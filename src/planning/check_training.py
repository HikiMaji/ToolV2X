"""One real shared-driver update on one train frame; not an adaptation study."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from planning.adaptation_data import encode_supervision, supervised_examples
from planning.inputs import make_prompt
from planning.v2vgot import CHECKPOINT, MODEL_CONFIG, V2VGoTPlanner, model_features
from planning.run_connection import save_json, snapshot_code


def rows_for_check(run, data, decoding):
    connection = json.loads((run / 'connection.json').read_text())
    sample_id = '%s:%d' % (connection['scene'], connection['local_frame'])
    index = {r['sample_id']: r for r in map(json.loads, (data / 'online_index/train.jsonl').read_text().splitlines())}
    if sample_id not in index or index[sample_id]['role'] != 'train':
        raise ValueError('update check requires an indexed training recording')
    labels = {r['sample_id']: r for r in map(json.loads, (data / 'offline_labels/train.jsonl').read_text().splitlines())}
    label = labels[sample_id]
    if label['role'] != 'train' or label['g'] != connection['g'] or not all(label['valid']):
        raise ValueError('incorrect training label alignment')
    rows = []
    for action in ('Ego', 'P', 'F', 'PF'):
        plan = json.loads((run / action / 'plan.json').read_text())
        if decoding == 'direct':
            plan = dict(decoding='direct', q8_executed=False, evidence_used=plan['evidence_used'],
                        evidence_selection=plan['evidence_selection'],
                        q9_prompt=make_prompt('Trajectory', connection['ego_motion'], plan['evidence_used'],
                                              evidence_format=plan['evidence_selection']['evidence_format']))
        rows.extend(supervised_examples('train', sample_id, action, connection['ego_motion'],
                                        plan, label, str(run / 'ego_features.npz')))
    expected = {'Trajectory'} if decoding == 'direct' else {'Q8', 'Q9'}
    if len(rows) != 4 * len(expected) or any({r['task'] for r in rows if r['action'] == a} != expected
                                           for a in ('Ego', 'P', 'F', 'PF')):
        raise ValueError('readiness frame must have all four actions and valid saved parents')
    return rows


def check(run, data, out, checkpoint, decoding):
    rows = rows_for_check(run, data, decoding)
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    (out / 'examples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    planner = V2VGoTPlanner(checkpoint=checkpoint, trainable_lora=True,
                           adapter_directory=out / 'input_adapter', evidence_format='compact')
    model = planner.model
    trainable = {}
    for name, parameter in model.named_parameters():
        enabled = any(part in name for part in ('.lora_A.', '.lora_B.', '.mm_projector.'))
        parameter.requires_grad_(enabled)
        if enabled:
            parameter.data = parameter.data.float()
            trainable[name] = parameter
    if not trainable or not any('.lora_A.' in name for name in trainable):
        raise ValueError('original LoRA was merged or missing')
    before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    lora = [p for name, p in trainable.items() if '.mm_projector.' not in name]
    projector = [p for name, p in trainable.items() if '.mm_projector.' in name]
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': 2e-4}, {'params': projector, 'lr': 2e-5}],
                                  weight_decay=0., foreach=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    report = dict(status='accumulating_gradients',
        scope='one real optimizer update; infrastructure only, not shared-driver adaptation quality',
        model=planner.provenance, decoding=decoding, input_run=str(run), data=str(data),
        original_mtr_evidence='historical readiness frame; not the stability-selected weight',
        q8_parent_source='old saved GoT generation' if decoding == 'q8_q9' else 'not applicable',
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        trainable_parameter_tensors=len(trainable), base_compute_dtype='bfloat16',
        trainable_parameter_dtype='float32', lora_lr=2e-4, projector_lr=2e-5,
        gradient_clip=1., weight_decay=0., seed=20, optimizer_steps=0, rows=[],
        actual_rgb_input=False, future_fields_in_prompt=False, test_access=False)
    save_json(out / 'training_check.json', report)
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    for row in rows:
        with np.load(row['feature_path'], allow_pickle=False) as saved:
            features = dict(saved)
        count = int(features['active_agent_mask'].sum()) * 270
        batch = {key: value.to(model.device) for key, value in
                 encode_supervision(row, planner.tokenizer, count).items()}
        tensors = model_features(features, model.device, model.dtype)
        position = int(torch.nonzero(batch['input_ids'][0] == -200)[0])
        supervised = int((batch['labels'] != -100).sum())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prepared = model.prepare_inputs_labels_for_multimodal(
                batch['input_ids'], None, torch.ones_like(batch['input_ids']), None, batch['labels'],
                torch.zeros((1, 3, 336, 336), device=model.device, dtype=model.dtype), [(336, 336)],
                my_model_config=MODEL_CONFIG, **tensors)
            assert int((prepared[5] != -100).sum()) == supervised
            assert (prepared[5][0, position:position + count] == -100).all()
            assert prepared[4].shape[1] <= planner.context_limit
            output = model(inputs_embeds=prepared[4], attention_mask=prepared[2], position_ids=prepared[1],
                           labels=prepared[5], use_cache=False, return_dict=True)
            if not torch.isfinite(output.loss):
                raise ValueError('nonfinite original supervised loss')
            loss = output.loss / len(rows)
        loss.backward()
        record = dict(action=row['action'], task=row['task'], loss=float(output.loss.detach()),
                      sequence_tokens=int(prepared[4].shape[1]), supervised_tokens=supervised,
                      feature_tokens=count, feature_labels_masked=True)
        report['rows'].append(record)
        save_json(out / 'training_check.json', report)
        print(json.dumps(record), flush=True)
        del prepared, output, loss, batch, tensors
    gradients = {name: float(p.grad.float().norm()) if p.grad is not None else None
                 for name, p in trainable.items()}
    if any(value is None or not np.isfinite(value) for value in gradients.values()):
        raise ValueError('missing or nonfinite gradient on a trainable parameter')
    frozen_gradients = [name for name, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    if frozen_gradients:
        raise ValueError('frozen original parameters received gradients')
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1., error_if_nonfinite=True)
    optimizer.step()
    report.update(status='updated_save_pending', optimizer_steps=1)
    save_json(out / 'training_check.json', report)
    differences = {name: float((p.detach().cpu() - before[name]).abs().max()) for name, p in trainable.items()}
    if not any(value > 0 for name, value in differences.items() if '.lora_' in name):
        raise ValueError('no real LoRA update')
    if not any(value > 0 for name, value in differences.items() if '.mm_projector.' in name):
        raise ValueError('no real projector update')
    destination = out / 'checkpoint'
    model.save_pretrained(destination, safe_serialization=True)
    model.get_base_model().config.save_pretrained(destination)
    planner.tokenizer.save_pretrained(destination)
    non_lora = {name: p.detach().cpu() for name, p in model.named_parameters()
                if '.mm_projector.' in name or '.mm_scene_projector.' in name}
    torch.save(non_lora, destination / 'non_lora_trainables.bin')
    from safetensors import safe_open
    with safe_open(str(destination / 'adapter_model.safetensors'), framework='pt') as saved:
        expected = {name.replace('.default.', '.'): p for name, p in trainable.items()
                    if '.mm_projector.' not in name}
        assert set(saved.keys()) == set(expected)
        assert all(torch.equal(saved.get_tensor(name), p.detach().cpu()) for name, p in expected.items())
    saved_projector = torch.load(str(destination / 'non_lora_trainables.bin'), map_location='cpu', mmap=True)
    assert saved_projector.keys() == non_lora.keys()
    assert all(torch.equal(saved_projector[name], value) for name, value in non_lora.items())
    save_json(out / 'parameter_updates.json', dict(gradient_norms=gradients, maximum_absolute_changes=differences))
    report.update(status='updated_and_saved', optimizer_steps=1, elapsed_s=perf_counter() - begin,
                  gradient_norm_before_clip=float(norm), frozen_gradients=frozen_gradients,
                  changed_tensors=sum(value > 0 for value in differences.values()),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint=str(destination), optimizer_state_saved=False,
                  saved_adapter_matches_live=True, saved_projector_matches_live=True,
                  reload_verified=False, formal_adaptation_complete=False)
    save_json(out / 'training_check.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('model', 'rows')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/framework_compact_train_v1')
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/adaptation_data_v1')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--decoding', choices=('direct', 'q8_q9'), default='direct')
    args = parser.parse_args()
    try:
        check(args.run.resolve(), args.data.resolve(), args.out.resolve(), args.checkpoint.resolve(), args.decoding)
    except Exception as exc:
        if args.out.is_dir() and not isinstance(exc, FileExistsError):
            save_json(args.out / 'failure.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise
