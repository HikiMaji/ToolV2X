"""Shared original GoT driver adaptation with resumable optimizer checkpoints."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def epoch_order(rows, seed, epoch, start=0):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['sample_id']].append(index)
    rng = random.Random(seed + epoch * 100003)
    samples = sorted(groups)
    rng.shuffle(samples)
    order = []
    for sample in samples:
        group = list(groups[sample])
        rng.shuffle(group)
        order.extend(group)
    if not 0 <= start <= len(order):
        raise ValueError('resume position outside epoch')
    return order[start:]


def answer_only_loss(model, inputs_embeds, attention_mask, position_ids, labels):
    """Original Llama/LoRA decoder, projecting only causally supervised positions."""
    import torch.nn.functional as F
    base = model.get_base_model()
    if model.active_peft_config.peft_type != 'LORA' or base.config.pretraining_tp != 1:
        raise ValueError('answer-only head requires the current LoRA and single output head')
    shifted_labels = labels[..., 1:]
    selected = shifted_labels != -100
    if not selected.any():
        raise ValueError('no causally predictable supervision')
    # LoRA layers remain inside the original decoder. No prompt states are removed.
    with model._enable_peft_forward_hooks():
        hidden = base.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False, return_dict=True)[0]
        logits = base.lm_head(hidden[..., :-1, :][selected]).float()
    return F.cross_entropy(logits, shifted_labels[selected])


def forward_loss(planner, row, answer_head=False):
    import numpy as np
    import torch
    from planning.adaptation_data import encode_supervision
    from planning.v2vgot import MODEL_CONFIG, model_features
    with np.load(row['feature_path'], allow_pickle=False) as saved:
        features = dict(saved)
    model = planner.model
    count = int(features['active_agent_mask'].sum()) * 270
    batch = {k: v.to(model.device) for k, v in encode_supervision(row, planner.tokenizer, count).items()}
    tensors = model_features(features, model.device, model.dtype)
    position = int(torch.nonzero(batch['input_ids'][0] == -200)[0])
    supervised = int((batch['labels'] != -100).sum())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prepared = model.prepare_inputs_labels_for_multimodal(
            batch['input_ids'], None, torch.ones_like(batch['input_ids']), None, batch['labels'],
            torch.zeros((1, 3, 336, 336), device=model.device, dtype=model.dtype), [(336, 336)],
            my_model_config=MODEL_CONFIG, **tensors)
        if (int((prepared[5] != -100).sum()) != supervised or not (prepared[5][0, position:position+count] == -100).all()
                or prepared[4].shape[1] > planner.context_limit):
            raise ValueError('supervision or feature masking contract failed')
        if answer_head:
            loss = answer_only_loss(model, prepared[4], prepared[2], prepared[1], prepared[5])
        else:
            loss = model(inputs_embeds=prepared[4], attention_mask=prepared[2], position_ids=prepared[1],
                         labels=prepared[5], use_cache=False, return_dict=True).loss
    if not torch.isfinite(loss):
        raise ValueError('nonfinite driver loss')
    return loss


def validation_loss(planner, rows, answer_head=False):
    import torch
    from common.audit_protocol import recording
    groups = defaultdict(list)
    planner.model.eval()
    with torch.no_grad():
        for row in rows:
            groups[(recording(row['scene']), row['action'])].append(float(forward_loss(planner, row, answer_head)))
    means = {scene+'|'+action: sum(values)/len(values) for (scene, action), values in groups.items()}
    planner.model.train()
    return dict(macro_loss=sum(means.values())/len(means), groups=means, examples=len(rows))


def save_checkpoint(planner, optimizer, destination, state):
    import numpy as np
    import torch
    final = destination
    if final.exists():
        raise FileExistsError(str(final))
    destination = final.with_name('.'+final.name+'.incomplete')
    destination.mkdir(parents=True, exist_ok=False)
    model = planner.model
    model.save_pretrained(destination, safe_serialization=True)
    model.get_base_model().config.save_pretrained(destination)
    planner.tokenizer.save_pretrained(destination)
    projectors = {name: p.detach().cpu() for name, p in model.named_parameters()
                  if '.mm_projector.' in name or '.mm_scene_projector.' in name}
    torch.save(projectors, destination/'non_lora_trainables.bin')
    torch.save(dict(state=state, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state()),
               destination/'training_state.pt')
    (destination/'training_state.json').write_text(json.dumps(state, indent=2)+'\n')
    # Publish only after every file is durably written. Interrupted staging is ignored.
    for path in destination.iterdir():
        if path.is_file():
            with path.open('rb') as saved:
                os.fsync(saved.fileno())
    descriptor = os.open(str(destination), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    destination.rename(final)
    descriptor = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_recovery(planner, optimizer, out, state):
    """Keep the newest two periodic states of this run; epoch checkpoints are untouched."""
    destination = out/('recovery-step%06d' % state['steps'])
    if not destination.exists():
        save_checkpoint(planner, optimizer, destination, state)
    elif json.loads((destination/'training_state.json').read_text()) != state:
        raise ValueError('existing recovery state differs')
    pointer = out/'latest_checkpoint.json'
    temporary = out/'latest_checkpoint.tmp'
    temporary.write_text(json.dumps(dict(checkpoint=str(destination), steps=state['steps'],
        epoch=state['epoch'], next_row=state['next_row']), indent=2)+'\n')
    with temporary.open('rb') as saved:
        os.fsync(saved.fileno())
    temporary.replace(pointer)
    descriptor = os.open(str(out), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    # Only delete this run's explicitly named completed recovery directories after publication.
    previous = sorted(p for p in out.glob('recovery-step[0-9]*')
                      if p.is_dir() and not p.is_symlink() and
                      p.name[len('recovery-step'):].isdigit() and (p/'training_state.json').is_file())
    for path in previous[:-2]:
        shutil.rmtree(path)
    return destination


def benchmark_loss(planner, rows, out):
    """No optimizer updates; compare identical stochastic examples and all gradients."""
    import statistics
    import torch
    from planning.run_connection import save_json
    model = planner.model
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    reference = None
    reports = []
    try:
        # First two executions warm up kernels. Three interleaved repetitions follow.
        for run, optimized in enumerate([False, True, False, True, False, True, False, True]):
            model.zero_grad(set_to_none=True)
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = perf_counter()
            losses = []
            for row in rows:
                loss = forward_loss(planner, row, optimized)
                losses.append(float(loss.detach()))
                (loss/len(rows)).backward()
                del loss
            torch.cuda.synchronize()
            elapsed = perf_counter()-begin
            report = dict(answer_head=optimized, warmup=run < 2, seconds=elapsed, losses=losses,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
                raise ValueError('missing or nonfinite benchmark gradient')
            if reference is None:
                reference = {n: p.grad.detach().cpu().clone() for n, p in parameters.items()}
                reference_losses = losses
            numerator = denominator = max_relative = 0.
            for name, parameter in parameters.items():
                difference = parameter.grad.detach().cpu()-reference[name]
                squared_error = float(difference.square().sum())
                squared_norm = float(reference[name].square().sum())
                numerator += squared_error
                denominator += squared_norm
                max_relative = max(max_relative, (squared_error/max(squared_norm, 1e-20))**0.5)
            report.update(gradient_relative_l2=(numerator/max(denominator, 1e-20))**0.5,
                          maximum_parameter_relative_l2=max_relative,
                          loss_max_abs=max(abs(a-b) for a,b in zip(losses, reference_losses)))
            reports.append(report)
            print(json.dumps(dict(benchmark=report)), flush=True)
        baseline = [r for r in reports if not r['answer_head'] and not r['warmup']]
        candidate = [r for r in reports if r['answer_head'] and not r['warmup']]
        seconds_before = statistics.median(r['seconds'] for r in baseline)
        seconds_after = statistics.median(r['seconds'] for r in candidate)
        tolerance = dict(loss_max_abs=1e-3, gradient_relative_l2=1e-2, maximum_parameter_relative_l2=.05)
        baseline_stable = all(r[k] <= limit for r in baseline for k, limit in tolerance.items())
        numerics_pass = baseline_stable and all(r[k] <= limit for r in candidate for k, limit in tolerance.items())
        save_json(out/'performance.json', dict(status='completed', optimizer_updates=0,
            examples=[dict(sample_id=r['sample_id'], action=r['action']) for r in rows],
            runs=reports, tolerance=tolerance, baseline_stable=baseline_stable, numerics_pass=numerics_pass,
            seconds_before=seconds_before, seconds_after=seconds_after,
            speedup=seconds_before/seconds_after,
            enable_answer_head=numerics_pass and seconds_after <= .97*seconds_before,
            scope='same full-model batch forward/backward; excludes optimizer and full-run ETA'))
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        torch.cuda.set_rng_state_all(cuda_rng)


def train(data, out, checkpoint, epochs=3, batch_size=8, seed=20, max_steps=0, resume=None, integration_only=False,
          answer_head=False, benchmark_only=False, checkpoint_steps=25, pause_after_steps=0):
    import numpy as np
    import torch
    from planning.run_framework import read_jsonl
    from planning.run_connection import save_json, snapshot_code
    from planning.v2vgot import V2VGoTPlanner
    from common.audit_protocol import recording

    if epochs < 1 or batch_size < 1 or max_steps < 0 or checkpoint_steps < 1 or pause_after_steps < 0:
        raise ValueError('invalid training budget')
    manifest = json.loads((data/'manifest.json').read_text())
    prepared_config = json.loads((Path(manifest['prepared_root'])/'config.json').read_text())
    integration_only = bool(integration_only or max_steps or prepared_config['per_recording'])
    if resume:
        previous = json.loads((resume/'training_state.json').read_text())
        if (max_steps and previous['steps'] >= max_steps) or previous['epoch'] >= epochs:
            raise ValueError('resume already reached the requested training budget')
    rows, val_rows = read_jsonl(data/'train.jsonl'), read_jsonl(data/'validation_loss.jsonl')
    if (not rows or not val_rows or any(r.get('role') != 'train' for r in rows) or
            any(r.get('role') != 'validation' for r in val_rows) or
            {recording(r['scene']) for r in rows} & {recording(r['scene']) for r in val_rows} or
            any(r.get('input_layout') != 'source_blocks_v1' for r in rows+val_rows)):
        raise ValueError('training/validation role or input contract mismatch')
    # Rebuild input fields from their actual online episode, rather than trusting edited prompts.
    from planning.inputs import make_prompt
    for row in rows+val_rows:
        task = json.loads(Path(row['source_run']).read_text())
        prepared = task['prepared']
        prompt = make_prompt('Trajectory', task['ego_motion'], prepared['evidence_used'], evidence_format='compact',
                             remote_evidence=prepared['remote_evidence_used'])
        if (prompt != row['prompt'] or row['feature_path'] != task['feature_path'] or row['role'] != task['role'] or
                row['sample_id'] != task['sample_id'] or row['action'] != task['policy']):
            raise ValueError('training input does not match its causal task')
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    config = dict(data=str(data), original_checkpoint=str(checkpoint), epochs=epochs, batch_size=batch_size,
        seed=seed, learning_rate=2e-5, projector_learning_rate=2e-5, weight_decay=0., gradient_clip=1.,
        patience=2, maximum_optimizer_steps=max_steps or None, resume=str(resume) if resume else None,
        train_rows=len(rows), validation_rows=len(val_rows), input_layout='source_blocks_v1',
        decoding='direct', mtr_updated=False, actual_rgb_input=False, optimizer='AdamW',
        answer_only_head=answer_head, benchmark_only=benchmark_only,
        checkpoint_steps=checkpoint_steps, pause_after_steps=pause_after_steps or None,
        checkpoint_retention='two periodic recovery points plus epoch checkpoints',
        selection='macro validation answer loss over recording and policy; never F improvement',
        integration_only=integration_only,
        scope='framework training integration' if integration_only else 'single-configuration shared driver adaptation')
    save_json(out/'config.json', config)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    planner = V2VGoTPlanner(checkpoint=resume or checkpoint, trainable_lora=True,
                           adapter_directory=out/'loaded_adapter', evidence_format='compact')
    model = planner.model
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = any(part in name for part in ('.lora_A.', '.lora_B.', '.mm_projector.'))
        parameter.requires_grad_(enabled)
        if enabled:
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0., foreach=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    state = dict(epoch=0, next_row=0, steps=0, examples_seen=0, best_loss=None, best_checkpoint=None,
                 bad_epochs=0, data=str(data), batch_size=batch_size, seed=seed, history=[])
    if resume:
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        # Restore FP32 training values after the model builder's BF16 initialization.
        adapter_state = load_file(str(resume/'adapter_model.safetensors'), device='cpu')
        set_peft_model_state_dict(model, adapter_state)
        projector_state = torch.load(str(resume/'non_lora_trainables.bin'), map_location='cpu')
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in projector_state:
                    parameter.copy_(projector_state[name])
        actual_adapter = get_peft_model_state_dict(model)
        restored = dict(adapter_exact=all(torch.equal(actual_adapter[k].detach().cpu(), v) for k,v in adapter_state.items()),
            projectors_exact=all(torch.equal(p.detach().cpu(), projector_state[n]) for n,p in model.named_parameters()
                                 if n in projector_state))
        del actual_adapter
        del adapter_state, projector_state
        stored = torch.load(str(resume/'training_state.pt'), map_location='cpu')
        previous = stored['state']
        if any(previous[k] != state[k] for k in ('data', 'batch_size', 'seed')):
            raise ValueError('resume data/order configuration changed')
        state = previous
        optimizer.load_state_dict(stored['optimizer'])
        torch.set_rng_state(stored['torch_rng'])
        torch.cuda.set_rng_state_all(stored['cuda_rng'])
        random.setstate(stored['python_rng'])
        np.random.set_state(stored['numpy_rng'])
        actual_optimizer = optimizer.state_dict()
        restored['optimizer_exact'] = actual_optimizer['param_groups'] == stored['optimizer']['param_groups'] and all(
            torch.equal(value.cpu(), stored['optimizer']['state'][key][name]) if torch.is_tensor(value)
            else value == stored['optimizer']['state'][key][name]
            for key, values in actual_optimizer['state'].items() for name, value in values.items())
        restored['rng_exact'] = (torch.equal(torch.get_rng_state(), stored['torch_rng']) and
            all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(), stored['cuda_rng'])) and
            random.getstate() == stored['python_rng'] and
            np.array_equal(np.random.get_state()[1], stored['numpy_rng'][1]) and
            np.random.get_state()[2:] == stored['numpy_rng'][2:])
        save_json(out/'resume_verification.json', dict(restored, resume=str(resume), next_row=state['next_row'],
            steps=state['steps'], scope='exact saved-state restoration; CUDA backward may be nondeterministic'))
        if not all(restored.values()):
            raise ValueError('saved training state was not restored exactly')
        del stored
    save_json(out/'model.json', dict(planner.provenance, trainable_parameters=sum(p.numel() for p in trainable)))
    if benchmark_only:
        order = epoch_order(rows, seed, state['epoch'], state['next_row'])
        benchmark_loss(planner, [rows[i] for i in order[:batch_size]], out)
        return
    begin = perf_counter()
    initial_examples = state['examples_seen']
    with (out/'steps.jsonl').open('w') as log:
        while state['epoch'] < epochs:
            order = epoch_order(rows, seed, state['epoch'], state['next_row'])
            for offset in range(0, len(order), batch_size):
                group = order[offset:offset+batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in group:
                    loss = forward_loss(planner, rows[index], answer_head)
                    losses.append(float(loss.detach()))
                    (loss/len(group)).backward()
                    del loss
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                state['steps'] += 1
                state['examples_seen'] += len(group)
                state['next_row'] += len(group)
                record = dict(epoch=state['epoch'], next_row=state['next_row'], steps=state['steps'],
                    examples_seen=state['examples_seen'], loss=sum(losses)/len(losses), gradient_norm=float(norm),
                    elapsed_seconds=perf_counter()-begin)
                log.write(json.dumps(record)+'\n')
                log.flush()
                save_json(out/'progress.json', dict(record, status='training'))
                if state['steps'] % 10 == 0 or max_steps:
                    print(json.dumps(record), flush=True)
                if (state['steps'] == 1 or state['steps'] % checkpoint_steps == 0 or
                        state['next_row'] == len(rows) or
                        pause_after_steps and state['steps'] >= pause_after_steps):
                    save_json(out/'progress.json', dict(record, status='saving_checkpoint'))
                    saving_started = perf_counter()
                    recovery = save_recovery(planner, optimizer, out, state)
                    print(json.dumps(dict(checkpoint=str(recovery), steps=state['steps'],
                        checkpoint_seconds=perf_counter()-saving_started)), flush=True)
                    save_json(out/'progress.json', dict(record, status='training', checkpoint=str(recovery)))
                    if pause_after_steps and state['steps'] >= pause_after_steps:
                        save_json(out/'result.json', dict(status='training_paused', checkpoint=str(recovery),
                            optimizer_steps=state['steps'], examples_seen=state['examples_seen'],
                            formal_adaptation_complete=False, generation_evaluated=False))
                        return
                if max_steps and state['steps'] >= max_steps:
                    destination = out/('checkpoint-step%06d' % state['steps'])
                    save_checkpoint(planner, optimizer, destination, state)
                    save_json(out/'result.json', dict(status='integration_budget_completed', checkpoint=str(destination),
                        optimizer_steps=state['steps'], examples_seen=state['examples_seen'],
                        new_training_examples=state['examples_seen']-initial_examples,
                        formal_adaptation_complete=False, generation_evaluated=False))
                    return
            save_json(out/'progress.json', dict(state, status='validating', elapsed_seconds=perf_counter()-begin))
            # Keep the original validation calculation for comparable checkpoint selection.
            validation = validation_loss(planner, val_rows)
            state['epoch'] += 1
            state['next_row'] = 0
            destination = out/('checkpoint-epoch%02d' % state['epoch'])
            improved = state['best_loss'] is None or validation['macro_loss'] < state['best_loss']
            state['bad_epochs'] = 0 if improved else state['bad_epochs']+1
            if improved:
                state['best_loss'], state['best_checkpoint'] = validation['macro_loss'], str(destination)
            state['history'].append(dict(epoch=state['epoch'], validation=validation, improved=improved))
            save_checkpoint(planner, optimizer, destination, state)
            save_json(out/'selection.json', state)
            print(json.dumps(state['history'][-1]), flush=True)
            if state['bad_epochs'] >= 2:
                break
    save_json(out/'result.json', dict(status='training_completed', optimizer_steps=state['steps'], epochs=state['epoch'],
        checkpoint=state['best_checkpoint'], validation_loss=state['best_loss'], generation_evaluated=False,
        examples_seen=state['examples_seen'], new_training_examples=state['examples_seen']-initial_examples,
        formal_adaptation_complete=not integration_only, method_effectiveness_established=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=0, help='explicit integration-only optimizer budget')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--integration-only', action='store_true', help='exercise full epoch/validation without a formal adaptation claim')
    parser.add_argument('--answer-only-head', action='store_true', help='project only answer-loss positions; full decoder context retained')
    parser.add_argument('--benchmark-only', action='store_true', help='compare loss paths without any optimizer update, then exit')
    parser.add_argument('--checkpoint-steps', type=int, default=25, help='save full recovery state every N steps, also at step 1')
    parser.add_argument('--pause-after-steps', type=int, default=0, help='save and exit at this total step; later resume continues the budget')
    args = parser.parse_args()
    if args.checkpoint is None:
        from planning.v2vgot import CHECKPOINT
        args.checkpoint = CHECKPOINT
    try:
        train(args.data.resolve(), args.out.resolve(), args.checkpoint.resolve(), args.epochs,
              args.batch_size, args.seed, args.max_steps, args.resume.resolve() if args.resume else None, args.integration_only,
              args.answer_only_head, args.benchmark_only, args.checkpoint_steps, args.pause_after_steps)
    except Exception as exc:
        if args.out.is_dir() and not isinstance(exc, FileExistsError):
            (args.out/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)))+'\n')
        raise
