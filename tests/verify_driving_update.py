"""Reload a one-update checkpoint through the standard original-model inference path."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.v2vgot import V2VGoTPlanner


def verify(run):
    output = run / 'reload_check.json'
    if output.exists():
        raise FileExistsError(str(output))
    training = json.loads((run / 'training_check.json').read_text())
    assert training['optimizer_steps'] == 1 and training['status'] == 'updated_and_saved'
    assert training['saved_adapter_matches_live'] and training['saved_projector_matches_live']
    checkpoint = Path(training['checkpoint'])
    planner = V2VGoTPlanner(checkpoint=checkpoint, adapter_directory=run / 'reload_adapter')
    assert not planner.provenance['trainable_lora_loaded']
    assert planner.provenance['attention_implementation'] == 'sdpa'
    model = planner.model
    state = model.state_dict()
    original_index = json.loads((Path(planner.provenance['base']) / 'pytorch_model.bin.index.json').read_text())['weight_map']
    config = json.loads((checkpoint / 'adapter_config.json').read_text())
    scale = config['lora_alpha'] / config['r']
    checks = []
    with safe_open(str(checkpoint / 'adapter_model.safetensors'), framework='pt') as saved:
        for name in ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.gate_proj.weight',
                     'model.layers.31.self_attn.v_proj.weight'):
            base = torch.load(str(Path(planner.provenance['base']) / original_index[name]), map_location='cpu', mmap=True)
            prefix = 'base_model.model.' + name[:-len('weight')]
            a = saved.get_tensor(prefix + 'lora_A.weight').to(device=model.device, dtype=model.dtype)
            b = saved.get_tensor(prefix + 'lora_B.weight').to(device=model.device, dtype=model.dtype)
            indices = torch.arange(0, b.shape[0], max(1, b.shape[0] // 128), device=model.device)
            expected = base[name][indices.cpu()].to(device=model.device, dtype=model.dtype) + ((b @ a) * scale)[indices]
            actual = state[name][indices]
            error = float((actual - expected).abs().max())
            assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-5)
            checks.append(dict(parameter=name, compared_values=expected.numel(), maximum_absolute_error=error))
            del base, a, b, indices, expected, actual
    projector = torch.load(str(checkpoint / 'non_lora_trainables.bin'), map_location='cpu', mmap=True)
    projector_checks = {name: torch.equal(state[name[len('base_model.model.'):]].cpu(), value.to(dtype=model.dtype))
                        for name, value in projector.items()}
    assert all(projector_checks.values())
    rows = [json.loads(line) for line in (run / 'examples.jsonl').read_text().splitlines()]
    row = rows[-1]
    with np.load(row['feature_path'], allow_pickle=False) as saved:
        features = dict(saved)
    raw, cost = planner._generate(features, row['prompt'], 256)
    report = dict(status='PASS', scope='serialization, merge, projector and actual generation after reload; not driving quality',
        training_report=str(run / 'training_check.json'), model=planner.provenance,
        sampled_lora_merge_checks=checks, projector_exact_checks=projector_checks,
        actual_generation=True, generated_sample_id=row['sample_id'], generated_action=row['action'],
        task=row['task'], prompt_source='saved training prompt; no target answer supplied',
        raw_generation=raw, generation_cost=cost, optimizer_steps_in_this_verifier=0,
        formal_adaptation_complete=False)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], raw=raw, cost=cost)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    verify(parser.parse_args().run.resolve())
