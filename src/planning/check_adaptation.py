"""Execute native supervised forward passes; no gradients or optimizer updates."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from planning.adaptation_data import encode_supervision
from planning.v2vgot import V2VGoTPlanner, model_features, MODEL_CONFIG
from planning.run_connection import save_json, snapshot_code


@torch.inference_mode()
def check(examples, out):
    rows = [json.loads(line) for line in examples.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('no materialized training examples')
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    planner = V2VGoTPlanner(evidence_format='compact')
    model, tokenizer = planner.model, planner.tokenizer
    report = dict(scope='native supervised forward only; no backward or optimization',
                  examples=str(examples), model=planner.provenance, rows=[], optimizer_steps=0)
    for row in rows:
        with np.load(row['feature_path'], allow_pickle=False) as saved:
            features = dict(saved)
        count = int(features['active_agent_mask'].sum()) * 270
        batch = {key: value.to(model.device) for key, value in encode_supervision(row, tokenizer, count).items()}
        tensors = model_features(features, model.device, model.dtype)
        image_position = int(torch.nonzero(batch['input_ids'][0] == -200)[0])
        prepared = model.prepare_inputs_labels_for_multimodal(
            batch['input_ids'], None, torch.ones_like(batch['input_ids']), None, batch['labels'],
            torch.zeros((1, 3, 336, 336), device=model.device, dtype=model.dtype), [(336, 336)],
            my_model_config=MODEL_CONFIG, **tensors)
        labels = prepared[5]
        supervised = int((batch['labels'] != -100).sum())
        assert int((labels != -100).sum()) == supervised
        assert (labels[0, image_position:image_position + count] == -100).all()
        assert prepared[4].shape[1] <= planner.context_limit
        begin = perf_counter()
        output = model(inputs_embeds=prepared[4], attention_mask=prepared[2], position_ids=prepared[1],
                       labels=labels, use_cache=False, return_dict=True)
        loss = float(output.loss)
        if not np.isfinite(loss):
            raise ValueError('nonfinite original model supervised loss')
        record = dict(sample_id=row['sample_id'], action=row['action'], task=row['task'],
                      loss=loss, seconds=perf_counter() - begin, sequence_tokens=prepared[4].shape[1],
                      feature_tokens=count, supervised_tokens=supervised, feature_labels_masked=True)
        report['rows'].append(record)
        save_json(out / 'forward_check.json', report)
        print(json.dumps(record), flush=True)
        del output, prepared, tensors
    report['all_losses_finite'] = True
    save_json(out / 'forward_check.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    check(args.examples, args.out)
