"""Materialize Ego/F inputs with a frozen local block; never import GT labels."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt

from planning.context import CONTEXT, GENERATION, PEER_RESERVE, input_size, fit_local, fit_remote


def materialize(out):
    import torch
    from transformers import AutoTokenizer
    from common import v2v4real_meta as M
    from prediction import cmp_adapter as C
    from planning.inputs import load_ego_features, load_ego_motion
    from planning.run_connection import load_window, save_json, snapshot_code
    from planning.v2vgot import CHECKPOINT
    from tools.vehicle import VehicleTools, decode_response, make_evidence

    config = json.loads((out / 'config.json').read_text())
    if (out / 'online_status.json').exists():
        raise FileExistsError('preserve prior attempts; use a new output directory')
    if config['context_limit'] != CONTEXT or config['peer_reserve'] != PEER_RESERVE:
        raise ValueError('run configuration does not match selector')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, provenance = C.load_model(checkpoint=Path(config['mtr_checkpoint']), device=config['device'])
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    save_json(out / 'mtr_loading.json', provenance)
    snapshot_code(out)
    rows = [json.loads(line) for role in ('train', 'validation')
            for line in (out / 'online_index' / (role + '.jsonl')).read_text().splitlines()]
    (out / 'frames').mkdir()
    save_json(out / 'online_status.json', dict(status='running', completed=0, expected=len(rows)))

    @lru_cache(maxsize=2)
    def archive(path):
        with path.open('rb') as handle:
            return pickle.load(handle)

    begin = perf_counter()
    for index, row in enumerate(rows):
        directory = out / row['directory']
        directory.mkdir()
        paths, calls = [], []
        def window(source):
            value = load_window(row['scene'], row['local_frame'], source, paths, archive)
            if int(value['g']) != row['g']:
                raise ValueError('window does not match indexed decision')
            return value

        def predict(value):
            start = perf_counter()
            result = C.predict(model, value, batch_size=config['target_batch_size'])
            path = directory / ('forecast_' + value['source'] + '.npz')
            np.savez(str(path), **result)
            calls.append(dict(source=value['source'], g=int(value['g']), seconds=perf_counter() - start,
                              model_targets=int(result['model_used'].sum()), targets=len(result['track_ids']),
                              prediction_path=str(path)))
            return result

        ego = window('no_fusion')
        features = load_ego_features(M.V2VGOT_ROOT, 'train', row['g'])
        motion, motion_paths = load_ego_motion(M.V2VGOT_ROOT, 'train', row['g'])
        if (features['read_paths'] != row['feature_read_paths'] or motion != row['ego_motion'] or
                motion_paths != row['motion_read_paths']):
            raise ValueError('indexed ego inputs changed')
        feature_tokens = int(features['active_agent_mask'].sum()) * 270
        np.savez(str(directory / 'ego_features.npz'),
                 **{k: v for k, v in features.items() if isinstance(v, np.ndarray)})
        local_prediction = predict(ego)
        if index == 0:
            # Check this execution optimization against the existing CPU/one-target path.
            reference_model, _ = C.load_model(checkpoint=Path(config['mtr_checkpoint']), device='cpu')
            reference = C.predict(reference_model, ego, batch_size=1)
            differences = {key: float(np.max(np.abs(local_prediction[key].astype(float) - reference[key].astype(float))))
                           if reference[key].size else 0. for key in reference}
            for key in ('track_ids', 'states', 'model_used'):
                np.testing.assert_array_equal(local_prediction[key], reference[key])
            for key in ('means', 'local_gmm'):
                np.testing.assert_allclose(local_prediction[key], reference[key], rtol=1e-4, atol=.01)
            np.testing.assert_allclose(local_prediction['scores'], reference['scores'], rtol=1e-4, atol=1e-5)
            save_json(out / 'execution_equivalence.json', dict(scope='one real first frame; portable torch execution only',
                max_absolute_difference=differences, native_CUDA_parity=False, rtol=1e-4, trajectory_atol=.01,
                scores_atol=1e-5, cpu_target_batch=1, materialization_target_batch=config['target_batch_size']))
            del reference_model, reference
        local_full = make_evidence(ego, local_prediction, [], predict)
        local, local_selection = fit_local(tokenizer, motion, local_full, feature_tokens)
        ego_prompt = make_prompt('Trajectory', motion, local, evidence_format='compact')
        assert len(paths) == 1 and len(calls) == 1
        # Commit the Ego input before any peer window read or query.
        save_json(directory / 'ego_input.json', dict(evidence_full=local_full, evidence_used=local,
            evidence_selection=local_selection, prompt=ego_prompt, remote_archive_reads=0,
            request_bytes=0, response_bytes=0))
        service = VehicleTools(lambda: window('no_fusion_cav1'), predict, row['scene'], row['g'])
        response = service.query('F')
        (directory / 'F_response.json').write_bytes(response['wire'])
        save_json(directory / 'F_request.json', response['request'])
        packet = decode_response(response['wire'], row['scene'], row['g'])
        full = make_evidence(ego, local_prediction, [packet], predict)
        remote_full = dict(full, objects=[o for o in full['objects'] if o['source'] != 'ego'])
        remote, remote_selection = fit_remote(tokenizer, motion, local, remote_full, feature_tokens)
        prompt = make_prompt('Trajectory', motion, local, evidence_format='compact', remote_evidence=remote)
        if not prompt.startswith(ego_prompt.split('Output six numeric')[0]):
            raise AssertionError('literal ego input changed')
        save_json(directory / 'F_input.json', dict(evidence_full=remote_full, evidence_used=remote,
            local_evidence_path=str(directory / 'ego_input.json'), evidence_selection=remote_selection,
            prompt=prompt, tool_costs=response['cost'], remote_archive_reads=len(paths) - 1))
        save_json(directory / 'sample.json', dict(sample_id=row['sample_id'], role=row['role'], g=row['g'],
            feature_path=str(directory / 'ego_features.npz'), feature_frames=features['frame_indices'],
            feature_read_paths=features['read_paths'], feature_tokens=feature_tokens, ego_motion=motion,
            truncated_boxes=features['truncated_boxes'], window_reads=paths, predictor_calls=calls,
            actual_rgb_input=False, driving_model_executed=False, q8_executed=False, q9_executed=False,
            actual_original_cmp_executed=any(c['model_targets'] for c in calls)))
        status = dict(status='running', completed=index + 1, expected=len(rows),
                      elapsed_seconds=perf_counter() - begin, last_sample=row['sample_id'])
        save_json(out / 'online_status.json', status)
        if index % 25 == 0 or index + 1 == len(rows):
            print(json.dumps(status), flush=True)
    status['status'] = 'completed'
    save_json(out / 'online_status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    try:
        materialize(args.out.resolve())
    except Exception as exc:
        path = args.out / 'online_failure.json'
        path.write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc))) + '\n')
        raise
