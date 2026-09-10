"""GT-blocked, frozen-time replay of the shared learned predictor on validation."""
import argparse
import builtins
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from probe import kinematic_tools as K
from probe import learned_predictor as L
from probe.learned_data import scene_inputs
from probe.run_vehicle_probe import causal_reference

ACTION_SERVICES = {'ego': [], 'P': ['P'], 'F': ['F'], 'PF': ['P', 'F'],
                   'P_history': ['P_history'], 'ego_cv': [], 'F_cv': ['F_cv']}


def load_model(path):
    with np.load(str(path), allow_pickle=False) as d:
        weights, metadata = d['weights'], json.loads(str(d['metadata']))
    if metadata['predictor'] != L.PREDICTOR or metadata['test_used'] is not False:
        raise ValueError('wrong model provenance')
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    if metadata['train_scenes'] != manifest['train_scenes']:
        raise ValueError('model trained with a different split')
    if weights.shape != (50, L.FEATURES, 2) or not np.isfinite(weights).all():
        raise ValueError('invalid model weights')
    return weights, metadata


def run(model_path, out):
    model_path, out = Path(model_path).resolve(), Path(out)
    weights, model_meta = load_model(model_path)
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    real_open, real_pose = builtins.open, M.load_pose
    cutoff = [None]
    guard_counts = {'checked_pose_reads': 0, 'blocked_reads': 0}

    def guarded_open(file, mode='r', *args, **kwargs):
        path = str(file)
        if 'r' in mode and ('_gt' in path or '/data/' in path or path.endswith('.pth')
                or (path.endswith('.npz') and Path(path).resolve() != model_path)):
            guard_counts['blocked_reads'] += 1
            raise AssertionError('forbidden generation input: ' + path)
        return real_open(file, mode, *args, **kwargs)

    def guarded_pose(split, g, cav='ego'):
        if cutoff[0] is None or g > cutoff[0] or g < cutoff[0] - 10 or split != 'train':
            raise AssertionError('pose outside current causal history')
        guard_counts['checked_pose_reads'] += 1
        return real_pose(split, g, cav)

    with patch('builtins.open', side_effect=guarded_open), \
            patch.object(M, 'load_gt', side_effect=AssertionError('GT disabled in generation')), \
            patch.object(M, 'ego_future_waypoints', side_effect=AssertionError('future ego GT disabled')), \
            patch.object(M, 'load_pose', side_effect=guarded_pose):
        for scene in manifest['validation_scenes']:
            split, frames, windows = scene_inputs('validation', scene)
            dest = out / scene
            dest.mkdir()
            meta = {'protocol': 'vehicle_learned_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scene': scene, 'role': 'validation', 'source_split': split, 'frames': frames,
                'model_path': str(model_path), 'model_metadata': model_meta, 'actions': ACTION_SERVICES,
                'input_generation_GT': False, 'reference': 'ego past 1s OLS, 3s extrapolation',
                'receiver': 'current position Hungarian 2m, average shared trajectories, retain peer-only',
                'scope': 'fixed-cache validation capability screen, not test or closed loop',
                'physical_network_used': False, 'detector_training_independence': 'unverified',
                'RSU': False, 'I': False, 'MTR': False, 'LLM': False, 'router': False}
            (dest / 'run.json').write_bytes(K.encode(meta) + b'\n')
            totals = {'frames': 0, 'same_information_equal_frames': 0, 'PF_equals_F_frames': 0,
                      'P_F_different_prediction_frames': 0, 'ego_F_different_scale_frames': 0}
            with open(dest / 'decisions.jsonl', 'wb') as decisions, open(dest / 'messages.jsonl', 'wb') as messages:
                for t in frames:
                    local_w, peer_w = windows['no_fusion'][t], windows['no_fusion_cav1'][t]
                    g = local_w['g']
                    cutoff[0] = g
                    if peer_w['g'] != g or peer_w['source'] != 'no_fusion_cav1' or local_w['source'] != 'no_fusion':
                        raise ValueError('source/time mismatch')
                    local_history = K.history_packet(local_w)
                    ego_cv, ego = K.forecast(local_history), L.forecast(local_history, weights)
                    received, wires = {}, {}
                    for service in ('P', 'F', 'P_history', 'F_cv'):
                        begin = perf_counter()
                        history = K.history_packet(peer_w)
                        packet = (L.summary_packet(history) if service == 'P' else
                                  L.forecast(history, weights) if service == 'F' else
                                  history if service == 'P_history' else K.forecast(history))
                        sender_ms = (perf_counter() - begin) * 1000
                        request = {'version': K.VERSION, 'scene': scene, 'g': g, 'service': service,
                            'scope': 'all_current_peer_tracks', 'coordinate_frame': 'ego_at_t',
                            'ego_pose_at_t': M.load_pose(split, g).tolist(), 'history_seconds': 1.,
                            'horizon_seconds': 5.}
                        req, res = K.encode(request), K.encode(packet)
                        received[service] = json.loads(res)
                        wires[service] = {'request_bytes': len(req), 'response_bytes': len(res)}
                        messages.write(K.encode(dict(scene=scene, t=t, g=g, service=service,
                            request_json=req.decode(), response_json=res.decode(), sender_processing_ms=sender_ms,
                            network_ms=None, sender_detection_tracking_ms=None,
                            transport='in_process_json_roundtrip', **wires[service])) + b'\n')
                    if L.forecast(received['P_history'], weights) != received['F']:
                        raise ValueError('same-information local/remote forecast mismatch')
                    if {o['key']: o['state'] for o in received['P']['objects']} != {
                            o['key']: o['state'] for o in received['F']['objects']}:
                        raise ValueError('P/F anchors differ')
                    reference = causal_reference(split, g)
                    actions = {}
                    for action, services in ACTION_SERVICES.items():
                        begin = perf_counter()
                        remote = (L.forecast(received[action], weights) if action in ('P', 'P_history') else
                                  received['F'] if action in ('F', 'PF') else
                                  received['F_cv'] if action == 'F_cv' else None)
                        base = ego_cv if action in ('ego_cv', 'F_cv') else ego
                        objects, audit = K.receive(base, remote_f=remote)
                        plan = K.choose_speed(reference, objects)
                        actions[action] = dict(used_services=services, objects=objects, association=audit, plan=plan,
                            receiver_planning_ms=(perf_counter() - begin) * 1000,
                            request_bytes=sum(wires[s]['request_bytes'] for s in services),
                            response_bytes=sum(wires[s]['response_bytes'] for s in services))
                    if not actions['F']['objects'] == actions['PF']['objects'] == actions['P_history']['objects']:
                        raise ValueError('F/PF/full-history receiver mismatch')
                    decisions.write(K.encode(dict(scene=scene, t=t, g=g, reference=reference.tolist(),
                        ego_forecast=ego, actions=actions)) + b'\n')
                    totals['frames'] += 1
                    totals['same_information_equal_frames'] += 1
                    totals['PF_equals_F_frames'] += 1
                    totals['P_F_different_prediction_frames'] += int(actions['P']['objects'] != actions['F']['objects'])
                    totals['ego_F_different_scale_frames'] += int(actions['ego']['plan']['scale'] != actions['F']['plan']['scale'])
            (dest / 'generation_summary.json').write_bytes(K.encode(totals) + b'\n')
            print(json.dumps(dict(scene=scene, **totals)), flush=True)
    (out / 'input_guard.json').write_bytes(K.encode(dict(status='PASS', **guard_counts,
        GT_functions_disabled=True, future_pose_access_disabled=True,
        note='cached causal windows verified separately; model loaded before generation')) + b'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path')
    ap.add_argument('out')
    a = ap.parse_args()
    run(a.model_path, a.out)
