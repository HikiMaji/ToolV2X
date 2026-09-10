"""Generate GT-free, frozen-time vehicle-only P/F interface replay on ONE scene.

No RSU, partner plan, CoBEVT fusion, MTR, LLM, training or physical network.
Counterfactual enumeration: all action results are generated, not a live router.
Measured bytes are exact UTF-8 JSON application payloads; network latency unknown.
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from probe import kinematic_tools as K


def causal_reference(split, g):
    """OLS velocity of past ego positions in ego(t); no future pose or route."""
    pose_t = M.load_pose(split, g)
    past = np.stack([np.linalg.solve(pose_t, np.r_[M.load_pose(split, h)[:3, 3], 1.])[:2]
                     for h in range(g - 10, g + 1)])
    times = np.arange(-10, 1) * .1
    centered = times - times.mean()
    velocity = (centered[:, None] * (past - past.mean(axis=0))).sum(axis=0) / (centered @ centered)
    return np.arange(1, 31)[:, None] * .1 * velocity


def run(scene=None, role='validation', out=None):
    with open(ROOT / 'outputs/protocol_audit/split_manifest.json') as f:
        manifest = json.load(f)
    selected = manifest[role + '_scenes']
    scene = scene or sorted(selected)[0]
    if scene not in selected:
        raise ValueError('scene not in selected role manifest')
    split = 'test' if role == 'test' else 'train'
    with open(ROOT / ('outputs/protocol_audit/%s_frames.json' % role)) as f:
        times = [int(t) for s, t in json.load(f) if s == scene]
    if not times or len(times) != len(set(times)):
        raise ValueError('empty or duplicate manifest frames')
    source_data = {}
    paths = {}
    for source in ('no_fusion', 'no_fusion_cav1'):
        p = ROOT / 'outputs/causal_windows_v1' / split / source / (scene + '.pkl')
        with open(p, 'rb') as f:
            artifact = pickle.load(f)
        meta = artifact['meta']
        if (meta['protocol'] != 'causal_windows_v1' or meta['coordinate_frame'] != 'ego_at_t'
                or meta['scene'] != scene or meta['split'] != split or meta['gt_access'] is not False):
            raise ValueError('wrong source metadata')
        if not set(times).issubset(artifact['windows']):
            raise ValueError('missing source window')
        source_data[source] = artifact['windows']
        paths[source] = str(p)
    out = Path(out) if out else ROOT / 'outputs/vehicle_probe_v1' / scene
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'protocol': K.VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scene': scene, 'role': role, 'source_split': split, 'frames': times,
                'source_paths': paths, 'forecast': 'ols_cv_history_v1',
                'P_semantics': 'full remote track history; same-information local-compute control',
                'F_semantics': 'remote constant-velocity forecast from the identical history',
                'reference': 'past ego pose OLS extrapolation, no future reference',
                'receiver': 'position Hungarian <=2m; matched predictions equal average; retain peer-only',
                'timing': 'frozen observation cutoff t, no simulated arrival shift',
                'scope': 'one-scene interface replay, not method or closed-loop evaluation',
                'gt_used_in_generation': False, 'physical_network_used': False,
                'detector_training_independence': 'unverified', 'RSU': False,
                'MTR': False, 'LLM': False, 'training': False, 'partner_plan_I': False}
    (out / 'run.json').write_bytes(K.encode(metadata) + b'\n')
    totals = {'frames': 0, 'frames_with_peer_only': 0, 'peer_only_states': 0,
              'same_information_equal_frames': 0, 'ego_changed_by_peer_frames': 0,
              'P_request_bytes': 0, 'P_response_bytes': 0, 'F_request_bytes': 0, 'F_response_bytes': 0}
    with open(out / 'decisions.jsonl', 'wb') as decisions, open(out / 'messages.jsonl', 'wb') as messages:
        for t in times:
            local_w = source_data['no_fusion'][t]
            peer_w = source_data['no_fusion_cav1'][t]
            g = local_w['g']
            if peer_w['g'] != g or local_w['source'] != 'no_fusion' or peer_w['source'] != 'no_fusion_cav1':
                raise ValueError('source frame mismatch')
            ego = K.forecast(K.history_packet(local_w))
            begin = perf_counter()
            p = K.history_packet(peer_w)
            p_ms = (perf_counter() - begin) * 1000
            begin = perf_counter()
            f = K.forecast(p)
            f_ms = (perf_counter() - begin) * 1000
            wires = {}
            received = {}
            for kind, packet in [('P', p), ('F', f)]:
                request = {'version': K.VERSION, 'scene': scene, 'g': g, 'service': kind,
                           'scope': 'all_current_peer_tracks', 'coordinate_frame': 'ego_at_t',
                           'ego_pose_at_t': M.load_pose(split, g).tolist(),
                           'history_seconds': 1., 'horizon_seconds': 5.}
                request_wire, response_wire = K.encode(request), K.encode(packet)
                received[kind] = json.loads(response_wire)
                wires[kind] = {'request_bytes': len(request_wire), 'response_bytes': len(response_wire)}
                messages.write(K.encode({'scene': scene, 't': t, 'g': g, 'service': kind,
                    'request_json': request_wire.decode('utf-8'), 'response_json': response_wire.decode('utf-8'),
                    'request_bytes': len(request_wire), 'response_bytes': len(response_wire),
                    'sender_processing_ms': p_ms if kind == 'P' else p_ms + f_ms,
                    'sender_detection_tracking_ms': None, 'network_ms': None,
                    'transport': 'in_process_json_roundtrip', 'upstream_inputs': 'cached_tracks'}) + b'\n')
                totals[kind + '_request_bytes'] += len(request_wire)
                totals[kind + '_response_bytes'] += len(response_wire)
            # Deliberately compute again after decoding P: information-equivalent local baseline.
            local_from_p = K.forecast(received['P'])
            if local_from_p != received['F']:
                raise ValueError('same-information local/remote compute mismatch')
            reference = causal_reference(split, g)
            actions = {}
            for action in ('ego', 'P', 'F', 'PF'):
                used = [] if action == 'ego' else list(action)
                begin = perf_counter()
                objects, audit = K.receive(ego, remote_p=received['P'] if 'P' in used else None,
                                          remote_f=received['F'] if 'F' in used else None)
                plan = K.choose_speed(reference, objects)
                actions[action] = {'used_services': used, 'objects': objects, 'association': audit,
                                   'plan': plan, 'receiver_planning_ms': (perf_counter() - begin) * 1000,
                                   'request_bytes': sum(wires[k]['request_bytes'] for k in used),
                                   'response_bytes': sum(wires[k]['response_bytes'] for k in used)}
            if not actions['P']['objects'] == actions['F']['objects'] == actions['PF']['objects']:
                raise ValueError('same-information receiver mismatch')
            decisions.write(K.encode({'scene': scene, 't': t, 'g': g,
                'reference': reference.tolist(), 'ego_forecast': ego, 'actions': actions}) + b'\n')
            totals['frames'] += 1
            n_peer = actions['F']['association']['peer_only']
            totals['frames_with_peer_only'] += int(n_peer > 0)
            totals['peer_only_states'] += n_peer
            totals['same_information_equal_frames'] += 1
            totals['ego_changed_by_peer_frames'] += int(actions['ego']['plan']['scale'] != actions['F']['plan']['scale'])
    (out / 'generation_summary.json').write_bytes(K.encode(totals) + b'\n')
    print(json.dumps({'output': str(out), **totals}, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene')
    ap.add_argument('--role', choices=['validation', 'train', 'test'], default='validation')
    ap.add_argument('--out')
    a = ap.parse_args()
    run(a.scene, a.role, a.out)
