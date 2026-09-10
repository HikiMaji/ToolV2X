"""Independent polynomial/linear-algebra reconstruction of learned replay artifacts."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M


def independent_predictions(window, weights):
    result = {}
    for i, tid in enumerate(window['track_ids']):
        state = window['states'][i].astype(float)
        mask = window['valid'][i]
        times = window['time_seconds'].astype(float)
        velocity = np.polyfit(times[mask], state[mask, :2], 1)[0] if mask.sum() >= 2 else np.zeros(2)
        yaw = state[-1, 6]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        x = np.zeros(53)
        x[:8] = [1., *(velocity @ rotation / 10.), *(state[-1, 3:6] / 10.),
                 window['scores'][i, -1], mask.sum() / 11.]
        summary = x.copy()
        residual = (state[:, :2] - state[-1, :2] - times[:, None] * velocity) @ rotation
        residual[~mask] = 0.
        x[8:] = np.r_[residual.ravel(), mask, window['scores'][i] * mask, 1.]
        cv = state[-1, :2] + np.arange(1, 51)[:, None] * .1 * velocity
        full = cv + np.stack([x @ w for w in weights]) @ rotation.T if mask.sum() >= 2 else cv.copy()
        current = cv + np.stack([summary @ w for w in weights]) @ rotation.T if mask.sum() >= 2 else cv.copy()
        result['%s:%d' % (window['source'], tid)] = dict(state=state[-1], velocity=velocity,
            cv=cv, history=full, summary=current, x_history=x, x_state=summary)
    return result


def verify_supervision(root, weights, manifest):
    counts = dict(supervision_targets_checked=0, supervision_frames_checked=0,
                  max_label_residual_abs_error_m=0., online_targets_in_training_artifacts=0)
    for role in ('train', 'validation'):
        for scene in manifest[role + '_scenes']:
            with np.load(str(root / 'data' / role / (scene + '.npz')), allow_pickle=False) as d:
                data = {k: d[k] for k in d.files}
            windows = {}
            for source in ('no_fusion', 'no_fusion_cav1'):
                with open(ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl'), 'rb') as f:
                    windows[source] = pickle.load(f)['windows']
            frames = [t for s, t in json.loads((ROOT / ('outputs/protocol_audit/%s_frames.json' % role)).read_text()) if s == scene]
            assert len(data['t']) == sum(len(w[t]['track_ids']) for w in windows.values() for t in frames)
            counts['online_targets_in_training_artifacts'] += len(data['t'])
            for t in sorted({frames[0], frames[len(frames) // 2], frames[-1]}):
                g = windows['no_fusion'][t]['g']
                pose = M.load_pose('train', g).astype(float)
                current, ids = M.load_gt('train', g)
                keep = np.linalg.norm(current[:, :2], axis=1) >= 1.5
                current, ids = current[keep], ids[keep]
                future = {}
                for h in range(1, 51):
                    boxes, fids = M.load_gt('train', g + h)
                    relative = np.linalg.inv(pose) @ M.load_pose('train', g + h).astype(float)
                    for box, oid in zip(boxes, fids):
                        if np.linalg.norm(box[:2]) < 1.5:
                            continue
                        xy = (relative @ np.r_[box[:3].astype(float), 1.])[:2]
                        future.setdefault(int(oid), np.full((50, 2), np.nan))[h - 1] = xy
                for sid, source in enumerate(('no_fusion', 'no_fusion_cav1')):
                    w = windows[source][t]
                    rows = np.where((data['t'] == t) & (data['source'] == sid))[0]
                    np.testing.assert_array_equal(data['track_id'][rows], w['track_ids'])
                    np.testing.assert_array_equal(data['eligible'][rows], w['valid'].sum(axis=1) >= 2)
                    expected = independent_predictions(w, weights)
                    usable = np.where(np.linalg.norm(w['states'][:, -1, :2], axis=1) >= 1.5)[0]
                    matches = {}
                    if len(usable) and len(ids):
                        distance = np.linalg.norm(w['states'][usable, -1, :2, None].transpose(0, 2, 1) - current[None, :, :2], axis=-1)
                        ii, jj = linear_sum_assignment(np.where(distance <= 2., distance, 1e6))
                        matches = {int(usable[i]): int(ids[j]) for i, j in zip(ii, jj) if distance[i, j] <= 2.}
                    for i, row in enumerate(rows):
                        o = expected['%s:%d' % (source, w['track_ids'][i])]
                        for view in ('history', 'state'):
                            np.testing.assert_allclose(data['x_' + view][row], o['x_' + view], atol=2e-5, rtol=0)
                        truth = future.get(matches.get(i), np.full((50, 2), np.nan))
                        mask = np.isfinite(truth).all(axis=1)
                        np.testing.assert_array_equal(data['mask'][row], mask)
                        yaw = o['state'][6]
                        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                        residual = (truth[mask] - o['cv'][mask]) @ rotation
                        if mask.any():
                            error = float(np.max(np.abs(data['residual'][row, mask] - residual)))
                            counts['max_label_residual_abs_error_m'] = max(counts['max_label_residual_abs_error_m'], error)
                            np.testing.assert_allclose(data['residual'][row, mask], residual, atol=2e-5, rtol=0)
                        counts['supervision_targets_checked'] += 1
                counts['supervision_frames_checked'] += 1
    return counts


def verify(root):
    root = Path(root)
    with np.load(str(root / 'model/model.npz'), allow_pickle=False) as d:
        weights, metadata = d['weights'], json.loads(str(d['metadata']))
    with np.load(str(root / 'model/fit_statistics.npz'), allow_pickle=False) as d:
        penalty = np.eye(53)
        penalty[0, 0] = 0.
        lhs = np.matmul(d['gram'] / d['count'][:, None, None] + metadata['chosen_alpha'] * penalty, weights)
        rhs = d['rhs'] / d['count'][:, None, None]
        normal_error = float(np.max(np.abs(lhs - rhs)))
        np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    assert metadata['train_scenes'] == manifest['train_scenes'] and not metadata['test_used']
    counts = dict(frames=0, messages=0, source_forecasts_recomputed=0, received_trajectories_recomputed=0,
                  same_information_frames=0, max_trajectory_abs_error_m=0., normal_equation_abs_error=normal_error)
    counts.update(verify_supervision(root, weights, manifest))
    for scene in manifest['validation_scenes']:
        path = root / 'replay' / scene
        meta = json.loads((path / 'run.json').read_text())
        with open(path / 'decisions.jsonl') as f:
            records = [json.loads(line) for line in f]
        with open(path / 'messages.jsonl') as f:
            messages = [json.loads(line) for line in f]
        lookup = {(m['t'], m['service']): m for m in messages}
        frames = [t for s, t in json.loads((ROOT / 'outputs/protocol_audit/validation_frames.json').read_text()) if s == scene]
        assert [r['t'] for r in records] == frames == meta['frames']
        assert len(lookup) == len(messages) == 4 * len(records)
        windows = {}
        for source in ('no_fusion', 'no_fusion_cav1'):
            with open(ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl'), 'rb') as f:
                windows[source] = pickle.load(f)['windows']
        for record in records:
            t, g = record['t'], record['g']
            expected = {s: independent_predictions(w[t], weights) for s, w in windows.items()}
            packets = {}
            for service in ('P', 'F', 'P_history', 'F_cv'):
                m = lookup[(t, service)]
                req, packet = json.loads(m['request_json']), json.loads(m['response_json'])
                packets[service] = packet
                assert req['g'] == packet['g'] == m['g'] == g and req['scene'] == scene
                assert len(m['request_json'].encode()) == m['request_bytes']
                assert len(m['response_json'].encode()) == m['response_bytes']
                assert m['network_ms'] is None and m['sender_detection_tracking_ms'] is None
                assert {o['key'] for o in packet['objects']} == set(expected['no_fusion_cav1'])
            for i, o in enumerate(packets['P_history']['objects']):
                np.testing.assert_array_equal(o['history'], windows['no_fusion_cav1'][t]['states'][i])
                np.testing.assert_array_equal(o['valid'], windows['no_fusion_cav1'][t]['valid'][i])
                np.testing.assert_array_equal(o['scores'], windows['no_fusion_cav1'][t]['scores'][i])
            for o in packets['P']['objects']:
                assert 'history' not in o and 'trajectory' not in o
                np.testing.assert_allclose(o['velocity'], expected['no_fusion_cav1'][o['key']]['velocity'], atol=2e-6, rtol=0)
            for source, packet in [('no_fusion', record['ego_forecast']), ('no_fusion_cav1', packets['F'])]:
                for o in packet['objects']:
                    np.testing.assert_allclose(o['trajectory'], expected[source][o['key']]['history'], atol=2e-5, rtol=0)
                    counts['source_forecasts_recomputed'] += 1
            local = {k: o for k, o in expected['no_fusion'].items() if np.linalg.norm(o['state'][:2]) >= 1.5}
            peer = {k: o for k, o in expected['no_fusion_cav1'].items() if np.linalg.norm(o['state'][:2]) >= 1.5}
            for action, a in record['actions'].items():
                services = meta['actions'][action]
                seen = [k for o in a['objects'] for k in o['evidence']]
                assert set(seen) == (set(local) | set(peer) if services else set(local))
                assert len(seen) == len(set(seen))
                for field in ('request_bytes', 'response_bytes'):
                    assert a[field] == sum(lookup[(t, s)][field] for s in services)
                for o in a['objects']:
                    traces = []
                    for key in o['evidence']:
                        mode = 'cv' if action in ('ego_cv', 'F_cv') else 'summary' if key in peer and action == 'P' else 'history'
                        traces.append((local if key in local else peer)[key][mode])
                    trajectory = np.mean(traces, axis=0)
                    error = float(np.max(np.abs(np.asarray(o['trajectory']) - trajectory)))
                    counts['max_trajectory_abs_error_m'] = max(counts['max_trajectory_abs_error_m'], error)
                    np.testing.assert_allclose(o['trajectory'], trajectory, atol=2e-5, rtol=0)
                    counts['received_trajectories_recomputed'] += 1
            a = record['actions']
            assert a['F']['objects'] == a['PF']['objects'] == a['P_history']['objects']
            pose = M.load_pose('train', g).astype(float)
            past = [(np.linalg.inv(pose) @ M.load_pose('train', h).astype(float))[:2, 3] for h in range(g - 10, g + 1)]
            v = np.polyfit(np.arange(-10, 1) * .1, past, 1)[0]
            np.testing.assert_allclose(record['reference'], np.arange(1, 31)[:, None] * .1 * v, atol=2e-5, rtol=0)
            counts['same_information_frames'] += 1
            counts['frames'] += 1
        counts['messages'] += len(messages)
    result = dict(status='PASS', scope='numerical/interface checks only; no capability-gain claim', **counts)
    with open(root / 'artifact_verification.json', 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    verify(ap.parse_args().root)
