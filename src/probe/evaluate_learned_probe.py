"""Score completed replay offline, with fixed common-target labels for differences."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from probe import kinematic_tools as K
from probe import learned_data as D
from probe import learned_predictor as L
from probe.run_learned_probe import ACTION_SERVICES, load_model


def common_arrays(a, b, current, ids, future):
    by_key = {o['key']: o for o in b}
    a = [o for o in a if o['key'] in by_key]
    b = [by_key[o['key']] for o in a]
    if any(x['state'] != y['state'] for x, y in zip(a, b)):
        raise ValueError('common-target current anchors differ')
    truth, mask, _ = D.match_labels(a, current, ids, future)
    return (np.asarray([o['trajectory'] for o in a]).reshape(-1, 50, 2),
            np.asarray([o['trajectory'] for o in b]).reshape(-1, 50, 2), truth, mask)


def evaluate(replay):
    replay = Path(replay)
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    result = {'scope': 'validation used for model selection; fixed-cache diagnostic only',
              'test_used': False, 'scenes': [], 'GT_access': 'offline after generation only'}
    frame_rows = []
    for scene in manifest['validation_scenes']:
        dest = replay / scene
        meta = json.loads((dest / 'run.json').read_text())
        weights, _ = load_model(meta['model_path'])
        with open(dest / 'decisions.jsonl') as f:
            records = [json.loads(line) for line in f]
        frames = [t for s, t in json.loads((ROOT / 'outputs/protocol_audit/validation_frames.json').read_text()) if s == scene]
        if [r['t'] for r in records] != frames or frames != meta['frames']:
            raise ValueError('replay frame set differs from independent manifest')
        with open(dest / 'messages.jsonl') as f:
            messages = [json.loads(line) for line in f]
        packets = {(m['t'], m['service']): json.loads(m['response_json']) for m in messages}
        if len(packets) != len(records) * 4 or len(messages) != len(packets):
            raise ValueError('missing/duplicate messages')
        cache = D.gt_cache('train', records[0]['g'], records[-1]['g'] + 50)
        comparisons = {}
        association_audit = {'ego_matched_target_records': 0, 'peer_only_target_records': 0,
            'peer_only_matched_GT_records': 0, 'peer_only_new_GT_records': 0,
            'merged_pairs_with_conflicting_GT_ids': 0, 'merged_pairs_both_GT_matched': 0}
        for record in records:
            if record['scene'] != scene or set(record['actions']) != set(ACTION_SERVICES):
                raise ValueError('wrong scene or action coverage')
            t, g = record['t'], record['g']
            if M.seq_of(g, 'train')[1] != t:
                raise ValueError('global/local frame mismatch')
            current, ids, future = D.truth_at(cache, 'train', g)
            actions = record['actions']
            ego = actions['ego']['objects']
            peer = packets[(t, 'F')]['objects']
            state = L.forecast(packets[(t, 'P')], weights)['objects']
            pairs = {
                'ego_CV_to_learned': (actions['ego_cv']['objects'], ego),
                'peer_CV_to_learned': (packets[(t, 'F_cv')]['objects'], peer),
                'peer_state_to_history': (state, peer),
                'ego_to_P_common': (ego, actions['P']['objects']),
                'ego_to_F_common': (ego, actions['F']['objects']),
                'P_to_F_same_targets': (actions['P']['objects'], actions['F']['objects']),
                'F_cv_to_F_common': (actions['F_cv']['objects'], actions['F']['objects'])}
            for name, (a, b) in pairs.items():
                comparisons.setdefault(name, []).append(common_arrays(a, b, current, ids, future))
            _, _, ego_ids = D.match_labels(ego, current, ids, future)
            _, _, peer_ids = D.match_labels(peer, current, ids, future)
            ego_by_key = {o['key']: int(oid) for o, oid in zip(ego, ego_ids)}
            peer_by_key = {o['key']: int(oid) for o, oid in zip(peer, peer_ids)}
            known = set(ego_ids) - {-1}
            association_audit['ego_matched_target_records'] += int((ego_ids >= 0).sum())
            for o in actions['F']['objects']:
                keys = o['evidence']
                if len(keys) == 2:
                    a, b = ego_by_key[keys[0]], peer_by_key[keys[1]]
                    association_audit['merged_pairs_both_GT_matched'] += int(a >= 0 and b >= 0)
                    association_audit['merged_pairs_with_conflicting_GT_ids'] += int(a >= 0 and b >= 0 and a != b)
                elif keys[0] in peer_by_key:
                    oid = peer_by_key[keys[0]]
                    association_audit['peer_only_target_records'] += 1
                    association_audit['peer_only_matched_GT_records'] += int(oid >= 0)
                    association_audit['peer_only_new_GT_records'] += int(oid >= 0 and oid not in known)
            ego_truth = M.ego_future_waypoints('train', g, horizon_frames=30, step=1)
            coverage = sum(np.isfinite(x[:30]).all(axis=1).sum() for x in future.values())
            for action, a in actions.items():
                plan = np.asarray(a['plan']['trajectory'])
                frame_rows.append(dict(scene=scene, t=t, action=action, scale=a['plan']['scale'],
                    feasible=a['plan']['feasible'], ego_L2_3s=float(np.linalg.norm(plan - ego_truth, axis=1).mean()),
                    observed_proximity_3s=int(any((np.linalg.norm(x[:30] - plan, axis=1) < 2.5).any() for x in future.values())),
                    observed_GT_future_points_3s=int(coverage), GT_future_target_count=len(future),
                    application_bytes=a['request_bytes'] + a['response_bytes'],
                    peer_only=a['association']['peer_only']))
        paired = {name: D.paired_errors(*[np.concatenate([v[i] for v in arrays]) for i in range(4)])
                  for name, arrays in comparisons.items()}
        plans = {}
        for action in ACTION_SERVICES:
            rows = [r for r in frame_rows if r['scene'] == scene and r['action'] == action]
            plans[action] = {k: float(np.mean([r[k] for r in rows])) for k in
                ('ego_L2_3s', 'observed_proximity_3s', 'application_bytes', 'scale', 'feasible', 'peer_only')}
        item = dict(scene=scene, frames=len(records), paired=paired, plans=plans, association_audit=association_audit)
        result['scenes'].append(item)
        print(json.dumps(item), flush=True)
    result['macro_scene_paired'] = {name: {k: float(np.mean([s['paired'][name][k] for s in result['scenes']]))
        for k in ('ADE_a', 'ADE_b', 'delta_ADE_b_minus_a', 'harm_fraction')}
        for name in result['scenes'][0]['paired']}
    with open(replay / 'offline_summary.json', 'x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    with open(replay / 'offline_frames.csv', 'x') as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('replay')
    evaluate(ap.parse_args().replay)
