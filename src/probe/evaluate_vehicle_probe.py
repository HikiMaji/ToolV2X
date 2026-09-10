"""Offline GT scoring for a SAVED vehicle kinematic probe. Never called by tools."""
import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from probe import kinematic_tools as K


def evaluate(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'run.json').read_text())
    with open(run_dir / 'decisions.jsonl') as f:
        decisions = [json.loads(line) for line in f]
    if [r['t'] for r in decisions] != meta['frames'] or any(r['scene'] != meta['scene'] for r in decisions):
        raise ValueError('decision frame manifest mismatch')
    split = meta['source_split']

    @lru_cache(maxsize=None)
    def gt_frame(g):
        boxes, ids = M.load_gt(split, g)
        # Offline ego annotation exclusion. Cooperating CAV remains a target.
        keep = np.linalg.norm(boxes[:, :2], axis=1) >= K.SELF_METRES
        return M.boxes_to_world(boxes[keep], M.load_pose(split, g)), ids[keep]

    rows = []
    for record in decisions:
        g = record['g']
        if M.seq_of(g, split)[1] != record['t'] or set(record['actions']) != {'ego', 'P', 'F', 'PF'}:
            raise ValueError('action/frame provenance mismatch')
        pose = M.load_pose(split, g)
        current_world, current_ids = gt_frame(g)
        current = M.boxes_to_ego(current_world, pose)
        future = {}
        for k in range(1, 51):
            if M.seq_of(g + k, split)[0] != M.seq_of(g, split)[0]:
                raise ValueError('GT horizon crosses scene')
            world, ids = gt_frame(g + k)
            boxes = M.boxes_to_ego(world, pose)
            for box, oid in zip(boxes, ids):
                future.setdefault(int(oid), np.full((50, 2), np.nan))[k - 1] = box[:2]
        ego_future = M.ego_future_waypoints(split, g, horizon_frames=30, step=1)
        if ego_future is None:
            raise ValueError('missing offline ego reference')
        observed = sum(np.isfinite(x).all(axis=1).sum() for x in future.values())
        for action, result in record['actions'].items():
            objects = result['objects']
            pairs = K.match_positions([o['state'][:2] for o in objects], current[:, :2])
            errors = []
            final_errors = []
            for i, j in pairs:
                truth = future.get(int(current_ids[j]))
                if truth is None:
                    continue
                mask = np.isfinite(truth).all(axis=1)
                error = np.linalg.norm(np.asarray(objects[i]['trajectory']) - truth, axis=1)
                if mask.any():
                    errors.append(float(error[mask].mean()))
                if mask[-1]:
                    final_errors.append(float(error[-1]))
            trajectory = np.asarray(result['plan']['trajectory'])
            proximity = any((np.linalg.norm(truth[:30] - trajectory, axis=1) < 2.5).any()
                            for truth in future.values())
            row = {'scene': record['scene'], 't': record['t'], 'g': g, 'action': action,
                   'input_targets': len(objects), 'gt_targets_at_t': len(current_ids),
                   'matched_at_t': len(pairs), 'unmatched_predictions': len(objects) - len(pairs),
                   'scored_targets': len(errors),
                   'mean_target_ADE_observed': float(np.mean(errors)) if errors else None,
                   'mean_target_FDE5_observed': float(np.mean(final_errors)) if final_errors else None,
                   'ego_L2_observed_3s': float(np.linalg.norm(trajectory - ego_future, axis=1).mean()),
                   'observed_proximity_3s': int(proximity),
                   'GT_future_coverage': observed / float(50 * len(future)) if future else None,
                   'scale': result['plan']['scale'], 'feasible': int(result['plan']['feasible']),
                   'peer_only': result['association']['peer_only'],
                   'application_bytes': result['request_bytes'] + result['response_bytes']}
            rows.append(row)
    with open(run_dir / 'offline_per_frame.csv', 'x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'scope': 'single-scene kinematic interface diagnostic', 'scene': meta['scene'],
               'frames': len(decisions), 'GT_access': 'offline scoring only', 'actions': {}}
    for action in ('ego', 'P', 'F', 'PF'):
        subset = [r for r in rows if r['action'] == action]
        scores = {}
        for field in ('input_targets', 'matched_at_t', 'unmatched_predictions', 'mean_target_ADE_observed',
                      'ego_L2_observed_3s', 'observed_proximity_3s', 'application_bytes', 'peer_only'):
            values = [r[field] for r in subset if r[field] is not None]
            scores[field] = float(np.mean(values)) if values else None
        scores['frames_without_scored_targets'] = sum(r['scored_targets'] == 0 for r in subset)
        summary['actions'][action] = scores
    with open(run_dir / 'offline_summary.json', 'x') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    evaluate(ap.parse_args().run_dir)
