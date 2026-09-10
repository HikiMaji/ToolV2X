"""Local data audit and independent recording-group/frame manifests. No model run."""
import glob
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M
from prediction.tracks_to_cmp import cmp_scene_names

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'outputs', 'protocol_audit')


def recording(scene):
    match = re.fullmatch(r'(testoutput_CAV_data_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_\d+)?', scene)
    if not match:
        raise ValueError('unrecognized recording name: ' + scene)
    return match.group(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = {s: cmp_scene_names(s) for s in ('train', 'test')}
    test_groups = {recording(n) for n in names['test']}
    candidates = [n for n in names['train'] if recording(n) not in test_groups]
    groups = sorted({recording(n) for n in candidates})
    # Deterministic chronological holdout; no performance-based split selection.
    validation_groups = set(groups[-2:])
    manifest = {'protocol': 'recording_holdout_v1', 'source_scenes': names,
                'train_scenes': [n for n in candidates if recording(n) not in validation_groups],
                'validation_scenes': [n for n in candidates if recording(n) in validation_groups],
                'test_scenes': names['test'],
                'excluded_train_scenes': [n for n in names['train'] if recording(n) in test_groups],
                'note': 'Downstream split only; pretrained detector training independence is NOT established.'}
    poses = {s: np.stack([M.load_pose(s, g) for g in range(M.num_frames(s))]) for s in names}
    tree = cKDTree(poses['train'][:, :3, 3])
    pairs = []
    for test_g, p in enumerate(poses['test']):
        for train_g in tree.query_ball_point(p[:3, 3], 1e-3):
            if np.allclose(poses['train'][train_g], p, rtol=0, atol=1e-4):
                pairs.append([int(train_g), int(test_g)])
    excluded = set(manifest['excluded_train_scenes'])
    residual = [p for p in pairs if names['train'][M.seq_of(p[0], 'train')[0]] not in excluded]
    if residual:
        raise ValueError('pose duplicates remain outside recording embargo')
    frames = {}
    for role in ('train', 'validation', 'test'):
        source = 'test' if role == 'test' else 'train'
        selected = set(manifest[role + '_scenes'])
        # Entire horizon available in the recording; NEVER require a target's future validity.
        frames[role] = [[n, t] for (_, start, end), n in zip(M.seq_ranges(source), names[source])
                        if n in selected for t in range(10, end - start - 50)]
    track_summary = []
    for split in ('train', 'test'):
        for config in ('no_fusion', 'no_fusion_cav1', 'cobevt'):
            path = os.path.join(ROOT, 'outputs', 'tracks', split, config + '_world.pkl')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            counts = []
            for seq, start, end in M.seq_ranges(split):
                part = data['seqs'][seq]
                if set(part) != set(range(start, end)):
                    raise ValueError('missing tracking frames: ' + path)
                for rows in part.values():
                    if rows.ndim != 2 or rows.shape[1] != 9 or not np.isfinite(rows).all():
                        raise ValueError('invalid tracking output: ' + path)
                counts.append(sum(len(a) for a in part.values()))
            track_summary.append({'split': split, 'config': config, 'scenes': len(counts),
                                  'frames': M.num_frames(split), 'states': sum(counts)})
    checks = {'run_utc': datetime.now(timezone.utc).isoformat(),
              'pose_tolerance': {'translation_search_m': .001, 'matrix_atol': .0001, 'rtol': 0},
              'pose_duplicate_pairs': pairs,
              'distinct_duplicated_test_frames': len({p[1] for p in pairs}),
              'remaining_pose_duplicate_pairs_after_embargo': len(residual),
              'role_frame_counts': {k: len(v) for k, v in frames.items()},
              'tracks': track_summary,
              'training_gate': 'HOLD',
              'reasons': ['Native CMP dataset uses GT identity and future-valid target selection',
                          'Native CMP aggregator omits remote-only target outputs',
                          'Native CMP uses 0.01s timestamps for 10Hz records',
                          'No measured P/F payload and latency ledger',
                          'Pretrained detector training split provenance not established']}
    for name, value in [('split_manifest.json', manifest), ('audit.json', checks)] + [
            (k + '_frames.json', v) for k, v in frames.items()]:
        with open(os.path.join(OUT, name), 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write('\n')
    print(json.dumps({'pose_duplicate_pairs': len(pairs),
                      'duplicate_test_frames': checks['distinct_duplicated_test_frames'],
                      'scene_counts': {k: len(manifest[k + '_scenes']) for k in ('train', 'validation', 'test')},
                      'excluded_train_scenes': len(excluded),
                      'frame_counts': checks['role_frame_counts'], 'training_gate': 'HOLD'}, indent=2))


if __name__ == '__main__':
    main()
