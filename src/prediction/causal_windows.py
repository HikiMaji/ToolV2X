"""GT-free windows from existing WORLD tracks; model integration is separate.

Each window contains only [t-10,t], expressed in ego(t), in metres/radians.
Raw IDs remain source-local. Missing states are masked, never interpolated.
All current tracks are retained; eligible marks >=2 available tracker states.
An available state can be a causal tracker prediction on a missed detection;
valid is not a detection-match mask.
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def make_window(frames, t, start, ego_pose_t, source):
    if t - 10 < start:
        raise ValueError('history crosses scene boundary')
    current = np.asarray(frames[t])
    ids = current[:, 0].astype(np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('duplicate current track ID')
    states = np.zeros((len(ids), 11, 7), dtype=np.float32)
    valid = np.zeros((len(ids), 11), dtype=bool)
    scores = np.zeros((len(ids), 11), dtype=np.float32)
    index = {int(i): k for k, i in enumerate(ids)}
    for h, g in enumerate(range(t - 10, t + 1)):
        rows = np.asarray(frames[g])
        if rows.ndim != 2 or rows.shape[1] != 9 or not np.isfinite(rows).all():
            raise ValueError('invalid world tracks at frame %d' % g)
        boxes = M.boxes_to_ego(rows[:, 1:8], ego_pose_t)
        for j, row in enumerate(rows):
            k = index.get(int(row[0]))
            if k is not None:
                states[k, h] = boxes[j]
                valid[k, h] = True
                scores[k, h] = row[8]
    return {'source': source, 'g': t, 'track_ids': ids, 'states': states,
            'valid': valid, 'scores': scores, 'eligible': valid.sum(axis=1) >= 2,
            'time_seconds': np.arange(-10, 1, dtype=np.float32) / M.FPS}


def export(config, split, scene_names, scenes=None):
    path = os.path.join(ROOT, 'outputs', 'tracks', split, config + '_world.pkl')
    with open(path, 'rb') as f:
        tracks = pickle.load(f)
    if tracks['meta']['frame'] != 'world' or tracks['meta']['split'] != split:
        raise ValueError('wrong tracking provenance')
    out = os.path.join(ROOT, 'outputs', 'causal_windows_v1', split, config)
    os.makedirs(out, exist_ok=True)
    summary = {'config': config, 'split': split, 'scenes': 0, 'frames': 0,
               'current_targets': 0, 'eligible_targets': 0}
    if len(scene_names) != len(M.seq_ranges(split)):
        raise ValueError('scene manifest does not match source split')
    for (seq, start, end), name in zip(M.seq_ranges(split), scene_names):
        if scenes is not None and name not in scenes:
            continue
        frames = tracks['seqs'][seq]
        if set(frames) != set(range(start, end)):
            raise ValueError('incomplete track scene ' + name)
        windows = {}
        for g in range(start + 10, end):
            windows[g - start] = make_window(frames, g, start, M.load_pose(split, g), config)
        dest = os.path.join(out, name + '.pkl')
        if os.path.exists(dest):
            raise FileExistsError('preserve previous artifact: ' + dest)
        with open(dest, 'wb') as f:
            pickle.dump({'meta': {'protocol': 'causal_windows_v1', 'source_path': path,
                                 'scene': name, 'split': split, 'coordinate_frame': 'ego_at_t',
                                 'yaw_unit': 'radian', 'dt_seconds': .1, 'gt_access': False,
                                 'native_cmp_ready': False}, 'windows': windows}, f)
        summary['scenes'] += 1
        summary['frames'] += len(windows)
        summary['current_targets'] += sum(len(w['track_ids']) for w in windows.values())
        summary['eligible_targets'] += sum(int(w['eligible'].sum()) for w in windows.values())
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=['no_fusion', 'no_fusion_cav1', 'cobevt'])
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--manifest', required=True, help='protocol_audit split_manifest.json')
    a = ap.parse_args()
    with open(a.manifest) as f:
        manifest = json.load(f)
    selected = manifest['train_scenes'] + manifest['validation_scenes'] if a.split == 'train' else manifest['test_scenes']
    print(json.dumps(export(a.config, a.split, manifest['source_scenes'][a.split], set(selected)), ensure_ascii=False))
