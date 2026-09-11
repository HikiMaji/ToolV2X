"""Materialize only offline MTR labels; online windows remain in separate files."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from common.audit_protocol import recording
from prediction.supervision import PROTOCOL, make_labels

SOURCES = ('no_fusion', 'no_fusion_cav1')


def load_windows(scene, source):
    path = ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl')
    with path.open('rb') as handle:
        artifact = pickle.load(handle)
    meta = artifact['meta']
    if (meta['scene'] != scene or meta['split'] != 'train' or meta['coordinate_frame'] != 'ego_at_t' or
            meta['gt_access'] is not False or meta['dt_seconds'] != .1 or meta['yaw_unit'] != 'radian'):
        raise ValueError('unexpected online window provenance')
    return path, artifact['windows']


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest_dir = ROOT / 'outputs/protocol_audit'
    manifest = json.loads((manifest_dir / 'split_manifest.json').read_text())
    groups = {role: {recording(scene) for scene in manifest[role + '_scenes']}
              for role in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('recording holdout overlap')
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    ledger, reads = [], []
    for role in ('train', 'validation'):
        (out / role).mkdir()
        frames = json.loads((manifest_dir / (role + '_frames.json')).read_text())
        index = []
        for scene in manifest[role + '_scenes']:
            times = [int(t) for name, t in frames if name == scene]
            inputs = {source: load_windows(scene, source) for source in SOURCES}
            first = inputs[SOURCES[0]][1][times[0]]['g']
            start, end = M.seq_of(first, 'train')[2:]
            cache, poses = {}, {}
            for g in range(first, min(end, inputs[SOURCES[0]][1][times[-1]]['g'] + 51)):
                try:
                    pose = np.asarray(M.load_pose('train', g), dtype=float)
                    if pose.shape != (4, 4) or not np.isfinite(pose).all():
                        raise ValueError('invalid offline localization')
                    poses[g] = pose
                    boxes, ids = M.load_gt('train', g)
                    cache[g] = (M.boxes_to_world(boxes.astype(float), pose), ids)
                except FileNotFoundError:
                    cache[g] = None
            reads.append(dict(scene=scene, physical_split='train', frame_start=first,
                frame_end_exclusive=min(end, inputs[SOURCES[0]][1][times[-1]]['g'] + 51),
                gt_directory=M.config_dir('no_fusion', 'train'), pose_directory=M.pose_dir('train'), offline_only=True))
            for source, (window_path, windows) in inputs.items():
                labels_by_t = {}
                counts = Counter()
                for t in times:
                    w = windows[t]
                    g = int(w['g'])
                    if w['source'] != source or g != start + t:
                        raise ValueError('online source/time mismatch')
                    pose = poses.get(g)
                    if pose is None or cache.get(g) is None:
                        current = (np.empty((0, 7)), np.empty(0, np.int64))
                        future = [None] * 50
                    else:
                        current = (M.boxes_to_ego(cache[g][0], pose), cache[g][1])
                        future = [(M.boxes_to_ego(cache[g + h][0], pose), cache[g + h][1])
                                  if g + h < end and cache.get(g + h) is not None else None
                                  for h in range(1, 51)]
                    y = make_labels(w, *current, future)
                    labels_by_t[t] = y
                    counts.update(y['status'].tolist())
                    counts['online_targets'] += len(y['track_ids'])
                    counts['future_points'] += int(y['valid'].sum())
                    counts['supervised_model_targets'] += int(((w['valid'].sum(axis=1) >= 2) & y['valid'].any(axis=1)).sum())
                    index.append(dict(role=role, scene=scene, recording=recording(scene), source=source, t=t, g=g,
                        window_path=str(window_path), label_path=str(Path(role) / (scene + '_' + source + '.pkl'))))
                path = out / role / (scene + '_' + source + '.pkl')
                with path.open('wb') as handle:
                    pickle.dump(dict(protocol=PROTOCOL, role=role, scene=scene, labels=labels_by_t), handle)
                row = dict(role=role, scene=scene, source=source, frames=len(times), counts=dict(counts))
                ledger.append(row)
                print(json.dumps(row), flush=True)
        (out / (role + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in index))
    result = dict(protocol=PROTOCOL, scenes=ledger, read_access=reads, test_labels_read=False,
        history_mask_semantics='available causal tracker output, not per-frame detection match',
        matching=dict(distance_m=2., size_ratio=[.5, 2.], heading_mod_pi_rad=float(np.pi / 3), ego_exclusion_m=1.5),
        pretrained_provenance_verified=False, online_inputs_rewritten=False)
    (out / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    prepare(parser.parse_args().out)
