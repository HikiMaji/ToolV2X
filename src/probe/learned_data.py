"""Offline supervision and paired scoring, kept out of online predictor imports."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from probe import kinematic_tools as K
from probe import learned_predictor as L


def match_labels(objects, current, ids, future):
    n = len(objects)
    y, mask, matched_ids = np.zeros((n, 50, 2)), np.zeros((n, 50), bool), np.full(n, -1, int)
    indices = [i for i, o in enumerate(objects) if np.linalg.norm(o['state'][:2]) >= K.SELF_METRES]
    for a, j in K.match_positions([objects[i]['state'][:2] for i in indices], current[:, :2]):
        i = indices[a]
        matched_ids[i] = int(ids[j])
        if int(ids[j]) in future:
            truth = future[int(ids[j])]
            mask[i] = np.isfinite(truth).all(axis=1)
            y[i, mask[i]] = truth[mask[i]]
    return y, mask, matched_ids


def paired_errors(a, b, truth, mask):
    a, b, truth, mask = map(np.asarray, (a, b, truth, mask))
    if a.shape != b.shape or a.shape != truth.shape or mask.shape != a.shape[:2]:
        raise ValueError('paired prediction dimensions differ')
    ea = np.linalg.norm(a - truth, axis=-1)
    eb = np.linalg.norm(b - truth, axis=-1)
    selected = mask.any(axis=1)
    ma = (ea * mask).sum(axis=1)[selected] / mask.sum(axis=1)[selected]
    mb = (eb * mask).sum(axis=1)[selected] / mask.sum(axis=1)[selected]
    end = mask[:, -1]
    return {'targets': int(selected.sum()), 'future_points': int(mask.sum()),
            'ADE_a': float(ma.mean()) if len(ma) else None,
            'ADE_b': float(mb.mean()) if len(mb) else None,
            'delta_ADE_b_minus_a': float((mb - ma).mean()) if len(ma) else None,
            'harm_fraction': float((mb > ma + 1e-6).mean()) if len(ma) else None,
            'FDE5_a': float(ea[end, -1].mean()) if end.any() else None,
            'FDE5_b': float(eb[end, -1].mean()) if end.any() else None,
            'FDE5_targets': int(end.sum())}


def scene_inputs(role, scene):
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    if scene not in manifest[role + '_scenes']:
        raise ValueError('scene outside role manifest')
    split = 'test' if role == 'test' else 'train'
    frames = [int(t) for s, t in json.loads(
        (ROOT / ('outputs/protocol_audit/%s_frames.json' % role)).read_text()) if s == scene]
    windows = {}
    for source in ('no_fusion', 'no_fusion_cav1'):
        path = ROOT / 'outputs/causal_windows_v1' / split / source / (scene + '.pkl')
        with open(path, 'rb') as f:
            artifact = pickle.load(f)
        meta = artifact['meta']
        if (meta['protocol'] != 'causal_windows_v1' or meta['scene'] != scene or meta['split'] != split
                or meta['coordinate_frame'] != 'ego_at_t' or meta['gt_access'] is not False
                or not set(frames).issubset(artifact['windows'])):
            raise ValueError('invalid source provenance or coverage')
        windows[source] = artifact['windows']
    if not frames or len(frames) != len(set(frames)):
        raise ValueError('empty or duplicate frame manifest')
    return split, frames, windows


def gt_cache(split, start, end):
    cache = {}
    for g in range(start, end + 1):
        boxes, ids = M.load_gt(split, g)
        keep = np.linalg.norm(boxes[:, :2], axis=1) >= K.SELF_METRES
        # Double precision avoids quantizing large world-coordinate translations.
        cache[g] = M.boxes_to_world(boxes[keep].astype(float), M.load_pose(split, g).astype(float)), ids[keep]
    return cache


def truth_at(cache, split, g):
    if M.seq_of(g, split)[0] != M.seq_of(g + 50, split)[0]:
        raise ValueError('future labels cross scene boundary')
    pose = M.load_pose(split, g).astype(float)
    current_world, ids = cache[g]
    current = M.boxes_to_ego(current_world, pose)
    future = {}
    for h in range(1, 51):
        world, fids = cache[g + h]
        for box, oid in zip(M.boxes_to_ego(world, pose), fids):
            future.setdefault(int(oid), np.full((50, 2), np.nan))[h - 1] = box[:2]
    return current, ids, future


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())
    (out / 'manifest.json').write_bytes(K.encode(manifest) + b'\n')
    ledger = []
    for role in ('train', 'validation'):
        (out / role).mkdir()
        for scene in manifest[role + '_scenes']:
            split, frames, windows = scene_inputs(role, scene)
            g0 = windows['no_fusion'][frames[0]]['g']
            cache = gt_cache(split, g0, windows['no_fusion'][frames[-1]]['g'] + 50)
            columns = {k: [] for k in ('x_history', 'x_state', 'residual', 'mask', 'eligible', 't', 'source', 'track_id')}
            for t in frames:
                g = windows['no_fusion'][t]['g']
                current, ids, future = truth_at(cache, split, g)
                for source_id, source in enumerate(('no_fusion', 'no_fusion_cav1')):
                    window = windows[source][t]
                    if window['g'] != g or window['source'] != source:
                        raise ValueError('source frame mismatch')
                    history = K.history_packet(window)
                    x, base = L.features(history)
                    state, _ = L.features(L.summary_packet(history))
                    y, mask, _ = match_labels(base['objects'], current, ids, future)
                    baseline = np.asarray([o['trajectory'] for o in base['objects']]).reshape(-1, 50, 2)
                    residual = np.einsum('nhc,nck->nhk', y - baseline, L.rotations(base['objects']))
                    residual[~mask] = 0.
                    n = len(x)
                    for key, value in [('x_history', x), ('x_state', state), ('residual', residual),
                        ('mask', mask), ('eligible', window['eligible']), ('t', np.full(n, t)),
                        ('source', np.full(n, source_id)), ('track_id', window['track_ids'])]:
                        columns[key].append(value)
            arrays = {k: np.concatenate(v).astype(np.float32 if k in ('x_history', 'x_state', 'residual')
                                                else bool if k in ('mask', 'eligible') else np.int64)
                      for k, v in columns.items()}
            np.savez_compressed(str(out / role / (scene + '.npz')), **arrays)
            row = {'role': role, 'scene': scene, 'frames': len(frames), 'all_online_targets': len(arrays['mask']),
                   'labelled_targets': int(arrays['mask'].any(axis=1).sum()),
                   'labelled_future_points': int(arrays['mask'].sum()),
                   'eligible_labelled_targets': int((arrays['mask'].any(axis=1) & arrays['eligible']).sum())}
            ledger.append(row)
            print(json.dumps(row), flush=True)
    (out / 'preparation.json').write_bytes(K.encode({'scope': 'offline supervision only; no test labels',
        'all_online_targets_retained': True, 'GT_used_for_labels_only': True, 'scenes': ledger}) + b'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    prepare(ap.parse_args().out)
