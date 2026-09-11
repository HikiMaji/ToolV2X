"""Offline target matching and original CMP loss inputs; never imported online."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from probe.kinematic_tools import history_packet

PROTOCOL = 'mtr_causal_supervision_v1'
MATCH_METRES = 2.
SELF_METRES = 1.5


def make_labels(window, current_boxes, current_ids, future):
    """All GT boxes are in fixed ego(t). Future is 50 (boxes, IDs) pairs or None."""
    history_packet(window)
    boxes, ids = np.asarray(current_boxes, float), np.asarray(current_ids)
    if (boxes.shape != (len(ids), 7) or len(set(ids.tolist())) != len(ids) or
            not np.isfinite(boxes).all() or len(future) != 50):
        raise ValueError('invalid offline GT frames')
    states = np.asarray(window['states'])[:, -1]
    count = len(states)
    matched = np.full(count, -1, np.int64)
    status = np.full(count, 'unmatched', dtype='<U24')
    self_mask = np.linalg.norm(states[:, :2], axis=1) < SELF_METRES
    status[self_mask] = 'ego_self'
    candidates = np.flatnonzero(~self_mask)
    truth = np.flatnonzero(np.linalg.norm(boxes[:, :2], axis=1) >= SELF_METRES)
    distance = np.full(count, np.nan)
    if len(candidates) and len(truth):
        a, b = states[candidates], boxes[truth]
        dist = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
        ratio = a[:, None, 3:6] / b[None, :, 3:6]
        yaw = np.abs((a[:, None, 6] - b[None, :, 6] + np.pi / 2) % np.pi - np.pi / 2)
        allowed = (dist <= MATCH_METRES) & (ratio >= .5).all(-1) & (ratio <= 2).all(-1) & (yaw <= np.pi / 3)
        left, right = linear_sum_assignment(np.where(allowed, dist, 1e6))
        for i, j in zip(left, right):
            if allowed[i, j]:
                k = candidates[i]
                matched[k], distance[k], status[k] = ids[truth[j]], dist[i, j], 'matched_no_future'
    xy = np.zeros((count, 50, 2), np.float32)
    valid = np.zeros((count, 50), bool)
    for step, frame in enumerate(future):
        if frame is None:
            continue
        frame_boxes, frame_ids = map(np.asarray, frame)
        if (frame_boxes.shape != (len(frame_ids), 7) or len(set(frame_ids.tolist())) != len(frame_ids) or
                not np.isfinite(frame_boxes).all()):
            raise ValueError('invalid future labels')
        lookup = {int(i): row for i, row in zip(frame_ids, frame_boxes)}
        for k in np.flatnonzero(matched >= 0):
            if int(matched[k]) in lookup:
                xy[k, step] = lookup[int(matched[k])][:2]
                valid[k, step] = True
    status[valid.any(axis=1)] = 'labelled'
    return dict(protocol=PROTOCOL, source=window['source'], g=int(window['g']),
                track_ids=np.asarray(window['track_ids']).copy(), matched_gt_ids=matched,
                status=status, current_match_distance=distance, xy=xy, valid=valid)


def check_labels(window, labels):
    n = len(window['track_ids'])
    if (labels['protocol'] != PROTOCOL or labels['source'] != window['source'] or labels['g'] != window['g'] or
            not np.array_equal(labels['track_ids'], window['track_ids']) or
            np.shape(labels['xy']) != (n, 50, 2) or np.shape(labels['valid']) != (n, 50) or
            np.asarray(labels['valid']).dtype != bool or not np.isfinite(labels['xy']).all()):
        raise ValueError('supervision source/time/target mismatch')


def subset(window, labels, roi):
    check_labels(window, labels)
    xy = window['states'][:, -1, :2]
    keep = np.flatnonzero((xy[:, 0] >= roi[0]) & (xy[:, 1] >= roi[1]) &
                         (xy[:, 0] <= roi[2]) & (xy[:, 1] <= roi[3]))
    w = dict(window)
    for key in ('states', 'valid', 'scores', 'track_ids', 'eligible'):
        if key in w:
            w[key] = np.asarray(w[key])[keep]
    y = dict(labels)
    for key in ('track_ids', 'matched_gt_ids', 'status', 'current_match_distance', 'xy', 'valid'):
        y[key] = np.asarray(y[key])[keep]
    return w, y


def training_batch(window, labels, center_indices):
    import torch
    from prediction import cmp_adapter as C
    check_labels(window, labels)
    indices = np.asarray(center_indices, dtype=np.int64)
    if not labels['valid'][indices].any(axis=1).all():
        raise ValueError('training centers need valid future supervision')
    batch = C.make_batch(window, indices)
    centers = window['states'][indices, -1]
    delta = labels['xy'][None] - centers[:, None, None, :2]
    c, s = np.cos(centers[:, 6]), np.sin(centers[:, 6])
    rotation = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    # Row vectors times R(yaw) express coordinates in the current target frame.
    future = np.einsum('bnhc,bcd->bnhd', delta, rotation).astype(np.float32)
    mask = np.broadcast_to(labels['valid'], future.shape[:-1]).copy()
    future[~mask] = 0.
    target_mask = labels['valid'][indices]
    last = np.where(target_mask, np.arange(50), -1).max(axis=1)
    batch['input_dict'].update(center_gt_trajs=torch.from_numpy(future[np.arange(len(indices)), indices]),
        center_gt_trajs_mask=torch.from_numpy(target_mask.copy()), center_gt_final_valid_idx=torch.from_numpy(last),
        obj_trajs_future_state=torch.from_numpy(future), obj_trajs_future_mask=torch.from_numpy(mask))
    return batch
