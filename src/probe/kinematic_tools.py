"""Vehicle-only interface probe: causal history -> constant-velocity forecasts.

P transmits the FULL observed 1s history; F computes from exactly that history.
Their prediction equivalence is an intended control, not evidence of tool utility.
No GT, filesystem, model checkpoint or network access in this module.
"""
import copy
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = 'vehicle_kinematic_v1'
DT = .1
HORIZON = 50
MATCH_METRES = 2.
SELF_METRES = 1.5


def encode(packet):
    return json.dumps(packet, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), sort_keys=True).encode('utf-8')


def history_packet(window):
    states = np.asarray(window['states'], dtype=float)
    valid = np.asarray(window['valid'], dtype=bool)
    scores = np.asarray(window['scores'], dtype=float)
    times = np.asarray(window['time_seconds'], dtype=float)
    ids = np.asarray(window['track_ids'])
    n = len(ids)
    if (states.shape != (n, 11, 7) or valid.shape != (n, 11) or scores.shape != (n, 11)
            or times.shape != (11,) or not np.allclose(times, np.arange(-10, 1) * DT)
            or not valid[:, -1].all() or len(set(ids.tolist())) != n
            or not np.isfinite(states).all() or not np.isfinite(scores).all()):
        raise ValueError('invalid causal history window')
    if ((states[:, -1, 3:6] <= 0).any() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('invalid box sizes or scores')
    objects = [{'key': '%s:%d' % (window['source'], int(ids[i])),
                'history': states[i].tolist(), 'valid': valid[i].tolist(),
                'scores': scores[i].tolist()} for i in range(n)]
    return {'version': VERSION, 'kind': 'P_history', 'source': window['source'],
            'g': int(window['g']), 'coordinate_frame': 'ego_at_t', 'dt_seconds': DT,
            'time_seconds': times.tolist(), 'status': 'ok' if n else 'empty', 'objects': objects}


def check_packet(packet, kind):
    if (packet.get('version') != VERSION or packet.get('kind') != kind
            or packet.get('coordinate_frame') != 'ego_at_t' or packet.get('dt_seconds') != DT
            or not isinstance(packet.get('g'), int) or not isinstance(packet.get('objects'), list)
            or packet.get('status') != ('ok' if packet['objects'] else 'empty')):
        raise ValueError('invalid packet contract')
    keys = [o['key'] for o in packet['objects']]
    if len(set(keys)) != len(keys) or any(not k.startswith(packet['source'] + ':') for k in keys):
        raise ValueError('invalid source-local identity')
    encode(packet)  # Reject non-finite values even in fields unused by the planner.


def forecast(history):
    check_packet(history, 'P_history')
    times = np.asarray(history['time_seconds'])
    if times.shape != (11,) or not np.allclose(times, np.arange(-10, 1) * DT):
        raise ValueError('invalid history time axis')
    objects = []
    future_t = np.arange(1, HORIZON + 1) * DT
    for obj in history['objects']:
        states = np.asarray(obj['history'])
        valid = np.asarray(obj['valid'], dtype=bool)
        scores = np.asarray(obj['scores'])
        if (states.shape != (11, 7) or valid.shape != (11,) or scores.shape != (11,)
                or not valid[-1] or (states[-1, 3:6] <= 0).any()
                or (scores < 0).any() or (scores > 1).any()):
            raise ValueError('invalid target history')
        v = np.zeros(2)
        fallback = None
        if valid.sum() >= 2:
            t = times[valid] - times[valid].mean()
            xy = states[valid, :2]
            v = (t[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / (t @ t)
        else:
            fallback = 'stationary_short_history'
        trajectory = states[-1, :2] + future_t[:, None] * v
        objects.append({'key': obj['key'], 'state': states[-1].tolist(),
                        'velocity': v.tolist(), 'trajectory': trajectory.tolist(),
                        'confidence': float(scores[-1]), 'history_count': int(valid.sum()),
                        'fallback': fallback})
    out = {k: copy.deepcopy(v) for k, v in history.items() if k not in ('objects', 'time_seconds')}
    out.update(kind='F_cv', predictor='ols_cv_history_v1', horizon_frames=HORIZON,
               objects=objects)
    return out


def check_forecast(packet):
    predictors = {'F_cv': 'ols_cv_history_v1', 'F_ridge': 'ridge_residual_history_v1'}
    kind = packet.get('kind')
    if kind not in predictors:
        raise ValueError('unexpected forecast kind')
    check_packet(packet, kind)
    if packet.get('predictor') != predictors[kind] or packet.get('horizon_frames') != HORIZON:
        raise ValueError('unexpected predictor')
    for o in packet['objects']:
        if (np.asarray(o['trajectory']).shape != (HORIZON, 2)
                or np.asarray(o['state']).shape != (7,)
                or np.asarray(o['velocity']).shape != (2,)
                or (np.asarray(o['state'])[3:6] <= 0).any()
                or not 0 <= o['confidence'] <= 1):
            raise ValueError('invalid forecast dimensions or state')


def match_positions(local_xy, remote_xy, threshold=MATCH_METRES):
    if not len(local_xy) or not len(remote_xy):
        return []
    distance = np.linalg.norm(np.asarray(local_xy)[:, None] - np.asarray(remote_xy)[None], axis=-1)
    i, j = linear_sum_assignment(np.where(distance <= threshold, distance, 1e6))
    # ponytail: position-only association can confuse adjacent tracks; use history/box gates for learned-model experiments.
    return [(int(a), int(b)) for a, b in zip(i, j) if distance[a, b] <= threshold]


def receive(ego, remote_p=None, remote_f=None):
    check_forecast(ego)
    remote = None
    if remote_p is not None:
        remote = forecast(remote_p)
    if remote_f is not None:
        check_forecast(remote_f)
        if remote is not None:
            if remote['source'] != remote_f['source'] or remote['g'] != remote_f['g']:
                raise ValueError('P/F source or timestamp mismatch')
            anchors_p = {o['key']: o['state'] for o in remote['objects']}
            anchors_f = {o['key']: o['state'] for o in remote_f['objects']}
            if anchors_p != anchors_f:
                raise ValueError('P/F target anchor mismatch')
        remote = remote_f  # Same peer observations count once in PF.
    if remote is not None and (remote['g'] != ego['g'] or remote['source'] == ego['source']):
        raise ValueError('remote timestamp/source mismatch')
    ego_objs = copy.deepcopy(ego['objects'])
    peer_objs = copy.deepcopy(remote['objects']) if remote is not None else []
    n_before = len(ego_objs) + len(peer_objs)
    # Observable ego-centre heuristic only; partner identity is never excluded.
    ego_objs = [o for o in ego_objs if np.linalg.norm(o['state'][:2]) >= SELF_METRES]
    peer_objs = [o for o in peer_objs if np.linalg.norm(o['state'][:2]) >= SELF_METRES]
    pairs = match_positions([o['state'][:2] for o in ego_objs], [o['state'][:2] for o in peer_objs])
    mapping = dict(pairs)
    used_peer = {b for _, b in pairs}
    out = []
    for i, obj in enumerate(ego_objs):
        obj['evidence'] = [obj['key']]
        if i in mapping:
            peer = peer_objs[mapping[i]]
            obj['trajectory'] = ((np.asarray(obj['trajectory']) + peer['trajectory']) * .5).tolist()
            obj['velocity'] = ((np.asarray(obj['velocity']) + peer['velocity']) * .5).tolist()
            obj['evidence'].append(peer['key'])
        out.append(obj)
    for j, obj in enumerate(peer_objs):
        if j not in used_peer:
            obj['evidence'] = [obj['key']]
            out.append(obj)
    return out, {'ego_targets': len(ego_objs), 'peer_targets': len(peer_objs),
                 'matched': len(pairs), 'peer_only': len(peer_objs) - len(used_peer),
                 'ego_filtered': n_before - len(ego_objs) - len(peer_objs),
                 'fallback_targets': sum(o['fallback'] is not None for o in out)}


def choose_speed(reference, objects):
    """Instant speed scaling on a history-extrapolated reference; proximity proxy only."""
    reference = np.asarray(reference)
    if reference.shape != (30, 2) or not np.isfinite(reference).all():
        raise ValueError('invalid causal reference')
    for scale in (1., .8, .6, .4, .2, 0.):
        trajectory = reference * scale
        conflict = any((np.linalg.norm(np.asarray(o['trajectory'])[:30] - trajectory, axis=1) < 2.5).any()
                       for o in objects)
        if not conflict:
            return {'scale': scale, 'trajectory': trajectory.tolist(), 'feasible': True}
    return {'scale': 0., 'trajectory': np.zeros((30, 2)).tolist(), 'feasible': False}
