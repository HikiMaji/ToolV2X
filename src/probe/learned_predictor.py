"""Shared, single-target ridge residual baseline. Online operations use no GT.

This is a capability screen, not MTR or a proposed ToolV2X architecture.
Both summary and history views train the SAME weights. No source/ID/scene feature.
"""
import copy
import numpy as np
from probe import kinematic_tools as K

PREDICTOR = 'ridge_residual_history_v1'
CURRENT_FEATURES = 8
FEATURES = CURRENT_FEATURES + 22 + 11 + 11 + 1


def summary_packet(history):
    base = K.forecast(history)
    packet = {k: copy.deepcopy(v) for k, v in base.items()
              if k not in ('objects', 'predictor', 'horizon_frames')}
    packet['kind'] = 'P_state'
    packet['objects'] = [{k: copy.deepcopy(v) for k, v in o.items() if k != 'trajectory'}
                         for o in base['objects']]
    return packet


def features(packet):
    if packet['kind'] == 'P_history':
        base = K.forecast(packet)
    elif packet['kind'] == 'P_state':
        K.check_packet(packet, 'P_state')
        base = copy.deepcopy(packet)
        base.update(kind='F_cv', predictor='ols_cv_history_v1', horizon_frames=50)
        for o in base['objects']:
            o['trajectory'] = (np.asarray(o['state'][:2]) +
                np.arange(1, 51)[:, None] * .1 * o['velocity']).tolist()
        K.check_forecast(base)
    else:
        raise ValueError('expected P_history or P_state')
    x = np.zeros((len(base['objects']), FEATURES), dtype=np.float64)
    for i, o in enumerate(base['objects']):
        yaw = o['state'][6]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        x[i, :CURRENT_FEATURES] = np.r_[1., np.asarray(o['velocity']) @ rotation / 10.,
            np.asarray(o['state'][3:6]) / 10., o['confidence'], o['history_count'] / 11.]
        if packet['kind'] == 'P_history':
            h = packet['objects'][i]
            valid = np.asarray(h['valid'], bool)
            residual = (np.asarray(h['history'])[:, :2] - o['state'][:2] -
                np.asarray(packet['time_seconds'])[:, None] * o['velocity']) @ rotation
            residual[~valid] = 0.
            x[i, CURRENT_FEATURES:] = np.r_[residual.ravel(), valid,
                np.asarray(h['scores']) * valid, 1.]
    if not np.isfinite(x).all():
        raise ValueError('non-finite model features')
    return x, base


def rotations(objects):
    yaw = np.array([o['state'][6] for o in objects])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c, -s, s, c], axis=-1).reshape(-1, 2, 2)


def forecast(packet, weights):
    if weights.shape != (50, FEATURES, 2) or not np.isfinite(weights).all():
        raise ValueError('invalid learned weights')
    x, base = features(packet)
    residual = np.einsum('nf,hfc->nhc', x, weights)
    # Row vectors: residual in target-heading coordinates -> ego(t) coordinates.
    correction = np.einsum('nhc,nkc->nhk', residual, rotations(base['objects']))
    for i, o in enumerate(base['objects']):
        if o['history_count'] >= 2:
            o['trajectory'] = (np.asarray(o['trajectory']) + correction[i]).tolist()
    base.update(kind='F_ridge', predictor=PREDICTOR)
    K.check_forecast(base)
    return base
