"""v2 task semantics and configurable execution parameters; no model execution.

P uses a causal tracking-state motion proxy, not detector-hit history or MTR.
Receipt validation requires the provider's own issued-response registry.
"""
from dataclasses import asdict, dataclass, fields
import re

import numpy as np

from probe.kinematic_tools import encode, history_packet

TASK_VERSION = 'toolv2x_task_v2'
TIMES = [.5, 1., 1.5, 2., 2.5, 3.]
FUTURE_INDICES = [4, 9, 14, 19, 24, 29]
SCORE_DECIMALS = 12
REQUEST_FIELDS = {'version', 'request_id', 'provider', 'scene', 'g', 'coordinate_frame',
                  'tool', 'mode', 'times', 'tau_new', 'tau_old', 'execution_spec',
                  'acquired_field_manifest'}
REF_FIELDS = {'provider', 'scene', 'g', 'track_handle', 'field_kind',
              'producer_version', 'context_version'}


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('invalid ' + label + ' fields')


def _text(value):
    if not isinstance(value, str) or not value or not value.strip() or '\x00' in value:
        raise ValueError('expected nonempty identifier')
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('expected integer at least %d' % minimum)
    return value


def _version(value):
    _text(value)
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', value) is None:
        raise ValueError('semantic version identifiers must not be paths')
    return value


def _version_pair(value):
    _keys(value, ('name', 'revision'), 'version identity')
    return _version(value['name']), _version(value['revision'])


def _array(value, shape):
    try:
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in 'fi' or not np.isfinite(array).all():
            raise ValueError('invalid finite numeric array')
        return array.astype(float)
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid numeric array') from exc


@dataclass(frozen=True)
class ExecutionSpec:
    """Full versioned run configuration; protocol formulas live outside this class."""
    version: str = 'toolv2x_execution_v1'
    profile: str = 'contract_v1'
    sigma_m: float = 5.
    max_targets: int = 4
    max_request_bytes: int = 4096
    max_response_bytes: int = 8192
    max_episode_bytes: int = 24576
    ego_geometry: str = 'circumscribed_circle'
    ego_length_m: float = 4.8
    ego_width_m: float = 2.
    max_plan_speed_mps: float = 80.
    max_plan_acceleration_mps2: float = 30.

    def __post_init__(self):
        if self.version != 'toolv2x_execution_v1' or self.ego_geometry != 'circumscribed_circle':
            raise ValueError('unsupported execution specification')
        _version(self.profile)
        for key in ('max_targets', 'max_request_bytes', 'max_response_bytes', 'max_episode_bytes'):
            _integer(getattr(self, key), 1)
        for key in ('sigma_m', 'ego_length_m', 'ego_width_m', 'max_plan_speed_mps',
                    'max_plan_acceleration_mps2'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError('execution parameter must be finite and positive: ' + key)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _keys(value, [f.name for f in fields(cls)], 'ExecutionSpec')
        return cls(**value)


def validate_provenance(value):
    _keys(value, ('version', 'tracking', 'prediction', 'context'), 'provider provenance')
    if value['version'] != 'toolv2x_provenance_v1':
        raise ValueError('unsupported provenance version')
    for key in ('tracking', 'prediction', 'context'):
        _version_pair(value[key])


def field_key(ref):
    """Content identity excludes request/receipt IDs and execution ranking parameters."""
    _keys(ref, REF_FIELDS, 'field reference')
    if ref['field_kind'] not in ('anchor', 'history', 'forecast'):
        raise ValueError('unknown remote field kind')
    return (_text(ref['provider']), _text(ref['scene']), _integer(ref['g']),
            _integer(ref['track_handle']), ref['field_kind'],
            _version_pair(ref['producer_version']), _version_pair(ref['context_version']))


def _request_spec(request):
    _keys(request, REQUEST_FIELDS, 'task request')
    if (request['version'] != TASK_VERSION or request['coordinate_frame'] != 'ego_at_t' or
            request['tool'] not in ('P', 'F') or request['mode'] not in ('current', 'change')):
        raise ValueError('invalid task protocol/time/coordinate contract')
    for key in ('request_id', 'provider', 'scene'):
        _text(request[key])
    _integer(request['g'])
    if not np.array_equal(_array(request['times'], (6,)), TIMES):
        raise ValueError('invalid request time axis')
    spec = ExecutionSpec.from_dict(request['execution_spec'])
    paths = [_array(request['tau_new'], (6, 2))]
    if request['mode'] == 'change':
        paths.append(_array(request['tau_old'], (6, 2)))
        if np.array_equal(paths[0], paths[1]):
            raise ValueError('change requires different old/new plans')
    elif request['tau_old'] is not None:
        raise ValueError('current must not contain an old plan')
    for path in paths:
        velocity = np.diff(np.vstack([np.zeros((1, 2)), path]), axis=0) / .5
        if (np.linalg.norm(velocity, axis=1).max() > spec.max_plan_speed_mps or
                np.linalg.norm(np.diff(velocity, axis=0) / .5, axis=1).max() > spec.max_plan_acceleration_mps2):
            raise ValueError('plan exceeds configured admissibility bounds')
    if not isinstance(request['acquired_field_manifest'], list):
        raise ValueError('invalid acquired-field manifest')
    try:
        size = len(encode(request))
    except (TypeError, OverflowError) as exc:
        raise ValueError('request must be JSON serializable') from exc
    if size > spec.max_request_bytes:
        raise ValueError('complete request exceeds byte limit')
    return spec


def validate_task_request(request, known_receipts):
    """Only provider-issued *remote* fields can be acknowledged for v2 deduplication.

    known_receipts maps receipt IDs to field refs from actual prior responses.
    It is owned by VehicleTools, never supplied through the request wire.
    """
    _request_spec(request)
    if not isinstance(known_receipts, dict):
        raise ValueError('provider receipt registry is required')
    seen = set()
    for entry in request['acquired_field_manifest']:
        _keys(entry, ('receipt_id', 'ref'), 'manifest entry')
        receipt_id = _text(entry['receipt_id'])
        key = field_key(entry['ref'])
        if key[:3] != (request['provider'], request['scene'], request['g']):
            raise ValueError('manifest source/scene/time mismatch')
        issued = known_receipts.get(receipt_id)
        if issued is None or key not in {field_key(ref) for ref in issued}:
            raise ValueError('field is not proven by a provider-issued receipt')
        if key in seen:
            raise ValueError('duplicate acknowledged field')
        seen.add(key)


def _check_window(window):
    required = {'source', 'scene', 'g', 'track_ids', 'states', 'valid', 'scores', 'time_seconds'}
    if (not isinstance(window, dict) or not required <= set(window) or
            set(window) - required - {'eligible'}):
        raise ValueError('unexpected or missing causal window fields')
    history_packet(window)
    _integer(window['g'])
    _text(window['source'])
    _text(window['scene'])
    ids = np.asarray(window['track_ids'])
    _array(window['states'], (len(ids), 11, 7))
    _array(window['scores'], (len(ids), 11))
    if (ids.dtype.kind not in 'iu' or (ids < 0).any() or
            np.asarray(window['valid']).dtype != np.dtype(bool) or
            (np.asarray(window['time_seconds']) > 0).any()):
        raise ValueError('invalid causal tracking-state identities/mask/time')
    if 'eligible' in window:
        eligible = np.asarray(window['eligible'])
        if (eligible.dtype != np.dtype(bool) or
                not np.array_equal(eligible, np.asarray(window['valid']).sum(axis=1) >= 2)):
            raise ValueError('inconsistent causal history eligibility')


def history_proxy(window):
    """Latest-two-state extrapolation for retrieval only; never an MTR forecast."""
    _check_window(window)
    states = np.asarray(window['states'], dtype=float)
    valid = np.asarray(window['valid'])
    times = np.asarray(window['time_seconds'], dtype=float)
    output = np.zeros((len(states), 1, 6, 2), dtype=float)
    status = []
    for i in range(len(states)):
        indices = np.flatnonzero(valid[i])
        velocity = np.zeros(2)
        tag = 'single_state_static_proxy'
        if len(indices) >= 2:
            a, b = indices[-2:]
            velocity = (states[i, b, :2] - states[i, a, :2]) / (times[b] - times[a])
            tag = 'causal_tracking_state_motion_proxy'
        output[i, 0] = states[i, -1, :2] + np.asarray(TIMES)[:, None] * velocity
        status.append(tag)
    return output, status


def relation_scores(rho_old, rho_new):
    new = np.asarray(rho_new, dtype=float)
    old = np.asarray(rho_old, dtype=float)
    if (new.ndim != 3 or old.shape != new.shape or not all(new.shape[1:]) or
            not np.isfinite(new).all() or not np.isfinite(old).all() or
            (new < 0).any() or (new > 1).any() or (old < 0).any() or (old > 1).any()):
        raise ValueError('expected bounded aligned [target, mode, time] relations')
    return new.max(axis=(1, 2)), np.abs(new - old).max(axis=(1, 2))


def rank_targets(window, request, forecast=None):
    """Pure provider-side scoring; callers authenticate receipts before private access."""
    spec = _request_spec(request)
    _check_window(window)
    if (window.get('source'), window.get('scene'), window.get('g')) != (
            request['provider'], request['scene'], request['g']):
        raise ValueError('task and provider window differ')
    n = len(window['track_ids'])
    states = np.asarray(window['states'], dtype=float)
    if request['tool'] == 'P':
        xy, status = history_proxy(window)
    else:
        if (not isinstance(forecast, dict) or
                not {'track_ids', 'states', 'means', 'scores', 'model_used'} <= set(forecast) or
                not np.array_equal(forecast['track_ids'], window['track_ids']) or
                not np.array_equal(forecast['states'], states[:, -1])):
            raise ValueError('F requires an aligned complete-context prediction')
        means = _array(forecast['means'], (n, 6, 50, 2))
        scores = _array(forecast['scores'], (n, 6))
        used = np.asarray(forecast['model_used'])
        if (used.shape != (n,) or used.dtype != np.dtype(bool) or (scores < 0).any() or
                (scores.sum(axis=1) > 1.0001).any()):
            raise ValueError('invalid predictor mode metadata')
        if (not np.array_equal(used, np.asarray(window['valid']).sum(axis=1) >= 2) or
                not np.all(means[~used] == states[~used, None, -1:, :2])):
            raise ValueError('predictor fallback disagrees with causal history/static anchor')
        xy = means[:, :, FUTURE_INDICES, :]
        status = ['mtr' if value else 'stationary_short_history' for value in used]
    radii = np.hypot(states[:, -1, 3], states[:, -1, 4]) * .5
    radii = radii + np.hypot(spec.ego_length_m, spec.ego_width_m) * .5

    def relation(path):
        delta = xy - np.asarray(path)[None, None]
        distance = np.maximum(0., np.hypot(delta[..., 0], delta[..., 1]) - radii[:, None, None])
        # Ratio before squaring also accommodates small/large configured sigma.
        with np.errstate(over='ignore'):
            return np.exp(-.5 * (distance / spec.sigma_m) ** 2)

    new = relation(request['tau_new'])
    old = relation(request['tau_old']) if request['mode'] == 'change' else new
    current, change = relation_scores(old, new)
    scores = current if request['mode'] == 'current' else change
    rows = [dict(track_handle=int(handle), score=round(float(scores[i]), SCORE_DECIMALS),
                 proxy_status=status[i]) for i, handle in enumerate(window['track_ids'])]
    return sorted(rows, key=lambda row: (-row['score'], row['track_handle']))
