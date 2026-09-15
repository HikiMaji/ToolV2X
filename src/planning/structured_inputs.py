"""Causal numeric evidence preparation with source-local entity association.

This module has no tokenizer, predictor, torch or label access. Input geometry is
already expressed in ego(t); preparation never applies another vehicle transform.
"""
import copy
from dataclasses import asdict, dataclass, fields
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.task_spec import _check_json_native, _keys, field_key


@dataclass(frozen=True)
class StructuredDriverSpec:
    """Complete versioned numeric receiver and architecture contract."""
    version: str = 'toolv2x_structured_driver_v1'
    hidden_dim: int = 256
    attention_heads: int = 4
    interaction_layers: int = 2
    max_entities: int = 64
    max_forecast_sets_per_entity: int = 4
    p_processing: str = 'observations_only'
    position_scale_m: float = 50.
    size_scale_m: float = 10.
    time_scale_s: float = 3.
    association_distance_m: float = 2.
    association_history_distance_m: float = 2.
    association_size_ratio_min: float = .5
    association_size_ratio_max: float = 2.
    association_heading_rad: float = math.pi / 3
    association_ambiguity_margin_m: float = .25
    ego_position_threshold_m: float = 1.5
    ego_history_distance_m: float = .5
    ego_heading_rad: float = math.pi / 6
    ego_min_common_history: int = 2
    ego_length_min_m: float = 3.
    ego_length_max_m: float = 6.
    ego_width_min_m: float = 1.4
    ego_width_max_m: float = 3.
    ego_height_min_m: float = 1.
    ego_height_max_m: float = 3.

    def __post_init__(self):
        if self.version not in ('toolv2x_structured_driver_v1', 'toolv2x_structured_driver_v2'):
            raise ValueError('unsupported structured driver specification')
        if self.p_processing not in ('observations_only', 'local_mtr'):
            raise ValueError('structured P processing must be explicit')
        for name in ('hidden_dim', 'attention_heads', 'interaction_layers', 'max_entities',
                     'max_forecast_sets_per_entity', 'ego_min_common_history'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError('positive integer structured parameter required: ' + name)
        if self.hidden_dim % self.attention_heads:
            raise ValueError('attention heads must divide hidden dimension')
        positive = ('position_scale_m', 'size_scale_m', 'time_scale_s',
                    'association_distance_m', 'association_history_distance_m',
                    'association_size_ratio_min', 'association_size_ratio_max',
                    'association_heading_rad', 'ego_position_threshold_m',
                    'ego_history_distance_m', 'ego_heading_rad', 'ego_length_min_m',
                    'ego_length_max_m', 'ego_width_min_m', 'ego_width_max_m',
                    'ego_height_min_m', 'ego_height_max_m')
        for name in positive:
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError('positive finite structured parameter required: ' + name)
        margin = self.association_ambiguity_margin_m
        if type(margin) not in (int, float) or not np.isfinite(margin) or margin < 0:
            raise ValueError('finite nonnegative ambiguity margin required')
        if (self.association_size_ratio_min > 1 or self.association_size_ratio_max < 1 or
                self.association_size_ratio_min >= self.association_size_ratio_max):
            raise ValueError('association size-ratio gates must straddle one')
        for low, high in (('ego_length_min_m', 'ego_length_max_m'),
                          ('ego_width_min_m', 'ego_width_max_m'),
                          ('ego_height_min_m', 'ego_height_max_m')):
            if getattr(self, low) >= getattr(self, high):
                raise ValueError('ego dimension bounds must increase')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _keys(value, [item.name for item in fields(cls)], 'StructuredDriverSpec')
        return cls(**value)


def _valid_pose(value):
    return (value.shape == (4, 4) and np.isfinite(value).all() and
            np.allclose(value[3], [0., 0., 0., 1.]) and
            np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5) and
            np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-5))


def load_ego_history(root, split, g):
    """Load only t-10..t localization and express it once in ego(t)."""
    from planning.inputs import frame_location
    directory, start = frame_location(root, split, g)
    pose_dir = directory / 'ego'
    poses = {}
    paths = []
    for frame in range(max(start, int(g) - 10), int(g) + 1):
        path = pose_dir / ('%04d_lidar_pose.npy' % frame)
        paths.append(str(path))
        try:
            pose = np.load(path, allow_pickle=False)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if _valid_pose(np.asarray(pose)):
            poses[frame] = np.asarray(pose, dtype=float)
    states = np.zeros((11, 3), dtype=float)
    valid = np.zeros(11, dtype=bool)
    origin = poses.get(int(g))
    if origin is not None:
        inverse = np.linalg.inv(origin)
        for frame, pose in poses.items():
            slot = frame - (int(g) - 10)
            relative = inverse @ pose
            states[slot, :2] = relative[:2, 3]
            states[slot, 2] = math.atan2(relative[1, 0], relative[0, 0])
            valid[slot] = True
    return dict(states=states.tolist(), valid=valid.tolist(),
                times=(np.arange(-10, 1) / 10.).tolist(), read_paths=paths)


def _field_value(kind, value):
    """Use the wire validator while retaining actual local/subset context labels."""
    from tools.vehicle import _validate_task_value
    check = copy.deepcopy(value)
    if kind == 'forecast':
        if check.get('context_scope') not in ('ego_full_at_t', 'provider_full_at_t',
                                               'receiver_acquired_subset'):
            raise ValueError('unknown forecast context scope')
        check['context_scope'] = 'provider_full_at_t'
    _validate_task_value(kind, check)


def _ref_sort(ref):
    return field_key(ref)


def _alias_sort(alias, local_source):
    return (0 if alias[0] == local_source else 1, alias[0], alias[1])


def _alias_value(alias):
    return dict(source=alias[0], track_handle=alias[1])


def _entity_id(aliases, local_source):
    canonical = sorted(aliases, key=lambda value: _alias_sort(value, local_source))[0]
    return '%s:%d' % canonical


def _history_from_record(record):
    value = record['value']
    return dict(states=copy.deepcopy(value['history']), valid=copy.deepcopy(value['history_valid']),
                scores=copy.deepcopy(value['history_scores']), times=copy.deepcopy(value['history_times']),
                proxy_status=value['proxy_status'])


def _tracks(ledger, units):
    local = {}
    for record in ledger['local_fields']:
        ref, value = record['ref'], record['value']
        if (record.get('origin') != 'local' or ref['provider'] != ledger['local_source'] or
                (ref['scene'], ref['g']) != (ledger['scene'], ledger['g'])):
            raise ValueError('invalid local ledger field')
        key = field_key(ref)
        _field_value(ref['field_kind'], value)
        alias = (ref['provider'], ref['track_handle'])
        track = local.setdefault(alias, dict(alias=alias, source_role='local', records={}, forecasts=[]))
        if ref['field_kind'] in track['records']:
            raise ValueError('duplicate local field identity')
        track['records'][ref['field_kind']] = record
        if ref['field_kind'] == 'forecast':
            track['forecasts'].append(dict(value=copy.deepcopy(value), primary_refs=[copy.deepcopy(ref)],
                anchor_ref=None, parent_refs=[]))
    remote = {}
    records = {field_key(record['ref']): record
               for record in ledger['acquired_fields'] + ledger['derived_fields']}
    for unit in units:
        obj = unit['object']
        alias = (obj['source'], int(obj['track_id']))
        if alias[0] == ledger['local_source']:
            raise ValueError('remote unit reused local source identity')
        track = remote.setdefault(alias, dict(alias=alias, source_role='remote', records={}, forecasts=[]))
        anchor = dict(ref=copy.deepcopy(unit['anchor_ref']), value=dict(box=copy.deepcopy(obj['box']),
                                                                       score=obj['score']))
        previous = track['records'].get('anchor')
        if previous is not None and previous != anchor:
            raise ValueError('remote units disagree on a shared anchor')
        track['records']['anchor'] = anchor
        kind = unit['kind']
        metadata = obj['field_metadata'][kind]
        if kind == 'history':
            value = dict(history=copy.deepcopy(obj['history']),
                history_valid=copy.deepcopy(obj['history_valid']),
                history_scores=copy.deepcopy(obj['history_scores']),
                history_times=copy.deepcopy(obj['history_times']), proxy_status=obj['proxy_status'])
            existing = track['records'].get('history')
            entry = dict(ref=copy.deepcopy(unit['primary_refs'][0]), value=value,
                         primary_refs=copy.deepcopy(unit['primary_refs']))
            if existing is not None and existing != entry:
                raise ValueError('remote units disagree on history')
            track['records']['history'] = entry
        else:
            value = dict(forecast=copy.deepcopy(obj['forecast']),
                forecast_scores=copy.deepcopy(obj['forecast_scores']),
                forecast_times=copy.deepcopy(obj['forecast_times']), model_used=obj['model_used'],
                context_scope=metadata['context_scope'])
            parents = []
            for ref in unit['primary_refs']:
                record = records[field_key(ref)]
                parents.extend(copy.deepcopy(record.get('parent_refs', [])))
            unique = {field_key(ref): ref for ref in parents}
            track['forecasts'].append(dict(value=value, primary_refs=copy.deepcopy(unit['primary_refs']),
                anchor_ref=copy.deepcopy(unit['anchor_ref']),
                parent_refs=[copy.deepcopy(unique[k]) for k in sorted(unique)]))
    for track in list(local.values()) + list(remote.values()):
        if 'anchor' not in track['records']:
            raise ValueError('every source-local track requires a current anchor')
        anchor = track['records']['anchor']['value']
        if 'history' in track['records']:
            history = track['records']['history']['value']
            if (history['history'][-1] != anchor['box'] or
                    history['history_scores'][-1] != anchor['score']):
                raise ValueError('history and current anchor disagree')
    return ([local[k] for k in sorted(local, key=lambda a: _alias_sort(a, ledger['local_source']))],
            [remote[k] for k in sorted(remote, key=lambda a: _alias_sort(a, ledger['local_source']))])


def _box_axis_distance(a, b):
    """Axis-equivalent box yaw distance; front/back is not used for association."""
    return abs((float(a) - float(b) + math.pi / 2) % math.pi - math.pi / 2)


def _directed_angle_distance(a, b):
    """Directed heading distance for ego-motion corroboration."""
    return abs((float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi)


def _pair_metrics(a, b):
    aa = np.asarray(a['records']['anchor']['value']['box'], dtype=float)
    bb = np.asarray(b['records']['anchor']['value']['box'], dtype=float)
    current = float(np.linalg.norm(aa[:2] - bb[:2]))
    ratio = aa[3:6] / bb[3:6]
    heading = _box_axis_distance(aa[6], bb[6])
    common = 0
    history_distance = 0.
    if 'history' in a['records'] and 'history' in b['records']:
        av = np.asarray(a['records']['history']['value']['history_valid'], dtype=bool)
        bv = np.asarray(b['records']['history']['value']['history_valid'], dtype=bool)
        mask = av & bv
        common = int(mask.sum())
        if common:
            ah = np.asarray(a['records']['history']['value']['history'], dtype=float)[mask, :2]
            bh = np.asarray(b['records']['history']['value']['history'], dtype=float)[mask, :2]
            history_distance = float(np.sqrt(np.mean(np.sum((ah - bh) ** 2, axis=1))))
    return dict(current_distance_m=current, history_distance_m=history_distance,
                common_history_steps=common, size_ratio=ratio.tolist(), heading_difference_rad=heading,
                cost_m=current + (history_distance if common else 0.))


def _associate(local, remote, spec):
    """Gated Hungarian matching; near-tied row or column candidates stay separate."""
    if not local or not remote:
        return [], set(), {}
    metrics = {}
    allowed = np.zeros((len(local), len(remote)), dtype=bool)
    costs = np.full(allowed.shape, np.inf)
    for i, a in enumerate(local):
        for j, b in enumerate(remote):
            value = _pair_metrics(a, b)
            ratio = np.asarray(value['size_ratio'])
            ok = (value['current_distance_m'] <= spec.association_distance_m and
                  (ratio >= spec.association_size_ratio_min).all() and
                  (ratio <= spec.association_size_ratio_max).all() and
                  value['heading_difference_rad'] <= spec.association_heading_rad and
                  (not value['common_history_steps'] or
                   value['history_distance_m'] <= spec.association_history_distance_m))
            metrics[(i, j)] = dict(value, allowed=bool(ok))
            allowed[i, j] = ok
            if ok:
                costs[i, j] = value['cost_m']
    ambiguous_local, ambiguous_remote = set(), set()
    for i in range(len(local)):
        values = sorted(costs[i, np.isfinite(costs[i])].tolist())
        if len(values) > 1 and values[1] - values[0] <= spec.association_ambiguity_margin_m:
            ambiguous_local.add(i)
            ambiguous_remote.update(np.flatnonzero(np.isfinite(costs[i])).tolist())
    for j in range(len(remote)):
        values = sorted(costs[np.isfinite(costs[:, j]), j].tolist())
        if len(values) > 1 and values[1] - values[0] <= spec.association_ambiguity_margin_m:
            ambiguous_remote.add(j)
            ambiguous_local.update(np.flatnonzero(np.isfinite(costs[:, j])).tolist())
    usable = allowed.copy()
    for i in ambiguous_local:
        usable[i, :] = False
    for j in ambiguous_remote:
        usable[:, j] = False
    pairs = []
    if usable.any():
        rows, columns = linear_sum_assignment(np.where(usable, costs, 1e12))
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns) if usable[i, j]]
    ambiguous = {('local', i) for i in ambiguous_local} | {('remote', j) for j in ambiguous_remote}
    return pairs, ambiguous, metrics


def _observation(track):
    anchor_record = track['records']['anchor']
    if 'history' in track['records']:
        history_record = track['records']['history']
        value = _history_from_record(history_record)
        history_ref = copy.deepcopy(history_record['ref'])
        primary_refs = copy.deepcopy(history_record.get('primary_refs', [history_ref]))
    else:
        box = copy.deepcopy(anchor_record['value']['box'])
        value = dict(states=[[0.] * 7 for _ in range(10)] + [box],
                     valid=[False] * 10 + [True], scores=[0.] * 10 + [anchor_record['value']['score']],
                     times=(np.arange(-10, 1) / 10.).tolist(), proxy_status='current_anchor_only')
        history_ref = None
        primary_refs = []
    return dict(source=track['alias'][0], source_role=track['source_role'],
                anchor_ref=copy.deepcopy(anchor_record['ref']), history_ref=history_ref,
                primary_refs=primary_refs, value=value)


def _forecast(track, item):
    return dict(source=track['alias'][0], source_role=track['source_role'],
        context_scope=item['value']['context_scope'], model_used=item['value']['model_used'],
        forecast=copy.deepcopy(item['value']['forecast']),
        forecast_scores=copy.deepcopy(item['value']['forecast_scores']),
        forecast_times=copy.deepcopy(item['value']['forecast_times']),
        primary_refs=copy.deepcopy(item['primary_refs']), anchor_ref=copy.deepcopy(
            item['anchor_ref'] or track['records']['anchor']['ref']),
        parent_refs=copy.deepcopy(item['parent_refs']))


def _representative(tracks):
    return sorted(tracks, key=lambda t: (0 if t['source_role'] == 'local' else 1,
        -float(t['records']['anchor']['value']['score']), t['alias']))[0]


def _representative_value(track):
    return dict(source=track['alias'][0], track_handle=track['alias'][1],
        box=copy.deepcopy(track['records']['anchor']['value']['box']),
        score=track['records']['anchor']['value']['score'],
        selection='local_then_score_then_canonical_alias')


def _role(tracks, ego_history, spec):
    representative = _representative(tracks)
    box = np.asarray(representative['records']['anchor']['value']['box'], dtype=float)
    if np.linalg.norm(box[:2]) > spec.ego_position_threshold_m:
        return 'obstacle'
    if ego_history is None:
        return 'unresolved'
    if not (spec.ego_length_min_m <= box[3] <= spec.ego_length_max_m and
            spec.ego_width_min_m <= box[4] <= spec.ego_width_max_m and
            spec.ego_height_min_m <= box[5] <= spec.ego_height_max_m):
        return 'unresolved'
    if 'history' not in representative['records']:
        return 'unresolved'
    observed = representative['records']['history']['value']
    valid = np.asarray(observed['history_valid'], dtype=bool) & np.asarray(ego_history['valid'], dtype=bool)
    if valid.sum() < spec.ego_min_common_history:
        return 'unresolved'
    states = np.asarray(observed['history'], dtype=float)[valid]
    ego = np.asarray(ego_history['states'], dtype=float)[valid]
    distance = float(np.sqrt(np.mean(np.sum((states[:, :2] - ego[:, :2]) ** 2, axis=1))))
    heading = max((_directed_angle_distance(a, b) for a, b in zip(states[:, 6], ego[:, 2])),
                  default=math.inf)
    return 'ego' if distance <= spec.ego_history_distance_m and heading <= spec.ego_heading_rad else 'unresolved'


def _validate_ego_history(value):
    if value is None:
        return None
    _keys(value, ('states', 'valid', 'times', 'read_paths'), 'ego history')
    states = _numeric_array(value['states'], (11, 3))
    valid = _bool_array(value['valid'], (11,))
    times = _numeric_array(value['times'], (11,))
    if not np.allclose(times, np.arange(-10, 1) / 10.) or any(type(p) is not str for p in value['read_paths']):
        raise ValueError('invalid causal ego history time/provenance')
    if valid.any() and (not valid[-1] or not np.allclose(states[-1], 0.)):
        raise ValueError('ego history must be expressed in ego(t)')
    return copy.deepcopy(value)


def _numeric_array(value, shape):
    try:
        raw = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
            raise ValueError('booleans are not numeric tensor values')
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid numeric tensor') from exc
    if result.shape != shape or result.dtype.kind not in 'fiu' or not np.isfinite(result).all():
        raise ValueError('invalid numeric tensor shape or value')
    return result.astype(float)


def _bool_array(value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.dtype(bool):
        raise ValueError('invalid boolean tensor shape or value')
    return result


def _known_records(ledger):
    result = {}
    for record in ledger['local_fields'] + ledger['acquired_fields'] + ledger['derived_fields']:
        key = field_key(record['ref'])
        if key in result:
            raise ValueError('duplicate known evidence reference')
        result[key] = record
    return result


def _parents_closed(keys, records):
    pending = list(keys)
    closed = set(keys)
    while pending:
        record = records[pending.pop()]
        for ref in record.get('parent_refs', []):
            key = field_key(ref)
            if key not in records:
                raise ValueError('unknown evidence parent')
            if key not in closed:
                closed.add(key)
                pending.append(key)
    return closed


def _make_entity(tracks, status, metrics, ledger, ego_history, spec):
    tracks = sorted(tracks, key=lambda t: _alias_sort(t['alias'], ledger['local_source']))
    aliases = [track['alias'] for track in tracks]
    representative = _representative(tracks)
    observations = [_observation(track) for track in tracks]
    forecasts = [_forecast(track, item) for track in tracks for item in track['forecasts']]
    parents = {field_key(ref): ref for forecast in forecasts for ref in forecast['parent_refs']}
    association = dict(status=status, rule='gated_hungarian_v1_ambiguous_separate',
                       metrics=copy.deepcopy(metrics))
    return dict(entity_id=_entity_id(aliases, ledger['local_source']),
        aliases=[_alias_value(alias) for alias in aliases], role=_role(tracks, ego_history, spec),
        association=association,
        representative_anchor=_representative_value(representative),
        observations=observations, forecast_sets=forecasts,
        parent_refs=[copy.deepcopy(parents[k]) for k in sorted(parents)], tensor_index=None)


def _entities(local, remote, ledger, ego_history, spec):
    pairs, ambiguous, metrics = _associate(local, remote, spec)
    paired_local = {i for i, _ in pairs}
    paired_remote = {j for _, j in pairs}
    result = []
    for i, j in pairs:
        result.append(_make_entity([local[i], remote[j]], 'matched', metrics[(i, j)],
                                   ledger, ego_history, spec))
    for role, tracks, used in (('local', local, paired_local), ('remote', remote, paired_remote)):
        for index, track in enumerate(tracks):
            if index in used:
                continue
            candidate_metrics = [dict(peer=_alias_value((remote[j] if role == 'local' else local[j])['alias']),
                                      **metrics[(index, j) if role == 'local' else (j, index)])
                                 for j in range(len(remote if role == 'local' else local))
                                 if metrics[(index, j) if role == 'local' else (j, index)]['allowed']]
            status = 'ambiguous' if (role, index) in ambiguous else 'unmatched'
            result.append(_make_entity([track], status,
                dict(candidates=candidate_metrics), ledger, ego_history, spec))
    return sorted(result, key=lambda entity: (
        float(np.hypot(*entity['representative_anchor']['box'][:2])), entity['entity_id']))


def _forecast_order(value):
    full = value['context_scope'] in ('ego_full_at_t', 'provider_full_at_t')
    refs = tuple(_ref_sort(ref) for ref in value['primary_refs'])
    return (0 if full else 1, -len(value['parent_refs']),
            0 if value['source_role'] == 'local' else 1, value['source'], refs)


def _tensors(entities, spec):
    e, c = spec.max_entities, spec.max_forecast_sets_per_entity
    observations = np.zeros((e, 2, 11, 10), dtype=float)
    observation_mask = np.zeros((e, 2, 11), dtype=bool)
    observation_sources = np.broadcast_to(np.array([0, 1]), (e, 2)).copy()
    forecasts = np.zeros((e, c, 6, 6, 5), dtype=float)
    forecast_mask = np.zeros((e, c, 6, 6), dtype=bool)
    forecast_context = np.zeros((e, c, 2), dtype=float)
    entity_mask = np.zeros(e, dtype=bool)
    selected = [entity for entity in entities if entity['role'] != 'ego'][:e]
    for row, entity in enumerate(selected):
        entity['tensor_index'] = row
        entity_mask[row] = True
        for observation in entity['observations']:
            slot = 0 if observation['source_role'] == 'local' else 1
            value = observation['value']
            states = np.asarray(value['states'], dtype=float)
            valid = np.asarray(value['valid'], dtype=bool)
            scores = np.asarray(value['scores'], dtype=float)
            times = np.asarray(value['times'], dtype=float)
            # v1 archives keep their original NumPy encoding. v2 uses one scalar
            # path so allocation-dependent ufunc rounding cannot break exact replay.
            if spec.version == 'toolv2x_structured_driver_v2':
                sine = [math.sin(float(yaw)) for yaw in states[:, 6]]
                cosine = [math.cos(float(yaw)) for yaw in states[:, 6]]
            else:
                sine, cosine = np.sin(states[:, 6]), np.cos(states[:, 6])
            encoded = np.stack([states[:, 0] / spec.position_scale_m,
                states[:, 1] / spec.position_scale_m, states[:, 2] / spec.position_scale_m,
                states[:, 3] / spec.size_scale_m, states[:, 4] / spec.size_scale_m,
                states[:, 5] / spec.size_scale_m, sine, cosine,
                scores, times / spec.time_scale_s], axis=-1)
            observations[row, slot, valid] = encoded[valid]
            observation_mask[row, slot] = valid
        ordered = sorted(entity['forecast_sets'], key=_forecast_order)
        for column, value in enumerate(ordered[:c]):
            paths = np.asarray(value['forecast'], dtype=float)
            scores = np.asarray(value['forecast_scores'], dtype=float)
            times = np.asarray(value['forecast_times'], dtype=float)
            forecasts[row, column, :, :, 0] = paths[:, :, 0] / spec.position_scale_m
            forecasts[row, column, :, :, 1] = paths[:, :, 1] / spec.position_scale_m
            forecasts[row, column, :, :, 2] = times[None, :] / spec.time_scale_s
            forecasts[row, column, :, :, 3] = scores[:, None]
            forecasts[row, column, :, :, 4] = float(value['model_used'])
            forecast_mask[row, column] = True
            forecast_context[row, column] = [float(value['source_role'] == 'local'),
                float(value['context_scope'] in ('ego_full_at_t', 'provider_full_at_t'))]
    return dict(observations=observations.tolist(), observation_mask=observation_mask.tolist(),
        observation_sources=observation_sources.tolist(), forecasts=forecasts.tolist(),
        forecast_mask=forecast_mask.tolist(), forecast_context=forecast_context.tolist(),
        entity_mask=entity_mask.tolist()), selected


def _admission_groups(entities, spec):
    field_groups, local_groups = [], []
    for entity in entities:
        admitted_entity = entity['tensor_index'] is not None
        use = 'ego_filter' if entity['role'] == 'ego' else (
            'tensor' if admitted_entity else 'entity_capacity')
        for observation in entity['observations']:
            group = dict(entity_id=entity['entity_id'], tensor_index=entity['tensor_index'],
                kind='observation', use=use, source_role=observation['source_role'],
                primary_refs=copy.deepcopy(observation['primary_refs']),
                anchor_ref=copy.deepcopy(observation['anchor_ref']),
                tensor_locations=[] if not admitted_entity else [dict(
                    array='observations', entity=entity['tensor_index'],
                    source_slot=0 if observation['source_role'] == 'local' else 1)])
            (local_groups if observation['source_role'] == 'local' else field_groups).append(
                dict(ref=copy.deepcopy(observation['anchor_ref']), **group)
                if observation['source_role'] == 'local' else group)
        for column, forecast in enumerate(sorted(entity['forecast_sets'], key=_forecast_order)):
            admitted_set = admitted_entity and column < spec.max_forecast_sets_per_entity
            group = dict(entity_id=entity['entity_id'], tensor_index=entity['tensor_index'],
                kind='forecast', use='tensor' if admitted_set else (
                    'ego_filter' if entity['role'] == 'ego' else 'structured_capacity'),
                source_role=forecast['source_role'], primary_refs=copy.deepcopy(forecast['primary_refs']),
                anchor_ref=copy.deepcopy(forecast['anchor_ref']), tensor_locations=[dict(
                    array='forecasts', entity=entity['tensor_index'], forecast_set=column)]
                    if admitted_set else [])
            if forecast['source_role'] == 'local':
                local_groups.append(dict(ref=copy.deepcopy(forecast['primary_refs'][0]), **group))
            else:
                field_groups.append(group)
    return field_groups, local_groups


def build_structured_plan_input(motion, ledger, spec, *, ego_history=None,
                                previous_plan=None, previous_parent_refs=()):
    """Build a JSON-native numeric record from a verified cumulative ledger."""
    from planning.evidence import remote_units
    if not isinstance(spec, StructuredDriverSpec):
        raise ValueError('StructuredDriverSpec instance required')
    if (ledger.get('version') not in ('toolv2x_evidence_ledger_v1', 'toolv2x_evidence_ledger_v2') or
            ledger.get('coordinate_frame') != 'ego_at_t' or ledger.get('status') != 'ready' or
            ledger.get('p_processing') != spec.p_processing):
        raise ValueError('ledger and structured receiver contract mismatch')
    _validate_motion(motion)
    ego_history = _validate_ego_history(ego_history)
    if previous_plan is not None:
        _numeric_array(previous_plan, (6, 2))
        previous_plan = copy.deepcopy(previous_plan)
    if not isinstance(previous_parent_refs, (list, tuple)):
        raise ValueError('previous parent refs must be an ordered collection')
    previous_parent_refs = [copy.deepcopy(ref) for ref in previous_parent_refs]
    if previous_plan is None and previous_parent_refs:
        raise ValueError('absent previous plan cannot carry parent refs')
    units = remote_units(ledger)  # Receipt, derived-parent and equivalence validation.
    records = _known_records(ledger)
    previous_keys = []
    for ref in previous_parent_refs:
        key = field_key(ref)
        if key not in records:
            raise ValueError('previous plan names unknown evidence')
        previous_keys.append(key)
    local, remote = _tracks(ledger, units)
    entities = _entities(local, remote, ledger, ego_history, spec)
    tensors, selected = _tensors(entities, spec)

    admitted = set(previous_keys)
    for entity in entities:
        # Association, capacity ordering and ego filtering consume every anchor/history.
        for observation in entity['observations']:
            admitted.add(field_key(observation['anchor_ref']))
            admitted.update(field_key(ref) for ref in observation['primary_refs'])
        if entity in selected:
            for forecast in sorted(entity['forecast_sets'], key=_forecast_order)[
                    :spec.max_forecast_sets_per_entity]:
                admitted.update(field_key(ref) for ref in forecast['primary_refs'])
    admitted = _parents_closed(admitted, records)
    known_keys = set(records)
    dropped_keys = known_keys - admitted
    acquired_refs = [copy.deepcopy(r['ref']) for r in ledger['acquired_fields']]
    derived_refs = [copy.deepcopy(r['ref']) for r in ledger['derived_fields']]

    field_groups, local_groups = _admission_groups(entities, spec)
    dropped = [dict(ref=copy.deepcopy(records[key]['ref']), reason='structured_capacity')
               for key in sorted(dropped_keys)]
    report = dict(acquired_field_refs=acquired_refs, derived_field_refs=derived_refs,
        admitted_field_refs=[copy.deepcopy(records[key]['ref']) for key in sorted(admitted)],
        dropped_field_refs=[copy.deepcopy(records[key]['ref']) for key in sorted(dropped_keys)],
        dropped=dropped, field_groups=field_groups, local_field_groups=local_groups,
        observation_receipt_ids=[receipt['receipt_id'] for receipt in ledger['receipts']])
    prepared = dict(input_layout='structured_evidence_v1', decoding='numeric',
        driver_spec=spec.to_dict(), scene=ledger['scene'], g=ledger['g'],
        ego_motion=copy.deepcopy(motion), entities=entities, tensor_inputs=tensors,
        admission_report=report, ego_history_used=copy.deepcopy(ego_history),
        previous_plan=previous_plan, previous_parent_refs=previous_parent_refs)
    validate_structured_prepared(prepared)
    return prepared


def _validate_motion(value):
    _keys(value, ('speed_mps', 'yaw_rate_rps'), 'ego motion')
    for item in value.values():
        if item is not None and (type(item) not in (int, float) or not np.isfinite(item)):
            raise ValueError('ego motion must be finite or explicitly missing')


def validate_structured_prepared(prepared):
    """Strictly validate JSON-native shapes, masks, capacities and dependencies."""
    _check_json_native(prepared)
    _keys(prepared, ('input_layout', 'decoding', 'driver_spec', 'scene', 'g', 'ego_motion',
        'entities', 'tensor_inputs', 'admission_report', 'ego_history_used', 'previous_plan',
        'previous_parent_refs'), 'structured prepared input')
    if (prepared['input_layout'] != 'structured_evidence_v1' or prepared['decoding'] != 'numeric' or
            type(prepared['scene']) is not str or not prepared['scene'] or
            type(prepared['g']) is not int or prepared['g'] < 0):
        raise ValueError('invalid structured prepared identity')
    spec = StructuredDriverSpec.from_dict(prepared['driver_spec'])
    _validate_motion(prepared['ego_motion'])
    _validate_ego_history(prepared['ego_history_used'])
    if prepared['previous_plan'] is not None:
        _numeric_array(prepared['previous_plan'], (6, 2))
    if type(prepared['previous_parent_refs']) is not list:
        raise ValueError('invalid previous parent refs')
    previous_keys = [field_key(ref) for ref in prepared['previous_parent_refs']]
    if (len(previous_keys) != len(set(previous_keys)) or
            prepared['previous_plan'] is None and previous_keys):
        raise ValueError('duplicate previous parent ref')

    tensors = prepared['tensor_inputs']
    _keys(tensors, ('observations', 'observation_mask', 'observation_sources', 'forecasts',
                    'forecast_mask', 'forecast_context', 'entity_mask'), 'structured tensors')
    e, c = spec.max_entities, spec.max_forecast_sets_per_entity
    observations = _numeric_array(tensors['observations'], (e, 2, 11, 10))
    observation_mask = _bool_array(tensors['observation_mask'], (e, 2, 11))
    sources = _numeric_array(tensors['observation_sources'], (e, 2))
    forecasts = _numeric_array(tensors['forecasts'], (e, c, 6, 6, 5))
    forecast_mask = _bool_array(tensors['forecast_mask'], (e, c, 6, 6))
    contexts = _numeric_array(tensors['forecast_context'], (e, c, 2))
    entity_mask = _bool_array(tensors['entity_mask'], (e,))
    if not np.array_equal(sources, np.broadcast_to([0, 1], (e, 2))):
        raise ValueError('observation source slots changed')
    if np.any(observations[~observation_mask] != 0):
        raise ValueError('masked observations must be zero')
    expanded = np.repeat(forecast_mask[..., None], 5, axis=-1)
    if np.any(forecasts[~expanded] != 0):
        raise ValueError('masked forecasts must be zero')
    set_mask = forecast_mask.any(axis=(2, 3))
    if np.any(contexts[~set_mask] != 0) or not np.all(
            forecast_mask == set_mask[:, :, None, None]):
        raise ValueError('forecast set masks/context disagree')
    count = int(entity_mask.sum())
    if not np.array_equal(entity_mask, np.arange(e) < count):
        raise ValueError('entity padding must follow admitted entities')

    if type(prepared['entities']) is not list:
        raise ValueError('entities must be a list')
    seen_ids, tensor_indices, entity_refs, derived_parents = set(), [], set(), {}
    semantic_tracks = {'local': [], 'remote': []}
    semantic_aliases = set()
    for entity in prepared['entities']:
        _keys(entity, ('entity_id', 'aliases', 'role', 'association', 'representative_anchor',
            'observations', 'forecast_sets', 'parent_refs', 'tensor_index'), 'entity')
        if (type(entity['entity_id']) is not str or entity['entity_id'] in seen_ids or
                entity['role'] not in ('ego', 'obstacle', 'unresolved')):
            raise ValueError('invalid entity identity/role')
        seen_ids.add(entity['entity_id'])
        aliases = entity['aliases']
        if type(aliases) is not list or not aliases:
            raise ValueError('entity aliases required')
        alias_keys = []
        for alias in aliases:
            _keys(alias, ('source', 'track_handle'), 'entity alias')
            if type(alias['source']) is not str or type(alias['track_handle']) is not int:
                raise ValueError('invalid source-local alias')
            alias_keys.append((alias['source'], alias['track_handle']))
        if (len(alias_keys) != len(set(alias_keys)) or len(alias_keys) > 2 or
                entity['entity_id'] != '%s:%d' % alias_keys[0]):
            raise ValueError('aliases must be unique and begin with the canonical alias')
        index = entity['tensor_index']
        if index is not None:
            if type(index) is not int or not 0 <= index < count or entity['role'] == 'ego':
                raise ValueError('invalid entity tensor index')
            tensor_indices.append(index)
        for ref in entity['parent_refs']:
            entity_refs.add(field_key(ref))
        _keys(entity['association'], ('status', 'rule', 'metrics'), 'entity association')
        if (entity['association']['status'] not in ('matched', 'ambiguous', 'unmatched') or
                entity['association']['rule'] != 'gated_hungarian_v1_ambiguous_separate'):
            raise ValueError('invalid association result')
        if ((entity['association']['status'] == 'matched') != (len(alias_keys) == 2)):
            raise ValueError('association status and alias cardinality disagree')
        representative = entity['representative_anchor']
        _keys(representative, ('source', 'track_handle', 'box', 'score', 'selection'),
              'representative anchor')
        if ((representative['source'], representative['track_handle']) not in alias_keys or
                representative['selection'] != 'local_then_score_then_canonical_alias'):
            raise ValueError('representative anchor is not a source observation')
        _field_value('anchor', dict(box=representative['box'], score=representative['score']))
        roles = []
        for observation in entity['observations']:
            _keys(observation, ('source', 'source_role', 'anchor_ref', 'history_ref',
                                'primary_refs', 'value'), 'entity observation')
            if (observation['source_role'] not in ('local', 'remote') or
                    (observation['source'], observation['anchor_ref']['track_handle']) not in alias_keys or
                    observation['anchor_ref']['provider'] != observation['source'] or
                    observation['anchor_ref']['field_kind'] != 'anchor'):
                raise ValueError('observation source and alias disagree')
            roles.append(observation['source_role'])
            alias = (observation['source'], observation['anchor_ref']['track_handle'])
            if alias in semantic_aliases:
                raise ValueError('source-local observation appears in multiple entities')
            semantic_aliases.add(alias)
            entity_refs.add(field_key(observation['anchor_ref']))
            primary = [field_key(ref) for ref in observation['primary_refs']]
            entity_refs.update(primary)
            value = observation['value']
            _keys(value, ('states', 'valid', 'scores', 'times', 'proxy_status'),
                  'entity observation value')
            states = _numeric_array(value['states'], (11, 7))
            valid = _bool_array(value['valid'], (11,))
            scores = _numeric_array(value['scores'], (11,))
            times = _numeric_array(value['times'], (11,))
            if (not valid[-1] or not np.allclose(times, np.arange(-10, 1) / 10.) or
                    (states[valid, 3:6] <= 0).any() or (scores < 0).any() or (scores > 1).any()):
                raise ValueError('invalid causal entity observation')
            if observation['history_ref'] is None:
                if primary or value['proxy_status'] != 'current_anchor_only' or valid.sum() != 1:
                    raise ValueError('anchor-only observation has hidden history')
            else:
                history_key = field_key(observation['history_ref'])
                if observation['history_ref']['field_kind'] != 'history' or history_key not in primary:
                    raise ValueError('observation history reference mismatch')
                check = dict(history=value['states'], history_valid=value['valid'],
                             history_scores=value['scores'], history_times=value['times'],
                             proxy_status=value['proxy_status'])
                _field_value('history', check)
            track = dict(alias=alias, source_role=observation['source_role'], records={
                'anchor': dict(ref=copy.deepcopy(observation['anchor_ref']), value=dict(
                    box=copy.deepcopy(value['states'][-1]), score=value['scores'][-1]))}, forecasts=[])
            if observation['history_ref'] is not None:
                track['records']['history'] = dict(ref=copy.deepcopy(observation['history_ref']),
                    value=dict(history=copy.deepcopy(value['states']),
                        history_valid=copy.deepcopy(value['valid']),
                        history_scores=copy.deepcopy(value['scores']),
                        history_times=copy.deepcopy(value['times']),
                        proxy_status=value['proxy_status']))
            semantic_tracks[observation['source_role']].append(track)
        if len(roles) != len(set(roles)) or not roles:
            raise ValueError('entity source observation slots must be unique')
        for forecast in entity['forecast_sets']:
            _keys(forecast, ('source', 'source_role', 'context_scope', 'model_used', 'forecast',
                'forecast_scores', 'forecast_times', 'primary_refs', 'anchor_ref', 'parent_refs'),
                'entity forecast set')
            if (forecast['source_role'] not in ('local', 'remote') or
                    (forecast['source'], forecast['anchor_ref']['track_handle']) not in alias_keys or
                    forecast['anchor_ref']['provider'] != forecast['source'] or
                    forecast['anchor_ref']['field_kind'] != 'anchor'):
                raise ValueError('forecast source and alias disagree')
            entity_refs.add(field_key(forecast['anchor_ref']))
            primary = [field_key(ref) for ref in forecast['primary_refs']]
            parents = [field_key(ref) for ref in forecast['parent_refs']]
            if not primary or any(key[-3] != 'forecast' for key in primary):
                raise ValueError('forecast set requires forecast primary refs')
            entity_refs.update(primary)
            entity_refs.update(parents)
            _field_value('forecast', {key: copy.deepcopy(forecast[key]) for key in (
                'forecast', 'forecast_scores', 'forecast_times', 'model_used', 'context_scope')})
            for key in primary:
                derived_parents.setdefault(key, set()).update(parents)
    if sorted(tensor_indices) != list(range(count)):
        raise ValueError('entity metadata and mask capacity disagree')
    local_sources = {track['alias'][0] for track in semantic_tracks['local']}
    if len(local_sources) > 1:
        raise ValueError('structured entities cannot mix local sources')
    local_source = next(iter(local_sources)) if local_sources else None
    expected_entities = _entities(semantic_tracks['local'], semantic_tracks['remote'],
        dict(local_source=local_source), prepared['ego_history_used'], spec)
    identity = lambda entity: tuple((alias['source'], alias['track_handle'])
                                    for alias in entity['aliases'])
    expected_by_alias = {identity(entity): entity for entity in expected_entities}
    if set(expected_by_alias) != {identity(entity) for entity in prepared['entities']}:
        raise ValueError('entity aliases do not match recomputed association')
    for entity in prepared['entities']:
        expected = expected_by_alias[identity(entity)]
        if (entity['aliases'] != expected['aliases'] or entity['role'] != expected['role'] or
                entity['association'] != expected['association'] or
                entity['representative_anchor'] != expected['representative_anchor']):
            raise ValueError('entity association, role or representative anchor changed')
    canonical = copy.deepcopy(prepared['entities'])
    for entity in canonical:
        entity['tensor_index'] = None
    try:
        expected_tensors, _ = _tensors(canonical, spec)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError('entity metadata cannot reproduce structured tensors') from exc
    expected_indices = {entity['entity_id']: entity['tensor_index'] for entity in canonical}
    if expected_tensors != tensors or any(
            entity['tensor_index'] != expected_indices[entity['entity_id']]
            for entity in prepared['entities']):
        raise ValueError('entity metadata and structured tensors disagree')

    report = prepared['admission_report']
    _keys(report, ('acquired_field_refs', 'derived_field_refs', 'admitted_field_refs',
        'dropped_field_refs', 'dropped', 'field_groups', 'local_field_groups',
        'observation_receipt_ids'), 'admission report')
    acquired = [field_key(ref) for ref in report['acquired_field_refs']]
    derived = [field_key(ref) for ref in report['derived_field_refs']]
    admitted = [field_key(ref) for ref in report['admitted_field_refs']]
    dropped = [field_key(ref) for ref in report['dropped_field_refs']]
    expected_field_groups, expected_local_groups = _admission_groups(prepared['entities'], spec)
    if (report['field_groups'] != expected_field_groups or
            report['local_field_groups'] != expected_local_groups):
        raise ValueError('admission groups or tensor locations changed')
    if (len(acquired) != len(set(acquired)) or len(derived) != len(set(derived)) or
            len(admitted) != len(set(admitted)) or len(dropped) != len(set(dropped)) or
            set(admitted) & set(dropped) or not set(previous_keys) <= set(admitted)):
        raise ValueError('admission references overlap, duplicate or omit prior parents')
    known = set(acquired) | set(derived)
    for group in report['field_groups'] + report['local_field_groups']:
        local_group = 'ref' in group
        _keys(group, ('entity_id', 'tensor_index', 'kind', 'use', 'source_role',
            'primary_refs', 'anchor_ref', 'tensor_locations') + (('ref',) if local_group else ()),
            'field group')
        if (group['entity_id'] not in seen_ids or group['kind'] not in ('observation', 'forecast') or
                group['use'] not in ('tensor', 'ego_filter', 'entity_capacity', 'structured_capacity') or
                group['source_role'] not in ('local', 'remote') or
                (local_group != (group['source_role'] == 'local'))):
            raise ValueError('invalid field group identity')
        for name in ('ref', 'anchor_ref'):
            if name in group:
                known.add(field_key(group[name]))
        for ref in group.get('primary_refs', []):
            known.add(field_key(ref))
        if type(group['tensor_locations']) is not list:
            raise ValueError('invalid structured tensor locations')
        for location in group['tensor_locations']:
            names = ('array', 'entity', 'source_slot') if group['kind'] == 'observation' else (
                'array', 'entity', 'forecast_set')
            _keys(location, names, 'structured tensor location')
            expected_array = 'observations' if group['kind'] == 'observation' else 'forecasts'
            if (location['array'] != expected_array or location['entity'] != group['tensor_index'] or
                    type(location[names[-1]]) is not int):
                raise ValueError('field group tensor location mismatch')
    if (entity_refs - known or set(admitted) | set(dropped) != known or
            (set(acquired) & set(derived))):
        raise ValueError('admission report does not partition known fields')
    for key, parents in derived_parents.items():
        if key in set(derived) and (parents - set(acquired) or
                                    (key in set(admitted) and not parents <= set(admitted))):
            raise ValueError('derived forecast parent closure is incomplete')
    if any(key in set(derived) and key not in derived_parents for key in admitted):
        raise ValueError('admitted derived forecast lacks parent metadata')
    if any(key not in set(admitted) | set(dropped) for key in previous_keys):
        raise ValueError('previous parent is not known to admission')
    if any(key in set(derived) and not derived_parents.get(key, set()) <= set(admitted)
           for key in previous_keys):
        raise ValueError('previous derived parent closure is incomplete')
    for item in report['dropped']:
        _keys(item, ('ref', 'reason'), 'dropped field')
        if field_key(item['ref']) not in set(dropped) or type(item['reason']) is not str:
            raise ValueError('dropped field detail mismatch')
    if ({field_key(item['ref']) for item in report['dropped']} != set(dropped) or
            len(report['observation_receipt_ids']) != len(set(report['observation_receipt_ids'])) or
            any(type(value) is not str or not value for value in report['observation_receipt_ids'])):
        raise ValueError('invalid observation receipt IDs')
    try:
        json.dumps(prepared, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('prepared input is not finite JSON') from exc
    return prepared
