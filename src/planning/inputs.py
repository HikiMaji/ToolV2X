"""Ego-only shallow features and causal Q8/Q9 inputs; never read QA labels."""
import ast
import copy
import json
import re
from pathlib import Path
import numpy as np
from common.v2v4real_meta import LEN_RECORD, seq_of

FEATURE_SHAPES = {'regression_map': (1, 14, 50, 88), 'classification_map': (1, 2, 50, 88)}
SPEEDS = ('fast', 'moderate', 'slow', 'very slow', 'stop')
STEERING = ('left', 'slightly left', 'straight', 'slightly right', 'right')
OBJECT_FIELDS = {'source', 'track_id', 'box', 'score', 'history', 'history_valid',
                 'history_times', 'forecast', 'forecast_scores', 'forecast_times', 'model_used'}
TASK_OBJECT_FIELDS = OBJECT_FIELDS | {'history_scores', 'proxy_status', 'field_metadata'}


def frame_location(root, split, g):
    if split not in LEN_RECORD or isinstance(g, bool) or not isinstance(g, (int, np.integer)) or not 0 <= g < LEN_RECORD[split][-1]:
        raise ValueError('invalid physical split/global frame')
    _, _, start, _ = seq_of(g, split)
    return Path(root) / (('train_' if split == 'train' else '') + 'no_fusion_keep_all') / 'npy', start


def load_ego_features(root, split, g):
    """Match upstream shallow feature layout, keeping an explicit missing-past mask.

    Only regression/classification maps and detection_box_score are used by the
    original shallow model. Deep BEV/object features and RGB are not read.
    """
    directory, start = frame_location(root, split, g)
    directory = directory / 'co_llm/ego'
    frames = [int(g), int(g - 1) if g > start else None]
    result = {'frame_indices': frames, 'read_paths': [], 'truncated_boxes': []}
    maps = {key: [] for key in FEATURE_SHAPES}
    boxes = []
    for frame in frames:
        for key, shape in FEATURE_SHAPES.items():
            if frame is None:
                value = np.zeros(shape, np.float32)
            else:
                path = directory / ('%04d_%s.npy' % (frame, key))
                value = np.load(path, allow_pickle=False)
                result['read_paths'].append(str(path))
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('invalid feature map: ' + str(path))
            maps[key].append(value.astype(np.float32, copy=False))
        if frame is None:
            value = np.empty((0, 8), np.float32)
        else:
            path = directory / ('%04d_detection_box_score.npy' % frame)
            value = np.load(path, allow_pickle=False)
            result['read_paths'].append(str(path))
            if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
                raise ValueError('invalid detection features: ' + str(path))
            # Exact axis permutation in upstream load_single_frame_detection_box_score.
            value = value[:, [0, 1, 2, 3, 5, 4, 6, 7]]
        result['truncated_boxes'].append(max(0, len(value) - 50))
        padded = np.zeros((1, 50, 8), np.float32)
        padded[0, :min(len(value), 50)] = value[:50]
        boxes.append(padded)
    result.update({key: np.stack(value)[None] for key, value in maps.items()})
    result['detection_box_score'] = np.stack(boxes)[None]
    result['active_agent_mask'] = np.array([[[frame is not None] for frame in frames]], bool)
    return result


def load_ego_motion(root, split, g):
    directory, start = frame_location(root, split, g)
    paths, poses = [], []
    for frame in ([g] if g == start else [g, g - 1]):
        path = directory / 'ego' / ('%04d_lidar_pose.npy' % frame)
        pose = np.load(path, allow_pickle=False)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError('invalid current/past localization: ' + str(path))
        paths.append(str(path))
        poses.append(pose)
    if len(poses) == 1:
        return dict(speed_mps=None, yaw_rate_rps=None), paths
    yaw = [np.arctan2(p[1, 0], p[0, 0]) for p in poses]
    delta = np.arctan2(np.sin(yaw[0] - yaw[1]), np.cos(yaw[0] - yaw[1]))
    return dict(speed_mps=float(np.linalg.norm(poses[0][:2, 3] - poses[1][:2, 3]) / .1),
                yaw_rate_rps=float(delta / .1)), paths


def validate_evidence(evidence):
    if isinstance(evidence, dict) and evidence.get('evidence_version') == 'toolv2x_driver_evidence_v2':
        return _validate_task_evidence(evidence)
    allowed = {'as_of_g', 'coordinate_frame', 'objects', 'relations', 'queries'}
    if not isinstance(evidence, dict) or set(evidence) - allowed or 'as_of_g' not in evidence or 'objects' not in evidence:
        raise ValueError('invalid evidence fields')
    if not isinstance(evidence['objects'], list):
        raise ValueError('objects must be a list')
    for obj in evidence['objects']:
        if not isinstance(obj, dict) or set(obj) - OBJECT_FIELDS:
            raise ValueError('unexpected object evidence fields')
    for relation in evidence.get('relations', []):
        if set(relation) - {'ego_id', 'peer_id', 'distance_m', 'status'}:
            raise ValueError('unexpected relation fields')
    for query in evidence.get('queries', []):
        if set(query) - {'tool', 'roi', 'response_bytes', 'status'}:
            raise ValueError('unexpected query fields')
    # Fail on NaN/inf rather than sending nonstandard JSON to the driving model.
    json.dumps(evidence, allow_nan=False)


def _validate_task_evidence(evidence):
    from tools.task_spec import _keys, _integer, _text, _version_pair, _array
    from tools.vehicle import _validate_task_value
    _keys(evidence, ('evidence_version', 'as_of_g', 'coordinate_frame', 'objects', 'observations'), 'driver v2 evidence')
    _integer(evidence['as_of_g'])
    if evidence['coordinate_frame'] != 'ego_at_t' or not isinstance(evidence['objects'], list):
        raise ValueError('invalid v2 driver coordinates/objects')
    for obj in evidence['objects']:
        if set(obj) - TASK_OBJECT_FIELDS or not {'source', 'track_id', 'box', 'score', 'field_metadata'} <= set(obj):
            raise ValueError('invalid v2 driver object fields')
        _text(obj['source'])
        _integer(obj['track_id'])
        _validate_task_value('anchor', dict(box=obj['box'], score=obj['score']))
        kinds = {k for k in ('history', 'forecast') if k in obj}
        if not kinds:
            raise ValueError('driver object has no primary bundle')
        _keys(obj['field_metadata'], kinds, 'driver field metadata')
        expected = {'source', 'track_id', 'box', 'score', 'field_metadata'}
        for kind in kinds:
            meta = obj['field_metadata'][kind]
            _keys(meta, ('context_version', 'producer_version', 'context_scope') if kind == 'forecast'
                  else ('context_version', 'producer_version'), 'driver field provenance')
            for key in ('context_version', 'producer_version'):
                _version_pair(meta[key])
            fields = {'history', 'history_valid', 'history_scores', 'history_times', 'proxy_status'} if kind == 'history' else {
                'forecast', 'forecast_scores', 'forecast_times', 'model_used'}
            if not fields <= set(obj):
                raise ValueError('incomplete driver primary bundle')
            value = {k: obj[k] for k in fields}
            if kind == 'forecast':
                if meta['context_scope'] not in ('provider_full_at_t', 'receiver_acquired_subset'):
                    raise ValueError('unknown driver forecast context scope')
                paths = _array(value['forecast'], (6, 6, 2))
                scores = _array(value['forecast_scores'], (6,))
                # Display rounding can add at most 0.005 per mode. Never normalize.
                if ((scores < 0).any() or (scores > 1).any() or scores.sum() > 1.0001 + 6 * .005 or
                        not np.array_equal(_array(value['forecast_times'], (6,)), [.5, 1., 1.5, 2., 2.5, 3.]) or
                        type(value['model_used']) is not bool or
                        (not value['model_used'] and not np.all(paths == np.asarray(obj['box'][:2])))):
                    raise ValueError('invalid rounded driver forecast')
            else:
                _validate_task_value(kind, value)
            expected |= fields
        if set(obj) != expected:
            raise ValueError('orphan driver numeric fields')
    if not isinstance(evidence['observations'], list) or not evidence['observations']:
        raise ValueError('remote v2 evidence requires paid source observations')
    for item in evidence['observations']:
        _keys(item, ('provider', 'coverage', 'history_complete'), 'driver observation')
        _text(item['provider'])
        if item['coverage'] != 'not_established' or type(item['history_complete']) is not bool:
            raise ValueError('invalid driver observation boundary')
    json.dumps(evidence, allow_nan=False)


def pack_evidence(evidence):
    """Lossless structural compression; precision is decided before this codec."""
    validate_evidence(evidence)
    objects = copy.deepcopy(evidence['objects'])
    if any(value is None for obj in objects for value in obj.values()):
        raise ValueError('null object fields are reserved for absent columns')
    shared = {}
    if objects:
        for key, value in objects[0].items():
            if (key not in {'source', 'track_id', 'box', 'forecast', 'history'} and
                    all(key in obj and obj[key] == value for obj in objects)):
                shared[key] = value
        for obj in objects:
            for key in shared:
                del obj[key]
    for obj in objects:
        if 'forecast' not in obj:
            continue
        unique, indices = [], []
        for path in obj['forecast']:
            if path not in unique:
                unique.append(path)
            indices.append(unique.index(path))
        if len(unique) < len(indices):
            obj['forecast'] = dict(unique_paths=unique, mode_to_path=indices)
    columns = sorted(set().union(*(set(obj) for obj in objects))) if objects else []
    result = {key: copy.deepcopy(value) for key, value in evidence.items() if key != 'objects'}
    result.update(evidence_format='compact_v1', object_shared=shared, object_columns=columns,
                  object_rows=[[obj.get(key) for key in columns] for obj in objects])
    return result


def unpack_evidence(packed):
    if packed.get('evidence_format') != 'compact_v1':
        raise ValueError('unknown evidence format')
    columns, shared = packed['object_columns'], packed['object_shared']
    allowed = TASK_OBJECT_FIELDS if packed.get('evidence_version') == 'toolv2x_driver_evidence_v2' else OBJECT_FIELDS
    if (len(set(columns)) != len(columns) or set(columns) & set(shared) or
            (set(columns) | set(shared)) - allowed):
        raise ValueError('invalid evidence columns')
    objects = []
    for row in packed['object_rows']:
        if len(row) != len(columns):
            raise ValueError('evidence row length mismatch')
        obj = dict(copy.deepcopy(shared), **{key: copy.deepcopy(value) for key, value in zip(columns, row) if value is not None})
        if isinstance(obj.get('forecast'), dict):
            forecast = obj['forecast']
            if set(forecast) != {'unique_paths', 'mode_to_path'}:
                raise ValueError('invalid forecast references')
            paths, indices = forecast['unique_paths'], forecast['mode_to_path']
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(paths) for i in indices):
                raise ValueError('invalid forecast path index')
            obj['forecast'] = [copy.deepcopy(paths[i]) for i in indices]
        objects.append(obj)
    result = {key: copy.deepcopy(value) for key, value in packed.items()
              if key not in {'evidence_format', 'object_columns', 'object_shared', 'object_rows'}}
    result['objects'] = objects
    validate_evidence(result)
    return result


def make_prompt(task, ego_state, evidence, q8_answer=None, evidence_format='json', remote_evidence=None):
    if set(ego_state) != {'speed_mps', 'yaw_rate_rps'}:
        raise ValueError('unexpected ego state fields')
    for value in ego_state.values():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
            raise ValueError('invalid ego motion state')
    validate_evidence(evidence)
    if evidence_format not in ('json', 'compact'):
        raise ValueError('unknown evidence serialization')
    rendered = evidence if evidence_format == 'json' else pack_evidence(evidence)
    legend = '' if evidence_format == 'json' else (
        'Object rows follow object_columns; object_shared applies to every row; null means absent. '
        'Forecasts list ordered modes, or use mode_to_path to index unique_paths; retain every mode score without renormalizing.\n')
    context = ('Current ego motion: ' + json.dumps(ego_state, sort_keys=True, allow_nan=False) + '\n'
               + legend +
               'Acquired evidence (missing observations do not imply free space; source forecasts may disagree): '
               + json.dumps(rendered, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    if remote_evidence is not None:
        validate_evidence(remote_evidence)
        task_evidence = remote_evidence.get('evidence_version') == 'toolv2x_driver_evidence_v2'
        if (task != 'Trajectory' or evidence.get('queries') or evidence.get('relations') or
                any(obj.get('source') != 'ego' for obj in evidence['objects']) or
                any(obj.get('source') == 'ego' for obj in remote_evidence['objects']) or
                evidence.get('coordinate_frame') != 'ego_at_t' or
                any(remote_evidence.get(key) != evidence.get(key) for key in ('as_of_g', 'coordinate_frame')) or
                (not remote_evidence.get('queries') and not task_evidence)):
            raise ValueError('paired direct prompt requires an isolated ego block and a queried same-time remote block')
        remote = remote_evidence if evidence_format == 'json' else pack_evidence(remote_evidence)
        # Encode separately: adding peer objects must not change the local block's
        # compact columns/shared values or the literal local text.
        context += ('Additional queried neighbor evidence: '
                    + json.dumps(remote, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    if task == 'Q8':
        if q8_answer is not None:
            raise ValueError('Q8 must not receive an answer')
        context += ('Use speed: fast, moderate, slow, very slow, or stop; '
                    'steering: left, slightly left, straight, slightly right, or right.\n')
        question = 'What are the suggested speed and steering settings to avoid collision with nearby objects?'
    elif task in ('Q9', 'Trajectory'):
        context += ('Output six numeric (x,y) waypoints in the current ego LiDAR frame, '
                    'at 0.5, 1, 1.5, 2, 2.5 and 3 seconds.\n')
        if task == 'Q9':
            if not q8_answer:
                raise ValueError('Q9 requires the actual generated Q8 answer')
            parse_q8(q8_answer)
            context += 'Context from the generated action answer: ' + q8_answer + '\n'
        elif q8_answer is not None:
            raise ValueError('direct trajectory task cannot receive an action parent')
        question = 'What is the suggested future trajectory to avoid collision with nearby objects?'
    else:
        raise ValueError('unknown driving task')
    # Keep the native task last. Literal answer placeholders were copied by the
    # released checkpoint; long trailing evidence instead elicited object QA.
    return context + 'I am CAV_EGO at [0.0, 0.0]. ' + question


def parse_q8(text):
    result = {}
    for key, choices in (('speed', SPEEDS), ('steering', STEERING)):
        answers = re.findall(r'\b' + key + r' setting is\s*:\s*([^.;\n]+)', text, re.IGNORECASE)
        if len(answers) != 1 or answers[0].strip().lower() not in choices:
            raise ValueError('invalid or ambiguous Q8 ' + key)
        result[key] = answers[0].strip().lower()
    return result


def parse_q9(text):
    match = re.search(r'(?:suggested\s+)?(?:future\s+)?trajectory is\s*:?\s*(\[.*\])', text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError('missing Q9 trajectory')
    try:
        points = ast.literal_eval(match.group(1))
        if not isinstance(points, (list, tuple)) or len(points) != 6:
            raise ValueError('expected six waypoints')
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in point):
                raise ValueError('expected numeric xy waypoints')
        value = np.asarray(points, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite trajectory')
        return value
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        raise ValueError('invalid Q9 trajectory') from exc


def refinement_prompt(prepared):
    """Regenerate the explicit controls-only model-answer slot; never accept free text."""
    r = prepared['refinement']
    if (set(r) != {'version','slot_tokens','previous_plan'} or r['version'] != 'toolv2x_refinement_v1' or
            type(r['slot_tokens']) is not int or r['slot_tokens'] <= 0):
        raise ValueError('invalid refinement specification')
    previous = r['previous_plan']
    if previous is not None:
        points = np.asarray(previous, dtype=float)
        if points.shape != (6,2) or not np.isfinite(points).all():
            raise ValueError('invalid model-generated refinement parent')
    origin = 'model_generated' if previous is not None else 'none_initial'
    if prepared['previous_plan_origin'] != origin:
        raise ValueError('refinement parent origin mismatch')
    slot = ('Review the trajectory using only the existing evidence. The previous trajectory '
            'below is a model output, not an observation. Produce a new six-point answer.\n'
            'Previous model trajectory: ' + json.dumps(previous, separators=(',',':'),allow_nan=False) + '\n')
    base = make_prompt('Trajectory', prepared['ego_motion'], prepared['evidence_used'],
                      evidence_format='compact', remote_evidence=prepared.get('remote_evidence_used'))
    return slot + base


def build_refinement_input(prepared, previous_plan, *, tokenizer=None, token_counter=None, slot_tokens=256):
    """Fixed E/Z; reject overflow instead of selecting or dropping any evidence."""
    if prepared['input_layout'] not in ('source_blocks_v2','source_blocks_v2_refinement'):
        raise ValueError('refinement requires common v2 receiver')
    result=copy.deepcopy(prepared)
    points=None
    if previous_plan is not None:
        points=parse_q9(previous_plan['raw']).tolist()
        if points != previous_plan['waypoints']:
            raise ValueError('refinement parent differs from actual model answer')
    result.update(input_layout='source_blocks_v2_refinement',
        previous_plan_origin='model_generated' if points is not None else 'none_initial',
        refinement=dict(version='toolv2x_refinement_v1',slot_tokens=slot_tokens,previous_plan=points))
    base=make_prompt('Trajectory',result['ego_motion'],result['evidence_used'],
        evidence_format='compact',remote_evidence=result.get('remote_evidence_used'))
    if token_counter is None:
        from planning.v2vgot import prompt_tokens
        token_counter=lambda text:len(prompt_tokens(tokenizer,text))-1
    prompt=refinement_prompt(result)
    count=token_counter(prompt)+result['evidence_selection']['feature_tokens']
    if (token_counter(prompt)-token_counter(base)>slot_tokens or
            count+result['receiver_spec']['generation_reserve']>result['receiver_spec']['context_limit']):
        raise ValueError('refinement slot/context overflow; evidence cannot be changed')
    result['q9_prompt']=prompt
    result['evidence_selection']['input_tokens']=count
    return result
