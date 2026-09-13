"""Freeze a complete method-run contract before any original model is loaded."""
import copy
from collections.abc import Mapping
import math
from pathlib import Path
import shutil

import numpy as np

from common.audit_protocol import recording
from planning.method_controls import normalize_control
from planning.method_episode import validate_limits
from planning.query_data import _semantic_binding, validate_utility_spec
from tools.task_spec import _check_json_native, _version_pair, validate_provenance


RUN_VERSION = 'toolv2x_method_run_spec_v1'
QUERY_SPEC = dict(state_version='toolv2x_query_state_v1',
    feature_version='toolv2x_query_features_v1', action_version='toolv2x_value_actions_v1')
CHECKPOINT_KINDS = dict(request='shared', bundle='bundle_terminal')


def predictor_load_identity(value):
    """Path-free identity of the MTR state that the loader actually accepted."""
    required = {'checkpoint_metadata', 'model_class', 'state_items_loaded',
        'missing_keys', 'unexpected_keys', 'parameter_count'}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError('complete actual predictor load metadata required')
    metadata = value['checkpoint_metadata']
    if not isinstance(metadata, dict) or set(metadata) != {'epoch', 'it', 'version'}:
        raise ValueError('complete predictor checkpoint metadata required')
    if any(v is not None and (type(v) is not int or v < 0) for v in (metadata['epoch'], metadata['it'])):
        raise ValueError('invalid predictor checkpoint epoch or iteration')
    if metadata['version'] is not None and (not isinstance(metadata['version'], str) or not metadata['version']):
        raise ValueError('invalid predictor checkpoint version')
    if (not isinstance(value['model_class'], str) or not value['model_class'] or
            any(type(value[k]) is not int or value[k] < 1 for k in ('state_items_loaded', 'parameter_count')) or
            any(not isinstance(value[k], list) or any(not isinstance(item, str) for item in value[k])
                for k in ('missing_keys', 'unexpected_keys'))):
        raise ValueError('invalid predictor load result')
    result = dict(version='toolv2x_predictor_load_identity_v1',
        checkpoint_metadata=copy.deepcopy(metadata), model_class=value['model_class'],
        state_items_loaded=value['state_items_loaded'], missing_keys=copy.deepcopy(value['missing_keys']),
        unexpected_keys=copy.deepcopy(value['unexpected_keys']), parameter_count=value['parameter_count'])
    _check_json_native(result)
    return result


def _runtime_binding(value):
    if not isinstance(value, dict) or set(value) != {'driver', 'predictor'}:
        raise ValueError('explicit driver and predictor runtime binding required')
    driver, predictor = value['driver'], value['predictor']
    if not isinstance(driver, dict) or not isinstance(predictor, dict):
        raise ValueError('invalid runtime binding')
    if not {'model_version', 'settings'} <= set(predictor):
        raise ValueError('predictor model version and settings required')
    _version_pair(predictor['model_version'])
    settings = predictor['settings']
    if (not isinstance(settings, dict) or 'predictor_load_identity' not in settings or
            predictor_load_identity(settings['predictor_load_identity']) != settings['predictor_load_identity']):
        raise ValueError('actual path-free predictor load identity required')
    _check_json_native(value)
    return copy.deepcopy(value)


def freeze_method_run_spec(value, rows):
    """Validate explicit JSON configuration and bind its exact causal sample set."""
    value = copy.deepcopy(value)
    _check_json_native(value)
    required = {'version', 'limits', 'local_provenance', 'control', 'utility_spec',
        'recording_folds', 'recording_roles', 'runtime_binding', 'query_spec'}
    if not isinstance(value, dict) or set(value) != required or value['version'] != RUN_VERSION:
        raise ValueError('complete versioned method run specification required')
    value['limits'] = validate_limits(value['limits'])
    validate_provenance(value['local_provenance'])
    value['control'] = normalize_control(value['control'])
    value['utility_spec'] = validate_utility_spec(value['utility_spec'])
    value['runtime_binding'] = _runtime_binding(value['runtime_binding'])
    if value['query_spec'] != QUERY_SPEC:
        raise ValueError('explicit supported query state, feature and action versions required')
    if value['runtime_binding']['predictor']['model_version'] != value['local_provenance']['prediction']:
        raise ValueError('runtime predictor and local prediction provenance differ')
    semantic = _semantic_binding(value['runtime_binding'], value)
    value['runtime_binding'] = dict(driver=semantic['driver']['settings'], predictor=semantic['predictor'])

    folds, roles = value['recording_folds'], value['recording_roles']
    if (not isinstance(folds, dict) or not folds or not isinstance(roles, dict) or not roles or
            set(folds) != set(roles) or
            any(recording(k) != k or not isinstance(v, str) or not v for k, v in folds.items()) or
            any(recording(k) != k or v not in ('train', 'validation') for k, v in roles.items())):
        raise ValueError('whole-recording folds and research roles required')
    rows = copy.deepcopy(list(rows))
    online_fields = {'sample_id', 'scene', 'g', 'role', 'physical_split', 'local_frame',
        'ego_motion', 'feature_read_paths', 'motion_read_paths'}
    samples = []
    for row in rows:
        _check_json_native(row)
        group = recording(row.get('scene', ''))
        if (set(row) != online_fields or row['physical_split'] != 'train' or group not in folds or
                roles.get(group) != row['role'] or row['role'] not in ('train', 'validation') or
                type(row['g']) is not int or row['g'] < 0 or
                type(row['local_frame']) is not int or row['local_frame'] < 0):
            raise ValueError('causal sample role, fold or identity differs from the run specification')
        samples.append(dict(sample_id=row['sample_id'], scene=row['scene'], g=row['g'],
            local_frame=row['local_frame'], physical_recording=group, role=row['role'], fold=folds[group]))
    identities = [(r['sample_id'], r['scene'], r['g'], r['local_frame']) for r in samples]
    if (not samples or len(identities) != len(set(identities)) or
            len({r['sample_id'] for r in samples}) != len(samples)):
        raise ValueError('nonempty unique exact sample keys required')
    value.update(samples=samples, value_checkpoints={})
    _check_json_native(value)
    return value


def validate_runtime_binding(actual, spec):
    """Compare semantic model/query bindings before reading a sample or calling a driver."""
    actual = _runtime_binding(actual)
    expected = _semantic_binding(spec['runtime_binding'], spec)
    if _semantic_binding(actual, spec) != expected:
        raise ValueError('runtime binding differs from the frozen run specification')
    return expected


def _same_checkpoint(left, right):
    """Compare saved semantics and weights directly, including NumPy/Torch arrays."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_checkpoint(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same_checkpoint(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if hasattr(left, 'detach') or hasattr(right, 'detach'):
        if not (hasattr(left, 'detach') and hasattr(right, 'detach')):
            return False
        a, b = left.detach().cpu(), right.detach().cpu()
        return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool((a == b).all().item())
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return type(left) is type(right) and left == right


def prepare_value_checkpoints(selections, out, expected_binding, utility_spec, *, resume_from=None,
                              previous_manifest=None, policy_loader=None, checkpoint_reader=None):
    """Validate selected policy roles, compare any resume source, then copy snapshots."""
    if not isinstance(selections, dict) or not selections or set(selections) - set(CHECKPOINT_KINDS):
        raise ValueError('explicit request and/or bundle value checkpoint selection required')
    if policy_loader is None:
        from planning.query_value import load_query_policy
        policy_loader = load_query_policy
    if checkpoint_reader is None:
        from planning.query_value import _torch_load
        checkpoint_reader = _torch_load
    control = expected_binding.get('control')
    if 'bundle' in selections and (not isinstance(control, dict) or control.get('name') != 'one_shot'):
        raise ValueError('bundle value checkpoint requires a one_shot control')
    out = Path(out);directory = out/'value_checkpoints'
    directory.mkdir(parents=True, exist_ok=False)
    policies, saved, snapshots = {}, {}, {}
    try:
        for role, source in selections.items():
            source = Path(source);snapshot = directory/(role + source.suffix)
            shutil.copy2(source, snapshot)
            policy = policy_loader(snapshot, expected_binding=expected_binding, policy_kind=CHECKPOINT_KINDS[role])
            if policy.kind != CHECKPOINT_KINDS[role]:
                raise ValueError('wrong checkpoint policy kind for ' + role)
            if policy.config['utility_spec'] != utility_spec:
                raise ValueError('checkpoint utility differs from the run specification')
            expected_id = (control['bundle']['continuation_policy_id'] if role == 'bundle'
                else expected_binding['limits']['policy_id'])
            if policy.config['policy_id'] != expected_id:
                raise ValueError('checkpoint policy identifier differs from its run role')
            _check_json_native(policy.provenance)
            policies[role], saved[role], snapshots[role] = policy, checkpoint_reader(snapshot), snapshot

        if resume_from is not None:
            if not isinstance(previous_manifest, dict) or set(previous_manifest) != set(selections):
                raise ValueError('resume checkpoint roles changed')
            for role in selections:
                previous = previous_manifest[role]
                if (previous.get('policy_kind') != CHECKPOINT_KINDS[role] or
                        previous.get('provenance') != policies[role].provenance):
                    raise ValueError('resume checkpoint provenance changed')
                relative = Path(previous.get('snapshot', ''))
                if relative.is_absolute() or relative.parent != Path('value_checkpoints'):
                    raise ValueError('invalid resume checkpoint snapshot reference')
                snapshot = Path(resume_from) / relative
                if not snapshot.is_file() or not _same_checkpoint(saved[role], checkpoint_reader(snapshot)):
                    raise ValueError('selected value checkpoint changed since the prior run')
    except BaseException:
        shutil.rmtree(directory)
        try:out.rmdir()
        except OSError:pass
        raise

    manifest = {}
    for role, snapshot in snapshots.items():
        relative = snapshot.relative_to(out)
        manifest[role] = dict(version='toolv2x_value_checkpoint_snapshot_v1', role=role,
            policy_kind=CHECKPOINT_KINDS[role], snapshot=str(relative),
            provenance=copy.deepcopy(policies[role].provenance))
    return manifest, policies
