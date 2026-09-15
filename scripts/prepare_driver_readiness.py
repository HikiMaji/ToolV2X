"""Prepare and validate frozen metadata/configs without reading arrays or models."""
import argparse
from collections import Counter, OrderedDict
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from common.audit_protocol import recording
from planning.method_controls import control_spec, episode_bundle, normalize_control
from planning.method_episode import DIAGNOSTIC_POLICY_IDS
from planning.structured_inputs import StructuredDriverSpec
from planning.train_structured_driver import _config
from probe.kinematic_tools import encode
from tools.control_bundle import validate_bundle_envelope
from tools.task_spec import ExecutionSpec, _check_json_native


FRAME_FIELDS = {'sample_id', 'scene', 'g', 'role', 'physical_split', 'local_frame',
                'ego_motion', 'feature_read_paths', 'motion_read_paths'}
ACCEPTANCE_FIELDS = {'sample_id', 'scene', 'g', 'role', 'physical_split', 'local_frame'}
CONDITION_POLICIES = {'Ego': 'stop', 'P': 'p_current', 'F': 'f_current',
                      'PF': 'p_current_f_current'}
EXPECTED_COUNTS = {'train': 2987, 'validation': 108}
EXPECTED_ROLES = {
    'testoutput_CAV_data_2022-03-15-10-09-50': 'train',
    'testoutput_CAV_data_2022-03-15-16-00-17': 'train',
    'testoutput_CAV_data_2022-03-17-10-43-13': 'train',
    'testoutput_CAV_data_2022-03-17-12-04-22': 'train',
    'testoutput_CAV_data_2022-03-17-12-14-33': 'train',
    'testoutput_CAV_data_2022-03-17-14-44-47': 'train',
    'testoutput_CAV_data_2022-03-17-14-59-30': 'train',
    'testoutput_CAV_data_2022-03-17-15-55-26': 'train',
    'testoutput_CAV_data_2022-03-17-16-06-11': 'validation',
    'testoutput_CAV_data_2022-03-21-09-50-20': 'validation',
}
EXPECTED_EXCLUSIONS = {
    'train': {'original': 3027, 'kept': 2987, 'excluded': 40, 'recordings': 8},
    'validation': {'original': 108, 'kept': 108, 'excluded': 0, 'recordings': 2},
}
EXECUTION_SPEC = {
    'version': 'toolv2x_execution_v1', 'profile': 'real_smoke_v1', 'sigma_m': 5.0,
    'max_targets': 4, 'max_request_bytes': 16384, 'max_response_bytes': 65536,
    'max_episode_bytes': 196608, 'ego_geometry': 'circumscribed_circle',
    'ego_length_m': 4.8, 'ego_width_m': 2.0, 'max_plan_speed_mps': 80.0,
    'max_plan_acceleration_mps2': 30.0,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def training_config(epochs=20, interval=100):
    return {
        'version': 'toolv2x_structured_training_v2',
        'driver_spec': StructuredDriverSpec().to_dict(),
        'seed': 7,
        'optimizer': {'name': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.01},
        'batch_size': 2,
        'epochs': epochs,
        'refinement_depth': 1,
        'save_interval': 25,
        'device': 'cuda',
        'objective': {'name': 'selected_stage_and_refinement', 'reduction': 'mean'},
        'row_weighting': 'frame_condition_equal',
        'validation': {'interval_batches': interval, 'initial': True, 'final': True},
        'selection': {'name': 'invalid_rate_prefix_l2_earliest', 'output': 'selected_stage'},
    }


def controls():
    feedback = control_spec('feedback')
    one_shot = control_spec('one_shot', driver_calls=3, bundle={
        'continuation_policy_id': 'diagnostic_conditional_v1',
        'candidate_sources': ['initial', 'slower', 'constant_motion'],
        'slowdown_scale': 0.5,
        'max_candidates': 3,
        'wrapper_reserve_bytes': 2048,
        'extra_generation': 'self_refinement',
        'summary_spec': {
            'version': 'toolv2x_bundle_summary_v1',
            'selection': 'distance_then_field',
            'field_kinds': ['anchor', 'forecast'],
            'max_fields': 8,
            'max_bytes': 1536,
        },
        'response_budget': {
            'version': 'toolv2x_bundle_budget_v1',
            'mode': 'episode_aggregate',
            'outer_response_cap': None,
            'primitive_response_caps': None,
        },
    })
    return {'feedback': feedback, 'one_shot': one_shot}


def readiness(source):
    return {
        'version': 'toolv2x_structured_driver_readiness_v1',
        'lifecycle': 'prepared_not_executed',
        'runtime_load_identity': None,
        'execution_spec': copy.deepcopy(EXECUTION_SPEC),
        'driver_spec': StructuredDriverSpec().to_dict(),
        'local_provenance': copy.deepcopy(source['local_provenance']),
        'source': {
            'frame_manifest': 'frames.jsonl',
            'acceptance_manifest': 'acceptance_frames.json',
            'source_index': source['source_index'],
            'source_split_manifest': source['source_split_manifest'],
            'source_exclusions_config': source['source_exclusions_config'],
            'paths_are_audit_only': True,
            'portable_validation': 'full structural and deterministic selection validation; optional exact local source comparison',
        },
        'frame_counts': copy.deepcopy(source['counts']),
        'acceptance_counts': {'train': 16, 'validation': 4},
        'recording_roles': copy.deepcopy(source['recording_roles']),
        'existing_exclusion_counts': copy.deepcopy(source['existing_exclusion_counts']),
        'selection': {
            'acceptance_per_physical_recording': 2,
            'method': 'first_and_last_eligible_in_source_order',
            'basis': 'causal identity coverage only; no labels, quality, gains, or model outputs',
            'all_failures_remain_in_denominator': True,
            'adaptation_population': 'all 3095 accepted frame metadata rows',
        },
        'collection': {
            'conditions': copy.deepcopy(CONDITION_POLICIES),
            'diagnostic_policy_ids': list(DIAGNOSTIC_POLICY_IDS),
            'max_remote_calls': 2,
            'max_driver_attempts': 3,
            'shared_driver_and_inputs': True,
            'p_processing': 'observations_only',
            'runtime_requirements': ['actual driver load identity', 'actual predictor load identity',
                                     'fresh non-existing output paths', 'output capacity preflight'],
            'ready_to_run_method_spec': False,
        },
        'controls': controls(),
        'bootstrap': {
            'status': 'conditional_recipe_only',
            'source': 'retained initial Ego stage-0 prepared inputs from the 20-frame integration attempt',
            'expected_training_rows': 16,
            'expected_validation_rows': 4,
            'expected_batches_per_epoch': 8,
            'eligible_label_count_certified': False,
            'runtime_interval_rule': 'ceil(eligible_train_rows/2)',
            'maximum_epochs': 3,
            'acceptance_after_each_epoch': True,
            'failed_and_missing_coverage_retained': True,
            'full_adaptation_initialization': 'fresh_seeded_initialization_after_recollection',
            'warm_start_api_available': False,
        },
        'metrics': {
            'diagnostic': ['finite_execution_valid_initial_rate', 'requested_acquisition_boundary_rate',
                           'budget_empty_rate', 'history_receipt_direct_use_audit'],
            'selection': ['weighted_invalid_plan_rate', 'got_prefix_L2_avg', 'earliest_processed_batch'],
            'interpretation': 'integration and offline imitation diagnostics; not paper superiority or closed-loop safety',
        },
        'stage_gates': {
            'B': 'audit all 20x4 initial tasks; conditionally bootstrap at most 3 epochs only if validity blocks evidence coverage; otherwise stop on unexpected failures',
            'C': 'after full adaptation, recollect all four conditions with one frozen shared driver and report every failure/STOP/call count',
            'D': 'compare feedback and controls under the same weights, inputs, evidence rules, and total bidirectional budget',
        },
        'execution_status': {'collection': False, 'training': False, 'inference': False,
                             'provider_access': False, 'model_access': False},
    }


def validate_source_metadata(source):
    expected = {'source_index', 'source_bytes', 'source_split_manifest',
                'source_exclusions_config', 'source_execution_spec', 'counts',
                'recording_roles', 'existing_exclusion_counts', 'execution_spec',
                'local_provenance', 'approved_row_fields',
                'forbidden_original_fields_removed', 'scope'}
    if (not isinstance(source, dict) or set(source) != expected or
            source['counts'] != EXPECTED_COUNTS or source['recording_roles'] != EXPECTED_ROLES or
            source['existing_exclusion_counts'] != EXPECTED_EXCLUSIONS or
            source['execution_spec'] != EXECUTION_SPEC or set(source['approved_row_fields']) != FRAME_FIELDS or
            set(source['forbidden_original_fields_removed']) !=
            {'window_archive_refs', 'action_queries', 'tool_outputs_materialized', 'directory', 'feature_frames'} or
            type(source['source_bytes']) is not int or source['source_bytes'] < 1 or
            any(not isinstance(source[key], str) or not source[key]
                for key in ('source_index', 'source_split_manifest', 'source_exclusions_config',
                            'source_execution_spec', 'scope'))):
        raise ValueError('source metadata differs from the controller-verified frozen population')
    ExecutionSpec.from_dict(source['execution_spec'])


def acceptance_rows(rows):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault((row['role'], recording(row['scene'])), []).append(row)
    selected = []
    for group in groups.values():
        for row in (group[:1] if len(group) == 1 else (group[0], group[-1])):
            selected.append({key: copy.deepcopy(row[key]) for key in
                             ('sample_id', 'scene', 'g', 'role', 'physical_split', 'local_frame')})
    return selected


def validate_frames(rows, selected, config):
    sample_ids, scene_frames, scene_indices = set(), set(), set()
    roles = {}
    counts = Counter()
    for row in rows:
        _check_json_native(row)
        if set(row) != FRAME_FIELDS or row['role'] not in ('train', 'validation') or row['physical_split'] != 'train':
            raise ValueError('invalid or noncausal frame metadata fields')
        identities = (row['sample_id'], (row['scene'], row['local_frame']),
                      (row['scene'], row['g']))
        if (identities[0] in sample_ids or identities[1] in scene_frames or
                identities[2] in scene_indices or not isinstance(row['sample_id'], str) or
                row['sample_id'] != '%s:%d' % (row['scene'], row['local_frame'])):
            raise ValueError('duplicate or invalid frame identity')
        sample_ids.add(identities[0])
        scene_frames.add(identities[1])
        scene_indices.add(identities[2])
        if type(row['g']) is not int or row['g'] < 0 or type(row['local_frame']) is not int or row['local_frame'] < 0:
            raise ValueError('invalid frame number')
        motion = row['ego_motion']
        if (not isinstance(motion, dict) or set(motion) != {'speed_mps', 'yaw_rate_rps'} or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in motion.values())):
            raise ValueError('invalid causal ego motion')
        for key in ('feature_read_paths', 'motion_read_paths'):
            if not isinstance(row[key], list) or not row[key] or any(not isinstance(path, str) or not path for path in row[key]):
                raise ValueError('invalid audit-only read paths')
        group = recording(row['scene'])
        if group in roles and roles[group] != row['role']:
            raise ValueError('physical recording crosses research roles')
        roles[group] = row['role']
        counts[row['role']] += 1
    if dict(counts) != config['frame_counts'] or roles != config['recording_roles']:
        raise ValueError('frozen frame counts or recording roles changed')
    if any(not isinstance(row, dict) or set(row) != ACCEPTANCE_FIELDS for row in selected):
        raise ValueError('invalid acceptance identity fields')
    if selected != acceptance_rows(rows):
        raise ValueError('acceptance frame selection changed')
    if dict(Counter(row['role'] for row in selected)) != config['acceptance_counts']:
        raise ValueError('acceptance role counts changed')


def verify_one_shot(config):
    execution = config['execution_spec']
    state = {
        'provider': 'peer', 'scene': 'public_contract_fixture', 'g': 0,
        'ego_motion': {'speed_mps': 2.0, 'yaw_rate_rps': 0.0},
        'local_evidence': [], 'response_receipts': [],
        'current_plan': {'waypoints': [[float(i), 0.0] for i in range(1, 7)]},
        'remaining_budget': {'calls': 2, 'bytes': execution['max_episode_bytes']},
        'execution_spec': execution,
    }
    spec = normalize_control(config['controls']['one_shot'])
    request = episode_bundle(state, spec, 'P')
    validate_bundle_envelope(request)
    size = len(encode(request))
    outer = request['limits']['outer_response_cap']
    if size > execution['max_request_bytes'] or size + outer > execution['max_episode_bytes']:
        raise ValueError('actual one-shot request and permitted response exceed execution budget')
    return {'request_bytes': size, 'outer_response_cap': outer,
            'candidate_ids': [candidate['candidate_id'] for candidate in request['candidates']]}


def validate_package(package, source_frames=None):
    package = Path(package)
    config = read_json(package / 'readiness.json')
    source = read_json(package / 'source_metadata.json')
    rows = read_jsonl(package / 'frames.jsonl')
    selected = read_json(package / 'acceptance_frames.json')
    validate_source_metadata(source)
    if config != readiness(source):
        raise ValueError('readiness metadata differs from the frozen preparation recipe')
    if config['lifecycle'] != 'prepared_not_executed' or config['runtime_load_identity'] is not None:
        raise ValueError('preparation cannot claim runtime identity or execution')
    if ExecutionSpec.from_dict(config['execution_spec']).to_dict() != EXECUTION_SPEC:
        raise ValueError('execution settings changed')
    driver = StructuredDriverSpec.from_dict(config['driver_spec']).to_dict()
    if driver != StructuredDriverSpec().to_dict() or driver['p_processing'] != 'observations_only':
        raise ValueError('shared driver must use the complete observations-only default')
    validate_frames(rows, selected, config)
    if source_frames is not None and rows != read_jsonl(source_frames):
        raise ValueError('packaged frame manifest differs from the whitelisted local source')
    train = read_json(package / 'training_v2.json')
    bootstrap = read_json(package / 'bootstrap_training_expected_v2.json')
    if _config(train) != train or train != training_config():
        raise ValueError('full adaptation training config changed')
    if _config(bootstrap) != bootstrap or bootstrap != training_config(3, 8):
        raise ValueError('expected bootstrap training config changed')
    if train['driver_spec'] != driver or bootstrap['driver_spec'] != driver:
        raise ValueError('training driver specification differs from collection')
    if normalize_control(config['controls']['feedback']) != config['controls']['feedback']:
        raise ValueError('invalid feedback control')
    result = {'frame_counts': config['frame_counts'], 'acceptance_counts': config['acceptance_counts'],
              'one_shot': verify_one_shot(config), 'source_comparison': source_frames is not None}
    return result


def prepare_package(source_frames, acceptance_preview, source_metadata, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('preparation output already exists: ' + str(out))
    source = read_json(source_metadata)
    rows = read_jsonl(source_frames)
    selected = read_json(acceptance_preview)
    config = readiness(source)
    validate_frames(rows, selected, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + '.', dir=str(out.parent)) as temporary:
        target = Path(temporary) / out.name
        target.mkdir()
        shutil.copyfile(source_frames, target / 'frames.jsonl')
        shutil.copyfile(acceptance_preview, target / 'acceptance_frames.json')
        shutil.copyfile(source_metadata, target / 'source_metadata.json')
        for name, value in (('readiness.json', config), ('training_v2.json', training_config()),
                            ('bootstrap_training_expected_v2.json', training_config(3, 8))):
            (target / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        validate_package(target, source_frames=source_frames)
        target.replace(out)
    return validate_package(out, source_frames=source_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--source-frames', type=Path)
    parser.add_argument('--acceptance-preview', type=Path)
    parser.add_argument('--source-metadata', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        if args.acceptance_preview is not None or args.source_metadata is not None:
            parser.error('--acceptance-preview and --source-metadata are preparation-only')
        result = validate_package(args.out, source_frames=args.source_frames)
    else:
        missing = [name for name in ('source_frames', 'acceptance_preview', 'source_metadata')
                   if getattr(args, name) is None]
        if missing:
            parser.error('preparation requires --source-frames, --acceptance-preview, and --source-metadata')
        result = prepare_package(args.source_frames, args.acceptance_preview, args.source_metadata, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
