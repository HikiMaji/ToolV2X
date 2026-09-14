"""Structured numeric receiver contracts; no model, tokenizer or training access."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_task_spec import provenance, task_request
from test_vehicle_tools import fixture_prediction, window
from tools.task_spec import FrozenPredictor
from tools.vehicle import VehicleTools


def one_track(source, x, track_id=7, y=0., score=1.):
    value = window(source)
    value['track_ids'] = np.array([track_id])
    value['states'] = value['states'][:1].copy()
    value['states'][0, :, 0] = x
    value['states'][0, :, 1] = y
    value['valid'] = value['valid'][:1].copy()
    value['scores'] = value['scores'][:1].copy()
    value['scores'][:] = score
    return value


def empty_window(source='ego'):
    value = window(source)
    for name in ('track_ids', 'states', 'valid', 'scores'):
        value[name] = value[name][:0].copy()
    return value


def predictor_for(callable_=fixture_prediction):
    return FrozenPredictor(callable_, provenance()['prediction'],
                           dict(adapter='cmp_causal_window_v1', batch_size=1))


def request(tool='P', request_id='q0', max_targets=4):
    return task_request(tool=tool, request_id=request_id, max_targets=max_targets,
        max_request_bytes=30000, max_response_bytes=30000,
        max_episode_bytes=100000, max_plan_acceleration_mps2=100.)


class StructuredModuleContractTests(unittest.TestCase):
    def test_module_and_opt_in_history_api_exist(self):
        module = ROOT / 'src/planning/structured_inputs.py'
        self.assertTrue(module.is_file(), 'structured numeric input module is missing')
        from planning.evidence import new_ledger
        self.assertIn('include_local_history', inspect.signature(new_ledger).parameters)

    def test_spec_is_strict_versioned_and_defaults_to_p_state(self):
        from planning.structured_inputs import StructuredDriverSpec
        spec = StructuredDriverSpec()
        self.assertEqual(spec.p_processing, 'observations_only')
        self.assertEqual(spec.max_entities, 64)
        self.assertEqual(spec.max_forecast_sets_per_entity, 4)
        self.assertEqual(StructuredDriverSpec.from_dict(spec.to_dict()), spec)
        for change in ({'extra': 1}, {'position_scale_m': 0.}, {'version': 'unknown'},
                       {'p_processing': 'implicit'}):
            value = spec.to_dict()
            value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                StructuredDriverSpec.from_dict(value)

    def test_default_ledger_remains_v1_and_history_is_exactly_opt_in(self):
        from planning.evidence import new_ledger
        local = window('ego')
        predictor = predictor_for()
        baseline = new_ledger(local, predictor(local), predictor=predictor,
                              local_provenance=provenance())
        explicit = new_ledger(local, predictor(local), predictor=predictor,
                              local_provenance=provenance(), include_local_history=False)
        self.assertEqual(explicit, baseline)
        self.assertEqual(baseline['version'], 'toolv2x_evidence_ledger_v1')
        self.assertEqual({r['ref']['field_kind'] for r in baseline['local_fields']},
                         {'anchor', 'forecast'})

        enriched = new_ledger(local, predictor(local), predictor=predictor,
                              local_provenance=provenance(), include_local_history=True)
        self.assertEqual(enriched['version'], 'toolv2x_evidence_ledger_v2')
        histories = [r for r in enriched['local_fields']
                     if r['ref']['field_kind'] == 'history']
        self.assertEqual(len(histories), len(local['track_ids']))
        for row, record in enumerate(histories):
            value = record['value']
            self.assertEqual(record['ref']['producer_version'], provenance()['tracking'])
            self.assertEqual(value['history'], local['states'][row].tolist())
            self.assertEqual(value['history_valid'], local['valid'][row].tolist())
            self.assertEqual(value['history_scores'], local['scores'][row].tolist())
            self.assertEqual(value['history_times'], local['time_seconds'].tolist())
            self.assertEqual(value['proxy_status'], 'causal_tracking_state_motion_proxy')
        json.dumps(enriched, allow_nan=False)

    def test_ego_history_reads_only_past_within_recording_and_masks_missing(self):
        from planning.inputs import frame_location
        from planning.structured_inputs import load_ego_history
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, start = frame_location(root, 'test', 2)
            self.assertEqual(start, 0)
            pose_dir = base / 'ego'
            pose_dir.mkdir(parents=True)
            current = np.eye(4)
            current[0, 3] = 10.
            past = np.eye(4)
            past[0, 3] = 8.
            np.save(pose_dir / '0002_lidar_pose.npy', current)
            np.save(pose_dir / '0000_lidar_pose.npy', past)
            result = load_ego_history(root, 'test', 2)
        self.assertEqual(result['times'], (np.arange(-10, 1) / 10.).tolist())
        self.assertEqual(result['valid'], [False] * 8 + [True, False, True])
        self.assertEqual(result['states'][8], [-2., 0., 0.])
        self.assertEqual(result['states'][10], [0., 0., 0.])
        self.assertEqual(len(result['read_paths']), 3)

    def test_ego_history_masks_an_invalid_current_pose(self):
        from planning.inputs import frame_location
        from planning.structured_inputs import load_ego_history
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = frame_location(root, 'test', 0)
            pose_dir = base / 'ego'
            pose_dir.mkdir(parents=True)
            singular = np.eye(4)
            singular[:3, :3] = 0.
            np.save(pose_dir / '0000_lidar_pose.npy', singular)
            result = load_ego_history(root, 'test', 0)
        self.assertEqual(result['valid'], [False] * 11)
        self.assertEqual(result['states'], [[0., 0., 0.]] * 11)


class StructuredReceiverTests(unittest.TestCase):
    def ledger(self, local, *, p_processing='observations_only', history=True,
               predictor=None):
        from planning.evidence import new_ledger
        predictor = predictor or predictor_for()
        prediction = fixture_prediction(local)
        return new_ledger(local, prediction, predictor=predictor,
                          local_provenance=provenance(), p_processing=p_processing,
                          include_local_history=history), predictor

    def acquire(self, ledger, predictor, remote, *, tool='P', request_id='q0',
                max_targets=4, service_predictor=None):
        from planning.evidence import apply_response
        service = VehicleTools(lambda: remote, service_predictor or predictor,
                               remote['scene'], remote['g'], remote['source'],
                               task_provenance=provenance())
        response = service.query_task(request(tool, request_id, max_targets))
        return apply_response(ledger, response, predictor), service

    def build(self, ledger, **kw):
        from planning.structured_inputs import StructuredDriverSpec, build_structured_plan_input
        spec = kw.pop('spec', StructuredDriverSpec(p_processing=ledger['p_processing']))
        return build_structured_plan_input(dict(speed_mps=4., yaw_rate_rps=0.),
                                           ledger, spec, **kw)

    def test_source_local_track_id_is_not_global_and_near_origin_is_unresolved(self):
        local = one_track('ego', 10., 7)
        ledger, predictor = self.ledger(local)
        ledger, _ = self.acquire(ledger, predictor, one_track('peer', 30., 7))
        prepared = self.build(ledger)
        live = [e for e in prepared['entities'] if e['tensor_index'] is not None]
        self.assertEqual(len(live), 2)
        self.assertEqual({tuple((a['source'], a['track_handle']) for a in e['aliases'])
                          for e in live}, {(('ego', 7),), (('peer', 7),)})

        close, _ = self.ledger(one_track('ego', .1, 3))
        close_prepared = self.build(close)
        self.assertEqual(close_prepared['entities'][0]['role'], 'unresolved')
        self.assertTrue(close_prepared['tensor_inputs']['entity_mask'][0])

    def test_empty_local_detections_retain_paid_remote_entity(self):
        ledger, predictor = self.ledger(empty_window())
        ledger, _ = self.acquire(ledger, predictor, one_track('peer', 30., 8))
        prepared = self.build(ledger)
        self.assertEqual(len(prepared['entities']), 1)
        entity = prepared['entities'][0]
        self.assertEqual(entity['aliases'], [dict(source='peer', track_handle=8)])
        self.assertEqual(entity['observations'][0]['source_role'], 'remote')
        self.assertEqual(entity['tensor_index'], 0)
        self.assertFalse(any(prepared['tensor_inputs']['observation_mask'][0][0]))
        self.assertTrue(all(prepared['tensor_inputs']['observation_mask'][0][1]))

    def test_fully_empty_scene_builds_all_masked_structured_input(self):
        ledger, _ = self.ledger(empty_window())
        prepared = self.build(ledger)
        self.assertEqual(prepared['entities'], [])
        self.assertFalse(any(prepared['tensor_inputs']['entity_mask']))
        self.assertFalse(any(value for entity in prepared['tensor_inputs']['observation_mask']
                             for source in entity for value in source))
        self.assertFalse(any(value for entity in prepared['tensor_inputs']['forecast_mask']
                             for forecast in entity for mode in forecast for value in mode))

    def test_gated_match_uses_common_history_and_representative_local_anchor(self):
        local = one_track('ego', 10., 7, score=.8)
        remote = one_track('peer', 10.2, 99, score=.99)
        ledger, predictor = self.ledger(local)
        ledger, _ = self.acquire(ledger, predictor, remote)
        prepared = self.build(ledger)
        entity = next(e for e in prepared['entities'] if e['tensor_index'] is not None)
        self.assertEqual([(a['source'], a['track_handle']) for a in entity['aliases']],
                         [('ego', 7), ('peer', 99)])
        self.assertEqual(entity['association']['status'], 'matched')
        self.assertEqual(entity['representative_anchor']['source'], 'ego')
        self.assertAlmostEqual(entity['representative_anchor']['box'][0], 10.)
        self.assertEqual(sum(prepared['tensor_inputs']['entity_mask']), 1)

    def test_equal_competing_matches_stay_separate_and_row_order_is_irrelevant(self):
        local = one_track('ego', 10., 7)
        remote = window('peer')
        remote['track_ids'] = np.array([8, 9])
        remote['states'][:, :, 0] = [[9.9] * 11, [10.1] * 11]
        remote_reversed = copy.deepcopy(remote)
        for key in ('track_ids', 'states', 'valid', 'scores'):
            remote_reversed[key] = remote_reversed[key][::-1].copy()

        outputs = []
        for value in (remote, remote_reversed):
            ledger, predictor = self.ledger(local)
            ledger, _ = self.acquire(ledger, predictor, value)
            outputs.append(self.build(ledger))
        self.assertEqual(outputs[0]['tensor_inputs'], outputs[1]['tensor_inputs'])
        self.assertEqual(len([e for e in outputs[0]['entities']
                              if e['tensor_index'] is not None]), 3)
        self.assertTrue(all(e['association']['status'] == 'ambiguous'
                            for e in outputs[0]['entities']))

    def test_observations_only_p_never_calls_predictor_and_absent_values_are_masked(self):
        calls = []
        def forbidden(value):
            calls.append(value)
            raise AssertionError('P-state must not run MTR')
        predictor = predictor_for(forbidden)
        local = one_track('ego', 10.)
        ledger, _ = self.ledger(local, predictor=predictor)
        remote = one_track('peer', 30., 8)
        ledger, _ = self.acquire(ledger, predictor, remote,
                                 service_predictor=forbidden)
        prepared = self.build(ledger)
        self.assertEqual(calls, [])
        remote_index = next(e['tensor_index'] for e in prepared['entities']
                            if e['aliases'][0]['source'] == 'peer')
        tensors = prepared['tensor_inputs']
        self.assertEqual(tensors['observation_sources'][remote_index], [0, 1])
        self.assertFalse(any(tensors['observation_mask'][remote_index][0]))
        self.assertTrue(all(tensors['observation_mask'][remote_index][1]))
        self.assertFalse(any(v for item in tensors['forecast_mask'][remote_index]
                             for mode in item for v in mode))
        self.assertTrue(all(x == 0. for item in tensors['forecasts'][remote_index]
                            for mode in item for step in mode for x in step))

    def test_full_p_local_and_equivalent_f_have_identical_forecast_encoding(self):
        local = one_track('ego', 10.)
        ledger, predictor = self.ledger(local, p_processing='local_mtr')
        remote = one_track('peer', 30., 8)
        service = VehicleTools(lambda: remote, predictor, 'scene', 10, 'peer',
                               task_provenance=provenance())
        from planning.evidence import apply_response, known_field_manifest
        first = apply_response(ledger, service.query_task(request('P')), predictor)
        req = request('F', 'q1')
        req['acquired_field_manifest'] = known_field_manifest(first)
        second = apply_response(first, service.query_task(req), predictor)
        a, b = self.build(first), self.build(second)
        self.assertEqual(a['tensor_inputs']['forecasts'], b['tensor_inputs']['forecasts'])
        self.assertEqual(a['tensor_inputs']['forecast_mask'], b['tensor_inputs']['forecast_mask'])
        self.assertEqual(a['tensor_inputs']['forecast_context'], b['tensor_inputs']['forecast_context'])

    def test_paid_receipts_and_previous_parent_closure_are_strict(self):
        from planning.structured_inputs import validate_structured_prepared
        local = one_track('ego', 10.)
        ledger, predictor = self.ledger(local)
        ledger, _ = self.acquire(ledger, predictor, one_track('peer', 30., 8))
        parent = ledger['acquired_fields'][0]['ref']
        prepared = self.build(ledger, previous_plan=[[float(i), 0.] for i in range(6)],
                              previous_parent_refs=[parent])
        self.assertEqual(set(prepared['admission_report']), {
            'acquired_field_refs', 'derived_field_refs', 'admitted_field_refs',
            'dropped_field_refs', 'dropped', 'field_groups', 'local_field_groups',
            'observation_receipt_ids'})
        self.assertIn(parent, prepared['admission_report']['admitted_field_refs'])
        validate_structured_prepared(prepared)

        bad = copy.deepcopy(ledger)
        bad['acquired_fields'][0]['value']['box'][0] += 1.
        with self.assertRaises(ValueError):
            self.build(bad)
        unknown = copy.deepcopy(parent)
        unknown['track_handle'] = 999
        with self.assertRaises(ValueError):
            self.build(ledger, previous_plan=[[0., 0.]] * 6,
                       previous_parent_refs=[unknown])
        with self.assertRaises(ValueError):
            self.build(ledger, previous_parent_refs=[parent])

    def test_correlated_ego_history_excludes_only_confirmed_ego_from_tensor(self):
        local = one_track('ego', .1, 3)
        ledger, _ = self.ledger(local)
        ego_history = dict(states=[[0., 0., 0.]] * 11, valid=[True] * 11,
                           times=(np.arange(-10, 1) / 10.).tolist(), read_paths=[])
        prepared = self.build(ledger, ego_history=ego_history)
        self.assertEqual(prepared['entities'][0]['role'], 'ego')
        self.assertIsNone(prepared['entities'][0]['tensor_index'])
        self.assertFalse(any(prepared['tensor_inputs']['entity_mask']))
        self.assertTrue(prepared['ego_history_used'])

    def test_opposite_ego_heading_remains_unresolved_and_in_tensor(self):
        local = one_track('ego', .1, 3)
        local['states'][:, :, 6] = np.pi
        ledger, _ = self.ledger(local)
        ego_history = dict(states=[[0., 0., 0.]] * 11, valid=[True] * 11,
                           times=(np.arange(-10, 1) / 10.).tolist(), read_paths=[])
        prepared = self.build(ledger, ego_history=ego_history)
        self.assertEqual(prepared['entities'][0]['role'], 'unresolved')
        self.assertEqual(prepared['entities'][0]['tensor_index'], 0)
        self.assertTrue(prepared['tensor_inputs']['entity_mask'][0])

    def matched_prepared(self):
        ledger, predictor = self.ledger(one_track('ego', 10., 7, score=.8))
        ledger, _ = self.acquire(ledger, predictor, one_track('peer', 10.2, 99, score=.99))
        return self.build(ledger)

    def test_validator_recomputes_association_representative_and_role(self):
        from planning.structured_inputs import validate_structured_prepared
        prepared = self.matched_prepared()
        corruptions = []
        bad = copy.deepcopy(prepared)
        bad['entities'][0]['association']['status'] = 'unmatched'
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        bad['entities'][0]['association']['metrics']['current_distance_m'] += 1.
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        bad['entities'][0]['representative_anchor']['box'][0] += 1.
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        bad['entities'][0]['role'] = 'unresolved'
        corruptions.append(bad)
        for bad in corruptions:
            with self.subTest(), self.assertRaises(ValueError):
                validate_structured_prepared(bad)

    def test_validator_requires_exact_bounded_tensor_locations(self):
        from planning.structured_inputs import validate_structured_prepared
        prepared = self.matched_prepared()
        corruptions = []
        bad = copy.deepcopy(prepared)
        observation = next(group for groups in ('field_groups', 'local_field_groups')
                           for group in bad['admission_report'][groups]
                           if group['kind'] == 'observation' and group['tensor_locations'])
        observation['tensor_locations'][0]['source_slot'] = 99
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        forecast = next(group for groups in ('field_groups', 'local_field_groups')
                        for group in bad['admission_report'][groups]
                        if group['kind'] == 'forecast' and group['tensor_locations'])
        forecast['tensor_locations'][0]['forecast_set'] = 99
        corruptions.append(bad)
        for bad in corruptions:
            with self.subTest(), self.assertRaises(ValueError):
                validate_structured_prepared(bad)

    def test_validator_rejects_shape_mask_and_nonfinite_corruption(self):
        from planning.structured_inputs import validate_structured_prepared
        ledger, _ = self.ledger(one_track('ego', 10.))
        prepared = self.build(ledger)
        corruptions = []
        bad = copy.deepcopy(prepared)
        bad['tensor_inputs']['entity_mask'].pop()
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        bad['tensor_inputs']['observation_mask'][0][0][0] = 1
        corruptions.append(bad)
        bad = copy.deepcopy(prepared)
        bad['tensor_inputs']['observations'][0][0][0][0] = float('nan')
        corruptions.append(bad)
        for bad in corruptions:
            with self.subTest(), self.assertRaises(ValueError):
                validate_structured_prepared(bad)


if __name__ == '__main__':
    unittest.main()
