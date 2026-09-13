"""Tool boundaries: lazy remote access, full prediction context and wire validity."""
import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tools.vehicle import VehicleTools, decode_response, history_window, make_evidence


def window(source='peer'):
    states = np.zeros((2, 11, 7), np.float32)
    states[:, :, 0] = [[2.] * 11, [20.] * 11]
    states[:, :, 3:6] = [4., 2., 1.5]
    return dict(source=source, scene='scene', g=10, track_ids=np.array([7, 9]), states=states,
                valid=np.ones((2, 11), bool), scores=np.ones((2, 11), np.float32),
                time_seconds=np.arange(-10, 1) / 10.)


def fixture_prediction(w):
    # Test-only stand-in at the expensive model boundary; never used by a runner.
    means = np.repeat(w['states'][:, None, -1:, :2], 6, axis=1)
    means = np.repeat(means, 50, axis=2)
    means[:, :, :, 1] = len(w['track_ids'])
    return dict(track_ids=w['track_ids'], states=w['states'][:, -1], means=means,
                scores=np.full((len(means), 6), 1 / 6, np.float32), model_used=np.ones(len(means), bool))


class VehicleToolTests(unittest.TestCase):
    def test_no_remote_read_before_query_and_roi_does_not_remove_model_context(self):
        reads, model_contexts = [], []
        def load():
            reads.append('peer')
            return window()
        def predict(w):
            model_contexts.append(len(w['track_ids']))
            return fixture_prediction(w)
        tools = VehicleTools(load, predict, 'scene', 10, provider='peer')
        self.assertEqual(reads, [])
        response = tools.query('F', [0., -5., 5., 5.])
        packet = decode_response(response['wire'], 'scene', 10)
        self.assertEqual(model_contexts, [2])
        self.assertEqual(len(packet['objects']), 1)
        self.assertEqual(packet['objects'][0]['track_id'], 7)
        self.assertEqual(packet['objects'][0]['forecast'][0][0][1], 2.)
        self.assertEqual(response['cost']['response_bytes'], len(response['wire']))
        self.assertEqual(packet['forecast_times'], [.5, 1., 1.5, 2., 2.5, 3.])

    def test_full_p_history_round_trips_for_same_information_control(self):
        w = window()
        tools = VehicleTools(lambda: w, fixture_prediction, 'scene', 10, provider='peer')
        response = tools.query('P')
        p = decode_response(response['wire'], 'scene', 10)
        reconstructed = history_window(p)
        for key in ('states', 'valid', 'scores', 'track_ids', 'time_seconds'):
            np.testing.assert_allclose(reconstructed[key], w[key])
        f = decode_response(tools.query('F')['wire'], 'scene', 10)
        e = make_evidence(window('ego'), fixture_prediction(window('ego')), [p, f], fixture_prediction)
        self.assertEqual(len(e['objects']), 6)
        self.assertEqual({o['source'] for o in e['objects']}, {'ego', 'peer:P_local', 'peer:F'})
        self.assertTrue(all(len(o['forecast']) == 6 for o in e['objects']))

    def test_receiver_rejects_future_mismatched_and_extra_payload_fields(self):
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        with self.assertRaises(ValueError):
            decode_response(response['wire'], 'scene', 9)
        p = json.loads(response['wire'])
        p['objects'][0]['gt_future'] = [99.]
        with self.assertRaises(ValueError):
            decode_response(json.dumps(p).encode(), 'scene', 10)
        p = json.loads(response['wire'])
        p['objects'][0]['history_times'][-1] = .1
        with self.assertRaises(ValueError):
            decode_response(json.dumps(p).encode(), 'scene', 10)

    def test_provider_checks_query_and_keeps_empty_distinct_from_observed_free_space(self):
        tools = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer')
        with self.assertRaises(ValueError):
            tools.query('P', [5, 0, 0, 1])
        packet = decode_response(tools.query('P', [50, 50, 60, 60])['wire'], 'scene', 10)
        self.assertEqual(packet['status'], 'no_observed_targets')
        self.assertEqual(packet['coverage'], 'not_established')

    def test_evidence_rejects_different_record_even_with_same_frame_number(self):
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        packet = json.loads(response['wire'])
        packet['scene'] = 'different_recording'
        with self.assertRaises(ValueError):
            make_evidence(window('ego'), fixture_prediction(window('ego')), [packet], fixture_prediction)

    def test_provider_never_caches_a_window_from_another_record_or_time(self):
        for field, value in (('scene', 'different_recording'), ('g', 11)):
            bad = dict(window(), **{field: value})
            tools = VehicleTools(lambda: bad, fixture_prediction, 'scene', 10, provider='peer')
            for _ in range(2):
                with self.assertRaises(ValueError):
                    tools.query('F')

    def test_cost_separates_new_model_fallback_and_roi_returned_targets(self):
        def predict(w):
            result = fixture_prediction(w)
            result['model_used'] = np.array([True, False])
            return result

        tools = VehicleTools(lambda: window(), predict, 'scene', 10, provider='peer')
        # Target counts must not depend on the timer resolving a short operation.
        with patch('tools.vehicle.perf_counter', return_value=5.):
            first = tools.query('F', [0., -5., 5., 5.])
            cached = tools.query('F')
            perception = tools.query('P')
        self.assertEqual(first['cost']['model_targets_computed'], 1)
        self.assertEqual(first['cost']['fallback_targets_computed'], 1)
        self.assertEqual(first['cost']['returned_targets'], 1)
        self.assertFalse(first['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(cached['cost']['model_targets_computed'], 0)
        self.assertEqual(cached['cost']['fallback_targets_computed'], 0)
        self.assertEqual(cached['cost']['returned_targets'], 2)
        self.assertTrue(cached['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(perception['cost']['model_targets_computed'], 0)
        self.assertEqual(perception['cost']['fallback_targets_computed'], 0)
        self.assertFalse(perception['cost']['within_decision_forecast_cache_hit'])


class TaskVehicleTests(unittest.TestCase):
    """Real service/codec, with a context-sensitive fake only at the MTR boundary."""
    def setUp(self):
        self.assertTrue(hasattr(VehicleTools, 'query_task'), 'T2 task service is missing')
        from tools.vehicle import decode_task_response
        from test_task_spec import task_request, provenance
        self.decode = decode_task_response
        self.request = lambda **kw: task_request(max_request_bytes=16384,
            max_response_bytes=20000, max_episode_bytes=75000,
            max_plan_acceleration_mps2=100., **kw)
        self.provenance = provenance()

    def service(self, w=None, predictor=fixture_prediction):
        w = window() if w is None else w
        return VehicleTools(lambda: w, predictor, 'scene', 10, provider=w['source'],
                            task_provenance=self.provenance)

    def packet(self, service, request):
        result = service.query_task(request)
        return self.decode(result['wire'], request), result

    def manifest(self, packet):
        return [dict(receipt_id=packet['receipt_id'], ref=copy.deepcopy(r['ref']))
                for r in packet['records']]

    def test_p_task_changes_real_selection_and_never_runs_mtr(self):
        def forbidden(w):
            self.fail('P ran a predictor')
        selected = []
        for x in (2., 20.):
            req = self.request(x=x, max_targets=1)
            packet, result = self.packet(self.service(predictor=forbidden), req)
            selected.append(packet['ranking'][0]['track_handle'])
            self.assertEqual({r['ref']['field_kind'] for r in packet['records']}, {'anchor', 'history'})
            self.assertEqual(result['cost']['model_targets_computed'], 0)
            self.assertEqual(packet['request']['execution_spec'], req['execution_spec'])
        self.assertEqual(selected, [7, 9])

    def test_change_is_executed_not_just_a_request_label(self):
        selected = []
        for mode in ('current', 'change'):
            req = self.request(x=20., mode=mode, max_targets=1)
            if mode == 'change':
                req['tau_old'] = [[2., 0.]] * 6
            packet, _ = self.packet(self.service(), req)
            selected.append(packet['ranking'][0]['track_handle'])
        self.assertEqual(selected, [9, 7])

    def test_second_p_only_returns_unacknowledged_fields(self):
        service = self.service()
        first, _ = self.packet(service, self.request(max_targets=1))
        req = self.request(request_id='q1', mode='change', x=20., max_targets=1)
        req['acquired_field_manifest'] = self.manifest(first)
        second, result = self.packet(service, req)
        self.assertEqual({r['ref']['track_handle'] for r in first['records']}, {7})
        self.assertEqual({r['ref']['track_handle'] for r in second['records']}, {9})
        self.assertGreater(result['cost']['request_bytes'], 0)

    def test_f_then_p_references_anchor_and_adds_original_history(self):
        service = self.service()
        first, _ = self.packet(service, self.request(tool='F'))
        req = self.request(request_id='q1')
        req['acquired_field_manifest'] = self.manifest(first)
        second, _ = self.packet(service, req)
        self.assertEqual({r['ref']['field_kind'] for r in second['records']}, {'history'})
        self.assertEqual({r['ref']['field_kind'] for r in second['references']}, {'anchor'})

    def test_full_p_does_not_claim_receiver_derived_f_is_already_owned(self):
        contexts = []
        def predictor(w):
            contexts.append(len(w['track_ids']))
            return fixture_prediction(w)
        service = self.service(predictor=predictor)
        first, _ = self.packet(service, self.request())
        req = self.request(tool='F', request_id='q1')
        req['acquired_field_manifest'] = self.manifest(first)
        second, result = self.packet(service, req)
        self.assertEqual(contexts, [2])
        self.assertEqual({r['ref']['field_kind'] for r in second['records']}, {'forecast'})
        self.assertEqual(result['cost']['model_targets_computed'], 2)

    def test_f_keeps_complete_context_before_selection_and_accounts_cache(self):
        contexts = []
        def predictor(w):
            contexts.append(w['track_ids'].tolist())
            return fixture_prediction(w)
        service = self.service(predictor=predictor)
        req = self.request(tool='F', max_targets=1)
        first, result = self.packet(service, req)
        prediction = next(r['value'] for r in first['records'] if r['ref']['field_kind'] == 'forecast')
        self.assertEqual(prediction['forecast'][0][0][1], 2.)
        self.assertEqual(result['cost']['model_targets_computed'], 2)
        req = self.request(tool='F', request_id='q1', x=20., max_targets=1)
        req['acquired_field_manifest'] = self.manifest(first)
        _, cached = self.packet(service, req)
        self.assertEqual(contexts, [[7, 9]])
        self.assertTrue(cached['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(cached['cost']['model_targets_computed'], 0)
        self.assertGreater(cached['cost']['response_bytes'], 0)

    def test_forged_unknown_cross_context_and_derived_receipts_are_rejected(self):
        from test_task_spec import field_ref
        reads = []
        service = VehicleTools(lambda: reads.append(True) or window(), fixture_prediction,
            'scene', 10, provider='peer', task_provenance=self.provenance)
        req = self.request()
        req['acquired_field_manifest'] = [dict(receipt_id='invented', ref=field_ref())]
        with self.assertRaises(ValueError):
            service.query_task(req)
        self.assertEqual(reads, [])
        first, _ = self.packet(service, self.request())
        for key, value in [('provider', 'other'), ('scene', 'other'), ('g', 11),
                           ('track_handle', 999), ('field_kind', 'forecast'),
                           ('context_version', dict(name='causal_tracking_window', revision='other'))]:
            req = self.request(request_id='q1')
            req['acquired_field_manifest'] = self.manifest(first)
            req['acquired_field_manifest'][0]['ref'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.query_task(req)
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(service.task_records), 1)

    def test_registry_is_not_mutable_through_response_or_run_record(self):
        service = self.service()
        first, _ = self.packet(service, self.request())
        first['records'][0]['ref']['track_handle'] = 999
        record = service.task_records[0]
        record['response']['records'][0]['ref']['track_handle'] = 999
        req = self.request(request_id='q1')
        req['acquired_field_manifest'] = self.manifest(first)
        with self.assertRaises(ValueError):
            service.query_task(req)

    def test_no_new_fields_is_paid_and_not_free_space(self):
        service = self.service()
        first, _ = self.packet(service, self.request())
        req = self.request(request_id='q1')
        req['acquired_field_manifest'] = self.manifest(first)
        second, result = self.packet(service, req)
        self.assertEqual(second['status'], 'no_new_fields')
        self.assertEqual(second['coverage'], 'not_established')
        self.assertEqual(second['records'], [])
        self.assertGreater(result['cost']['response_bytes'], 0)

    def test_final_utf8_wire_cap_includes_header_and_whole_bundles(self):
        w = window('邻车')
        req = self.request(max_targets=1)
        req['provider'] = '邻车'
        req['execution_spec']['max_response_bytes'] = 9999
        first, result = self.packet(self.service(w), req)
        size = len(result['wire'])
        self.assertGreater(size, len(result['wire'].decode('utf-8')))
        # Same-sized decimal cap values avoid changing header length in the exact-boundary check.
        req['execution_spec']['max_response_bytes'] = size
        for cap, expected_count in [(size, 2), (size - 1, 0)]:
            bounded = copy.deepcopy(req)
            bounded['execution_spec']['max_response_bytes'] = cap
            packet, result = self.packet(self.service(w), bounded)
            self.assertLessEqual(len(result['wire']), cap)
            self.assertEqual(len(packet['records']), expected_count)
            self.assertEqual(result['cost']['response_bytes'], len(result['wire']))
            if not expected_count:
                self.assertEqual(packet['status'], 'budget_empty')
                self.assertTrue(packet['truncated'])

    def test_preflight_caps_and_episode_budget_reject_before_private_read(self):
        for limits in (dict(max_request_bytes=10), dict(max_response_bytes=10),
                       dict(max_episode_bytes=100)):
            reads = []
            service = VehicleTools(lambda: reads.append(True) or window(), fixture_prediction,
                'scene', 10, provider='peer', task_provenance=self.provenance)
            req = self.request()
            req['execution_spec'].update(limits)
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                service.query_task(req)
            self.assertEqual(reads, [])

    def test_two_call_bound_profile_freeze_and_replayed_ids(self):
        service = self.service()
        self.packet(service, self.request())
        with self.assertRaises(ValueError):
            service.query_task(self.request())
        changed = self.request(request_id='q1')
        changed['execution_spec']['sigma_m'] = 2.
        with self.assertRaises(ValueError):
            service.query_task(changed)
        self.packet(service, self.request(request_id='q1'))
        with self.assertRaises(ValueError):
            service.query_task(self.request(request_id='q2'))
        self.assertEqual(len(service.task_records), 2)

    def test_invalid_provider_window_is_not_forecast_and_failure_is_recorded(self):
        w = window()
        w['time_seconds'][-1] = .1
        def forbidden(w):
            self.fail('invalid/future window reached predictor')
        service = self.service(w, forbidden)
        with self.assertRaises(ValueError):
            service.query_task(self.request(tool='F'))
        record = service.task_records[0]
        self.assertEqual(record['status'], 'error')
        self.assertGreater(record['cost']['request_bytes'], 0)
        self.assertFalse(record['cost']['complete'])
        self.assertEqual(record['request']['execution_spec'], self.request()['execution_spec'])

    def test_decoder_rejects_tampered_schema_identity_reference_and_values(self):
        service = self.service()
        req = self.request()
        packet, _ = self.packet(service, req)
        corruptions = []
        bad = copy.deepcopy(packet)
        bad['records'][0]['ref']['scene'] = 'other'
        corruptions.append(bad)
        bad = copy.deepcopy(packet)
        bad['records'][0]['value']['gt_future'] = []
        corruptions.append(bad)
        bad = copy.deepcopy(packet)
        bad['records'][1]['value']['history_valid'][0] = 1
        corruptions.append(bad)
        bad = copy.deepcopy(packet)
        bad['references'] = [dict(receipt_id='invented', ref=bad['records'][0]['ref'])]
        corruptions.append(bad)
        bad = copy.deepcopy(packet)
        bad['request']['execution_spec']['sigma_m'] = 1.
        corruptions.append(bad)
        # Python considers True == 1; echo equality alone is not schema validation.
        narrow_req = self.request(max_targets=1)
        narrow_packet, _ = self.packet(self.service(), narrow_req)
        narrow_packet['request']['execution_spec']['max_targets'] = True
        with self.assertRaises(ValueError):
            self.decode(json.dumps(narrow_packet).encode('utf-8'), narrow_req)
        for bad in corruptions:
            with self.assertRaises(ValueError):
                self.decode(json.dumps(bad, ensure_ascii=False).encode('utf-8'), req)

    def test_predictor_cannot_revise_issued_context_or_pollute_it_on_failure(self):
        def mutating(w):
            w['states'][:, 0, 0] += 100.  # Current anchors remain equal; history does not.
            return fixture_prediction(w)
        service = self.service(predictor=mutating)
        first, _ = self.packet(service, self.request())
        req = self.request(tool='F', request_id='q1')
        req['acquired_field_manifest'] = self.manifest(first)
        with self.assertRaises(ValueError):
            service.query_task(req)

        def failing(w):
            w['states'][:, :, 0] += 100.
            raise RuntimeError('partial computation failed')
        service = self.service(predictor=failing)
        with self.assertRaises(RuntimeError):
            service.query_task(self.request(tool='F'))
        record = service.task_records[0]
        self.assertIsNone(record['cost']['model_targets_computed'])
        self.assertIsNone(record['cost']['fallback_targets_computed'])
        second, _ = self.packet(service, self.request(request_id='q1'))
        anchor = next(r['value']['box'] for r in second['records'] if r['ref']['field_kind'] == 'anchor')
        self.assertEqual(anchor[0], 2.)

    def test_decoder_checks_proxy_labels_and_fallback_values(self):
        req = self.request(tool='F')
        packet, _ = self.packet(self.service(), req)
        for corruption in ('model_used', 'ranking'):
            bad = copy.deepcopy(packet)
            if corruption == 'model_used':
                bad['records'][1]['value']['model_used'] = False
            else:
                bad['ranking'][0]['proxy_status'] = 'stationary_short_history'
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.decode(json.dumps(bad).encode('utf-8'), req)
        req = self.request()
        packet, _ = self.packet(self.service(), req)
        packet['ranking'][0]['proxy_status'] = 'single_state_static_proxy'
        with self.assertRaises(ValueError):
            self.decode(json.dumps(packet).encode('utf-8'), req)

    def test_receipts_exclude_computed_but_unsent_fields_and_other_sessions(self):
        service = self.service()
        first, _ = self.packet(service, self.request(tool='F', max_targets=1))
        req = self.request(tool='F', request_id='q1', max_targets=1)
        req['acquired_field_manifest'] = self.manifest(first)
        with self.assertRaises(ValueError):
            self.service().query_task(req)
        req['acquired_field_manifest'][1]['ref']['track_handle'] = 9
        with self.assertRaises(ValueError):
            service.query_task(req)

    def test_v1_does_not_prewarm_v2_and_unacknowledged_retransmission_keeps_identity(self):
        calls = []
        def predictor(w):
            calls.append(True)
            return fixture_prediction(w)
        service = self.service(predictor=predictor)
        service.query('F')
        first, result = self.packet(service, self.request(tool='F'))
        self.assertEqual(len(calls), 2)
        self.assertFalse(result['cost']['within_decision_forecast_cache_hit'])
        second, _ = self.packet(service, self.request(tool='F', request_id='q1'))
        self.assertEqual(first['records'], second['records'])
        self.assertNotEqual(first['receipt_id'], second['receipt_id'])

    def test_empty_source_and_valid_short_history_fallback_remain_explicit(self):
        w = window()
        for name in ('states', 'valid', 'scores', 'track_ids'):
            w[name] = w[name][:0]
        packet, _ = self.packet(self.service(w), self.request())
        self.assertEqual(packet['status'], 'no_observed_targets')
        self.assertEqual(packet['coverage'], 'not_established')
        w = window()
        w['valid'][0, :-1] = False
        def predictor(value):
            pred = fixture_prediction(value)
            pred['model_used'][0] = False
            pred['means'][0] = value['states'][0, -1, :2]
            return pred
        packet, result = self.packet(self.service(w, predictor), self.request(tool='F'))
        self.assertEqual(result['cost']['fallback_targets_computed'], 1)
        self.assertEqual(packet['ranking'][0]['proxy_status'], 'stationary_short_history')

    def test_remaining_episode_budget_cannot_be_reset_by_next_request(self):
        service = self.service()
        first = self.request()
        first['execution_spec']['max_episode_bytes'] = 25000
        packet, _ = self.packet(service, first)
        second = copy.deepcopy(first)
        second['request_id'] = 'q1'
        second['acquired_field_manifest'] = self.manifest(packet)
        with self.assertRaisesRegex(ValueError, 'remaining episode budget'):
            service.query_task(second)
        self.assertEqual(len(service.task_records), 1)

    def test_mixed_boolean_request_arrays_rejected_before_private_read(self):
        for field in ('times', 'tau_new', 'tau_old'):
            req = self.request(mode='change' if field == 'tau_old' else 'current')
            if field == 'times':
                req[field][1] = True
            else:
                req[field] = [list(point) for point in req[field]]
                req[field][0][1] = False
            reads = []
            service = VehicleTools(lambda: reads.append(1) or window(), fixture_prediction,
                'scene', 10, 'peer', task_provenance=self.provenance)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    service.query_task(req)
                self.assertEqual(reads, [])
                self.assertEqual(service.task_records, [])

    def test_request_requires_json_native_containers_before_private_read(self):
        for field in ('times', 'tau_new', 'tau_old'):
            req = self.request(mode='change' if field == 'tau_old' else 'current')
            req[field] = tuple(req[field]) if field == 'times' else [tuple(p) for p in req[field]]
            reads = []
            service = VehicleTools(lambda: reads.append(1) or window(), fixture_prediction,
                'scene', 10, 'peer', task_provenance=self.provenance)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    service.query_task(req)
                self.assertEqual(reads, [])
                self.assertEqual(service.task_records, [])

    def test_list_predictor_fields_rejected_without_poisoning_cache(self):
        for field in ('track_ids', 'states', 'means', 'scores', 'model_used'):
            calls = []
            def predictor(w):
                calls.append(1)
                result = fixture_prediction(w)
                if len(calls) == 1:
                    result[field] = result[field].tolist()
                return result
            service = self.service(predictor=predictor)
            with self.subTest(field=field):
                # Catch the old packing TypeError too, then assert the contract error.
                with self.assertRaises(Exception) as failure:
                    service.query_task(self.request(tool='F'))
                self.assertIsInstance(failure.exception, ValueError)
                self.assertIsNone(service._task_forecast)
                self.assertEqual(service.task_records[0]['error']['stage'], 'prediction')
                packet, result = self.packet(service, self.request(tool='F', request_id='q1'))
                self.assertEqual(packet['status'], 'ok')
                self.assertEqual(len(calls), 2)
                self.assertFalse(result['cost']['within_decision_forecast_cache_hit'])

    def test_decoder_rejects_boolean_numeric_payloads(self):
        for tool in ('P', 'F'):
            req = self.request(tool=tool)
            packet, _ = self.packet(self.service(), req)
            fields = [('anchor', 'box'), ('history', 'history'), ('history', 'history_scores'),
                      ('history', 'history_times')] if tool == 'P' else [('forecast', 'forecast'),
                                                                        ('forecast', 'forecast_scores')]
            for kind, field in fields:
                bad = copy.deepcopy(packet)
                value = next(r['value'][field] for r in bad['records'] if r['ref']['field_kind'] == kind)
                if field == 'history':
                    value[0][1] = False
                elif field == 'forecast':
                    value[0][0][1] = False
                else:
                    value[-1] = False
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.decode(json.dumps(bad).encode('utf-8'), req)

    def test_valid_history_dimensions_positive_missing_placeholders_allowed(self):
        for dimension in (3, 4, 5):
            for size in (0., -4.):
                w = window()
                w['states'][0, 0, dimension] = size
                with self.subTest(dimension=dimension, size=size), self.assertRaises(ValueError):
                    self.service(w).query_task(self.request())
        w = window()
        w['valid'][0, 0] = False
        w['states'][0, 0] = 0.
        w['scores'][0, 0] = 0.
        req = self.request()
        packet, _ = self.packet(self.service(w), req)
        history = next(r['value'] for r in packet['records'] if r['ref']['field_kind'] == 'history')
        self.assertFalse(history['history_valid'][0])
        self.assertEqual(history['history'][0], [0.] * 7)

    def test_decoder_rejects_nonpositive_valid_history_dimensions(self):
        req = self.request()
        packet, _ = self.packet(self.service(), req)
        for dimension in (3, 4, 5):
            bad = copy.deepcopy(packet)
            history = next(r['value'] for r in bad['records'] if r['ref']['field_kind'] == 'history')
            history['history'][0][dimension] = -4.
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                self.decode(json.dumps(bad).encode('utf-8'), req)

    def test_budget_empty_response_preserves_valid_prediction_cache(self):
        calls = []
        def predictor(w):
            calls.append(1)
            return fixture_prediction(w)
        service = self.service(predictor=predictor)
        for i in range(2):
            req = self.request(tool='F', request_id='q' + str(i))
            req['execution_spec']['max_response_bytes'] = 1600
            packet, result = self.packet(service, req)
            self.assertEqual(packet['status'], 'budget_empty')
            self.assertEqual(packet['records'], [])
            self.assertEqual(result['cost']['within_decision_forecast_cache_hit'], i == 1)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
