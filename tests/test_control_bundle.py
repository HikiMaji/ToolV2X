"""T6 controls execute real provider primitives; only prediction is injected."""
import copy
import json
import unittest

import numpy as np

from probe.kinematic_tools import encode
from tools.task_spec import rank_targets, validate_task_request, FrozenPredictor
from tools.vehicle import VehicleTools, decode_task_response
from planning.evidence import new_ledger, apply_response
from test_task_spec import task_request, provenance
from test_vehicle_tools import window, fixture_prediction


def envelope(max_calls=2, **spec_changes):
    request = task_request(**dict(dict(max_request_bytes=12000, max_response_bytes=20000,
                                      max_episode_bytes=40000), **spec_changes))
    return dict(version='toolv2x_bundle_v1', request_id='b0', bundle_id='b0',
        first_request=request, public_summary=dict(ego_motion=dict(speed_mps=4.)),
        candidates=[dict(candidate_id='tau0', waypoints=request['tau_new']),
                    dict(candidate_id='slow', waypoints=[[0., 0.]] * 6)],
        continuation_policy_id='conditional_fixture_v1',
        limits=dict(execution_spec=request['execution_spec'], max_calls=max_calls,
                    max_candidates=2, remaining_bytes=40000,
                    wrapper_reserve_bytes=2048,
                    primitive_response_caps=[8000] * max_calls))


def stop(_):
    return dict(tool='STOP', mode=None, candidate_id=None, reason='returned_evidence_suffices')


class ControlRankingTests(unittest.TestCase):
    def test_old_union_have_distinct_ranking_and_v2_rejects_control_modes(self):
        for tool in ('P', 'F'):
            req = task_request(tool=tool, x=20., max_plan_acceleration_mps2=100.)
            req.update(version='toolv2x_control_task_v1', tau_old=[[2., 0.]] * 6)
            w = window()
            pred = fixture_prediction(w) if tool == 'F' else None
            req['mode'] = 'old'
            try:
                old = rank_targets(w, req, pred)
            except ValueError as exc:
                self.fail('controls-only old ranking is missing: ' + str(exc))
            req['mode'] = 'union'
            union = rank_targets(w, req, pred)
            self.assertEqual([r['track_handle'] for r in old], [7, 9])
            self.assertEqual([r['score'] for r in union], [1., 1.])
            self.assertLess(old[1]['score'], 1.)
            req['version'] = 'toolv2x_task_v2'
            with self.assertRaises(ValueError):
                validate_task_request(req, {})

    def test_union_keeps_same_future_time_and_mode_coordinates(self):
        w = window()
        pred = fixture_prediction(w)
        # At .5 s targets are at 20, at every later time at 0. Ego does the
        # reverse; spatial union would falsely claim coincident occupied points.
        pred['means'][:, :, :, :] = [0., 0.]
        pred['means'][:, :, 4, :] = [20., 0.]
        req = task_request(tool='F', max_plan_acceleration_mps2=500.)
        req.update(version='toolv2x_control_task_v1', mode='union',
            tau_new=[[0., 0.]] + [[20., 0.]] * 5,
            tau_old=[[0., 1.]] + [[20., 1.]] * 5)
        try:
            scores = rank_targets(w, req, pred)
        except ValueError as exc:
            self.fail('controls-only aligned union is missing: ' + str(exc))
        self.assertTrue(all(0. < r['score'] < .02 for r in scores))

    def test_roi_uses_full_f_context_but_returns_only_current_roi_targets(self):
        req = task_request(tool='F')
        req.update(version='toolv2x_control_task_v1', mode='roi', roi=[0., -5., 5., 5.])
        service = VehicleTools(window, fixture_prediction, 'scene', 10, 'peer', task_provenance=provenance())
        try:
            response = service.query_task(req)
        except ValueError as exc:
            self.fail('controls-only ROI request is missing: ' + str(exc))
        packet = decode_task_response(response['wire'], req)
        self.assertEqual([r['track_handle'] for r in packet['ranking']], [7])
        self.assertEqual(response['cost']['model_targets_computed'], 2)
        self.assertEqual(response['cost']['fallback_targets_computed'], 0)
        outside = copy.deepcopy(packet)
        anchor = next(r for r in outside['records'] if r['ref']['field_kind'] == 'anchor')
        anchor['value']['box'][0] = 30.
        with self.assertRaises(ValueError):
            decode_task_response(encode(outside), req)
        req['tool'] = 'P'
        req['request_id'] = 'q1'
        packet = decode_task_response(service.query_task(req)['wire'], req)
        self.assertNotIn('context_certificate', packet)


class ControlBundleTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(VehicleTools, 'query_bundle'), 'bounded query_bundle is missing')
        self.prediction_windows = []
        def predict(w):
            self.prediction_windows.append(copy.deepcopy(w))
            return fixture_prediction(w)
        self.predictor = FrozenPredictor(predict, provenance()['prediction'], {})
        self.service = VehicleTools(window, self.predictor, 'scene', 10, 'peer', task_provenance=provenance())

    def register(self, policy):
        self.service.register_bundle_policy('conditional_fixture_v1', policy)

    def test_first_real_response_selects_stop_or_second_f_without_free_prediction(self):
        seen = []
        def conditional(state):
            self.assertEqual(set(state), {'envelope', 'first_response', 'available_actions'})
            self.assertEqual(self.prediction_windows, [])
            seen.append(copy.deepcopy(state))
            if state['first_response']['records']:
                return dict(tool='F', mode='change', candidate_id='slow', reason='history_received')
            return stop(state)
        self.register(conditional)
        sent = envelope()
        result = self.service.query_bundle(sent)
        self.assertEqual(result['cost']['rpc_rounds'], 1)
        self.assertEqual(result['cost']['capability_calls'], 2)
        self.assertEqual(len(self.prediction_windows), 1)
        self.assertEqual(len(self.prediction_windows[0]['track_ids']), 2)
        first, second = result['primitive_responses']
        first_packet = decode_task_response(first['wire'], first['request'])
        second_packet = decode_task_response(second['wire'], second['request'])
        self.assertEqual(seen[0]['first_response'], first_packet)
        self.assertEqual(second['request']['tau_new'], [[0., 0.]] * 6)
        self.assertTrue(second_packet['references'])
        self.assertEqual({r['ref']['field_kind'] for r in second_packet['records']}, {'forecast'})
        self.assertEqual(result['cost']['request_bytes'], len(encode(sent)))
        self.assertEqual(result['request_wire'], encode(sent))
        self.assertEqual(result['cost']['response_bytes'], len(result['wire']))
        self.assertLessEqual(len(result['wire']), sent['limits']['execution_spec']['max_response_bytes'])
        self.assertTrue(all(r['cost']['service_seconds'] >= 0. for r in result['primitive_responses']))
        self.assertGreaterEqual(result['cost']['service_seconds'], result['cost']['model_seconds'])
        with self.assertRaises(ValueError):
            self.service.query_bundle(sent)

        # The same registered policy sees an actual empty causal window.
        empty = window()
        for key in ('track_ids', 'states', 'scores', 'valid'):
            empty[key] = empty[key][:0]
        self.prediction_windows.clear()
        self.service = VehicleTools(lambda: empty, self.predictor, 'scene', 10, 'peer', task_provenance=provenance())
        self.register(conditional)
        result = self.service.query_bundle(envelope())
        self.assertEqual(result['cost']['capability_calls'], 1)
        self.assertEqual(self.prediction_windows, [])

    def test_second_p_and_f_first_use_the_same_executor_and_cache(self):
        self.register(lambda _: dict(tool='P', mode='current', candidate_id='slow', reason='select_history'))
        sent = envelope()
        sent['first_request']['tool'] = 'F'
        result = self.service.query_bundle(sent)
        self.assertEqual([r['request']['tool'] for r in result['primitive_responses']], ['F', 'P'])
        self.assertEqual(len(self.prediction_windows), 1)
        self.assertEqual(result['cost']['capability_calls'], 2)

    def test_receiver_uses_actual_outer_wire_and_provider_issued_inner_receipts(self):
        from tools.control_bundle import decode_bundle_response
        self.register(lambda _: dict(tool='F', mode='current', candidate_id='tau0', reason='get_forecast'))
        sent = envelope()
        result = self.service.query_bundle(sent)
        decoded = decode_bundle_response(result['wire'], sent)
        self.assertEqual(decoded['primitive_responses'], result['primitive_responses'])
        local = window('ego')
        ledger = new_ledger(local, fixture_prediction(local), predictor=self.predictor, local_provenance=provenance())
        for primitive in decoded['primitive_responses']:
            ledger = apply_response(ledger, primitive, self.predictor)
        self.assertEqual(len(ledger['receipts']), 2)
        self.assertEqual(len(ledger['acquired_fields']), 6)
        bad = json.loads(result['wire'])
        bad['responses'][1]['packet']['request']['tau_new'] = [[1., 0.]] * 6
        with self.assertRaises(ValueError):
            decode_bundle_response(encode(bad), sent)
        for cost_change in (dict(returned_targets=99), dict(model_targets_computed=-1),
                            dict(within_decision_forecast_cache_hit='yes')):
            bad = json.loads(result['wire'])
            bad['responses'][0]['cost'].update(cost_change)
            with self.subTest(cost_change=cost_change), self.assertRaises(ValueError):
                decode_bundle_response(encode(bad), sent)

    def test_policy_mutations_cannot_change_sent_candidates_or_budget(self):
        def mutate(visible):
            visible['envelope']['candidates'][1]['waypoints'] = [[99., 0.]] * 6
            visible['envelope']['limits']['max_calls'] = 200
            visible['first_response']['records'].clear()
            return dict(tool='P', mode='current', candidate_id='slow', reason='detached_input')
        self.register(mutate)
        sent = envelope()
        result = self.service.query_bundle(sent)
        self.assertEqual(result['request'], sent)
        self.assertEqual(result['cost']['capability_calls'], 2)
        self.assertEqual(result['primitive_responses'][1]['request']['tau_new'], [[0., 0.]] * 6)
        first = json.loads(result['primitive_responses'][0]['wire'])
        self.assertTrue(first['records'])

    def test_budget_candidates_summary_and_unknown_policy_fail_before_private_access(self):
        self.register(stop)
        for field in ('request_cap', 'response_cap', 'episode_cap', 'candidates', 'unknown_policy', 'receipt', 'features', 'future'):
            sent = envelope()
            if field == 'request_cap':
                sent['public_summary']['ego_motion']['description'] = '车' * 6000
            elif field == 'response_cap':
                sent['limits']['primitive_response_caps'] = [11000, 11000]
            elif field == 'episode_cap':
                sent['limits']['remaining_bytes'] = 500
            elif field == 'candidates':
                sent['limits']['max_candidates'] = 1
            elif field == 'unknown_policy':
                sent['continuation_policy_id'] = 'not_registered'
            elif field == 'receipt':
                from test_task_spec import field_ref
                sent['first_request']['acquired_field_manifest'] = [dict(receipt_id='invented', ref=field_ref())]
            elif field == 'features':
                sent['public_summary']['features'] = [1, 2]
            else:
                sent['public_summary']['gt_future'] = [[1., 2.]]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.service.query_bundle(sent)
        self.assertEqual(self.service.task_records, [])
        self.assertEqual(self.prediction_windows, [])

    def test_condition_cannot_invent_candidate_handle_or_enlarge_frozen_limits(self):
        for action in (dict(tool='F', mode='current', candidate_id='unseen', reason='invented'),
                       dict(tool='F', mode='current', candidate_id='tau0', reason='handle', track_handle=1234)):
            self.service = VehicleTools(window, self.predictor, 'scene', 10, 'peer', task_provenance=provenance())
            self.register(lambda _: action)
            with self.assertRaises(ValueError):
                self.service.query_bundle(envelope())
            record = self.service.bundle_records[-1]
            self.assertEqual(record['cost']['capability_calls'], 1)
            self.assertEqual(record['cost']['rpc_rounds'], 1)
            self.assertFalse(record['cost']['complete'])
            self.assertEqual(record['cost']['response_bytes'], 0)
            self.assertEqual(record['cost']['model_seconds'], 0.)
            self.assertEqual(len(record['primitive_responses']), 1)
            self.assertEqual(self.prediction_windows, [])

    def test_second_prediction_failure_preserves_charged_first_response_and_costs(self):
        def fail_predict(_):
            raise RuntimeError('injected predictor failure')
        self.service = VehicleTools(window, fail_predict, 'scene', 10, 'peer', task_provenance=provenance())
        self.register(lambda _: dict(tool='F', mode='current', candidate_id='tau0', reason='need_forecast'))
        with self.assertRaisesRegex(RuntimeError, 'injected predictor failure'):
            self.service.query_bundle(envelope())
        record = self.service.bundle_records[-1]
        self.assertEqual(record['status'], 'error')
        try:
            encode(record)
        except TypeError as exc:
            self.fail('failed provider record is not durably JSON serializable: ' + str(exc))
        self.assertEqual(record['cost']['capability_calls'], 2)
        self.assertEqual(record['cost']['response_bytes'], 0)
        self.assertIsNone(record['cost']['model_targets_computed'])
        self.assertEqual(len(record['primitive_responses']), 1)
        self.assertEqual(record['primitive_records'][1]['status'], 'error')
        self.assertGreaterEqual(record['cost']['model_seconds'], 0.)

    def test_max_one_call_does_not_run_continuation_and_costs_its_sent_candidates(self):
        def forbidden(_):
            self.fail('continuation ran after frozen max_calls=1')
        self.register(forbidden)
        sent = envelope(max_calls=1)
        result = self.service.query_bundle(sent)
        self.assertEqual(result['cost']['capability_calls'], 1)
        self.assertEqual(result['cost']['request_bytes'], len(encode(sent)))
        self.assertEqual(self.prediction_windows, [])
        saved = self.service.bundle_records[-1]
        self.assertEqual(bytes.fromhex(saved['wire_hex']), result['wire'])
        try:
            encode(saved)
            encode(result['provider_record'])
        except TypeError as exc:
            self.fail('completed provider record is not durably JSON serializable: ' + str(exc))

    def test_actual_outer_wrappers_are_not_free_when_reserve_is_too_small(self):
        self.register(stop)
        sent = envelope(max_calls=1, max_response_bytes=4000)
        sent['limits'].update(primitive_response_caps=[4000], wrapper_reserve_bytes=0)
        with self.assertRaisesRegex(ValueError, 'aggregate bundle response'):
            self.service.query_bundle(sent)
        record = self.service.bundle_records[-1]
        self.assertEqual(record['cost']['capability_calls'], 1)
        self.assertEqual(record['cost']['response_bytes'], 0)
        self.assertIsNone(record['wire_hex'])
        self.assertLess(len(bytes.fromhex(record['primitive_responses'][0]['wire_hex'])), 4000)
        self.assertGreater(record['cost']['service_seconds'], 0.)

    def test_f_then_f_records_paid_cache_reuse_without_extra_prediction(self):
        self.register(lambda _: dict(tool='F', mode='union', candidate_id='slow', reason='select_new_task'))
        sent = envelope()
        sent['first_request']['tool'] = 'F'
        result = self.service.query_bundle(sent)
        self.assertEqual(result['cost']['capability_calls'], 2)
        self.assertEqual(len(self.prediction_windows), 1)
        self.assertTrue(result['primitive_responses'][1]['cost']['within_decision_forecast_cache_hit'])
        self.assertEqual(result['primitive_responses'][1]['cost']['model_targets_computed'], 0)
        self.assertEqual(result['cost']['model_targets_computed'], 2)

    def test_first_private_load_failure_is_a_charged_bundle_attempt(self):
        def failed_load():
            raise OSError('unavailable causal window')
        self.service = VehicleTools(failed_load, self.predictor, 'scene', 10, 'peer', task_provenance=provenance())
        self.register(stop)
        sent = envelope()
        with self.assertRaisesRegex(OSError, 'unavailable causal window'):
            self.service.query_bundle(sent)
        record = self.service.bundle_records[-1]
        self.assertEqual(record['cost']['capability_calls'], 1)
        self.assertEqual(record['cost']['request_bytes'], len(encode(sent)))
        self.assertEqual(record['primitive_records'][0]['error']['stage'], 'window_loading')
        self.assertEqual(record['cost']['model_seconds'], 0.)
        self.assertFalse(record['cost']['complete'])

    def test_field_lengths_use_declared_complete_wire_budgets(self):
        name = '候选' * 40
        self.register(lambda _: dict(tool='P', mode='current', candidate_id=name, reason='r' * 300))
        sent = envelope()
        sent['candidates'][1]['candidate_id'] = name
        try:
            result = self.service.query_bundle(sent)
        except ValueError as exc:
            self.fail('legal fields were limited outside the declared wire budgets: ' + str(exc))
        self.assertEqual(result['cost']['capability_calls'], 2)
        self.assertLessEqual(len(result['wire']), sent['limits']['execution_spec']['max_response_bytes'])


if __name__ == '__main__':
    unittest.main()
