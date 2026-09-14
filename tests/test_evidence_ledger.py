"""T3 contracts only: real service and receiver, synthetic frozen predictor."""
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_vehicle_tools import window, fixture_prediction
from test_task_spec import task_request, provenance
from tools.vehicle import VehicleTools, decode_task_response


def ordered_prediction(w):
    result = fixture_prediction(w)
    result['means'][..., 1] = sum((i + 1) * int(t) for i, t in enumerate(w['track_ids']))
    return result


class LedgerFixture(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'src/planning/evidence.py').exists(), 'T3 ledger missing')
        self.e = importlib.import_module('planning.evidence')
        from tools.task_spec import FrozenPredictor
        self.predictor = FrozenPredictor(ordered_prediction, provenance()['prediction'],
                                        dict(adapter='cmp_causal_window_v1', batch_size=1))
        ego = window('ego')
        self.ledger = self.e.new_ledger(ego, self.predictor(ego), predictor=self.predictor,
                                       local_provenance=provenance())

    def service(self, w=None, predictor=None):
        return VehicleTools(lambda: window() if w is None else w, predictor or self.predictor,
                            'scene', 10, 'peer', task_provenance=provenance())

    def request(self, tool='P', request_id='q0', **kw):
        return task_request(tool=tool, request_id=request_id, max_request_bytes=30000,
            max_response_bytes=30000, max_episode_bytes=100000,
            max_plan_acceleration_mps2=100., **kw)

    def apply(self, result, ledger=None):
        return self.e.apply_response(self.ledger if ledger is None else ledger, result, self.predictor)


class EvidenceLedgerTests(LedgerFixture):
    def test_full_p_certificate_is_paid_and_restores_actual_input_order(self):
        w = window()
        for k in ('track_ids', 'states', 'scores', 'valid'):
            w[k] = w[k][::-1].copy()
        req = self.request()
        result = self.service(w).query_task(req)
        p = decode_task_response(result['wire'], req)
        self.assertEqual(p['context_certificate']['track_order'], [9, 7])
        self.assertEqual(result['cost']['response_bytes'], len(result['wire']))
        ledger = self.apply(result)
        context = ledger['contexts'][-1]
        self.assertEqual(context['window']['track_ids'], [9, 7])
        self.assertEqual(context['array_dtypes']['states'], 'float32')
        self.assertEqual(context['scope'], 'provider_full_at_t')
        json.dumps(ledger, allow_nan=False)
        self.assertEqual(self.ledger['acquired_fields'], [])

    def test_partial_p_never_certifies_unknown_tracks(self):
        req = self.request(max_targets=1)
        result = self.service().query_task(req)
        p = decode_task_response(result['wire'], req)
        self.assertNotIn('context_certificate', p)
        ledger = self.apply(result)
        self.assertEqual(ledger['contexts'][-1]['scope'], 'receiver_acquired_subset')
        self.assertEqual(ledger['contexts'][-1]['window']['track_ids'], [7])

    def test_two_p_accumulate_remote_fields_and_recompute_only_effective_context(self):
        s = self.service()
        first = self.apply(s.query_task(self.request(max_targets=1)))
        req = self.request(request_id='q1', max_targets=1)
        req['acquired_field_manifest'] = self.e.known_field_manifest(first)
        second = self.apply(s.query_task(req), first)
        self.assertEqual(len(first['acquired_fields']), 2)
        self.assertEqual(len(second['acquired_fields']), 4)
        self.assertEqual(len(second['contexts']), 2)
        self.assertEqual(second['contexts'][-1]['scope'], 'provider_full_at_t')
        self.assertEqual(len(second['derived_fields']), 3)
        self.assertEqual(len(self.e.known_field_manifest(second)), 4)
        self.assertTrue(all(x['ref']['field_kind'] != 'forecast'
                            for x in self.e.known_field_manifest(second)))

    def test_retransmission_is_paid_but_not_new_evidence_or_prediction(self):
        s = self.service()
        first = self.apply(s.query_task(self.request()))
        second = self.apply(s.query_task(self.request(request_id='q1')), first)
        self.assertEqual(len(second['acquired_fields']), len(first['acquired_fields']))
        self.assertEqual(len(second['derived_fields']), len(first['derived_fields']))
        self.assertEqual(len(second['receipts']), 2)
        self.assertTrue(second['receiver_events'][-1]['cache_hit'])
        self.assertGreater(second['receipts'][-1]['cost']['response_bytes'], 0)
        again = self.apply({
            'request': second['receipts'][-1]['request'],
            'wire': second['receipts'][-1]['wire_text'].encode(),
            'cost': second['receipts'][-1]['cost']}, second)
        self.assertEqual(again, second)

    def test_f_then_p_resolves_receipt_anchor_and_keeps_derived_separate(self):
        s = self.service()
        first = self.apply(s.query_task(self.request('F')))
        req = self.request(request_id='q1')
        req['acquired_field_manifest'] = self.e.known_field_manifest(first)
        second = self.apply(s.query_task(req), first)
        self.assertEqual(len(second['acquired_fields']), 6)
        self.assertEqual(len(second['derived_fields']), 2)
        self.assertTrue(all(r['origin'] == 'remote' for r in second['acquired_fields']))
        self.assertTrue(all(r['parent_refs'] and r['origin'] == 'receiver_derived'
                            for r in second['derived_fields']))
        units = self.e.remote_units(second)
        forecasts = [u for u in units if u['kind'] == 'forecast']
        self.assertEqual(len(forecasts), 2)
        self.assertTrue(all(len(u['primary_refs']) == 2 for u in forecasts))

    def test_equal_numbers_from_different_contexts_are_not_equivalent(self):
        from tools.task_spec import FrozenPredictor
        def constant(w):
            p = fixture_prediction(w)
            p['means'][..., 1] = 0.
            return p
        pred = FrozenPredictor(constant, provenance()['prediction'], dict(adapter='cmp_causal_window_v1'))
        ego = window('ego')
        ledger = self.e.new_ledger(ego, pred(ego), predictor=pred, local_provenance=provenance())
        s = self.service(predictor=pred)
        ledger = self.e.apply_response(ledger, s.query_task(self.request(max_targets=1)), pred)
        ledger = self.e.apply_response(ledger, s.query_task(self.request('F', 'q1', max_targets=1)), pred)
        self.assertEqual(len([u for u in self.e.remote_units(ledger) if u['kind'] == 'forecast']), 2)

    def test_same_version_different_predictor_binding_does_not_prove_equivalence(self):
        from tools.task_spec import FrozenPredictor
        other = FrozenPredictor(ordered_prediction, provenance()['prediction'],
                                dict(adapter='cmp_causal_window_v1', batch_size=1))
        s = self.service(predictor=other)
        ledger = self.apply(s.query_task(self.request()))
        ledger = self.apply(s.query_task(self.request('F', 'q1')), ledger)
        self.assertEqual(len([u for u in self.e.remote_units(ledger) if u['kind'] == 'forecast']), 4)

    def test_f_needs_its_own_matching_predictor_binding(self):
        from tools.task_spec import FrozenPredictor
        ledger = self.apply(self.service().query_task(self.request()))
        other = FrozenPredictor(ordered_prediction, provenance()['prediction'],
                                dict(adapter='cmp_causal_window_v1', batch_size=1))
        ledger = self.apply(self.service(predictor=other).query_task(self.request('F', 'q1')), ledger)
        self.assertEqual(len([u for u in self.e.remote_units(ledger) if u['kind'] == 'forecast']), 4)

    def test_certificate_omitted_at_cap_then_no_new_fields_can_change_context(self):
        from probe.kinematic_tools import encode
        w = window()
        for key in ('states', 'valid', 'scores', 'track_ids'):
            w[key] = w[key][::-1].copy()
        q = self.request()
        packet = decode_task_response(self.service(w).query_task(q)['wire'], q)
        del packet['context_certificate']
        q['execution_spec']['max_response_bytes'] = len(encode(packet))
        service = self.service(w)
        first_result = service.query_task(q)
        self.assertNotIn('context_certificate', json.loads(first_result['wire']))
        first = self.apply(first_result)
        self.assertEqual(first['contexts'][0]['window']['track_ids'], [7, 9])
        second_req = copy.deepcopy(q)
        second_req['request_id'] = 'q1'
        second_req['acquired_field_manifest'] = self.e.known_field_manifest(first)
        second_result = service.query_task(second_req)
        self.assertEqual(json.loads(second_result['wire'])['status'], 'no_new_fields')
        second = self.apply(second_result, first)
        self.assertEqual(len(second['acquired_fields']), len(first['acquired_fields']))
        self.assertEqual(second['contexts'][-1]['window']['track_ids'], [9, 7])
        self.assertEqual(second['receiver_events'][-1]['model_calls'], 1)
        self.assertLessEqual(len(second_result['wire']), q['execution_spec']['max_response_bytes'])

    def test_f_only_anchors_do_not_expand_local_p_prediction_context(self):
        service = self.service()
        first = self.apply(service.query_task(self.request(max_targets=1)))
        second = self.apply(service.query_task(self.request('F', 'q1', max_targets=1, x=20.)), first)
        self.assertEqual(len(first['contexts']), len(second['contexts']))
        self.assertTrue(second['receiver_events'][-1]['cache_hit'])

    def test_injected_or_changed_receipt_content_is_rejected(self):
        s = self.service()
        first = self.apply(s.query_task(self.request()))
        result = s.query_task(self.request(request_id='q1'))
        p = json.loads(result['wire'])
        next(r for r in p['records'] if r['ref']['field_kind'] == 'history')['value']['history'][0][0] += 1.
        from probe.kinematic_tools import encode
        result['wire'] = encode(p)
        result['cost']['response_bytes'] = len(result['wire'])
        with self.assertRaises(ValueError):
            self.apply(result, first)

    def test_failed_derivation_retains_paid_e_and_no_partial_derived_records(self):
        from tools.task_spec import FrozenPredictor
        def fail(w):
            raise RuntimeError('predictor failed')
        pred = FrozenPredictor(fail, provenance()['prediction'], dict(adapter='cmp_causal_window_v1'))
        ego = window('ego')
        ledger = self.e.new_ledger(ego, ordered_prediction(ego), predictor=pred, local_provenance=provenance())
        result = self.service(predictor=pred).query_task(self.request())
        with self.assertRaises(self.e.EvidenceUpdateError) as failure:
            self.e.apply_response(ledger, result, pred)
        saved = failure.exception.ledger
        self.assertEqual(len(saved['acquired_fields']), 4)
        self.assertEqual(saved['derived_fields'], [])
        self.assertFalse(saved['receiver_events'][-1]['complete'])
        self.assertGreater(saved['receipts'][0]['cost']['response_bytes'], 0)

    def test_invalid_paid_response_preserves_raw_attempt_but_no_legal_e(self):
        result = self.service().query_task(self.request())
        result['wire'] = b'{broken response'
        result['cost']['response_bytes'] = len(result['wire'])
        with self.assertRaises(self.e.EvidenceUpdateError) as failure:
            self.apply(result)
        saved = failure.exception.ledger
        self.assertEqual(saved['acquired_fields'], [])
        self.assertEqual(saved['receipts'], [])
        attempt = saved['failed_receives'][0]
        self.assertEqual(bytes.fromhex(attempt['wire_hex']), result['wire'])
        self.assertEqual(attempt['reported_cost'], result['cost'])
        self.assertGreaterEqual(attempt['receiver_seconds'], 0.)


class TaskReceiverTests(LedgerFixture):
    def build(self, ledger, limits=None):
        import planning.context as context
        self.assertTrue(hasattr(context, 'build_task_plan_input'), 'T3 receiver missing')
        build_task_plan_input = context.build_task_plan_input
        return build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 0,
            limits=limits or dict(context_limit=18000, generation_reserve=256, peer_reserve=5000),
            token_counter=len)

    def test_full_p_and_equivalent_f_have_identical_z_and_prompt(self):
        s = self.service()
        first = self.apply(s.query_task(self.request()))
        second = self.apply(s.query_task(self.request('F', 'q1')), first)
        a, b = self.build(first), self.build(second)
        self.assertEqual(a['q9_prompt'], b['q9_prompt'])
        self.assertEqual(a['remote_evidence_used'], b['remote_evidence_used'])
        self.assertEqual(len(second['acquired_fields']), 6)
        self.assertEqual(len(b['admission_report']['field_groups']), 4)
        self.assertTrue(any(len(g['primary_refs']) == 2 for g in b['admission_report']['field_groups']))

    def test_same_final_information_ignores_request_labels_and_old_derivations(self):
        s = self.service()
        partial = self.apply(s.query_task(self.request(max_targets=1)))
        req = self.request(request_id='q1', max_targets=1)
        req['acquired_field_manifest'] = self.e.known_field_manifest(partial)
        cumulative = self.apply(s.query_task(req), partial)
        direct = self.apply(self.service().query_task(self.request(request_id='unrelated')))
        self.assertEqual(self.build(cumulative)['q9_prompt'], self.build(direct)['q9_prompt'])
        self.assertGreater(len(cumulative['derived_fields']), len(direct['derived_fields']))

    def test_serialized_ledger_rebuilds_the_same_prompt(self):
        ledger = self.apply(self.service().query_task(self.request()))
        restored = json.loads(json.dumps(ledger, sort_keys=True))
        self.assertEqual(self.build(ledger)['q9_prompt'], self.build(restored)['q9_prompt'])

    def test_small_capacity_retains_e_and_reports_exact_dropped_refs(self):
        from planning.context import build_task_plan_input
        s = self.service()
        far = self.apply(s.query_task(self.request(x=20., max_targets=1)))
        req = self.request(request_id='q1', max_targets=1)
        req['acquired_field_manifest'] = self.e.known_field_manifest(far)
        more = self.apply(s.query_task(req), far)
        # A contract-only counter isolates atomic receiver capacity, not LLM cost.
        def count(prompt):
            line = next((l for l in prompt.splitlines() if l.startswith('Additional queried neighbor evidence: ')), None)
            if line is None:
                return 20
            from planning.inputs import unpack_evidence
            remote = unpack_evidence(json.loads(line.split(': ', 1)[1]))
            return 20 + 100 * sum(int('history' in o) + int('forecast' in o) for o in remote['objects'])
        limits = dict(context_limit=180, generation_reserve=50, peer_reserve=100)
        build = lambda l: build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), l, 0,
                                               limits=limits, token_counter=count)
        first, second = build(far), build(more)
        self.assertEqual(first['remote_evidence_used']['objects'][0]['track_id'], 9)
        self.assertEqual(second['remote_evidence_used']['objects'][0]['track_id'], 7)
        self.assertEqual(len(more['acquired_fields']), 4)
        self.assertTrue(any(r['track_handle'] == 9 for r in second['admission_report']['dropped_field_refs']))
        self.assertLessEqual(second['evidence_selection']['input_tokens'] + 50, 180)
        self.assertEqual(first['evidence_used'], second['evidence_used'])

    def test_fabricated_acquired_field_cannot_enter_z(self):
        ledger = self.apply(self.service().query_task(self.request('F')))
        ledger['acquired_fields'][1]['value']['forecast'][0][0][0] += 1.
        with self.assertRaises(ValueError):
            self.build(ledger)

    def test_derived_z_requires_exact_nonempty_context_parent_chain(self):
        ledger = self.apply(self.service().query_task(self.request()))
        for corruption in ('empty', 'partial', 'wrong_context', 'wrong_target'):
            bad = copy.deepcopy(ledger)
            record = bad['derived_fields'][0]
            if corruption == 'empty':
                record['parent_refs'] = []
            elif corruption == 'partial':
                record['parent_refs'] = record['parent_refs'][:1]
            elif corruption == 'wrong_context':
                record['ref']['context_version']['revision'] = 'unknown'
            else:
                record['ref']['track_handle'] = 999
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.build(bad)

    def test_v2_codec_preserves_fields_and_rejects_future_payload(self):
        from planning.inputs import pack_evidence, unpack_evidence
        ledger = self.apply(self.service().query_task(self.request()))
        remote = self.build(ledger)['remote_evidence_used']
        self.assertEqual(unpack_evidence(pack_evidence(remote)), remote)
        remote['objects'][0]['gt_future'] = [[999., 999.]]
        with self.assertRaises(ValueError):
            pack_evidence(remote)

    def test_zero_peer_reserve_is_recorded_for_ego_max_context(self):
        result = self.build(self.ledger, dict(context_limit=18000, generation_reserve=256, peer_reserve=0))
        self.assertEqual(result['evidence_selection']['peer_reserved_tokens'], 0)
        self.assertIsNone(result['remote_evidence_used'])
        self.assertEqual(result['input_layout'], 'source_blocks_v2')
        self.assertFalse(result['language_model_executed'])

    def test_round_robin_receiver_spreads_units_then_keeps_distinct_contexts(self):
        from planning.context import build_task_plan_input
        from planning.inputs import unpack_evidence
        peer = window()
        for name in ('states', 'scores', 'valid'):
            peer[name] = np.concatenate([peer[name], peer[name][:1].copy()])
        peer['track_ids'] = np.array([7, 9, 11])
        peer['states'][2, :, 0] = 100.
        service = self.service(peer)
        ledger = self.apply(service.query_task(self.request(max_targets=2)))
        ledger = self.apply(service.query_task(self.request('F', 'q1', max_targets=2)), ledger)
        original = copy.deepcopy(ledger)
        def count(prompt):
            line = next((s for s in prompt.splitlines() if s.startswith('Additional queried neighbor evidence: ')), None)
            objects = unpack_evidence(json.loads(line.split(': ', 1)[1]))['objects'] if line else []
            return 20 + 100 * sum(len(o['field_metadata']) for o in objects)
        def build(version, capacity=280):
            try:
                return build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 0,
                    limits=dict(version=version, context_limit=capacity, generation_reserve=50, peer_reserve=200),
                    token_counter=count)
            except ValueError as exc:
                self.fail('declared receiver must accept the valid paid ledger: ' + str(exc))
        old, new = build('toolv2x_receiver_v1'), build('toolv2x_receiver_v2')
        refs = lambda result: [g['anchor_ref']['track_handle'] for g in result['admission_report']['field_groups']]
        self.assertEqual(refs(old), [7, 7])
        self.assertEqual(refs(new), [7, 9])
        self.assertEqual(new['evidence_used'], old['evidence_used'])
        self.assertEqual(new['evidence_selection']['input_tokens'], old['evidence_selection']['input_tokens'])
        self.assertEqual(new['admission_report']['acquired_field_refs'], old['admission_report']['acquired_field_refs'])
        full = build('toolv2x_receiver_v2', 1000)
        self.assertEqual(refs(full), [7, 9, 7, 9, 7, 9])
        scopes = [o['field_metadata']['forecast']['context_scope'] for o in full['remote_evidence_used']['objects']
                  if o['track_id'] == 7 and 'forecast' in o]
        self.assertCountEqual(scopes, ['provider_full_at_t', 'receiver_acquired_subset'])
        self.assertEqual(ledger, original)

    def test_round_robin_receiver_skips_oversize_unit_and_preserves_ego(self):
        from planning.context import build_task_plan_input
        from planning.inputs import unpack_evidence
        ledger = self.apply(self.service().query_task(self.request()))
        def count(prompt):
            line = next((s for s in prompt.splitlines() if s.startswith('Additional queried neighbor evidence: ')), None)
            objects = unpack_evidence(json.loads(line.split(': ', 1)[1]))['objects'] if line else []
            return 20 + sum(500 if o['track_id'] == 7 and kind == 'forecast' else 100
                            for o in objects for kind in o['field_metadata'])
        limits = dict(version='toolv2x_receiver_v2', context_limit=280, generation_reserve=50, peer_reserve=200)
        try:
            result = build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 0,
                limits=limits, token_counter=count)
        except ValueError as exc:
            self.fail('declared receiver must skip non-fitting units: ' + str(exc))
        groups = result['admission_report']['field_groups']
        self.assertEqual([(g['anchor_ref']['track_handle'], g['kind']) for g in groups], [(9, 'forecast'), (7, 'history')])
        self.assertEqual(result['evidence_selection']['input_tokens'], 220)
        ego = build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), self.ledger, 0,
            limits=limits, token_counter=count)
        self.assertIsNone(ego['remote_evidence_used'])
        self.assertEqual(ego['evidence_used'], result['evidence_used'])
        with self.assertRaises(ValueError):
            build_task_plan_input(None, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 0,
                limits=dict(limits, version='unimplemented'), token_counter=count)


if __name__ == '__main__':
    unittest.main()
