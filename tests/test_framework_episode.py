"""Exercise causal tool interaction and the shared training/inference input."""
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
from tools.vehicle import VehicleTools, make_evidence, decode_response


class EpisodeTests(unittest.TestCase):
    def runner(self):
        self.assertTrue((ROOT / 'src/planning/episode.py').is_file(), 'complete task runner is missing')
        return importlib.import_module('planning.episode').run_episode

    def test_stop_does_not_access_remote_source(self):
        def forbidden():
            self.fail('STOP read an unqueried peer')
        ego = window('ego')
        service = VehicleTools(forbidden, fixture_prediction, 'scene', 10, provider='peer')
        result = self.runner()(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                               service, fixture_prediction, 'Ego')
        self.assertEqual(result['cost']['request_bytes'], 0)
        self.assertEqual(result['cost']['response_bytes'], 0)
        self.assertEqual(result['cost']['calls'], 0)
        self.assertEqual(result['steps'][-1]['decision']['tool'], 'STOP')
        self.assertEqual({o['source'] for o in result['evidence']['objects']}, {'ego'})

    def test_actual_p_result_changes_second_query(self):
        run = self.runner()
        ego = window('ego')
        outcomes = []
        for peer_x in (20., 25.):
            peer, events = window(), []
            peer['states'][1, :, 0] = peer_x
            def load():
                events.append('load_peer')
                return peer
            def remote_predict(w):
                events.append('remote_predict')
                return fixture_prediction(w)
            service = VehicleTools(load, remote_predict, 'scene', 10, provider='peer')
            result = run(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                         service, fixture_prediction, 'rule')
            tools = [s['decision']['tool'] for s in result['steps']]
            outcomes.append(tools)
            self.assertEqual(events[0], 'load_peer')
            self.assertEqual(events.count('remote_predict'), int('F' in tools))
            self.assertIsNotNone(result['steps'][0]['decision']['roi'])
            self.assertEqual(result['cost']['response_bytes'],
                             sum(s.get('cost', {}).get('response_bytes', 0) for s in result['steps']))
        self.assertEqual(outcomes, [['P', 'STOP'], ['P', 'F', 'STOP']])

    def test_budget_stops_fixed_pf_after_real_p(self):
        ego = window('ego')
        service = VehicleTools(window, fixture_prediction, 'scene', 10, provider='peer')
        result = self.runner()(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                               service, fixture_prediction, 'PF', max_calls=1)
        self.assertEqual([s['decision']['tool'] for s in result['steps']], ['P', 'STOP'])
        self.assertEqual(result['stop_reason'], 'call_budget_exhausted')
        self.assertEqual(result['cost']['calls'], 1)

    def test_p_can_be_received_without_running_local_predictor(self):
        ego = window('ego')
        service = VehicleTools(window, fixture_prediction, 'scene', 10, provider='peer')
        packet = decode_response(service.query('P')['wire'], 'scene', 10)
        def forbidden(w):
            self.fail('observation-only P ran a prediction model')
        try:
            evidence = make_evidence(ego, fixture_prediction(ego), [packet], forbidden, predict_p=False)
        except TypeError as exc:
            self.fail('explicit observation-only P reception missing: ' + str(exc))
        remote = [o for o in evidence['objects'] if o['source'] != 'ego']
        self.assertEqual(len(remote), 2)
        self.assertTrue(all(o['source'] == 'peer:P' and 'history' in o and 'forecast' not in o for o in remote))

    def test_empty_p_does_not_become_observed_free_space(self):
        ego, peer = window('ego'), window()
        peer['states'][:, :, 0] = 100.
        service = VehicleTools(lambda: peer, fixture_prediction, 'scene', 10, provider='peer')
        result = self.runner()(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                               service, fixture_prediction, 'rule')
        self.assertEqual(result['steps'][0]['response']['coverage'], 'not_established')
        self.assertEqual(result['steps'][0]['response']['status'], 'no_observed_targets')
        self.assertEqual(result['stop_reason'], 'no_returned_targets')

    def test_receiver_failure_keeps_p_response_and_actual_cost(self):
        ego = window('ego')
        service = VehicleTools(window, fixture_prediction, 'scene', 10, provider='peer')
        def broken_predictor(w):
            raise RuntimeError('receiver failed after receiving P')
        result = self.runner()(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                               service, broken_predictor, 'PF')
        self.assertEqual(result['status'], 'tool_error')
        self.assertEqual(result['cost']['calls'], 1)
        self.assertGreater(result['cost']['response_bytes'], 0)
        self.assertEqual(result['steps'][0]['response']['tool'], 'P')
        self.assertEqual(result['steps'][0]['status'], 'error')
        self.assertEqual(result['error']['stage'], 'receiver_update')
        self.assertEqual(result['cost']['receiver_model_calls'], 1)

    def test_service_failure_keeps_attempt_and_marks_unknown_cost(self):
        ego = window('ego')
        def unavailable():
            raise RuntimeError('peer unavailable')
        service = VehicleTools(unavailable, fixture_prediction, 'scene', 10, provider='peer')
        result = self.runner()(ego, fixture_prediction(ego), {'speed_mps': 10., 'yaw_rate_rps': 0.},
                               service, fixture_prediction, 'P')
        self.assertEqual(result['cost']['calls'], 1)
        self.assertGreater(result['cost']['request_bytes'], 0)
        self.assertFalse(result['cost']['complete'])
        self.assertEqual(result['steps'][0]['request']['tool'], 'P')
        self.assertEqual(result['error']['stage'], 'service_query')


class SharedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT
        cls.tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)

    def builder(self):
        self.assertTrue((ROOT / 'src/planning/context.py').is_file(), 'shared model input builder is missing')
        return importlib.import_module('planning.context').build_plan_input

    def real_inputs(self):
        root = ROOT / 'outputs/paired_driving_data_v1/frames/train_g0719'
        local = json.loads((root / 'ego_input.json').read_text())
        peer = json.loads((root / 'F_input.json').read_text())
        sample = json.loads((root / 'sample.json').read_text())
        combined = dict(peer['evidence_full'],
                        objects=local['evidence_full']['objects'] + peer['evidence_full']['objects'])
        return local, peer, sample, combined

    def test_shared_builder_exactly_reproduces_saved_paired_inputs(self):
        build = self.builder()
        local, peer, sample, combined = self.real_inputs()
        for evidence, expected in ((local['evidence_full'], local), (combined, peer)):
            prepared = build(self.tokenizer, sample['ego_motion'], evidence, sample['feature_tokens'])
            self.assertEqual(prepared['q9_prompt'], expected['prompt'])
            self.assertEqual(prepared['evidence_used'], local['evidence_used'])
            self.assertEqual(prepared['input_layout'], 'source_blocks_v1')

    def test_driver_and_supervision_use_same_prepared_prompt(self):
        build = self.builder()
        from planning.v2vgot import V2VGoTPlanner
        from planning.adaptation_data import supervised_examples
        _, _, sample, combined = self.real_inputs()
        prepared = build(self.tokenizer, sample['ego_motion'], combined, sample['feature_tokens'])
        planner = V2VGoTPlanner.__new__(V2VGoTPlanner)
        planner.tokenizer, planner.context_limit = self.tokenizer, 4096
        planner.evidence_format = 'compact'
        captured = []
        def generate(features, prompt, budget):
            captured.append(prompt)
            return '[(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)]', {'output_tokens': 40}
        planner._generate = generate
        self.assertTrue(hasattr(planner, 'plan_prepared'), 'driver cannot execute the shared prepared input')
        with np.load(sample['feature_path'], allow_pickle=False) as stored:
            features = dict(stored)
        plan = planner.plan(features, sample['ego_motion'], combined,
                            decoding='direct', input_layout='source_blocks_v1')
        self.assertEqual(captured, [prepared['q9_prompt']])
        rows = supervised_examples('train', sample['sample_id'], 'F', sample['ego_motion'], plan,
                                   {'target_q9': 'target only'}, sample['feature_path'])
        self.assertEqual(rows[0]['prompt'], captured[0])
        poisoned = copy.deepcopy(prepared)
        poisoned['q9_prompt'] += ' future label'
        with self.assertRaises(ValueError):
            planner.plan_prepared(features, poisoned)

    def test_task_receiver_uses_original_tokenizer_without_model_generation(self):
        from planning.context import build_task_plan_input
        from planning.evidence import new_ledger, apply_response
        from planning.v2vgot import prompt_tokens
        from tools.task_spec import FrozenPredictor
        from test_task_spec import task_request, provenance
        predictor = FrozenPredictor(fixture_prediction, provenance()['prediction'],
                                    dict(adapter='cmp_causal_window_v1'))
        ego = window('ego')
        ledger = new_ledger(ego, predictor(ego), predictor=predictor, local_provenance=provenance())
        service = VehicleTools(window, predictor, 'scene', 10, 'peer', task_provenance=provenance())
        # Default execution/receiver budgets; test only tokenizer plus fake predictor.
        ledger = apply_response(ledger, service.query_task(task_request()), predictor)
        prepared = build_task_plan_input(self.tokenizer, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 64)
        exact = len(prompt_tokens(self.tokenizer, prepared['q9_prompt'])) - 1 + 64
        self.assertEqual(prepared['evidence_selection']['input_tokens'], exact)
        self.assertLessEqual(exact + 256, 4096)
        self.assertGreater(prepared['evidence_selection']['remote_units_retained'], 0)
        self.assertEqual(prepared['evidence_selection']['token_counting'], 'original_got_prompt_tokens')
        self.assertFalse(prepared['language_model_executed'])
        before = prepared['q9_prompt']
        ledger = apply_response(ledger, service.query_task(task_request(tool='F', request_id='q1')), predictor)
        after = build_task_plan_input(self.tokenizer, dict(speed_mps=4., yaw_rate_rps=0.), ledger, 64)
        self.assertEqual(before, after['q9_prompt'])


    def test_v2_driver_adapter_validates_original_budget_before_generation(self):
        from planning.context import build_task_plan_input
        from planning.evidence import new_ledger
        from planning.v2vgot import V2VGoTPlanner
        from tools.task_spec import FrozenPredictor
        from test_task_spec import provenance
        predictor = FrozenPredictor(fixture_prediction, provenance()['prediction'], dict(adapter='fixture'))
        ego = window('ego')
        ledger = new_ledger(ego, predictor(ego), predictor=predictor, local_provenance=provenance())
        features = dict(active_agent_mask=np.array([[[True], [False]]]))
        prepared = build_task_plan_input(self.tokenizer, dict(speed_mps=2., yaw_rate_rps=0.), ledger, 270)
        planner = V2VGoTPlanner.__new__(V2VGoTPlanner)
        planner.tokenizer, planner.context_limit = self.tokenizer, 4096
        seen = []
        def generate(features, prompt, budget):
            seen.append((prompt, budget))
            return 'The suggested trajectory is: [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)]', dict(output_tokens=40)
        planner._generate = generate
        output = planner.plan_prepared(features, prepared)
        self.assertEqual(output['status'], 'parsed')
        self.assertEqual(seen, [(prepared['q9_prompt'], 256)])
        for area, key, bad in [('evidence_selection', 'token_counting', 'injected_contract_counter'),
                              ('evidence_selection', 'input_tokens', 1),
                              ('evidence_selection', 'generation_reserve', 1),
                              ('receiver_spec', 'peer_reserve', 0),
                              ('receiver_spec', 'context_limit', 8192)]:
            poisoned = copy.deepcopy(prepared)
            poisoned[area][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                planner.plan_prepared(features, poisoned)
        self.assertEqual(len(seen), 1)


    def test_task_episode_calls_actual_driver_adapter_with_same_features_and_tokenizer(self):
        from planning.method_episode import run_task_episode, diagnostic_policy
        from planning.v2vgot import V2VGoTPlanner, prompt_tokens
        from tools.task_spec import FrozenPredictor
        from test_task_spec import provenance, spec_dict
        predictor = FrozenPredictor(fixture_prediction, provenance()['prediction'], dict(adapter='fixture'))
        ego = window('ego')
        service = VehicleTools(window, predictor, 'scene', 10, 'peer', task_provenance=provenance())
        planner = V2VGoTPlanner.__new__(V2VGoTPlanner)
        planner.tokenizer, planner.context_limit = self.tokenizer, 4096
        planner.provenance = dict(model_class='original_adapter_synthetic_generate')
        seen = []
        def generate(features, prompt, budget):
            seen.append((id(features), prompt))
            y = .1 if 'Additional queried neighbor evidence:' in prompt else 0.
            tau = [[float(i+1), y*(i+1)] for i in range(6)]
            return 'The suggested trajectory is: ' + repr(tau), dict(seconds=.1,
                input_tokens=len(prompt_tokens(self.tokenizer, prompt))-1+270, output_tokens=40, feature_tokens=270)
        planner._generate = generate  # Only expensive external generation is substituted.
        limits = dict(version='toolv2x_interaction_v1', max_calls=2, policy_id='p_current_f_change',
            driver_version=dict(name='adapter_contract', revision='v1'), execution_spec=spec_dict(),
            receiver_spec=dict(version='toolv2x_receiver_v1', context_limit=4096, generation_reserve=256,
                               peer_reserve=1536, numeric_decimal_places=2))
        result = run_task_episode(ego, predictor(ego), dict(speed_mps=2., yaw_rate_rps=0.),
            dict(active_agent_mask=np.array([[[True], [False]]])), service, predictor, planner,
            diagnostic_policy, limits, local_provenance=provenance())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['plans']), 3)
        self.assertEqual({item[0] for item in seen}, {seen[0][0]})
        self.assertEqual(result['requests'][1]['mode'], 'change')
        self.assertEqual(result['requests'][1]['tau_new'], result['plans'][1]['output']['waypoints'])
        self.assertTrue(all(p['prepared']['evidence_selection']['token_counting'] == 'original_got_prompt_tokens'
                            for p in result['plans']))


if __name__ == '__main__':
    unittest.main()
