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


if __name__ == '__main__':
    unittest.main()
