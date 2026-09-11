"""Decoder contracts only; stub generations are never used as driving evidence."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.inputs import make_prompt

MOTION = dict(speed_mps=2., yaw_rate_rps=0.)
EVIDENCE = dict(as_of_g=10, objects=[])
TRAJECTORY = 'The suggested future trajectory is [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)].'
ACTION = 'The suggested speed setting is: slow. The suggested steering setting is: straight.'


class DirectPlanningInputTests(unittest.TestCase):
    def test_direct_prompt_has_no_action_parent_and_old_chain_stays_strict(self):
        try:
            prompt = make_prompt('Trajectory', MOTION, EVIDENCE)
        except ValueError as exc:
            self.fail('direct trajectory task missing: ' + str(exc))
        self.assertIn('six numeric (x,y) waypoints', prompt)
        self.assertNotIn('Context from the generated action answer', prompt)
        with self.assertRaises(ValueError):
            make_prompt('Trajectory', MOTION, EVIDENCE, ACTION)
        with self.assertRaises(ValueError):
            make_prompt('Q9', MOTION, EVIDENCE)

    def test_direct_supervision_contains_only_trajectory_target(self):
        from planning.adaptation_data import supervised_examples
        prompt = make_prompt('Trajectory', MOTION, EVIDENCE)
        plan = dict(decoding='direct', q8_executed=False, q9_prompt=prompt,
                    evidence_used=EVIDENCE, evidence_selection={'evidence_format': 'json'})
        label = dict(target_q8='GT_ACTION_MUST_NOT_ENTER_THE_PROMPT', target_q9=TRAJECTORY)
        values = supervised_examples('train', 'sample', 'Ego', MOTION, plan, label, '/features.npz')
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['task'], 'Trajectory')
        self.assertEqual(values[0]['target'], TRAJECTORY)
        self.assertNotIn(label['target_q8'], values[0]['prompt'])
        self.assertEqual(values[0]['q8_parent_source'], 'not applicable')

    def test_direct_evaluation_excludes_q8_and_retains_trajectory_failures(self):
        from evaluation.planning import evaluate_plan, summarize
        label = dict(waypoints=[[i, 0.] for i in range(1, 7)], valid=[True] * 6,
                     times_seconds=[.5, 1., 1.5, 2., 2.5, 3.], target_q8=ACTION)
        plan = dict(decoding='direct', q8_executed=False, q9_executed=True,
                    q9_prompt='direct trajectory prompt', q9_raw=TRAJECTORY, status='parsed')
        try:
            good = evaluate_plan(plan, label, MOTION)
        except ValueError as exc:
            self.fail('explicit direct trajectory must not require Q8: ' + str(exc))
        failed = evaluate_plan(dict(plan, q9_raw='five points', status='invalid_q9'), label, MOTION)
        report = summarize([good, failed])
        self.assertEqual(report['q8_labelled_attempts'], 0)
        self.assertIsNone(report['q8_joint_accuracy'])
        self.assertEqual(report['q9_failure_fraction'], .5)
        self.assertEqual(report['ADE3'], 0.)
        self.assertFalse(good['q8_applicable'])
        with self.assertRaises(ValueError):
            evaluate_plan(dict(plan, q8_raw=ACTION), label, MOTION)
        with self.assertRaises(ValueError):
            evaluate_plan(dict(plan, decoding='q8_q9'), label, MOTION)


class DirectPlanningModelTests(unittest.TestCase):
    def test_direct_calls_generation_once_with_same_evidence_as_two_stage(self):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT, V2VGoTPlanner
        planner = object.__new__(V2VGoTPlanner)
        planner.tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        planner.context_limit, planner.evidence_format = 4096, 'json'
        calls = []

        def generated(features, prompt, limit):
            calls.append(limit)
            return (ACTION if limit == 128 else TRAJECTORY), dict(test_only=True, output_tokens=10)

        planner._generate = generated
        features = {'active_agent_mask': np.ones((1, 2, 1), bool)}
        evidence = copy.deepcopy(EVIDENCE)
        evidence['objects'] = [dict(source='ego', track_id=7, box=[2., 0, 0, 4, 2, 1.5, 0], score=.8)]
        try:
            direct = planner.plan(features, MOTION, evidence, decoding='direct')
        except TypeError as exc:
            self.fail('planner does not expose direct decoding: ' + str(exc))
        self.assertEqual(calls, [256])
        self.assertFalse(direct['q8_executed'])
        self.assertEqual(direct['status'], 'parsed')
        staged = planner.plan(features, MOTION, evidence)
        self.assertEqual(calls, [256, 128, 256])
        self.assertEqual(direct['evidence_used'], staged['evidence_used'])
        self.assertEqual(direct['evidence_selection'], staged['evidence_selection'])


if __name__ == '__main__':
    unittest.main()
