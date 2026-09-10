"""Offline supervision must never supply the online Q8 parent answer."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class AdaptationDataTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'src/planning/adaptation_data.py'
        self.assertTrue(path.is_file(), 'offline adaptation preparation is not implemented')
        from planning import adaptation_data
        return adaptation_data

    def test_native_labels_use_six_future_poses_in_fixed_current_frame(self):
        module = self.module()
        calls = []
        def pose(split, g):
            calls.append(g)
            value = np.eye(4)
            value[:2, :2] = [[0., -1.], [1., 0.]]
            value[:2, 3] = [1000., 2000. + (g - 10) * .4]
            return value
        label = module.ego_label('test', 10, pose)
        np.testing.assert_allclose(label['waypoints'], [[2, 0], [4, 0], [6, 0], [8, 0], [10, 0], [12, 0]])
        self.assertEqual(label['target_q8'], 'The suggested speed setting is: very slow. The suggested steering setting is: straight.')
        self.assertEqual(set(calls), {10, 15, 20, 25, 30, 35, 40})
        self.assertTrue(all(label['valid']))

    def test_missing_or_cross_record_future_keeps_mask_without_fabricating_label(self):
        module = self.module()
        calls = []
        def pose(split, g):
            calls.append(g)
            if g == 15:
                raise FileNotFoundError('missing offline pose')
            return np.eye(4)
        missing = module.ego_label('test', 10, pose)
        self.assertFalse(missing['valid'][0])
        self.assertIsNone(missing['target_q8'])
        self.assertIsNone(missing['target_q9'])
        calls.clear()
        boundary = module.ego_label('test', 140, pose)
        self.assertEqual(boundary['valid'], [True, False, False, False, False, False])
        self.assertEqual(calls, [140, 145])

    def test_supervised_q9_uses_generated_parent_even_when_q9_prediction_failed(self):
        module = self.module()
        from planning.inputs import make_prompt
        state, evidence = dict(speed_mps=4., yaw_rate_rps=0.), dict(as_of_g=10, objects=[])
        generated = 'The suggested speed setting is: stop. The suggested steering setting is: straight.'
        truth = 'The suggested speed setting is: fast. The suggested steering setting is: right.'
        plan = dict(q8_raw=generated, q8_prompt=make_prompt('Q8', state, evidence),
                    q9_prompt=make_prompt('Q9', state, evidence, generated),
                    status='invalid_q9', evidence_used=evidence, evidence_selection={'evidence_format': 'json'})
        label = dict(target_q8=truth, target_q9='The suggested future trajectory is [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)].')
        rows = module.supervised_examples('train', 'sample', 'P', state, plan, label, '/features.npz')
        self.assertEqual(len(rows), 2)
        self.assertIn(generated, rows[1]['prompt'])
        self.assertNotIn(truth, rows[1]['prompt'])
        self.assertEqual(rows[0]['target'], truth)
        self.assertEqual(rows[1]['target'], label['target_q9'])
        with self.assertRaises(ValueError):
            module.supervised_examples('validation', 'sample', 'P', state, plan, label, '/features.npz')
        plan['q8_raw'] = 'There is a car.'
        rows = module.supervised_examples('train', 'sample', 'P', state, plan, label, '/features.npz')
        self.assertEqual(len(rows), 1)

    def test_runner_selects_authorized_training_frame_without_using_validation(self):
        from planning import run_connection
        self.assertTrue(hasattr(run_connection, 'select_decision'), 'training frame selection is not implemented')
        self.assertEqual(run_connection.select_decision('train', 0, 10),
                         'testoutput_CAV_data_2022-03-15-10-09-50_0')
        with self.assertRaises(ValueError):
            run_connection.select_decision('test', 0, 10)
        with self.assertRaises(ValueError):
            run_connection.select_decision('train', -1, 10)

    def test_training_tokens_mask_all_prompt_and_feature_positions(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'encode_supervision'), 'supervised token masking is not implemented')
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT
        from planning.inputs import make_prompt
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        row = dict(prompt=make_prompt('Q8', dict(speed_mps=4., yaw_rate_rps=0.), dict(as_of_g=10, objects=[])),
                   target='The suggested speed setting is: stop. The suggested steering setting is: straight.')
        batch = module.encode_supervision(row, tokenizer, 540)
        labels = batch['labels'][0]
        active = labels != -100
        self.assertGreater(int((~active).sum()), 20)
        self.assertTrue((batch['input_ids'][0][active] == labels[active]).all())
        self.assertEqual(tokenizer.decode(labels[active], skip_special_tokens=True).strip(), row['target'])
        self.assertTrue((labels[batch['input_ids'][0] == -200] == -100).all())
        with self.assertRaises(ValueError):
            module.encode_supervision(row, tokenizer, 4096)


if __name__ == '__main__':
    unittest.main()
