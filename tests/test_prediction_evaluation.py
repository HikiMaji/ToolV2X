"""Prediction quality uses score-selected top-1 and explicit horizon coverage."""
import importlib.util
from pathlib import Path
import sys
import unittest
from copy import deepcopy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class PredictionEvaluationTests(unittest.TestCase):
    def test_top1_is_score_selected_and_missing_endpoint_is_not_scored(self):
        self.assertTrue((ROOT / 'src/evaluation/prediction.py').is_file(), 'prediction evaluator missing')
        from evaluation.prediction import target_metrics
        means = np.zeros((6, 50, 2))
        means[0, :, 0] = 3.
        mask = np.ones(50, bool)
        mask[-1] = False
        result = target_metrics(means, np.array([.8, .1, 0, 0, 0, 0]), np.zeros((50, 2)), mask)
        self.assertEqual(result['top1_ADE5'], 3.)
        self.assertEqual(result['minADE5'], 0.)
        self.assertIsNone(result['top1_FDE5'])
        self.assertEqual(result['top1_FDE3'], 3.)
        self.assertEqual(result['valid_points5'], 49)

    def test_no_label_targets_and_nonfinite_predictions_are_explicit(self):
        self.assertTrue((ROOT / 'src/evaluation/prediction.py').is_file(), 'prediction evaluator missing')
        from evaluation.prediction import target_metrics
        means = np.zeros((6, 50, 2))
        scores = np.ones(6) / 6
        result = target_metrics(means, scores, np.zeros((50, 2)), np.zeros(50, bool))
        self.assertIsNone(result['top1_ADE5'])
        means[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            target_metrics(means, scores, np.zeros((50, 2)), np.ones(50, bool))

    def test_pairing_keeps_target_identity_and_rejects_changed_coverage(self):
        from evaluation.prediction import METRICS, paired
        before = [dict(context='full', scene='a', source='ego', t=10, track_id=i,
                       recording='a', model_used=True, matched_gt_id=i,
                       **{key: float(i + 1) for key in METRICS}) for i in range(2)]
        after = deepcopy(before)
        for key in METRICS:
            after[0][key] -= .5
            after[1][key] += .25
        result = paired(before, after[::-1])['full']['top1_ADE5']
        self.assertEqual(result['targets'], 2)
        self.assertEqual(result['after_minus_before_m'], -.125)
        self.assertEqual(result['worsened_fraction'], .5)
        with self.assertRaises(ValueError):
            paired(before, after[:1])
        after[0]['top1_ADE5'] = None
        with self.assertRaises(ValueError):
            paired(before, after)


if __name__ == '__main__':
    unittest.main()
