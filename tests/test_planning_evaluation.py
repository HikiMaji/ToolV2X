"""Failures retain their denominator; dynamics begin at the actual ego origin."""
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def label(points):
    return dict(waypoints=points, valid=[True] * 6, times_seconds=[.5, 1., 1.5, 2., 2.5, 3.],
        target_q8='The suggested speed setting is: moderate. The suggested steering setting is: straight.')


class PlanningEvaluationTests(unittest.TestCase):
    def test_repeated_far_points_retain_initial_jump_but_stationary_origin_is_valid(self):
        from evaluation.planning import trajectory_metrics
        points = np.array([[49.3, 1.4]] * 6)
        result = trajectory_metrics(points, label(points.tolist()), 14.8377)
        self.assertEqual(result['ADE3'], 0.)
        self.assertTrue(result['all_future_points_equal'])
        self.assertGreater(result['first_segment_speed_mps'], 98.)
        self.assertGreater(result['initial_acceleration_estimate_mps2'], 330.)
        stopped = trajectory_metrics(np.zeros((6, 2)), label([[0., 0.]] * 6), 0.)
        self.assertEqual(stopped['peak_acceleration_estimate_mps2'], 0.)

    def test_parse_failure_and_wrong_q8_remain_in_aggregate(self):
        from evaluation.planning import evaluate_plan, summarize
        text = 'The suggested speed setting is: fast. The suggested steering setting is: straight.'
        invalid = dict(status='invalid_q9', q8_raw=text, q9_executed=True,
            q9_prompt='Context from the generated action answer: ' + text + '\n',
            q9_raw='The suggested future trajectory is [(1,0),(2,0),(3,0),(4,0),(5,0)].')
        row = evaluate_plan(invalid, label([[0., 0.]] * 6), {'speed_mps': 0.})
        report = summarize([row])
        self.assertEqual(report['attempts'], 1)
        self.assertEqual(report['q9_parse_successes'], 0)
        self.assertEqual(report['q8_speed_correct'], 0)
        self.assertEqual(report['q8_steering_correct'], 1)
        self.assertEqual(report['ADE3_targets'], 0)
        self.assertIsNone(report['ADE3'])

    def test_missing_last_label_does_not_substitute_an_earlier_endpoint(self):
        from evaluation.planning import trajectory_metrics
        y = label([[0., 0.]] * 6)
        y['valid'][-1], y['waypoints'][-1] = False, None
        result = trajectory_metrics(np.ones((6, 2)), y, 0.)
        self.assertIsNone(result['FDE3'])
        self.assertAlmostEqual(result['ADE3'], np.sqrt(2.))
        self.assertEqual(result['valid_label_points'], 5)


if __name__ == '__main__':
    unittest.main()
