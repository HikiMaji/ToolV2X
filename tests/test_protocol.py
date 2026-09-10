"""Regression checks for causal inputs and honest offline oracle reporting."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prediction import tracks_to_cmp as T
from oracle import oracle_analysis as O


class ProtocolTests(unittest.TestCase):
    def test_causal_window_rebases_history_into_current_ego_frame(self):
        from prediction.causal_windows import make_window
        frames = {g: np.array([[7, 100 + g, 0, 0, 4, 2, 1, 0, .9]]) for g in range(12)}
        pose = np.eye(4)
        pose[0, 3] = 100
        a = make_window(frames, 10, 0, pose, 'no_fusion')
        frames[11][:, 1] = 999999
        b = make_window(frames, 10, 0, pose, 'no_fusion')
        np.testing.assert_array_equal(a['states'], b['states'])
        np.testing.assert_allclose(a['states'][0, :, 0], np.arange(11))
        np.testing.assert_allclose(a['time_seconds'], np.arange(-10, 1) * .1)
        self.assertEqual(a['track_ids'].tolist(), [7])
        self.assertEqual(a['source'], 'no_fusion')

    def test_window_keeps_masked_gaps_and_remote_only_targets(self):
        from prediction.causal_windows import make_window
        frames = {g: np.zeros((0, 9)) for g in range(11)}
        frames[9] = np.array([[77, 9, 0, 0, 4, 2, 1, 0, .9]])
        frames[10] = np.array([[77, 10, 0, 0, 4, 2, 1, 0, .9]])
        a = make_window(frames, 10, 0, np.eye(4), 'no_fusion_cav1')
        self.assertEqual(a['track_ids'].tolist(), [77])
        self.assertEqual(a['valid'].sum(), 2)
        self.assertFalse(a['valid'][0, 8])
        self.assertTrue(a['eligible'][0])

    def test_raw_export_does_not_fill_a_gap_with_future_observations(self):
        empty = np.zeros((0, 9))
        frames = {0: np.array([[7, 0, 0, 0, 4, 2, 1, 0, .9]]),
                  1: empty, 2: np.array([[7, 20, 0, 0, 4, 2, 1, 0, .9]])}
        with patch.object(T.M, 'load_pose', return_value=np.eye(4)), \
                patch.object(T.M, 'load_gt', side_effect=AssertionError('GT read')):
            full, _ = T.build_scene(frames, 'test', 0, 3, 'raw')
            prefix, _ = T.build_scene({k: v for k, v in frames.items() if k < 2},
                                      'test', 0, 2, 'raw')
        self.assertEqual(full['data'][7][:2], prefix['data'][7])
        self.assertEqual(full['data'][7][1][7], 0)

    def test_cooperating_vehicle_remains_an_obstacle(self):
        boxes = np.array([[0, 0, 0, 4, 2, 1, 0], [10, 0, 0, 4, 2, 1, 0]])
        pose = np.eye(4)
        pose[0, 3] = 10
        with patch.object(O.M, 'load_gt', return_value=(boxes, np.array([1, 2]))), \
                patch.object(O.os.path, 'exists', return_value=True), \
                patch.object(O.np, 'load', return_value=pose):
            self.assertEqual(O.cav_self_ids('test', 0), {1})

    def test_late_ego_annotation_is_not_a_future_obstacle(self):
        def gt(split, g):
            if g == 0:
                return np.zeros((0, 7)), np.array([], dtype=int)
            return np.array([[0, 0, 0, 4, 2, 1, 0], [10, 0, 0, 4, 2, 1, 0]]), np.array([1, 2])
        with patch.object(O.M, 'load_gt', side_effect=gt), \
                patch.object(O.M, 'load_pose', return_value=np.eye(4)):
            self.assertEqual(set(O.gt_other_futures('test', 0, horizon=2)), {2})

    def test_multiple_evaluation_directories_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as d:
            for run in ('a', 'b'):
                os.makedirs(os.path.join(d, 'cfg', 'default', 'eval', run, 'inference_result'))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                O.load_results(d, 'cfg', 'default')

    def test_missing_action_frame_is_rejected_before_scoring(self):
        data = {k: {('scene', 10): {}} for k in O.CFGS}
        data['F'] = {}
        with tempfile.TemporaryDirectory() as d, patch.object(O, 'scene_to_global', return_value={'scene': 0}):
            with self.assertRaisesRegex(ValueError, 'frame'):
                O.evaluate('test', data, os.path.join(d, 'out.csv'))

    def test_invalid_prediction_is_not_treated_as_clear_road(self):
        for traj, score in [(np.full((1, 30, 2), np.nan), np.array([1.])),
                            (np.zeros((1, 29, 2)), np.array([1.])),
                            (np.zeros((1, 30, 2)), np.array([np.nan]))]:
            with self.assertRaises(ValueError):
                O.preds_to_others({1: (traj, score)})

    def test_summary_does_not_invent_bandwidth_or_four_way_ties(self):
        row = {'scene': 's', 'occ_critical': 0, 'gt_coverage': 1.0}
        for k, cost in zip(O.CFGS, [0., 0., 5., 6.]):
            row.update({k + '_cost': cost, k + '_l2': cost, k + '_coll': 0., k + '_nobj': 1})
        md = O.summarize([row])
        self.assertNotIn('带宽', md)
        self.assertNotIn('四者打平', md)
        self.assertNotIn('10–15', md)
        self.assertIn('GT', md)


if __name__ == '__main__':
    unittest.main()
