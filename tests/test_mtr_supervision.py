"""Offline labels must not select or modify online targets using future data."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from test_vehicle_tools import window


class MTRSupervisionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('prediction.supervision'), 'offline MTR supervision missing')
        from prediction import supervision
        return supervision

    def test_labels_match_current_only_keep_missing_and_preserve_inputs(self):
        S = self.module()
        w = window()
        before = copy.deepcopy(w)
        boxes = w['states'][:, -1].astype(float)
        ids = np.array([100, 200])
        future = []
        for step in range(50):
            value = boxes.copy()
            value[:, 0] += (step + 1) * .2
            future.append((value, ids) if step != 2 else (value[:1], ids[:1]))
        labels = S.make_labels(w, boxes, ids, future)
        self.assertEqual(labels['matched_gt_ids'].tolist(), [100, 200])
        self.assertEqual(labels['valid'].sum(axis=1).tolist(), [50, 49])
        self.assertFalse(labels['valid'][1, 2])
        np.testing.assert_array_equal(labels['xy'][1, 2], [0, 0])
        np.testing.assert_allclose(labels['xy'][0, -1], [12, 0])
        empty = S.make_labels(w, boxes, ids, [None] * 50)
        np.testing.assert_array_equal(empty['matched_gt_ids'], labels['matched_gt_ids'])
        self.assertEqual(empty['valid'].sum(), 0)
        for key in ('states', 'valid', 'scores', 'track_ids'):
            np.testing.assert_array_equal(w[key], before[key])

    def test_current_geometry_gates_and_ego_exclusion_leave_targets_in_ledger(self):
        S = self.module()
        w = window()
        w['states'][0, :, 0] = 0.
        gt = w['states'][:, -1].astype(float)
        gt[1, 3] = 20.
        labels = S.make_labels(w, gt, np.array([3, 4]), [(gt, np.array([3, 4]))] * 50)
        self.assertEqual(labels['track_ids'].tolist(), [7, 9])
        self.assertEqual(labels['matched_gt_ids'].tolist(), [-1, -1])
        self.assertEqual(labels['status'].tolist(), ['ego_self', 'unmatched'])
        self.assertFalse(labels['valid'].any())

    def test_native_training_labels_rotate_around_detected_center_and_preserve_context(self):
        S = self.module()
        from prediction import cmp_adapter as C
        w = window()
        w['states'][0, :, 6] = np.pi / 2
        boxes = w['states'][:, -1].astype(float)
        future = boxes.copy()
        future[:, 1] += 2.
        labels = S.make_labels(w, boxes, np.array([100, 200]), [(future, np.array([100, 200]))] * 50)
        native = S.training_batch(w, labels, [0])['input_dict']
        online = C.make_batch(w, [0])['input_dict']
        np.testing.assert_allclose(native['center_gt_trajs'][0, 0], [2., 0.], atol=1e-5)
        np.testing.assert_allclose(native['obj_trajs_future_state'][0, 1, 0], [2., -18.], atol=1e-5)
        self.assertEqual(native['center_gt_final_valid_idx'].tolist(), [49])
        np.testing.assert_array_equal(native['obj_trajs'], online['obj_trajs'])
        self.assertFalse(any('gt' in key or 'future' in key for key in online))
        bad = dict(labels, g=11)
        with self.assertRaises(ValueError):
            S.training_batch(w, bad, [0])


if __name__ == '__main__':
    unittest.main()
