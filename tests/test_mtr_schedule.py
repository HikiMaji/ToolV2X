"""Fixed update budgets must survive ROI exclusions and empty supervision."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction import train_mtr as T
from prediction.supervision import make_labels
from test_vehicle_tools import window


def examples():
    result = []
    for i, offset in enumerate((0., 200., 0.)):
        w = window()
        w['states'][1, :, 0] = 100.
        w['states'][:, :, 0] += offset
        boxes = w['states'][:, -1].copy()
        ids = np.array([100, 200])
        y = make_labels(w, boxes, ids, [(boxes, ids)] * 50 if i < 2 else [None] * 50)
        result.append(({'t': i}, w, y))
    return result


class MTRScheduleTests(unittest.TestCase):
    def setUp(self):
        self.iterate = getattr(T, 'epoch_examples', None)
        self.assertTrue(callable(self.iterate), 'missing fixed-update training iterator')

    def test_full_and_roi_have_same_updates_with_repeated_sampling_and_no_mutation(self):
        data = examples()
        original = copy.deepcopy(data)
        for context in ('full', 'roi'):
            np.random.seed(20)
            rows = list(self.iterate(data, context, steps_per_epoch=5))
            trained = [r for r in rows if len(r[3])]
            self.assertEqual(len(trained), 5)
            if context == 'roi':
                self.assertEqual([r[0]['t'] for r in trained], [0] * 5)
                self.assertTrue(all(len(r[1]['track_ids']) == 1 and len(r[3]) == 1 for r in trained))
            else:
                self.assertEqual({r[0]['t'] for r in trained}, {0, 1})
                self.assertTrue(all(len(r[1]['track_ids']) == 2 and len(r[3]) == 2 for r in trained))
        for (_, w, y), (_, ow, oy) in zip(data, original):
            np.testing.assert_array_equal(w['states'], ow['states'])
            np.testing.assert_array_equal(y['valid'], oy['valid'])

    def test_default_visits_each_source_frame_once_and_seed_replays_order(self):
        data = examples()
        np.random.seed(21)
        first = list(self.iterate(data, 'full'))
        self.assertEqual(sorted(r[0]['t'] for r in first), [0, 1, 2])
        self.assertEqual(sum(bool(len(r[3])) for r in first), 2)
        np.random.seed(21)
        replay = list(self.iterate(data, 'full'))
        self.assertEqual([r[0]['t'] for r in first], [r[0]['t'] for r in replay])

    def test_empty_roi_pool_and_invalid_settings_fail_before_sampling(self):
        data = examples()
        for values, mode, steps in (([data[1]], 'roi', 5), ([data[2]], 'full', 5),
                                    (data, 'invalid', 5), (data, 'full', 0)):
            with self.subTest(mode=mode, steps=steps), self.assertRaises(ValueError):
                list(self.iterate(values, mode, steps_per_epoch=steps))


if __name__ == '__main__':
    unittest.main()
