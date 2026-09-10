"""Scientific contracts for the learned capability screen."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from probe import kinematic_tools as K
from probe import learned_predictor as L
from probe import learned_data as D


def history(source='peer'):
    states = np.zeros((3, 11, 7))
    states[:, :, 3:6] = [4., 2., 1.5]
    states[:, :, 0] = np.arange(11)[None] + np.array([10., 30., 50.])[:, None]
    valid = np.ones((3, 11), bool)
    valid[2, :-1] = False
    return K.history_packet(dict(source=source, g=10, states=states, valid=valid,
        scores=np.ones((3, 11)) * .9, track_ids=np.array([8, 9, 10]),
        time_seconds=np.arange(-10, 1) * .1))


class LearnedProbeTests(unittest.TestCase):
    def test_summary_is_velocity_capable_and_excludes_history(self):
        p = L.summary_packet(history())
        self.assertTrue(all('history' not in o for o in p['objects']))
        np.testing.assert_allclose(p['objects'][0]['velocity'], [10., 0.])
        a, _ = L.features(p)
        b, _ = L.features(history())
        np.testing.assert_array_equal(a[:, :L.CURRENT_FEATURES], b[:, :L.CURRENT_FEATURES])
        self.assertFalse(np.array_equal(a, b))

    def test_zero_model_is_cv_and_same_information_survives_wire(self):
        p = history()
        weights = np.zeros((50, L.FEATURES, 2))
        remote = L.forecast(p, weights)
        self.assertEqual(remote, L.forecast(json.loads(K.encode(p)), weights))
        for a, b in zip(remote['objects'], K.forecast(p)['objects']):
            np.testing.assert_allclose(a['trajectory'], b['trajectory'])
        weights[:, 0] = 10.
        changed = L.forecast(p, weights)
        self.assertEqual(changed['objects'][2]['trajectory'], remote['objects'][2]['trajectory'])
        self.assertNotEqual(changed['objects'][0]['trajectory'], remote['objects'][0]['trajectory'])
        merged, audit = K.receive(L.forecast(history('ego'), weights), remote_f=changed)
        self.assertEqual(audit['matched'], 3)
        self.assertEqual(len(merged), 3)

    def test_masked_positions_and_identity_do_not_enter_features(self):
        p = history()
        p['objects'][0]['valid'][3] = False
        a, _ = L.features(p)
        p['objects'][0]['history'][3][:2] = [9999., -9999.]
        p['objects'][0]['key'] = 'peer:987'
        b, _ = L.features(p)
        np.testing.assert_array_equal(a, b)

    def test_labels_keep_unmatched_targets_and_mask_missing_future(self):
        objects = K.forecast(history())['objects']
        current = np.array([[20., 0., 0., 4., 2., 1.5, 0.]])
        future = {77: np.full((50, 2), np.nan)}
        future[77][0] = [21., 0.]
        y, mask, ids = D.match_labels(objects, current, np.array([77]), future)
        self.assertEqual(y.shape, (3, 50, 2))
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(ids.tolist(), [77, -1, -1])
        np.testing.assert_array_equal(y[0, 0], [21., 0.])

    def test_paired_scoring_uses_common_keys_and_same_truth(self):
        truth = np.zeros((3, 50, 2))
        mask = np.ones((3, 50), bool)
        a, b = truth.copy(), truth.copy()
        a[0, :, 0] = 2.
        b[0, :, 0] = 1.
        mask[1:] = False
        score = D.paired_errors(a, b, truth, mask)
        self.assertEqual(score['targets'], 1)
        self.assertEqual(score['delta_ADE_b_minus_a'], -1.)
        self.assertEqual(score['harm_fraction'], 0.)

    def test_ridge_recovers_known_mapping_with_missing_labels(self):
        from probe.train_learned import sufficient_statistics, solve
        x = np.zeros((12, L.FEATURES))
        x[:, 0] = 1.
        x[:, 1] = np.arange(12) - 6.
        truth = np.tile((3. * x[:, 1] + 2.)[:, None, None], (1, 50, 2))
        mask = np.ones((12, 50), bool)
        mask[:4, 20:] = False
        truth[~mask] = 999999.
        gram, rhs, count = sufficient_statistics(x, truth, mask)
        w = solve(gram, rhs, count, .000001)
        np.testing.assert_allclose(w[:, 0], 2., atol=1e-5)
        np.testing.assert_allclose(w[:, 1], 3., atol=1e-5)

    def test_learned_residual_rotates_back_from_target_heading(self):
        p = history()
        for o in p['objects']:
            for box in o['history']:
                box[6] = np.pi / 2
        weights = np.zeros((50, L.FEATURES, 2))
        weights[:, 0, 0] = 2.
        a = L.forecast(p, weights)['objects'][0]['trajectory']
        b = K.forecast(p)['objects'][0]['trajectory']
        np.testing.assert_allclose(np.asarray(a) - b, np.tile([0., 2.], (50, 1)), atol=1e-10)

    def test_common_target_scores_exclude_new_targets_from_accuracy_delta(self):
        from probe.evaluate_learned_probe import common_arrays
        a = K.forecast(history())['objects'][:1]
        b = copy.deepcopy(a)
        b.append(K.forecast(history())['objects'][1])
        current = np.array([[20., 0., 0., 4., 2., 1.5, 0.]])
        future = {77: np.tile([20., 0.], (50, 1))}
        aa, bb, y, mask = common_arrays(a, b, current, np.array([77]), future)
        self.assertEqual(aa.shape, (1, 50, 2))
        self.assertEqual(D.paired_errors(aa, bb, y, mask)['delta_ADE_b_minus_a'], 0.)


if __name__ == '__main__':
    unittest.main()
