"""Breaks on peer/future reads, wrong axes, silent missing data or invalid answers."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning import inputs
from planning.inputs import load_ego_features, make_prompt, parse_q8, parse_q9


class PlanningInputTests(unittest.TestCase):
    def write_frame(self, root, g, value):
        path = Path(root) / 'no_fusion_keep_all/npy/co_llm/ego'
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / ('%04d_regression_map.npy' % g), np.full((1, 14, 50, 88), value, np.float32))
        np.save(path / ('%04d_classification_map.npy' % g), np.full((1, 2, 50, 88), value, np.float32))
        np.save(path / ('%04d_detection_box_score.npy' % g), np.array([[1.5, 2., 4., 10., 3., 20., .2, .9]], np.float32))

    def test_only_ego_past_and_current_are_loaded_with_original_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_frame(directory, 9, 9.)
            self.write_frame(directory, 10, 10.)
            f = load_ego_features(directory, 'test', 10)
            self.assertEqual(f['frame_indices'], [10, 9])
            self.assertEqual(f['regression_map'].shape, (1, 2, 1, 14, 50, 88))
            self.assertEqual(f['regression_map'][0, 1, 0, 0, 0, 0], 9.)
            np.testing.assert_allclose(f['detection_box_score'][0, 0, 0, 0], [1.5, 2., 4., 10., 20., 3., .2, .9])
            self.assertEqual(f['active_agent_mask'].shape, (1, 2, 1))
            self.assertTrue(f['active_agent_mask'].all())
            self.assertEqual(len(f['read_paths']), 6)
            self.assertTrue(all('/co_llm/ego/' in p for p in f['read_paths']))

    def test_boundary_does_not_copy_current_frame_into_missing_past(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_frame(directory, 147, 2.)
            f = load_ego_features(directory, 'test', 147)
            self.assertEqual(f['frame_indices'], [147, None])
            self.assertEqual(f['active_agent_mask'].tolist(), [[[True], [False]]])
            self.assertEqual(len(f['read_paths']), 3)

    def test_missing_real_frame_and_invalid_index_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_frame(directory, 10, 1.)
            with self.assertRaises(FileNotFoundError):
                load_ego_features(directory, 'test', 10)
            with self.assertRaises(ValueError):
                load_ego_features(directory, 'test', -1)

    def test_prompt_requires_generated_q8_and_rejects_extra_state_fields(self):
        state = {'speed_mps': 4., 'yaw_rate_rps': .1}
        p8 = make_prompt('Q8', state, {'objects': [], 'as_of_g': 10})
        self.assertIn('speed', p8)
        with self.assertRaises(ValueError):
            make_prompt('Q9', state, {'objects': [], 'as_of_g': 10})
        with self.assertRaises(ValueError):
            make_prompt('Q8', dict(state, gt_future=[1, 2]), {'objects': [], 'as_of_g': 10})
        q8 = 'The suggested speed setting is: slow. The suggested steering setting is: straight.'
        p9 = make_prompt('Q9', state, {'objects': [], 'as_of_g': 10}, q8_answer=q8)
        self.assertIn(q8, p9)

    def test_native_answer_parsing_rejects_ambiguity_and_wrong_length(self):
        q8 = 'The suggested speed setting is: very slow. The suggested steering setting is: slightly left.'
        self.assertEqual(parse_q8(q8), {'speed': 'very slow', 'steering': 'slightly left'})
        with self.assertRaises(ValueError):
            parse_q8(q8 + ' The suggested speed setting is: fast.')
        points = [(float(i), -.5) for i in range(1, 7)]
        np.testing.assert_allclose(parse_q9('The suggested trajectory is ' + str(points) + '.'), points)
        with self.assertRaises(ValueError):
            parse_q9('The suggested trajectory is [(1, 2)].')
        with self.assertRaises(ValueError):
            parse_q9('The suggested trajectory is [(1e999, 0)] * 6.')

    def test_task_question_follows_evidence_without_copyable_answer_placeholders(self):
        # Real checkpoint copied <speed> or answered object motion when the new
        # evidence followed the operative task. Preserve the native task suffix.
        state = {'speed_mps': 4., 'yaw_rate_rps': .1}
        evidence = {'objects': [], 'as_of_g': 10}
        q8 = 'The suggested speed setting is: slow. The suggested steering setting is: straight.'
        p8 = make_prompt('Q8', state, evidence)
        p9 = make_prompt('Q9', state, evidence, q8)
        self.assertTrue(p8.endswith('I am CAV_EGO at [0.0, 0.0]. What are the suggested speed and steering settings to avoid collision with nearby objects?'))
        self.assertTrue(p9.endswith('I am CAV_EGO at [0.0, 0.0]. What is the suggested future trajectory to avoid collision with nearby objects?'))
        self.assertNotIn('<speed>', p8)
        self.assertNotIn('x1,y1', p9)
        self.assertLess(p9.index(q8), p9.rindex('What is the suggested future trajectory'))

    def test_ego_motion_uses_past_localization_without_future_pose(self):
        self.assertTrue(hasattr(inputs, 'load_ego_motion'), 'causal motion loader not implemented')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'no_fusion_keep_all/npy/ego'
            path.mkdir(parents=True)
            before, now = np.eye(4), np.eye(4)
            now[0, 3] = .4
            now[:2, :2] = [[np.cos(.1), -np.sin(.1)], [np.sin(.1), np.cos(.1)]]
            np.save(path / '0009_lidar_pose.npy', before)
            np.save(path / '0010_lidar_pose.npy', now)
            state, paths = inputs.load_ego_motion(directory, 'test', 10)
            self.assertAlmostEqual(state['speed_mps'], 4.)
            self.assertAlmostEqual(state['yaw_rate_rps'], 1.)
            self.assertEqual(len(paths), 2)


if __name__ == '__main__':
    unittest.main()
