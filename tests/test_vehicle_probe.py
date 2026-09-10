"""Behavior checks for the vehicle-only kinematic interface probe."""
import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from probe import kinematic_tools as K


def window(source, positions, ids=None):
    n = len(positions)
    state = np.zeros((n, 11, 7), dtype=np.float32)
    for i, xy in enumerate(positions):
        state[i, :, :2] = xy
        state[i, :, 3:6] = [4., 2., 1.5]
    return {'source': source, 'g': 10, 'states': state,
            'track_ids': np.array(list(range(n)) if ids is None else ids),
            'valid': np.ones((n, 11), dtype=bool),
            'scores': np.full((n, 11), .8),
            'time_seconds': np.arange(-10, 1) * .1}


class VehicleProbeTests(unittest.TestCase):
    def test_offline_score_on_hand_checked_stationary_target(self):
        from pathlib import Path
        from probe import evaluate_vehicle_probe as E
        truth = np.array([[20., 0., 0., 4., 2., 1.5, 0.]])
        ref = np.column_stack([np.arange(1, 31) * .1, np.zeros(30)])
        prediction = K.forecast(K.history_packet(window('ego', [(20, 0)])))
        objects, audit = K.receive(prediction)
        result = {'objects': objects, 'association': audit, 'plan': K.choose_speed(ref, objects),
                  'request_bytes': 0, 'response_bytes': 0}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'run.json').write_text(json.dumps({'frames': [10], 'scene': 'fixture', 'source_split': 'test'}))
            (p / 'decisions.jsonl').write_text(json.dumps({'t': 10, 'g': 10, 'scene': 'fixture',
                'actions': {k: copy.deepcopy(result) for k in ['ego', 'P', 'F', 'PF']}}) + '\n')
            with patch.object(E.M, 'load_gt', return_value=(truth, np.array([7]))), \
                    patch.object(E.M, 'load_pose', return_value=np.eye(4)), \
                    patch.object(E.M, 'seq_of', side_effect=lambda g, split: (0, g, 0, 100)), \
                    patch.object(E.M, 'ego_future_waypoints', return_value=ref), patch('builtins.print'):
                scores = E.evaluate(p)['actions']['ego']
            self.assertEqual(scores['mean_target_ADE_observed'], 0.)
            self.assertEqual(scores['ego_L2_observed_3s'], 0.)
            self.assertEqual(scores['observed_proximity_3s'], 0.)
            self.assertEqual(scores['matched_at_t'], 1.)

    def test_reference_uses_only_past_ego_poses(self):
        from probe.run_vehicle_probe import causal_reference
        def pose(split, g):
            self.assertLessEqual(g, 10)
            result = np.eye(4)
            result[0, 3] = float(g)
            return result
        with patch('probe.run_vehicle_probe.M.load_pose', side_effect=pose):
            reference = causal_reference('train', 10)
        np.testing.assert_allclose(reference[9], [10., 0.], atol=1e-6)

    def test_known_velocity_and_short_history_fallback(self):
        w = window('ego', [(10, 0), (20, 0)])
        w['states'][0, :, 0] = np.arange(11)
        w['valid'][1, :-1] = False
        records = K.forecast(K.history_packet(w))['objects']
        np.testing.assert_allclose(records[0]['trajectory'][9], [20, 0], atol=1e-6)
        self.assertEqual(records[1]['fallback'], 'stationary_short_history')
        np.testing.assert_allclose(records[1]['trajectory'][-1], [20, 0])

    def test_history_transferred_to_local_matches_remote_compute(self):
        w = window('peer', [(15, 3)])
        w['states'][0, :, 0] += np.arange(11) * .3
        history = K.history_packet(w)
        local = K.forecast(json.loads(K.encode(history)))
        remote = json.loads(K.encode(K.forecast(history)))
        self.assertEqual(local, remote)

    def test_id_collision_does_not_remove_a_remote_only_target(self):
        ego = K.forecast(K.history_packet(window('ego', [(10, 0)], [7])))
        peer = K.forecast(K.history_packet(window('peer', [(30, 0)], [7])))
        merged, audit = K.receive(ego, remote_f=peer)
        self.assertEqual(len(merged), 2)
        self.assertEqual(audit['peer_only'], 1)

    def test_one_to_one_geometry_and_pf_deduplication(self):
        ego = K.forecast(K.history_packet(window('ego', [(10, 0)], [5])))
        p = K.history_packet(window('peer', [(10.5, 0), (25, 0)], [8, 9]))
        f = K.forecast(p)
        p_out, _ = K.receive(ego, remote_p=p)
        f_out, _ = K.receive(ego, remote_f=f)
        pf_out, audit = K.receive(ego, remote_p=p, remote_f=f)
        self.assertEqual(p_out, f_out)
        self.assertEqual(f_out, pf_out)
        self.assertEqual(len(pf_out), 2)
        self.assertEqual(audit['matched'], 1)

    def test_only_ego_footprint_is_filtered(self):
        ego = K.forecast(K.history_packet(window('ego', [(0, 0), (10, 0)])))
        merged, audit = K.receive(ego)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['state'][0], 10)
        self.assertEqual(audit['ego_filtered'], 1)

    def test_empty_packet_is_valid_but_stale_or_nan_is_rejected(self):
        ego = K.forecast(K.history_packet(window('ego', [])))
        peer = K.forecast(K.history_packet(window('peer', [])))
        self.assertEqual(K.receive(ego, remote_f=peer)[0], [])
        peer['g'] = 11
        with self.assertRaises(ValueError):
            K.receive(ego, remote_f=peer)
        bad = window('ego', [(10, 0)])
        bad['states'][0, -1, 0] = np.nan
        with self.assertRaises(ValueError):
            K.history_packet(bad)

    def test_stop_candidate_is_marked_infeasible_if_all_candidates_conflict(self):
        ref = np.column_stack([np.arange(1, 31) * .1, np.zeros(30)])
        obstacle = {'trajectory': np.zeros((50, 2)).tolist()}
        plan = K.choose_speed(ref, [obstacle])
        self.assertEqual(plan['scale'], 0.)
        self.assertFalse(plan['feasible'])

    def test_ego_result_is_invariant_to_peer_contents(self):
        ego = K.forecast(K.history_packet(window('ego', [(10, 0)])))
        before = copy.deepcopy(K.receive(ego))
        K.receive(ego, remote_f=K.forecast(K.history_packet(window('peer', [(10, 1)]))))
        self.assertEqual(before, K.receive(ego))


if __name__ == '__main__':
    unittest.main()
