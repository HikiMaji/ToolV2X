"""Audit sampling and coverage must not manufacture cooperative coverage."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('evidence_audit',
    Path(__file__).resolve().parents[1] / 'scripts/audit_training_evidence.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_training_evidence import check_window
import report_training_evidence as report


class EvidenceAuditTests(unittest.TestCase):
    def test_fixed_unique_group_quota_and_no_downstream_fields(self):
        rows = [dict(sample_id='%s:%s' % (group, i), scene=group, recording=group,
                     local_frame=i, g=i, ego_count=i % 3, future_label='forbidden')
                for group in ('a', 'b') for i in range(19)]
        selected = audit.select_frames(rows)
        self.assertEqual(selected, audit.select_frames(list(reversed(rows))))
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({r['sample_id'] for r in selected}), 16)
        for group in ('a', 'b'):
            chosen = [r for r in selected if r['recording'] == group]
            self.assertEqual(len(chosen), 8)
            self.assertEqual(sum(r['selection_reason'].startswith('time') for r in chosen), 4)
        self.assertTrue(all('future_label' not in r for r in selected))

    def test_insufficient_group_is_not_silently_replaced(self):
        rows = [dict(sample_id=str(i), scene='a', recording='a', local_frame=i,
                     g=i, ego_count=2) for i in range(7)]
        with self.assertRaises(ValueError):
            audit.select_frames(rows)

    def test_pf_deduplicates_peer_ids_but_preserves_capability_records(self):
        def obj(source, track, x=10):
            return dict(source=source, track_id=track, box=[x, 0, 0, 4, 2, 1, 0])
        full = dict(objects=[obj('ego', 1), obj('peer:P_local', 1), obj('peer:F', 1),
                            obj('peer:P_local', 2), obj('peer:F', 2, 80)],
                    relations=[dict(ego_id=1, peer_id=1)])
        # The receiver dropped the relation. This does not make peer 1 novel.
        used = dict(objects=full['objects'][1:4], relations=[])
        counts = audit.coverage(full, used)
        self.assertEqual(counts['remote_records_total'], 4)
        self.assertEqual(counts['remote_records_retained'], 3)
        self.assertEqual(counts['remote_unique_total'], 2)
        self.assertEqual(counts['remote_unique_retained'], 2)
        self.assertEqual(counts['remote_unassociated_total'], 1)
        self.assertEqual(counts['remote_unassociated_retained'], 1)
        self.assertEqual(counts['ego_retained'], 0)

    def test_independent_history_check_detects_wrong_reference_frame(self):
        pose = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1.]])
        world = np.array([[7, 8, 23, 1, 4, 2, 1, np.pi / 2 + .2, .9]])
        frames = {g: world for g in range(11)}
        window = dict(track_ids=np.array([7]), valid=np.ones((1, 11), dtype=bool),
            scores=np.full((1, 11), .9), time_seconds=np.arange(-10, 1) / 10,
            states=np.tile([3., 2., 1., 4., 2., 1., .2], (1, 11, 1)))
        self.assertTrue(check_window(window, frames, 10, pose)['within_tolerance'])
        window['states'][0, 3, 0] += 1
        self.assertFalse(check_window(window, frames, 10, pose)['within_tolerance'])

    def test_history_check_rejects_future_timestamp(self):
        window = dict(track_ids=np.array([], dtype=int), time_seconds=np.arange(-10, 1) / 10)
        window['time_seconds'][-1] = .1
        with self.assertRaises(AssertionError):
            check_window(window, {}, 10, np.eye(4))

    def test_near_stationary_position_noise_is_not_a_turn(self):
        points = np.array([[.01, 0], [0, .01], [-.01, 0], [0, -.01], [.01, 0], [0, .01]])
        result = report.motion_geometry(points)
        self.assertFalse(result['path_heading_reliable'])
        self.assertIsNone(result['heading_change_deg'])
        moving = report.motion_geometry(np.array([[2, 0], [4, .2], [6, 1], [7, 2], [8, 4], [8, 6]]))
        self.assertTrue(moving['path_heading_reliable'])
        self.assertGreater(moving['heading_change_deg'], 45)


if __name__ == '__main__':
    unittest.main()
