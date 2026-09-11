"""Saved packet equality cannot be inferred from the receiver's evidence view."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning import compare_evidence as comparison
from probe.kinematic_tools import encode
from test_vehicle_tools import fixture_prediction, window
from tools.vehicle import VehicleTools, make_evidence


class WireComparisonTests(unittest.TestCase):
    def compare(self, baseline, expanded, action):
        check = getattr(comparison, 'compare_saved_wire', None)
        self.assertTrue(callable(check), 'comparison needs an actual saved-packet byte check')
        return check(baseline, expanded, action)

    def test_equal_evidence_and_equal_sizes_do_not_hide_different_wire(self):
        response = VehicleTools(lambda: window(), fixture_prediction, 'scene', 10, provider='peer').query('P')
        first = json.loads(response['wire'])
        first['objects'][0]['history_scores'][0] = .8
        second = copy.deepcopy(first)
        second['objects'][0]['history_scores'][0] = .9
        left, right = encode(first), encode(second)
        self.assertEqual(len(left), len(right))
        self.assertNotEqual(left, right)
        local = window('ego')
        self.assertEqual(make_evidence(local, fixture_prediction(local), [first], fixture_prediction),
                         make_evidence(local, fixture_prediction(local), [second], fixture_prediction))
        with tempfile.TemporaryDirectory() as directory:
            baseline, expanded = Path(directory) / 'baseline', Path(directory) / 'expanded'
            for path, wire in ((baseline, left), (expanded, right)):
                path.mkdir()
                (path / 'P_request.json').write_bytes(encode(response['request']))
                (path / 'P_response.json').write_bytes(wire)
            result = self.compare(baseline, expanded, 'P')
            self.assertTrue(result['checked'])
            self.assertFalse(result['equal'])
            self.assertEqual(result['different_files'], ['P_response.json'])
            (expanded / 'P_response.json').write_bytes(left)
            changed = dict(response['request'], provider='another_peer')
            (expanded / 'P_request.json').write_bytes(encode(changed))
            result = self.compare(baseline, expanded, 'P')
            self.assertFalse(result['equal'])
            self.assertEqual(result['different_files'], ['P_request.json'])

    def test_missing_packets_are_unchecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = self.compare(path, path / 'missing', 'PF')
            self.assertFalse(result['checked'])
            self.assertIsNone(result['equal'])
            self.assertTrue(result['missing_files'])
            self.assertEqual(result['compared_files'], [])

    def test_archived_wire_equality_is_checked_for_every_requested_tool(self):
        baseline = ROOT / 'outputs/framework_connection_v3'
        expanded = ROOT / 'outputs/framework_compact_v1'
        for action, count in (('Ego', 0), ('P', 2), ('F', 2), ('PF', 4)):
            with self.subTest(action=action):
                result = self.compare(baseline / action, expanded / action, action)
                self.assertTrue(result['checked'])
                self.assertTrue(result['equal'])
                self.assertEqual(len(result['compared_files']), count)
                self.assertEqual(result['missing_files'], [])
                self.assertEqual(result['different_files'], [])


if __name__ == '__main__':
    unittest.main()
