"""A codec must preserve source identities, masks and every ordered mode."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning import inputs


class CompactEvidenceTests(unittest.TestCase):
    def fixture(self):
        path = [[float(i), .2] for i in range(6)]
        obj = dict(source='ego', track_id=7, box=[2., 1., 0., 4., 2., 1.5, .01],
                   score=.85, model_used=True, forecast=[copy.deepcopy(path) for _ in range(6)],
                   forecast_scores=[.11, .1, .09, .08, .07, .06],
                   forecast_times=[.5, 1., 1.5, 2., 2.5, 3.])
        peer = copy.deepcopy(obj)
        peer.update(source='peer:P_local', history=[[2., 1., 0., 4., 2., 1.5, .01]] * 11,
                    history_valid=[False] + [True] * 10,
                    history_times=[i / 10 for i in range(-10, 1)])
        peer['forecast'][5][0][1] = 9.
        return dict(as_of_g=10, coordinate_frame='ego_at_t', objects=[obj, peer],
                    relations=[dict(ego_id=7, peer_id=7, distance_m=0., status='candidate')], queries=[])

    def test_compact_roundtrip_preserves_modes_sources_missing_fields_and_masks(self):
        self.assertTrue(hasattr(inputs, 'pack_evidence'), 'compact codec is not implemented')
        from planning.inputs import pack_evidence, unpack_evidence
        evidence = self.fixture()
        before = copy.deepcopy(evidence)
        packed = pack_evidence(evidence)
        self.assertEqual(unpack_evidence(packed), before)
        self.assertEqual(evidence, before)
        self.assertLess(len(str(packed)), len(str(evidence)))
        # Altering a unique mode must survive; equal IDs never merge sources.
        restored = unpack_evidence(packed)
        self.assertEqual(len(restored['objects']), 2)
        self.assertEqual(restored['objects'][1]['forecast'][5][0], [0., 9.])
        self.assertEqual(restored['objects'][0]['forecast_scores'], [.11, .1, .09, .08, .07, .06])

    def test_compact_does_not_round_or_merge_nearly_equal_modes(self):
        self.assertTrue(hasattr(inputs, 'pack_evidence'), 'compact codec is not implemented')
        from planning.inputs import pack_evidence, unpack_evidence
        evidence = self.fixture()
        for i in range(6):
            evidence['objects'][0]['forecast'][i][0][0] = i * 1e-7
        restored = unpack_evidence(pack_evidence(evidence))
        self.assertEqual(restored, evidence)
        self.assertEqual(len({p[0][0] for p in restored['objects'][0]['forecast']}), 6)

    def test_compact_rejects_unknown_gt_fields_and_invalid_path_reference(self):
        self.assertTrue(hasattr(inputs, 'pack_evidence'), 'compact codec is not implemented')
        from planning.inputs import pack_evidence, unpack_evidence
        evidence = self.fixture()
        evidence['objects'][0]['gt_future'] = [1., 2.]
        with self.assertRaises(ValueError):
            pack_evidence(evidence)
        del evidence['objects'][0]['gt_future']
        packed = pack_evidence(evidence)
        column = packed['object_columns'].index('forecast')
        packed['object_rows'][0][column]['mode_to_path'][0] = 99
        with self.assertRaises(ValueError):
            unpack_evidence(packed)

    def test_prompt_uses_decodable_compact_data_and_keeps_native_task_last(self):
        evidence = self.fixture()
        self.assertTrue(hasattr(inputs, 'pack_evidence'), 'compact codec is not implemented')
        prompt = inputs.make_prompt('Q8', dict(speed_mps=4., yaw_rate_rps=0.), evidence,
                                    evidence_format='compact')
        self.assertIn('mode_to_path', prompt)
        self.assertTrue(prompt.endswith('What are the suggested speed and steering settings to avoid collision with nearby objects?'))


if __name__ == '__main__':
    unittest.main()
