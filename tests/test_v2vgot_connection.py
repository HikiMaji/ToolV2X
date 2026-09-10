"""Reference checks against the original V2V-GoT projector, not a toy planner."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.v2vgot import load_projector, model_features, project_ego_features


class V2VGoTConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.projector, cls.provenance = load_projector(device='cpu')

    def fixture(self):
        return {'regression_map': np.zeros((1, 2, 1, 14, 50, 88), np.float32),
                'classification_map': np.zeros((1, 2, 1, 2, 50, 88), np.float32),
                'detection_box_score': np.zeros((1, 2, 1, 50, 8), np.float32),
                'active_agent_mask': np.ones((1, 2, 1), bool)}

    def test_real_original_projector_uses_current_then_past_ego_features(self):
        f = self.fixture()
        f['regression_map'][0, 0] = .2
        f['regression_map'][0, 1] = -.1
        f['detection_box_score'][0, 0, 0, 0, :4] = [1.5, 2., 4., 10.]
        got = project_ego_features(self.projector, f)
        from planning.v2vgot import MODEL_CONFIG
        tensors = model_features(f, 'cpu', torch.float32)
        with torch.inference_mode():
            scene = self.projector.generate_scene_level_features(MODEL_CONFIG, None, tensors['regression_map'], tensors['classification_map'], tensors['active_agent_mask'])
            objects = self.projector.generate_object_level_features(MODEL_CONFIG, tensors['detection_box_score'], tensors['object_features'])
        self.assertEqual(tuple(got.shape), (1, 540, 4096))
        torch.testing.assert_close(got[:, :220], scene[:, 0, 0])
        torch.testing.assert_close(got[:, 220:270], objects[:, 0, 0])
        torch.testing.assert_close(got[:, 270:490], scene[:, 1, 0])
        self.assertGreater(float((got[:, :220] - got[:, 270:490]).abs().max()), 1e-5)
        self.assertEqual(self.provenance['missing_keys'], [])
        self.assertEqual(self.provenance['unexpected_keys'], [])

    def test_missing_history_tokens_are_omitted_and_peer_features_rejected(self):
        f = self.fixture()
        f['active_agent_mask'][0, 1, 0] = False
        got = project_ego_features(self.projector, f)
        self.assertEqual(tuple(got.shape), (1, 270, 4096))
        f['regression_map'] = np.zeros((1, 2, 2, 14, 50, 88), np.float32)
        with self.assertRaises(ValueError):
            model_features(f, 'cpu', torch.float32)

    def test_context_selection_reports_drops_and_reserves_space_for_both_tasks(self):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT, fit_evidence
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        objects = [dict(source='peer:F', track_id=i, box=[float(i + 1), 0, 0, 4, 2, 1.5, 0],
            score=.9, forecast=[[[float(i + 1), 0.] for _ in range(6)] for _ in range(6)],
            forecast_scores=[1 / 6] * 6, forecast_times=[.5, 1., 1.5, 2., 2.5, 3.], model_used=True)
            for i in range(12)]
        evidence = dict(as_of_g=10, coordinate_frame='ego_at_t', objects=objects)
        view, report = fit_evidence(tokenizer, dict(speed_mps=4., yaw_rate_rps=0.), evidence, 270, 1536)
        self.assertGreater(len(view['objects']), 0)
        self.assertGreater(len(report['dropped_objects']), 0)
        self.assertEqual(view['objects'][0]['track_id'], 0)
        self.assertEqual(len(evidence['objects']), 12)
        self.assertLessEqual(report['input_token_bound'] + report['reserved_generation_tokens'], 1536)

    def test_context_relations_do_not_confuse_equal_ids_from_different_sources(self):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT, fit_evidence
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        evidence = dict(as_of_g=10, objects=[
            dict(source='peer:F', track_id=7, box=[1., 0, 0, 4, 2, 1.5, 0]),
            dict(source='ego', track_id=9, box=[2., 0, 0, 4, 2, 1.5, 0])],
            relations=[dict(ego_id=7, peer_id=9, distance_m=1., status='candidate')])
        view, _ = fit_evidence(tokenizer, dict(speed_mps=4., yaw_rate_rps=0.), evidence, 270, 1536)
        self.assertEqual(len(view['objects']), 2)
        self.assertEqual(view.get('relations', []), [])

    def test_compact_budget_admits_more_records_without_changing_retained_values(self):
        from transformers import AutoTokenizer
        from planning.v2vgot import CHECKPOINT, fit_evidence, rounded
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
        objects = [dict(source='peer:F', track_id=i, box=[float(i + 1), 0, 0, 4, 2, 1.5, 0],
            score=.9, forecast=[[[float(i + 1), 0.] for _ in range(6)] for _ in range(6)],
            forecast_scores=[1 / 6] * 6, forecast_times=[.5, 1., 1.5, 2., 2.5, 3.], model_used=True)
            for i in range(12)]
        evidence = dict(as_of_g=10, coordinate_frame='ego_at_t', objects=objects)
        state = dict(speed_mps=4., yaw_rate_rps=0.)
        legacy, _ = fit_evidence(tokenizer, state, evidence, 270, 1536)
        compact, report = fit_evidence(tokenizer, state, evidence, 270, 1536, evidence_format='compact')
        self.assertGreater(len(compact['objects']), len(legacy['objects']))
        for obj in compact['objects']:
            self.assertEqual(obj, rounded(objects[obj['track_id']]))
        self.assertLessEqual(report['input_token_bound'] + 256, 1536)


if __name__ == '__main__':
    unittest.main()
