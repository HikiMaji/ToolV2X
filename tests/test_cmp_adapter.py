"""Checks for adapting CMP's existing MTR, including original-kernel semantics."""
import sys
import ast
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction import cmp_adapter as C


class CMPAdapterTests(unittest.TestCase):
    def window(self):
        states = np.zeros((3, 11, 7), np.float32)
        states[:, :, 3:6] = [4., 2., 1.5]
        states[:, :, 0] = np.arange(11)[None] + np.array([0., 20., 40.])[:, None]
        states[0, :, 6] = np.pi / 2
        mask = np.ones((3, 11), bool)
        mask[1, 3] = False
        mask[2, :-1] = False
        return dict(states=states, valid=mask, track_ids=np.array([11, 22, 33]),
                    source='peer', g=10, time_seconds=np.arange(-10, 1) / 10.,
                    scores=np.ones((3, 11), np.float32))

    def test_centered_features_keep_context_and_mask_without_future_fields(self):
        batch = C.make_batch(self.window(), [0])
        inp = batch['input_dict']
        self.assertEqual(tuple(inp['obj_trajs'].shape), (1, 3, 11, 22))
        np.testing.assert_allclose(inp['obj_trajs'][0, 1, -1, :2], [0., -20.], atol=2e-6)
        self.assertEqual(float(inp['obj_trajs'][0, 1, 3].abs().sum()), 0.)
        np.testing.assert_allclose(inp['obj_trajs'][0, 0, :, 19], np.arange(11) / 10.)
        self.assertFalse(any('gt' in key or 'future' in key for key in inp))
        self.assertEqual(inp['track_index_to_predict'].tolist(), [0])

    def test_portable_attention_handles_batch_offsets_padding_and_gradients(self):
        C.add_vendor_path()
        from mtr.ops.attention import attention_torch as A
        counts = torch.tensor([2, 1], dtype=torch.int32)
        batches = torch.tensor([0, 0, 1], dtype=torch.int32)
        indices = torch.tensor([[1, -1], [0, 1], [0, -1]], dtype=torch.int32)
        q = torch.tensor([[[1., 2.]], [[2., 0.]], [[3., 1.]]], requires_grad=True)
        k = torch.tensor([[[1., 0.]], [[0., 1.]], [[2., 2.]]], requires_grad=True)
        score = A.attention_weight_computation(counts, counts, batches, indices, q, k)
        np.testing.assert_allclose(score.detach().numpy().squeeze(-1), [[2., 0.], [2., 0.], [8., 0.]])
        result = A.attention_value_computation(counts, counts, batches, indices, torch.ones_like(score), k)
        np.testing.assert_allclose(result.detach().numpy().squeeze(1), [[0., 1.], [1., 1.], [2., 2.]])
        (score.sum() + result.sum()).backward()
        self.assertTrue(torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all())

    def test_knn_does_not_cross_sources_or_fill_missing_neighbors_with_zero(self):
        C.add_vendor_path()
        from mtr.ops.knn.knn_utils import knn_batch_mlogk
        xyz = torch.tensor([[0., 0., 0.], [10., 0., 0.], [0., 0., 0.]])
        got = knn_batch_mlogk(xyz, xyz, torch.tensor([0, 0, 1]), torch.tensor([0, 2, 3]), 4)
        self.assertEqual(set(got[0].tolist()), {0, 1, -1})
        self.assertEqual(got[2].tolist().count(-1), 3)
        self.assertEqual(set(got[2].tolist()) - {-1}, {0})

    def test_feature_layout_matches_original_cmp_helper_with_corrected_timestamps(self):
        C.add_vendor_path()
        from mtr.utils import common_utils
        source = C.VENDOR / 'reference/v2v4real_multiego_dataset.py'
        tree = ast.parse(source.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'V2V4RealMultiEgoDataset')
        original.bases = []  # Static helpers do not need the heavy original dataset constructor.
        original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in
                         ('transform_trajs_to_center_coords', 'generate_centered_trajs_for_agents')]
        module = ast.Module(body=[original])
        module.type_ignores = []
        namespace = dict(np=np, torch=torch, common_utils=common_utils)
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        w = self.window()
        past = np.concatenate([w['states'], w['valid'][..., None]], -1).astype(np.float32)
        # Dummy labels only satisfy the original helper signature in this reference test.
        reference = namespace['V2V4RealMultiEgoDataset'].generate_centered_trajs_for_agents(
            past[[0, 1], -1], past, np.array(['TYPE_VEHICLE'] * 3), np.array([0, 1]), 0,
            np.arange(11, dtype=np.float32) * .1, np.zeros((3, 50, 8), np.float32))
        actual = C.make_batch(w, [0, 1])['input_dict']
        np.testing.assert_allclose(actual['obj_trajs'], reference[0], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(actual['obj_trajs_mask'], reference[1].astype(bool))


if __name__ == '__main__':
    unittest.main()
