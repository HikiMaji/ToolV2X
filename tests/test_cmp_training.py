"""A real original MTR loss must backpropagate into encoder and decoder."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction import cmp_adapter as C
from prediction.supervision import make_labels, training_batch
from test_vehicle_tools import window


class CMPTrainingTests(unittest.TestCase):
    def test_original_loss_updates_real_parameters_and_batching_preserves_predictions(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        try:
            model, info = C.load_model(device='cpu')
        except TypeError as exc:
            self.fail('original model needs explicit training device: ' + str(exc))
        w = window()
        boxes = w['states'][:, -1].copy()
        future = boxes.copy()
        future[:, 0] += 3.
        labels = make_labels(w, boxes, np.array([100, 200]), [(future, np.array([100, 200]))] * 50)
        model.train()
        parameter = model.motion_decoder.motion_reg_heads[-1][-1].weight
        before = parameter.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        _, loss, logged, _ = model(training_batch(w, labels, [0, 1]))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('loss_dense_prediction', logged)
        loss.backward()
        for module in (model.context_encoder, model.motion_decoder):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        optimizer.step()
        self.assertGreater(float((parameter.detach() - before).abs().max()), 0.)
        model.eval()
        single = C.predict(model, w)
        batched = C.predict(model, w, batch_size=2)
        np.testing.assert_allclose(single['means'], batched['means'], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(single['scores'], batched['scores'], atol=1e-5, rtol=1e-4)
        self.assertEqual(info['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
