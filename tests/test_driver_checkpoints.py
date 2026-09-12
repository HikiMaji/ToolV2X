"""A failed save must not advertise a complete checkpoint or remove recovery state."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)),
            LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss.backward()
        optimizer.step()
        tokenizer = SimpleNamespace(save_pretrained=lambda p: (p/'tokenizer_config.json').write_text('{}'))
        state = dict(epoch=0, next_row=8, steps=1, examples_seen=8, best_checkpoint=None)
        return SimpleNamespace(model=model, tokenizer=tokenizer), optimizer, state

    def test_failed_write_never_publishes_checkpoint(self):
        from planning import train_driver as T
        planner, optimizer, state = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'checkpoint'
            with patch('torch.save', side_effect=OSError('injected disk failure')):
                with self.assertRaises(OSError):
                    T.save_checkpoint(planner, optimizer, destination, state)
            self.assertFalse(destination.exists(), 'partial weights must not appear as a completed checkpoint')

    def test_recovery_rotation_keeps_two_complete_states_and_epoch_checkpoint(self):
        import torch
        from planning import train_driver as T
        self.assertTrue(hasattr(T, 'save_recovery'), 'periodic complete-state save missing')
        planner, optimizer, state = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            epoch = out/'checkpoint-epoch01'
            epoch.mkdir()
            for step in [1, 25, 50]:
                state.update(steps=step, next_row=8*step, examples_seen=8*step)
                T.save_recovery(planner, optimizer, out, state)
            self.assertEqual(sorted(p.name for p in out.glob('recovery-step*')),
                             ['recovery-step000025', 'recovery-step000050'])
            self.assertTrue(epoch.exists())
            latest = json.loads((out/'latest_checkpoint.json').read_text())
            checkpoint = Path(latest['checkpoint'])
            saved = torch.load(checkpoint/'training_state.pt', map_location='cpu')
            self.assertEqual(saved['state'], state)
            self.assertEqual(saved['optimizer']['param_groups'], optimizer.state_dict()['param_groups'])
            self.assertTrue(all(k in saved for k in ['torch_rng','cuda_rng','python_rng','numpy_rng']))
            state.update(steps=75)
            with patch('torch.save', side_effect=OSError('injected disk failure')):
                with self.assertRaises(OSError):
                    T.save_recovery(planner, optimizer, out, state)
            self.assertEqual(json.loads((out/'latest_checkpoint.json').read_text()), latest)
            self.assertTrue(checkpoint.exists())

    def test_mid_epoch_order_resumes_remaining_examples_including_short_batch(self):
        from planning.train_driver import epoch_order
        rows = [dict(sample_id=str(i//5)) for i in range(37)]
        full = epoch_order(rows, 20, 0)
        self.assertEqual(full[24:], epoch_order(rows, 20, 0, 24))
        self.assertEqual([], epoch_order(rows, 20, 0, len(rows)))
        self.assertEqual(len(full), len(set(full)))


if __name__ == '__main__':
    unittest.main()
