"""Loader regression: directory naming cannot choose a different model branch."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class TrainablePlannerTests(unittest.TestCase):
    def test_training_retains_adapter_and_inference_remains_default(self):
        from planning import v2vgot as V
        from llava.model import builder
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096,
                                                        tokenizer_model_max_length=2048), eval=lambda: None)
        with patch.object(builder, 'load_pretrained_model', return_value=(None, model, None, 2048)) as loader, \
                patch.object(V, 'local_adapter', return_value=Path('/candidate/custom_directory')):
            V.V2VGoTPlanner()
            self.assertIn('llava', loader.call_args.args[2])
            self.assertIn('lora', loader.call_args.args[2])
            self.assertFalse(loader.call_args.kwargs.get('trainable_lora', False))
            self.assertEqual(loader.call_args.kwargs['attn_implementation'], 'sdpa')
            try:
                V.V2VGoTPlanner(trainable_lora=True)
            except TypeError as exc:
                self.fail('native LoRA training load is unavailable: ' + str(exc))
            self.assertTrue(loader.call_args.kwargs['trainable_lora'])
            self.assertEqual(loader.call_args.kwargs['attn_implementation'], 'flash_attention_2')


if __name__ == '__main__':
    unittest.main()
