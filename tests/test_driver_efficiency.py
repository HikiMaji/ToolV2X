"""Loss optimization must preserve context, causal shift and LoRA gradients."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class AnswerHeadTests(unittest.TestCase):
    def make_model(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(23)
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=97, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=64)), LoraConfig(r=4,
            lora_alpha=8, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'],
            task_type='CAUSAL_LM', init_lora_weights=False))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        return model

    def test_loss_and_all_gradients_match_with_dropout_and_causal_mask(self):
        import torch
        from planning import train_driver as T
        self.assertTrue(hasattr(T, 'answer_only_loss'), 'answer-only loss is not implemented')
        model = self.make_model()
        embeddings = torch.randn(2, 17, 32, requires_grad=True)
        labels = torch.full((2, 17), -100, dtype=torch.long)
        labels[0, [5, 8, 9, 16]] = torch.tensor([4, 9, 20, 2])
        labels[1, 12:] = torch.tensor([3, 4, 5, 7, 2])
        attention = torch.ones(2, 17, dtype=torch.long)
        kwargs = dict(inputs_embeds=embeddings, attention_mask=attention, position_ids=None, labels=labels)
        torch.manual_seed(45)
        reference = model(**kwargs, use_cache=False).loss
        reference.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        input_gradient = embeddings.grad.clone()
        model.zero_grad(set_to_none=True)
        embeddings.grad = None
        torch.manual_seed(45)
        actual = T.answer_only_loss(model, **kwargs)
        actual.backward()
        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(embeddings.grad, input_gradient, rtol=2e-5, atol=2e-7)
        self.assertGreater(float(embeddings.grad[:, :5].abs().sum()), 0., 'context must still receive gradients')

    def test_empty_shifted_supervision_is_rejected(self):
        import torch
        from planning import train_driver as T
        self.assertTrue(hasattr(T, 'answer_only_loss'), 'answer-only loss is not implemented')
        model = self.make_model()
        labels = torch.full((1, 4), -100, dtype=torch.long)
        labels[0, 0] = 2  # No preceding position can predict token zero.
        with self.assertRaisesRegex(ValueError, 'supervis'):
            T.answer_only_loss(model, torch.randn(1, 4, 32), torch.ones(1, 4), None, labels)


if __name__ == '__main__':
    unittest.main()
