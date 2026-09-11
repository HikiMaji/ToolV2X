"""Compare the installed, head-dimension-128 kernel with explicit FP32 attention."""
import json
from pathlib import Path
import torch
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func


def reference(q, k, v):
    scores = (q.transpose(1, 2) / (128 ** .5)) @ k.transpose(1, 2).transpose(-1, -2)
    causal_mask = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    return (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)


torch.manual_seed(20)
records = []
for dtype, rtol, atol in ((torch.float16, .01, .002), (torch.bfloat16, .04, .02)):
    for varlen in (False, True):
        shape = (96, 2, 128) if varlen else (1, 64, 2, 128)
        inputs = [torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True) for _ in range(3)]
        gold_inputs = [x.detach().float().requires_grad_() for x in inputs]
        if varlen:
            cumulative = torch.tensor([0, 32, 96], device='cuda', dtype=torch.int32)
            actual = flash_attn_varlen_func(*inputs, cumulative, cumulative, 64, 64, causal=True)
            expected = torch.cat([reference(*(x[start:end][None] for x in gold_inputs))[0]
                                  for start, end in ((0, 32), (32, 96))])
        else:
            actual = flash_attn_func(*inputs, causal=True)
            expected = reference(*gold_inputs)
        output_gradient = torch.randn_like(actual) * .05
        actual_gradients = torch.autograd.grad(actual, inputs, output_gradient)
        expected_gradients = torch.autograd.grad(expected, gold_inputs, output_gradient.float())
        errors = {}
        for name, value, gold in zip(('output', 'dq', 'dk', 'dv'),
                                     (actual,) + actual_gradients, (expected,) + expected_gradients):
            assert torch.isfinite(value).all() and torch.isfinite(gold).all()
            assert torch.allclose(value.float(), gold, rtol=rtol, atol=atol), (dtype, varlen, name)
            errors[name] = float((value.float() - gold).abs().max())
        records.append(dict(dtype=str(dtype), varlen=varlen, causal=True, rtol=rtol, atol=atol,
                            maximum_absolute_errors=errors))
result = dict(status='PASS', flash_attn_version=flash_attn.__version__,
              imported_package=flash_attn.__file__, torch=torch.__version__,
              scope='small explicit FP32 attention reference; not full upstream package certification',
              results=records)
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
