"""GPU compatibility checks for the local head_dim=128 FlashAttention wheel."""

import argparse
import json
from pathlib import Path

import torch

from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func


def _finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise AssertionError("non-finite " + name)


def _finish(q, k, v, out, label):
    _finite(out, label + ".out")
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.grad is None:
            raise AssertionError(label + "." + name + ".grad is None")
        _finite(tensor.grad, label + "." + name + ".grad")
    return {
        "label": label,
        "shape": list(out.shape),
        "out_abs_mean": float(out.detach().float().abs().mean()),
        "q_grad_abs_mean": float(q.grad.float().abs().mean()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def fixed(dtype, causal, seqlen=256, nheads=2, label=None):
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn((1, seqlen, nheads, 128), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    torch.cuda.synchronize()
    result = _finish(q, k, v, out, label or (str(dtype) + ".causal=" + str(causal)))
    del q, k, v, out
    torch.cuda.empty_cache()
    return result


def varlen(dtype, causal):
    torch.cuda.reset_peak_memory_stats()
    lengths = [64, 48]
    total = sum(lengths)
    cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device="cuda")
    q = torch.randn((total, 2, 128), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        dropout_p=0.0,
        causal=causal,
    )
    torch.cuda.synchronize()
    result = _finish(q, k, v, out, "varlen." + str(dtype) + ".causal=" + str(causal))
    del q, k, v, out, cu
    torch.cuda.empty_cache()
    return result


def reject_head_dim_64():
    q = torch.randn((1, 8, 1, 64), device="cuda", dtype=torch.float16)
    try:
        flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
    except RuntimeError as exc:
        message = str(exc)
        if "only supports head dimension 128" not in message:
            raise AssertionError(message)
        return {"label": "unsupported_head_dim_64", "message": message.splitlines()[0]}
    raise AssertionError("head_dim=64 unexpectedly succeeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long", action="store_true", help="also test 2248 x 32 x 128 fp16 backward")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for causal in (False, True):
            results.append(fixed(dtype, causal))
            results.append(varlen(dtype, causal))
    results.append(reject_head_dim_64())
    if args.long:
        results.append(fixed(torch.float16, True, seqlen=2248, nheads=32, label="long_fp16_2248x32x128"))
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
