"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention
"""

from . import attention_torch

__all__ = {
    'v1': attention_torch,
    'v2': attention_torch,
}
