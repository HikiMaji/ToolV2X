# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from . import knn_cuda
except ImportError:
    knn_cuda = None


class KNNBatch(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda
        assert query_xyz.is_contiguous() and query_xyz.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda

        idx = torch.cuda.IntTensor(n, k).zero_()

        knn_cuda.knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None
    

knn_batch = KNNBatch.apply


class KNNBatchMlogK(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        # assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda
        assert query_xyz.is_contiguous() and query_xyz.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda
        assert k <= 128
        idx = torch.cuda.IntTensor(n, k).zero_()

        knn_cuda.knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None
   
def knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, k):
    """Same per-batch nearest-neighbor contract, padding absent keys with -1.

    This portable reference sorts neighbors; native CUDA uses heap order, and
    equal-distance boundary ties may select different keys. No GPU parity claim.
    """
    result = torch.full((len(xyz), k), -1, dtype=torch.int32, device=xyz.device)
    for batch in range(len(query_batch_offsets) - 1):
        selected = torch.where(batch_idxs == batch)[0]
        start, end = int(query_batch_offsets[batch]), int(query_batch_offsets[batch + 1])
        count = min(k, end - start)
        if count and len(selected):
            distance = (xyz[selected, None] - query_xyz[None, start:end]).square().sum(-1)
            result[selected, :count] = distance.argsort(dim=1, stable=True)[:, :count].int()
    return result
