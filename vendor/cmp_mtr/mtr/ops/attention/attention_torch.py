"""Device-portable reference for CMP's indexed CUDA dot products and sums.

The surrounding projections, scaling, masking and softmax remain in CMP's
MultiheadAttentionLocal. Native CUDA parity has not been measured in this runtime.
"""
import torch


def gather_keys(key_batch_cnt, index_pair_batch, index_pair, features):
    offsets = torch.cat([key_batch_cnt.new_zeros(1), key_batch_cnt.cumsum(0)[:-1]])
    valid = index_pair >= 0
    indices = offsets[index_pair_batch.long(), None] + index_pair.clamp_min(0)
    return features[indices.long()] * valid[..., None, None], valid


def attention_weight_computation(query_batch_cnt, key_batch_cnt, index_pair_batch,
                                 index_pair, query_features, key_features):
    keys, _ = gather_keys(key_batch_cnt, index_pair_batch, index_pair, key_features)
    return (query_features[:, None] * keys).sum(dim=-1)


def attention_value_computation(query_batch_cnt, key_batch_cnt, index_pair_batch,
                                index_pair, attn_weight, value_features):
    values, _ = gather_keys(key_batch_cnt, index_pair_batch, index_pair, value_features)
    return (attn_weight[..., None] * values).sum(dim=1)
