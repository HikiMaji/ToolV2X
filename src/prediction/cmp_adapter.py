"""Causal windows -> CMP's ORIGINAL MotionTransformer and existing checkpoint.

The 22-channel layout follows V2V4RealMultiEgoDataset.generate_centered_trajs_for_agents.
GT-dependent selection/labels stay outside inference; physical time is corrected to 0.1s.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CMP = Path(os.environ.get('TOOLV2X_CMP_ROOT', str(ROOT.parent / 'CMP')))
VENDOR = ROOT / 'vendor/cmp_mtr'
CHECKPOINT = Path(os.environ.get('TOOLV2X_MTR_CHECKPOINT',
    str(CMP / 'MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth')))
CONFIG = VENDOR / 'configs/v2v4real_multiego_no_coop.yaml'


def add_vendor_path():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def make_batch(window, center_indices):
    from probe.kinematic_tools import history_packet
    history_packet(window)  # Reuse causal-window validation, no predictions generated here.
    add_vendor_path()
    from mtr.utils.common_utils import rotate_points_along_z
    states = torch.as_tensor(window['states'], dtype=torch.float32)
    valid = torch.as_tensor(window['valid'], dtype=torch.bool)
    indices = torch.as_tensor(center_indices, dtype=torch.long)
    if not len(indices) or len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= len(states)).any():
        raise ValueError('invalid center-object selection')
    centers = states[indices, -1]
    count, objects, steps = len(indices), len(states), 11
    xyz = states[None, :, :, :3] - centers[:, None, None, :3]
    xyz = rotate_points_along_z(xyz.reshape(count, -1, 3), -centers[:, 6]).reshape(count, objects, steps, 3)
    sizes = states[None, :, :, 3:6].expand(count, -1, -1, -1)
    types = torch.zeros(count, objects, steps, 2)
    types[..., 0] = 1.
    types[torch.arange(count), indices, :, 1] = 1.
    times = torch.zeros(count, objects, steps, steps + 1)
    times[:, :, torch.arange(steps), torch.arange(steps)] = 1.
    times[..., -1] = torch.arange(steps) * .1
    heading = states[None, :, :, 6] - centers[:, None, None, 6]
    features = torch.cat([xyz, sizes, types, times, torch.stack([heading.sin(), heading.cos()], -1)], -1)
    mask = valid[None].expand(count, -1, -1).clone()
    features[~mask] = 0.
    return {'batch_sample_count': [count], 'input_dict': {
        'obj_trajs': features, 'obj_trajs_mask': mask, 'obj_trajs_last_pos': features[:, :, -1, :3].clone(),
        'track_index_to_predict': indices, 'center_objects_type': np.array(['TYPE_VEHICLE'] * count),
        'center_objects_world': centers, 'center_objects_id': np.asarray(window['track_ids'])[indices.numpy()]}}


def load_model(checkpoint=CHECKPOINT):
    add_vendor_path()
    from mtr.config import cfg, cfg_from_yaml_file
    from mtr.models_v2v4real.model import MotionTransformer
    from easydict import EasyDict
    config = EasyDict()
    cfg_from_yaml_file(str(CONFIG), config)
    cfg.ROOT_DIR = VENDOR
    config.MODEL.MOTION_DECODER.DEVICE = 'cpu'
    # mmap and assign reuse checkpoint storage; optimizer tensors are not materialized.
    saved = torch.load(str(checkpoint), map_location='cpu', mmap=True)
    prefix = 'motion_transformer.'
    if any(not k.startswith(prefix) for k in saved['model_state']):
        raise ValueError('expected CMP no-cooperation checkpoint containing only MotionTransformer')
    state = {k[len(prefix):]: v for k, v in saved['model_state'].items()}
    with torch.device('meta'):
        model = MotionTransformer(config.MODEL)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    evidence = {'checkpoint': str(checkpoint), 'config': str(CONFIG), 'model_class': 'CMP MotionTransformer',
        'state_items_loaded': len(state), 'missing_keys': list(loaded.missing_keys),
        'unexpected_keys': list(loaded.unexpected_keys), 'parameter_count': sum(p.numel() for p in model.parameters()),
        'checkpoint_metadata': {k: saved.get(k) for k in ('epoch', 'it', 'version')},
        'source': str(VENDOR / 'mtr/models_v2v4real/model.py'),
        'upstream_source': str(CMP / 'MTR/mtr/models_v2v4real/model.py'), 'device': 'cpu',
        'operators': 'CMP layers with portable torch KNN/indexed attention reference',
        'native_CUDA_parity_measured': False, 'training_provenance_verified': False,
        'input_distribution': 'causal fixed-frame history; differs from original GT-aligned moving-frame dataset'}
    return model, evidence


@torch.inference_mode()
def predict(model, window):
    states = np.asarray(window['states'])
    valid = np.asarray(window['valid'])
    n = len(states)
    means = np.zeros((n, 6, 50, 2), np.float32)
    means[:] = states[:, None, -1:, :2]
    scores = np.zeros((n, 6), np.float32)
    scores[:, 0] = 1.
    local_gmm = np.zeros((n, 6, 50, 5), np.float32)
    model_used = valid.sum(axis=1) >= 2
    for i in np.flatnonzero(model_used):
        batch = make_batch(window, [int(i)])
        output = model(batch)
        raw = output['pred_trajs'][0].cpu()
        probabilities = output['pred_scores'][0].cpu().numpy()
        if tuple(raw.shape) != (6, 50, 5) or not torch.isfinite(raw).all() or not np.isfinite(probabilities).all():
            raise ValueError('invalid original MTR output')
        from mtr.utils.common_utils import rotate_points_along_z
        xy = rotate_points_along_z(raw[None, :, :, :2].reshape(1, -1, 2),
            torch.tensor([states[i, -1, 6]])).reshape(6, 50, 2).numpy()
        means[i] = xy + states[i, -1, :2]
        scores[i] = probabilities
        local_gmm[i] = raw.numpy()
    if not np.isfinite(means).all() or (scores < 0).any() or (scores.sum(axis=1) > 1.0001).any():
        raise ValueError('invalid predictions/scores')
    return dict(track_ids=np.asarray(window['track_ids']), states=states[:, -1], means=means,
                scores=scores, local_gmm=local_gmm, model_used=model_used)
