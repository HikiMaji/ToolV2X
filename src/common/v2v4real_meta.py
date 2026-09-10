"""
V2V4Real / V2V-GoT npy 数据的统一读取层。

所有下游脚本（跟踪、预测、oracle 分析）只通过这个模块读取检测结果和位姿，
避免各处重复硬编码路径与坐标约定。格式细节见 docs/data_format_v2vgot_npy.md。

坐标约定：
- npy 中 gt/pred 是 (N,8,3) 角点，位于当前帧 ego LiDAR 系（x 前, y 右, z 上），yaw 弧度绕 z。
- lidar_pose 是 4x4 ego LiDAR -> world 齐次矩阵，只存在于 no_fusion_keep_all 目录，
  fusion 配置按同一全局帧号复用。
"""
import os
import sys
from pathlib import Path
import numpy as np

V2VGOT_ROOT = os.environ.get('TOOLV2X_V2VGOT_ROOT',
                             str(Path(__file__).resolve().parents[3] / 'V2V-GoT'))
OPENCOOD_ROOT = os.path.join(V2VGOT_ROOT, 'DMSTrack', 'V2V4Real')
AB3DMOT_ROOT = os.path.join(V2VGOT_ROOT, 'DMSTrack', 'AB3DMOT')

VAL_LEN_RECORD = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
TRAIN_LEN_RECORD = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524,
                    3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636,
                    5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
LEN_RECORD = {'test': VAL_LEN_RECORD, 'train': TRAIN_LEN_RECORD}
FPS = 10

# 检测配置名 -> (目录名, 子目录)。'ego_cav1' 是 CAV1 的单车检测（已在 ego 系）。
DET_CONFIGS = {
    'no_fusion': ('no_fusion_keep_all', 'ego'),
    'no_fusion_cav1': ('no_fusion_keep_all', '1'),
    'early': ('early', 'ego'),
    'attfuse': ('attfuse', 'ego'),
    'cobevt': ('cobevt', 'ego'),
    'v2xvit': ('v2xvit', 'ego'),
}


def config_dir(config, split):
    dirname, sub = DET_CONFIGS[config]
    prefix = 'train_' if split == 'train' else ''
    return os.path.join(V2VGOT_ROOT, prefix + dirname, 'npy', sub)


def pose_dir(split, cav='ego'):
    prefix = 'train_' if split == 'train' else ''
    return os.path.join(V2VGOT_ROOT, prefix + 'no_fusion_keep_all', 'npy', cav)


def num_frames(split):
    return LEN_RECORD[split][-1]


def seq_ranges(split):
    """返回 [(seq_idx, start, end_exclusive), ...]"""
    out, start = [], 0
    for k, end in enumerate(LEN_RECORD[split]):
        out.append((k, start, end))
        start = end
    return out


def seq_of(g, split):
    for k, start, end in seq_ranges(split):
        if g < end:
            return k, g - start, start, end
    raise IndexError(g)


def _opencood():
    if OPENCOOD_ROOT not in sys.path:
        sys.path.insert(0, OPENCOOD_ROOT)
    from opencood.utils import box_utils
    return box_utils


def corners_to_center(corners):
    """(N,8,3) -> (N,7) [x,y,z,l,w,h,yaw]，ego 系。"""
    if corners.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)
    box_utils = _opencood()
    c = box_utils.corner_to_center(corners, order='hwl')  # x,y,z,h,w,l,yaw
    return np.concatenate([c[:, 0:3], c[:, 5:6], c[:, 4:5], c[:, 3:4], c[:, 6:7]], axis=1)


def load_pose(split, g, cav='ego'):
    return np.load(os.path.join(pose_dir(split, cav), '%04d_lidar_pose.npy' % g))


def load_dets(config, split, g):
    """返回 (boxes (M,7) [x,y,z,l,w,h,yaw] ego 系, scores (M,))"""
    d = config_dir(config, split)
    corners = np.load(os.path.join(d, '%04d_pred.npy' % g))
    scores = np.load(os.path.join(d, '%04d_pred_score.npy' % g))
    return corners_to_center(corners), scores.astype(np.float32)


def load_gt(split, g, config='no_fusion'):
    """返回 (boxes (N,7) ego 系, object_ids (N,))。默认取 no_fusion 的 GT（范围最全）。"""
    d = config_dir(config, split)
    corners = np.load(os.path.join(d, '%04d_gt.npy' % g))
    ids = np.load(os.path.join(d, '%04d_gt_object_id.npy' % g))
    return corners_to_center(corners), ids.astype(np.int64)


def boxes_to_world(boxes, pose):
    """(N,7) ego 系 -> world 系，pose 为 4x4 ego->world。"""
    if boxes.shape[0] == 0:
        return boxes.copy()
    xyz1 = np.concatenate([boxes[:, :3], np.ones((boxes.shape[0], 1))], axis=1)
    xyz_w = (pose @ xyz1.T).T[:, :3]
    yaw_w = boxes[:, 6] + np.arctan2(pose[1, 0], pose[0, 0])
    out = boxes.copy()
    out[:, :3] = xyz_w
    out[:, 6] = wrap_angle(yaw_w)
    return out


def boxes_to_ego(boxes_w, pose):
    inv = np.linalg.inv(pose)
    return boxes_to_world(boxes_w, inv)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ego_future_waypoints(split, g, horizon_frames=30, step=5):
    """
    与 V2V-GoT inference.py:2926-2958 一致：inv(P_t) @ P_{t+k} 的平移 [0:2]。
    返回 (H,2) 或 None（跨序列边界时）。step=5 即每 0.5s 一个点。
    """
    _, _, _, end = seq_of(g, split)
    if g + horizon_frames >= end:
        return None
    p0_inv = np.linalg.inv(load_pose(split, g))
    pts = []
    for k in range(step, horizon_frames + 1, step):
        pk = load_pose(split, g + k)
        rel = p0_inv @ pk
        pts.append(rel[0:2, 3])
    return np.stack(pts)
