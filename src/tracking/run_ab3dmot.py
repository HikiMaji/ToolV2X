"""
用 AB3DMOT（V2V-GoT 自带的拷贝，无需训练）对任一检测配置做逐序列跟踪，输出世界坐标系轨迹。

用法（dmstrack 环境）:
  python run_ab3dmot.py --config cobevt --split test [--frame ego|world] [--score_thr 0.2]

输出: /root/autodl-tmp/ToolV2X/outputs/tracks/{split}/{config}_{frame}.pkl
  {
    'meta': {...},
    'seqs': { seq_idx: { global_frame: ndarray (K, 9) [track_id, x, y, z, l, w, h, yaw, score] } }
  }
  坐标系由 --frame 决定：world = 用 ego lidar_pose 变换后再跟踪（默认，供 MTR 使用）；
  ego = 与 V2V4Real 跟踪基准一致（每帧 ego 系、无 ego-motion 补偿）。

AB3DMOT 内部使用 KITTI 相机系 [h,w,l,x,y,z,theta]，这里沿用 V2V-GoT 的做法：
kitti_x = x, kitti_y = z(高), kitti_z = y(横)，BEV 匹配在 x-z 平面进行，与原始 x-y 平面等价。
"""
import os
import sys
import argparse
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M  # noqa: E402

sys.path.insert(0, M.AB3DMOT_ROOT)
sys.path.insert(0, os.path.join(M.AB3DMOT_ROOT, 'Xinshuo_PyToolbox'))
from AB3DMOT_libs.model import AB3DMOT  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402


class _NullLog:
    def write(self, *_):
        pass

    def flush(self):
        pass


def to_kitti(boxes):
    """(N,7) [x,y,z,l,w,h,yaw] -> (N,7) [h,w,l,x,z,y,yaw]"""
    return np.stack([boxes[:, 5], boxes[:, 4], boxes[:, 3],
                     boxes[:, 0], boxes[:, 2], boxes[:, 1], boxes[:, 6]], axis=1)


def from_kitti(k):
    """(N,7) [h,w,l,x,z,y,yaw] -> (N,7) [x,y,z,l,w,h,yaw]"""
    return np.stack([k[:, 3], k[:, 5], k[:, 4], k[:, 2], k[:, 1], k[:, 0], k[:, 6]], axis=1)


def make_tracker():
    cfg = edict(dataset='v2v4real', det_name='cobevt', ego_com=False, vis=False,
                affi_pro=False, num_hypo=1, score_threshold=-1e4)
    return AB3DMOT(cfg, 'Car', hw={'image': None, 'lidar': None}, log=_NullLog())


def run(config, split, frame_mode, score_thr):
    out = {'meta': dict(config=config, split=split, frame=frame_mode, score_thr=score_thr,
                        columns=['track_id', 'x', 'y', 'z', 'l', 'w', 'h', 'yaw', 'score']),
           'seqs': {}}
    for seq_idx, start, end in M.seq_ranges(split):
        tracker = make_tracker()
        seq_out = {}
        for g in range(start, end):
            boxes, scores = M.load_dets(config, split, g)
            keep = scores >= score_thr
            boxes, scores = boxes[keep], scores[keep]
            if frame_mode == 'world':
                boxes = M.boxes_to_world(boxes, M.load_pose(split, g))
            dets = to_kitti(boxes)
            # info 列: AB3DMOT 会把 info 原样带出；这里放 [score, 0..] 共 7 列以匹配其 15 列输出约定
            info = np.zeros((dets.shape[0], 7), dtype=np.float64)
            info[:, 0] = scores
            results, _ = tracker.track({'dets': dets, 'info': info}, g - start, '%04d' % seq_idx)
            r = results[0]  # (K,15): h,w,l,x,y,z,theta, id, info(7)
            if r.shape[0] == 0:
                seq_out[g] = np.zeros((0, 9), dtype=np.float32)
                continue
            boxes_out = from_kitti(r[:, :7])
            row = np.concatenate([r[:, 7:8], boxes_out, r[:, 8:9]], axis=1)
            seq_out[g] = row.astype(np.float32)
        out['seqs'][seq_idx] = seq_out
        n_ids = len({int(i) for f in seq_out.values() for i in f[:, 0]})
        print('[%s/%s] seq %d: %d frames, %d track ids' % (config, split, seq_idx, end - start, n_ids), flush=True)

    save_dir = os.path.join(HERE, '..', '..', 'outputs', 'tracks', split)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, '%s_%s.pkl' % (config, frame_mode))
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    print('saved', os.path.abspath(path))
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(M.DET_CONFIGS))
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--frame', default='world', choices=['world', 'ego'])
    ap.add_argument('--score_thr', type=float, default=0.2)
    a = ap.parse_args()
    run(a.config, a.split, a.frame, a.score_thr)
