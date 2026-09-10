"""
把 run_ab3dmot.py 的世界系跟踪结果转换成 CMP / MTR 读取的逐场景轨迹 pickle。

用法（dmstrack 环境）:
  python tracks_to_cmp.py --config no_fusion --split test [--id_mode gt|raw] [--out_name ...]

输出目录:  outputs/cmp_trajs/<out_name>/<split>/<scene>-<cav>-traj.pickle
  与 CMP/preprocessed_data/v2v4real/gt_multiego_speedless 同构:
  {'data': {obj_id: [[x,y,z,l,w,h,yaw_rad,valid] * T]}, 'timestamps': ['000000', ...]}
  坐标 = 每帧 ego(cav0) LiDAR 系；yaw 弧度（与 V2V-GoT gt.npy 一致；CMP 自己的 GT 文件 cav0 也是弧度）。

id_mode:
  gt  : 每帧把轨迹框与 GT 框做 BEV IoU 匹配（IoU>=0.5 或中心距<2m），轨迹 id 取匹配 GT id 众数，
        并只保留匹配上的轨迹 —— 复现 CMP 的做法（其 GT id 通过 IoU 泄漏进"感知"），MTR dataset 需要 key=GT id。
  raw : 默认。保留 AB3DMOT id，不读 GT、不插值，保存因果原始序列。
        注意：这不是 CMP 原 dataset 可直接使用的格式；时间窗口应使用 causal_windows.py。
  gt 是历史 GT 辅助诊断协议，仍保留整段 GT 匹配和插值，禁止作为在线结果。

只产出 cav=0（ego）的文件。CMP 的 cav=1 文件是 CAV1 视角的 GT，本项目 Remote F 由
CAV1 的单车跟踪(no_fusion_cav1)在同一 ego 系下单独产出一份，命名 <scene>-1-traj.pickle。
"""
import os
import sys
import glob
import copy
import pickle
import argparse
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M  # noqa: E402

CMP_GT_DIR = '/root/autodl-tmp/CMP/preprocessed_data/v2v4real/gt_multiego_speedless'
OUT_ROOT = os.path.join(HERE, '..', '..', 'outputs', 'cmp_trajs')


def cmp_scene_names(split):
    """按 CMP GT 文件名排序得到 seq_idx -> scene 名，并校验帧数与 len_record 一致。"""
    files = sorted(glob.glob(os.path.join(CMP_GT_DIR, split, '*-0-traj.pickle')))
    names = [os.path.basename(f)[:-len('-0-traj.pickle')] for f in files]
    ranges = M.seq_ranges(split)
    assert len(names) == len(ranges), (len(names), len(ranges))
    for f, (k, s, e) in zip(files, ranges):
        with open(f, 'rb') as fh:
            T = len(pickle.load(fh)['timestamps'])
        assert T == e - s, (f, T, e - s)
    return names


def bev_iou(a, b):
    """a:(N,7) b:(M,7) [x,y,z,l,w,h,yaw] -> (N,M) 旋转框 BEV IoU（shapely）。"""
    from shapely.geometry import Polygon
    from shapely import affinity

    def polys(boxes):
        out = []
        for x, y, _, l, w, _, yaw in boxes:
            p = Polygon([(-l / 2, -w / 2), (l / 2, -w / 2), (l / 2, w / 2), (-l / 2, w / 2)])
            p = affinity.rotate(p, np.degrees(yaw), origin=(0, 0))
            out.append(affinity.translate(p, x, y))
        return out
    pa, pb = polys(a), polys(b)
    iou = np.zeros((len(pa), len(pb)), dtype=np.float32)
    for i, p in enumerate(pa):
        for j, q in enumerate(pb):
            inter = p.intersection(q).area
            if inter > 0:
                iou[i, j] = inter / (p.area + q.area - inter)
    return iou


def v2vgot_id_to_cmp_id(i):
    """
    V2V4Real yaml 里对象 id: 有 ass_id 的用 ass_id（跨 CAV 一致，<100），只被 CAV1 标注的用 object_id + offset*cav_id。
    V2V-GoT offset=100（base_postprocessor.py:198），CMP offset=1000（V2V4Real2dict.py:129）。
    两边都只有 cav_id=1 会加 offset，所以 100<=i<1000 -> i-100+1000。
    """
    i = int(i)
    return i - 100 + 1000 if 100 <= i < 1000 else i


def match_track_ids_to_gt(seq_tracks_ego, split, start, end, iou_thr=0.5, dist_thr=2.0):
    """返回 {track_id: gt_id}（按逐帧匹配众数），未匹配的 track 不在字典里。"""
    votes = defaultdict(Counter)
    for g in range(start, end):
        tr = seq_tracks_ego[g]
        if tr.shape[0] == 0:
            continue
        gt_boxes, gt_ids = M.load_gt(split, g)
        if gt_boxes.shape[0] == 0:
            continue
        iou = bev_iou(tr[:, 1:8], gt_boxes)
        d = np.linalg.norm(tr[:, 1:3, None] - gt_boxes[None, :, :2].transpose(0, 2, 1), axis=1)  # (N,M)
        # 贪心一对一
        cand = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
                if iou[i, j] >= iou_thr or d[i, j] <= dist_thr]
        cand.sort(key=lambda t: -t[0])
        used_i, used_j = set(), set()
        for _, i, j in cand:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            votes[int(tr[i, 0])][v2vgot_id_to_cmp_id(gt_ids[j])] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def interpolate_gaps(states):
    """复制 CMP interpolate_car_info 的行为：对内部缺失帧做线性插值，两端不补。"""
    T = len(states)
    valid = [s[7] == 1 for s in states]
    idx = [i for i in range(T) if valid[i]]
    if len(idx) < 2:
        return states
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a == 1:
            continue
        pa, pb = np.array(states[a]), np.array(states[b])
        for k in range(a + 1, b):
            w = (k - a) / (b - a)
            s = pa.copy()
            s[0:2] = (1 - w) * pa[0:2] + w * pb[0:2]
            dy = M.wrap_angle(pb[6] - pa[6])
            s[6] = M.wrap_angle(pa[6] + w * dy)
            s[7] = 1
            states[k] = s.tolist()
    return states


def build_scene(seq_tracks_world, split, start, end, id_mode):
    T = end - start
    # 世界系 -> 逐帧 ego 系
    seq_ego = {}
    for g in range(start, end):
        tr = seq_tracks_world[g]
        if tr.shape[0] == 0:
            seq_ego[g] = tr
            continue
        pose = M.load_pose(split, g)
        boxes = M.boxes_to_ego(tr[:, 1:8], pose)
        seq_ego[g] = np.concatenate([tr[:, :1], boxes, tr[:, 8:9]], axis=1)

    if id_mode == 'gt':
        remap = match_track_ids_to_gt(seq_ego, split, start, end)
    else:
        remap = None

    data = {}
    for g in range(start, end):
        for row in seq_ego[g]:
            tid = int(row[0])
            if remap is not None:
                if tid not in remap:
                    continue
                oid = remap[tid]
            else:
                oid = tid
            if oid not in data:
                data[oid] = [[0.0] * 8 for _ in range(T)]
            x, y, z, l, w, h, yaw = row[1:8]
            data[oid][g - start] = [float(x), float(y), float(z), float(l), float(w), float(h),
                                    float(M.wrap_angle(yaw)), 1.0]
    if id_mode == 'gt':
        for oid in data:
            data[oid] = interpolate_gaps(data[oid])
    timestamps = ['%06d' % i for i in range(T)]
    meta = {'protocol': 'raw_causal_v1' if id_mode == 'raw' else 'legacy_gt_assisted',
            'coordinate_frame': 'ego_at_each_observation', 'yaw_unit': 'radian',
            'uses_gt_identity': id_mode == 'gt', 'interpolated': id_mode == 'gt',
            'native_cmp_ready': False}
    return {'data': data, 'timestamps': timestamps, 'meta': meta}, (len(remap) if remap else None)


def run(config, split, id_mode, out_name, cav):
    tracks_path = os.path.join(HERE, '..', '..', 'outputs', 'tracks', split, '%s_world.pkl' % config)
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)['seqs']
    names = cmp_scene_names(split)
    out_dir = os.path.join(OUT_ROOT, out_name, split)
    os.makedirs(out_dir, exist_ok=True)
    total_obj, total_valid = 0, 0
    for (k, start, end), name in zip(M.seq_ranges(split), names):
        scene, n_matched = build_scene(tracks[k], split, start, end, id_mode)
        n_obj = len(scene['data'])
        n_valid = sum(int(s[7]) for tr in scene['data'].values() for s in tr)
        total_obj += n_obj
        total_valid += n_valid
        path = os.path.join(out_dir, '%s-%d-traj.pickle' % (name, cav))
        with open(path, 'wb') as f:
            pickle.dump(scene, f)
        print('[%s/%s/%s] seq %d %s: %d objs (%s tracks matched), %d valid states'
              % (config, split, id_mode, k, name[-25:], n_obj, n_matched, n_valid), flush=True)
    print('saved to', os.path.abspath(out_dir), '| objs', total_obj, 'valid states', total_valid)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(M.DET_CONFIGS))
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--id_mode', default='raw', choices=['gt', 'raw'])
    ap.add_argument('--cav', type=int, default=0, help='写入文件名的 cav 编号（ego=0, CAV1=1）')
    ap.add_argument('--out_name', default=None, help='默认 tracking_trajs_<config>_<id_mode>')
    a = ap.parse_args()
    out_name = a.out_name or 'tracking_trajs_%s_%s' % (
        a.config, 'raw_causal_v1' if a.id_mode == 'raw' else 'gt')
    run(a.config, a.split, a.id_mode, out_name, a.cav)
