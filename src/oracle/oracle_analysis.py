"""
Historical CMP diagnostic ONLY: four GT-indexed prediction sets, GT reference
path, longitudinal speed selection, observed center-distance proximity proxy.
Not a causal predictor, online planner, body collision metric or bandwidth test.
Requires an independent scene/frame manifest and explicit legacy protocol name.
Every action must cover exactly the manifest; absent frames are errors.
No real model results have been produced by this script's regression tests.
"""
import os
import sys
import glob
import pickle
import argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import v2v4real_meta as M  # noqa: E402

CFGS = {'ego': 'local_f_egoP', 'P': 'local_f_coopP', 'F': 'remote_f_egoP', 'PF': 'remote_f_coopP'}
SPEED_SCALES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
H_FRAMES = 30          # 3s
WP_STEP = 5            # 0.5s
COLL_RADIUS = 2.5      # m, 近似 车长/2 + margin
MODE_PROB_MIN = 0.1
# This reader handles only historical CMP GT-indexed diagnostics, not causal inference.


def load_results(root, cfg, tag):
    """返回 {(scene, t): {obj_id: (pred_trajs (6,50,2), pred_scores (6,))}}，只取 ego_cav_id==0。"""
    d = glob.glob(os.path.join(root, cfg, tag, 'eval', '**', 'inference_result'), recursive=True)
    if len(d) != 1:
        raise ValueError('expected exactly one inference_result directory, found %d for %s' % (len(d), cfg))
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d[0], '*.pkl')):
        name = os.path.basename(f)[:-4]
        left, obj = name.split('--')
        scene, ego_cav, t = left.rsplit('-', 2)
        if int(ego_cav) != 0:
            continue
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        out[(scene, int(t))][int(obj)] = (r['pred_trajs'][:, :, :2], r['pred_scores'])
    return out


def scene_to_global(split='test'):
    files = sorted(glob.glob('/root/autodl-tmp/CMP/preprocessed_data/v2v4real/gt_multiego_speedless/%s/*-0-traj.pickle' % split))
    names = [os.path.basename(f)[:-len('-0-traj.pickle')] for f in files]
    return {n: s for n, (_, s, _) in zip(names, M.seq_ranges(split))}


SELF_RADIUS = 1.5  # m: offline ego-ID heuristic; ambiguity is an error.


def cav_self_ids(split, g):
    """仅排除 ego；合作车辆依然是障碍。GT 身份仅限离线诊断。"""
    boxes, ids = M.load_gt(split, g)
    return ego_ids_from_boxes(boxes, ids)


def ego_ids_from_boxes(boxes, ids):
    """Offline ego annotation exclusion in that annotation's own ego frame."""
    if boxes.shape[0] == 0:
        return set()
    near = ids[np.linalg.norm(boxes[:, :2], axis=1) < SELF_RADIUS]
    if len(near) > 1:
        raise ValueError('ambiguous ego identity')
    return set(near.tolist())


def gt_other_futures(split, g, horizon=H_FRAMES):
    """GT 他车未来轨迹（ego(t) 系），缺帧 nan；仅排除 ego，保留合作车。"""
    p0 = M.load_pose(split, g)
    exclude = cav_self_ids(split, g)
    out = defaultdict(lambda: np.full((horizon, 2), np.nan))
    for k in range(1, horizon + 1):
        boxes, ids = M.load_gt(split, g + k)
        if boxes.shape[0] == 0:
            continue
        step_exclude = exclude | ego_ids_from_boxes(boxes, ids)
        bw = M.boxes_to_world(boxes, M.load_pose(split, g + k))
        be = M.boxes_to_ego(bw, p0)
        for b, i in zip(be, ids):
            if int(i) in step_exclude:
                continue
            out[int(i)][k - 1] = b[:2]
    return dict(out)


def path_points(ego_future):
    """ego GT 未来 30 帧位置 (30,2) -> 累计弧长 s(k) 与插值函数。"""
    pts = np.concatenate([np.zeros((1, 2)), ego_future], axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    return pts, s


def plan(ego_future, others, scale_list=SPEED_SCALES):
    """others: {id: (n_modes,30,2) 或 (30,2)}. 返回 (选中 scale, waypoints (30,2))"""
    pts, s = path_points(ego_future)
    total = s[-1]
    for sc in scale_list:
        traj = np.zeros((H_FRAMES, 2))
        for k in range(1, H_FRAMES + 1):
            target = min(total, sc * s[k])
            traj[k - 1] = np.array([np.interp(target, s, pts[:, 0]), np.interp(target, s, pts[:, 1])])
        collide = False
        for tr in others.values():
            tr = np.asarray(tr)
            if tr.ndim == 2:
                tr = tr[None]
            d = np.linalg.norm(tr[:, :H_FRAMES] - traj[None], axis=-1)  # (modes,30)
            if np.nanmin(d) < COLL_RADIUS:
                collide = True
                break
        if not collide:
            return sc, traj
    return 0.0, np.zeros((H_FRAMES, 2))


def preds_to_others(pred_dict, exclude=()):
    """pred_dict 的 key 是 CMP id 空间；exclude 里是 V2V-GoT id，需转换。"""
    from prediction.tracks_to_cmp import v2vgot_id_to_cmp_id
    ex = {v2vgot_id_to_cmp_id(i) for i in exclude}
    others = {}
    for oid, (trajs, scores) in pred_dict.items():
        trajs, scores = np.asarray(trajs), np.asarray(scores)
        if (trajs.ndim != 3 or trajs.shape[0] == 0 or trajs.shape[1] < H_FRAMES
                or trajs.shape[2] != 2 or scores.shape != (trajs.shape[0],)
                or not np.isfinite(trajs).all() or not np.isfinite(scores).all()
                or (scores < 0).any() or not np.isclose(scores.sum(), 1., atol=1e-3)):
            raise ValueError('invalid prediction or mode probabilities for object %s' % oid)
        if oid in ex:
            continue
        keep = scores >= MODE_PROB_MIN
        if not keep.any():
            keep = scores == scores.max()
        others[oid] = trajs[keep][:, :H_FRAMES]
    return others


def evaluate(split, results, out_csv, expected_frames=None):
    s2g = scene_to_global(split)
    if set(results) != set(CFGS):
        raise ValueError('expected all four action frame sets')
    frame_set = set(results['ego']) if expected_frames is None else set(expected_frames)
    if not frame_set or any(set(r) != frame_set for r in results.values()):
        raise ValueError('frame coverage mismatch: every action must cover the independent frame manifest')
    frames = sorted(frame_set)
    rows = []
    for scene, t in frames:
        g = s2g[scene] + t
        ego_fut = M.ego_future_waypoints(split, g, horizon_frames=H_FRAMES, step=1)
        if ego_fut is None:
            raise ValueError('frame horizon crosses scene boundary: %s/%d' % (scene, t))
        gt_others = gt_other_futures(split, g)
        gt_sc, gt_plan = plan(ego_fut, gt_others)          # GT 上界规划
        row = {'scene': scene, 't': t, 'g': g, 'gt_scale': gt_sc,
               'n_gt_others': len(gt_others)}
        observed = sum(np.isfinite(tr).all(axis=1).sum() for tr in gt_others.values())
        row['gt_coverage'] = observed / float(H_FRAMES * len(gt_others)) if gt_others else 1.0
        inv_ids = set(np.load(os.path.join(M.pose_dir(split), '..', '%04d_gt_object_id_invisible_to_ego.npy' % g)).tolist()) \
            if os.path.exists(os.path.join(M.pose_dir(split), '..', '%04d_gt_object_id_invisible_to_ego.npy' % g)) else set()
        row['occ_critical'] = int(len(inv_ids) > 0)
        for key in CFGS:
            others = preds_to_others(results[key][(scene, t)], exclude=cav_self_ids(split, g))
            sc, traj = plan(ego_fut, others)
            # 规划代价：与 GT 上界规划的 waypoint L2（每 0.5s 取点）+ 与 GT 他车的碰撞
            wp_err = np.linalg.norm(traj[WP_STEP - 1::WP_STEP] - gt_plan[WP_STEP - 1::WP_STEP], axis=1)
            coll = 0
            for tr in gt_others.values():
                if np.nanmin(np.linalg.norm(tr - traj, axis=1)) < COLL_RADIUS:
                    coll = 1
                    break
            row['%s_scale' % key] = sc
            row['%s_l2_1s' % key] = wp_err[1]
            row['%s_l2_2s' % key] = wp_err[3]
            row['%s_l2_3s' % key] = wp_err[5]
            row['%s_l2' % key] = wp_err.mean()
            row['%s_coll' % key] = coll
            row['%s_nobj' % key] = len(others)
            row['%s_cost' % key] = row['%s_l2' % key] + 10.0 * coll  # diagnostic quality proxy, NOT communication cost
        rows.append(row)
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows):
    if not rows:
        raise ValueError('no evaluation frames')
    keys = list(CFGS)
    stack = np.array([[r[k + '_cost'] for k in keys] for r in rows])
    if not np.isfinite(stack).all():
        raise ValueError('non-finite diagnostic quality')
    lines = ['# 历史 CMP / GT 参考路径诊断', '',
             '仅用于给定 GT 路径、GT 身份辅助的速度选择诊断；不是在线规划或闭环安全结果。',
             'proximity 是已观测 GT 轨迹的中心距 <2.5m 事件；不是车体几何碰撞率。',
             'quality = 与 GT 条件速度规划的 waypoint L2 + 10 × proximity；不含通信成本。',
             'GT 未来缺标位置未计为安全；coverage 只衡量已出现目标的观测比例，不证明标注完整。', '',
             '帧数: %d；场景数: %d；平均 GT coverage: %.4f' % (
                 len(rows), len({r['scene'] for r in rows}), np.mean([r['gt_coverage'] for r in rows])), '',
             '| 动作 | quality | L2 (m) | observed proximity |',
             '|---|---|---|---|']
    for i, k in enumerate(keys):
        lines.append('| %s | %.4f | %.4f | %.4f |' % (
            k, stack[:, i].mean(), np.mean([r[k + '_l2'] for r in rows]),
            np.mean([r[k + '_coll'] for r in rows])))
    oracle = stack.min(axis=1)
    lines += ['', '离线 quality oracle: %.4f；PF−oracle: %.4f' % (
        oracle.mean(), (stack[:, 3] - oracle).mean()), '',
        '严格最优以 quality 差值 >0.05 定义，仅描述分布，不作为方向生死阈值。']
    strict = []
    for i, k in enumerate(keys):
        win = stack[:, i] + .05 < np.delete(stack, i, axis=1).min(axis=1)
        strict.append(win)
        lines.append('- %s 严格最优: %.1f%%' % (k, 100 * win.mean()))
    lines.append('- 无唯一严格最优: %.1f%%（可能仅部分动作接近，不能解释为全部动作相同）' % (
        100 * (~np.any(strict, axis=0)).mean()))
    lines += ['', '场景内配对 PF−oracle（描述性结果；未提供统计置信区间）：',
              '| 场景 | 帧数 | PF−oracle |', '|---|---|---|']
    for scene in sorted({r['scene'] for r in rows}):
        mask = np.array([r['scene'] == scene for r in rows])
        lines.append('| %s | %d | %.4f |' % (scene, mask.sum(), (stack[mask, 3] - oracle[mask]).mean()))
    lines += ['', '尚无实测消息成本 ledger，不计算通信节省、质量—成本 Pareto 或路由部署收益。',
              '该结果不证明顺序调用必要性；后续必须比较相同信息与预算下的规则和小模型。']
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='/root/autodl-tmp/CMP/MTR/output')
    ap.add_argument('--tag', default='default')
    ap.add_argument('--protocol', required=True, choices=['legacy-gt-reference-diagnostic'])
    ap.add_argument('--frames_json', required=True, help='independent JSON list of [scene, local_t]')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out_md', default=os.path.join(HERE, '..', '..', 'docs', 'oracle_legacy_diagnostic.md'))
    a = ap.parse_args()
    results = {k: load_results(a.results_root, cfg, a.tag) for k, cfg in CFGS.items()}
    for k, r in results.items():
        print(k, 'frames', len(r), 'objs', sum(len(v) for v in r.values()))
    import json
    with open(a.frames_json) as f:
        expected = [tuple(x) for x in json.load(f)]
    if len(set(expected)) != len(expected):
        raise ValueError('duplicate frame in manifest')
    rows = evaluate(a.split, results, os.path.join(HERE, '..', '..', 'outputs', 'oracle', 'legacy_per_frame.csv'), expected)
    md = summarize(rows)
    with open(a.out_md, 'w') as f:
        f.write(md + '\n')
    print(md)
