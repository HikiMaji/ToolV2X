# V2V-GoT / V2V4Real 检测结果 npy 数据格式说明

> 盘点日期 2026-09-09。数据根目录 `/root/autodl-tmp/V2V-GoT`。本文档自包含，供后续接入 AB3DMOT 跟踪 + MTR 轨迹预测的 agent 使用。
> 所有 python 请用 `/root/autodl-tmp/conda-envs/dmstrack/bin/python`（含 numpy）。**不要解压 zip、不要修改 V2V-GoT 目录中的文件。**

---

## 0. 一句话总览

- 10 个检测配置目录：`no_fusion_keep_all / early / attfuse / cobevt / v2xvit`（测试集，各 **1993 帧**，编号 `0000–1992`）和带 `train_` 前缀的 5 个训练集版本（各 **7105 帧**，`0000–7104`）。
- 帧编号是**全局连续编号**，跨 9 个（测试）/ 32 个（训练）序列；序列边界由 `len_record` 累积长度表给出（§3）。
- `gt.npy` / `pred.npy` 都是 **(N, 8, 3) 角点**，在**当前帧 ego LiDAR 坐标系**。`lidar_pose.npy` 是 **4x4 ego LiDAR→world** 齐次矩阵。只有 `no_fusion_keep_all` / `train_no_fusion_keep_all` 含 `lidar_pose` 和 CAV1 (`1/`) 子目录；fusion 配置复用同编号的 no_fusion pose 即可（同一个 ego、同一个全局帧号）。
- 目前**没有**任何已经跑好的 AB3DMOT / DMSTrack 跟踪结果，也没有 npy→KITTI txt 的中间产物。

---

## 1. 目录结构

```
/root/autodl-tmp/V2V-GoT/
├── no_fusion_keep_all/          # 单车检测(PointPillar), 保留全部框, 测试集
│   ├── config.yaml, net_epoch60.pth
│   ├── eval.yaml, short_eval.yaml, middle_eval.yaml, long_eval.yaml   # OpenCOOD AP 结果(见§4)
│   └── npy/
│       ├── XXXX_gt.npy, XXXX_gt_object_id.npy, XXXX_pred.npy, XXXX_pred_score.npy   # 与 ego/ 完全相同的副本
│       ├── XXXX_gt_object_id_{visible,invisible}_to_{ego,1}.npy                    # 仅 no_fusion
│       ├── ego/    XXXX_{gt,gt_object_id,pred,pred_score,lidar_pose,transformation_matrix,projected_lidar}.npy
│       ├── 1/      同上 (CAV 1 视角)
│       └── co_llm/ XXXX_detection_box_score.npy ; ego/ 与 1/ 下各有 XXXX_{classification_map,regression_map,detection_box_score}.npy
├── early/ attfuse/ cobevt/ v2xvit/          # 协作融合检测, 测试集
│   └── npy/
│       ├── XXXX_{gt,gt_object_id,pred,pred_score}.npy      # 副本
│       ├── ego/   XXXX_{gt,gt_object_id,pred,pred_score,transformation_matrix}.npy   # 无 lidar_pose, 无 projected_lidar
│       └── co_llm/ XXXX_detection_box_score.npy ; ego/{classification_map,regression_map,detection_box_score}
├── train_no_fusion_keep_all/ train_early/ train_attfuse/ train_cobevt/ train_v2xvit/   # 训练集, 结构同上, 7105 帧
│   （train_no_fusion_keep_all/npy/ego 与 1/ 没有 projected_lidar）
├── DMSTrack/
│   ├── DMSTrack/main_dkf.py             # DMSTrack 跟踪入口, 含 len_record 硬编码
│   ├── AB3DMOT/main.py                  # AB3DMOT 跟踪入口
│   ├── AB3DMOT/scripts/KITTI/v2v4real_{val,train}_label/*.txt            # KITTI 格式 GT 跟踪标签
│   ├── AB3DMOT/scripts/KITTI/v2v4real_{val,train}_evaluate_tracking.seqmap.val
│   └── V2V4Real/
│       ├── opencood/tools/inference.py  # QA 生成器 nq1..nq9, waypoint GT 计算, npy→AB3DMOT txt 转换
│       └── official_models/no_fusion_keep_all/npy/   # 指向上面 no_fusion_keep_all/npy 的软链接 + co_llm/*.json (QA 数据集)
└── runs/                                # 之前 LLM 推理 run 的输出
```

各子目录文件数（测试集）：`no_fusion_keep_all/npy/ego`、`/1` 各 1993×7=13951 个；fusion 配置 `ego/` 1993×5=9965 个；`co_llm/` 每配置 1993 个 `detection_box_score` + 子目录。

---

## 2. 每类 npy 文件的格式

以下 shape/范围取自 `no_fusion_keep_all/npy/ego/{0000,1000}_*.npy` 并对全部 1993 帧做了统计核对。

| 文件 | shape / dtype | 含义 / 坐标系 |
|---|---|---|
| `XXXX_gt.npy` | `(N, 8, 3) float32` | GT 3D 框 **8 角点**，**当前帧 ego LiDAR 坐标系**（不是世界坐标）。角点 0–3 为底面，4–7 为顶面，且 `corner[i+4] = corner[i] + (0,0,h)`；边 0→3 / 1→2 为车长方向，0→1 / 3→2 为车宽方向。测试集全部 1993 帧共 31421 个 GT 框，单帧最多 40。 |
| `XXXX_gt_object_id.npy` | `(N,) int64` | 每个 GT 框的 V2V4Real 轨迹 ID（跨帧一致，可直接作跟踪 GT）。**行数与 `gt` 完全一致**（全帧验证）。 |
| `XXXX_pred.npy` | `(M, 8, 3) float32` | 检测框 8 角点，与 `gt` 同一坐标系、同一角点顺序。 |
| `XXXX_pred_score.npy` | `(M,) float32` | 检测置信度，与 `pred` 行对齐，**已按降序排序**。全局范围 0.200–0.893（检测阈值 0.2）。 |
| `XXXX_lidar_pose.npy` | `(4, 4) float32` | **ego LiDAR → world 的齐次变换矩阵 `P_ego(t)`**（不是 6 维 `[x,y,z,roll,yaw,pitch]`）。平移量可达数千米（如 `[-747.5, 246.4, 4.6]`、`[252.9, -3921.3, -7.8]`），说明是全局（GPS/RTK）世界坐标。10 Hz，ego 每帧中位移动 0.54 m。仅 `no_fusion_keep_all` 与 `train_no_fusion_keep_all` 有此文件（`ego/` 与 `1/` 各自的位姿）。 |
| `XXXX_transformation_matrix.npy` | `(4, 4) float32` | **该 CAV 的 LiDAR → ego LiDAR** 变换。`ego/` 下恒为单位阵；`1/` 下 `T_1→ego(t) ≈ inv(P_ego(t)) @ P_1(t)`（数值验证最大误差 2e-4）。fusion 配置 `ego/` 下也是单位阵，无信息量。 |
| `XXXX_projected_lidar.npy` | `(P, 4) float32` | 点云 `x, y, z, intensity`，**已投影到 ego 坐标系**（CAV1 的点云中心约在 ego 前方 -73 m 处即 CAV1 位置）。仅测试集 `no_fusion_keep_all/npy/{ego,1}` 有。 |
| `co_llm/XXXX_detection_box_score.npy` | `(M, 8) float32` | `[h, w, l, x, y_kitti, z_kitti, yaw, score]`：由 `pred` 角点经 `corner_to_center(order='hwl')` 得到 7 维再**按 AB3DMOT/KITTI 习惯交换了 y、z 列**。即 `col3 = x(前)`, `col4 = 原 z(高度, ≈-1.2)`, `col5 = 原 y(横向)`, `col6 = yaw(弧度)`, `col7 = score`。yaw 全局范围 -1.04–2.60 rad。与 `co_llm/ego/XXXX_detection_box_score.npy` 内容相同。 |
| `co_llm/{ego,1}/XXXX_classification_map.npy` | `(1, 2, H, W) float32` | 检测头原始分类 logits（2 anchor）。BEV 尺寸随模型不同：no_fusion `50×88`，cobevt/v2xvit `48×128`，attfuse `48×176`，early `50×176`。 |
| `co_llm/{ego,1}/XXXX_regression_map.npy` | `(1, 14, H, W) float32` | 检测头原始回归输出（2 anchor × 7）。 |
| `npy/XXXX_gt_object_id_{visible,invisible}_to_{ego,1}.npy` | `(K,) int64` | GT ID 子集：对 ego / CAV1 可见或不可见（仅 no_fusion）。 |

坐标轴约定（引自 `inference.py:2942` 注释）：V2V4Real/OpenCOOD 为 "x forward, y right, z up"（左手），yaw 为绕 z 轴、以 +x 为 0 的弧度角（`atan2(corner0-corner3)`）；DMSTrack/AB3DMOT KITTI 风格为 "x forward, y up, z right"，因此 `detection_box_score` 与 AB3DMOT txt 都做了 y/z 交换。

### 2.1 各配置差异
- **fusion 配置（early/attfuse/cobevt/v2xvit）的 GT 与 no_fusion 略有不同**：fusion 的 GT 被裁到融合检测范围（x≤100），少数远处目标被去掉（如 0000 帧 no_fusion 2 个 GT、fusion 1 个）。ID 空间相同。
- `1/pred.npy`（CAV1 单车检测）**已经在 ego 坐标系**（验证：原始框与 GT 中心距 <1.3 m；再乘一次 `T_1→ego` 会偏 5–100 m）。不要重复变换。
- 训练集 `train_*` 序列 0000（帧 0–146）的 ego pose 与测试集序列 0000 完全相同（重叠的 147 帧）。

---

## 3. 帧编号 → 场景（序列）映射

帧编号是跨序列的**全局连续索引**。序列边界硬编码于
`/root/autodl-tmp/V2V-GoT/DMSTrack/DMSTrack/main_dkf.py:823-827`：

```python
val_len_record   = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]   # 测试集 9 序列 (0000..0008)
train_len_record = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648,
                    3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254,
                    6389, 6532, 6681, 6846, 6997, 7105]                 # 训练集 32 序列 (0000..0031)
```
序列 k 的全局帧范围是 `[len_record[k-1], len_record[k])`（k=0 时起点为 0），`local_frame = global - start`（见 `main_dkf.py:66-83 get_global_timestamp_index`）。

测试集 9 序列明细：

| seq | 全局帧范围 | 帧数 |
|---|---|---|
| 0000 | 0–146 | 147 |
| 0001 | 147–260 | 114 |
| 0002 | 261–404 | 144 |
| 0003 | 405–602 | 198 |
| 0004 | 603–782 | 180 |
| 0005 | 783–1092 | 310 |
| 0006 | 1093–1396 | 304 |
| 0007 | 1397–1617 | 221 |
| 0008 | 1618–1992 | 375 |

独立验证：用 `lidar_pose` 平移量相邻帧跳变 >10 m 检测断点，测试集断点恰为 `[147,261,405,603,783,1093,1397,1618]`，训练集断点恰为上表 31 个内部值。同样的边界也写在 `/root/autodl-tmp/V2V-GoT/DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_evaluate_tracking.seqmap.val`（格式 `0000 empty 000000 000146`）。

---

## 4. `eval.yaml` / `short|middle|long_eval.yaml`

OpenCOOD 检测 AP 结果（不是逐帧、不是 GT 列表）。键：`ap_50, ap_70, mpre_50, mpre_70, mrec_50, mrec_70`，其中 mpre/mrec 各为约 33k 点的 P-R 曲线（故文件 0.6–2.8 MB）。short/middle/long 为距离分段（0–30 / 30–50 / 50–100 m）。

| 配置 | AP50 | AP70 |
|---|---|---|
| no_fusion_keep_all | 0.399 | 0.220 |
| early | 0.597 | 0.321 |
| attfuse | 0.647 | 0.336 |
| cobevt | 0.665 | 0.360 |
| v2xvit | 0.649 | 0.368 |

---

## 5. QA 数据集与 waypoint GT 生成

- `/root/autodl-tmp/V2V-GoT/no_fusion_keep_all/npy/co_llm/` 下**没有** json。QA 数据集位于
  `/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/`：
  - `v2v4real_3d_grounding_qa_dataset_v2vgot.json` 49,188,900 B，31,014 条样本
  - `v2v4real_3d_grounding_qa_dataset_v2vgot_simplified_perception.json` 54 MB
  - 单节点版本 `..._nq1sm3w0d.json … _nq9sm3w6dc.json`（4–7 MB），`_v2vllmq5.json`
  - 该目录下 npy 均为指向顶层 `no_fusion_keep_all/npy` 的软链接。
- 样本字段：`id, conversations, scenario_index, local_timestamp_index, global_timestamp_index, asker_cav_id ('ego'|'1'), future_trajectory_str_in_ego, future_trajectory_str_in_self, cav_ego_lidar_pose, cav_1_lidar_pose (展平的 4x4), qa_source, qa_type_id`。
- **生成脚本**：`/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py`（8928 行）中的 `generate_3d_grounding_qa_dataset_nq1 … nq9`；nq8（planning：speed/steering 分类）在 **6502 行**，nq9（建议未来轨迹）紧随其后。包装脚本 `opencood/tools/generate_single_frame_qa.py`（`NODE_GENERATORS`：nq3/nq4 0 waypoint，nq5–nq7 1 waypoint，nq8/nq9 6 waypoint）。
- **ego 未来 waypoint GT 的计算**（`inference.py:2926-2958 get_cav_ego_future_trajectory`，双车版 `2962+ get_double_cavs_future_trajectory`）：

```python
# inference.py:2938-2953
initial_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % global_timestamp_index))
for i in range(global_timestamp_index+1, global_timestamp_index + num_future_frames + 1):
    lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % i))
    lidar_pose_in_initial_frame = x1_to_x2(lidar_pose, initial_lidar_pose)   # = inv(P_t) @ P_i
    location_2d = np.array([lidar_pose_in_initial_frame[0, 3], lidar_pose_in_initial_frame[1, 3]])
    cav_ego_future_trajectory[i - global_timestamp_index - 1] = location_2d
```
`x1_to_x2`（`opencood/utils/transformation_utils.py:53-95`）对两个 4x4 矩阵即 `np.linalg.inv(x2) @ x1`。nq8/nq9 使用 `time_horizon=3 s, frame_rate=10 → num_future_frames=30`，每 0.5 s 取一个 waypoint 共 6 个；只对 `global_idx <= seq_end - 30` 的帧生成（`inference.py:6580-6594`，遍历 `len_record` 保证不跨序列）。

---

## 6. 跟踪入口与现状

- **没有任何已跑好的跟踪输出**：`DMSTrack/AB3DMOT/data/`、`DMSTrack/AB3DMOT/results/`、`DMSTrack/DMSTrack/results/` 均不存在；也没有 `npy/ab3dmot_detection/` 中间 txt。
- 已有：KITTI 格式 GT 跟踪标签 `/root/autodl-tmp/V2V-GoT/DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_label/0000..0008.txt`（31,419 行）和 `v2v4real_train_label/`（32 个）。行格式（空格分隔）：`frame track_id Car 0 0 0 0 0 0 0 h w l x y z yaw`（KITTI 轴序，y/z 已交换）。
- **AB3DMOT 入口**：`/root/autodl-tmp/V2V-GoT/DMSTrack/AB3DMOT/main.py --dataset v2v4real --det_name cobevt`；配置 `AB3DMOT/configs/v2v4real.yml`。读取 `./data/v2v4real/detection/{det_name}_Car_{split}/{seq}.txt`（逗号分隔 15 列：`frame,type(2=Car),x1,y1,x2,y2,score,h,w,l,x,y,z,rot_y,alpha`，见 `AB3DMOT_libs/io.py:9-33`），输出 `./results/v2v4real/{det_name}_Car_val_H1/data_0/{seq}.txt`。评估：`python scripts/KITTI/evaluate.py cobevt_Car_val_H1 1 3D 0.25`。
- **DMSTrack 入口**：`/root/autodl-tmp/V2V-GoT/DMSTrack/DMSTrack/main_dkf.py`（用法见 `DMSTrack/docs/INFERENCE.md`）。检测 txt 路径 `../AB3DMOT/data/v2v4real/detection/{det_name}_Car_{split}/{cav_id}/{seq}.txt`（`main_dkf.py:388`）；npy 特征路径**硬编码**为 `../V2V4Real/official_models/{train_,}no_fusion_keep_all/npy/` 或 `.../cobevt/npy/`（`main_dkf.py:832-837`），无命令行参数。
- **npy → AB3DMOT txt 转换器**：`inference.py` 中 `transform_and_save_detection_to_ab3dmot_format(len_record, npy_save_path, cav_id_set)`（约 648 行起，写 `npy/ab3dmot_detection/{cav_id}/{scenario:04d}.txt`）；核心转换逻辑见 `inference.py:706-749`（`corner_to_center(order='hwl')` → `[h,w,l,x,y,z,yaw]` → 交换 y/z）。要接 AB3DMOT 可直接复用/仿写这段。

---

## 7. 配置 × split 总结表

| 配置 | test 帧 | train 帧 | pred | gt | gt_object_id | lidar_pose | CAV1 目录 `1/` | transformation_matrix |
|---|---|---|---|---|---|---|---|---|
| no_fusion_keep_all | 1993 | 7105 | 有 | 有 | 有 | **有**（ego 与 1） | 有 | ego=I，1=CAV1→ego |
| early | 1993 | 7105 | 有 | 有 | 有 | 无 | 无 | 仅 I |
| attfuse | 1993 | 7105 | 有 | 有 | 有 | 无 | 无 | 仅 I |
| cobevt | 1993 | 7105 | 有 | 有 | 有 | 无 | 无 | 仅 I |
| v2xvit | 1993 | 7105 | 有 | 有 | 有 | 无 | 无 | 仅 I |

**把某配置第 t 帧检测框转到世界坐标**：
- 用 `P = {no_fusion_keep_all | train_no_fusion_keep_all}/npy/ego/{t:04d}_lidar_pose.npy`（fusion 配置没有自己的 pose，同编号复用即可，ego 是同一辆车）。
- 位置：`X_world = P @ [x, y, z, 1]^T`；朝向：`yaw_world = yaw_ego + atan2(P[1,0], P[0,0])`；尺寸不变。
- 若直接用 `detection_box_score` 的 8 列，先还原轴序：`x = col3, y = col5, z = col4`。
- CAV1 的 `1/pred.npy` 已在 ego 系，同样乘 `P_ego`；若需 CAV1 自身系→world 用 `1/{t}_lidar_pose.npy`。
- ego 未来 waypoint（MTR 目标）：`inv(P_ego(t)) @ P_ego(t+k)` 的 `[0:2, 3]`，k 不得跨越 `len_record` 序列边界。

---

## 8. 可直接运行的读取示例

```bash
/root/autodl-tmp/conda-envs/dmstrack/bin/python /tmp/v2vgot_demo.py
```

`/tmp/v2vgot_demo.py`（已验证可运行）：

```python
import numpy as np

ROOT = '/root/autodl-tmp/V2V-GoT'
VAL_LEN_RECORD = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
TRAIN_LEN_RECORD = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648,
                    3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254,
                    6389, 6532, 6681, 6846, 6997, 7105]

def seq_of(g, len_record=VAL_LEN_RECORD):
    """全局帧号 -> (seq_idx, local_frame, seq_start, seq_end_inclusive)"""
    for k, end in enumerate(len_record):
        if g < end:
            start = 0 if k == 0 else len_record[k - 1]
            return k, g - start, start, end - 1
    raise IndexError(g)

def corners_to_center7(c):
    """(N,8,3) 角点 -> (N,7) [x,y,z,l,w,h,yaw(rad)]，ego LiDAR 系"""
    xyz = c.mean(axis=1)
    l = np.linalg.norm(c[:, 0, :2] - c[:, 3, :2], axis=1)
    w = np.linalg.norm(c[:, 0, :2] - c[:, 1, :2], axis=1)
    h = np.abs(c[:, 4:, 2].mean(1) - c[:, :4, 2].mean(1))
    yaw = np.arctan2(c[:, 0, 1] - c[:, 3, 1], c[:, 0, 0] - c[:, 3, 0])
    return np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], l, w, h, yaw], axis=1)

def load_frame(cfg, g, cav='ego'):
    """读取某配置某帧；lidar_pose 从对应 split 的 no_fusion 目录取"""
    d = f'{ROOT}/{cfg}/npy/{cav}'
    out = {k: np.load(f'{d}/{g:04d}_{k}.npy') for k in ['gt', 'gt_object_id', 'pred', 'pred_score']}
    pose_cfg = 'train_no_fusion_keep_all' if cfg.startswith('train_') else 'no_fusion_keep_all'
    out['lidar_pose'] = np.load(f'{ROOT}/{pose_cfg}/npy/{cav}/{g:04d}_lidar_pose.npy')  # 4x4 LiDAR->world
    return out

def to_world(box7, P):
    """(N,7) ego 系框 + 4x4 pose -> (N,7) 世界系框"""
    xyz1 = np.c_[box7[:, :3], np.ones(len(box7))]
    xyz_w = (P @ xyz1.T).T[:, :3]
    yaw_w = box7[:, 6] + np.arctan2(P[1, 0], P[0, 0])
    return np.c_[xyz_w, box7[:, 3:6], yaw_w]

def ego_future_waypoints(g, num_future=30, step=5,
                         pose_dir=f'{ROOT}/no_fusion_keep_all/npy/ego', len_record=VAL_LEN_RECORD):
    """复现 inference.py:2926-2958：未来 num_future 帧 ego 位置在当前帧坐标系下的 (x,y)，每 step 帧取一个"""
    _, _, _, end = seq_of(g, len_record)
    assert g + num_future <= end, 'future horizon crosses sequence boundary'
    P0 = np.load(f'{pose_dir}/{g:04d}_lidar_pose.npy')
    wp = []
    for i in range(g + 1, g + num_future + 1):
        Pi = np.load(f'{pose_dir}/{i:04d}_lidar_pose.npy')
        T = np.linalg.inv(P0) @ Pi
        wp.append(T[:2, 3])
    return np.array(wp)[step - 1::step]          # (6,2) 对应 0.5s..3.0s

if __name__ == '__main__':
    g = 1000
    print('seq/local/start/end:', seq_of(g))                       # (5, 217, 783, 1092)
    f = load_frame('cobevt', g)
    print('gt', f['gt'].shape, 'ids', f['gt_object_id'].shape, 'pred', f['pred'].shape, f['pred_score'][:3])
    b7 = corners_to_center7(f['pred'])
    print('pred[0] ego  :', b7[0].round(3))
    print('pred[0] world:', to_world(b7, f['lidar_pose'])[0].round(3))
    print('ego waypoints (0.5s..3s):\n', ego_future_waypoints(g).round(2))
    dbs = np.load(f'{ROOT}/cobevt/npy/co_llm/{g:04d}_detection_box_score.npy')   # [h,w,l,x,z,y,yaw,score]
    print('detection_box_score[0]:', dbs[0].round(3), '-> x,y,z =', dbs[0, 3], dbs[0, 5], dbs[0, 4])
    # CAV1 (只有 no_fusion 有)，pred 已在 ego 系
    p1 = np.load(f'{ROOT}/no_fusion_keep_all/npy/1/{g:04d}_pred.npy')
    T1 = np.load(f'{ROOT}/no_fusion_keep_all/npy/1/{g:04d}_transformation_matrix.npy')  # CAV1->ego
    print('CAV1 pred', p1.shape, 'CAV1 position in ego frame:', T1[:3, 3].round(2))
```

预期输出（关键行）：`seq/local/start/end: (5, 217, 783, 1092)`；`pred[0] ego: [-26.027 2.735 -1.237 4.692 2.049 1.730 0.042]`；`pred[0] world: [251.70 -3895.14 -8.33 ...]`；waypoints 第一行约 `[5.44, 0.16]`。

### 8.1 写 AB3DMOT 检测 txt 的最小片段（仿 `inference.py:706-749`）

```python
# 输出 ./data/v2v4real/detection/{det_name}_Car_val/{seq:04d}.txt，逗号分隔 15 列
# frame,type,x1,y1,x2,y2,score,h,w,l,x,y_kitti,z_kitti,rot_y,alpha  （y_kitti = ego z, z_kitti = ego y）
for k in range(len(VAL_LEN_RECORD)):
    start = 0 if k == 0 else VAL_LEN_RECORD[k-1]; end = VAL_LEN_RECORD[k]
    with open(f'{out_dir}/{k:04d}.txt', 'w') as fh:
        for g in range(start, end):
            f = load_frame('cobevt', g); b7 = corners_to_center7(f['pred'])
            for (x, y, z, l, w, h, yaw), s in zip(b7, f['pred_score']):
                fh.write(f'{g-start},2,0,0,0,0,{s:.6f},{h},{w},{l},{x},{z},{y},{yaw},0\n')
```
