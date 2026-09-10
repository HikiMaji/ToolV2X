# ToolV2X 第一阶段流水线规格（P/F 四配置 → oracle 实验）

> **2026-09-10 更新：正式四配置训练 HOLD。** 已完成协议审查、因果窗口与评价缺陷修正；当前以 [protocol_audit.md](protocol_audit.md) 和 [experiment_protocol.md](experiment_protocol.md) 为准。下文保留 9 月 9 日历史记录，其中“F 补充远端独有目标”“CMP 完整版”“10–15% 停止线”、旧成本与环境/权重状态不可直接作为当前事实。原 CMP dataset/聚合器尚未完成因果接入。


最后更新：2026-09-09。本文是实现规格，背景与决策见 `STATUS.md`。数据格式细节见 `data_format_v2vgot_npy.md`、`cmp_repo_notes.md`。

---

## 1. 四个配置的定义（全部在 V2V4Real test 集 1993 帧上）

| 配置 | 感知来源 | 预测来源 | 在 ToolV2X 中 |
|---|---|---|---|
| **Ego-only** | `no_fusion_keep_all`（ego 单车 PointPillar） | MTR-local（吃 ego 单车跟踪） | 不调用任何远端能力 |
| **+P** | `cobevt`（协作融合检测，同 CMP 检测器） | MTR-local（吃协作跟踪） | 调 Remote Perception |
| **+F** | `no_fusion_keep_all` | MTR-local + 远端（CAV1）预测聚合 | 调 Remote Forecast |
| **+PF** | `cobevt` | MTR + 聚合（= CMP 完整版） | 两个都调 |

说明：
- 「Remote F」在 CMP 中 = CAV1 用自己的跟踪跑 MTR，把预测轨迹发给 ego，ego 用 `MotionAggregatorTransformer` 融合（`MTR/mtr/models_v2v4real/multi_ego_mtr_model.py:1073`）。F 的独立信息量来自 CAV1 对 ego 不可见目标的**历史跟踪**。
- 「+F 而不 +P」在 CMP 代码里没有现成配置（CMP 的聚合器默认叠在 cobevt 上），需要用 `no_coop` 的感知 + `Transformer` 聚合器组合一个新 cfg。这是本项目相对 CMP 唯一需要新增的组合。

## 2. 数据流

```
V2V-GoT npy (已有, 5 配置 × train/test)
   │  src/common/v2v4real_meta.py       统一读取 + ego→world 变换
   ▼
src/tracking/run_ab3dmot.py             AB3DMOT(无需训练), 每序列独立, 世界系
   │  outputs/tracks/{split}/{config}_world.pkl
   ▼
src/prediction/tracks_to_cmp.py (待写)  → CMP 轨迹 pickle 格式 (见 §3)
   │  outputs/cmp_trajs/{perception_name}/{split}/{scene}-{cav}-traj.pickle
   ▼
CMP/MTR  train_multiego.py / test_multiego.py  (Local F / Remote F)
   │  每帧每目标 6 模态 × 50 帧未来轨迹 + 置信度
   ▼
src/oracle/ (待写)  简单规划器 → ego waypoint → 4 配置逐帧误差 → oracle 分析
```

## 3. CMP 轨迹 pickle 格式（已用仓库自带文件核实）

文件：`<dir>/<split>/<scene>-<cav>-traj.pickle`，`scene` 是 V2V4Real 原始目录名（如 `testoutput_CAV_data_2022-03-15-09-54-40_0`），`cav` ∈ {0,1}。

```python
{
  'data': { object_id(int): [[x, y, z, l, w, h, yaw, valid], ...]  # 长度 = 序列帧数 T, list of list
  },
  'timestamps': ['000000', '000001', ...]                             # 长度 T
}
```

- **坐标系：每帧 cav0 (ego) 的 LiDAR 系**（随 ego 运动，不是世界系）。已用 seq0 帧0 / seq3 帧0 与 V2V-GoT `gt.npy` 数值逐项比对，x,y,z,l,w,h 一致（误差 <2e-3）。
- **yaw 单位在 CMP 自己的数据里是不一致的（CMP 的 bug，已核实）**：`DatasetPreprocess/V2V4Real2dict.py` 对 cav0 直接写 yaml 的 `angle[1]`（V2V4Real yaml 中是**弧度**，与 V2V-GoT `gt.npy` 逐目标比对 3.011 vs 3.010 一致），对 cav1 却做了 `np.rad2deg`（L143）。对仓库全部 GT pickle 统计：所有 `*-0-traj.pickle` max|yaw|≤3.14（弧度），所有 `*-1-traj.pickle` 达 ±180（度）。跟踪 pickle（`multi_ego_inference_v2v4real.py:455,608`）则统一 `rad2deg` 为度。MTR dataset 直接对该列做 sin/cos（`radians()` 被注释）。
  **结论**：CMP 官方权重是在"ego=cav0 时 GT 弧度、跟踪度"这种混乱输入上训的；我们**自己的适配器统一用弧度**（与 V2V-GoT 一致），并且只用 ego=cav0 的样本。如果要加载 CMP 官方权重，其输入约定与我们不同，数值不可直接对齐——这也是倾向自训 MTR 的一个理由（见 §5）。
- 缺失帧整行为 0（valid=0）。
- **key 必须是 GT object_id**：MTR dataset 取 GT 与跟踪 pickle 的 id 交集（`v2v4real_multiego_dataset.py:601-607`）。CMP 的做法是把检测框与 GT IoU≥0.7 匹配拿 GT id、只跟踪匹配上的框。**这是把 GT 泄漏进"感知"的做法**，我们必须复现它才能与 CMP 数字对齐，但要在论文里说明；同时保留一份纯 AB3DMOT id 的版本用于诚实评估。
- **两边 GT id 空间有一个 offset 差异（已处理）**：V2V4Real yaml 里只被 CAV1 标注（无 `ass_id`）的对象，V2V-GoT 编号为 `object_id + 100*cav_id`（`base_postprocessor.py:198`），CMP 为 `object_id + 1000*cav_id`（`V2V4Real2dict.py:129`）。有 `ass_id` 的对象两边一致（<100）。适配器 `v2vgot_id_to_cmp_id()` 做 `100≤i<1000 → i−100+1000`。remap 后我们 test 集 247 个对象里 237 个能在 CMP GT 中找到，剩余 10 个是 CMP GT 文件里没有的 CAV1-only 对象（CMP 只把它们写进 cav1 文件）。
- V2V-GoT 还提供 `gt_object_id_{visible,invisible}_to_{ego,1}.npy`（no_fusion 目录），可直接用来定义 **occlusion-critical 子集**（对 ego 不可见但存在的 GT）。

### 3.2 适配器产出（`src/prediction/tracks_to_cmp.py`，test 集已生成）

| 输出目录 `outputs/cmp_trajs/` | 来源 | 对象数 | 有效状态 | MTR 可用 (obj,t) 样本 |
|---|---|---|---|---|
| `tracking_trajs_no_fusion_gt/test/*-0-*` | ego 单车跟踪，GT id | 247 | 23724 | 13081 |
| `tracking_trajs_cobevt_gt/test/*-0-*` | cobevt 协作跟踪，GT id | 322 | 30530 | 16776 |
| `tracking_trajs_no_fusion_cav1_gt/test/*-1-*` | CAV1 单车跟踪（ego 系），GT id | 281 | 25079 | 13247 |
| `tracking_trajs_no_fusion_raw/test/*-0-*` | ego 单车跟踪，原始 AB3DMOT id | 1020 | 27097 | —（与 GT 无交集，仅用于诚实评估） |

对照：CMP 仓库自带 test 跟踪结果 sinbevt 4944 / cobevt 5780 样本，GT 上界 14674。我们的样本数明显多于 CMP，原因是 (a) 我们保留 score≥0.2 的全部检测（CMP 只跟踪 IoU≥0.7 匹配上的框），(b) 匹配阈值更宽（IoU≥0.5 或中心距≤2m），(c) 做了 CMP 风格的内部缺帧插值。这意味着我们的"感知"比 CMP 的更接近 GT，P 与 GT 的差距更小——**oracle 实验里 P 的价值可能被低估**，做敏感性分析时把 IoU 阈值收紧到 0.7 再跑一版。
- 历史 11 帧 `[t-10, t]`、未来 50 帧都要求全 valid，否则该目标不进样本。

GT pickle 仓库已给全：`CMP/preprocessed_data/v2v4real/gt_multiego_speedless/{train,test}/`（train 64 文件 = 32 场景 × 2 CAV；test 18 文件 = 9 × 2）。

### 3.1 V2V-GoT 序列号 ↔ CMP scene 名对照（test）

| V2V-GoT seq | 全局帧 | CMP scene 名 | T |
|---|---|---|---|
| 0 | 0–146 | testoutput_CAV_data_2022-03-15-09-54-40_0 | 147 |
| 1 | 147–260 | testoutput_CAV_data_2022-03-15-10-29-43_3 | 114 |
| 2 | 261–404 | testoutput_CAV_data_2022-03-15-10-29-43_4 | 144 |
| 3 | 405–602 | testoutput_CAV_data_2022-03-17-10-50-46_0 | 198 |
| 4 | 603–782 | testoutput_CAV_data_2022-03-17-10-50-46_1 | 180 |
| 5 | 783–1092 | testoutput_CAV_data_2022-03-17-11-02-23_1 | 310 |
| 6 | 1093–1396 | testoutput_CAV_data_2022-03-17-11-02-23_2 | 304 |
| 7 | 1397–1617 | testoutput_CAV_data_2022-03-17-11-51-42_0 | 221 |
| 8 | 1618–1992 | testoutput_CAV_data_2022-03-21-09-35-07_7 | 375 |

帧数逐一吻合，顺序一致。train 集的 32 个对照可用同样方法（按 `gt_multiego_speedless/train` 文件名排序 + `TRAIN_LEN_RECORD` 帧数）生成，写适配器时自动校验帧数。

## 4. 已有跟踪输出（`outputs/tracks/`）

`run_ab3dmot.py` 输出：`{'meta', 'seqs': {seq_idx: {global_frame: (K,9) [track_id,x,y,z,l,w,h,yaw,score]}}}`，**世界系**（用 ego `lidar_pose` 变换后再跟踪，AB3DMOT 参数沿用 V2V4Real cobevt 基准：hungarian, giou_3d, thr −0.2, min_hits 3, max_age 2，score≥0.2）。

no_fusion/test 结果：465 条轨迹，长度中位数 13 帧，≥60 帧 86 条。**注意 AB3DMOT max_age=2 会让遮挡 >2 帧的目标断 id**，这对 MTR 需要连续 11 帧历史很不利；写适配器时评估是否需要放宽 max_age 或按 GT id 重新拼接（CMP 用 GT id 匹配天然规避了这个问题）。

## 5. MTR 环境

- CMP 要求 py3.7 / torch 1.13.1 / spconv-cu117 / torch-geometric 2.3.1；`MTR/mtr/ops/{attention,knn}` 附带 cp37 预编译 `.so`，**这两个 op 与 torch 版本绑定**，在 dmstrack 环境（torch 1.12）大概率加载失败。
- 建议：新建 conda env `cmp`（按 `CMP/environment.yml`），装到 `/root/autodl-tmp/conda-envs/cmp`。磁盘只剩 20G，先删 `V2V-GoT/dataset_processed_features_and_gt.root-partial.zip`（9.5G，需用户确认）。
- **预训练权重**：Google Drive 链接（`CMP/docs/prepare_dataset_checkpoints.md:38,73`），含 `v2v4real_multiego_{no_coop, cobevt_c256_no_agg, cobevt_c256, v2vnet}` 四个 MTR 权重。**本机直连和学术加速均无法访问 drive.google.com**。需要用户在本地下载后上传，或提供可访问的镜像。拿到权重则 Local F / Remote F **不需要训练**，只需推理。
- 若拿不到权重：训练量约 1 万样本，BATCH 1，30 epoch，单卡 4090 估计 Local F 半天、Remote F（需先有 no_agg 权重再训聚合器）再半天。

## 6. Remote F 对 fused BEV feature 的依赖

CMP 的聚合器输入包含 ego 的 fused BEV feature `(1,256,48,128)`（`multi_ego_mtr_model.py:77-117`），dataset 加载时**检查该文件存在否则跳过样本**（`v2v4real_multiego_dataset.py:178`）。这些特征文件仓库没给，需要跑 CMP 的 `opencood/tools/multi_ego_inference_v2v4real.py` 从**原始点云**生成——而原始 V2V4Real 点云不在本机。

两条路：
1. 下载 V2V4Real test 集原始数据（约 30–40G，磁盘不够，需清理）跑 CMP 感知端，同时可顺带得到与 CMP 完全一致的检测/跟踪；
2. 改 `multi_ego_mtr_model.py` 让聚合器不吃 BEV feature（只用轨迹特征），并改 dataset 跳过检查。Remote F 会略弱于 CMP 原版，但仍是合法的「远端预测聚合」。

**建议先走路 2 做 oracle 实验**（快），方向确认后再走路 1 对齐 CMP 数字。

## 7. oracle 实验的规划器与指标（待实现，`src/oracle/`）

- 规划器：先用 V2V-GoT 已实现的规则/学习 waypoint 逻辑之外的**最简版本**——以 GT ego 未来 3s 轨迹为参考路径，规划器只决定纵向速度剖面（跟车/让行），碰撞检测基于各配置给出的他车预测轨迹。这样四配置的差异只来自感知/预测输入。
- 指标：waypoint L2@{1,2,3}s、与 GT 他车未来轨迹的 collision rate、以及 occlusion-critical 子集（存在 `gt_object_id_invisible_to_ego` 非空的帧，V2V-GoT 已给该文件）。
- 输出四个数字：oracle 上界增益、P/F 可分性比例、STOP/单工具比例、P 结果与 F 增益相关性。判据见 `STATUS.md §5`。
