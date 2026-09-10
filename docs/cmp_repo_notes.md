# CMP 仓库调研笔记（面向 "V2V4Real 检测/跟踪结果 -> CMP/MTR 预测" 的接入）

> **2026-09-10 资源状态更正：** 下文“无权重”“目录不存在”等是首次克隆时的历史快照，已不代表当前机器。四个 CMP MTR 权重现已在本地，单车 MTR 已严格加载并参与真实 P/F→V2V-GoT Q8/Q9 串联。当前资源以 [resource_downloads.md](resource_downloads.md) 为准，实际接入与局限见 [planning_connection.md](planning_connection.md)。数据格式与代码分析仍可参考，资源缺项须重新核实。

- 仓库：https://github.com/tasl-lab/CMP ，已 `git clone --depth 1` 到 **/root/autodl-tmp/CMP**（无 .pth 权重；含 564MB preprocessed_data）。
- 论文：CMP: Cooperative Motion Prediction with Multi-Agent Communication（IEEE TIV 2025 / arXiv 2403.17916）。
- 本文档所有路径均为绝对路径，行号对应克隆时的代码。仅只读调研，未安装任何依赖、未训练。

---

## 0. 是否需要训练 MTR（结论先行）

### (a) 仓库内有没有预训练权重 / 外链
- **仓库内没有任何 `.pth/.ckpt/.bin` 权重文件**（`find CMP -name "*.pth"` 为空；`CMP/MTR/output/` 与 `CMP/pretrained/` 目录都不存在）。
- README 及 docs 给出 3 个 Google Drive 外链（未下载，需自行判断可访问性）：
  1. 感知模型（CoBEVT / V2VNet / SinBEVT，OPV2V+V2V4Real）+ OPV2V 用 Swin 预训练：
     `https://drive.google.com/drive/folders/1EizY6ZFMi__HnqeFPQ2Wf9yRJeD_-S82` （`CMP/docs/prepare_dataset_checkpoints.md:38`）
     期望放到 `CMP/pretrained/v2v4real/{point_pillar_cobevt_multiego_1x, point_pillar_cobevt_multiego_256x, point_pillar_v2vnet_multiego, point_pillar_sinbevt}/{config.yaml, net_epochXX.pth}`。
  2. **MTR 预测模型权重**（`v2v4real_multiego_cobevt_c256`, `..._c256_no_agg`, `v2v4real_multiego_no_coop`, `v2v4real_multiego_v2vnet`，以及同名 opv2v 版本）：
     `https://drive.google.com/drive/folders/1ZUJ5a5VuNfxV34I9FmIefHDGixaJ7gM2` （`CMP/docs/prepare_dataset_checkpoints.md:73`）
     期望放到 `CMP/MTR/output/<cfg_name>/ckpt/best_model.pth`。
  3. 感知中间结果（检测/跟踪/fused feature 等）：
     `https://drive.google.com/drive/folders/1Xqfo1FPNPlunQrj2Apnpg8NQYyXCVmQa` （`CMP/docs/perception_eval.md:29`）
- 没有 Box / HuggingFace 链接。

### (b) preprocessed_data 里有没有现成 V2V4Real pkl
有，但只覆盖"轨迹"层，不含 BEV fused feature：

| 文件 | 大小 | 内容 |
|---|---|---|
| `CMP/preprocessed_data/v2v4real/gt_multiego_speedless/{train,test}/<scene>-<cav>-traj.pickle` | train 64 个文件 / test 18 个文件 | **GT 轨迹**（MTR 的 future label），字段见 §2.3。train 共 14209 帧，test 共 3986 帧 |
| `CMP/preprocessed_data/v2v4real/point_pillar_{sinbevt,cobevt_multiego_1x,cobevt_multiego_256x,v2vnet_multiego}_detection_result_for_tracking_test.pkl` | 6–8.6 MB | **test 集**逐帧检测结果缓存（字段见 §5） |
| `CMP/preprocessed_data/v2v4real/point_pillar_*_tracking_result_for_prediction_test.pkl` | 15–19 MB | **test 集** AB3DMOT 跟踪轨迹（MTR 的 past 输入），结构同 GT pickle |
| `CMP/preprocessed_data/v2v4real/v2v4real_multiego_gt_multiego_speedless_tracking_trajs_point_pillar_*_test_dataset_cache.pkl` | ~2.8 MB | MTR dataset 的 records 缓存（test，sinbevt 2720 条 / cobevt_c256 2858 条 / v2vnet 2802 条） |
| `CMP/preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl` | 1.2 KB | `{'TYPE_VEHICLE': (64,2)}` MTR 的 intention points |

**缺失项**：
- 没有 `tracking_trajs_*/train/*.pickle`（train 集跟踪结果），也没有 `tracking_trajs_*/test/` 目录形式（只有合并 pkl；dataset 类按 per-scene pickle 路径读，需要自己拆，见 §3）。
- 没有 `fused_features_*` 目录（BEV 特征 `.pkl.npy`，每帧 (1,256,48,128)）。**dataset 类会因 fused_feature 文件不存在而跳过所有样本**（`v2v4real_multiego_dataset.py:178-180`），MTR 主干本身并不用它，只有 aggregator 用。
- 注意 cache pkl 里写的路径是作者机器的 `/data1/Datasets/V2V4Real/...` 与 `/data3/zehao/CMP_v5/...`，直接用会失败，需删掉缓存重建。

### (c) 若无权重，训练 MTR 的数据规模和步骤
- 数据规模（V2V4Real）：train 32 个场景 × 2 CAV = 64 条 GT 轨迹文件、14209 帧；test 9 场景 × 2 = 18 文件、3986 帧。MTR 一条样本 = (场景, ego_cav, 当前帧 t)，要求 t≥10 且 t+50 < 序列长度（`v2v4real_multiego_dataset.py:586`），且该帧至少有一辆车 past 10 帧 + future 50 帧全 valid。cfg `SAMPLE_INTERVAL: {train: 5, test: 1}` 只在 waymo dataset 里生效（`waymo_dataset.py:32`），V2V4Real dataset 没有用它，所以 train 大约 (14209 − 64×61) ≈ 1 万条样本量级，test 缓存显示 2700–2900 条。
- 模型规模：MTR encoder D=256、6 层 + decoder D=512、6 层、64 intention query、6 模态；cfg `BATCH_SIZE_PER_GPU: 1`（一个 batch = 一个场景里所有 CAV 的所有 center object），`NUM_EPOCHS: 30`，AdamW lr 1e-4。官方脚本用 8 GPU（`dist_train_multiego.sh`），单卡训 ~1 万样本 × 30 epoch 是可承受量级（小时级到十小时级，取决于每帧 agent 数）。
- 训练分两步（作者流程）：
  1. `v2v4real_multiego_*_no_agg.yaml`（`MOTION_AGGREGATOR.TYPE: 'None'`）从头训 MTR；
  2. `v2v4real_multiego_cobevt_c256.yaml` 把 `PRETRAINED_MOTION_TRANSFORMER` 指向第 1 步的 best_model.pth，训 aggregator（可 `--freeze_mtr`，`train_multiego.py:203-211`）。
  无协作版 `v2v4real_multiego_no_coop.yaml` 只需第 1 步。
- 训练所需的中间文件：GT pickle（仓库已给）+ 你自己的 tracking pickle（train+test 都要）+ （仅当 TYPE≠None 或不改 dataset 时）fused feature npy。**如果只想跑 MTR 主干（no_coop / no_agg），可以把 dataset 里的 fused_feature 检查与加载改掉，就不需要 BEV 特征。**
- 结论：**要么拿到 Google Drive 的 `v2v4real_multiego_no_coop`/`_no_agg` 权重做推理，要么自己训**。即使有权重，因为你的检测/跟踪输入分布不同于作者的 sinbevt/cobevt 轨迹，也建议至少 fine-tune。

---

## 1. 目录结构与 README 中 V2V4Real 流程

```
CMP/
├── AB3Dmot/            修改版 AB3DMOT（3D MOT），含 KITTI 风格 tracking 评估脚本
├── DatasetPreprocess/  OPV2V2dict.py / V2V4Real2dict.py：从数据集 yaml 生成 GT 轨迹 pickle
├── MTR/                修改版 Motion Transformer：mtr/{datasets,models_opv2v,models_v2v4real,ops,utils}, tools/{train,test}_multiego.py, tools/cfgs
├── Plotter/            预测结果可视化（main.py / pcd_processor.py）
├── docs/               env.md, prepare_dataset_checkpoints.md, perception_eval.md, prediction_eval.md, visualization.md
├── opencood/           修改版 OpenCOOD：协作检测（CoBEVT/V2VNet/SinBEVT）+ multi_ego_inference_*.py（检测->跟踪一条龙）
├── preprocessed_data/  {opv2v,v2v4real}/ GT 轨迹 pickle、检测/跟踪缓存 pkl、intention 聚类中心、dataset cache
├── environment.yml / requirements.txt
```

README 本身只有链接；V2V4Real 流程写在 docs：
1. `docs/prepare_dataset_checkpoints.md`：改 `opencood/hypes_yaml/*` 与 `MTR/tools/cfgs/*` 里的 `root_dir/validate_dir/train_dir` 指向 V2V4Real 的 train/test；下载感知与预测 ckpt。
2. `docs/perception_eval.md`：`python opencood/tools/multi_ego_inference_v2v4real.py --perception_model_name point_pillar_cobevt_multiego_256x`（协作），或 `--perception_model_name point_pillar_sinbevt --no_coop`（无协作）。该脚本做检测 + AB3DMOT 跟踪，输出到 `preprocessed_data/v2v4real/`。跟踪评估：`python AB3Dmot/scripts/KITTI/evaluate.py --perception_model_name ... --dataset v2v4real`。
3. `docs/prediction_eval.md`：`bash MTR/tools/scripts/dist_test_multiego.sh --cfg_file MTR/tools/cfgs/v2v4real/v2v4real_multiego_cobevt_c256.yaml --batch_size 8 --ckpt MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth --save_to_file`。
4. `docs/env.md`：PYTHONPATH 需加 `AB3Dmot`, `AB3Dmot/Xinshuo_PyToolbox`, `MTR`, `opencood`；所有命令在仓库根目录执行（脚本内用 `os.getcwd()` 与相对路径 `preprocessed_data/...`）。

---

## 2. DatasetPreprocess：GT 轨迹生成（MTR 的 label 来源）

### 2.1 入口
`/root/autodl-tmp/CMP/DatasetPreprocess/V2V4Real2dict.py`
- `__main__`（L265-267）依次对 train/test 调 `GetGTForSplit()`。
- 参数：`--cfg_file`（MTR yaml，读 `DATA_CONFIG.train_dir/validate_dir`）、`--output_path`（默认 `./preprocessed_data`）、`--tag`（默认 `gt`；仓库已给的是 `gt_multiego_speedless`）。输出 `<output_path>/<tag>/{train,test}/<scene>-<cav_id>-traj.pickle`（L173-174）。
- 注意：它 `import opencood...` 与 `mtr.datasets.v2v4real_multiego_dataset`（L12-21），需要环境。

### 2.2 输入
V2V4Real 原始数据集目录：`<split>/<scene>/<cav_id 0|1>/<timestamp>.yaml`（+`.pcd`）。读取 yaml 字段：
- `lidar_pose`：[x,y,z,roll,yaw,pitch]，**角度为度**（世界坐标）。
- `vehicles[obj_id]`：`location`, `center`, `extent`(半长半宽半高), `angle`([roll,yaw,pitch] 度), `obj_type`, `ass_id`。

坐标系（关键）：
- **ego=cav '0' 的 yaml 里的 vehicles 已经是 cav0 的 LiDAR 坐标**（L83-100 直接用 `location+center`，`2*extent`，`angle[1]` 作为 yaw，单位度）。
- cav '1' 的 vehicles 在 cav1 LiDAR 系，L101-143 用 `x1_to_x2(cav_lidar_pose[i], ego_lidar_pose[i])` 把 cav1 的框投影到 **同一帧 cav0 的 LiDAR 系**；ID：`ass_id != -1` 用 `ass_id`（跨车关联后的统一 ID），否则 `obj_id + 1000*cav_id`（L129-132）。
- L73-74：`if cav_id == '1': continue` —— 外层循环跳过 cav1，所以当前版本脚本**只产出 `<scene>-0-traj.pickle`**，且其中只含 cav0 yaml 里的 vehicles（L101-143 的 cav1 投影分支在 V2V4Real 只有 0/1 两车时实际不会执行）。但仓库里的 `gt_multiego_speedless` 同时有 `-0-` 和 `-1-` 两份，同一物体同一帧 xyz 完全一致、仅 yaw 数值不同（cav0 文件 -3.06°，cav1 文件 -175.5°，相差 180° 是 corner_to_center 反算导致的朝向歧义），说明 `-1-` 文件由早期版本脚本生成并已投影到 cav0 系。**结论：所有 GT/跟踪轨迹的统一坐标系 = 每帧 cav0 (ego) 的 LiDAR 系，随 ego 运动（不是固定世界系）。**
- 只保留 `obj_type == 'Car'`（L86, L103）。
- **不包含 ego 自身轨迹**（`get_gt_traj` 注释 L466 "we do not extract the ego traj"；vehicles 里也没有 ego）。

### 2.3 输出 pickle 结构（GT 与 tracking 完全同构）
```python
{
  'data': OrderedDict{ object_id(int) : list[T] of [x, y, z, l, w, h, yaw_deg, valid] },
  'timestamps': list[str] 长度 T，如 ['000000', ..., '000197']
}
```
- 每个 object 都是**整段序列长度 T** 的列表（缺失帧填 `[0]*8`，valid=0）。
- `yaw_deg` 是**度**（`np.rad2deg`, L143）；MTR dataset 直接对该列做 sin/cos 与 rotate（`v2v4real_multiego_dataset.py:938-946, 1004-1005`，且 L639-641 的 radians 转换被注释掉）——即模型内部把"度数"当弧度用，这是作者代码现状，接入时**保持一致（喂度数）**，否则与官方 ckpt 不兼容。
- 预处理：`padding past`（L151-166：物体首次出现前的帧复制第一帧状态但 valid=0）；`interpolate_car_info`（`v2v4real_multiego_dataset.py:405-461`：中间缺失帧线性插值 x,y,yaw 并置 valid=1）。
- 长度与 cfg：`PAST_FRAMES: 10`, `FUTURE_FRAMES: 50`（cfg L14-15）。past 取 `[t-10, t]` 共 **11 帧**（`extract_frames` L587），future 取 `[t+1, t+50]` 50 帧（L589），10Hz。
- 地图：`map_infos`/`map_polylines` **完全不用**。dataset `ret_dict` 没有 map 键（L663-689），encoder 中 map 相关行被注释（`models_v2v4real/context_encoder/mtr_encoder.py:161,164,170`），decoder 的 map cross-attn 未调用（`mtr_decoder.py:310-360` 只有 obj cross-attn）。cfg 里的 `NUM_INPUT_ATTR_MAP` 等是死配置。
- object type：全部硬编码 `'TYPE_VEHICLE'`（L627），one-hot 第 0 位（L996）。不需要提供类型。
- ego 自身：不需要、也不在输入里；`sdc_track_index = 0` 是占位（L615 注释 "useless"）。

### 2.4 MTR 每条样本的最终张量（`organize_trajectory_timestamp_into_dict` L609-696, `collate_batch` L775-834）
- `obj_trajs` (Nc, No, 11, 22)：[x,y,z,l,w,h](6, 已转到 center object 坐标) + onehot(2: is_vehicle, is_center) + time embedding(12 = 11 one-hot + 时间秒) + [sin yaw, cos yaw]；`NUM_INPUT_ATTR_AGENT: 22` 与此对应（+1 mask 在 encoder 内拼接）。
- `obj_trajs_mask` (Nc, No, 11) bool；`obj_trajs_pos` (Nc,No,11,3)；`obj_trajs_last_pos` (Nc,No,3)。
- `center_objects_world` (Nc, 8) [x,y,z,l,w,h,yaw_deg,valid]（cav0 LiDAR 系当前帧）。
- `obj_trajs_future_state` (Nc, No, 50, 2) [x,y]；`center_gt_trajs` (Nc,50,2)；`center_gt_trajs_mask` (Nc,50)；`center_gt_final_valid_idx` (Nc)；`center_gt_trajs_src` (Nc, 61, 8)。
- `fused_feature` list of (1,256,48,128) numpy（aggregator 用）。
- `batch_dict = {'num_cavs', 'input_dict', 'batch_sample_count'}`；一个 batch = 一个 record 的所有 CAV。
- 一个 track 成为 center object 的条件：在 pred(跟踪) 数据里 past 11 帧全 valid **且** 在 GT 数据里 future 50 帧全 valid，且 ID 同时存在于 GT 与 tracking dict（`extract_common_key_value_pairs_and_sort` L601-607）。**因此 tracking pickle 的 object_id 必须与 GT 的 object_id 对齐**（作者靠检测框与 GT 的 IoU 匹配把 GT id 赋给 track，见 §5）。

---

## 3. MTR 训练/推理入口与 dataset

- 训练：`/root/autodl-tmp/CMP/MTR/tools/train_multiego.py`（`--cfg_file --batch_size --epochs --ckpt --pretrained_model --freeze_mtr --launcher none|pytorch`）；多卡脚本 `MTR/tools/scripts/dist_train_multiego.sh`。模型按 cfg 名字选择 `mtr.models_opv2v` 或 `mtr.models_v2v4real` 的 `MotionTransformerWithMultiEgoAggregation`（L190-194）。
- 推理：`/root/autodl-tmp/CMP/MTR/tools/test_multiego.py`（`--cfg_file --ckpt --batch_size --save_to_file --eval_all`），多卡 `MTR/tools/scripts/dist_test_multiego.sh`（硬编码 8 GPU，单卡可直接 `python MTR/tools/test_multiego.py --launcher none ...`）。结果落在 `MTR/output/<cfg>/default/eval/inference_results`。
- 评估用 waymo 指标（minADE/minFDE/MR/mAP），`eval_second = FUTURE_FRAMES//10 = 5`（`v2v4real_multiego_dataset.py:1159-1161`）；`waymo_eval.py` 依赖 `waymo-open-dataset-tf-2.6.0`（`requirements.txt`）。
- cfg（V2V4Real 4 个）：`/root/autodl-tmp/CMP/MTR/tools/cfgs/v2v4real/`
  - `v2v4real_multiego_no_coop.yaml`：perception=`point_pillar_sinbevt`，`MOTION_AGGREGATOR.TYPE: 'None'`，`PRETRAINED_MOTION_TRANSFORMER: ''`
  - `v2v4real_multiego_cobevt_c256_no_agg.yaml`：perception=`point_pillar_cobevt_multiego_256x`，TYPE None
  - `v2v4real_multiego_cobevt_c256.yaml`：同上 + TYPE `'Transformer'` + PRETRAINED 指向 no_agg ckpt（**CMP 完整版**）
  - `v2v4real_multiego_v2vnet.yaml`
  关键路径键（cfg L4-12）：`train_dir/validate_dir`（原始数据集，dataset 类只用它列目录/读 yaml 做 convex hull）、`preprocessed_gt_traj_dir`、`preprocessed_pred_traj_dir`、`cobevt_fused_features_dir`、`DATASET_CACHE_DIR`。
- dataset 类：`/root/autodl-tmp/CMP/MTR/mtr/datasets/v2v4real_multiego_dataset.py` `V2V4RealMultiEgoDataset`
  - 构造（L22-247）：遍历 `<root_dir>/<scene>/<cav>/*.yaml` 得到时间戳；为每 (scene,cav,t) 记录 `preprocessed_gt_data = <preprocessed_gt_traj_dir>/<split>/<scene>-<cav>-traj.pickle`、`preprocessed_pred_data = <preprocessed_pred_traj_dir>/<split>/<scene>-<cav>-traj.pickle`、`fused_feature = <cobevt_fused_features_dir>/fused_feature_<scene>_<cav>_<t_idx>.pkl.npy`（L123-130）。校验（L143-217）：两 pickle 都存在、至少一辆 CAV 有 valid 段、**fused_feature 文件存在**；写 cache pkl。
  - **注意 pred pickle 文件名需为 per-scene per-cav**，与 `multi_ego_inference_v2v4real.py:558` 落盘一致；仓库只给了合并版 `*_tracking_result_for_prediction_test.pkl`（结构 `{scene: {cav_id(int): {'data','timestamps'}}}`），要自己拆成文件。
  - `__getitem__` → `retrieve_base_data`（L698-773）：读 record 里 `cav_ids_that_have_valid_data` 每辆 CAV 的 GT/pred pickle → `organize_trajectory_timestamp_into_dict`。
  - 数据集要求原始 V2V4Real 数据目录存在（列 yaml；`CONVEX_HULL_THRESHOLD=-1` 时不读 yaml 内容）。若没有原始数据，需改 L35-38、L89-99 从 pickle 的 `timestamps` 取时间戳。

---

## 4. AB3Dmot：修改版 AB3DMOT

- `/root/autodl-tmp/CMP/AB3Dmot/` 基于官方 AB3DMOT（含 `AB3DMOT_libs`, `Xinshuo_PyToolbox`, `configs/{KITTI,nuScenes,OPV2V}.yml`, `scripts/KITTI/evaluate.py`），**有修改**：
  - `AB3DMOT_libs/model.py:50-51, 82-83` 新增 `cfg.dataset == 'opv2v' / 'v2v4real'` 参数：Car 用 `hungar, giou_3d, thres -0.2, min_hits 3, max_age 2`。
  - `track()`（L389-460）输入 `dets_all = {'dets': (N,7) [h,w,l,x,y,z,theta], 'info': (N,7) [ori, type=2, xmin,ymin,xmax,ymax, score], 'cav_id': (N,), 'vel': (N,2)}`；`KF.__init__(bbox3D, info, ID, cav_id)`（`kalman_filter.py:5,15`）新增 `cav_id` 字段（实际存的是**匹配到的 GT object id**，不是 CAV 编号）。输出每行 16 列：`[h,w,l,x,y,z,theta, track_ID, ori, type, xmin,ymin,xmax,ymax, conf, cav_id]`（L452 注释；`io.py:88-89` 解析）。
  - `ego_com` 需要 oxts，CMP 里 `oxts=None`（`multi_ego_inference_v2v4real.py:352`），所以**没有 ego 运动补偿**——跟踪在随 ego 运动的 LiDAR 系里直接做（V2V4Real 都是低速/同向场景，作者接受了这点）。
  - `io.py:load_detection/get_frame_det`（L9-42）文本格式每行 18 列：`frame, type(2=Car), xmin, ymin, xmax, ymax, score, h, w, l, x, y, z, theta, ori, matched_gt_id, vel_x, vel_y`（写入方 `multi_ego_inference_v2v4real.py:57-100`；注意 standup box 写的顺序是 `[0],[2],[1],[3]`，且 h,w,l 取的是 `pred_box3d_np[:,5],[4],[3]`）。
  - CMP 主流程**不走 `AB3Dmot/main.py` 和 txt 文件**，而是在 `opencood/tools/multi_ego_inference_v2v4real.py::TrackByAB3DMOT`（L313-561）里内存中直接构造 `dets_frame` 调用 `tracker.track()`，txt 只是旁路日志（`path_to_detection_output/<scene>/<cav>.txt`, `ab3dmot_logs_*`）。
  - `scripts/KITTI/evaluate.py` 是 KITTI MOT 评估（AMOTA/AMOTP/MOTA...），GT 标签已在 `scripts/KITTI/v2v4real_label/test/<scene>-<cav>.txt`（KITTI tracking 格式 `frame id Car 0 0 0 0 0 0 0 h w l x y z ry`），seqmap `v2v4real_val_evaluate_tracking.seqmap.val`。

---

## 5. opencood：协作检测 → 跟踪 → 预测轨迹的接口

入口 `/root/autodl-tmp/CMP/opencood/tools/multi_ego_inference_v2v4real.py`（`--perception_model_name`, `--no_coop`, `--use_train_set`, `--remove_tracking_speed`(store_false，默认 True)）。

### 5.1 Stage 1 检测（`PerformDetectionWithPretrained` L103-310）
- 读 `pretrained/v2v4real/<name>/config.yaml` 建 opencood dataset（`IntermediateFusionDatasetMultiEgo` 或 no_fusion），逐 (scene, ego_cav, t) 推理。`post_process(..., for_tracking=True)` 返回 `pred_box_tensor (N,8,3) 角点, pred_box3d (N,7), pred_score (N,), gt_box_tensor (M,8,3), gt_object_id_tensor (M,)`（`intermediate_fusion_dataset_multi_ego.py:1070-1075`）。
- **`pred_box3d` 的 7 列顺序 = `[x, y, z, l, w, h, yaw_rad]`**。原因：网络回归输出是 hwl（cfg `postprocess.order: 'hwl'`），但 `voxel_postprocessor.py:295-297` 调 `boxes_to_corners_3d(boxes3d, order='hwl')`，该函数对 torch 张量**就地**执行 `boxes3d[:,3:6] = boxes3d[:,[5,4,3]]`（`box_utils_v2v4real.py:170-171`，`check_numpy_to_torch` 对 torch 输入返回同一对象），于是返回给调用方的 `boxes3d` 已被改成 lwh。实测 dump：`[-40.6, 36.1, -1.28, 4.59, 2.00, 1.81, -0.22]`（第 4 列 4.59 = l）。代码注释 L367 "(l,w,h,z,y,x,yaw)" 是错的。TrackByAB3DMOT L384 `[:, [5,4,3,0,1,2,6]]` 得到 (h,w,l,x,y,z,θ) 正是 AB3DMOT 期望的顺序；写回时 L450-452 `l=item[2], w=item[1], h=item[0]` 一致。yaw 为弧度，写入 pickle 时 `np.rad2deg`（L455）。
- 坐标系：`pred_box3d` 在 **当前帧 ego LiDAR 系**（ego = 当前迭代的 ego_cav_id，可为 0 或 1）；`cur_lidar_pose_np` (4,4) 或 (6,)；协作模式下 ego=1 时还保存 `transformation_matrix`（cav1→cav0，L169-170）。
- **GT id 关联**：`eval_utils.caluclate_tp_fp(..., iou_thresh=0.7, gt_object_id_tensor)`（`opencood/utils/eval_utils.py:43-110`）按分数排序做 BEV IoU 贪心匹配，`matched_indices[i]` = 匹配到的 GT `object_id`（无匹配 -1）——注意它按**分数降序**填 list，而 `pred_box3d_np` 是原顺序，L263 直接 `[:len(pred_score_np)]` 切片，**顺序未对齐**（潜在 bug，但作者数据就是这样）。GT id 来源 `base_postprocessor.generate_object_center` L155-157：`ass_id != -1 ? ass_id : object_id + 1000*cav_id`。
- 每帧字典（即 `*_detection_result_for_tracking_test.pkl[scene][ego_cav_id(int)][t_idx]`）：
  ```
  timestamp_idx:int, timestamp_key:'000000', cur_lidar_pose_np:(4,4), standup_box_np:(N,4) [xmin,ymin,xmax,ymax],
  pred_score_np:(N,), pred_box3d_np:(N,7), matched_car_id:list[N] GT id/-1, speeds:(N,2) (由相邻检测差分,几乎无意义),
  transformation_matrix:None|(4,4), path_to_fused_feature:str
  ```
  fused feature：`np.save(<fused_features_dir>/fused_feature_<scene>_<cav>_<t>.pkl)` → 实际文件名 `.pkl.npy`，shape (1,256,48,128)（no_coop 用 `spatial_features_2d`，协作用 `fused_feature`，L272-277）。

### 5.2 Stage 2 跟踪（`TrackByAB3DMOT` L313-561）
- **只保留 `matched_car_id != -1` 的检测**（L375-381）——即只跟踪与 GT 匹配上的框（这是作者能让 track id == GT id 的关键，也意味着 FP 全部被丢掉）。
- 逐帧 `tracker.track()`；输出转 `car_state = [x,y,z,l,w,h,yaw_deg, vx=0, vy=0, conf, valid=1]`（L455）；协作且 ego==1 时用 `transformation_matrix` 投到 cav0 系（L457-458）；no_coop 且 ego==1 时按 yaml 的 lidar_pose 投到 cav0 系（L529-538，`get_transformation_matrix` 读 `dataset_path` 硬编码 `/data1/Datasets/V2V4Real/`）。**所以两种模式最终轨迹都在 cav0 LiDAR 系。**
- track id → GT id：每条 track 的 `cav_id`（其实是 matched GT id）取众数（L499-520），冲突时保留有效帧多的；无匹配的 track 用 `-track_id` 做 key。
- `remove_tracking_speed`(默认 True) 删掉 `[vx,vy,conf]`，状态变 8 维（L541-545）。
- 落盘：`preprocessed_data/v2v4real/tracking_trajs_<name>/<split>/<scene>-<ego_cav>-traj.pickle` = `{'data': OrderedDict{gt_id: [T×8]}, 'timestamps': [...]}`（L549-559），以及合并缓存 `*_tracking_result_for_prediction_<split>.pkl`。**这就是 MTR 读的 pred 输入。**

---

## 6. 无协作 (SinBEVT) vs 协作 (CMP) 的差别 & 预测聚合模块

| 层 | 脚本/参数 | 差别 |
|---|---|---|
| 感知 | `opencood/tools/multi_ego_inference_v2v4real.py --perception_model_name point_pillar_sinbevt --no_coop` vs `--perception_model_name point_pillar_cobevt_multiego_256x`（无 `--no_coop`） | `--no_coop` → `inference_utils.inference_no_fusion`（L198）、fused feature 取 `spatial_features_2d`、ego=1 轨迹用 yaml pose 投影；否则 `inference_intermediate_fusion`（L193）、用 batch 里的 `transformation_matrix` |
| 预测数据 | MTR cfg `preprocessed_pred_traj_dir` / `cobevt_fused_features_dir` / `perception_model_name` | no_coop cfg 指向 `tracking_trajs_point_pillar_sinbevt`；CMP 指向 `..._cobevt_multiego_256x` |
| 预测模型 | MTR cfg `MODEL.MOTION_AGGREGATOR.TYPE` | `'None'`（no_coop, no_agg）vs `'Transformer'`（CMP）；`DATA_CONFIG.PRETRAINED_MOTION_TRANSFORMER` 指向 no_agg ckpt |

`diff v2v4real_multiego_cobevt_c256.yaml v2v4real_multiego_no_coop.yaml` 只差这 5 行（L8-10, L12, L135）。

**预测聚合模块代码**：`/root/autodl-tmp/CMP/MTR/mtr/models_v2v4real/multi_ego_mtr_model.py`
- 顶层 `MotionTransformerWithMultiEgoAggregation`（L1394-）：`forward` 先对 batch 内所有 CAV 一起跑 `self.motion_transformer`（MTR），再按 `batch_sample_count` 切分每个 CAV 的 `pred_trajs (n,6,50,5)`/`pred_scores`/`center_objects_id`；对每个 ego_cav，把自己放列表首位、其他 CAV 的预测去掉第 0 帧（模拟 100ms 延迟，L1500-1502）后一起送 `self.motion_aggregator(features, scores, fused_feature[ego], batch_sample_count[ego], center_ids)`。
- `MotionAggregatorTransformer`（L1073-1215，cfg 用的）：每条 (6,50,5) 轨迹 flatten→MLP(250→512)；ego BEV (1,256,48,128) →Conv+Linear→128；按 `center_object_id` 把各 CAV 对同一物体的预测拼在一起 + BEV embedding → 5 层 TransformerEncoder(d=640) → 取前 6 个 token 解码为 (6,50,5) 与 6 个分数。**聚合按 object id 对齐，所以所有 CAV 的 track id 必须是同一套（GT id）。**
- 其他候选实现：MLP/MLPV2/GCN/MOE..V6/TransformerV2（同文件），由 cfg TYPE 选择。

---

## 7. preprocessed_data 目录清单

见 §0(b) 表格。另：
- `preprocessed_data/v2v4real/gt_multiego_speedless/test/{test.py, visualize.py}` 两个小脚本演示读取 pickle。
- train GT 覆盖 32 个场景（如 `testoutput_CAV_data_2022-03-15-09-54-40_0/1/2`, `..._2022-03-21-09-35-07_2..9`, `..._2022-03-17-14-44-47_1..6` 等），test 9 个场景。
- 仓库没有 fused feature、没有 train 集跟踪结果、没有任何 ckpt。

---

## 8. 环境依赖（`/root/autodl-tmp/CMP/environment.yml`, `requirements.txt`）
- python 3.7.11；torch 1.13.1 / torchvision 0.14.1；spconv-cu117（→ CUDA 11.7）；torch-geometric 2.3.1；numpy 1.21.6；numba 0.49.0；scipy 1.5.4；shapely 2.0.0；open3d；opencv 4.5.5.62；easydict；filterpy（AB3DMOT KF）；transformers/timm（OPV2V Swin）；`requirements.txt` 额外 `waymo-open-dataset-tf-2.6.0`（仅评估用，MTR 的 waymo_eval 依赖它）。
- 预编译的 `MTR/mtr/ops/{attention,knn}/*.cpython-37m-x86_64-linux-gnu.so` 已在仓库（py3.7）；其他 python 版本需 `cd MTR && python setup.py develop` 重编（需要 nvcc）。opencood 需 `python opencood/utils/setup.py build_ext --inplace`（box_overlaps cython）。

---

## 9. 接入总结：把 V2V4Real npy（每帧检测 (N,7)[x,y,z,l,w,h,yaw] + score + lidar_pose + GT 框 + GT object_id）喂给 CMP/MTR

### 9.1 需要产出的中间文件
对每个 split（train/test）、每个场景 scene、每个 ego cav（0 和 1；若只做无协作/单 ego 可只做 0）：

1. **GT 轨迹** `preprocessed_data/v2v4real/<gt_tag>/<split>/<scene>-<cav>-traj.pickle`
   - 仓库已提供 `gt_multiego_speedless`，可直接复用（前提：你的 scene 名与 timestamps 数量和它一致，均为 `testoutput_CAV_data_*`，10Hz，'000000' 起）。若你的 GT object_id 体系与 V2V4Real yaml 的 `ass_id`/`obj_id+1000*cav` 不同，需自己按 §2.3 格式重建。
2. **跟踪轨迹（MTR past 输入）** `preprocessed_data/v2v4real/<pred_tag>/<split>/<scene>-<cav>-traj.pickle`
   ```python
   {'data': OrderedDict{ gt_object_id: [[x,y,z,l,w,h,yaw_deg,valid] × T] }, 'timestamps': ['000000',...]}
   ```
   - 坐标：**每帧 cav0 的 LiDAR 系**（与 GT 一致；ego=1 的检测用 `x1_to_x2(cav1_lidar_pose, cav0_lidar_pose)` 投过去，`opencood/utils/transformation_utils.py:81`，pose 为 [x,y,z,roll,yaw,pitch] 度）。
   - yaw 用**度**；l,w,h 是全长；T 为整段序列，缺失帧 `[0]*8`；建议同样做 padding past + 线性插值（可直接调用 `V2V4RealMultiEgoDataset.interpolate_car_info`）。
   - **key 必须是 GT object_id**（与 GT pickle 的 key 相同），否则 `extract_common_key_value_pairs_and_sort` 取交集后为空。作者做法：检测框与 GT 框 BEV IoU≥0.7 匹配得 GT id，只跟踪匹配上的框，track 的 id 取匹配 GT id 的众数。你手头已有 GT 框 + GT id，可以复现同样的匹配；或者如果你的跟踪器已输出与 GT 对齐的 id，直接用。
3. （可选）**BEV fused feature** `preprocessed_data/v2v4real/<feat_dir>/fused_feature_<scene>_<cav>_<t_idx>.pkl.npy`，(1,256,48,128) float32。只有 `MOTION_AGGREGATOR.TYPE != 'None'` 才真正使用；否则可用全零占位或改代码跳过。
4. 可复用：`preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl`（intention points，cfg `INTENTION_POINTS_FILE`）。
5. 删除旧的 `preprocessed_data/v2v4real/v2v4real_multiego_*_dataset_cache.pkl`（里面是作者机器路径）。

### 9.2 最少需要改的文件
1. **新建一个转换脚本**（建议放 `CMP/DatasetPreprocess/npy2traj.py` 或放在你自己的仓库）：读你的 npy → 生成 §9.1 第 2 项 pickle（若需要也重建第 1 项）。核心逻辑可从 `opencood/tools/multi_ego_inference_v2v4real.py:437-559`（跟踪结果→pickle）与 `DatasetPreprocess/V2V4Real2dict.py:101-174`（cav1→cav0 投影 + padding）抄。若要复用 AB3DMOT 做跟踪，按 §4 的 `dets_frame` 字典格式调用 `AB3DMOT_libs.model.AB3DMOT.track()`，`cfg.dataset='v2v4real'`。
2. **新建/复制一个 cfg**：拷贝 `MTR/tools/cfgs/v2v4real/v2v4real_multiego_no_coop.yaml` → 改 `train_dir/validate_dir`（原始 V2V4Real 目录，本机可能在 `/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real` 下，需确认）、`preprocessed_pred_traj_dir`、`cobevt_fused_features_dir`、`perception_model_name`；主干模型用 `TYPE: 'None'`。
3. **`MTR/mtr/datasets/v2v4real_multiego_dataset.py`**（视情况）：
   - 若无 fused feature：注释/放宽 L178-180 的存在性检查，并把 L692-694 改为返回零张量或删除 `'fused_feature'` 键（同时 `collate_batch` L824 分支相应处理）。
   - 若没有原始 V2V4Real yaml 目录：改 L35-38、L89-99 让 timestamps 来自 GT pickle 的 `timestamps`。
   - 若你的 scene 名不是 `testoutput_CAV_data_*` 或 cav 名不是 '0'/'1'，检查 L69-141 的路径拼接。
4. **`MTR/tools/scripts/dist_*_multiego.sh`** 硬编码 8 GPU，单卡直接 `python MTR/tools/train_multiego.py --cfg_file <cfg> --launcher none --batch_size 1` / `python MTR/tools/test_multiego.py --cfg_file <cfg> --ckpt <pth> --launcher none --batch_size 1 --save_to_file`（在 CMP 根目录、PYTHONPATH 含 MTR 与 opencood；dataset 顶部 `from opencood.hypes_yaml.yaml_utils import load_yaml` 需要 opencood 可 import，但不需要编译 spconv 等）。
5. 不需要改 `AB3Dmot/`、`opencood/` 模型代码、MTR 模型代码。

### 9.3 关键对齐点（踩坑清单）
- 历史 = 11 帧 `[t-10, t]`（1s + 当前帧），未来 = 50 帧，10Hz；序列至少 63 帧。
- past 11 帧与 future 50 帧都要求 **全部 valid** 才能成为 center object（`extract_frames` L593）；因此插值补洞很重要。
- yaw 列是"度"但被当弧度用（作者代码现状），保持一致。
- 无地图、无类别、无 ego 自车轨迹；ego 自车也不会被预测。
- 轨迹坐标系随 ego (cav0) 运动，不是固定世界系；GT 和 tracking 必须用同一系。
- 想用官方权重（若能下载）必须完全对齐上述格式；否则用 `no_coop`/`no_agg` cfg 自己训 MTR，约 1 万样本、30 epoch。

---

## 10. 2026-09-10：官方准备文档复核与当前实际状态

本节是对前文“尚未下载权重”状态的更新；前文保留作为历史调研记录。

### 10.1 官方文档实际要求

官方 `docs/prepare_dataset_checkpoints.md` 要求三类准备工作：

1. 配置原始 OPV2V/V2V4Real 数据路径；
2. 下载感知模型权重（每个目录包含 `config.yaml` 和 `net_epochXX.pth`）；
3. 下载 MTR 预测模型权重（4 个 setting）；
4. tracking label 与 motion prediction GT 由仓库/预处理结果提供。

官方没有把 `fused_features_*` 列成独立的 checkpoint 下载项。`docs/perception_eval.md` 说明，运行 `multi_ego_inference_v2v4real.py` 时会生成 detection、tracking 和 fused feature 等中间结果；也可以使用作者提供的中间结果。

### 10.2 已下载并验证通过

仓库位置：`/root/autodl-tmp/CMP`。

MTR 预测权重（4 个均可读，ZIP 条目校验通过）：

```text
MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth
MTR/output/v2v4real_multiego_cobevt_c256_no_agg/ckpt/best_model.pth
MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth
MTR/output/v2v4real_multiego_v2vnet/ckpt/best_model.pth
```

V2V4Real 感知权重：

```text
pretrained/v2v4real/point_pillar_cobevt_multiego_256x/{config.yaml,net_epoch75.pth}
pretrained/v2v4real/point_pillar_sinbevt/{config.yaml,net_epoch60.pth}
pretrained/v2v4real/point_pillar_v2vnet_multiego/{config.yaml,net_epoch75.pth}
```

`point_pillar_cobevt_multiego_1x/config.yaml` 已有，但对应的 Google Drive `net_epoch75.pth` 源文件在两次独立下载中都出现相同的 CRC 错误，已标记为 `.invalid_upstream`，不可使用。

### 10.3 已有的 V2V4Real 预处理结果

`preprocessed_data/v2v4real/` 当前有并已逐个 pickle 读取验证：

- 4 套 detection result 合并缓存；
- 3 套 tracking result 合并缓存（`cobevt_256x`、`sinbevt`、`v2vnet`）；
- GT 轨迹目录；
- 3 个 MTR dataset cache；
- `v2v4real_cluster_64_center_dict.pkl`。

这些结果足以作为后续 MTR smoke test 的输入基础，但尚未实际完成官方 MTR 全量推理验证。

### 10.4 当前仍缺的内容

1. **原始 V2V4Real 数据**：当前容器没有 `/data1/Datasets/V2V4Real/train` 和 `/data1/Datasets/V2V4Real/test`。官方感知推理需要 `.pcd`、`.yaml` 原始数据；配置文件仍需改成实际路径。
2. **fused features**：本地没有 `preprocessed_data/v2v4real/fused_features_*`。它们不是新的模型 checkpoint，而是感知阶段生成的 BEV 中间特征。完整 `cobevt_c256` CMP 主模型的 Motion Aggregator 会用到它们。
3. **`cobevt_multiego_1x` 感知权重**：只有在复现 1x 无压缩 CoBEVT 感知/跟踪 setting 时才是硬缺口；当前 MTR 四组预测测试不依赖它。
4. **OPV2V 全套文件**：目前只准备了 V2V4Real；若要复现 OPV2V，还需另下 OPV2V 原始数据、感知权重、MTR 权重及其 Swin 预训练模型。

### 10.5 结论边界与建议顺序

- 当前不是“所有东西都缺”，而是 V2V4Real 的 4 个 MTR 权重、主要感知权重和预处理 cache 已准备好。
- 可以先尝试 MTR 单次 smoke test，验证环境、权重加载和 cache 链路；这一步的成功与否尚未在当前记录中宣称。
- 不能把没有 fused feature 的运行结果称为官方完整 CMP 复现结果。
- 若要完整复现：先准备原始 V2V4Real 数据，再运行 `multi_ego_inference_v2v4real.py --perception_model_name point_pillar_cobevt_multiego_256x` 生成 fused features，最后运行 MTR 四个官方测试命令。

官方参考：

- https://github.com/tasl-lab/CMP/blob/main/docs/prepare_dataset_checkpoints.md
- https://github.com/tasl-lab/CMP/blob/main/docs/perception_eval.md
- https://github.com/tasl-lab/CMP/blob/main/docs/prediction_eval.md
