# ToolV2X 项目状态（给后续 agent 的交接文档）

最后更新：2026-09-10。

**GitHub 首版已发布：** [HikiMaji/ToolV2X](https://github.com/HikiMaji/ToolV2X)，当前为公开仓库，分支 `main`。仓库提供源码、原框架依赖源码/配置、当前文档和精选运行文本；权重、完整数据、点云特征、缓存及重复运行快照留在工作站。审查入口见根目录 [README](../README.md) 和 [github_review.md](github_review.md)。本次核心资源路径适配后，62 项本机集成测试、隔离导出副本的 51 项轻量检查通过；原 MTR 对旧验证帧 21 个目标的重算与归档数组完全一致。详见 [准备核验记录](github_preparation_validation.json)。本次没有新 7B 生成或训练；下面历史实验报告的快照和绝对路径仍属于原运行。

**最新推进：紧凑证据与适配训练入口已实现。** 见 [evidence_adaptation.md](evidence_adaptation.md)。同信息紧凑格式在原 7B 上四动作均可解析；相同预算可保留记录数由 5/4/5/4 增至 19/9/19/8，但 P 增加目标后只输出五点，故默认仍是 JSON。较大压缩收益依赖当前两位小数舍入后近重合的 MTR 模态，不能泛化成一般压缩率或规划收益。

3027 个训练帧、108 个验证帧的因果输入索引和独立原 Q8/Q9 标签已落盘；未读取测试数据。另一个真实训练帧物化了四动作共 8 条监督样本，实际原模型监督前向 loss 均有限，提示/证据/点云位置均屏蔽损失。58 项测试与独立复核通过，优化器步数为 0。产物在 `outputs/adaptation_data_v1/` 和 `outputs/adaptation_forward_v1/`。下一步先适配原 MTR，再物化全量工具证据并训练共享驾驶模型；完整 SFT 数据和训练优化器尚未完成。

**此前完成的 JSON 接入基线。** 用户批准设计后，已实际执行 CMP 原 MTR、独立 P/F 报文、P 本地重算、源分离接收，以及完整原 V2V-GoT 7B + LoRA 的 Q8→Q9。在一个真实验证决策帧上，Ego/P/F/PF 四动作均输出可解析行为和六点轨迹；当时的 48 项测试及落盘独立复核通过。完整报告见 [planning_connection.md](planning_connection.md)，运行产物在 `outputs/framework_connection_v3/`。

**当前未完成适配训练、正式规划质量评价、学习查询策略和闭环。** 接通和监督前向不能宣称方法有效。完整 P 历史本地重算与 F 在 30 个邻车目标上预测/分数完全一致；这只验证同信息计算等价。MTR 权重元数据 epoch=1、it=6，训练来源及算子适配与原 CUDA 的一致性未核实；V2V-GoT 也未针对新证据训练。默认 JSON 在该验证帧保留 4–5 条目标记录，紧凑候选增加至上面的数量，完整账本与舍弃项均已保存。

紧凑表示和监督边界已按上条推进，训练与质量对照尚未完成。两车 P/F 继续作为实施范围，RSU/I 后置，最终机制未确定。恒速/岭回归只保留为历史诊断或简单对照，其结果不能否定原模型方向。

新 agent 阅读顺序：

1. [user_requirements.md](user_requirements.md)：用户授权与最新下载分工。
2. [evidence_adaptation.md](evidence_adaptation.md)、[planning_connection.md](planning_connection.md)：最新表示/训练准备及此前实际接入记录。
3. [framework_design.md](framework_design.md)、[planning_implementation.md](planning_implementation.md)：已批准的目标设计与已完成接入范围。
4. [resource_downloads.md](resource_downloads.md)：当前资源。完整 LLaVA/CLIP 已从本机缓存链接并成功离线加载，无立即下载缺项。

用户允许 GPT-5.6-Luna Max 资源子 agent；github.com、githubusercontent.com、githubassets.com、huggingface.co 相关下载交给用户。不要重复下载已存在的模型，也不要通过镜像/跳转绕过分工。

维护原则：**只写已核实的事实和已做出的决定**。下面是 2026-09-09 及早期推进的历史正文，其中“CMP 单仓库 + 自建规划器”“用小模型 oracle 决定方向生死”“路由已经确定”“尚无权重/规划推理”等均不能当作当前设定；旧四配置训练脚本也仍有未解决的 GT/协议问题，不能直接启动为正式实验。以本文顶部和新接入报告为准。

---

## 0. 一句话现状

方案已从「V2V-GoT 底座 + 异构能力路由」调整为「**CMP 单仓库流水线（CoBEVT 检测 → AB3DMOT 跟踪 → MTR 预测）+ 自建 LLM 规划器**，在 V2V4Real 上验证 P/F 路由」。
当前阶段：**数据/环境准备**。目标是先跑出一个**不需要 LLM 的 oracle 实验**（见 §5），用它决定方向生死。

---

## 1. 已确定的研究设定（经过多轮评估后的结论）

- **叙事**：不讲「远端节点能力异构」（V2V4Real 只有两辆同构车，讲不通），改讲「**规划驱动的协作消息抽象层级选择**」——P（远端感知，回答"现在有什么"）与 F（远端预测，回答"未来会怎样"）是同一远端节点的两个抽象层级，差异在带宽、时延、灵活性、对远端模型质量的依赖。
- **必须有显式 cost**（字节数/时延），主实验报 cost–accuracy Pareto 曲线，否则 Always-PF 就是上界、路由无存在理由。
- **路由不用 prompt 让 LLM 自由选**，改为在 LLM 规划 hidden state 上接小 routing head，oracle 标签监督；同时必须报强 Rule-based 基线（置信度 + 预测熵阈值）。
- **第一版路由空间**：{STOP, P, F, PF}，P/F 独立可调用，不强制 P→F。Sequential（P 结果决定是否调 F）用 oracle 数据先做相关性分析再决定是否保留。
- **对照组**：Ego-only / Always-P / Always-F / Always-PF / Rule-based / ToolV2X / V2V-GoT（固定多阶段 LLM 协同推理）。
- **强度判断**：oracle 实验正面 → CCF-C 有把握，冲 CCF-B 需要 cost 曲线 + occlusion-critical 子集 collision rate。
- **V2X-Real + RSU** 只作为后续增强，不进第一版（见 §6）。

---

## 2. 底座选型的核实记录（避免重复踩坑）

| 候选 | 核实结果（2026-09-09） | 结论 |
|---|---|---|
| **V2XPnP** (github.com/Zewei-Zhou/V2XPnP) | 克隆后只有 15 个 py（OpenCOOD 工具函数），`datasets/__init__.py` 引用的 5 个 dataset 文件不存在，无 yaml/模型/权重。README 的 "2025/12 Codebase 1.0" 未勾选。**代码未发布**，假定不会更新。 | ❌ 不可用 |
| **TurboTrain** (ucla-mobility) | 只有 README | ❌ |
| **QuantV2X** (ucla-mobility) | 300 py / 80 yaml，完整；支持 V2X-Real(VC/IC/V2V/I2I)、OPV2V、DAIR-V2X；**只做检测，无 V2V4Real 加载器，无预测** | 仅用于后续 V2X-Real 的 P |
| **V2V4Real 官方** (ucla-mobility/V2V4Real) | 81 py / 10 yaml，OpenCOOD 分支，检测可用 | 已包含在本机 V2V-GoT/DMSTrack/V2V4Real 内 |
| **CMP** (github.com/tasl-lab/CMP) | GitHub API 确认顶层含 `AB3Dmot/ DatasetPreprocess/ MTR/ Plotter/ opencood/ preprocessed_data/ docs/ environment.yml`。论文：检测 = 修改版 CoBEVT，跟踪 = AB3DMOT（无需训练），预测 = MTR，历史 1.0s / 未来 5.0s，含 SinBEVT 无协作基线（V2V4Real minADE6@5s: SinBEVT 5.0250）。**未见预训练权重声明**。 | ✅ 主底座 |
| **V2V-GoT** | 本机已跑通（见 §3） | 保留：waypoint 生成参考 + 对照 baseline |

近邻工作边界见 `docs/analysis_report.md`、`docs/literature_screening.md`；PDF 在 `docs/papers/`。

---

## 3. 本机资源盘点（2026-09-09）

### 硬件 / 环境
- GPU：**1× RTX 4090 24GB**，空闲。
- 磁盘：`/root/autodl-tmp` 185G，剩 ~79G（2026-09-09 清理后）。`/` 剩 9G。
- conda 环境（`/root/autodl-tmp/conda-envs/`）：
  - `dmstrack`：py3.7.16, torch 1.12.0+cu113, spconv-cu113 2.1.25, numpy 1.21.6, open3d 0.17, filterpy, numba, shapely。**GPU 可用**。用于 OpenCOOD / AB3DMOT / 数据处理。
  - `llava`：torch 2.1.2+cu118, transformers 4.37.2。**GPU 可用**。用于 LLM 训练推理。
  - 另有 `/root/miniconda3/envs/{LangCoopCarla, tcp_codriving}`，与本项目无关。
- 网络：GitHub / HuggingFace 需 `source /etc/network_turbo`，**用完立刻 `unset http_proxy https_proxy`**（代理会拖慢其他网络）。

### 数据（全部在 `/root/autodl-tmp/V2V-GoT/`，已解压）
**V2V4Real 原始点云不在本机**（找不到 .pcd），但 **V2V-GoT 已落盘了 5 种检测配置的逐帧输出**，这就是我们的 P 原料：

| 目录 | 含义 | 在 ToolV2X 中的角色 |
|---|---|---|
| `no_fusion_keep_all/npy/{ego,co_llm,1}` | 单车检测（test） | **Local P** |
| `cobevt/npy/{ego,co_llm}` | CoBEVT 协作检测（test） | **Remote P**（与 CMP 同检测器，对比干净） |
| `early/`, `attfuse/`, `v2xvit/` | 其他协作检测（test） | 备选 Remote P |
| `train_*` 同名目录 | 训练集版本 | 训练 MTR / routing head 用 |
| `*/net_epoch60.pth` | 各检测器权重 | 已有，不需重训检测器 |
| `*/{eval,short_eval,middle_eval,long_eval}.yaml` | 0.6–2.9MB，内容待盘点 | — |

每帧文件：`XXXX_{gt, gt_object_id, lidar_pose, pred, pred_score, projected_lidar, transformation_matrix}.npy`，`co_llm/` 下另有 `XXXX_detection_box_score.npy`。
test 帧数：`no_fusion_keep_all/npy/ego` 约 1993 帧（15947 文件 / 8 类）。**shape / 坐标系 / 帧-场景映射待盘点（后台 agent 正在做，结果写入 `docs/data_format_v2vgot_npy.md`）**。

其他：
- `V2V-GoT/DMSTrack/`：含 `AB3DMOT/`、`DMSTrack/`、`V2V4Real/`(OpenCOOD 分支)。
- `V2V-GoT/LLaVA/`（45G）：LLaVA 1.5 + V2V-GoT 微调 checkpoint。
- `V2V-GoT/dataset_processed_features_and_gt.zip`（21G）与 `.root-partial.zip`（9.5G）：已解压，**可删以腾空间**（root-partial 疑为不完整重复下载；删前请用户确认）。
- `V2V-GoT/dataset_jsons.zip`：V2V-GoT-QA / V2V-QA。

---

## 4. 待办与分工

| # | 任务 | 状态 | 产出 |
|---|---|---|---|
| 1 | 克隆 CMP，摸清 MTR 输入格式、config、AB3DMOT I/O、聚合模块、权重 | ✅ | `docs/cmp_repo_notes.md`；仓库在 `/root/autodl-tmp/CMP` |
| 2 | 盘点 V2V-GoT npy 格式、帧-场景映射、waypoint GT 代码 | ✅ | `docs/data_format_v2vgot_npy.md` |
| 3 | 验证 dmstrack / llava 环境 GPU | ✅ | 本文 §3 |
| 4 | 统一读取层 + AB3DMOT 世界系跟踪脚本 | ✅ | `src/common/v2v4real_meta.py`, `src/tracking/run_ab3dmot.py` |
| 5 | 跑 6 组跟踪（no_fusion / no_fusion_cav1 / cobevt × test / train） | ✅ 6 组全部完成 | `outputs/tracks/{split}/{config}_world.pkl`，日志 `outputs/logs/track_all.log` |
| 6 | **决定 MTR 是否需训练** | ✅ 自训（§4.1） | — |
| 7 | tracks.pkl → CMP MTR 轨迹 pickle 适配器 | ✅ test 集 4 份已生成 | `src/prediction/tracks_to_cmp.py`；产出统计见 `pipeline_spec.md §3.2` |
| 8 | 新建 `cmp` conda 环境 | 🔄 后台安装中 | `conda-envs/cmp`，日志 `outputs/logs/setup_cmp_env.log`，脚本 `outputs/logs/setup_cmp_env.sh`；末尾出现 `ENV_DONE` 即成功 |
| 9 | MTR 四个 cfg + 去 BEV feature 依赖 | ✅ | `CMP/MTR/tools/cfgs/toolv2x/{local_f_egoP, local_f_coopP, remote_f_egoP, remote_f_coopP}.yaml`；CMP dataset 打了一个小补丁（`ALLOW_MISSING_FUSED_FEATURE`，缺 BEV 时喂全零），`cd CMP && git diff` 可见 |
| 10 | V2V4Real 目录骨架（空 yaml，替代原始点云） | ✅ | `src/prediction/make_v2v4real_stub.py` → `outputs/v2v4real_stub/` |
| 11 | MTR 训练 / 测试一键脚本 | ✅ 未运行 | `src/prediction/run_mtr_stage.sh {prepare|train|test}` |
| 12 | 生成 train 集轨迹 pickle | ✅ | `outputs/cmp_trajs/*/train/`（no_fusion 1027 objs / cobevt 1322 / cav1 1069），合并目录 `remote_egoP`, `remote_coopP` 各 64 文件 |
| 13 | 训练 4 个 MTR（顺序：local_f_* → remote_f_*） | ⏳ | `run_mtr_stage.sh train <cfg>`，ckpt 在 `CMP/MTR/output/<cfg>/default/ckpt/` |
| 14 | oracle 分析脚本（规划器 + 四配置对比 + 四个判据数字） | ✅ 已写好并用 GT 输入 smoke test | `src/oracle/oracle_analysis.py`，等 #13 的推理结果后运行，输出 `docs/oracle_results.md` |

### 4.0.1 一个重要的数据事实：GT 里包含 CAV 自身
V2V4Real 的 GT 框包含 ego 自身（距 ego 原点 <1.5m 的框，test 集 80% 的帧都有）和 CAV1。做规划碰撞检测时必须排除，否则 ego 永远"撞上自己"。`oracle_analysis.cav_self_ids()` 用 ego 原点与 CAV1 位置（`1/<t>_transformation_matrix.npy` 的平移）半径 1.5m 内的 GT id 排除。排除后，用 GT 他车做规划，27 个抽样帧里 26 帧选择全速沿 GT 路径，平均 waypoint 误差 0.2m，规划器行为正常。

### 4.0 MTR 四个 cfg 与四配置的对应

| cfg | 感知输入（pred_traj_dir） | 聚合器 | 对应 ToolV2X 配置 |
|---|---|---|---|
| `local_f_egoP` | ego 单车跟踪 | None | **Ego-only** |
| `local_f_coopP` | cobevt 协作跟踪 | None | **+P** |
| `remote_f_egoP` | ego 单车跟踪 + CAV1 单车跟踪（`remote_egoP/` 合并目录） | Transformer | **+F** |
| `remote_f_coopP` | cobevt 跟踪 + CAV1 单车跟踪（`remote_coopP/`） | Transformer | **+PF** |

GT 统一用 `outputs/cmp_gt_radians/`（CMP 的 cav0 GT 文件复制为 cav0 与 cav1 两份，全部弧度，规避 CMP 原 cav1 文件的度数问题）。remote_f_* 的 `PRETRAINED_MOTION_TRANSFORMER` 指向对应 local_f_* 的 epoch 30 ckpt，与 CMP 的两阶段训练一致。

### 4.1 关于"是否需要训练 MTR"——结论：**自训**（已决定）

- CMP 仓库内无 `.pth`；官方权重在 Google Drive，**本机两种网络方式都不可达**。
- 即便拿到权重也不兼容：CMP GT pickle yaw 单位不一致（cav0 弧度、cav1 度，跟踪结果度），且聚合器依赖需原始点云生成的 BEV feature。详见 `pipeline_spec.md §3, §6`。
- 自训规模：test 集 MTR 可用样本 1.3–1.7 万 (obj,t) 对，train 集约 4–5 倍；BATCH 1 × 30 epoch，4090 单卡预计每个 cfg 数小时到半天，四个 cfg 顺序跑（remote 依赖 local ckpt）。

### 4.2 磁盘
用户已清理，`/root/autodl-tmp` 现剩 ~79G（两个大 zip 已删）。

---

## 5. 第一个决定性实验：oracle 路由（不需要 LLM）

对 test 集每帧跑 4 配置：Ego-only / +P / +F / +PF，用**同一个简单规划器**（先用 V2V-GoT 已有 waypoint 逻辑或规则规划器）生成 ego waypoint，计算：

1. **Oracle 上界**：逐帧取 4 者最优 vs Always-PF vs Ego-only。若 oracle 相对 Always-PF 提升可忽略且 Always-PF 相对 Ego-only 也微小 → **停止该方向**。
2. **P/F 可分性**：P-only 最优、F-only 最优的样本占比及误差差距分布。**低于 10–15% → 核心假设不成立**。
3. **成本节省空间**：oracle 下 STOP / 单工具比例 → 带宽节省。
4. **Sequential 是否值得**：P 结果（新增检测数、置信度变化）与"F 是否有帮助"的相关性；弱则砍掉 sequential。

若方向被证伪，已搭好的 CMP 流水线转做「协作预测对 LLM 规划器的增益」。

---

## 6. 后续增强路径（不进第一版）

V2X-Real 有 2 CAV + 2 RSU（路口场景四者同时在线，走廊场景只有 2 CAV），RSU 传感器（Ouster 40m 俯视）与车载（RoboSense 200m）真实异构。VC 子集 = ego 是车、协作者含 RSU。
- P：QuantV2X（代码完整，原生支持 V2X-Real）
- F：V2XPnP-Seq **数据**已放出（68 场景轨迹 + 地图，UCLA Box），把 CMP 的 MTR 迁到该轨迹数据上训。
- 原版 V2X-Real 无跟踪 ID / 预测 / 规划标注；LLM 层无任何现成工作，需自建。

---

## 7. 文件索引

- `docs/STATUS.md`（本文）— 交接总览，**新 agent 先读这个**
- `docs/pipeline_spec.md` — 四配置定义、数据流、CMP 轨迹 pickle 格式（含 yaw 单位 bug 记录）、序列号对照表、MTR 环境与权重问题、oracle 实验设计
- `docs/data_format_v2vgot_npy.md` — V2V-GoT npy 每类文件的 shape/坐标系/公式、帧-场景映射、waypoint GT 代码位置、可运行示例
- `docs/cmp_repo_notes.md` — CMP 仓库结构、MTR dataset/模型/训练入口、AB3DMOT 修改点、最少改动清单（注意：其中"cav0 GT yaw 为度"的说法不准确，以 pipeline_spec §3 为准）
- `docs/user_requirements.md` — 用户原始需求
- `docs/analysis_report.md`、`docs/literature_screening.md` — 文献与边界分析
- `docs/papers/` — 近邻论文 PDF 与文本
- `docs/evidence/v2v_got_inference_source.py` — V2V-GoT 推理源码摘录
- `start.md` — 早期完整方案设想（部分叙事已被 §1 修正，以 §1 为准）
- `src/common/v2v4real_meta.py` — 唯一的数据读取入口（路径、len_record、角点→中心、ego↔world、ego 未来 waypoint）
- `src/tracking/run_ab3dmot.py` — 跟踪脚本，用法见文件头
- `outputs/tracks/` — 跟踪结果 pkl；`outputs/logs/` — 运行日志
- `/root/autodl-tmp/CMP` — CMP 仓库克隆（浅克隆，含 `preprocessed_data/v2v4real/gt_multiego_speedless` GT 轨迹）

## 8. 运行约定

- 数据处理 / 跟踪：`/root/autodl-tmp/conda-envs/dmstrack/bin/python`（不要 `conda activate`，直接用绝对路径）。
- 网络：只在访问 GitHub / HuggingFace 时 `source /etc/network_turbo`，之后立即 `unset http_proxy https_proxy`。Google Drive 两种方式都不通。
- 不修改 `/root/autodl-tmp/V2V-GoT/` 下任何文件；所有新产物放 `ToolV2X/outputs/`。
