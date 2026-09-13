# T8 实施记录：普通请求价值模块与冻结策略接口

日期：2026-09-13。范围：T8 代码、CPU 合成契约与回归验证。没有运行真实录制上的价值训练、真实分支采集、GoT/MTR 训练或推理，没有重建缓存，没有进入 T9 或推送 GitHub。

## 实际完成

ToolV2X 主线保持为参数化 P/F 请求、真实返回、同一 driver 修订、依修订再请求或 STOP，最多两次远端调用。T8 只提供标准监督求解器，不把小网络结构包装成创新。

- `src/planning/query_value.py`：71 维因果状态特征，附加 6 维动作编码，普通 `77 → 64 → 64 → 1` MLP；默认 9,217 个参数。P/F 与 current/change/old/union 使用同一动作条件网络。主方法实际合法集合仍为 STOP/P_current/F_current/P_change/F_change；old/union 只用于已有对照。
- 特征包括 old/current 共 24 个坐标、运动与缺失位、方案差分、local/acquired/derived/admitted/dropped 字段计数、仅由已知 anchor/forecast 计算的关系距离、剩余资源、首工具、阶段和信息可用性标记。字段身份、raw 文本、GT、离线损失、未执行候选的实际成本均不输入网络。
- STOP 固定为 0，不训练 STOP 输出。全负/零值时 STOP；正值平局按版本化动作顺序，bundle 再按发送候选顺序。executor 的可行集合与本地可见预算共同屏蔽动作，初始不可 change。
- T7 标签按真实状态、动作、续动作、终态与前缀后实际费用重新绑定。先筛物理录制训练组，再求规范化统计；首步教师同时排除自身折与共享策略的外层留出组。共享网络每次更新各抽一批 first 和 terminal，不让样本数量较多的一层淹没另一层。
- checkpoint 保存网络、Adam、步数、损失轨迹、两层抽样次数、NumPy/Torch RNG、规范化、训练配置和实际训练表。第 0 步先落盘，再按 `checkpoint_every` 保存完整状态；恢复时比对实际训练表和配置，禁止换数据续跑。保存采用同目录临时文件替换。
- `run_task_episode(..., policy=loaded_policy)` 在 driver 调用前核对模型/receiver/query/control 绑定，并记录策略来源。`make_control_policy` 透传该检查。现有 `interact` CLI 仍为诊断入口；本批提供的是可部署 Python callable，不自动切换真实运行配置。

## 新增 API 与版本化记录

```python
feature_spec()
state_features(visible_state, feature_spec)
feasible_queries(visible_state)
choose_query(values, feasible_actions)
training_config(checkpoint=..., train_recordings=..., binding=..., utility_spec=..., ...)
training_examples(targets, kind, config)
fit_terminal_policy(targets, train_recordings, config)
fit_shared_policy(first_targets, terminal_targets, config)
load_query_policy(checkpoint, expected_binding=..., policy_kind=...)
bundle_features(visible_state, action, feature_spec)
fit_bundle_continuation(targets, train_recordings, config)
compare_training_contracts(left.training_contract, right.training_contract)
register_value_continuation(service, checkpoint, expected_binding)
```

主要记录为 `toolv2x_query_features_v1`、`toolv2x_value_actions_v1`、`toolv2x_query_training_v1`、`toolv2x_query_checkpoint_v1`、`toolv2x_frozen_value_source_v1`、`toolv2x_continuation_v2`、`toolv2x_query_policy_v2`、`toolv2x_bundle_targets_v1`。

教师的 `teacher_id` 绑定持久化 fit ID 与保存步数。`frozen_source` 保存完整不含 checkpoint 位置的训练规格、有序特征名和规范化；具体路径仅为文件位置，不作语义身份。fit ID 是本地训练过程标识，不是内容指纹，也不证明人为修改过的文件可信。T7 兼容原 `toolv2x_continuation_v1` 测试契约，新增 v2 的快照来源校验。

## 条件单轮对照的边界

`BundlePolicy` 是独立拟合的对照副本，复用相同网络容量与数值特征定义。它只读取已发送摘要/候选以及实际首返回。对每个候选，用其已发送数值轨迹和首请求轨迹形成特征；不能补入真实 ego 修订 τ1、未发送本地完整上下文或未执行 F。其 admitted/Z 在此时不可知，使用明确 missing 标记；不能把未知解释为已入模数量为零的观察结论。

加载类型必须为 `bundle_terminal`。provider 注册和实际执行前均核对真实 FrozenPredictor 的版本/设置与任务 provenance；预算模式、内外容量、候选生成配置、首请求与摘要上限必须匹配 checkpoint。不会因为调用方传了旧 `expected_binding` 就信任不同的实际提供方。当前运行框架对两车采用同一版本化任务 provenance；异构提供方身份尚不在这一首版绑定中。

独立 bundle 标签表要求：

- `kind=bundle_terminal`，绑定自己的 control、utility、录制折和监督来源；不能把 T7 的 alternating terminal 表改名复用。
- 每组共享同一 actual envelope/first_response；STOP 的终态基准是“只返回首结果，再执行最终 driver”。
- 候选标签使用相对该 STOP 的损失变化和外层 wire / 完整终态计算增量；所有公开可执行候选与 STOP 必须覆盖。计算时间增量允许有符号，因为同一次最终 driver 的耗时可能变化。
- 表适配器验证结构、同前缀、动作覆盖、录制和效用算式。它不是 bundle 分支采集器，也不代替真实原始终态/成本的实验审计。本轮仅使用 `origin=synthetic_contract`。

**未完成、也未声称完成的实验工作：** T7 当前仍不采集独立 bundle 条件分支监督；尚无真实 one-shot 训练表、首请求/continuation 的公平拟合结果或共同 rollout。因此不能把上述接口称为已经训练好的 strong one-shot。后续需在获准的有限真实采集阶段补对应监督并冻结采样/更新预算，不能借用交替 τ1 的标签。

`training_contract` 比较特征、容量、动作定义、录制组、优化配置、更新数、首末层抽样数、候选监督数量、receiver/query 预算、共同效用与底座身份。不同 control 模板不会被当成模型不一致；真实数据量/更新预算不同会明确报差异。比较器只报告，既不自动补平样本，也不把比较通过称为实验公平性或效果证明。

## 文件改动与计划调整

- 新增：`src/planning/query_value.py`、`tests/test_query_value.py`、本记录。
- 接入：`src/planning/method_episode.py`、`src/planning/method_controls.py`。
- 审查后必要补充：`src/planning/query_data.py` 兼容并校验具体冻结教师来源；`src/tools/vehicle.py` 和 `src/tools/control_bundle.py` 增加学习对照的 provider/执行规格绑定钩子。原 P/F 执行、v1 query/decode、排序和装包语义不变。
- 更新：`scripts/check_review.py`、`docs/STATUS.md`、原 T1–T9 计划的 T8 状态。

上述附加修改用于固定监督和实际执行身份，没有增加第三套机制、预测骨干、联合 RL、RSU/I 或 peer 学习排序器。数据/归一化/检查点/绑定属于工程基础，bundle 副本和训练比较器属于对照代码。

## 验证

主工作区完整轻量检查：271/271，通过；其中 T8 新增 12 项不导入 torch 的检查。独立 CPU 测试组新增 9 项，通过；包含小网络实际回归、T7 教师到 first/shared 的接入、保存重载、精确恢复、中断前检查点、录制隔离、终态标签污染、provider/预算版本漂移。测试分支和模型均在临时目录，predictor/driver 是契约替身；不能解释为模型效果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/check_review.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_query_value.ModelTests test_query_value.RecoveryTests \
  test_query_value.ReviewRegressionTests test_query_value.BundleModelTests -v
python outputs/framework_epoch01_quick_eval_2026_09_12/verify_review.py
```

旧归档 160 条 raw answers、32 帧和五策略 ADE/FDE 复算通过。与 T8 前基线比较，原 v1 `query/decode_response/make_evidence`、`choose_action/run_episode` 和三个旧评价函数共 8 个 AST 不变。未再次运行真实 tokenizer/模型权重相关全量集成，轻量通过不代表真实模型运行。

独立审查确认的四项缺口已局部修补：教师具体来源、首步终态/费用重绑定、实际 provider/单轮执行绑定、公平性中的共同效用/底座身份。最终复审与主工作区交付核验另记下方。

最终独立复审：原 4 项问题全部关闭，另行运行 6 项定向测试通过；复现确认注册后 provider 漂移仍在私有读取前拒绝，恢复保留 fit ID 并推进 step 身份。没有剩余必须修复项。

最终交付：11 个 T8 文件已同步本地 main，源码和测试与工作树逐文件一致。主工作区完整轻量 **271/271（104.720 秒）**；工作树专门 CPU 组 **9/9（115.593 秒）**，最后元数据接口调整后在 main 重跑受影响的 3 项 CPU 接入测试，**3/3（31.763 秒）**。另外 2 项只接收深拷贝元数据、不能改宿主状态的测试由独立审查再次通过，保留原四项关闭结论。绑定 hook 不向策略传递 driver/predictor 对象。旧证据复算通过，8 个 v1 AST 与用户原审查文档保持不变。未提交、未推送，停在 T8。
