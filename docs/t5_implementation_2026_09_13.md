# T5：多次生成、失败与完整已测成本评价

日期：2026-09-13。用户授权“开始 T5”，本批到 T5 停止。没有训练、真实 MTR/GoT 推理、新方法效果实验或数据缓存重建。

T5 已增加独立的 v2 离线评价路径，能将 T4 归档转换为逐任务、逐阶段和汇总结果。全部初始/修订驾驶调用都计入费用，缺失与失败保留在预期分母中；原始轨迹误差、协议异常范围检查和任务是否正常结束分别报告。评价通过不证明真实模型能够利用证据或方法有效。

## 1. 实际修改与 API

| 文件 | 改动 |
| --- | --- |
| `src/evaluation/framework.py` | 新增 v2 任务/阶段评价、非重叠成本汇总、完整预期任务读取和显式 CLI。三个旧 v1 评价函数保持原样。 |
| `src/planning/method_episode.py` | 输入构造失败时，把原已计入累计值的同一耗时也保存为 `input_build` 事件；没有更改模型、请求、receiver 或 STOP 行为。 |
| `tests/test_method_evaluation.py`（新增） | 16 项资源无关测试，包括旧 160 条原始回答的指标兼容。 |
| `tests/test_framework_pipeline.py` | 新增实际 `interact` 归档写入 → v2 离线评价集成；外部模型加载与生成使用测试替身。 |
| `scripts/check_review.py` | 注册 T5 测试，不引入 PyTorch/Transformers。 |
| 本报告、`docs/STATUS.md`、执行计划 | 记录实际能力、统计口径与验证边界。 |

```python
method_plan_row(plan, label, motion, execution_spec)
evaluate_method_task(task, label, *, policy_id=None, branch_id=None)
summarize_method(rows)
evaluate_method(episodes_root, out, labels_root)
```

命令：

```bash
python src/evaluation/framework.py EXISTING_INTERACT_ROOT NEW_EVALUATION_OUT \
  --method --labels EXPLICIT_OFFLINE_LABELS_DIRECTORY
```

`labels_root` 中是 `train.jsonl` / `validation.jsonl`；只读本批预期角色所需的标签。调用者必须显式提供离线标签目录。旧命令不带 `--method` 时继续执行原 v1 评价。

结果使用 `toolv2x_method_evaluation_v1`，输出：

- `rows.jsonl`：每个预期 `(sample_id, policy_id, branch_id)` 一行，含失败、缺失和归档问题。
- `plan_rows.jsonl`：实际记录到的各次 driver 回答、解析、异常范围检查及原始轨迹指标。
- `summary.json`：分 policy/branch、物理录制组汇总，给出预期次数、成功/失败、标签覆盖、已知/未知费用数量和归档问题。

评价不读取模型权重、输入特征、peer 数据缓存或恢复样本的 `artifact_root`。T4 已结束样本复用原产物时，现有 JSON 回答与费用足以评价。

## 2. 质量与失败的口径

| 字段 | 含义 |
| --- | --- |
| `parse_valid` | 最后一次实际记录的 driver 回答是否可解析；服务失败后的最后回答可能只是此前合法前缀。 |
| `admissibility` | 已解析路径是否满足该 episode 完整 ExecutionSpec 的数值、速度和加速度范围。不是碰撞或闭环安全判断。 |
| `task_success` | 实际终态是 STOP，最终指针对应最后一次真实有效回答，已有阶段满足契约且归档一致。不是误差低于某个阈值的驾驶成功率。 |
| `ADE3/FDE3` | 成功终态对应的平均/末点位置误差；失败不以 `last_valid_plan_id` 回填成功方案。 |
| `raw_ADE3/raw_FDE3` | 最后记录回答的诊断误差；各阶段也有独立原始指标，可能属于失败样本的前缀，不混进成功终态指标。 |
| `label_status` | complete / partial / no_valid_points / missing / identity_mismatch / invalid / duplicate。缺标签不改变模型解析或执行成功状态。 |
| `artifact_status` | completed / recorded_failure / incomplete / missing_artifact / duplicate_artifact / invalid_artifact。 |

复用原 `evaluate_plan` 的回答一致性检查及 `trajectory_metrics` 数值定义。ADE3 在有效未来点上计算，最后一点无效时 FDE3 为未知；无有效标签时二者为 `None`，不生成替代真值。每项指标都报告有效数量。

标签按角色和 sample_id 查找，并核对 g 及存在的 scene 字段；不能用 g 单独索引不同录制的数据。未来标签始终只在本离线评价路径读取，不进入 T4 决策。

预期分母来自 T4 `selected_index.jsonl` 和运行配置。T4 CLI 当前每个运行只有一个 policy，branch_id 与 policy_id 相同；后续 T7 的多分支树需要自己的预期分支清单，不能拿现有单策略清单推断未声明分支。

缺少 manifest 行、缺文件、重复记录和中断不会缩小分母。无法确认唯一产物时不擅自选取一个成功结果，保留该预期行及产物路径用于诊断。`evaluation_completed` 只表示本次读取评价完成；源运行未完成时，原 progress 和问题会显式保留。

已判为 `invalid_artifact` 的互相冲突记录，保留原始回答和费用诊断，但不进入阶段质量或成本均值；汇总包含排除数量。正常执行失败的真实前缀与已测费用仍保留。

## 3. 成本范围

`driver_calls` 是全部已开始的 driver 尝试，`calls` 是全部请求尝试；中断后的尝试计数不能解释为已完成 forward 数或 FLOPs。

`total_compute_seconds` 使用 `toolv2x_measured_stages_v1`：

```text
local_model_seconds
+ input_build_seconds
+ driver_seconds
+ service_seconds
+ receiver_seconds
```

- `driver_seconds` 汇总全部 driver 外层尝试时长。三个阶段为 1/2/3 秒时总计 6 秒，不仅统计最后一次。
- `generation_seconds`、`peer_model_seconds`、`receiver_model_seconds` 是嵌套诊断项，分别已包含在 driver/service/receiver 外层时长中，不能再加一次。
- receiver 的累计事件只从最后一份 ledger snapshot 读取，不能把每次累计快照再累加。
- `request_bytes/response_bytes` 核对实际 JSON 请求编码与原始 wire 字节长度；没有实际响应，也没有合法的已执行 provider 记录时，不能沿用旧累计 0 冒充免费调用。
- `input_tokens/output_tokens/feature_tokens` 覆盖全部已记录生成。生成异常导致某项未知时，该项完整总量为 `None`，`known_cost` 保留已知前缀。
- 每项费用均有 `known_count/unknown_count/known_sum/mean_known`。未知不以 0 加入均值；归档互相冲突的数值也不当作可信均值。`cost_complete=False` 的行不能直接用于需要完整成本的后续价值监督。

这个总数是上述已测阶段的合计，不是端到端时延：当前未记录 policy/executor 的控制开销、写盘 I/O、模型加载或网络传输；`policy_seconds` 与 `end_to_end_seconds` 显式为 `None`。本地共享 MTR 在每个对照任务中计一次，不能把它因被另一个任务复用而记为零。T8 引入实际请求价值网络后，其运行费用还需纳入明确的新口径。

## 4. 验证

| 检查 | 结果 |
| --- | --- |
| 修改前轻量基线 | 174/174 通过。 |
| T5 新测试 | 16/16 通过。 |
| T5＋framework pipeline | 23/23 通过。 |
| 完整轻量审查 | 191/191 通过，不导入 PyTorch/Transformers。 |
| 旧 160 条原始回答 | 新阶段 reader、原指标函数及已发布 ADE/FDE 逐条一致。 |
| 原便携复算 `verify_review.py` | 160 条回答、32 帧、五策略均值全部通过。 |
| v1 兼容 | `evaluate`、`summarize_episodes`、`validate_generation_run` 的 AST 与修改前一致。 |
| 独立子 agent 复审 | 定向 23 项通过，无剩余重要/严重问题。 |
| `git diff --check` / CLI help | 通过。 |

可运行：

```bash
PYTHONPATH=src:tests python -m unittest test_method_evaluation test_framework_pipeline -v
python scripts/check_review.py
python outputs/framework_epoch01_quick_eval_2026_09_12/verify_review.py
```

新增测试使用实际 T4 执行器、P/F service、ledger、receiver 与归档写入函数；昂贵模型使用测试替身，人工耗时只用于独立验算费用。归档/新 CLI 集成在临时目录中验证，不形成真实新方法实验结果。没有运行完整模型测试集，没有重算 MTR 或生成 GoT 回答。

覆盖了零/两次远端调用、三次 driver 累计、模型子项不重复、初始/中间无效输出、异常轨迹保留原始误差、生成异常的未知 token、请求/响应处中断、失败输入构造计时、缺标签与末点无效、同 g 不同录制、缺文件/重复产物、错误身份/冲突费用排除，以及原归档指标兼容。

## 5. 计划细化及结论边界

唯一涉及执行端的补充是 T4 `input_build` 失败的阶段计时记录：此前同一时段只在累计费用中存在。本批补记录，不增加执行或改动机制。旧归档若缺少足够阶段证据，评价保持未知，不补造历史数据或重建缓存。

本批属于评价工程基础，验证了完整分母、已测费用和原始误差的读取/统计前提。没有新 Method 创新、价值模块、收益结论或实时性结论。T6 的强单轮条件委托、同证据额外推理、冻结反馈等关键对照仍待后续授权实施。
