# ToolV2X T9 两帧真实 smoke 独立审计

审计日期：2026-09-13（Asia/Hong_Kong）
审计对象：`/root/autodl-tmp/ToolV2X/outputs/t9_real_smoke_2026_09_13_v1`
审计方式：只读；未运行模型、未训练、未扩大样本、未修改仓库或 Git。推理进程退出后才读取 `outputs/paired_driving_data_v1/offline_labels/validation.jsonl`，随后使用默认 Python/NumPy 1.24.2 独立重算；重算没有导入仓库 evaluator。

## 结论

**两帧真实接线 smoke：PASS。** 六个预声明任务全部形成终态，`completed_tasks=6`、`failed_tasks=0`、实际 GoT driver attempts=12。逐任务的原始回答、计划合法性、请求动作、外层与原语 wire、字节账本、P/F 计算边界、receipt/derived/admission 证据链以及离线 ADE3/FDE3 均通过独立检查。

这个 PASS 只说明冻结组件和协议在两个真实帧上接通。它不支持学习型请求策略有效、序贯调用优于单轮、协作带来平均收益、安全改善、闭环改善或测试集泛化。第二帧的协作终轨迹误差反而显著大于 Ego；正负结果均应保留。

## 冻结配置与实际模型

- 样本是两个 validation recording 各自索引中的最早合法帧：`g5526/local_frame=10` 与 `g7007/local_frame=10`。物理文件来自 train archive，但 research role 为 validation；未访问 test。
- GoT 使用 `outputs/framework_baseline_restart_v1/training/checkpoint-epoch01`，保存状态确认 epoch 1。CMP 使用 `outputs/mtr_stability_v1/full_seed20/best_model.pth`，实际加载元数据为 epoch 9、iteration 8964、版本 `ToolV2X causal MTR adaptation v1`。
- GoT 与 MTR 均是已有训练后冻结组件；本次没有更新参数。请求策略 `stop`、`p_current_f_change` 和 one-shot continuation `diagnostic_conditional_v1` 都是手写诊断规则，不是训练出的 value/query policy。
- 三臂共同使用 `max_calls=2`、`max_targets=4`、request/response/episode caps `16384/65536/196608` bytes，以及 receiver `context_limit=4096`、`generation_reserve=256`、`peer_reserve=1536`。
- one-shot 的 `per_rpc` 外层 response cap 为 65,536 bytes；扣除 2,048-byte wrapper reserve 后，两个内部 primitive cap 各为 31,744 bytes。实际 primitive wires 约 12.8 KB 和 14 KB，均未碰到该 cap。本次所有 P/F response 的 `truncated=true` 来自 `max_targets=4`，不是 byte cap。

## 真实输入与因果边界

- **没有真实 RGB 输入。** 实际 driver provenance 为 `actual_rgb_input=false`、`point_cloud_feature_input=true`。GoT 读取当前与前一帧的 `regression_map`、`classification_map`、`detection_box_score`，以及当前/前一帧 ego pose；LLaVA API 中的零图像张量只是接口占位，信息来自 shallow point-cloud/detection features。
- 同一帧三臂保存的 `ego_features.npz` 和 `local_forecast.npz` 逐数组完全一致。Ego 只读取 `no_fusion` 的所选窗口；两个工具臂额外读取同一时刻 `no_fusion_cav1` 的所选窗口。归档虽然包含其他窗口，但保存的 access ledger 明示且代码只把 `local_frame=10` 对应窗口送入模型。
- online window 的时间轴要求不大于 0；GoT feature/motion 只用 `t` 与 `t-1`。未来 ego pose 仅在推理进程退出后通过离线 label 文件读取，用于观察轨迹误差。
- 六个 episode 都保存 `gt_labels_read=false`、`execution_kind=original_got`。所有 GoT 输出均为 direct Trajectory：`q8_executed=false`，不存在 Q8 parent；Q9 raw 文本可独立解析成恰好 6 个有限二维点，并与保存的 `waypoints` 完全一致。

## 动作与因果链接

两个 Ego 任务均执行一次 GoT 后 `STOP(diagnostic_stop)`，没有 peer window 读取、请求或 response。

两个 alternating 任务的完整链为：

1. `plan_0` 产生后发送 `P(current)`，其 `tau_new` 逐点等于 `plan_0`，`tau_old=null`。
2. P receipt 经 receiver 更新后生成 `plan_1`。两个帧的 `plan_1` 都与 `plan_0` 实际不同，因此 `F(change)` 才成为可用动作并被诊断策略选择。
3. F request 的 `tau_old` 逐点等于 `plan_0`，`tau_new` 逐点等于 `plan_1`；F receipt 后生成 `plan_2`，随后因两次 capability allowance 用尽而 STOP。

两个 one-shot 任务均只有一个外层 RPC：bundle 的第一个原语为 `P(current, tau_new=plan_0)`；真实 P response 有 records 后，peer 内诊断 continuation 选择 `F(current, candidate_id=initial)`。第二个原语的 `tau_new` 仍是预先发送的 `plan_0`，并通过 actual P receipt manifest 证明已收到的字段；完整 bundle 返回后才进行第二次 GoT。它是“P 后条件 F”的单轮诊断执行，不是“看见新 GoT 计划后再选 F”。

所有 event IDs 连续，`caused_by_event_id` 无断裂；每个后续 plan 的 `request_id` 都指向实际先前 request。没有动作被 byte preflight 覆盖或静默替换。

## P、F 与 receiver MTR 边界

- 四个 P 原语的 provider record 均为 `model_seconds=0`、`model_targets_computed=0`。P 在 peer 端只按完整因果 tracking window 做目标关系排序并返回 anchor/history，**没有 provider/remote MTR**。
- P 到达后，receiver 在购得的 4-target history 子集上运行一次同一冻结 MTR；每个 P 对应 `model_calls=1`、`model_targets_computed=4`。因此只能表述为“P 无远端 MTR”，不能表述为“P 路径完全没有 MTR”。
- 四个 F 原语都在 provider 端先对完整 peer time-t window 运行冻结 CMP MTR，再按请求关系排序和 `max_targets=4` 打包。`g5526` 的完整 context 有 30 targets，其中 27 个走 MTR、3 个短历史 fallback；`g7007` 有 19 targets，19 个走 MTR。wire 内每个 forecast 均保存 `context_scope=provider_full_at_t`。
- F 到达 receiver 后，先前 P 子集预测 context 命中 cache，没有重复运行 receiver MTR。provider MTR 与 receiver MTR 时间是嵌套诊断项，未在 total compute 中重复加算。

## Wire、成本与证据账本重算

独立重算对每个外层 request 使用 canonical JSON（排序 key、紧凑分隔符、UTF-8），对每个保存的 `wire_hex` 恢复原始 bytes，检查重新编码逐字节相等，并逐项核对 provider primitive costs、episode aggregate costs 与 completed flags。结果如下：

| frame | arm | GoT plans | outer RPC | capability calls | request bytes | response bytes | 非重叠 total compute (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| g5526 | Ego | 1 | 0 | 0 | 0 | 0 | 2.0774 |
| g5526 | alternating | 3 | 2 | 2 | 4,122 | 26,961 | 6.7589 |
| g5526 | one-shot | 2 | 1 | 2 | 3,030 | 27,477 | 4.4135 |
| g7007 | Ego | 1 | 0 | 0 | 0 | 0 | 1.9005 |
| g7007 | alternating | 3 | 2 | 2 | 4,134 | 26,785 | 6.7300 |
| g7007 | one-shot | 2 | 1 | 2 | 3,059 | 27,561 | 4.3897 |

这里的 total compute 独立相加：每任务 local MTR、input construction、GoT driver attempts、外层 service attempts、receiver updates 和 control computation。它排除模型加载、持久化 I/O 与网络传输；generation、peer MTR、receiver MTR 是上述外层阶段中的子项。

最终 evidence ledger 与 driver Z 的检查结果：

- alternating 的最终 ledger：`g5526 acquired=14/derived=4/receipts=2`，`g7007 acquired=12/derived=4/receipts=2`。
- one-shot 的最终 ledger：两个帧均为 `acquired=13/derived=4/receipts=2`。
- 每个 acquired field 都带 actual remote receipt；每个 receiver-derived forecast 都带完整 parent refs，所有 parent refs 都能回指 acquired anchor/history；admitted refs 都属于最终 ledger，没有未购买 peer field 进入 prompt。
- post-P 的 driver prompt 在 8 个 remote units 中只保留 2 个：最近目标的一份 receiver-derived forecast 和一份 history。
- final prompt 在 12 个 remote units 中只保留 2 个：同一最近目标的一份 provider-full forecast 与一份 receiver-subset forecast。`g5526` 对应 target 20，`g7007` 对应 target 0。其余字段由 1,536-token peer reserve 明确丢弃并记录，不是静默丢失。
- 两帧中 alternating 与 one-shot 的 final remote Z 逐对象完全一致，尽管它们的 F mode、target ranking、原始 acquired field 集和 wire bytes 不同。由此得到的 final GoT 轨迹也逐点相同。这个结果表明本 smoke 没有提供两种机制的有效差异证据。

## 离线轨迹重算

标签是 6 个 0.5-second 间隔的未来 ego observed trajectory，六点均有效；它不是最优规划或安全标签。独立重算的终轨迹如下：

| frame | arm | final trajectory | ADE3 (m) | FDE3 (m) |
|---|---|---|---:|---:|
| g5526 | Ego | `[(7.6,0.3),(15.2,0.7),(22.8,1.1),(30.4,1.5),(38.2,2.0),(45.8,2.5)]` | 0.948969 | 1.854059 |
| g5526 | alternating | `[(7.6,0.3),(15.1,0.6),(22.7,0.9),(30.3,1.3),(37.9,1.7),(45.4,2.1)]` | 0.700788 | 1.292016 |
| g5526 | one-shot | 同 alternating | 0.700788 | 1.292016 |
| g7007 | Ego | `[(7.2,0.2),(14.3,0.5),(21.5,0.8),(28.7,1.0),(35.9,1.3),(43.1,1.6)]` | 0.410510 | 1.189432 |
| g7007 | alternating | `[(7.0,0.2),(13.8,0.4),(20.6,0.6),(27.3,0.8),(34.1,1.0),(40.0,1.2)]` | 1.623345 | 4.309533 |
| g7007 | one-shot | 同 alternating | 1.623345 | 4.309533 |

所有 12 次 GoT plan 的峰值速度与峰值加速度均低于冻结的 80 m/s 和 30 m/s² admissibility 上限。`g5526` 的 alternating 阶段 ADE3 为 `0.948969 → 0.852557 → 0.700788`；`g7007` 为 `0.410510 → 1.623345 → 1.623345`，即 P 后恶化且 F 没有改变最终轨迹。

## 失败与结论边界

本次没有动态失败案例：六个 task、所有 provider primitive、bundle、receiver update 和 GoT parse 都完成。因此“失败尝试会被保存”的代码路径和事件持久化设计可从源码与空的 failed ledgers 核对，但本批没有实际触发该性质，不能声称已用真实失败验证恢复行为。最终 `progress.status=completed` 应始终与 `failed_tasks` 和逐 task status 一起读取；该字段本身只表示 runner 到达终点。

本结果适合写成“两帧真实原模型协议/receiver 联调完成，完整正负轨迹与实际成本已留档”。不能写成 learned-policy evaluation、method benefit、one-shot fairness 胜出、闭环安全结果、测试集结果或论文主实验。若后续研究机制，首先要面对本次 final Z 在 1,536-token receiver budget 下塌缩为同一目标的两份 forecast，以及两臂在两帧产生完全相同终轨迹这一事实。

## 核对依据

- 冻结配置：`outputs/t9_real_smoke_2026_09_13_v1/launch.json`
- 终态与原始任务：`progress.json`、`tasks/*.json`
- provider 原语/单轮记录：`inputs/*/provider_records.json`
- 实际模型与输入：`models.json`、`inputs/*/ego_features.npz`、`inputs/*/local_forecast.npz`
- 推理日志：`inference.log`
- 推理结束后读取的标签：`outputs/paired_driving_data_v1/offline_labels/validation.jsonl`
- 代码路径：`src/planning/method_episode.py`、`src/planning/method_controls.py`、`src/planning/evidence.py`、`src/planning/context.py`、`src/tools/vehicle.py`、`src/tools/control_bundle.py`、`src/planning/run_framework.py`

## Offline reducer 最终交叉核验

主流程随后生成 `offline/evaluation_rows.json`、`summary.json`、`results.csv`、`trace_summary.json`、两帧轨迹图和显式 label-access record。该 reducer 在 `2026-09-13T14:45:58.343387+00:00` 开始读标签，晚于推理终止时间 `2026-09-13T14:40:56.496181+00:00`；记录为 `model_calls=0`、`training=false`、`online_inference_modified=false`。

我将 reducer 六行与上面的独立重算逐项比较。六行的 ADE3、FDE3、driver calls、outer RPC、capability calls、request/response bytes 和 total compute 均在 `1e-12` 绝对误差内完全一致。另从原始 `cost_events`/receiver ledger 重新求和核对 input/output tokens、generation time、peer MTR time、receiver MTR time 和 control time，也全部一致。六行终态均为：

- `artifact_status=completed`
- `task_success=true`
- `cost_complete=true`
- `parse_valid=true`
- `admissibility=true`
- `label_status=complete`
- `cost_issues=[]`

第二次 wire 的 acquired dedup 也通过逐字段 identity 检查。每个 P wire 首先发送 8 个新字段（4 个 target 的 anchor+history）；第二个 F request 的 manifest 都包含这 8 个已购字段。F wire 不重复发送已购 anchor，而以 references 指回它们，只发送 forecast 和此前未购 target 的必要 anchor：

| frame | arm | P 新 records | F 新 records | F references | final unique acquired |
|---|---|---:|---:|---:|---:|
| g5526 | alternating | 8 | 6 | 2 | 14 |
| g5526 | one-shot | 8 | 5 | 3 | 13 |
| g7007 | alternating | 8 | 4 | 4 | 12 |
| g7007 | one-shot | 8 | 5 | 3 | 13 |

每个第二 wire 的新 record identities 与首个 wire 完全不相交，所有 references 都属于首个 wire 且出现在第二 request manifest 中，最终 ledger 的 acquired identities 恰好等于两次新 records 的并集。这里的 references 只做已购字段认证，不会凭空增加 acquired value。

“最终 Z 相同”可以加强为“实际最终 GoT 输入 prompt 逐字相同”，但不能加强为“完整收据/账本相同”。在两个帧上，alternating 与 one-shot 的：

- local evidence 完全相同；
- final remote evidence Z 完全相同；
- evidence selection 完全相同；
- `q9_prompt` 字符串逐字相同，长度分别为 4,657 和 4,659 characters；
- final raw GoT answer 与六点轨迹完全相同。

两臂的 `admission_report` 不相同，因为 F mode/ranking、actual receipts、acquired fields 和 dropped-field ledger 不同。因此准确表述是：“不同原始通信轨迹经 receiver 选择后形成相同的最终模型可见 Z 与 prompt”，而不是“两个 episode 的全部 evidence/ledger 相同”。

推理日志中的 `2165 > 2048` 是 checkpoint `tokenizer_config.json` 所带 2,048 长度元数据触发的通用 tokenizer 警告，应该保留披露。实际 GoT base `max_position_embeddings=4096`，runner 将执行 context limit 固定为 4,096，且每次保存的 `q9_cost.input_tokens` 与 receiver 的精确 token count 一致。12 次生成的 input tokens 为 2,292–3,702；最紧的一次为 `3702 + 256 reserve = 3958 <= 4096`，仍有 138 tokens 余量。实际生成 output 为 63 tokens/次，没有截断、index error 或 context overflow，全部成功解析。因此该警告在本批是 **旧 tokenizer 阈值与实际 4,096 模型合同不一致造成的非致命提示**，不是本次真实上下文越界；后续报告不能简单删除警告，也不能把它写成已经发生 indexing error。
