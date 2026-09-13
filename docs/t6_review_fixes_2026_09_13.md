# T6 定向修复：单轮摘要、可支付动作与成本口径

日期：2026-09-13。依据用户提供的 [9-13-2 审查意见](9-13-2.md) 并获授权“开始”。保留 T1–T6 结构，修复三处可复现问题，增加明确的单轮聚合预算配置。没有进入 T7、恢复训练、调用真实 GoT/MTR、重建缓存或生成真实方法实验。

## 1. 已完成的修改

| 文件 | 实际改动 |
| --- | --- |
| `src/planning/method_controls.py` | 版本化摘要与响应预算配置；确定性选择本地完整字段；区分每包限制与 episode 聚合限制。 |
| `src/planning/method_episode.py` | 策略前绑定候选请求并过滤不可支付动作；保存原选择、执行动作与强制原因；分段控制计时排除 decision 快照复制与同步保存。 |
| `src/tools/control_bundle.py` | 新增 bundle v2 外层响应上限，独立约束原语、外层 RPC 和 episode 总字节；保留 bundle v1。 |
| `src/evaluation/framework.py` | 识别 bundle v2 的内部原语次数；旧未版本化控制时间仅供诊断，新的可信成本要求新计时版本。 |
| `src/planning/run_framework.py` | 归档完整摘要/预算配置；单轮分支区分 `one_shot:per_rpc` 和 `one_shot:episode_aggregate`。 |
| `tests/test_budget_review.py`、`scripts/check_review.py` | 新增 11 项定向回归并纳入常规轻量审查。 |
| 本文、`STATUS.md`、T6 实施记录、方法计划 | 记录修复后的当前语义、验证与边界；用户审查原文保持不变。 |

本批属于执行基础与实验对照修复，没有新增 Method 创新点、工具或网络。

## 2. 单轮摘要和预算

`control_spec('one_shot', bundle=...)` 会补齐并保存以下可修改配置。数值属于版本化实验配置，不是新增协议常量。

```json
{
  "summary_spec": {
    "version": "toolv2x_bundle_summary_v1",
    "selection": "distance_then_field",
    "field_kinds": ["anchor", "forecast"],
    "max_fields": 8,
    "max_bytes": 1536
  },
  "response_budget": {
    "version": "toolv2x_bundle_budget_v1",
    "mode": "per_rpc",
    "outer_response_cap": null,
    "primitive_response_caps": null
  }
}
```

`select_bundle_summary(request, local_fields, spec)` 按当前 anchor 距离、track handle、声明字段次序、字段身份确定性排序。选择完整字段，不做数值舍入；预测只随已经选入的同目标 anchor 发送。可配置仅 anchor 或 anchor 加 forecast，分别约束字段数、摘要字节和最终整个请求的 UTF-8 字节。初始轨迹已在 first_request/candidates 中，不再额外重复其原始文本。

外发摘要包含 ego_motion、所选 local_evidence 和 `summary_status`（版本、selected_fields、truncated）。完整本地字段继续保留在本车台账，外发字段内容与身份可从实际报文复查。配置连必需的运动/状态头都装不下时明确报错；若完整固定请求仍无法支付，进入明确的不可行动作记录，不承诺任意小上限都能发送。

采用默认 ExecutionSpec（上行 4096 B、普通响应 8192 B、总双向 24576 B）、两次原语及 wrapper reserve 2048 B 时：

| 条件 | 每原语响应上限 | 外层响应上限 | episode 总双向上限 | 可以怎样解释 |
| --- | --- | --- | --- | --- |
| 普通两次请求 | 各 8192 B | 每次 8192 B | 24576 B | 每次实际请求与响应都计费。 |
| `one_shot:per_rpc` | 3072 + 3072 B | 8192 B | 24576 B | 相同每包上限条件，不能直接归因为纯反馈差异。 |
| `one_shot:episode_aggregate` | 8192 + 8192 B | 18432 B | 24576 B | 相同原语上限、原语次数和总双向预算，允许一次聚合返回。 |

`null` 表示按所选模式和 ExecutionSpec 在发送前计算容量，实际数值写入请求 limits；也可预声明合法外层/原语上限。聚合模式没有提高 episode 总预算。wrapper reserve 是配置预留，并不替代最终校验：外层实际 UTF-8 wire 超上限时失败，已经执行的原语成本仍保留。内部编码长度保留为诊断，网络费用只计外部完整请求和响应，不重复累计内层字节。

新构造使用 `toolv2x_bundle_v2` / `toolv2x_bundle_response_v2`，limits 新增 `budget_mode`、`outer_response_cap`；原 v1 envelope/response 仍按原容量语义处理。主方法的 task v2 与旧 query/decode_response 不变。

默认两目标合成案例实际保留一个目标的完整 anchor 与 forecast：per_rpc 请求 2983 B，aggregate 请求 2994 B。20 目标案例也实际发出一次请求；另独立核对中文 scene/provider 的 UTF-8 长度及反转输入后的确定性。聚合测试完成一次外部 RPC、两次内部原语，实际返回超过 8192 B，仍满足外层 18432 B、每原语 8192 B 和总双向 24576 B。以上仅证明契约，不是实际数据失败率或模型收益。

## 3. 动作可行性与执行记录

新增 `prepare_request(state, action, request_id, manifest, control_spec=None)` 和 `request_budget(request, remaining_bytes)`。策略调用前，为每个候选绑定本车已有轨迹、真实回执 manifest 和冻结配置，检查实际序列化请求长度及公开响应预留。没有提前读 peer loader、predictor 或未知响应来估算可支付性。

策略只看到过滤后的 `available_actions`，以及 `action_feasibility` 中的请求字节、响应预留和不可行原因。已绑定请求在选中后复用，并保留发送前最终校验。长 change 不可支付但 current 可支付时，后者仍能被策略选择并真实执行。

每条 decision 保存：

- `action`：原策略选择；未调用策略时为显式执行器 STOP。
- `executed_action`：实际发送动作或 STOP；在持久化的待执行快照中允许为 null。
- `policy_called`、`forced_reason`：区分策略主动 STOP、调用次数耗尽、没有可支付请求。
- `request_preflight`、`infeasible_actions`：保留完整候选检查依据。
- `execution_override`：最终检查改变执行动作时保留原因，不覆盖原策略提议。

全无可支付请求时不调用策略，不能把该 STOP 当成学得的“无收益”。最终防线的独立故障注入也确认：原 P/current 保留，executed_action=STOP，未发请求且未调用服务。

## 4. 控制时间及旧归档

控制计算在 `emit('decision')` 前累计一次，回调返回后重新计时；保留快照与失败记录，排除快照 deep copy 和同步持久化回调。注入相同 2 秒策略计算，保存分别耗时 0 / 7 秒时，两次 `control_seconds` 均为 2 秒。

新增 control cost event 标记 `timing_version=toolv2x_control_compute_v2`。有 control 的新评价使用 `cost_scope=toolv2x_measured_stages_v3`。旧无标记事件仍可读取，原数值保留为 `reported_control_seconds` 诊断，但 `control_seconds` / `total_compute_seconds` 为未知、`cost_complete=False`，不能事后假称已扣除 I/O。旧归档和原始成本记录没有被覆盖。

该范围仍不是端到端时延；网络和未测 executor/I/O 不计入计算效用。服务、receiver、driver 与控制计算按已测非重叠区间计费，内部预测等子项不重复相加。

## 5. 验证

| 检查 | 结果与范围 |
| --- | --- |
| 修改前基线 | 226/226 轻量测试通过。 |
| 原问题复现 | 单轮默认预算、保存计时、可支付 current 三项先失败，再修复通过；旧计时解释也先失败再通过。 |
| 新增定向回归 | 11/11，通过真实执行器/服务/receiver API，驾驶与预测使用合成替身。 |
| 最终轻量审查 | 237/237 通过，检查入口确保未导入 torch/transformers。 |
| 旧证据便携复算 | 160 条原始回答、32 帧、Ego/P/F/PF/rule 各 ADE/FDE 与均值通过。 |
| v1 源码兼容 | vehicle.py 与旧 episode.py 文件保持原样；三个旧评价函数 AST 保持原样。 |
| 独立复审 | 有限范围 APPROVE，无未决 Critical/Important；77 项不同定向测试通过，另有 UTF-8/private 隔离/最终 STOP 注入核对。 |
| `git diff --check` | 通过。 |

可复查命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_review.py
PYTHONPATH=src:tests python -m unittest test_budget_review -v
python outputs/framework_epoch01_quick_eval_2026_09_12/verify_review.py
```

没有重新运行依赖模型环境的完整 integration suite，也没有将历史 tokenizer 测试算作本轮新验证。独立复审报告位于本机 `/tmp/toolv2x-budget-review.md`；本报告已保留关键结果与复查入口。

## 6. 结论边界和下一步

三项问题已修复，单轮预算条件已有明确可执行配置。这不等于完成 strong one-shot 的策略拟合，也不证明交互带来驾驶收益。摘要选择、候选集、预算模式与策略仍需在后续正式实验前冻结并公平选择。

E / derived / Z、真实回执、P 不运行 MTR、F 使用完整合法 context、同信息对照、同一 driver 修订和旧五策略均保留。P 继续是 causal tracking-state motion proxy。没有新增第三套机制。

本轮停止在 T6 定向修复。后续如获授权再实施 T7 的分支制作接口；本文件不授权自动运行真实分支或训练。
