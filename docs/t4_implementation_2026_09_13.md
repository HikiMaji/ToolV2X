# T4：GoT 修订参与下一次 P/F 请求

日期：2026-09-13。承接用户“继续下一步”，本批仅实施已批准计划的 T4；T5 及以后未执行。没有启动训练、真实 MTR/GoT 推理、真实新方法实验或外部数据缓存重建。

交替执行代码已接通：同一驾驶器生成初始方案，实际调用 P/F，合并 T3 的已购与派生证据，使用共同 receiver 构造新输入，再调用同一驾驶器；真实返回的旧/新方案绑定第二请求或决定 STOP。**本批验证的是执行契约及原驾驶适配接口，不是驾驶收益、真实模型证据敏感性或完整方法已完成。**

## 1. 实际修改

| 文件 | T4 改动 |
| --- | --- |
| `src/planning/method_episode.py`（新增） | 最多两次远端调用的交替执行器、可见决策状态、实际方案参数绑定、失败和逐阶段留痕。 |
| `src/planning/v2vgot.py` | `plan_prepared` 显式接纳 `source_blocks_v2`；核对原 tokenizer 的实际计数、receiver/feature/context 配置和 256-token 生成预算。 |
| `src/planning/run_framework.py` | 新增 `interact`、一次加载原模型、每样本独立 service、原子快照、已终态样本前缀恢复。旧 `prepare/generate` 保留。 |
| `src/tools/task_spec.py` | 抽出已有数值及异常轨迹校验为 `validate_plan`，使初始方案、修订、STOP 结果和外发请求服从同一 ExecutionSpec。排序/装包语义不改。 |
| `tests/test_method_episode.py`（新增） | 18 项资源无关执行、失败、因果路径和持久化测试。 |
| `tests/test_framework_episode.py` | 新增 2 项原 tokenizer / 原 `plan_prepared` 适配与三阶段集成测试，只替换昂贵的 `_generate`。 |
| `scripts/check_review.py` | 注册 T4 资源无关测试，不引入模型依赖。 |
| `docs/STATUS.md`、本报告及执行计划 | 标记实际完成范围和后续边界。 |

相对于已交付 T3，本批未修改 `episode.py` 的 Ego/P/F/PF/rule、旧 v1 `query/decode_response`、旧提示构造或训练入口。工作区中此前 T1/T2 边界修复及 T3 未提交内容仍保留，不应混算成本批新增。

## 2. API 与记录

```python
run_task_episode(local_window, local_prediction, motion, features,
                 service, predictor, driver, policy, limits,
                 on_progress=None, *, local_provenance,
                 sample_id=None, branch_id='main', token_counter=None)
```

- `predictor` 是 T3 的同一个 `FrozenPredictor`；receiver 与 provider 共用该绑定。provider 为每个样本新建，不继承另一 episode 的回执或预测缓存。
- `driver` 是同一对象。执行器固定本车特征快照、运动、时刻和配置，每次调用同一个 `plan_prepared`；原模型推理仍为 direct、greedy、单 beam、最多 256 新 token。
- `limits` 是完整 JSON：`version=toolv2x_interaction_v1`、`max_calls`（0–2）、`policy_id`、稳定 `driver_version={name,revision}`、完整 `execution_spec` 和完整 `receiver_spec`。服务排序、大小和异常轨迹参数继续由版本化 ExecutionSpec 管理。
- `policy(visible_state)` 仅返回 `{tool, mode, reason}`。状态含实际当前/上一方案、本车合法证据、已购/派生字段、入模报告、实际回执和剩余预算；不传 service、loader、未购 peer 内容、点云特征对象、未来标签或完整分支树。传入的是脱离引用的快照。策略是受控本地 callable，此接口不声称提供任意 Python 插件沙箱。
- executor 自己绑定 `tau_new/tau_old`、manifest、source/scene/g 和预算；策略不能自行填路径、receipt 或已知字段清单。
- `token_counter` 只允许契约测试注入。原 GoT v2 适配器拒绝 `injected_contract_counter`，重新使用原 tokenizer 核对真实输入长度。

返回 `toolv2x_episode_v2`，包含：

- `plans`：每阶段 prepared/Z、原始 driver output、解析/可执行状态、实际调用耗时、来源 request 和 ledger snapshot 索引。
- `requests/responses`：实际请求和返回原始 bytes 的 `wire_hex` 表示、已报告费用。这里是可还原的原始字节存储，不是内容指纹。
- `ledger_snapshots`：初始及每次接收后的 T3 台账；派生失败也保存异常携带的已付 E。
- `decisions`：实际可见状态与策略选择；`events`：连续 event_id、sample/branch/stage、plan/request 关联、前一事件引用。
- `cost_events/cost`：每次构造、driver、service、receiver 的记录和已知累计费用；缺失生成费用保持 `None/complete=False`。模型子项和外层调用时间都留存，不能相加重复计费。
- `final_plan_id`：仅成功 STOP 指向当前真实有效方案。中间失败保持 `None`，`last_valid_plan_id` 只作诊断，不降级为成功终态。

## 3. 核心时序与边界

```text
GoT τ0 → P/F_current → 实际响应 → T3 E/derived/Z → GoT τ1
       → STOP
       或 P/F_current/change → 实际响应 → T3 E/derived/Z → GoT τ2 → STOP
```

初始不存在 change；旧/新方案完全相同时，change 不在合法动作集合中。STOP 不读取 peer、不再生成。两次调用后直接 STOP，策略不能要求第三次。

每个初始/修订回答都重新解析其原文，并与 driver 返回的 waypoints 一致性核对，再执行 ExecutionSpec 的数值、速度与加速度范围检查。解析失败或异常轨迹终止该样本并保留原文、已购信息及已发生成本，不用几何代理或旧方案补成“GoT 修订”。相同证据生成相同方案是合法结果。

第二次请求总是包含 T3 的完整合法远端 manifest；在发送前，用完整请求 UTF-8 bytes 和 response cap 预留检查预算。不会通过删掉 manifest 重买已知字段以挤过上限。T3 本地派生 F 不冒充远端 receipt，也不在 T4 中预先屏蔽 F。

每次合法响应都会进入 receiver 和随后驾驶调用，包括空字段响应；例如完整上下文证书首次到达可改变合法本地计算上下文，不能仅按 `records=[]` 跳过该阶段。

## 4. 运行入口与中断恢复

新增命令（本批只运行了 `--help`，未启动此真实模型路径）：

```bash
python src/planning/run_framework.py interact NEW_OUT \
  --data EXISTING_CAUSAL_INDEX \
  --checkpoint EXISTING_DRIVER_CHECKPOINT \
  --spec FULL_VERSIONED_INTERACTION_SPEC.json \
  --role validation --per-recording 1
```

spec 文件顶层为 `limits` 和 `local_provenance`，数值由调用者明确填写并随运行保存，不能把本批合成 fixture 的宽容量当作实验默认。

CLI 现仅开放三个明确的接线诊断策略：`stop`、`p_current`、`p_current_f_change`。后者先请求 P；若 GoT 实际方案变化，再请求 F_change，否则 STOP。通用 API 已支持 P→P/P→F/F→P/F→F 和可见状态 callable，但 **T8 的请求价值网络尚未实现**；诊断策略不是正式学习方法或新创新点。

运行目录保存完整 spec、在线选样清单、代码副本、模型来源、本车输入资源、逐样本 task 和总体 progress。请求前、响应真实到达后、receiver 完成后、driver 前后都会立即写原子 task 快照；写盘回调失败直接中断，不在未记录的情况下继续购买或生成。

`--resume-from` 只复用连续的已终态样本，包括失败样本；检查配置、在线索引和已归档 Python 源码一致。响应已收到但尚未修订的尾部留在原目录，在新目录重试时使用新 service、从 τ0 开始。旧尝试的费用不消失，也不自动并入重试的成功 episode。复用样本的输入资源通过保留的 `artifact_root` 指向原产物，不复制或重建缓存；不要删除原目录。

这是样本前缀复用，不是进程中间状态恢复，也不提供模型权重内容认证。冻结权重/模型版本仍是受控运行要求。

## 5. 验证与研究结论边界

- T4 资源无关测试：**18/18 通过**。使用真实 `VehicleTools.query_task`、T3 ledger/receiver，预测器和驾驶生成是测试替身。
- 全部轻量审查：**174/174 通过**，不导入 PyTorch/Transformers。
- 原模型环境的 `SharedContextTests`：**5/5 通过**。包含旧已保存提示逐字回归、原 tokenizer v2 预算、拒绝伪计数/不同预算，以及原 `V2VGoTPlanner.plan_prepared` 的三阶段链；只有 `_generate` 被替换，没有加载权重或实际模型 forward。
- 原 v1 工具/五策略回归包含在以上轻量测试；原 query、decoder/helpers 和五策略/prepare/generate 函数另做 AST 比较。
- 独立子 agent 定向复审通过：T4＋旧 Episode＋TaskSpec 共 43 项、原 tokenizer/driver 适配 5 项，合计 48 项；缺失成本及非 bytes 返回失败路径均已复核，无剩余阻塞性问题。
- `git diff --check` 和 `interact --help` 通过。完整模型测试集未执行。

关键反例已覆盖：改变第一次真实 P 内容会使第二动作从 F_change 变 STOP；STOP 和 0/1/2 次上限分别只有 1/2/3 次 driver；初始/中间无效回答阻止后续查询；策略不能篡改实际任务参数；付费后 predictor/decoder/生成失败保留记录；缺失响应成本不再导致原始 wire 丢失；同时拦截 `builtins.open` 与 `io.open`，改变未来标签文件不影响实际 interact 的请求/提示；中途响应后的快照不被当作完成，恢复只复用首个终态样本。

原 tokenizer 在旧已保存长提示回归中仍有 `2456 > 2048` 警告，执行器显式上下文为 4096；该测试没有模型 forward，不据此宣称长上下文质量或性能。

**可验证：** C0 及 H2 的执行前提——驾驶修订值确实进入下一次任务或停止决策，且真实原 driver 适配接口可被这条代码链调用。

**不能据此断言：** 真实 GoT 会响应证据、修订更准确、change 优于 current、两轮优于强单轮、实时性成立或论文创新已验证。三次 GoT 的正式评价汇总、强单轮条件委托、同证据额外生成、价值监督/训练与真实分支实验分别仍属于 T5–T9。

## 6. 与原计划的细化

1. 增加 `task_spec.validate_plan` 的共享调用，以免 executor 和 provider 的异常轨迹判断分叉；没有新增数值阈值。
2. `interact` 先使用明确标注的诊断策略，正式请求价值模块仍留给 T8，没有在 T4 另造策略网络。
3. 恢复粒度明确为已终态样本前缀；不实现半程 receipt registry / driver 状态恢复。
4. T4 留齐每次调用的原始成本及少量累计值，T5 的正式评价口径和失败分母汇总尚未实现，不能以本批累计字段替代 T5。

本批 Method 核心是交替生成及实际修订参与后续任务。配置冻结、适配校验、台账关联和持久化属于工程基础；合成测试与诊断策略属于契约验证，不是效果实验。当前不增加新网络、RSU/I、RL、peer 学习排序器或额外推理对照。
