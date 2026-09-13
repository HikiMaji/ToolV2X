# ToolV2X 任务请求与驾驶修订反馈 Implementation Plan

> **For agentic workers:** 用户确认相应批次后，使用 superpowers:executing-plans 按任务实施；各步骤用复选框跟踪。本文件不授权启动训练或实验，也不要求并行 agent。先阅读两份 Spec，不能把本计划当成已完成记录。

> **执行更新（2026-09-13）：** 用户已授权且仅授权 T1/T2；两步代码与契约测试已完成，详见 [实施记录](../../t1_t2_implementation_2026_09_13.md)。T3 及以后未实施，不自动继续。以下采用用户补充硬约束：协议与完整版本化 ExecutionSpec 分开；provider 的 manifest 只接受自身真实远端响应回执；本地派生等价留给 T3，不能作为 T2 回执；provenance/model/context 使用稳定结构版本，路径不作为语义身份。默认数值只用于当前契约配置，不是模型效果已验证的实验参数。

> **T3 执行更新（2026-09-13）：** 用户随后授权 T3。已实现已购 E、派生父链、共同 receiver/Z 和最小完整 P 上下文证明，详见 [T3 实施记录](../../t3_implementation_2026_09_13.md)。T4 以后仍未实施、不自动执行。下方此前仅授权 T1/T2 的表述是当时批次边界。

> **T4 执行更新（2026-09-13）：** 用户“继续下一步”授权 T4。本批交替执行、原 driver 的 v2 适配、interact 与逐阶段持久化已完成代码和契约验证，详见 [T4 实施记录](../../t4_implementation_2026_09_13.md)。没有运行真实模型、训练或新方法实验；停止在 T4，T5 以后未执行。

> **T5 执行更新（2026-09-13）：** 用户授权“开始 T5”后，v2 离线评价、全部 driver/服务/receiver 成本、缺失失败分母及旧指标兼容已完成；见 [T5 实施记录](../../t5_implementation_2026_09_13.md)。191 项轻量测试通过，未训练、未执行真实模型或新方法效果实验；停止在 T5，T6 以后不自动执行。

> **T6 执行更新（2026-09-13）：** 用户授权“先实现 T6”；关键对照及共享真实前缀、同槽 refinement、单轮条件委托和 T5 成本/分支接入完成，见 [T6 实施记录](../../t6_implementation_2026_09_13.md)。226 项轻量、2 项原 tokenizer/planner 契约和旧 160 条回答复算通过；未训练、未真实模型执行或新方法效果实验。停止在 T6，T7 以后不自动执行。

**Goal:** 在现有 P/F、CMP MTR 和 V2V-GoT 上实现最多两次远端能力调用，验证任务参数与真实驾驶修订反馈的作用。

**Architecture:** 保留旧五策略执行路径；增加参数化 P/F 接口及交替调用同一 GoT 的执行路径。使用一个普通本车请求价值模块，提供方保持确定性检索；累计取得的字段、接收端派生字段、实际驾驶输入和全部成本分别留痕。

**Tech Stack:** 现有 Python 3.8、标准库、NumPy、unittest；沿用现有 PyTorch、CMP MTR、LLaVA/GoT、tokenizer 和 compact 表示，不增加依赖。

**Spec:** [9-12-1.md](../../9-12-1.md) 是方法、因果、执行与实验优先依据；[9-12-2.md](../../9-12-2.md) 约束 Tool 语义。旧 [核心机制 v0.1](../../core_mechanism_v0_1_2026_09_12.md) 只作背景，冲突时服从前两份材料。

## Global Constraints

- 计划经用户逐批授权已实施 T1–T6；本批到 T6 停止。未授权恢复训练或执行真实新方法实验，T7 以后不自动执行。
- 首版只有远端 P/F；current/change 是任务参数，不是新增工具。
- 固定决策时刻 t、ego_at_t 坐标、同一冻结 GoT 和同一冻结 MTR；最多两次远端能力调用。
- 请求前只能使用本车信息、已经合法取得/计算的信息、公共配置和预算；未来 GT 只进入独立离线评价/监督。
- 不增加 RSU、I、新预测骨干、联合 RL、peer 学习目标排序器、额外价值归因头或自然语言工具规划。
- 不禁止完整 P 的本地 MTR 重算；不把小网络结构、API 包装、预算回归或工程台账列为独立创新。
- 原 Ego/P/F/PF/rule、原检查点、原实验输出保留。新执行格式、新接收器和新结果必须有明确版本。
- 用户确认首批代码后再实施；实现完成也不等于已获授权恢复此前停止的驾驶训练。

---

## 1. 审查结论与当前事实

**ToolV2X 主线可以保留。当前已有真实工具服务和驾驶底座，尚未具备两份材料要求的完整工具使用机制。** 最值得验证的候选是：第一次实际返回引起的驾驶修订，是否帮助第二次任务委托取得更有用的 P/F 证据。它不是已成立的创新结论。

本次直接阅读了两份新材料，以及 `vehicle.py`、`episode.py`、`context.py`、`inputs.py`、`v2vgot.py`、`run_framework.py`、`cmp_adapter.py`、评价代码和相关测试。没有用文档的完成标记替代调用链检查；本轮没有重新检索近邻文献，文献判断沿用用户提供的材料，不新增首创性断言。

### 1.1 最新实验状态

- 当前分支：`fix/review-2026-09-10`。工作区已有若干未提交研究文档，本计划不覆盖它们。
- `outputs/framework_baseline_restart_v1/user_stop.json` 记录用户在 09-12 18:24 左右停止训练；日志停在 step1941，完整恢复点为 step1925。本轮进程只读检查未发现对应训练/管理命令运行。
- 已评价的是 `training/checkpoint-epoch01`，不是 step1941。第一轮完成 14935 条训练行和 540 行验证。
- `outputs/framework_epoch01_quick_eval_2026_09_12/evaluation/summary.json`：两个开发验证录制组、32 帧、五条件共 160 次生成，格式失败 0。ADE3：Ego 1.7412、P 1.9374、F 1.8213、PF 1.7223、rule 1.6413 米；恒速直行参考 1.5327 米。
- 上述是已有冻结任务的驾驶生成；没有 current/change 分支、交替 GoT 轨迹树或请求价值模型。这些数字不能作为新机制收益或序贯监督。
- 当前实际驾驶输入为点云特征和文本，没有真实 RGB；属于固定时刻的离线开放环轨迹模仿评价。预训练来源及独立测试边界的历史限制仍在。

### 1.2 四类实现差距

| 类别 | 能力/位置 | 核实结果及处理 |
| --- | --- | --- |
| 已具备 | `src/tools/vehicle.py:VehicleTools.query` | query 后才读取邻车私有窗口；P 返回 11 个时刻的源内跟踪状态，F 运行/复用完整窗口 MTR 后按当前框 ROI 返回。真实能力可复用。 |
| 已具备 | `decode_response`、`history_window` | 白名单、时间/来源/坐标/形状验证；P 可重建因果历史。无目标不等于观测到自由空间。 |
| 已具备 | `cmp_adapter.make_batch/predict` | 同一原 MTR、完整邻居上下文、六模态；历史不足有显式 model_used=False 的静态回退。 |
| 已具备 | `episode.run_episode` | Ego/P/F/PF/rule、最多两次调用、真实首个 P 影响 rule 后续 F/STOP、失败保存已付成本。**循环内没有 GoT。** |
| 已具备 | `context.build_plan_input`、`V2VGoTPlanner.plan_prepared` | 共享 source_blocks_v1/direct 提示、固定本车块、实际点云输入和六点生成。可复用同一实例连续生成。 |
| 已具备 | 训练/评价/归档 | 驾驶 SFT、检查点恢复、独立未来标签、录制级划分、格式失败分母、真实报文字节与单次驾驶成本。 |
| 复用但改接口 | `VehicleTools`、`decode_response` | 增加独立 v2 task request/response 分支。保留 `query(tool, roi)` 和 v1 严格字段，不能直接向旧报文塞字段。 |
| 复用但改接口 | `make_evidence`、`history_window` | 原 v1 逻辑保留；v2 需字段身份、已购/派生区别、上下文正确的 P 本地重算与字段投影，不能逐包重复追加。 |
| 复用但改接口 | `context.py`、`inputs.py` | 复用 tokenizer/compact 编码；增加共同 v2 接收选择和字段到提示位置的映射。旧白名单不能无条件放开。 |
| 复用但改接口 | `plan_prepared` | 主方法继续直接调用它；如需携带新表示版本，应显式校验。只有等证据迭代对照需要单独、可核对的上一答案槽。 |
| 复用但改接口 | `run_framework.py`、`evaluation/framework.py` | 复用加载、样本清单、持久化和指标；新交替路径需事件序列和多次 GoT 计费，不复用旧“全部准备后统一生成”的时序。 |
| 必须新增 | 请求执行规格 | current/change 的逐时刻关系得分、参数绑定、合法字段引用、上限与确定性装包。 |
| 必须新增 | 交替执行状态机 | τ0→q0→r0→τ1→q1/STOP→r1→τ2；真实修订进入下一决策与任务参数。 |
| 必须新增 | E/Z 与派生字段台账 | acquired fields、derived fields、admitted/dropped、上下文版本、缓存语义、请求/响应/driver 事件关联。 |
| 必须新增 | 有限分支、两步监督、请求价值模块 | 全部分支来自真实请求及真实驾驶生成；按物理录制交叉拟合末步教师，再制作首步终态标签。 |
| 必须新增 | 核心反方对照 | 强单轮条件委托、同证据额外生成/迭代、冻结反馈、等候选数 old/union、P/F 同目标诊断、Ego-max-context。 |
| 冲突/需明确 | 旧 PF 重复表示 | `make_evidence` 将 P_local/F 分别追加；即使相同预测也占两条记录。v2 按字段/上下文去重，旧 PF 保留其历史语义。 |
| 冲突/需明确 | P 本地缓存 | `episode.py` 缓存只用 track IDs，在旧固定 t 路径有其前提；v2 不能忽略历史内容、模型和上下文版本。 |
| 冲突/需明确 | 任务优先检索被距离选择覆盖 | 当前 `_select` 按当前距离重新排序。若只改 provider，任务选择可能被 driver 截断抵消；若只给新方法换 receiver，又会混淆归因。 |
| 冲突/需明确 | Ego 的远端预留 | 当前默认留 1536 token 给远端，无通信 Ego 也留着；必须同时保留固定本车块参照和 Ego-max-context。 |
| 冲突/需明确 | 一次往返与一次能力执行 | 强单轮可以在 peer 内执行最多两个 P/F 原语；不能限制为只能取一份 P 或 F。 |
| 冲突/需明确 | 规格尚缺常数和部分语义 | σ、尺寸代理、短历史、平局、容量、异常轨迹与费用尺度未被新文档全部定死。下文是建议默认值，确认后才冻结，不冒称文档已经规定。 |

两份材料没有需要另造第三机制的实质矛盾。材料二的“历史/观测检索”在本仓库应精确解释为**因果跟踪状态检索**；`history_valid` 包含漏检时跟踪器维持的状态，不是每帧检测命中。

## 2. 最小方法与确定执行规格

### 2.1 动作和状态

初始：同一 GoT 在本车输入上实际生成 τ0。

```text
remaining=2: STOP / P_current / F_current
remaining=1: STOP / P_current / F_current / P_change / F_change
remaining=0: STOP
```

STOP 复用当前真实方案，不再生成。P→P、P→F、F→P、F→F 均允许，但相同字段不得再次购买。首步无旧/新修订对，change 不合法。

每次有效真实响应后：合并 E，计算有必要的 P_local 派生结果，选择 Z，再用**同一模型实例、同一检查点、同一特征时刻、同一解码配置**生成 τnew。这里“修订”是重新推理的结果，允许与旧方案相同；首版不要求 GoT 在提示中看到旧答案，也不把人工几何改写称作 GoT 修订。真实旧/新输出都进入策略状态，change 请求绑定两者。

建议状态字典：

```text
DecisionState
  sample_id, physical_recording, scene, g, provider
  ego_motion, local_evidence, current_plan, previous_plan
  acquired_fields, derived_fields
  admitted_field_refs, dropped_field_refs, admission_report
  response_receipts, cost_ledger, remaining_budget
```

策略只接收这个可见状态的确定性摘要；不接收 service 对象、邻车窗口 loader、未购买候选结果、离线标签或全部轨迹树。

### 2.2 v2 请求和结果

沿用字典及严格校验，不引入工具注册框架/RPC 框架。增加 `VehicleTools.query_task(request)`；旧 `query(tool, roi)` 保留。

| 结构 | 必须字段/语义 |
| --- | --- |
| TaskRequest | version、request_id、provider、scene、g、coordinate_frame=ego_at_t、tool=P/F、mode=current/change、times=[.5,1,1.5,2,2.5,3]、tau_new、tau_old（current 为 None）、execution_spec、acquired_field_manifest。 |
| ExecutionSpec | 完整版本化配置，集中记录 sigma、max_targets、请求/响应/episode 字节上限、ego 几何代理与尺寸、异常轨迹阈值；不另设重复 ego_size/limits。最多两次远端调用属于首版机制边界；后续 driver 的 context/generation 配置也需完整留痕。所有上限在发送前可知。 |
| FieldRef | provider、scene、g、track_handle、field_kind、producer_version、context_version。历史 bundle、当前 anchor bundle、完整六模态 forecast bundle 分别识别；不能以整目标为去重单位。 |
| FieldRecord | ref、value、origin=remote/receiver_derived、receipt_id 或 parent_refs、context_scope、model_used/fallback。request_id 是取得路径，不是内容身份的一部分。 |
| TaskResponse | echo/request_id、同源同刻坐标、records、references、执行/排序规格版本、truncated、结果状态、完整上下文证书（若成立）、history/fallback/cache 边界。 |
| Receipt / CostEvent | 实际发送/接收字节、原语次数、service/peer MTR/receiver/GoT 时间、token、缓存命中、成本完整性；失败已发生的代价保留。 |

实现时 `FieldRef` 为 JSON 字典，`field_key` 返回可排序身份元组；不增加数据库。固定决策 t 内 handle 为 provider 的实际源内 track ID，跨场景/提供方/时间不可互认。首版策略不能凭空指定未返回的 handle；发给 provider 的 manifest 只允许引用其真实远端返回回执。本地派生证明保留在后续 T3 接收端台账，不能冒充 provider 回执。

持久化的 `acquired_fields`、`derived_fields` 均为 FieldRecord 列表，身份索引只在内存中按 `field_key(record['ref'])` 建立；`admitted_field_refs`、`dropped_field_refs` 是 FieldRef 列表。避免把不可 JSON 序列化的元组键字典直接写进 episode。

模型和上下文语义版本用稳定 `{name, revision}` 结构，冻结资源路径只作旁路定位，不能作为字段身份。**同一个字符串不是等价证明**：后续 T3 上下文登记必须对应真实输入数组、掩码、目标顺序、坐标、时间与模型配置。完整上下文证书不能偷偷列出尚未支付的目标和历史；标识只用于一致性，不编码额外私有信息。T1/T2 未实现内容 fingerprint 或本地派生等价证明。

### 2.3 current/change：建议冻结的检索规则

以下是实施默认建议，不是已运行参数，也不是新增算法贡献。

1. 时间：只比较六个既有驾驶时刻；不读取未来真实位置，不跨 t 更新传感器输入。
2. 尺度：目标半径 `r_i = 0.5*hypot(length,width)`，取它在 t 的跟踪框；ego 用公共配置 `length=4.8 m,width=2.0 m`。这是包围圆间距代理，不是精确车体碰撞。配置可被真实公开车辆尺寸替换，但整轮比较冻结一致，不从未来 GT 估计。
3. `d=max(0, norm(peer_xy(h,m)-tau(h))-r_i-r_ego)`，`rho=exp(-d*d/(2*sigma_m**2))`。σ 从 ExecutionSpec 读取，当前默认 5 米。
4. P：从同一 ego_at_t 历史中取最新两个 valid 状态，用它们真实时间差估计平面速度，以当前点恒速外推六个时刻，仅用于检索。少于两个 valid 时零速度并标记 `single_state_static_proxy`。该代理不作为 P 报文中的新 forecast，不调用 MTR。
5. F：完整 peer 窗口先运行/复用冻结 MTR，按已有六模态、六时刻计算关系；全部模态按原顺序保留，不因 ego 方案改变预测、不重归一化模式分数。model_used=False 沿用既有静态回退并标记。
6. current 为 `max(rho_new)`；change 为 `max(abs(rho_new-rho_old))`，相同时间和同一份预测中的相同模式先作差，再聚合。不得分别匹配两次独立 MTR 模式。
7. 排序按 `(-round(score,12), track_handle)`；同一候选内部字段顺序固定。无随机排序、不加最小得分阈值、不让后来的请求 ID 影响内容身份。old/new 完全相同时，change 可由本车合法地屏蔽。
8. 先剔除已知的等价字段，再对仍有新字段的目标评分并装包；同目标已取得 F 时，P 历史仍可入选。当前框的固定 ROI 不再是主方法检索语义，只留作对照；不能在 current/change 前偷偷加方向 ROI。

必须覆盖两个反例：`rho_old=(1,0),rho_new=(0,1)` 时 change=1；远处 100→80 米的间距变化不应凭原始距离差胜过近处 2→4 米的相关性变化。P 与 F 得分不必相同，前者是历史代理、后者是模型预测；不将差异称为交互反应预测或校准风险。

### 2.4 已购 E、派生信息与入模 Z

严格保持 `E_new=E_old ∪ 实际新返回字段`。本地 MTR 的输出放入单独 `derived_fields`，每项指向已付费 P 历史，不能写成又收到了 F。Z 只允许引用本车合法输入、E 或由 E 得出的派生字段。

每次驾驶保存：完整 E 的引用、派生关系、Z 的精确字段引用、实际舍入后数值、prompt、token 数、被丢弃项及原因。`acquired` 单调；`admitted` 不必单调；新字段挤掉旧字段必须可追踪。字段级入模率和目标数分别报，不能把“已购”写成“driver 已使用/理解”。

Z 还包括真正进入提示的来源、缺失/截断、fallback 等语义元数据，不能仅用“目标引用集合相同”判定同证据。实际返回的空响应边界来自已付费回执；若进入 driver，必须在 Z 中有来源。请求 ID、请求名称和价格账本留在旁路，不因新请求到达就改变 driver 文本。等证据控制核对完整证据文本/数值及这些元数据，只有显式允许的上一模型答案槽可以不同。

为缩小首版改动，推荐：

- 沿用 compact 数值/模式编码及固定本车块；查询/ledger 元数据保存在旁路，driver 只接收对驾驶有意义的来源与能力边界，不塞入整份协议。
- 字段先归一化和去重，再组成完整可解码对象；同一预测只有一份数值，可以有多个来源路径。不同上下文的预测必须保留区别。
- v2 接收选择仍用确定性当前距离顺序，平局按 provider/handle/field_kind/context_version。它不依赖策略名、请求名称或到达顺序，避免重复请求改写排序。
- 按“共享 anchor + 完整 history bundle / 完整 forecast bundle”装入；不因缺容量裁掉半个历史或部分模态。只按实际 tokenizer 检查是否容纳，舍入与 wire bytes 分离。
- source_blocks_v2 用最小必要的字段/上下文标识支持上述对象；旧 source_blocks_v1 的 prompt 与测试原样保持。若能在 v1 数值编码中表达，只复用 codec，不额外造第二种压缩器。
- **所有新对比臂共享这个 receiver。** 当前/变化检索即使部分被距离排序抵消，也先如实报告；若以后改为任务优先 receiver，必须所有对照同步改并重做对应标签。

### 2.5 重复购买、上下文与 P/F 等价

| 已有内容 | 再请求 | 处理 |
| --- | --- | --- |
| 同源同刻 history | 同 history | skip/reference，不重传数值；换 current/change 不改变身份。 |
| 某子集 P + 本地 MTR | 完整 peer 上下文 F | 不认为等价；局部背景与完整背景不同。 |
| 完整 P + 同模型同上下文本地重算 | 同目标完整 F | 验证的等价预测已知，跳过。若全部 F 已可得，可在本车屏蔽 F；不靠虚构互补强制购买。 |
| F forecast | P history | anchor 可引用，历史是新字段，允许返回。 |
| 某上下文预测 | 新增 P 改变了本地输入上下文 | 产生新 context_version 的派生预测，不能覆盖旧身份或复用错误缓存。 |
| 已知一个目标 | 新目标 | 返回新字段；不准把跨 provider 的同号 ID 合并。 |

本车不能在请求前知道未知目标是否存在。可以屏蔽**已证明**重复的动作；否则发出请求后 provider 才能判定 `no_new_fields`。空新内容响应也收费，不用“先执行看看有没有新东西”免费选择动作。重复字段的引用/空响应头都计字节；相同字段换序或重复强调不能记作新信息收益。

P 本地 MTR 默认对**累计已购 P 历史的确定排序集合**重算，缓存键包括完整父字段集合、模型和预处理版本；新 P 到来改变集合时重算。不得因只返回少数目标而把它标为完整上下文。

继续保留两种不同目的的 P/F 检查：

1. **预测同信息检查**：完整、无数值损失的 P 重建 provider 全窗口，比较同 MTR、同时间、目标、模式和背景的预测数组；history_valid、顺序、坐标、fallback 也一致。若主方法预算装不下完整 P，这个诊断单独声明更高的公共容量，不能免费绕过主实验预算。
2. **驾驶终态诊断**：固定目标集合和本车背景，比较 `相同 P 历史+局部预测` / `相同 P 历史+完整预测` / `相同完整预测但无 P 历史`；锁定相同槽位和容量，不让去掉历史后悄悄填入别的目标。它定位上下文/历史作用，不证明序贯必要性。P/F 的 source 标签不同也需规范化或单列标签控制，不能把“输入相等”和“预测数值相等”混说。

### 2.6 硬预算、失败与成本建议

首个集成 profile 建议：每次最多 4 个目标、request 上限 4096 B、response 上限 8192 B、一次 episode 总双向报文上限 24576 B；最多 2 个 P/F 原语、3 次 GoT，context=4096、generation=256，本车固定块 peer reserve=1536。它们是可测试默认值，正式质量—成本曲线使用预注册的多个预算档，不把一个档的胜负当普遍结论。

- 本车在发送前序列化完整 request（含两条轨迹、manifest、头部），据实际 request 长度和公开 response cap 做预算可行性判断，并预留至少一次后续 GoT 的计算/生成名额。没有可支付动作就 STOP。
- provider 按排序尝试整字段 bundle，每次按最终 JSON UTF-8 字节检查上限；连头部都装不下时请求前拒绝，候选装不下则跳过并继续检查后续候选。返回 truncated/empty 状态也必须在完整报文 cap 内。正式数值不为凑字节而额外舍入。
- 硬限制可严格保证调用数、完整报文字节、token/context；GPU 实际秒数使用测量和预期成本，**没有抢占/超时机制就不称严格实时 deadline 保证**。本轮不增加真实网络部署。
- F 第一次完整计算、同 t 缓存命中分开记；每个独立反事实 episode 从同样冷缓存开始。不能让离线分支 A 的 F 为分支 B 免费预热；共享缓存复用用于节省制作成本时，仍单列部署反事实成本，不能用近零回放时间训练效用。
- 计划可用性建议：六点有限、时间轴正确；把 t=0 原点加入后任一 0.5s 段速度不超过 80m/s，段间加速度不超过 30m/s²。此宽松规则只挡明显异常，不是安全验证；静止合法，不以 GT 决定合法性。原始回答、格式失败与异常指标全部保存。
- τ0 无效：本 episode 终止计失败，不读取 peer。中间修订无效：终止计该路径失败，不用它形成第二请求；保留以前的有效方案供诊断，但不把它暗中替换为成功终态。首版不加 CV 回退方案、不伪称回退为 GoT。所有新对照共享规则；旧历史指标不回写。
- 服务/接收/生成错误保留已发请求、响应及成本；无法得知的费用标 unknown/incomplete，不能填 0。

总计算时间使用互不重叠的外层阶段：`local_MTR + Σ(service_elapsed + receiver_elapsed) + Σ(driver_attempt_elapsed) + policy/executor_elapsed`。peer MTR 是 service 子项、P_local MTR 是 receiver 子项，不能再加一次造成重复计费。另报各子项、driver 生成核心时间、输入/输出 token、GoT 次数和双向字节；部署加载成本、实网传输不在当前实测内，必须写范围。

效用首版建议 `L=min(ADE3/10m,1)`，无效终态 L=1；另存未截断 ADE/FDE/失败率，不能让该有界训练目标掩盖坏样本。`c=λb*(实际双向字节/24576)+λt*(增量实测计算秒/训练侧标定的一次GoT秒数)`。实施默认 λb=λt=0.01，时间尺度取事先指定训练侧标定录制的初始 GoT 实测中位数，规则在收集前登记、数值在制作标签前冻结；缺失/非正尺度拒绝运行。这个默认权重不代表已验证的最优效用，另保留零软成本、相同硬预算的质量参照，以及预登记成本敏感性。开发结果不反推惩罚，原始成本照常报告。没有明确效用配置时，标签工具应拒绝运行，不能暗填“实验最优权重”。

## 3. 核心对照的可执行定义

### 3.1 基线矩阵

| 对照 | 必须相同 | 唯一有意改变及允许结论 |
| --- | --- | --- |
| 固定 ROI 价值策略 vs current | 同状态摘要、价值网络容量、训练录制/预算、receiver、MTR/GoT | 请求程序。可检验任务参数作用，不能归功于换网络。 |
| current+change vs current+old / current+union | 首步 3 动作、末步 5 动作，P/F 各两种合理模板；监督量与预算相同 | old 用 τold；union 用 `max(rho_old,rho_new)` 后聚合，真正涵盖两条方案的时间关系。不能用无关区域充数。 |
| 真反馈 vs frozen τ0 | **固定实际同一 q0、r0、τ1、E1、Z1**；同决策器输入/解码和预算 | 第二请求中的方案依据。冻结臂把方案相关特征和请求参数统一替换为 τ0/τ0，不能只改报文仍把 τ1 特征泄露给策略。实际 driver 仍用已购证据产生终态。它是因果消融；另配独立同预算拟合的冻结策略控制分布失配。 |
| 两轮 vs strong one-shot conditional delegation | 相同服务、信息来源、能力调用上限和总资源；支持合法摘要/候选/条件程序/bundle | 有无真实中途 ego 往返反馈。只有超过充分训练/选配的一轮方案族，才能支持该资源档下交互增益。 |
| 新证据修订 vs same-evidence extra-generation | 同一 prefix、同一 checkpoint、总生成次数/配额，精确固定对照的 E/Z | 信息获取 vs 额外推理。重复样本中选 GT 最优输出不合法。 |
| 固定 Ego/P/F/PF/rule | 原调度、原 v1 路径和历史产物保留 | 用作历史基线。另在共同 v2 receiver 下生成兼容版本用于新方法归因，不拿两个 receiver 的差直接算查询收益。 |
| Ego-fixed-block vs Ego-max-context | 本车同源输入、检查点、总 context/generation | 后者 peer_reserve=0、不获取远端；验证无通信基线是否因空留容量受损。 |
| 同目标 P/F 控制 | 目标、背景、表示、容量、模型、时间 | 定位预测上下文或历史字段增量，不推断闭环安全/两轮必要。 |

另保留预算内宽覆盖/完整 P 或 F 参照，检测错误初始方案把 current/change 都引向错误区域。若完整报文超预算，必须标超预算诊断，或使用与其他方法相同的确定性宽覆盖装包；不只给新方法加全景回退。

### 3.2 strong one-shot conditional delegation

**单轮是一次网络往返，不是只允许一个工具原语。** 建议新增 controls-only `query_bundle(envelope)`：

```text
ego: 生成 τ0；构造合法摘要/候选方案/有限条件程序；发送一个 envelope
peer: 执行第一个 P 或 F → 读取该真实中间结果 → 条件选择第二个 P/F 或 STOP
peer: 最多两个能力原语，共同装包返回一次
ego: 用共同 receiver 与同一 GoT 生成最终方案
```

envelope 允许携带本车运动、已持有的本地证据摘要、τ0，以及仅依赖本车 time-t 信息生成的候选方案。首版可使用 τ0、由它得到的减速候选和因果恒速/恒转率候选，通过同一个方案可用性检查；候选选择及数量在训练侧确定并记录。允许更充分的合法摘要/更好的候选来源，只要符合相同总预算；不能人为限成弱的固定 PF 包。

条件 continuation 接收 envelope 可见信息、第一真实结果以及 provider 依法可计算的信息；第二任务仍调用相同确定性 P/F executor。为了实现有竞争力的条件选择，可复用 Task 8 的普通价值回归代码，**对照自己的一个冻结策略副本**部署在 bundle 执行端，仅选任务/STOP，不学习目标排序。它是替代主方法策略的实验臂，不是给主方法增加第二个网络。训练录制组、容量、监督机会与超参选择预算对齐。若采用固定条件程序，应在同一训练侧选择合理阈值并与此拟合版本一起验证，不能默认一个任意规则就是“strong”。

这里需要区分用户禁止的 peer 学习排序器与条件委托对照：主方法 provider 的目标排序始终确定；控制臂里选择条件任务的策略不参与主方法。这是材料一要求强单轮可替代性检验的实现，不增加第三套主方法。

公平账本至少记录：

- `rpc_rounds=1`，`capability_calls=0/1/2`；内部第二请求虽不占网络上行，但执行/构造时间计费，回包及外发摘要算真实 bytes。
- 初始候选若需要额外 GoT/MTR，生成/筛选成本全部计入；不能免费取得本车收到 r0 后才生成的真实 τ1。
- 检查条件若需要未执行的 F，那个 MTR 计算也是 F 能力成本/预算；不能先免费跑完所有 F 再称自己只买 P。
- 若廉价摘要足以替代反馈，承认单轮优势。若在某预算档传完整本车输入并在 peer 跑同一 GoT 可行，不以“必须留在 ego”人为禁止，应按实际双向字节和计算纳入更强代理；是否可行依报文大小核对，不能凭名称排除。
- 控制臂也可用剩余 GoT 预算作同证据迭代；同时报告其自然低计算版本，质量—成本比较不能故意浪费单轮优势。

### 3.3 same-evidence extra-generation

需要两级控制，只有重复 greedy 解码是不够的。

1. **Exact repeat：** 完全相同 features、prompt、E、Z、checkpoint 和 greedy 解码；真实再调用 GoT，记录重复性，不调用 service。输出相同是正常结果，不把它包装成强推理基线。
2. **Self-refinement：** 固定相同的外部 E/Z，在独立 controls-only 提示中加入上一份实际生成的六点方案及固定“根据现有证据复核”指令，再运行同一 GoT。上一答案明确为模型输出，不是新观测；固定取最后一次输出，失败照计，不用 GT 挑最好。不得改变 E/Z 或重排已有字段来暗中增加证据。

为避免迭代提示挤掉证据，比较双方预留同样的上一答案槽并冻结 Z；主方法控制版本也使用相同的 refinement wrapper，仅新证据路径在相应步骤替换 Z。这样对比 `同 wrapper+新增证据` 与 `同 wrapper+固定证据`，可以排除提示形式本身。主方法默认仍是直接重算，不强制使用 wrapper；另报 exact/direct 对照。

先采用全 3 次 GoT 的预登记控制切片保证相同配额；实际策略另按 prefix 的已执行次数配对，并报告累计输入/输出 token 与秒数。三次调用不意味着 FLOPs 或时间恰好相等，需报告差距和质量—成本曲线；不能为填满预算生成无效答案再人为奖励主方法。

至少比较：固定 Z0 下的 3 次本车生成，与获取新证据的 3 次生成；固定真实 q0/r0/τ1/Z1 后，用一次额外同证据生成代替第二次查询后的生成。这分别检查总信息作用和第二请求的增量。迭代提示未经适配的失败也需报告；若需要适配，所有相关对照共享等量适配，不能只训练主方法后宣布迭代无用。

## 4. 监督边界与研究判据

末步：`Y1 = L(当前真实轨迹) - L(该动作真实终态) - 增量实际成本`；STOP=0。

首步：固定一个未在本录制组训练的末步教师，先只用在线可见状态选 continuation，再读取对应真实分支终态/成本计算 `Y2`。包含首步成本和教师真正继续时的第二步成本。离线未来标签可以监督收益，**逐帧 GT 最优续动作只能叫 oracle 上界，不能称可部署回报**。

部署用一个 `Linear(d,64)→ReLU→Linear(64,64)→ReLU→Linear(64,5)` 普通 MLP：输入两个实际方案、运动及缺失标记、阶段/剩余预算、E/Z/dropped 的字段与上下文统计、已知证据关系摘要；输出 STOP/P_current/F_current/P_change/F_change 的净价值，STOP 固定 0、合法性 mask 后取最大，非正时停止，平局优先 STOP 再按固定动作次序。输入特征名及缩放写入 checkpoint；不得加入候选实际收益、未知目标数量、未来标签或未购预测。结构不是创新。

先按物理录制组交叉拟合末步；同录制的两车、相邻帧、片段、全部分支必须同折。首步训练时混入末步监督以免遗忘；冻结生成首步标签的教师版本，最终实际 rollout 测的是最终单网络，不拿教师标签均值替代部署结果。

驾驶器训练过的录制组仍可用于实现诊断/学习，但交叉拟合小网络并不能自动消除驾驶器在这些场景上的乐观误差。既有两个开发组已多次查看，不能改称全新独立测试；正式泛化要另有未用于设计/选模的录制或清楚限定为开发结论。本轮不因此启动多套 7B 训练。

研究判据按依赖分层：

- **C0 契约成立：** 参数真正影响执行、无未来/查询前泄漏、E/Z/费用准确。不能由此宣称性能有效。
- **H1 任务参数：** 同预算优于固定 ROI；否则不支持任务检索收益。
- **H2 修订关系：** 优于等候选/监督的 current+old/union，且真实反馈消融有一致增量；否则不能把 change 单列贡献。
- **H3 两轮必要性：** 在所测总资源档优于强单轮条件委托；否则撤回该档下两轮优势。
- **H4 新信息作用：** 优于相同 E/Z、等计算迭代；否则可能只是额外计算或表示效应。
- **H5 可部署选择：** 实际单网络 rollout 的质量—成本关系优于强规则/固定调用，改善、恶化、失败和录制组差异都报告。不能从 oracle/训练标签推出这个结论。

---

## 5. 按依赖顺序的 implementation plan

依赖：`T1→T2→T3→T4→T5→T6→T7→T8→T9`。这是同一主机制的实施顺序；对照在制作正式价值标签前完成，不能先训练后补关键反方。T1–T6 主要是代码与测试；T7 的真实分支制作、T8 的价值训练、T9 的真实 rollout 均属于后续执行阶段，本轮不运行。

T7 先实现两个标签构建 API，测试注入一个冻结 continuation 函数；真实数据执行顺序为：收集分支 → 末步标签 → T8 训练各折末步教师 → T7 用教师制作首步标签 → T8 混合末步监督训练共享策略 → 实际 rollout。不能在末步教师产生前运行首步标签制作，也不能把这个数据依赖省略成一次性离线最优挑选。

所有命令以下列方式在仓库根目录执行；资源检查与模型检查分开。测试用假 predictor/driver 只用于验证执行契约，不能作为真实模型实验发表。

### T1：冻结请求、字段身份与时序检索函数【工程基础；承载核心请求语义】

**状态：已完成。** 新增 16 项测试通过；实际规格见实施记录。

**修改文件：** 新建 `src/tools/task_spec.py`、`tests/test_task_spec.py`；更新 `scripts/check_review.py` 的显式轻量测试清单。实验参数集中在版本化 `ExecutionSpec` 中，与协议定义分离，运行记录保存完整配置，不将数值散落为协议常量。

**API / 数据：**

```python
# 字典保持 JSON 可序列化；函数使用 NumPy/标准库，不导入 torch。
validate_task_request(request, known_receipts)  # 成功返回 None，违规抛 ValueError
field_key(ref)                                # 返回 §2.2 的身份元组
history_proxy(window)                          # 返回 xy[N,1,6,2] 及每目标 proxy 状态
relation_scores(rho_old, rho_new)               # 返回 current[N], change[N]
rank_targets(window, request, forecast=None)    # 返回按确定规则排序的 handle/score 列表
```

`rank_targets` 是主方法 provider 排序的唯一实现：P 只用 causal tracking-state motion proxy；F 必须传入注入 predictor 的实际输出（本批测试使用 fake，不运行真实模型）。首次请求 provider 私有 `known_receipts={}`；后续只对照此前真实远端响应的 receipt 校验，不接受请求方自填 registry 或派生证明。

- [x] 在 `unittest.TestCase` 中写最小得分反例，并先运行确认缺函数/错误公式会失败：

```python
def test_difference_precedes_time_aggregation(self):
    old = np.array([[[1., 0.]]])
    new = np.array([[[0., 1.]]])
    current, change = relation_scores(old, new)
    np.testing.assert_array_equal(current, [1.])
    np.testing.assert_array_equal(change, [1.])
```

- [x] 加入 `test_near_relation_change_beats_far_distance_change`：用 2→4、100→80 米输入核，断言近处 change 更大；`test_history_gap_uses_actual_dt`：valid 在 -0.4/0，位移 2m，应外推 5m/s，不能按 0.1s 算。
- [x] 加入单状态回退、时间/坐标错配、NaN、错误方案形状、未知字段、未来历史、跨 source 同号 handle、请求 ID 不改变 field_key、old=new 屏蔽 change、打乱输入目标顺序仍同结果的测试。
- [x] 最小实现上述公式和严格类型/字段检查；P 路径安装“调用即失败”的 predictor，保证不偷偷执行 MTR。
- [x] 运行 `PYTHONPATH=src:tests python -m unittest test_task_spec -v`；再运行 `python scripts/check_review.py`，确认没有导入模型依赖。代码已独立审查并带回主工作区；本批未执行 git commit 或 push，原有研究材料保留。

**该步可验证：** C0 中请求语义确定、因果、可复算；同一输入只改合法参数的执行效果符合公式。

**不允许结论：** current/change 能改善驾驶；P 代理等于真实预测；change 是校准风险或新颖算法。

### T2：接入真实 P/F、字段去重和完整报文上限【核心机制的执行部分＋工程基础】

**状态：已完成服务代码与契约验证。** 新增 20 项测试通过，未新运行真实 MTR；完整 P 后不提前屏蔽 F。具体回执确认、预算预留与失败语义见实施记录。

**修改文件：** 修改 `src/tools/vehicle.py`，扩展 `tests/test_vehicle_tools.py`。复用 T1，不另建服务进程、网络协议库或新预测器。

**API / 数据：** 增加 `VehicleTools.query_task(request)` 返回 `{request, wire, cost}`；增加 `decode_task_response(wire, request)`。v1 `VERSION`、`query`、`decode_response` 保留。服务维护同 t 的窗口/预测缓存、已返回字段回执和上下文登记；返回 TaskResponse/FieldRecord/Receipt。

```python
# query_task 中的必须顺序；这段描述执行边界，不是新运行命令。
# 1 validate request/limits/manifest against receipts
# 2 load same-time private window
# 3 P: history proxy; F: full-context prediction or lawful cache reuse
# 4 remove known equivalent fields, rank remaining targets
# 5 serialize whole bundles under response cap
# 6 validate response, record actual wire costs and receipts
```

- [x] 测试固定窗口、两条不同合法轨迹使选中目标按规格变化；使用现有 `tests/test_vehicle_tools.py:window/fixture_prediction`，fake predictor 的输出显式依赖上下文目标数，避免“上下文错误也碰巧通过”。
- [x] 测试先 P 再 P_change：同一 history 不重传，别的合法目标仍可返回；先 F 再 P 时 forecast 不重传、history 可以新增；未知 manifest 引用拒绝。
- [x] 测试 F 排序前 predictor 收到全部目标；只返回一目标仍保留完整模型上下文。第一次 F 计计算，第二次同 t F 缓存命中但服务/报文成本不免除。
- [x] 测试改 request_id/mode/字段顺序不会制造新身份；同 ID 的不同 provider 或 context 不混淆；以未验证完整 P 声称 F 已知必须被拒绝。
- [x] 测试最终 UTF-8 报文含 manifest、头部、truncated/empty 状态均不超 cap，目标数满足上限；所有 bundle 太大时合法空响应，cap 连最小头部都容不下时无私有读取。
- [x] 保持原 v1 请求/响应黄金样本和现有工具测试结果不变；运行 `PYTHONPATH=src:tests python -m unittest test_task_spec test_vehicle_tools -v` 及轻量审查。

关键断言应直接作用于真实 service 的返回，而非手工构造相同字典：

```python
first = service.query_task(request_current)
second = service.query_task(request_change_with_first_receipt)
a = decode_task_response(first['wire'], request_current)
b = decode_task_response(second['wire'], request_change_with_first_receipt)
keys_a = {field_key(r['ref']) for r in a['records']}
keys_b = {field_key(r['ref']) for r in b['records']}
assert keys_a.isdisjoint(keys_b)
assert second['cost']['response_bytes'] == len(second['wire'])
assert len(second['wire']) <= request_change_with_first_receipt['execution_spec']['max_response_bytes']
```

测试里的两个 request 使用 §2.2 的完整字典；第二个由第一个真实回执构造 manifest，不能通过预填未知 handle 绕过接口。

**该步可验证：** C0：任务参数真的影响原服务执行；重复信息不会冒充新返回；硬字节预算不依赖免费窥视。

**不允许结论：** 两轮优于单轮、F 天生优于 P、本地/远端网络时延已验证。

### T3：E/派生/Z 台账与共同 receiver【工程基础，不能独立列创新】

**状态：已完成代码与契约测试，未运行真实模型。** 实际 API、工具端最小证明扩展和验证边界见 T3 实施记录。

**修改文件：** 新建 `src/planning/evidence.py`、`tests/test_evidence_ledger.py`；修改 `src/planning/context.py`、`src/planning/inputs.py`；扩展 `tests/test_compact_evidence.py`、`tests/test_framework_episode.py:SharedContextTests`。`scripts/check_review.py` 只注册无需模型/tokenizer 资源的类。

**API / 数据：**

```python
new_ledger(local_window, local_prediction, *, predictor, local_provenance)                 # 返回 ledger 字典
apply_response(ledger, response, predictor, p_processing=None)   # response={request, wire, cost}；返回新 ledger，不接 GT
known_field_manifest(ledger)                               # 只输出 provider 可验证的 acquired 远端回执
build_task_plan_input(tokenizer, motion, ledger, feature_tokens, limits)
# 返回 source_blocks_v2 prepared，含精确 admitted/dropped 与字段到对象/提示位置映射
```

`apply_response` 读取已验证响应；只在新购 P 改变已知历史集合时重算本地 MTR。`p_processing` 在整个比较中冻结为 local_mtr，observations-only 只作明确标记的诊断。旧 make_evidence 与 build_plan_input 默认行为不改。

- [x] 构造同目标同上下文 P_local/F，断言 E 保留购买事实、derived 保留父引用，而 Z 不重复放同一 forecast；同目标不同上下文预测不能被去掉。
- [x] 用一个只能放一条完整字段 bundle 的小容量测试：第二轮收到新字段后 E 增长、Z 可替换，旧入模引用进入 dropped；再次引用旧字段不会增加 E。
- [x] 验证每个 Z 项可还原到本车或 acquired/derived 的父链；把未购 F 注入 Z 必须失败。历史 scores、mask、mode 顺序、数值舍入与原始 bytes 的区别分别核对。
- [x] v2 对相同 E 不受 policy 名称、报文到达顺序、request_id/current/change 标签影响；同 source 同 id 的不同计算上下文仍可解码区分。
- [x] 完整 P/F 同信息测试先用上下文敏感 fake predictor 检查数据路径；真实同 MTR 数组对照留到后续获准的模型验证，不在本步骤假称已经验证真实模型。
- [x] 验证新接收器对各方法共享，Ego-max-context 真为 peer_reserve=0；旧 v1 保存提示逐字回归。运行 `PYTHONPATH=src:tests python -m unittest test_evidence_ledger test_compact_evidence -v`；模型环境另运行 SharedContextTests。

```python
# 容量替换的关键不变量；fixture 让新字段排在旧字段前且不能同时装下。
before_keys = {field_key(r['ref']) for r in ledger_before['acquired_fields']}
after_keys = {field_key(r['ref']) for r in ledger_after['acquired_fields']}
assert after_keys >= before_keys
assert old_ref in prepared_after['admission_report']['dropped_field_refs']
assert new_ref in prepared_after['admission_report']['admitted_field_refs']
assert prepared_after['evidence_selection']['input_tokens'] + 256 <= context_limit
```

**该步可验证：** C0：已购与入模不是同一件事；同信息预测对照具备正确前提；所有方法接收公平。

**不允许结论：** driver 已理解所有已购字段、去重本身是方法创新、不同 context 代表一定有益。

### T4：真实 GoT 交替、STOP 与失败时序【Method 核心】

**状态：已完成代码、契约及原 tokenizer/driver 适配接口验证，未执行真实模型。** 恢复粒度、诊断策略和成本记录边界见 T4 实施记录。

**修改文件：** 新建 `src/planning/method_episode.py`、`tests/test_method_episode.py`；修改 `src/planning/run_framework.py` 增加显式 `interact` 子命令，修改 `src/planning/v2vgot.py:plan_prepared` 校验共同 v2 输入。原 `episode.py` 五策略和 prepare/generate 子命令不迁移、不替换。

**API / 数据：**

```python
run_task_episode(local_window, local_prediction, motion, features,
                 service, predictor, driver, policy, limits, on_progress=None)
# policy(visible_state) -> {tool, mode, reason}；executor 绑定真实 plan、manifest、预算
# 返回 episode_v2：plans、requests、responses、ledger_snapshots、events、cost、final_plan_id、status
```

运行器加载一次 driver，再多次调用 `plan_prepared`。每个事件包含 sample/branch/stage/request_id/plan_id，来源指向实际前一步；生成前后及真实响应到达后立即持久化，不等全部分支完成。无效初始输出也有 completion/失败记录。

- [x] 用可记录事件的 fake service/driver 检查真实顺序，不能仅检查调用次数：

```python
assert events == ['driver:0', 'query:P_current', 'driver:1',
                  'query:F_change', 'driver:2', 'STOP']
assert second_request['tau_old'] == plans[0]['waypoints']
assert second_request['tau_new'] == plans[1]['waypoints']
```

fixture 的 driver 输出依赖实际收到的字段，policy 读取实际修订；更改第一响应必须能改变第二 request 参数或 STOP，不用硬编码动作测试冒充反馈。

- [x] 零/一/两调用分别最多执行 1/2/3 次 GoT；STOP 零额外远端读取/生成；第三请求被硬拦；初始 τ0 无效时 service 从不读取。
- [x] 中间修订格式/动力范围失败阻止下次请求，保留原始回答和全部成本；同证据重复生成允许不变，不伪造修订。
- [x] 修改未来标签文件不影响在线执行；给 runner 注入禁止打开 offline_labels 的访问钩子；请求策略参数中不得有 service/完整树/GT。
- [x] 打断在 response_received 与 driver_started 之间，归档可判断未完成；恢复只复用完整前缀，不能补造中间方案或覆盖失败。
- [x] 运行 `PYTHONPATH=src:tests python -m unittest test_method_episode test_framework_episode.EpisodeTests -v`；真实模型短链留待代码确认和单独安排执行后验证。

**该步可验证：** C0，及 H2 的执行前提：确有真实驾驶修订参与下一任务。

**不允许结论：** fake driver 证明真实 GoT 对证据敏感；闭环控制已实现；调用合法即驾驶收益成立。

### T5：多次生成与失败成本评价【工程基础＋实验评价】

**状态：代码与契约/旧结果回归完成，未执行真实新模型。** 当前统计范围、未知费用和预期分母定义详见 T5 实施记录。

**修改文件：** 扩展 `src/evaluation/framework.py` 增加显式 v2 episode 读取/评价；复用 `src/evaluation/planning.py` 数值指标；新建 `tests/test_method_evaluation.py`，扩展 `tests/test_framework_pipeline.py`。旧 evaluate/summarize 路径的历史口径不回写。

**API / 数据：** `evaluate_method(episodes_root, out, labels_root)`；`summarize_method(rows)`。行主键为 `(sample_id, policy_id, branch_id)`，计划事件另有 stage；保存 parse_valid、task_success、admissibility、raw ADE/FDE、缺标签情况、完整分母和 cost_complete。

- [x] 3 次 driver 时长 1/2/3 秒的合成 episode 断言 driver_seconds=6；只有一次的旧基线仍=1，不能只记最终输出。
- [x] service=2 秒且其中 peer MTR=1 秒时，总时间只加 2；receiver 内 P_local 同理，子项另报。
- [x] 失败发生在请求、响应后接收、生成三个位置，已发生次数/bytes/time 都保留；unknown 成本不能用 0 参与成本均值/价值训练。
- [x] 预期任务集合有无效 τ0、缺文件、重复分支、不同 recording 的同 g；缺失不能以“完成样本”替代 expected denominator。
- [x] 相同结果按旧/新 reader 的可比指标一致，旧 160 条归档的便携复算仍可运行；本批已完成原便携复算。
- [x] 运行 `PYTHONPATH=src:tests python -m unittest test_method_evaluation test_framework_pipeline -v`。

```python
assert row['driver_calls'] == 3
assert row['driver_seconds'] == 6.0
assert row['total_compute_seconds'] == sum(non_overlapping_stage_seconds)
assert report['attempts'] == len(expected_episode_keys)
```

**该步可验证：** H1–H5 的成本/失败归因前提；三次 GoT 的代价不遗漏、不重复。

**不允许结论：** 没有实网传输测量却宣称端到端实时性；固定调用数就是等 FLOPs；仅有效样本 ADE 代表全部部署成功率。

### T6：关键反方、旧基线兼容与等证据控制【仅实验对照代码】

**状态：已完成代码与契约验证。** 单轮冻结策略可注入但尚未拟合；所有模型结果是契约替身。实际增补文件/成本范围及前缀接口见实施记录。

**修改文件：** 新建 `src/planning/method_controls.py`、`tests/test_method_controls.py`；在 `src/tools/vehicle.py` 增加 controls-only `query_bundle(envelope)`；扩展 `src/planning/inputs.py` 和 `v2vgot.py` 的显式 refinement 输入校验；`run_framework.py` 注册对照配置，不改变主方法默认 direct 提示。`episode.py:choose_action` 作为旧策略被复用，行为不改。

**API / 数据：**

```python
make_control_policy(control_spec, value_policy)       # 返回 policy(visible_state)
make_bundle(state, public_summary, candidates, continuation_policy_id, limits)
VehicleTools.query_bundle(envelope)                  # 返回单轮 wire + 原语执行账本
build_refinement_input(prepared, previous_plan)       # 固定 E/Z，加明确模型答案槽
```

bundle 只执行本地登记的有限条件程序/冻结策略 ID，不执行请求携带的任意 Python；策略实现复用 T8，先以注入确定性函数做契约测试。候选/摘要固定在请求前，内部最多两原语；不能读取 ego 未发送内容或中途再访问 ego driver。

- [x] 冻结反馈对照共享相同首轮 prefix，只替换方案条件；状态摘要中的 τ1/差分也被一致屏蔽。current+old、current+union 的末步动作数与 current+change 相同，union 确实按两个方案逐时刻关系评分。
- [x] bundle 在两种第一真实返回下分别 STOP/第二 P 或 F；只产生一个外发 request/response，原语数为 1/2，全部内部计算被记录。禁止以免费 F 预测构造 if 条件。
- [x] 不能取得未返回的目标 handle、ego 私有 features 或未来；合法摘要/候选的大小计 request；变更 budget 可限制候选/第二动作，不能事后调额度。
- [x] exact-repeat 精确相同 prompt/Z/features 且 service 从不调用；self-refinement 仅多上一模型输出，Z 引用与值不变，固定最后一次回答，GT 不可用。
- [x] 对照双方 refinement 槽容量相同、传入新证据与固定证据路径的模板相同；额外调用、token、耗时进入 T5。
- [x] 旧 v1 五条件的工具时序和提示黄金样本不变；共同 receiver 兼容版本标 `receiver=v2`，不能复用旧轨迹宣称新 receiver 有效。Ego-max-context 不留 peer 预算。
- [x] 运行 `PYTHONPATH=src:tests python -m unittest test_method_controls.ControlTests test_method_episode -v`，并回归 `test_direct_planning`，保持 direct 无 Q8 假父回答。

```python
assert bundle_result['cost']['rpc_rounds'] == 1
assert bundle_result['cost']['capability_calls'] <= 2
assert refinement_input['admission_report'] == original_input['admission_report']
assert refinement_input['previous_plan_origin'] == 'model_generated'
assert repeat_events.count('service_query') == 0
```

**该步可验证：** H1–H4 的反方对照在执行上成立、没有人为削弱或信息/成本不等。

**不允许结论：** 一个测试条件程序就是最强单轮；同证据 greedy 不变就排除一切额外推理收益；尚未拟合/评价的基线性能已确定。

### T7：合法分支台账和末步/首步监督接口【实验基础】

**修改文件：** 新建 `src/planning/query_data.py`、`tests/test_query_data.py`；`run_framework.py` 增加明确 `collect-method` 入口；复用 `common.audit_protocol.recording`、已有 online_index/独立 offline_labels，不更改原数据划分或原标签。

**API / 数据：**

```python
collect_branches(online_index, out, runtime, controls_spec)  # 无 labels 参数/访问
make_terminal_targets(branch_root, labels_root, utility_spec, out)
make_first_targets(branch_root, labels_root, frozen_continuation, utility_spec, out)
```

branch 行保存 prefix、动作、request/response、E/derived/Z、实际 plan_id、全部成本、有效/失败原因、driver/MTR/receiver/query 版本和 physical_recording。离线目标表与在线状态表分目录；不能把首步价值标签塞回 request 特征。

- [ ] fake 小树无失败/去重/复用时，初始 1、两种首步 2、每首步四种末步 8，合计 11 次 driver；STOP 只引用现有方案。它不包含额外对照输出。
- [ ] 相同 first prefix 的孩子共享其真实回执/τ1；不同分支不共享本来未购买的缓存，不提前把全部响应交给 selector。
- [ ] 失败节点保留，不展开其非法子树；预算屏蔽动作标 infeasible，不能以预测未知结果屏蔽；等价请求合法复用需记录理由及部署应付成本，而非把回放成本当实际成本。
- [ ] 末步标签核对正/负改善、失败惩罚和每阶段费用。首步刻意设置“GT 最优动作”不同于冻结 teacher 的动作，必须用 teacher 实际选择的终态。
- [ ] 两车、相邻帧、同录制不同 scene 段、所有分支不能跨 fold；修改 GT 只改变标签，不改变 collect 的查询/生成路径。
- [ ] 使用缺失权重配置、缺失成本、缺失实际 Z 或错误 teacher 录制来源时标签制作拒绝，而不是静默填值。运行 `PYTHONPATH=src:tests python -m unittest test_query_data -v`。

```python
assert fake_tree['driver_calls'] == 1 + 2 + 2 * 4
assert target['continuation_action'] == frozen_teacher_action
assert target['terminal_plan_id'] == branch_index[frozen_teacher_action]['final_plan_id']
assert record_group not in target['teacher_training_recordings']
```

**该步可验证：** 分支覆盖正确；非先知续策略的监督定义可执行；旧五条件确实不足以替代这些分支。

**不允许结论：** 11 个输出即可证明论文有效；offline oracle 等于策略性能；跨折价值教师解决了底座预训练/训练来源问题。

### T8：普通请求价值模块、分组教师与实际策略接口【标准求解器，不作为独立创新】

**修改文件：** 新建 `src/planning/query_value.py`、`tests/test_query_value.py`；接入 `method_episode.py` 的 policy callable 和 `method_controls.py` 的冻结条件 continuation。使用现有训练环境，**不修改 `train_driver.py`，不联合训练 GoT/MTR**。

**API / 数据：**

```python
state_features(visible_state, feature_spec)              # 有序浮点向量 + 特征名/缺失标记
choose_query(values, feasible_actions)                   # STOP=0，确定平局
fit_terminal_policy(targets, train_recordings, config)   # 普通监督回归
fit_shared_policy(first_targets, terminal_targets, config)
load_query_policy(checkpoint)                            # 返回 policy(visible_state)
```

一份 feature_spec 明确列出 τold/τcurrent 的 24 个坐标、运动值/缺失位、方案差分统计、已购/派生/入模/丢弃按字段类型的计数与合法关系摘要、剩余资源、首个工具和阶段；规范化只用训练侧统计。不能以网络拿不到具体前景为由给它全部未购 peer 张量。确需更丰富状态也只能来自已知字段，并让所有学习对照同步可用。

- [ ] 用污染测试给输入附上 GT/未购预测/候选实际 bytes，白名单应拒绝而不是忽略后误用；改变标签不改变 state_features。
- [ ] STOP=0、全部负值停止、平局停止、初始 change mask、用尽预算停止；所有选择只在 feasible_actions 中。
- [ ] 物理录制留一/分组折：teacher 训练组不得含当前组；保存教师权重来源与特征/查询/receiver/driver 版本，版本漂移拒绝用旧标签。
- [ ] 共享网络训练 batches 同时保留首步和末步监督；最终保存单网络及合法动作定义。对照替换 task templates 时同特征、容量、训练组、更新预算和候选监督数量。
- [ ] 保存/重载输出数值一致，resume 的优化器/进度/随机状态可恢复，不能再次出现长跑无检查点。小网络优先简单完整 checkpoint，不复制 7B 的复杂管理框架。
- [ ] CPU 合成监督只验证回归/保存/选择契约，不写入真实实验目录；轻量特征/选择测试不导入 torch，模型测试分开运行。

```python
assert choose_query({'P_current': -0.2, 'F_current': 0.0},
                    ['STOP', 'P_current', 'F_current']) == 'STOP'
assert choose_query({'P_current': 0.3, 'F_change': 99.0},
                    ['STOP', 'P_current']) == 'P_current'
```

运行：`PYTHONPATH=src:tests python -m unittest test_query_value -v`，在现有模型环境完成需 torch 的保存/回归单测。真实录制上的训练必须在 T7 获准生成分支、效用配置被冻结后另行执行；本计划交付不启动它。

**该步可验证：** H5 的可部署选择器实现、非短视标签与部署接口一致；能真正 STOP、不会查询先看结果。

**不允许结论：** 训练 loss 低证明性能提升；首步教师标签均值代表最终共享网络；普通 MLP 结构本身有创新。

### T9：冻结全链验证、结果归因与文档交接【实验验证，非新增方法】

**修改文件：** 扩展 `tests/test_method_episode.py`、`tests/test_method_evaluation.py`、`tests/test_query_data.py` 和 `scripts/check_review.py`；按真实执行状态更新 `README.md`、`docs/STATUS.md`、`docs/github_review.md`。新结果写独立输出目录，旧记录不覆盖。

**API / 数据：** 沿用 T4–T8，不加新在线网络。每次运行的 manifest 固定 query/receiver/driver/MTR/value/utility 版本、录制组、样本集合、预算、解码与对照族；归档 expected/completed/failed keys 及可便携复算的终态表。

- [ ] 完整集成测试：修改首返回能改变同一 GoT 的输入，再改变合法第二请求；对照共享该 prefix，终态都对应实际 Z，费用能由 events 重算。
- [ ] 所有对照有 expected sample keys，失败不删；按物理录制配对汇报 ADE/FDE、改善/恶化/失败、资源曲线，不能把相邻帧当独立样本制造显著性。只有两个开发录制组时明确统计强度有限。
- [ ] 在任何真实运行前，先完成轻量检查和模型环境完整回归；此阶段运行新真实模型链、有限分支、价值训练和 rollout 的范围要按用户后续授权安排，不能从“同意第一批代码”自动跳到全量训练。
- [ ] 新机制测试若只由 fake driver/predictor 通过，报告写“契约通过”；真实模型接通后也只写实际完成的链。核心反方没跑完时不能写 Method 有效。
- [ ] 完整检查命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'`。使用 README 的模型环境；`python scripts/check_review.py` 只说明资源无关检查。

```python
assert set(report['completed_keys']) | set(report['failed_keys']) == set(manifest['expected_keys'])
assert not (set(report['completed_keys']) & set(report['failed_keys']))
assert report['cost_accounting']['all_driver_attempts_included']
assert report['evaluation_scope'] == 'fixed_t_open_loop_development'
```

**该步可验证：** 实际完成且有足够数据时，分别检验 H1–H5；失败时能定位请求、反馈、接收、底座或成本问题。

**不允许结论：** 已证明所有驾驶条件泛化、真实网络实时性、闭环安全或论文必被 CCF-B/C 录用；不得把未跑的对照写进结果段。

## 6. 哪些是 Method，哪些不是

| 身份 | 首版内容 | 论文/实现边界 |
| --- | --- | --- |
| **Method 核心候选** | current/change 的可执行 P/F 任务请求；真实返回后的同一 GoT 修订；实际旧/新方案及证据状态组织第二请求/STOP | 三者构成一条机制。效果及相对强近邻的增量必须靠 H1–H4 支撑，不能拆成三个已证实创新。 |
| **标准求解器** | 一个普通小型价值网络、剩余预算、末步/首步监督、STOP | 支撑选择；不包装新架构、新 RL 或新通用价值算法。 |
| **工程基础** | 版本协议、字段/上下文身份、E/derived/Z、去重、P 本地缓存、预算、失败、保存、共同 receiver | 保证主张可核对；不是独立论文贡献。 |
| **仅实验对照** | old/union/fixedROI/frozen-feedback、strong one-shot/bundle、repeat/refinement、同信息 P/F、Ego-max-context、旧五策略兼容 | 不进入主方法默认执行，不把对照能力变成新的工具。 |
| **当前不实现** | RSU/I、第三工具、Risk/Critic/Safety API 包装、peer 学习目标排序器、新 MTR/预测骨干、额外归因头、联合 RL、GoT 自回归函数调用、自然语言思维链、实网/闭环、全量驾驶续训 | 没有必要为 Tool 名称增加；后续如另有研究假设再单独讨论。 |

## 7. 建议第一批代码改动：只实施 T1＋T2

**建议用户确认这批，而不是一次批准后面所有训练/实验：**

| 文件 | 第一批实际改动 |
| --- | --- |
| 新 `src/tools/task_spec.py` | v2 请求/FieldRef 白名单；current/change 公式、历史代理、平局及固定参数；manifest 合法引用与预算规格。 |
| 改 `src/tools/vehicle.py` | 独立 `query_task/decode_task_response`；使用现有私有窗口和 predictor，先按任务检索，按字段去重，再按完整报文字节装包；v1 行为不变。 |
| 新 `tests/test_task_spec.py` | 时间对齐反例、远近变化反例、因果短历史、尺寸/坐标/类型、重复身份/未知引用。 |
| 改 `tests/test_vehicle_tools.py` | 对真实 service 类验证任务参数影响、完整预测上下文、第二次新增字段、上限/缓存/错误回执、v1 回归。昂贵模型边界使用显式测试替身。 |
| 改 `scripts/check_review.py` | 登记上述资源无关测试，维持无 torch/transformers 的检查约束。 |

验收产物是**一个实际会执行任务参数、不会把同字段反复卖回来的 P/F 服务接口**，而不是一组空 API。第一批没有 GoT 交替、没有价值网络训练，也没有新实验结果；完成后进入 T3/T4 补 E/Z 和真实驾驶反馈，不能在 T2 就宣布方法完成。

待确认的实施默认值汇总：包围圆间距＋σ=5m；P 最新两个有效状态按真实 dt 外推，单状态静态；固定六时刻/六模式；12 位得分平局；4 目标/4096B 请求/8192B 响应/24576B 总预算；字段 bundle 不拆；首版确定性距离 receiver；初始或修订无效终止失败；80m/s、30m/s² 仅作为宽松异常上限。**这些是常数和失败规则的建议收敛，不是另加第三机制；用户确认后按版本冻结。** 效用权重在后续制作标签前单独登记，本轮不从已有开发结果选择。

## 8. 本轮交付与自查

- 已按 Spec 优先级完成源码/历史状态审查和计划；本轮只新增本文，没有修改任何 Python、配置、测试实现或实验输出。
- 没有运行单元测试、GoT/MTR 推理、分支制作、训练或新实验。表中测试/命令都是之后实施的验收要求，不能引用为本轮 PASS。
- 本轮只读确认最新已保存评价和停止记录；不把用户新材料中的方法视为已实现。
- 规格覆盖自查：任务参数 T1/T2；字段身份/去重 T1–T3；E/Z T3；真实修订 T4；三次 GoT 成本 T5；强单轮及同证据推理 T6；分支与因果标签 T7；交叉拟合/教师漂移 T8；实际反方验证与旧证据保留 T9。
- 第一批代码等待用户确认，这是用户本轮明确要求，不是额外审批流程。
