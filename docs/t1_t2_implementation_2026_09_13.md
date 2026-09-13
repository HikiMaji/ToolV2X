# T1/T2 实施与验证记录

日期：2026-09-13。范围：按用户确认的计划及三条补充硬约束，只实施 T1、T2，完成后停止。

**同日复审补丁已完成：** 针对 `9-13-1.md` 的四项 v2 边界问题，已补齐数值布尔值拒绝、JSON 原生请求容器、NumPy predictor 返回契约和有效历史尺寸检查。当前针对性 52/52、轻量回归 135/135 通过；下文 43/126 是原 T1/T2 交付时记录。补丁详情见文末第 6 节。

已实现确定的任务请求规格和独立 v2 P/F 服务入口；代码已放回原主工作区。没有实施 T3 的 E/derived/Z ledger，没有接入新的驾驶循环、价值网络或实验运行器；没有启动训练、GoT 新生成或真实新方法实验。下面的测试结果只支持执行契约与兼容性，不支持驾驶效果或创新成立。

## 1. 实际修改文件

| 文件 | 改动 |
| --- | --- |
| `src/tools/task_spec.py`（新增） | 版本化 ExecutionSpec、严格请求/来源/回执验证、结构化字段身份、因果历史代理、current/change 评分与确定性排序。 |
| `src/tools/vehicle.py` | 增加独立 `query_task`、`task_records`、`decode_task_response`，处理完整上下文、真实返回字段回执、原子装包和成本。旧 v1 方法及解码函数保持原样。 |
| `tests/test_task_spec.py`（新增） | 16 项 T1 契约测试。 |
| `tests/test_vehicle_tools.py` | 保留原 `VehicleToolTests` 的 7 项测试；新增 `TaskVehicleTests` 的 20 项测试。 |
| `scripts/check_review.py` | 将新规格测试加入明确的轻量审查清单，不增加模型依赖。 |

文档另新增本报告，并更新 `docs/STATUS.md` 与 `docs/superpowers/plans/2026-09-12-toolv2x-method.md` 的当前执行状态和补充约束。原工作区已有的研究文档内容保留。

## 2. 新 API 与数据结构

```python
ExecutionSpec(...).to_dict()
ExecutionSpec.from_dict(value)
validate_task_request(request, known_receipts)
validate_provenance(provenance)
field_key(ref)
history_proxy(window)
relation_scores(rho_old, rho_new)
rank_targets(window, request, forecast=None)

VehicleTools(window_loader, predictor, scene, g, provider,
             task_provenance=provenance)
service.query_task(request)      # {request, wire: bytes, cost}
service.task_records             # 脱离内部状态的副本，含完整请求/spec/响应/失败成本
decode_task_response(wire, request)
```

一个 VehicleTools 实例拥有一个固定 scene/source/g 的 v2 episode，最多两次开始执行的请求。v1 与 v2 窗口、预测缓存及回执状态独立，旧调用不能预热 v2。

### ExecutionSpec：实验参数集中记录

协议固定 current/change 公式、合法字段、同刻坐标/时间规则、排序与平局规则。以下数值集中在版本化 ExecutionSpec 中，可以配置，不是散落的协议常量。

| 字段 | 当前默认值 |
| --- | --- |
| version / profile | `toolv2x_execution_v1` / `contract_v1` |
| sigma_m / max_targets | 5.0 / 4 |
| max_request_bytes / max_response_bytes / max_episode_bytes | 4096 / 8192 / 24576 |
| ego_geometry | `circumscribed_circle` |
| ego_length_m / ego_width_m | 4.8 / 2.0 |
| max_plan_speed_mps / max_plan_acceleration_mps2 | 80.0 / 30.0 |

这些是契约实现的默认配置，尚未经真实新方法实验标定。请求携带完整 ExecutionSpec，响应完整回显请求，运行记录保留完整字典。同一个已开始的 episode 内不允许改配置重置预算；更换配置需新 episode。当前只支持包围圆代理，新增几何算法需要显式扩展版本校验。

### 请求、身份和回执

- `TaskRequest`：`version/request_id/provider/scene/g/coordinate_frame/tool/mode/times/tau_new/tau_old/execution_spec/acquired_field_manifest`。不再另设重复的 `ego_size` 或 `limits` 字段。
- `FieldRef`：`provider/scene/g/track_handle/field_kind/producer_version/context_version`。字段分为 `anchor/history/forecast`，request ID、receipt ID 和检索参数不参与内容身份。
- `producer_version`、`context_version` 均为 `{name, revision}`。提供方 provenance 为版本化 `tracking/prediction/context` 结构；名称和版本不能使用绝对路径或其他路径字符串。资源定位路径可以留在未来运行配置旁路。
- `acquired_field_manifest`：`[{receipt_id, ref}, ...]`。provider 仅对照自己记录的该 receipt 实际返回字段验证，不能信任请求者传来的 registry、内容声明或本地派生证明。
- `TaskResponse`：完整 `request`、`receipt_id/provenance/records/references/ranking/status/coverage/truncated`。新远端字段为 `{ref, value}`，已确认收到的字段仅引用合法旧 receipt；当前 T2 的 records 不混入 receiver-derived 项。
- `task_records`：完整请求和 ExecutionSpec、已完成响应或失败阶段、实际请求/响应字节、服务/模型耗时、计算/回退目标数、缓存命中、成本完整性。尚无 episode 运行器将其自动写盘；该持久化调用属于后续获准的执行路径。

## 3. 执行语义

### current/change 真正改变服务结果

时间固定为 0.5–3.0 秒的六个驾驶时刻，坐标为同一 `ego_at_t`。根据包围圆间距和 ExecutionSpec 中的 sigma 得到关系值 rho：current 取 `max(rho_new)`；change 先对同一时刻、同一预测模式计算 `abs(rho_new-rho_old)`，再取最大值。排序为 `(-round(score,12), track_handle)`。

current 的 `tau_old` 必须为 None；change 必须包含合法且不同的 old/new 方案。原点、形状、有限性及配置中的速度/加速度上限均校验。排序按实际任务参数执行，再按目标上限和实际报文字节装包，不只是更改标签。

P 采用 **causal tracking-state motion proxy**：取最近两个 valid 跟踪状态，按真实时间差外推；不足两个时显式静态回退。`history_valid` 包含跟踪器因果维持状态，不等于逐帧 detector hit。代理只用于检索，P 不调用 MTR，也不返回伪装成 MTR 的 forecast。

F 在排序和裁剪前调用注入的 predictor，输入为完整合法 peer context 的独立副本；返回目标、历史有效性、预测形状和静态回退必须相互一致。预测器不能改写用于已签发回执的固定窗口。第二次 F 可以复用同刻完整预测缓存，但仍记录服务时间和真实报文字节。此路径可接原 CMP predictor，本轮仅用上下文敏感的 fake predictor 做契约测试，没有新运行真实 MTR。

### T2 仅去重远端真实返回字段

例：第一次 F 返回某目标 anchor+forecast。第二次 P 附上该真实 receipt 后，可引用 anchor 并新增 history；未返回或被裁掉的其他目标，不能借该 receipt 声称已购。

完整 P 后再请求 F 仍合法，F 仍按完整上下文计算或使用合法缓存。本轮不根据本地 MTR 派生结果屏蔽 F。完整 P 与同模型 F 的等价证明和派生字段处理留给 T3。

伪造、未知、跨 provider/scene/g/context 的引用，以及其他实例的 receipt，均在私有窗口读取前拒绝。回执注册表只记录最终 wire 中真正返回的新字段，不记录候选、计算过但被裁掉的字段或旧字段引用。

当前 manifest 是明确的收到确认：如果请求者不附上此前合法的确认，服务允许再次传输该字段，并重新计字节；其 FieldRef 保持相同，不能宣称新增信息。后续合法请求构造器应从已购远端 ledger 完整生成 manifest，才能保证正常两步方法不重复购买。这是本批显式保留的接口行为，不是 T3 已完成。

### 字节与失败成本

最终 cap 对完整 UTF-8 wire 检查，包含请求回显、manifest、provenance、receipt、引用、排序、空/截断状态等全部字段。按“共享 anchor 或引用 + 完整 history/forecast”原子装包，不裁掉半个 bundle。测试覆盖非 ASCII 字符及 JSON `false/true` 长度差导致的边界。

发起服务执行前，按已花双向字节 + 当前请求实际字节 + 完整 response cap 保守预留 episode 预算；不先窥视未知 peer 内容预测是否能装下。最小完整头部也装不下、预算不足或非法引用的预检拒绝，不读取私有窗口、不消耗模型执行次数；调用侧未来需记录这些拒绝事件。

开始执行后的失败计入两次尝试上限并保留已花请求字节、服务时间与模型时间。模型失败时无法确认的完成目标数记为 None，而非猜成 0；失败预测不进入缓存。空响应仍付费，`coverage=not_established`，不等于观测到了自由空间。

## 4. 测试与回归

主工作区，Python 3.8.10，2026-09-13 实际执行：

```text
PYTHONPATH=src:tests python -m unittest test_task_spec test_vehicle_tools -v
Ran 43 tests ... OK

python scripts/check_review.py
Ran 126 tests ... OK

git diff --check
通过，无输出
```

| 范围 | 结果及关键断言 |
| --- | --- |
| T1 新增 | 16/16：公式反例、真实时间差、参数生效、未来/额外字段拒绝、结构身份、确定性排序、完整 F 对齐、回退一致性、数值精度与 list-backed 输入。 |
| T2 新增 | 20/20：真实服务选择、两步回执、完整 P 不屏蔽 F、完整上下文与缓存、伪造/未返回字段拒绝、最终 UTF-8 cap、两次调用、预算不可重置、预测器修改隔离、失败账本、解码篡改与空响应。 |
| 原 v1 工具测试 | 7/7，旧 `VehicleToolTests` 内容未改。 |
| 总轻量回归 | 原 90 项 + 新 36 项 = 126/126；不导入 torch/transformers。 |
| 旧归档黄金测试 | 保存的报文成本、早停、20 次旧驾驶生成/提示/指标及完整 P 与 F 的 30 个目标数组同信息检查等继续通过。读取旧产物，不产生新生成。 |

另将修改前 HEAD 的 v1 实现加载为独立模块，在固定计时器和相同上下文敏感 fake predictor 下，比对 P 全量、F 窄 ROI、F 全量缓存、P 空 ROI、F 空 ROI 五种连续调用的完整 request/wire/cost，均一致。AST 核对旧 query 和六个原辅助函数未改。独立只读审查复跑 43 项测试通过，提出的数值精度、回退一致性和预测器修改问题已补测试并修复。

这不是完整模型集成测试；没有重新运行真实 7B、真实 MTR 或新的驾驶评价。旧归档同信息检查不等于新的 v2 派生证明已经实现。

## 5. 与原计划的调整及当前边界

1. 按用户硬约束，将原计划的 execution_spec 常量改为完整版本化配置；原计划重复的 ego_size/limits 合并入 ExecutionSpec。
2. `known_receipts` 明确为 provider 私有字典，manifest 仅承认真实远端字段；原计划提及的“派生证明”不进入 T2 请求协议。
3. provenance/model/context 用稳定 `{name, revision}`，不使用路径作为语义身份。当前没有内容 fingerprint、跨实例持久化回执或本地派生等价证书；相同版本字符串不是等价证明。
4. episode 预算采用完整 response cap 的保守预留：即使未知实际响应可能更小，也可能提前拒绝。后续正式比较应共享该行为，不能把它当作已优化通信效率。
5. 未显式确认收到的旧字段可重传，身份不变；后续正常调用方需完整携带可验证远端 manifest。空响应、截断和未知内容均不能作为免费证据。
6. 当前 decoder 能检查引用是否合法出现在请求中，但不持有旧回执数值，无法独立比对被引用旧 anchor 的内容；provider 固定窗口保证本批响应一致，接收端跨回执数值/派生证明校验留给 T3 ledger。
7. T1/T2 承载方法的“任务参数决定能力执行”部分；版本、白名单、缓存隔离、回执、字节账本属于工程基础。新增 fake/归档测试属于验证代码，均不能单独包装成创新。

**停止位置：T1、T2 完成；T3 及以后未实施。** 实际 GoT 修订影响第二次请求/STOP、E 与 Z 分离、三次 GoT 成本、强单轮条件委托、同证据额外生成和请求价值学习仍是后续计划，不因服务契约通过而视为完成。

## 6. 09-13 复审边界补丁

用户要求先修复四项已核实问题，不因边界问题暂停整体方法推进、重建缓存或增加训练。本次仅修改 `src/tools/task_spec.py`、`src/tools/vehicle.py` 及两个对应测试文件，并同步本报告和 STATUS。

- `_array` 在 NumPy 隐式转换前检查原始数值叶子，拒绝混合 `bool`；纯布尔数组仍被拒绝。请求/响应数值数组、因果窗口的状态/分数/时间共用校验。输入在到达接口前已被调用方转换成普通数值的情况，接口无法反推其原始类型。
- 请求只接受 JSON 原生 dict/list/标量结构；tuple 等容器在读取私有窗口和登记执行之前拒绝，不引入容器转换层。
- F predictor 的五个必需字段 `track_ids/states/means/scores/model_used` 必须为 NumPy 数组，与原 CMP 输出一致；列表包装结果在写入可复用缓存之前拒绝。额外的原 CMP `local_gmm` 不改变此要求。
- provider 窗口和 v2 响应解码均检查所有 `valid=True` 历史状态的 length/width/height 为正；缺失历史的零占位继续允许。

保留原 current/change 公式、P 不调用 MTR、F 完整上下文、回执与预算语义。合法预测不会因当前 bundle 装不下而被清空；新增测试验证 `budget_empty` 后第二次 F 命中合法缓存，预测器只执行一次。没有重建任何现有数据缓存，也没有模型训练/生成或 T3 实施。

验证环境：Python 3.8.10、NumPy 1.24.2。新增 9 项测试，其中 8 项在修复前复现待修问题（30 个失败子例），另 1 项保护合法缓存保留行为。修复后：

```text
PYTHONPATH=src:tests python -m unittest test_task_spec test_vehicle_tools -v
52/52 通过
python scripts/check_review.py
135/135 通过
git diff --check
通过
```

旧 v1 query 和六个辅助函数的 AST 与补丁前一致，原 v1 工具及保存的黄金证据回归继续通过。未运行完整模型集成，fake predictor 不作为方法效果证据。
