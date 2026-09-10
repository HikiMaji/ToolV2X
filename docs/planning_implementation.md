# 原模型驱动的 ToolV2X 接入实施计划

日期：2026-09-10。用户已在框架设计后明确“OK，现在开始”，随后要求继续，授权按该设计实施。本计划在当前会话内执行，不再请求同一事项的确认。

**Goal:** 接通原 V2V-GoT 单车特征、原投影与 Q8/Q9 调用入口，并把原 CMP MTR 和真实 P/F 报文接到统一任务上下文。

**Architecture:** 只在 ToolV2X 添加适配代码，原 CMP 与 V2V-GoT 仓库作为只读组件来源。原 MTR 副本沿用既有适配；驾驶端导入本地 V2V-GoT 原模型。在线输入采用字段白名单，不复用 GT 生成的 QA 数据。

**Tech Stack:** 现有 Python 3.8 / PyTorch / NumPy / Transformers / V2V-GoT LLaVA / CMP MTR。使用 unittest，不新增服务框架。

**Spec:** [framework_design.md](framework_design.md)。本计划实现该设计的接入部分；适配训练、正式泛化评价和学习查询策略仍需在真实任务链接通后执行。

**执行结果（2026-09-10）：** 接入范围已实际执行，四个固定动作在一个真实验证帧上完成原 Q8→Q9，48 项测试和独立落盘复核通过。完整记录见 [planning_connection.md](planning_connection.md)。下方清单按实际实现与覆盖范围更新；这不代表完整研究框架已训练或评价。

用户再次要求“开始”后，已继续完成紧凑证据对照、因果输入与离线标签准备、真实训练帧样本及原模型监督前向；最新为 58 项测试通过、优化器 0 步，见 [evidence_adaptation.md](evidence_adaptation.md)。

## 全局约束

- 两车、P/F，RSU 和 I 后置；硬件不作为研究方向否定依据。
- 主驾驶模型只读取 Ego 当前/历史原特征；邻车证据只能通过报文进入。
- 不读取未来位姿、GT notable objects、GT Q8 答案或 GT 伙伴未来作为在线输入。
- 保留原模型、来源与实际执行状态，不用合成回答、岭回归或模拟规划结果代替原模型运行。
- 每项非平凡输入/解析/隔离逻辑先写可执行失败测试，再实现和验证。
- ToolV2X 当前不是 Git 仓库。直接在用户指定项目目录新增适配文件，保留原仓库与历史产物；不创建伪提交。

## Task 1：单车因果特征与任务提示

Files: `src/planning/inputs.py`, `tests/test_planning_inputs.py`。

接口：`load_ego_features(root, split, g)` 返回原 shallow 模式的张量数组与逐文件访问记录；`make_prompt(task, ego_state, evidence, q8_answer=None)` 仅接受明确字段；`parse_q8(text)` 和 `parse_q9(text)` 拒绝无效回答。

- [x] 测试与访问记录覆盖仅 Ego 的 t / t-1 特征、原框轴顺序、边界缺失、真实文件缺失错误、六点解析与 Q8 类别；测试夹具只提供因果 Ego 所需文件。
- [x] 提示末尾任务问题的回归测试先复现失败，再修复；真实模型失败和诊断产物保留。
- [x] 实现严格单车 loader、明确缺失状态、任务提示与解析；不加载原 GT QA JSON。
- [x] 执行新测试和既有回归测试。

## Task 2：原 V2V-GoT 组件调用

Files: `src/planning/v2vgot.py`, `tests/test_v2vgot_connection.py`。

接口：`load_projector()` 从发布的 non_lora_trainables.bin 严格加载原投影；`project_ego_features(features)` 调用原 `LlavaMetaForCausalLM.generate_point_features`。`V2VGoTPlanner.plan(features, ego_state, evidence)` 调用原模型生成 Q8，再用实际 Q8 回答生成 Q9，保留原文与解析结果。

- [x] 用原特征函数和投影权重测试 token 顺序、数值及输入响应；单测使用构造张量，真实数据另由 runner 验证。
- [x] 实现原组件导入与加载、缺项预检查和执行来源记录；完整 7B + 原 LoRA 已实际运行。
- [x] 用真实验证帧与发布投影权重运行，保存 `[1,540,4096]` token 及有限数值检查。

## Task 3：真实 P/F 报文与统一证据

Files: `src/tools/vehicle.py`, `tests/test_vehicle_tools.py`。

接口：`VehicleTools.query(tool, roi)` 只在调用后读取邻车窗口，P 返回合法观测历史，F 调用原 MTR；`make_evidence(local_window, local_prediction, responses)` 仅从已购报文构造源分离的规划上下文。第一阶段保留所有源目标与关联候选，不强制融合预测模式。

- [x] 验证不调用则不读邻车、F 保留完整内部上下文、时空契约、源分离和实际字节；完整账本保留目标，文本舍弃项单独记录。
- [x] 实现 P/F、白名单接收、源分离证据；P 本地预测与完整上下文等价对照复用同一 MTR。
- [x] 运行真实验证帧四个固定动作；30 个邻车目标的 P 本地重算与 F 预测/分数逐元素一致。

## Task 4：运行入口与状态交接

Files: `src/planning/run_connection.py`, `docs/planning_connection.md`, `docs/STATUS.md`。

- [x] 提供 `--mode prepare`，实际运行原预测/投影并保存规划输入；明确不产生驾驶成绩。
- [x] 提供 `--mode infer`，执行真实 Q8/Q9；运行异常非零退出，格式错误单独记为 `invalid_q8` / `invalid_q9`。
- [x] 当前环境完整推理已执行；v1 的 Q8 格式失败保留，修复后 v3 四动作均可解析，仍明确 `full_framework_complete=false`。
- [x] 保存运行命令、结果、代码副本、48 项测试输出和独立复核；更新资源、状态和用户分工约束。
