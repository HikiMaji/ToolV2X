# T6 实施记录：关键对照与同证据控制

日期：2026-09-13。用户授权“先实现 T6”。本批完成对照代码与契约验证，停在 T6；没有训练、真实 GoT/MTR 推理、真实新方法效果实验或缓存重建。全部对照都属于实验控制，不作为新的 Method 创新。

## 已实现的对照

| 对照 | 实际执行与归因边界 |
| --- | --- |
| `feedback` / `frozen_feedback` | 共享实际首轮查询、响应、驾驶修订和 E/Z；冻结臂把策略可见 current/previous 以及第二请求的轨迹依据一致替换为初始方案。仍保留真实修订原文。可注入独立拟合策略；本批没有拟合。 |
| `current_old` / `current_union` | 首步 3、发生修订后的末步 5 个候选动作，与 current/change 对齐。old 使用旧方案关系；union 在相同时刻、相同模态上计算两方案关系的最大值。真实改变 provider 检索装包。 |
| `exact_repeat` | 完整复制原 prepared 输入，固定 prompt、E/Z、特征、driver 和解码配置，真实调用 driver API 至声明配额，不调用 service。契约测试的 driver 为替身。 |
| `self_refinement` / `evidence_refinement` | 前者固定证据，后者取得新证据；双方从初始阶段就预留同样的模型答案槽，使用相同 wrapper。上一方案来自实际模型回答，固定取最后输出；失败不回填旧答案、不用 GT 选优。 |
| `one_shot` | 初始 driver → 一次外发 envelope → peer 执行首个真实 P/F → 注册的条件策略选第二原语或 STOP → 一次回包 → 同一 receiver/driver。内部最多两个原语，没有中途 ego driver。支持额外一次同证据 repeat/refinement，也保留自然两次 driver 的低计算版本。 |
| `legacy_v2` | 复用原 `choose_action` 的 Ego/P/F/PF/rule 调度及因果几何判定，使用显式 controls ROI 检索和共同 v2 receiver。旧 v1 路径/提示/原产物继续保留；不能把 v1/v2 差异归因为查询机制收益。 |
| `ego_max_context` | 不查询 peer，peer reserve=0；完整运行配置与实际 receiver 都保存该设置。 |

`exact_repeat` / `self_refinement` 支持 `repeat_after_calls=0/1`：分别比较 Z0 上的三次生成，以及真实第一响应与修订后，在 Z1 上再生成一次。三次调用是配额对齐，不声称 token、FLOPs 或秒数相等。

强单轮的执行接口已经具备，但“strong”仍是后续需要兑现的实验要求：当前 CLI 条件程序是明确标记的诊断程序，接口接受注册的冻结策略；T8 才拟合并公平选择其策略。不能据此宣称任意手写规则已经代表最强单轮。完整本车特征上传并在 peer 运行 GoT 的更强代理尚未实现；若其在实际预算下可行，后续下交互优势结论前仍需评估，不能仅凭名称排除。

## 实际修改文件

| 文件 | 改动性质 |
| --- | --- |
| `src/planning/method_controls.py`（新增） | 版本化控制配置、策略适配、旧调度适配、因果候选构造、共享真实前缀捕获与校验。 |
| `src/tools/control_bundle.py`（新增） | 单轮有限条件委托、合法任务候选、真实 wire 解码、外层费用与失败归档。 |
| `src/tools/task_spec.py`、`src/tools/vehicle.py` | 独立 controls task 版本；old/union/ROI；共享 P/F executor；注册条件策略、query_bundle 与回执校验。 |
| `src/planning/context.py`、`src/planning/inputs.py`、`src/planning/v2vgot.py` | 显式答案槽预留、固定证据 refinement wrapper、原 GoT 的新布局校验。默认 direct 提示不改。 |
| `src/planning/method_episode.py` | 在原执行器内增加控制状态、重复生成、内存前缀分支及 bundle 调用/接收；保持失败/持久化逻辑。 |
| `src/planning/run_framework.py` | `interact` 接受可选控制配置，保存完整配置和独立 branch_id；共同 receiver 的 Ego 容量如实归档。 |
| `src/evaluation/framework.py` | 区分外部往返与原语尝试；加入已测 control 阶段，校验控制配置/分支；旧 v1 评价函数不变。 |
| `tests/test_method_controls.py`、`tests/test_control_bundle.py`（新增） | 20 项本车侧/集成测试、15 项 provider 测试；另有 1 项原 tokenizer/planner 契约测试。 |
| `scripts/check_review.py` | 注册 35 项新增轻量测试，模型依赖测试单独执行。 |
| 本记录、`docs/STATUS.md`、原实施计划 | 记录验收、能力与结论边界。 |

## API 与版本

```python
control_spec(name, **options)
make_control_policy(control_spec, value_policy)
make_bundle(state, public_summary, candidates, continuation_policy_id, limits)
capture_control_prefix(**episode_inputs)
run_task_episode(..., control_spec=None, prefix=None)
build_refinement_input(prepared, previous_plan, *, tokenizer=None,
                       token_counter=None, slot_tokens=256)
VehicleTools.register_bundle_policy(policy_id, callable)
VehicleTools.query_bundle(envelope)
decode_bundle_response(wire, envelope)
```

- `toolv2x_controls_v1` 保存完整 name、driver_calls、repeat_after_calls、refinement_slot_tokens、baseline 和 bundle 配置；数值是显式实验配置，不散落为新增协议常量。driver_calls 为最多 3 次的声明配额，没有后续修订容量时不再请求。
- 主请求 `toolv2x_task_v2` 仍只有 current/change。额外模式使用 `toolv2x_control_task_v1`；ROI 版本复用原 source/window 顺序，F 仍先计算完整 context，部分 ROI 的 P 不得出具完整上下文证书。
- `source_blocks_v2_refinement` / `toolv2x_refinement_v1` 标记显式 wrapper；原 GoT 验证实际 prompt、原 tokenizer 计数、来源及固定槽容量。容量不足时失败，不偷偷重选/丢弃 Z。
- bundle 是 `toolv2x_bundle_v1`。外发 first_request、合法 public_summary、预先生成的候选、continuation_policy_id；limits 保存完整 ExecutionSpec、max_calls、max_candidates、remaining_bytes、primitive_response_caps、wrapper_reserve_bytes。所有容量在发送前冻结，最终 cap 检查完整 UTF-8 wire。

`capture_control_prefix` 实际执行到第一次有效返回后的有效驾驶修订，返回仅内存使用的 episode、实际 features 副本和同一 driver 引用。分支校验实际数组值/dtype、元数据、driver 实例、同源同刻、本地 ledger 与 provider 的真实首轮回包；不接受裸归档冒充可续接前缀。它不是进程恢复或 T7 分支数据集生成。大特征不写入 episode JSON；归档保存实际相同前缀及 prefix_origin。refinement 配对额外要求前缀已经预留相同答案槽，不能中途给 direct 前缀增加未预留容量。

单轮 continuation 只能接收外发内容、第一真实返回和合法动作清单，选择已发送候选及 P/F/STOP；不接收 service、window loader、predictor 或 ego driver。仅本地登记的可信冻结 callable 可执行，不能在请求中传 Python。第二 manifest 由真实首轮回执形成；不会将 receiver 派生预测当作远端收据。接收端从实际外层 wire 重建内层回包，不能信任返回对象的未传输 sidecar。

当前候选构造提供 initial、显式 slowdown_scale 的减速候选和由因果 speed/yaw-rate 构造的恒速/恒转率候选；缺失运动状态时不编造候选。候选经相同轨迹可用性检查。没有增加预测骨干、主方法 peer 学习排序器或网络模型。

## 入口与成本

原 `interact --spec` 的 `limits` / `local_provenance` 保留；新增可选 `control` 字段，必须按 `control_spec(...)` 规范化保存完整配置。CLI 仍只提供已有诊断请求策略；单轮默认可登记 `diagnostic_conditional_v1`，程序接口可注入正式冻结策略。没有新增自动训练/实验命令，也没有运行 `interact` 真实模型路径。

`branch_id` 显式区分控制臂，旧五策略 v2 臂附 baseline 名；评价核对归档和配置一致。未指定 control 的 T4 归档仍按旧 policy_id 读取。旧五策略 v1 的 prepare/generate/evaluate 接口不变。

T5 扩展：

- `rpc_rounds`：实际外部请求尝试数量。
- `calls`：能力原语尝试数量；bundle 内部为 1/2，发生失败仍记录已开始原语。普通请求沿用原尝试计数；中断 bundle 无已执行记录时数量为 unknown。
- 外部 `request_bytes/response_bytes` 只计一对 envelope/wire；内层规范编码长度留在 receipt 验证与诊断中，不重复加到网络费用。
- `control_seconds`：本批显式记录的本车控制决策/候选构造阶段。决策处中断但未留下计时事件时为 unknown。它不是完整 executor/I/O 时间。
- control 运行使用 `toolv2x_measured_stages_v2`：原五个非重叠外层阶段合计再加已测 control_seconds。bundle 的 continuation/internal request 时间已在 service 外层内，MTR/生成核心时间仍是子项，不重复相加。
- 未指定 control 的旧运行仍是 v1 成本范围；新增 control 阶段不适用，不补造旧 policy 时间。仍不宣称端到端时延或实网性能。

失败、中断、缺标签、缺产物的预期分母继续由 T5 保留。CLI/归档测试全部在临时目录使用测试替身，不形成真实新方法结果。

## 验证与边界

| 检查 | 结果 |
| --- | --- |
| 修改前完整轻量基线 | 191/191 通过。 |
| 新 T6 轻量测试 | 35/35 通过（本车/集成 20，provider 15）。 |
| 最终完整轻量审查 | 226/226 通过；不导入 torch/transformers。 |
| 原 tokenizer + 原 GoT planner 输入校验 | 2/2 通过：新增 refinement 契约与旧 direct 回归；`_generate` 被测试替身替换，没有加载权重或真实生成。 |
| 原已发布证据便携复算 | 160 条回答、32 帧、五策略全部 ADE/FDE 与均值通过。 |
| v1 源码兼容 | query/decode_response/make_evidence、choose_action/run_episode、三个旧评价函数 AST 与基线相同。 |
| 独立最终复审 | 79/79 项定向回归通过，无遗留 Important/Critical；另独立核对 CLI/Ego 分支身份与篡改控制配置拒绝。 |
| CLI help、diff 空白检查 | 通过。 |

独立复审发现并复现了“同一前缀可传入不同 features”的问题；已新增实际内存输入绑定及拒绝回归。另补齐了控制决策中断费用未知、声明生成配额、不同控制臂归档身份、refinement 必须预留同槽的测试。关键新增与修复均经过 RED→GREEN；测试 predictor 不作为效果证据。

可复查：

```bash
python scripts/check_review.py
PYTHONPATH=src:tests python -m unittest test_method_controls.ControlTests test_control_bundle -v
python outputs/framework_epoch01_quick_eval_2026_09_12/verify_review.py
```

模型环境仅跑输入契约：

```bash
TOOLV2X_V2VGOT_ROOT=/root/autodl-tmp/V2V-GoT \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
/root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_method_controls.ControlModelTests test_direct_planning.DirectPlanningModelTests -v
```

这批证明的是对照可以因果、公平、可计费地执行。尚无拟合的请求价值模型/强单轮策略、真实对照结果、交互收益、实时性或论文有效性结论。T7 分支制作、T8 策略训练、T9 实际 rollout 未执行。
