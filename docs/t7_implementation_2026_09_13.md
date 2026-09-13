# T7 实施记录：合法分支采集与末步/首步监督接口

日期：2026-09-13。用户授权“开始 T7”。本批完成 T7 代码、CLI 接入及契约验证；未启动真实 GoT/MTR、真实分支采集、价值训练、驾驶训练或缓存重建，未实施 T8。

## 1. 本批完成了什么

T7 将已实现的交替执行器用于有限分支采集，并把监督计算放在独立离线入口。采集阶段只接收因果 online index；教师只接收其实际第一返回后的可见状态；离线 GT 用于计算已选终态的损失，不能反向决定教师选择。

| 文件 | 实际修改 |
| --- | --- |
| `src/planning/query_data.py` | 新增采集、分支归档校验、版本化效用配置、末步与首步目标构建。 |
| `src/planning/method_episode.py` | 支持合法初始前缀续接；提取共同 `decision_state`，供实际策略调用与离线状态核对使用。原参数检索和控制计时语义保留。 |
| `src/planning/method_controls.py` | `capture_control_prefix` 可显式捕获初始或第一次返回后的实际有效方案。原默认仍为一次返回后。 |
| `src/tools/vehicle.py` | 新增 `fork_task()`，复制实际前缀的独立回执、窗口、预测缓存和计数，共享同一冻结模型与 loader。 |
| `src/planning/run_framework.py` | 新增 `collect_method` / `collect-method`，复用原模型加载、因果窗口、特征和运动加载接口。 |
| `tests/test_query_data.py`、`scripts/check_review.py` | 新增 22 项测试并纳入轻量检查。 |
| 本文、`STATUS.md`、方法计划 | 更新当前完成边界、数据依赖与验证记录。 |

本批主要是实验基础和标准监督接口，不增加新的 Method 创新、网络、工具或训练机制。此前未提交的 T6 定向修复作为本批基线保留，没有覆盖用户文件。

## 2. API 与数据结构

```python
collect_branches(online_index, out, runtime, controls_spec)
make_terminal_targets(branch_root, labels_root, utility_spec, out)
make_first_targets(branch_root, labels_root, frozen_continuation, utility_spec, out)

collection_spec(value)
validate_utility_spec(value)
decision_state(episode)  # 返回 visible state、已绑定候选请求、public preflight
VehicleTools.fork_task()
capture_control_prefix(after_calls=0, **episode_inputs)  # 也支持 1，默认 1
```

`runtime` 使用原 driver、FrozenPredictor、`load_inputs(row, directory)` 和 provenance；CLI 可提供零参数工厂，在输出目录创建且 online index/配置校验后加载模型。测试只注入合成模型。前缀仍绑定同一实际 driver 实例和逐值相同的 features，不能把磁盘 JSON 当成可恢复的运行状态。

`toolv2x_query_collection_v1` 要求完整 `limits`、`local_provenance`、`control`、`recording_folds`。`limits` 继续保存完整 ExecutionSpec/receiver/driver 版本；`recording_folds` 只能以 `common.audit_protocol.recording(scene)` 得到的完整物理录制为键。同录制的片段、车辆和分支继承同一 fold，不能按 frame/scene 后缀分折。同录制也不能跨本次 index 的研究 role。配置不改变旧数据划分。

每次采集使用一个交替动作语法：`feedback`、`frozen_feedback`、`current_old`、`current_union` 或 `evidence_refinement`。固定额外生成、旧固定策略和 one-shot bundle 继续使用 T6 入口；这些图不能冒充这里的逐次 ego 修订树。T7 不声称已制作全部对照的正式训练数据，one-shot continuation 的拟合和监督接入仍须在 T8 对应控制路径兑现。

归档布局：

```text
branch_root/
  config.json                 # 完整规格、稳定绑定、运行审计绑定
  selected_index.jsonl        # 因果索引，拒绝未知顶层字段
  online_states.jsonl         # 实际可见状态、source plan、prefix、录制/fold
  branches.jsonl              # 动作、可行性、失败原因、真实 task 路径、child
  prefixes/*.json             # 实际请求前缀、费用边界、输入读取记录
  tasks/*.json                # T4 episode 全量阶段快照、原回答、E/derived/Z/费用
  inputs/*/                   # 原 runtime 的本车输入产物
  progress.json               # 预期样本、已完成、初始失败、实际采集 driver 次数
  launch.json                 # CLI 调用配置，仅 CLI 路径
  code_snapshot/              # CLI 实际运行前保存的源码

target_out/
  config.json                 # utility、teacher provenance、绑定与来源
  targets.jsonl               # 离线监督；不得写回 online branch 目录
  coverage.json               # 预期样本、目标、不可行、失败终态、初始失败明细
```

`toolv2x_query_state_v1` 记录 `state_id`、depth、物理录制/fold、source_plan_id、prefix_path、cost_event_count 和实际 visible state。分支记录区分 completed / failed / infeasible；infeasible 没有伪造的请求或结果文件。正常初始失败没有合法监督源，保留在覆盖统计中，不编造目标。

`binding` 用于教师配置一致性：状态版本、声明的 driver 模型版本与执行设置、MTR 模型版本/设置、query/receiver/ExecutionSpec/control/provenance。`runtime_binding` 单独保存本次完整 driver provenance 和 predictor 的运行内句柄；路径、加载耗时及运行内句柄不作为跨运行语义身份。归档仍用自己的 runtime_binding 核对实际 episode。这是版本与配置核对，不新增权重内容等价证明；调用方必须正确冻结并标识模型版本。

## 3. 分支展开与成本复用

无失败且所有动作均可执行的小树：初始 GoT 一次，P/F 首步各一次，两个首步后各有 P_current/F_current/P_change/F_change 四个末步，共 `1 + 2 + 2 × 4 = 11` 次 driver 调用。STOP 引用当前实际方案，不额外调用 driver。实际合法动作受调用数、修订情况和字节 mask 限制，11 不是每个真实样本必须凑齐的配额。

初始方案只生成一次。每个成功首步的孩子共享该首步真实 request/response、回执、τ1、Z1 和费用。每个孩子从该服务前缀独立 fork，某兄弟分支执行过的 F 不能给另一个兄弟免费预热预测缓存。P 不调用 MTR；F 完整合法 context 的计算与裁剪仍走原服务。

服务 loader 的实际读取日志可能由运行时追加到一个共享列表。采集器按每次执行开始的读日志位置切出新增读取，再加该分支已经发生的前缀读取；不会保存兄弟分支读取，也不会漏掉自身读取。每个阶段仍原子保存 task，异常与未完成尾部保留。

本批只复用实际相同的前缀，没有实现跨兄弟分支的等价结果缓存。所有可支付的动作后缀实际执行。归档写明复用来源；每条部署路径保留完整前缀费用，不能因采集少算了一次 GoT 就把部署时那次 GoT 记为免费。`physical_driver_attempts` 仅表示制作分支的实际调用次数，不是部署策略的累计调用数。

成功非 STOP 首步必须连接唯一、匹配实际首响应的子状态，包括剩余动作只有 STOP 的子状态。失败首步不展开非法子树。缺失子树、重复动作、孤立分支、错配 task 文件、动作与真实参数化请求不符时，监督构建拒绝。

## 4. 效用与冻结教师

`toolv2x_query_utility_v1` 要求调用方显式提供全部字段，不提供实验权重默认值：

```text
version = toolv2x_query_utility_v1
label_coverage = all_six
quality_weights = {ADE3: ..., FDE3: ...}
cost_weights = {request_bytes: ..., response_bytes: ..., total_compute_seconds: ...}
failure_loss = ...
```

权重必须有限、非负，失败损失为正，至少一个质量权重为正。缺字段、bool、无穷值和未知计时均不能静默填值。失败损失与费用尺度需要在正式数据制作前明确冻结；合成测试里的数值没有实验推荐意义。

末步目标为“当前真实方案损失 − 该动作真实终态损失 − 请求前缀之后的实际新增成本”。源成本取实际 driver_completed 前缀，不能减去另一条重新计时的 STOP 轨迹。本地 MTR、已执行 GoT 和已购证据的公共费用取消；新 service、receiver、输入构造、driver、控制计算计入。沿用 T5 的非重叠 measured_stages_v3，生成/MTR 内部子项不再次相加。STOP 是价值零基准，目标固定为 0。

质量复用 `evaluate_method_task` / `method_plan_row` 的原轨迹解析与 ADE3/FDE3。需要完整六点、身份匹配的离线标签。完整计费的真实失败使用显式 failure_loss；不能用失败原文的 raw_ADE 或之前有效方案冒充成功终态。缺失费用、旧未版本化控制时间、缺实际 Z 或无效归档拒绝制作对应监督。

首步接口接受 `fold -> frozen callable` 映射。callable 具有完整 provenance：

```text
version = toolv2x_continuation_v1
teacher_id
held_out_fold
training_recordings
binding
utility_spec
feature_spec
model_version = {name, revision}
```

训练录制必须已明确分组且不属于当前 held-out fold，同录制其他片段不能借改名通过。query/receiver/driver/MTR/utility 绑定漂移、非法动作或教师调用时配置改变均拒绝。完整 provenance 保存于目标配置，首步目标保存教师 ID、训练录制、实际 continuation action、终态 branch/plan 身份与费用。

教师收到的状态从真实前缀通过共同 `decision_state` 重新构造并逐项核对，尊重 frozen_feedback 的计划隐藏规则。只比较两份状态副本不足以证明因果性。教师看不到标签、兄弟结果或候选实际收益，调用得到的是独立副本；先完成 continuation 选择，再打开 labels。目标使用该选择对应的实际终态，即使 STOP 在离线 GT 下更优，也不替换教师选择。

## 5. 验证与修复记录

| 检查 | 实际结果 |
| --- | --- |
| T7 前基线 | 237 项轻量检查通过。 |
| 新增 T7 契约 | 22 项通过，包含实际执行器的 11 次小树、失败剪枝、不可行记录、读取隔离、冻结/old/union 语法、CLI、教师与效用。 |
| 完整轻量检查 | 259/259 通过；入口确认未导入 torch/transformers。 |
| 旧证据复算 | 160 条原始回答、32 帧、五策略各 ADE/FDE 与均值通过。 |
| v1 兼容 | 原 query/decode_response/make_evidence、choose_action/run_episode、三个旧评价函数 AST 与 T7 前基线相同。 |
| 独立复审 | 五项问题修复后通过；独立 20 项 T7、49 项 T4/T6/预算回归通过，另核对语义绑定及仅余 STOP 的子树。其后新增控制语法及末步正负收益测试已包含在 259 项中。 |
| CLI help / `git diff --check` | 通过。 |

主要修复均先复现失败再验证通过，包括：初始前缀、服务 fork、三个新数据 API、CLI、分支动作与 task 错配、缺子树、教师状态注入字段、运行时读取日志串入兄弟分支、初始失败分母，以及路径/加载耗时误作语义身份。

注入时钟的监督测试独立给出精确期望：一次末步后缀为 8 个合成时间单位，两次请求路径相对初始前缀为 14；分别保留全部后缀阶段、不重复扣前缀。此数值只检验记账，不是实际模型耗时。测试还令固定教师选择比 STOP 差的 F，核对首步标签仍指向该 F；改变 GT 只改变离线目标，不改变采集提示或教师输入。

复查命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_review.py
PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_query_data -v
python outputs/framework_epoch01_quick_eval_2026_09_12/verify_review.py
PYTHONPATH=src python -m planning.run_framework collect-method --help
```

本机独立复审原文：`/tmp/toolv2x-t7-review.md`。未运行需要完整模型环境的 integration suite，未将历史 tokenizer 检查列成本轮新验证。

## 6. 使用范围与下一步依赖

新增 `collect-method` 在真正调用时会加载原模型并生成实际分支；本轮仅运行 help 与注入 runtime 的 CLI 测试。命令要求 `--checkpoint`、完整 `--spec`、既有因果 `--data`，可用 `--role` / `--per-recording` 指定范围；没有自动批量采集、隐式恢复或训练步骤。输出目录必须新建，不覆盖历史归档。中断尾部保留，首版不提供部分树恢复。

后续数据依赖仍是：授权小批量实际采集 → 制作末步标签 → T8 按录制折训练末步教师 → T7 用冻结教师制作首步标签 → T8 混合首步/末步监督训练单一策略 → 实际 rollout 与对照评价。

接口可执行、合成标签正确不证明机制收益，也不证明 strong one-shot 已拟合。原来反复查看的开发录制组不因分组教师而变成独立测试集。本批在 T7 代码与契约验证处结束。
