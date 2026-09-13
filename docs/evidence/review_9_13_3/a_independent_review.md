# 9-13-3 A1/A2 独立只读审查

日期：2026-09-14
审查对象：`/tmp/toolv2x-9-13-3-a-review.diff` 中 A1/A2 的 5 个文件；不包含 B admission 脚本。
审查工作树：`/root/autodl-tmp/ToolV2X/.worktrees/t9-supervision-admission-review`

## 结论

**规格符合性：PASS。代码质量与最小性：PASS。未发现必须修复项，也未发现会使 A1/A2 监督污染重新进入 normalization 或 optimizer 的路径。**

这只是训练数据契约与合成测试层面的工程结论。它不证明 value policy 的效果、公平性、泛化能力或任何真实驾驶科学主张；本审查也没有运行训练、GoT/MTR、真实采集或新实验。

## A1：bundle / one-shot 原始 research role

- `src/planning/bundle_data.py:150-237` 的既有 `validate_bundle_archive()` 先重新验证原始 `selected_index.jsonl`，并把原始 sample identity 绑定到 initial task；随后把 state 的 `sample_id`、physical recording、fold 绑定到该 initial row，并要求 terminal task row 与 initial row 完全相同。role 因此由原始 selected row 贯穿到实际归档，而不是取自目标表声明。
- `src/planning/bundle_data.py:253-259` 在完成上述原始归档验证后，按 `recording(identity['scene'])` 检查每个拟训练 recording 的原始 role，非 `train` 立即拒绝。
- `src/planning/query_value.py:587-590,649-655` 将训练配置中的 `train_recordings` 传入 measured loader，并要求 public fitter 只接收可回溯原始归档的 v2 文件路径。Python 会先完整求值 `_bundle_examples(...)`，之后才调用 `_fit(...)`；而 normalization、checkpoint 创建和 optimizer update 均在 `_fit` 内，因此污染拒绝时序符合“在 normalization / optimizer 前失败”。
- `tests/test_bundle_data.py:207-257` 覆盖合法 validation archive 被故意加入 `train_recordings` 的拒绝、合法 train archive 的正常准备，以及 dict/v1 伪装不能绕过 public fitter。测试用 patch 还确认拒绝路径没有进入 `_fit` 或创建 checkpoint。

裁定：原始 `sample_id / physical recording / role / fold` 链接完整；现有 train/validation fold 映射没有被修改。

## A2：alternating 质量监督重新绑定

- `src/planning/query_data.py:443-453,462-474,497-530` 在 T7 target 生成时额外保存独立 `labels.jsonl` 快照，并在 metadata 中明确引用。原 T7 `_source()` / `_target()` 质量、失败和成本公式位于 `src/planning/query_data.py:393-440`，本 diff 没有修改这些公式。
- `src/planning/query_value.py:277-301` 要求新 target archive 带有独立标签文件，缺失的旧 target 明确要求离线重新生成；重复 label identity 也会拒绝。
- `src/planning/query_value.py:255-274` 对每个可行 target 重新从已验证 branch archive 读取真实 source prefix identity，按 `(sample_id, role)` 连接 offline label，解析 first target 实际记录的 continuation，并复用原 `_target()` 从真实 source 最后方案、真实 terminal 最终方案、冻结 utility 与标签完整复算。目标行的公共字段必须与复算结果逐项相同，因此同时改 `source_loss/target` 或 `terminal_loss/target` 但保持代数成立仍会失败。
- `src/planning/query_value.py:331-345` 只在复算通过后提取训练 feature/target；标签不进入 `state_features()`。`tests/test_query_value.py:112-204,221-240` 覆盖 source/terminal 联动污染、first continuation、weighted FDE、失败终态、STOP=0、标签错配/重复/缺失，以及“独立改变标签并重新生成 targets 后 feature 不变、实际 source loss 改变”。

裁定：source ADE/FDE、terminal ADE/FDE、source/terminal loss 和最终 target 都重新绑定到实际归档与标签；失败使用既有 `failure_loss`，STOP 仍为零 target/零 incremental cost，普通 alternating 增量成本仍按真实 prefix 后的非负差值计算。

## API、质量与最小性

- public `fit_bundle_continuation()` 不再接收 dict 或 v1 synthetic table。该行为会使仓库外依赖旧 synthetic public fitter 的调用失败，但在本仓库中没有生产调用点；它与 A1“任何进入 fitter 的 recording 都必须能恢复原始 role”直接一致。仓库内 synthetic optimizer fixture 改为调用私有 `_bundle_examples()` + `_fit()`，只保留合成 CPU 契约用途，不冒充可审计训练入口。因此不判为本任务范围内的 API 回归。
- 实现复用了既有 `_target()` 和 `validate_bundle_archive()`，删除了 query loader 中重复的成本/terminal 校验逻辑。三个源文件合计增加 36 行、删除 30 行，净增 6 行；没有新依赖、数据库、认证层或模型。
- `git diff --check` 与 5 个改动 Python 文件的 `py_compile` 均通过。
- 最小性专项结论：**Lean already. Ship.** 没有可在保持 fail-closed 契约下合理删除的新增抽象或依赖。

## 已有验证证据

- 基线完整轻量：`/tmp/toolv2x-9-13-3-baseline.log`，290/290 PASS。
- RED：`/tmp/toolv2x-9-13-3-red.log`，4 个目标负例均在修复前复现失败；`/tmp/toolv2x-9-13-3-red-bundle-entry.log` 额外证明 public synthetic/v1 降级在修复前可进入 fitter。
- GREEN：`/tmp/toolv2x-9-13-3-green.log`，4/4 PASS。
- 扩展 DatasetTests：`/tmp/toolv2x-9-13-3-dataset-tests.log`，6/6 PASS，包括 source/terminal loss、标签、失败、STOP 和 feature isolation。审查没有重复运行这些测试。

## 有界剩余关注

1. 独立标签是随 target 导出的离线快照；若有人同时蓄意改写标签、targets 和其引用的 raw archive，本方案不提供防篡改证明。这不影响当前规格裁定，因为任务明确要求原始数据复算，同时明确禁止扩展成密码学认证或数据库机制。
2. 无标签的旧 alternating target archive 会按设计失败并要求重新离线生成。迁移是有意的 fail-closed 变化；实际重新生成不在本审查和本批授权范围内。
3. 测试均为合成契约/轻量回归。它们建立“污染会被拒绝”和“既有公式不变”的工程证据，不建立真实模型收益或科学结论。
