# 9-13-3：监督入口修复与两帧 admission 审计

完成日期：2026-09-14（UTC+8）。依据 [9-13-3 原始要求](9-13-3.md)，基于已发布的 T9 两帧 main 快照实施。A1/A2 已复现并修复；B 仅复算原归档的文本与 token，并模拟 admission，没有修改 receiver 或旧 T9 结果。

## A1：bundle / one-shot 恢复原始 research role

修复前，合法的 validation bundle archive 即使被错误写入 `train_recordings`，也能越过训练数据准备入口。已有的 fold 检查没有阻止这种配置错误。

现在公开 `fit_bundle_continuation()` 必须接收 measured v2 targets 文件路径。loader 先运行既有 `validate_bundle_archive()`，把原始 `selected_index.jsonl` 的 sample、scene/physical recording、role 与真实 initial/terminal task row 绑定，并核对 state 的 fold；随后检查每个拟训练 recording 的原始 role 必须为 `train`。检查在 `_fit()` 之前完成，因此 normalization、checkpoint 创建和 optimizer update 都不会先执行。没有调整任何录制划分。

合法 train archive 仍能通过；validation archive 被错误声明为训练、或把 measured 数据降级包装成 dict / synthetic v1 来绕过原始归档，都会被拒绝。小型 CPU 合成夹具直接使用已有私有数据准备与拟合函数，仅验证优化器契约，不是公开真实训练入口。

## A2：质量监督重新绑定原始方案与标签

修复前，同时增加 `source_loss` 与 `target`，或同时增加 `terminal_loss` 并相应降低 `target`，仍能通过代数一致性校验。first targets 中实际选中的 continuation 也有同一问题。

现在 T7 targets 导出时额外保存独立 `labels.jsonl`，metadata 明确引用它。T8 loader 重新读取真实 source prefix 的最后方案、实际选择的 terminal branch 最终方案、匹配 `(sample_id, role)` 的离线标签及冻结 utility，复用原 T7 `_target()` 完整复算 source/terminal ADE3、FDE3、加权 loss、实际增量费用和 target。所有目标表公共字段必须与复算结果一致。标签不进入在线 state features。

原 `_source()` / `_target()` 公式没有修改：失败终态仍使用冻结 `failure_loss`；STOP target 与增量成本仍为 0；普通动作的成本仍只计实际 source prefix 之后的增量。测试另覆盖非零 FDE 权重、真实选择的 first continuation、无最终方案的失败终态、缺失/重复/错配标签。

有两项有意的接口收紧：

- `load_measured_bundle_targets(path, *, train_recordings=())` 新增可选训练 recording 校验；默认空集合仍支持离线验证数据复算。公开 bundle fitter 要求 measured v2 原始归档路径。
- `training_examples()` 要求标签 sidecar；缺少它的旧 target archive 会明确报错，需要日后从原始分支与标签离线重新导出。**本批没有批量重建 targets、缓存或重跑模型**。元数据继续使用既有 target v1，新增必需的 labels 引用；旧文件不会被静默接受。

这是一致性复算，不是防恶意同时改写所有原始材料的认证机制；没有引入认证、数据库或新模型。

## 修改文件

| 文件 | 改动 |
| --- | --- |
| `src/planning/bundle_data.py` | 从原始归档验证拟训练 recording 的 research role，loader 传递训练组 |
| `src/planning/query_data.py` | terminal / first targets 导出独立离线标签 sidecar，原监督公式不变 |
| `src/planning/query_value.py` | bundle 公开入口约束；alternating 从真实方案、标签和成本重新计算监督 |
| `tests/test_bundle_data.py` | 原始 validation 污染、合法 train、公开入口降级绕过的回归 |
| `tests/test_query_value.py` | 联动损失污染、first continuation、失败/STOP/FDE、标签及特征隔离回归；合成 CPU 夹具适配 |
| `scripts/audit_t9_admission.py` | 只读 provider→ledger→receiver→token→Z 诊断与共同 round-robin 模拟 |
| `tests/test_t9_admission.py` | 模拟顺序、超限跳过和请求/证据可见阶段的契约测试 |
| `scripts/check_review.py` | 纳入不加载模型的 admission 测试 |
| `README.md`、`docs/STATUS.md`、`docs/github_review.md` | 更新当前执行与审查入口，保留历史结果 |
| 本报告、admission 报告、执行计划及 `docs/evidence/` 两个本批目录 | 保存要求、逐 unit 明细、复算断言与验证日志 |

P/F、receiver、GoT/MTR、旧 v1 query/decode 和五策略执行器均没有代码改动。

## 测试与证据

| 验证 | 结果与范围 |
| --- | --- |
| 修复前完整轻量基线 | 290/290 PASS |
| RED | 原始 validation→train 及三种 loss/target 联动污染均复现为缺少应有拒绝；另复现 dict / synthetic v1 公开入口绕过 |
| GREEN | 同组 4 个测试方法全部通过；含 validation/train 正负例及降级拒绝 |
| 扩展监督数据契约 | 6/6 PASS，覆盖标签、FDE 权重、失败、STOP、continuation 和 feature isolation |
| CPU 合成拟合/重载/恢复回归 | 10/10 PASS，显式禁用 GPU；仅临时合成小网络夹具，不是实际 value-policy 拟合 |
| 完整轻量回归 | 300/300 PASS，199.664 秒；不导入 PyTorch / Transformers，不运行真实模型；最后补齐 dropped 表后另通过 4 项定向检查与原 tokenizer 复算 |
| 原 32 帧 / 160 条驾驶回答 | 现有独立脚本复算 PASS；旧答案、标签、分母与 ADE/FDE 不变 |

日志、独立 A1/A2 及最终整体审查见 [本批验证证据](evidence/review_9_13_3/)。本批没有运行包含真实模型调用的完整模型集成套件；历史 T9 的 345 项结果属于上一批，不能算成本批的新模型验证。小型 CPU 优化器检查仅证明拟合/恢复接口仍可用，不证明策略收益。

## B：两个不同 provider ranking 为什么得到相同 Z

详细复算、完整脚本命令及所有列定义见 [admission 审计](t9_admission_audit_2026_09_13.md)。逐 unit 的 [实际表](evidence/t9_admission_audit_2026_09_13/remote_units_actual.csv) 与 [模拟表](evidence/t9_admission_audit_2026_09_13/remote_units_counterfactual.csv) 各有 64 行，覆盖六个已有远端证据阶段；全部 12 个保存的 plan stage 都核对了完整 prompt、token 总数、admission refs 与位置。provider 保存记录和 ledger receipt 也一致。补齐初始/Ego 阶段后，[dropped fields 表](evidence/t9_admission_audit_2026_09_13/dropped_fields.csv) 包含全部 12 阶段的 482 条丢弃字段（408 local、74 remote/receiver-derived），与旧归档逐项相等。

receiver v1 先按 anchor 距离排序，不沿用 provider rank；同一近目标的不同 field/context 连续进入候选顺序。最终阶段先装入该目标的 provider-full forecast，再装入 receiver-subset forecast，剩余候选逐个尝试后均超过固定预算。两种 forecast 的 context 不同，本报告没有将其认定为重复信息。

| frame / arm / 阶段 | 已有 units / targets | 入模 units / targets | 实际入模目标 | 固定预算 round-robin targets |
| --- | ---: | ---: | --- | --- |
| g5526 alternating，P 后 | 8 / 4 | 2 / 1 | 20 | 20、6 |
| g5526 alternating，final | 12 / 6 | 2 / 1 | 20 | 20、6 |
| g5526 one-shot，final | 12 / 5 | 2 / 1 | 20 | 20、6 |
| g7007 alternating，P 后 | 8 / 4 | 2 / 1 | 0 | 0、5 |
| g7007 alternating，final | 12 / 4 | 2 / 1 | 0 | 0、5 |
| g7007 one-shot，final | 12 / 5 | 2 / 1 | 0 | 0、5 |

六个 Ego/初始阶段没有远端 unit。g5526 的 F(change) `[6,20,3,8]` 与 F(current) `[3,6,20,4]` 最终都仅留下 target20；g7007 的 `[0,21,5,11]` 与 `[0,5,2,21]` 都仅留下 target0。两臂每帧的最终 Z、prompt 和保存轨迹相同。**这定位了两帧中差异消失的位置，不能外推为全部场景的主要瓶颈或新方法无效。**

独立加入一个 unit 的 token 增量如下；它包含 remote observation header，相对同阶段的 local-only prompt 计数，不能将各列简单相加。逐 unit 表同时保留每次完整 prompt 尝试总数与实际累计总数。

| frame | local-only tokens | history 增量 | receiver-subset forecast 增量 | provider-full forecast 增量 |
| --- | ---: | ---: | ---: | ---: |
| g5526 | 2292 | 777–794 | 714–737 | 705–743 |
| g7007 | 2300 | 789–805 | 726–746 | 711–727 |

context 固定 4096，generation reserve 固定 256，输入上限为 3840。g5526 final 依次为 3016→3573 tokens，再加同目标 history 会到 4194；g7007 final 为 3017→3572，再加 history 为 4191。后续其他目标也逐个超限，没有因为第一个失败而直接停止尝试。

共同 round-robin 模拟保留同样两份 unit，但分给两个目标，每阶段多覆盖 1 个目标；没有证明它达到最大覆盖，更没有生成新轨迹或计算新 ADE/FDE。

## receiver v2 候选，仅提案

| 共同候选 | 能改变什么 | 公平性与效果风险 |
| --- | --- | --- |
| 稳定 target round-robin | 每个目标先取一个 unit，再开始下一轮；已有两帧模拟可增加一个目标 | 多目标覆盖不等于任务相关性；第一轮字段顺序和大 unit 仍影响谁能入模 |
| 每 target 分轮 field 配额 | 所有目标先竞争首份证据，再按共同上限补第二份 | 配额可能压掉同目标不同 context 的互补证据，也可能留下无法利用的预算；track 碎片化会影响公平性 |
| 共同任务相关性 + 实际边际 token 成本 | 从已购内容、共享本车输入和共同初始方案 τ0 评分，逐步重新计算装包成本 | 评分权重、字段规模和顺序依赖可能偏向某类证据；规则需要固定并对所有臂开放，不能用 alternating 独有 τ1 |

三种候选均须由 Ego/P/F/PF/one-shot/alternating 共用，使用相同 context/generation budget；不读 GT、未知 peer 信息或 alternating 独有中间方案。当前优先可考虑最容易解释的 round-robin，但**本批没有实现任何 receiver v2**，也没有声称需要重建缓存、扩大 context 或增加训练。

## 停止状态

本批完成 A1/A2 修复、旧产物复算和 B 离线 admission 诊断后停止。没有实际策略训练、新帧/新分支采集、GoT/MTR 执行、receiver 改动或旧 T9 结果修改；本地代码交付不包含自动 GitHub 推送。
