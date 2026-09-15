# 9-15-2 核验：P/F/PF 的 10 组直接证据差异

**后续落实：** 用户批准的原因报告和角色转换回归已实施，见[修正记录](admission_reason_fix_2026_09_15.md)。下文保留最初只读核验的结论与当时建议。

结论：已从原始归档逐字段定位。10 个样本/远端 track 组合在 F-only 中为 `unresolved`，在 P 和 PF 中因历史与本车定位历史吻合而被判为 `ego`，继而触发 `ego_filter`。不是随机容量截断，也不是这 10 组目标在 Hungarian 关联中被合并或换了身份。本轮仅做离线归档检查，未修改生产代码、训练、运行预测器/驾驶器、重建缓存或重新采集。

## 复核范围与结果

使用 `outputs/structured_driver_fix_2026_09_15_v1/acceptance/` 中全部 20 帧的 P、F、PF 最终输入，共 60 份。重新执行真实 `validate_structured_prepared`，验证实体、关联、张量、容量和字段依赖；再用完整 source/scene/g/track/field/model/context 身份比较购买和入模状态。没有按有利结果筛样本。

| 条件与字段 | 已购 | 直接入模 | 未直接入模 | 实际原因 |
|---|---:|---:|---:|---|
| P history | 78 | 68 | 10 | ego_filter |
| F forecast | 78 | 78 | 0 | 无 |
| PF history | 78 | 68 | 10 | ego_filter |
| PF forecast | 78 | 68 | 10 | ego_filter |

20 帧中 F-only 与 PF 的 forecast 身份集合全部相同，78 份实际 forecast 内容也逐值相同。变化的恰好是其中 10 份的 admission；对应 P history 是同一批 10 个样本/track，而非仅仅总数碰巧相同。

## 确定的执行路径

1. F-only 有当前 anchor 和 forecast，这些目标未与本车本地检测形成跨源配对，`association.status=unmatched`。
2. 当前位置靠近本车，但没有该目标的已购 history；[`_role`](../src/planning/structured_inputs.py#L355) 按保守规则返回 `unresolved`。靠近原点不会自动删掉目标。
3. P 提供 causal tracking-state motion proxy 后，相同目标的 11 个有效历史状态与本车合法定位历史比较，满足固定位置、尺寸、历史 RMSE 和朝向门槛，角色变为 `ego`。
4. [`_tensors`](../src/planning/structured_inputs.py#L495) 在分配张量位置前排除 `role=ego` 的实体，因而同时排除其 observation 和 forecast。`tensor_index=None`。
5. 这些 F-only/PF 对比中的 aliases、代表 anchor 和 `unmatched` 状态保持相同。不是实体关联身份变化，而是**同一实体的角色判断改变**。

角色函数诊断中，仅移除已有历史后重算 `_role`，这 10 例都回到 `unresolved`。这只是私有函数层面的离线因果定位，不伪造在线 receipt、修改合法请求或生成驾驶结果。

## 逐例结果

以下 track handle 均属于该帧 `no_fusion_cav1` 的局部身份，不是全局 ID；全部例子有 11 个共同历史步。位置单位为米，朝向单位为弧度。

| g | peer track | 当前距本车原点 | 历史 RMSE | 最大朝向差 | PF 使用实体槽 / 上限 |
|---:|---:|---:|---:|---:|---:|
| 2932 | 593 | 0.1257 | 0.1089 | 0.1213 | 17 / 64 |
| 3597 | 1 | 0.1442 | 0.1848 | 0.1038 | 14 / 64 |
| 3658 | 10 | 0.0377 | 0.4377 | 0.0679 | 15 / 64 |
| 3766 | 2 | 0.0772 | 0.0961 | 0.0353 | 17 / 64 |
| 4736 | 3 | 0.3089 | 0.4280 | 0.0583 | 13 / 64 |
| 4950 | 3 | 0.1959 | 0.2460 | 0.1050 | 17 / 64 |
| 5526 | 20 | 0.6275 | 0.4843 | 0.1111 | 21 / 64 |
| 5585 | 20 | 0.5098 | 0.3835 | 0.0516 | 27 / 64 |
| 7007 | 0 | 0.1635 | 0.2298 | 0.0803 | 20 / 64 |
| 7054 | 0 | 0.2482 | 0.2767 | 0.0343 | 11 / 64 |

历史 RMSE 门槛为 0.5 米，朝向为 π/6，当前位置为 1.5 米；尺寸条件也全部满足。没有触及 64 实体容量，未发生预测集合容量截断。这确认了实现按既定规则判为 ego；本轮没有读取独立身份真值，不能把几何/运动一致性提升为零误识别保证。

## “没有直接入模”不等于“完全丢弃”

- P 与 PF 中这 10 份 history 仍保留在 acquired ledger，且属于 admitted dependency：它们用于角色判定/过滤，属于间接使用，不是完全被丢掉。
- PF 中这 10 份 forecast 保留已购记录和费用，但不在最终 admitted dependency 中，实际进入 dropped 集合。
- F-only 中同样的 10 份 forecast 是直接张量输入。

因此现有“60 个协作任务都有远端字段入模”的整体验收仍成立，但它没有保证 F-only 与 PF 的最终 forecast 入模集合相同。这是先前整体验收结论的限制，需要随结果一起披露。

## 一处确认的工程问题

[`build_structured_plan_input`](../src/planning/structured_inputs.py#L621) 把所有 `dropped` 项的顶层 reason 写成 `structured_capacity`。本批 10 份 PF forecast 的 `field_groups.use` 正确为 `ego_filter`，顶层标签却暗示容量原因。

这是**原因标注过粗/误导**，不是本次字段消失的根因。建议后续直接从已有 `field_groups` 推导并输出准确的排除原因，区分 ego filter、实体容量和预测集合容量。旧归档保留原记录；可以在派生审计中给出准确分类，避免为了修改报告标签破坏历史 prepared-input 的严格重放。

## 对实验和下一步的影响

文档要求先查清这 10 组是必要的，这项离线定位现已完成。若以后观察到 F 优于 PF，不能直接解释为 P 信息有害：P 还改变了角色判定和 F 的可用集合。相反，也不能先断言过滤一定正确、因此 PF 必然应更好。

建议接下来只补两项小工作，再继续既定的数据准备：

1. 准确的派生排除原因和一条 F-only → 加入 P history → 同一实体变为 ego 的回归检查，明确 history 的间接使用与 forecast 的排除。
2. 在实验规格中预声明角色过滤对照：固定同一批合法取得的 evidence、同一 driver 和推理次数，对比现行过滤与允许这些实体保留的诊断模式，以后才测角色过滤对轨迹误差的贡献。保留模式需要独立、明确的配置，不能临时改已发布模型规格。它属于解释 receiver 影响的实验对照，不是新 ToolV2X 方法机制。

当前不建议立刻放宽 ego 阈值、把 PF 已购历史偷偷给 F-only 使用、改造 P/F 协议，或为这项审计重建缓存/增加训练。合法信息增加引起身份判定变化本身并不违反因果契约；是否值得保留当前硬过滤，应由明确的角色语义和后续对照结果决定。

本轮仅新增本报告、紧凑审计证据及输出目录中的离线检查脚本。没有提交或推送 GitHub。

证据：[逐字段汇总与 10 例明细](evidence/admission_audit_2026_09_15/summary.json)。完整诊断位于 `outputs/admission_audit_2026_09_15_v1/`。
