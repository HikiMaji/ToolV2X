# 9-13-3 全分支独立只读终审

日期：2026-09-14（UTC+8）。工作树：`/root/autodl-tmp/ToolV2X/.worktrees/t9-supervision-admission-review`。

## 最终裁定

**规格符合性 PASS；正确性/最小性 PASS；没有剩余必须修复项。** 审查最初发现一个 P2 导出覆盖问题；主任务已做最小修复，本 reviewer 已独立验证修复后的全部 482 行。

依据为 `docs/9-13-3.md`、批准计划、完整源代码/测试/文档 diff，以及当前原始 T9 tasks/provider_records 和本批精选证据。A1/A2 已有 `docs/evidence/review_9_13_3/a_independent_review.md` 的独立专项 PASS，本审查复核共享入口与 B 集成，没有重复执行完整测试套件。审查只进行了源码/文档阅读、标准库 AST 和原始 JSON/CSV 直接关联检查；没有运行 tokenizer、模型、训练或采集，没有修改仓库文件或执行 Git 操作。

## 已修复的唯一发现

**P2，`scripts/audit_t9_admission.py:401-414`：无 remote unit 的阶段漏导出 dropped fields。**

触发：六个 Ego/initial plan 都没有 remote unit，但原 admission report 仍有被本车预算裁掉的字段。原 dropped export 循环在 `if units` 内，只产生 278 行；原始保存产物有 482 行，漏掉 204 行：g5526 三个初始阶段各 36，g7007 三个各 32。报告原先声称导出了每个保存 stage 的全部 dropped ref，因此确有输出与声明不一致。64 条 remote-unit 行和最终 admission 结论不受影响。

最小修复已经完成：只把既有 export 循环移到每个 plan 的作用域。主任务生成新的 v3 输出，保留旧 T9 以及先前 audit v1/v2。本 reviewer 用标准库逐行直接对照原 tasks，确认修复后的 482 行在 task、plan index/stage、frame、arm、完整 ref、origin、reason 和 context scope 上全部相等，顺序也相等；覆盖 12/12 stage。修改后的脚本 AST 解析通过。该项关闭。

## A1/A2 与全分支集成

- `bundle_data.py:253-259` 在已验证原 selected index/task/state identity 后复核拟训练 recording 的原始 role；`query_value.py:587-590,649-655` 将训练组传入 measured loader，并要求 public fitter 使用可恢复原始归档的 v2 路径。参数准备完整结束后才调用 `_fit`，故 role 拒绝先于 normalization、checkpoint 和 optimizer。没有改录制划分。
- `query_data.py:443-453` 保存独立离线 label sidecar；`query_value.py:255-274,292-301,331-345` 重新读取 source prefix、实际 chosen continuation/terminal、标签与 utility，复用原 `_target()` 比较全部公共监督字段。实际质量和代数联动污染不能只靠重写 loss/target 绕过；失败终态、STOP=0 和原增量费用定义保留。
- 公开 bundle fitter 不再接受 dict/v1 synthetic 数据、旧 alternating targets 缺 label sidecar 时 fail closed，均是明确披露且与需求一致的入口收紧。合成 CPU 夹具仍可走私有函数验证优化器契约；不构成真实训练。
- B 仅新增诊断与证据，不修改 receiver、provider 排序、GoT/MTR 或在线特征。没有新依赖、数据库、认证机制或模型。当前工作量来自逐字段可审计输出，没有需要进一步抽象或改写的最小性问题。

## B 独立关联核验

1. **请求阶段和服务事件。** `audit_t9_admission.py:54-107` 区分 `request_stage` 与 `first_visible_plan_stage`，绑定真实 `request_started` 及外层 request id。直接读取原始事件确认 alternating P 为 0→1、F 为 1→2；one-shot 两个 primitive 均由同一个 bundle stage 0 请求产生，在 stage 1 首次可见。对应 `response_received`、`receiver_completed` 事件都存在且 request/stage 一致。原 provider bundle 的完整 outer request 等于 episode request，内部 primitive 对象等于 provider primitive 记录。
2. **64 行完整映射。** 独立将所有实际 remote-unit 行关联到原 ledger 的完整 primary ref、context version/scope、anchor 距离、origin、receipt 的排名列表及位置。receiver-derived 行的完整 parent chain 也逐项吻合，来源于已购 P 字段。此冻结数据中每个 unit 有一个 primary ref，每行来源为一个 receipt；没有同目标多 context 被误合并或排名来源混淆。
3. **Admitted Z 和位置。** 对每个 admitted row 直接对照保存 admission group 的 primary/anchor refs、object index、prompt section、field positions；结果全部相等。保存原始 provider 的 request/response/cost 与 ledger receipt 也相等。
4. **真实 tokenizer 重放边界。** 源码 `audit_t9_admission.py:249-331` 只构造本地 tokenizer，调用既有 receiver/prompt codec；每个保存 plan 比较完整 local/remote evidence、receiver spec、admission report、evidence selection 和完整 q9 prompt，并核对原 prompt token 总数。另比较诊断 wrapper 的已选 unit 与保存 refs、最终 token 数。原 receiver 的 merge/order/token 规则与 wrapper 逐段一致。记录的 reproducibility 中六任务、12 stage 的全部相等断言 PASS；本 reviewer 检查这些记录及实现，没有重复启动 tokenizer。
5. **同预算 simulation。** target round-robin 保持合法候选和每目标内部顺序，不永久删除 context-distinct 字段，逐次渲染 prompt 并按原 context/generation budget 贪心尝试。输入不含 GT、offline labels、未知 peer 信息或 alternating 独有中间方案。它是 admission counterfactual，未生成新驾驶结果，也没有声称是最优覆盖算法。

保存输入支持报告的限定结论：g5526 两臂最终只有 target20 的两个 context unit，input tokens 为 3573；g7007 对应 target0、3572。距离排序与同目标多 field/context 占位使不同 provider ranking 在这两帧塌成相同 Z。共同 round-robin 模拟每个已购证据阶段从一个目标增加到两个，不能据此推断驾驶质量收益、普遍瓶颈或模式等价。

## 验证与完成边界

本 reviewer 的新增检查：八个改动 Python 文件标准库 AST PASS；64 条原始 row-to-receipt/context/rank/event/position 关联 PASS；修复后 482 条 dropped export 与原始全部 12 stage 精确相等 PASS。没有重复模型/轻量/CPU 套件。

最终完整轻量日志 `/tmp/toolv2x-9-13-3-final-light.log` 已完成：300/300 PASS，199.664 秒，末尾 OK；本 reviewer 已直接读取完成输出。CPU 合成日志为 10/10 PASS，相关 A 生产源代码此后未变。最终 B export 修复另外由原始 482 行直接比对和既有 B 四项契约检查覆盖。完整模型集成套件没有在本批执行。

最终工程审查通过不等于方法研究效果通过。当前交付停在监督入口修复、冻结产物复算及同预算 admission 模拟；不授权新训练、GoT/MTR 调用、扩帧/扩分支、receiver v2 实现或旧 T9 结果修改。
