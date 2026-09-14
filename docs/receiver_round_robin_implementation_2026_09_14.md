# 共同 receiver v2：可选目标轮询

日期：2026-09-14。用户批准共同 receiver 和原两帧实际调用链两步一起执行。本文件记录工程实现；实际模型结果见 [两帧报告](receiver_round_robin_smoke_2026_09_14.md)。

## 实现

`build_task_plan_input(..., limits=receiver_spec)` 新增接受 `version="toolv2x_receiver_v2"`。其余字段不变：`context_limit`、`generation_reserve`、`peer_reserve`、`numeric_decimal_places`。默认仍为 `toolv2x_receiver_v1`，旧运行不用迁移。

v1 按既有 `remote_units()` 顺序尝试；v2 先用相同顺序按 `(source, track_id)` 分组，再每组依次尝试一个 unit。组内顺序保持，超限仍跳过并继续；第一轮结束后允许同目标更多字段。不同 context 的 forecast 不合并为重复信息，不删除候选字段、不改已购/派生账本。

此选项由共同 receiver 实现，不接收方法名、任务分支、GT、未知 peer 信息或 alternating 独有的 τ1；本车装包、数值投影、prompt codec、generation reserve 和容量检查不变。`source_blocks_v2` 仍表示既有 evidence layout，与 receiver 的 v1/v2 策略版本分开。

运行示例沿用原两帧参数，仅替换版本：

```json
{
  "version": "toolv2x_receiver_v2",
  "context_limit": 4096,
  "generation_reserve": 256,
  "peer_reserve": 1536,
  "numeric_decimal_places": 2
}
```

## 修改范围

| 文件 | 内容 |
| --- | --- |
| `src/planning/context.py` | 迁入既有 `target_round_robin(units, target_key)`，在明确 v2 时用于共同排序 |
| `src/planning/v2vgot.py` | 原 `plan_prepared` 接受 v1/v2 receiver，其他真实 token/预算/feature/提示一致性检查保留 |
| `scripts/audit_t9_admission.py` | 引用相同轮询函数，避免模拟与实际执行维护两份算法 |
| `tests/test_evidence_ledger.py` | 两个新契约：同预算目标覆盖、后续轮次 context 保留；超限继续、Ego 不变、未知版本拒绝 |
| `tests/test_framework_episode.py` | 原 GoT adapter 接受明确 v2，同时继续拒绝未知版本；昂贵生成仅在该单元测试中替代 |

P/F 协议、MTR、ledger、policy、监督目标及五个旧执行策略没有改动。没有新依赖、网络或可学习模块。

## 已完成的定向验证

- 原 ledger/admission 25 项基线通过。
- RED：两个 receiver 正例因尚未支持 v2 失败；一个原 driver 正例因旧版本 gate 失败。随后 GREEN：27 项 ledger/admission 与 4 项原 tokenizer/driver 契约通过。
- 用旧真实归档的 12 个 plan stage，重新核对 v1 的完整 prompt、local/remote Z、admission refs/位置和 token 总数；全部与旧保存值一致。v2 的入模 refs、顺序、object index 与 token 总数和先前 round-robin counterfactual 逐项一致。
- 五个固定 Ego/P/F/PF/rule 的合成执行检查均经过同一 v2 receiver，实际请求序列符合原策略。它们验证接口共用，不是五策略真实模型效果实验。
- R1 独立审查规格与最小性均 PASS；完整轻量/模型环境回归和实际两帧结果统一记入本批模型报告及验证证据。

这项修改是让合法已购证据进入模型的工程基础，不单独包装成 ToolV2X 创新。此前固定旧返回的两帧模拟中，两臂最终仍选中同样字段；本批实际联调从新入模证据生成修订方案，再发送真实第二请求，没有把旧 F 返回拼接到新 τ1 上。实际 g5526 两臂最终 forecast context 不同，g7007 仍相同，详见两帧报告。

旧 v1 实验、代码快照和标签保持历史含义。新 receiver 的结果单独归档，今后若制作新 receiver 的策略监督，需要相应真实分支及完整绑定；本批不重建旧 targets、不训练 value policy。
