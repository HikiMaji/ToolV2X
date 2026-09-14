# 共同 receiver 轮询与两帧实际联调

用户已批准上一轮提出的两步，并明确要求一起完成。当前改动范围明确，沿用既有接口；不再重复请求实施许可。

## 固定范围

- 新增显式 `toolv2x_receiver_v2`：先按旧规则确定目标/同目标字段顺序，再每目标轮流尝试一个 unit，超限跳过，下一轮允许同目标更多字段。保留原 v1 默认与精确行为。
- 同一规则供所有方法使用，不输入方法名、GT、未购字段或 alternating 独有的中间方案；不删 context 不同的预测，不改变本车块、编码精度、context/generation budget。
- 两个原样本 g5526/g7007，冻结 GoT epoch01 与 CMP MTR epoch09。Ego / 诊断 alternating / 诊断 one-shot 在旧、新 receiver 下各运行一次，共 12 个任务，最多 24 次 GoT。重新执行合法服务请求；不把旧第二次响应接到新修订方案上。
- 这是两帧链路/receiver 对照，不是正式 value-policy 或 strong one-shot 效果实验。保持原诊断策略及其既有 wire cap 差别，披露所有成本；不训练、不扩大帧/枚举分支，不重建缓存，不改旧运行归档，不自动推送。

## 执行顺序与责任

| 工作 | 文件与接口 | 验证与依赖 |
| --- | --- | --- |
| R1 root | `planning/context.py` 共同轮询 helper 与可选 receiver version；`v2vgot.py` 接受已声明 v2；审计脚本复用 helper；对应契约测试 | RED→GREEN：固定预算增加目标覆盖、context/来源区分、超限继续、local 不变、v1 精确行为、真实 driver 合约接受 v2；不据此声称驾驶收益 |
| R2 agent 准备，root 执行 | 复用旧 `run_smoke.py` 和已有 runtime，准备新输出目录的冻结配置/代码快照/12 任务运行封装，随后离线评价与对照报告 | 先独立准备不运行模型；等 R1 检查完成后 root 启动。逐阶段校验新 τ1→新第二请求与真实 provider，旧/新 Z、轨迹、失败和完整调用成本；只使用两帧 |
| R3 root + 独立终审 | 完整轻量/适当模型环境回归，已有两帧 tokenizer 对照与运行后独立核验，当前状态与最终报告 | R1→R2 的 receiver 字段完整冻结，预算不变。本地 main 集成后停止 |

依赖核对：R1/R2 共享 receiver version 和预算，R2 只写新输出和自己的报告，不改 root 源码；实际推理等待 R1 GREEN。默认 v1 与新 opt-in v2 不冲突；通用轮询与冻结两帧模拟算法一致。小型合成契约测试不是实际 value-policy 训练。新两帧输入可能仍相同，不以“必须不同/改善”为通过条件。

## 执行完成

R1 已完成并通过独立规格/质量审阅。R2 有效运行 12/12 任务、24 次真实 GoT，完整轻量 302 项和模型环境 358 项回归通过；两帧链路、逐阶段轨迹、证据及完整成本由独立终审复核通过。首次运行的输入路径失败另存，未覆盖旧归档。详见 [实现记录](../../receiver_round_robin_implementation_2026_09_14.md) 与 [真实运行报告](../../receiver_round_robin_smoke_2026_09_14.md)。本批到此停止，未训练、扩帧或推送。
