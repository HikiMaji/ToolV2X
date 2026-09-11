# 首次 GitHub 审查

日期：2026-09-10。目标仓库：HikiMaji/ToolV2X。首次仓库用于审查实现、协议和训练准备；当前没有正式方法收益结论。

2026-09-11 本地补充：原 MTR 因果适配已完成，见 [实验结果](mtr_adaptation_results.md) 和 [Q8/Q9 质量诊断](planning_quality.md)。新增审查入口为 `src/prediction/{supervision,prepare_mtr,train_mtr}.py`、`src/evaluation/` 及 `tests/verify_mtr_*.py`。重点区分离线 GT 监督与在线因果输入、相同目标的原始/适配对照、验证选模与独立测试、完整/ROI 训练设置与真正的 P/F 方法收益；最新大体积产物和权重仍在本机。

最新补充：MTR 六组等预算训练及冻结选择见 [mtr_stability_results.md](mtr_stability_results.md)，驾驶初始化/解码真实接入见 [driving_decoder_results.md](driving_decoder_results.md)。`check_training.py` 与原 builder 的可训练 LoRA 路径已完成四组合一次真实更新、独立重载与生成，见 [训练入口结果](driving_training_readiness_results.md)；这些检查与正式适配必须分别核实。权重和完整缓存仍留在工作站，精选文件范围见 [本次更新说明](github_update_2026_09_11.md)。

## 审查入口

| 关注点 | 当前代码 |
| --- | --- |
| 当前/过去 Ego 特征、输入字段白名单、回答解析 | `src/planning/inputs.py` |
| 原 MTR 22 通道输入、缺失历史、完整上下文与预测输出 | `src/prediction/cmp_adapter.py`、`vendor/cmp_mtr/` |
| P/F 请求与响应、解码、P 本地重算、来源与模式保留 | `src/tools/vehicle.py` |
| 原 projector、上下文预算、原 7B 的 Q8→Q9 / direct | `src/planning/v2vgot.py`、`vendor/v2vgot_llava/llava/` |
| 固定四动作执行和访问/通信账本 | `src/planning/run_connection.py` |
| 离线标签、训练样本、实际生成的 Q8 父上下文 | `src/planning/adaptation_data.py` |
| 原监督前向、特征/提示损失屏蔽、一次真实参数更新检查 | `src/planning/check_adaptation.py`、`check_training.py` |
| 同信息编码对照 | `src/planning/compare_evidence.py` |

`src/probe`、`src/oracle` 和旧 CMP 四配置脚本保留历史实现。不要把这些代码运行过等同于当前主线已验证，也不要仅凭文件夹名称删除当前仍引用的校验函数。`vendor/cmp_mtr/reference/` 中原 dataset 只供参考测试；正式在线适配器不调用该 GT 驱动 dataset。

## 上传的运行证据

- `outputs/protocol_audit/`：录制级 split manifest、帧清单和早期协议核查。物理 train/test 名称不能代替研究 train/validation/test。
- `outputs/framework_connection_v3/`：一个验证决策帧的原模型 JSON 四动作接入；30 个邻车目标的 P 本地重算与 F 等价。
- `outputs/framework_compact_v1/`：同一验证帧扩充目标的紧凑表示运行；P 的原 Q9 输出失败仍保留。
- `outputs/evidence_comparison_v1/`：相同目标集合的真实紧凑格式生成，以及单独的完整文本 token 统计。
- `outputs/framework_compact_train_v1/`：一个真实训练帧的四动作记录。
- `outputs/adaptation_data_v1/`：3027/108 帧准备摘要、8 条已物化监督样本和历史独立复核报告；完整索引/标签仍在工作站。
- `outputs/adaptation_forward_v1/forward_check.json`：8 次原模型监督前向的历史记录，优化器步数为 0。
- `outputs/logs/`：明确挑选的历史复核脚本和测试日志。

这些文件保持原内容。它们不包含完整重放所需的点云特征、窗口归档或当时源码快照。`scripts/check_review.py` 在当前仓库重算可独立检查的事实；旧复核脚本中的绝对路径、源码快照一致性检查只适用于原工作站及当时版本。

## 可以交给审查者的任务

```text
请对整个 ToolV2X 仓库做只读代码审查。先读 README.md、docs/evidence_adaptation.md 和 docs/upstream_sources.md，再追踪当前规划/工具/离线监督路径以及实际调用的 vendor 源码。

优先检查：
1. 时间 t 因果性、查询前远端信息隔离、GT 标签与在线输入分离、录制级数据划分。
2. P/F 同信息对照、完整上下文与 ROI 顺序、坐标/时间、远端独有目标、多模态和实际通信成本。
3. q8_q9 模式的真实生成父回答、direct 模式无 Q8 输入且不能由缺失父回答自动切换、格式失败分母、上下文截断/舍入、监督损失掩码与原模型真实接入。

运行 python scripts/check_review.py；环境允许时再按 README 运行完整集成测试，并明确哪些未运行。
不要只复述报告或把设计文档视为已完成实现。每条问题给出具体触发条件、文件位置、可观察后果和最小验证办法；区分实现缺陷、未验证假设与明确后置工作。不要将可解析轨迹或有限 loss 视为方法有效。此任务先报告发现，不修改代码、不启动训练。
```

## 本次仓库准备带来的变化

只调整核心外部资源路径、原源码/配置的仓库内位置和审查测试入口；P/F 定义、预测网络参数、提示选择规则和训练目标未调整。V2V-GoT LLaVA 与原标签文件从本机逐文件复制，CMP 原配置和 intention points 也保持内容一致。原目录没有在本轮修改。

轻量环境仅安装 `requirements-review.txt`。完整集成仍使用原模型环境和本机资源。本次不重新执行 7B 生成、不更新参数、不把旧快照复核结果转记到新代码上。

## 本次准备的核验

修改前 58 项测试通过；加入资源路径和归档证据检查后，完整本机集成 62 项通过。从实际 Git 暂存内容导出的副本，在只安装 NumPy 1.24.2 / SciPy 1.10.1、未安装 PyTorch/Transformers 且外部资源路径不可用的隔离环境中，51 项轻量检查通过。

另外，用随仓库提供的 MTR 配置/辅助资源和本机原权重重算原验证帧的 21 个自车目标；目标、状态、预测、分数、局部 GMM 和 model-used 标记与原归档数组逐项完全一致。这是路径/打包改动回归检查，未评价预测质量。详细记录见 [github_preparation_validation.json](github_preparation_validation.json)。
