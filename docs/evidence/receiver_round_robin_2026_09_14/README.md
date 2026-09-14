# Receiver round-robin evidence

当前保存准备证据、`_v1` 环境失败证据及 `_v2` 有效运行结果。第一次尝试完成模型加载，但 12 个任务均在输入构造前失败，GoT/provider 调用为 0；它不构成方法或 receiver 结果。正式有效运行使用新 `_v2` 目录，12/12 任务成功、24/24 GoT plan 合法，且没有覆盖失败尝试或 2026-09-13 归档。

本目录归档冻结 `launch.json`、`code_version.json`、终态 `progress.json`、模型身份、离线 `summary.json`、LF 行尾 CSV、`receiver_comparison.json`、`prior_archive_comparison.json` 和 label access 声明。两张图位于 `docs/figures/receiver_round_robin_2026_09_14/`。原始 tasks、wire、provider records、NPZ、完整 trace/provider JSON 与源码快照保留在大输出目录。

`verification/` 保存 RED/GREEN、302 项轻量与 358 项模型环境完整回归、原 v1 的 12 阶段 tokenizer 回放、5 个固定策略合成契约、独立原始记录复算以及两次独立审阅。初始缺资源的模型环境日志与修正后的完整日志分别保留。

报告图由 `verification/render_actual_run.py` 从同一真实任务轨迹及已保存的离线标签重绘，将图例移至图外，保持等空间比例及原指标。原始输出目录内的图、数据和评价未修改。
