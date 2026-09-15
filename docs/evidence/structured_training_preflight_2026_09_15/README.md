# 真实数据小批训练预检证据

对应 [报告](../../structured_training_preflight_2026_09_15.md)。实际输入来自此前 20 帧、80 个数值 v2 任务；本批只有有界离线监督和恢复核验。

`summary.json` 与 `verify_count_corrected.log` 为最终独立核验；首次 `verify.log` 中特征文件数因变量复用误记为字段字典长度6，实际文件逐个比较已执行，修正计数后重新完整复核。`coverage.json` 保存按物理录制隔离后的阶段/条件权重；完整参数见 `training_config.json`、`execution_spec.json`、`plan.json`。`*_trace.jsonl` 记录实际 loss、更新及梯度诊断，不是在线方法效果评价。

`remote_gradients.log` 是第一次诊断脚本错误分类 F anchor 的失败日志，修正结果单列于 `remote_gradients_corrected.log`，两份脚本均保留。没有以修正覆盖旧失败，也没有修改 driver。

原始目录：`outputs/structured_training_preflight_2026_09_15_v1/`。这些有界脚本依据该原始目录定位已有归档，不应从 `docs/evidence` 直接运行。权重、完整任务、特征和重复快照保留在本机，未放入本证据目录。
