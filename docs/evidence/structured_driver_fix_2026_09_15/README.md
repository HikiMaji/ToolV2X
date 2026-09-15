# 数值驾驶 v2 修复与 20 帧复验证据

主报告：[修复、真实结果与边界](../../structured_driver_fix_2026_09_15.md)。

- `summary.json`：最终全部 80 任务、实际请求/字段/模型执行、动态约束与结论边界。
- `acceptance_audit.json`：独立读取整批归档后的输入、provider、receipt 和入模审计。
- `execution_plan.json` / `executed_readiness.json`：本批执行上限与完整实际规格。原冻结配置保留，driver spec 显式切换 v2。
- `binding_verification.json`：实际检查点绑定/模型身份、源码逐字节复核、原配置未修改。
- `source_comparison.json`：本地原始索引全部 3,095 行的独立字段比较，区别于公开包的结构单测。
- `light_regression_final.log` / `model_regression_final.log` / `v2_tests_final.log`：最终通过记录。旧失败日志保留以说明测试修复。
- `sine_output_layout.json`：真实历史角度的重复向量计算差异；`legacy_validator_capture.json` 等记录未重现旧异常的有限尝试，不把未复现当根因证明。

同目录中的运行脚本是原始操作副本，用于审查本批具体如何运行，不是生产 CLI；它们依赖原输出目录的相对结构，不能在此 evidence 目录直接启动。全部任务、输入、实际 provider 记录、完整源码快照及未训练检查点位于：

`/root/autodl-tmp/ToolV2X/outputs/structured_driver_fix_2026_09_15_v1/`

没有真实数据训练、GoT 生成、全量采集或查询价值网络更新。共同驾驶基础的连通不等于协作收益或 ToolV2X 方法效果。
