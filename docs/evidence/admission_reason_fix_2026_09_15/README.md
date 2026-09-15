# 排除原因修正证据

[实现说明](../../admission_reason_fix_2026_09_15.md)。

- `summary.json`：80 个旧任务、160 个阶段的新旧审计比较及准确排除分类。
- `red.log` / `targeted.log` / `light.log`：修改前失败、定向通过、完整轻量回归记录。
- `replay.py` / `replay.log`：仅离线重审旧归档，不加载模型，不改变原文件。脚本副本依赖原输出目录结构，不在本 evidence 目录直接执行。

完整新派生审计在 `outputs/admission_reason_fix_2026_09_15_v1/audits/`；旧任务及旧审计未覆盖。
