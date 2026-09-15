# 角色过滤审计

[核验回复](../../review_9_15_2_response.md)。`summary.json` 包含全部计数及逐例原因。

`audit.py` 是本地离线脚本副本，依赖原始输出目录相对位置，不应在此目录直接运行。原脚本和完整中间实体记录位于 `outputs/admission_audit_2026_09_15_v1/`。它只读取既有归档并重放 receiver 校验，没有加载模型或独立身份标签。
