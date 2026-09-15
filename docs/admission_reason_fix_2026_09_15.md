# 9-15-2 排除原因修正

落实[逐例核验](review_9_15_2_response.md)提出的原因报告与回归修正。本批只修改派生审计和测试，保留原始 prepared、驾驶输入、角色识别、过滤规则、P/F 协议和模型版本。没有训练、新模型生成、重新采集或缓存重建。

## 新增审计字段

`audit_structured_episode` 的每个阶段输出新增：

- `exclusion_reason_version = toolv2x_structured_exclusion_reasons_v1`
- `non_direct_primary_fields`：每个被排除的 primary 字段/实体组对应一项，包含完整 `ref`、`entity_id`、`source_role`、`reason`、`admitted_dependency`、`dropped` 和 `legacy_reason`。

`reason` 从已经严格校验的 field group 与实体槽位推导，取值为：

| 值 | 含义 |
|---|---|
| `ego_filter` | 被判为 ego，整个实体未进入直接张量 |
| `entity_capacity` | 受实体容量限制，整个实体未进入直接张量 |
| `forecast_set_capacity` | 实体已有张量槽位，但此预测集合未进入直接张量 |

`admitted_dependency` 表示字段仍在 admitted 依赖集合中；`dropped` 表示不在该集合。二者不能由“未直接入模”推断。`legacy_reason` 原样保留旧顶层 dropped 标签；字段未被完全排除时为 null。

原始 prepared 中粗粒度的 `dropped.reason=structured_capacity` 不回写，也不作为新的分类依据。原因应读取上述版本化派生字段。旧审计文件若缺少新字段，代表未提供此分类，不能据此填零；可用原始 episode 离线重新审计。

标准 `evaluate_method_task` 已调用该审计 API，自动携带新分类。其余原有统计与字段保持原义，不混用字段条数、实体数和预测集合数。

## 回归覆盖

新增两项测试，使用原有合成契约夹具，不作为模型效果证据：

1. 相同实体从 F-only 的 unresolved 变为加入 P history 后的 ego；验证字段内容、身份与旧输入不被审计改写，并区分 P→F 和 F→P。
2. 分别触发实体容量与预测集合容量，验证 forecast 的准确原因，即使原始顶层标签都写着 structured_capacity。

调用顺序的区别被明确保留：P→F 中 forecast 从未直接使用，可进入 dropped；F→P 中 forecast 已影响实际上一方案，当前被 ego_filter 排除后，仍可以通过 prior 保留依赖，不能报告为完全丢弃。history 参与身份判断，同样应报告 admitted dependency。

生产修改仅在 `src/evaluation/structured.py`；测试位于 `tests/test_structured_admission.py`。未改变 receiver 的 prepared 构造与校验、checkpoint 接口或实际驾驶行为。

## 验证与结果记录

先运行失败测试确认原接口缺少准确分类，再实施修改。定向审计/评价测试 25 项通过；完整轻量回归 **349 项通过**（200.877 秒）。旧真实归档 **80 个任务、160 个阶段**复核通过：移除两个新增审计字段后，全部旧审计字段与原始记录完全相同，原任务文件未改变。最终记录见[证据目录](evidence/admission_reason_fix_2026_09_15/README.md)。

离线复核读取先前真实 20 帧、80 个任务，比较新旧每个阶段审计：删除新增的两个字段后，应与原始审计完全相同，同时验证原任务文件未变。该复核不加载 Torch/Transformers，不调用预测器或 driver。

本批未实现角色过滤消融、未启动全量数据准备/训练，未提交或推送 GitHub。后续可继续既定数据准备；这项原因报告修正无需重新采集旧任务。
