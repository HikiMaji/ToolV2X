# 9-15-1 审查核验与下一工作包

日期：2026-09-15。依据：用户提供的 [9-15-1.md](9-15-1.md)、当前 main 源码与已保存验证记录。本轮完成代码阅读和三个资源无关合成复核；未修改实现/测试，未运行真实模型、训练、真实数据采集或新方法实验，未提交/推送本轮文档。

## 判断

五项意见与当前代码基本一致，但性质不同：第一项是已复现的成本计量缺口；第二项需要补充观测统计；第三项界定现有离线训练的适用范围；第四项要求确定训练目标与验证监控；第五项要求冻结公平对照配置。

接受“先验收并适配共享数值驾驶器，再比较反馈机制”的推进顺序。当前不需要扩充 Method、增大容量或增加预测网络。既有 322 项轻量/416 项完整回归证明当时测试范围通过，没有覆盖下面的空 control 计时缺口，也不代表新数值驾驶器已有任务能力。这两组完整测试本轮没有重跑。

## 逐项核验

### 1. 空 control 漏记决策时间：成立，应最先修复

- [method_episode.py](../src/planning/method_episode.py) 第 352–396 行：无论 control 是否为空，都会构建决策状态、预检请求并执行可调用的 policy；三个成功/异常记录位置均受 `if control_spec` 限制。
- [framework.py](../src/evaluation/framework.py) 第 176–182 行：只有显式 control 才建立预期决策计时集合，空集合被求和为 0。
- [method_run_spec.py](../src/planning/method_run_spec.py) 的 `freeze_method_run_spec` 经 `normalize_control` 允许 None，因此不是只有非法输入才能触发。
- 限定范围：`query_data.collection_spec` 已要求显式交替 control，`bundle_collection_spec` 已要求 one-shot；不能据此声称所有现有分支采集都漏记。

独立复核使用既有合成 driver/predictor，并替换执行器时钟；policy 仅使模拟时钟增加 5 秒后 STOP，没有真实等待或模型执行：

| 配置 | policy 实际调用 | control 事件数 | control_seconds | total_compute_seconds | cost_complete |
|---|---:|---:|---:|---:|---|
| None | 是 | 0 | 0 | 0.002 | true |
| feedback | 是 | 1 | 5.002 | 5.004 | true |

建议在共同执行器修复，覆盖空/显式 control、STOP、强制停止、策略异常及归档中断。新计量契约需要显式标记；历史归档缺失计时只能认定未知，不能补写为真实 0。保留已知阶段小计。`total_compute_seconds` 继续表示列明的测量阶段和，`end_to_end_seconds` 保持未知，不能声称实车端到端时延。

### 2. 直接数值输入与依赖闭包：成立，优先复用现有字段

[structured_inputs.py](../src/planning/structured_inputs.py) 的 `_entities`、`_tensors`、`_admission_groups` 和 `build_structured_plan_input` 已实现：按距离/身份稳定排序、实体容量与 forecast-set 容量、直接张量位置，以及完整来源依赖。

复核建立 local x=10 与 peer x=30 的两个不同实体，由真实服务代码返回合成 P receipt；仅将测试容量设为 1/2：

| max_entities | 远端已购字段 | 远端字段进入依赖闭包 | 远端直接 primary 字段 | 远端 observation group |
|---:|---:|---:|---:|---|
| 1 | 2 | 2 | 0 | entity_capacity，无 tensor_locations |
| 2 | 2 | 2 | 1 | tensor，定位到 observations 的远端槽 |

这里两个已购字段是 anchor/history，而直接 primary 字段是 history；字段身份数量本来就不应强求相等。这个例子仅证明统计区别，不能证明真实默认 64/4 已经触顶。

建议以每帧、每阶段为单位，从已有 receipt、ledger、field_groups、tensor_locations 与实际 mask 导出：新增已购字段、直接 observation/forecast 来源与有效覆盖、容量/ego 过滤、关联及排序依赖、prior 父依赖、实际输入变化。直接和间接作用可以重叠，不宜强行相加为互斥分区；“仅间接”应扣除直接部分。保留完整字段身份以识别引用复用和不同 context。

这应是派生审计输出，不新增第二套 receipt/ledger，不改变 `admitted_field_refs` 的闭包含义，也不先增加容量。

### 3. 当前 prior、固定归档证据：成立，不是已发现的泄漏

[train_structured_driver.py](../src/planning/train_structured_driver.py) 第 165–213 行：逐阶段读取保存的 prepared，清空 collator 中的归档 prior，再使用当前模型输出并 detach；目标只传入轨迹损失。exact-repeat 沿用已有 prior 槽，同证据 refinement 使用实际前次输出。

同一循环没有重新调用 provider。因此它训练的是“给定合法证据的预测/修订”，不能代替训练完成后由当前模型驱动的真实检索与交互评价。复核支持该调用链的 target-as-prior 隔离，不应扩大成对所有数据来源的无泄漏认证。

首批数据组织还需要明确一个实际依赖：当前 `run_task_episode` 先校验初始轨迹，再允许请求。未训练输出若不合法，P/F 条件可能根本采不到。初始化入口解决可加载性，不保证证据覆盖。阶段 B 必须统计这类失败并保留分母，不能通过挑选可运行或 F 有收益的帧隐去问题。

如初始化阻断覆盖，可在驾驶基础数据采集中使用预声明、仅依赖本车当前运动的合法请求轨迹，仍经真实 P/F 和 receipt 获得证据；必须明确标为采集条件，不能伪造为共享驾驶器输出，也不能把它当作主方法交互。当前 exporter 以真实 numeric episode 为输入，这种独立采集格式尚未实现，不能假定已有或直接绕过校验。阶段 A 应先确定采集方案，再决定是否需要这项最小接入。

### 4. 损失权重与验证：成立，正式训练前确定

导出器每阶段一行；`training_loss` 在该行上监督所有前缀与最终额外修订；`fit` 对有效行的 loss 求均值。以三阶段、每行等权、refinement_depth=0 为例，独立分数运算得到：

```text
各行：(L0), (L0 + L1)/2, (L0 + L1 + L2)/3
行均值：11/18 L0 + 5/18 L1 + 2/18 L2
累计系数比：11 : 5 : 2
```

这不是实际随机 minibatch 优化贡献的精确比例，更不是训练结果。缺失标签、不同 episode 长度、batch 大小和 refinement 会改变实际权重/更新过程。

建议首个正式配置将“监督哪些输出”和“怎样采样/加权”明确分开。一个较直观的选择是：每个阶段行只监督该阶段及其指定 refinement，前缀仍用于产生 detach 的合法 prior；按预声明的帧/证据条件权重训练。这样不会因导出更多后续行而反复增加初始阶段的监督权重。该选择需要新训练目标版本；现有前缀均值模式保留，不静默更改旧配置或恢复语义。本轮未实施或冻结这个选择。

`fit` 第 485–499 行仅在循环退出后算一次验证损失。建议复用现有验证逻辑，每个预声明验证间隔输出分阶段、分证据条件的轨迹误差与有效率；验证不得改变优化器、训练 RNG 或恢复位置。保留周期完整检查点，另外记录验证选模结果。训练 loss 仅作监控；选模应使用预声明的独立验证指标、失败处理与固定平局规则，所有证据条件共享同一个选中驾驶器。

### 5. 单轮总预算与能力范围：成立，配置可复用

[method_controls.py](../src/planning/method_controls.py) 的 `control_spec` 默认预算为 per_rpc；`episode_bundle` 第 165–185 行已有 aggregate 配置。用相同合成公开状态调用真实请求构造器，普通 response cap=30000、episode 双向预算=100000、wrapper reserve=2048：

| 模式 | 外层 response cap | 两个原语各自 response cap | 实际请求 bytes |
|---|---:|---|---:|
| per_rpc | 30000 | 13976 / 13976 | 2996 |
| episode_aggregate | 62048 | 30000 / 30000 | 3006 |

这是公开请求构造的结果，没有执行真实模型。aggregate 允许两个原语得到各一次普通容量，但本身不保证总成本相等：仍须固定相同 episode 双向总预算，并计入外层 wrapper、候选、summary、引用和实际服务/驾驶计算。不能只改 mode 名称就宣称公平。

候选明确为 initial/slower/constant_motion 等有限、预声明类别。后续“强”对照还需要相应条件委托策略的适配与公平评价，不能将当前诊断 continuation 当作已经训练的强对照，更不声称这是所有单轮程序的理论上界。

## 下一批：先完成阶段 A

这是建议实施范围，尚未执行。

| 顺序 | 最小工作与涉及文件 | 必须验证 | 不允许据此声称 |
|---|---|---|---|
| A1 | `method_episode.py`、`evaluation/framework.py`；必要时在已有运行规格绑定中记录计量版本 | 空/显式 control 计算均收费；STOP/异常/缺失日志/前缀复用可解释；旧归档不伪造已知成本 | 实车端到端时延、质量—成本优势 |
| A2 | 优先在已有评价/审计入口消费 `structured_inputs.py` 的现有 groups，不修改 provider/ledger 语义 | 已购但因容量未直接入模；远端独有；同字段引用；prior-only 依赖；所有张量位置和 mask 可复算 | 默认容量不足、P/F 已被模型有效利用 |
| A3 | `train_structured_driver.py` 及对应测试；版本化目标、验证计划和选择记录 | 手算 loss/梯度权重；无 GT prior；验证不更新训练状态；恢复连续性；分组指标与选模可复算 | 驾驶器已经训练好、序贯反馈有效 |
| A4 | 新批次运行配置和 manifest；复用 `StructuredDriverSpec`、`ExecutionSpec` 与原录制组角色 | Ego/P-state/F/PF 共用 observations_only 规格；明确初始采集条件、预算、种子、指标、失败分母和停止条件 | 配置冻结等于真实集成或训练完成 |

新增参数进入完整版本配置和运行记录，不散落到协议常量；`p_processing` 属于 driver spec，P-local 独立作为同上下文等价/计算委托控制，不能无记录混入主训练。

A1/A2 是计量工程；A3 是共享驾驶器训练基础；A4 是实验控制。均不包装为方法创新。现在不新增 RSU/I、联合 RL、peer 排序器或另一套预测骨干。

## 阶段 B–D 的推进与验收边界

1. **B：真实集成验收和共享驾驶器适配。** 在预声明训练/验证录制组上确定性抽样检查真实历史、关联/ego 识别、P/F 直接覆盖、初始失败与目标设备的保存恢复，再按已冻结配置训练。集成验收与质量评价分开；不按收益挑帧，不将两帧误差表作为完成门槛。
2. **B 的模型诊断。** 共用权重检查本地参照、移除 P/F、隔离的错配证据、prior 影响。当前 Ego 包含本地检测特征、运动/历史及本地 MTR 预测；它不等于仅运动状态参照。保留带远端信息的 prior 再删除直接 P/F，只能衡量新增直接证据的边际作用；若要衡量远端信息总体依赖，还应在合法前缀上重新生成无远端 prior。错配内容只能走清楚标识的诊断输入，不能伪装 provider receipt。
3. **C：冻结适配后的同一 D，重新执行真实交互。** 对照首返回、实际修订、去重后动作/新字段、直接和间接作用、最终质量/成本。完整报告初始 STOP、首返回后 STOP、两次调用及失败比例。没有要求每次修订都改变排序，也不强制用满调用。
4. **D：再拟合和比较查询策略。** 核心是同首轮真实前缀的 feedback vs frozen-feedback；固定证据下的有意义 refinement；明确能力范围和同总预算的 conditional one-shot。所有主机制对照共用同一驾驶器权重、adapter、输入和指标；不能用新数值路径胜过旧 GoT 来证明反馈机制。

ToolV2X 主线保持：本车任务参数调用远端 P/F，真实返回改变本车方案，修订再影响下一请求或 STOP。共享数值驾驶器是让这条机制能接受有效检验的共同基础。下一步的重点是把这项基础验收和适配好，不继续扩充方法描述。
