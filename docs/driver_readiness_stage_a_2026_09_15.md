# 共享数值驾驶器 Stage A 实施记录

日期：2026-09-15。四项实现、逐项独立审查、修复复审、整批独立审查及完整回归已完成，已同步本地main，未推送GitHub。本批依据[9-15-1核验与批准范围](review_9_15_1_response.md)，完成计量、训练基础和准备规格，没有真实数据训练、真实模型生成、provider推理、新采集或方法效果实验。

## 实际改动与接口

| 工作 | 文件 | 已实现行为 |
|---|---|---|
| A1 决策成本 | `src/planning/method_episode.py`、`src/evaluation/framework.py` | 全部实际决策/预检计时，空control和异常也记录；历史缺失费用为未知 |
| A2 证据审计 | `src/evaluation/structured.py`、共同评价入口 | acquired、receiver-derived、直接primary张量、关联/选择和prior依赖分开统计 |
| A3 训练与验证 | `src/planning/train_structured_driver.py`、`structured_validation.py` | 保留v1；新增v2阶段目标、固定帧条件权重、周期验证、严格选模和恢复 |
| A4 准备包 | `configs/structured_driver_readiness_v1/`、`scripts/prepare_driver_readiness.py`、既有diagnostic/CLI | 冻结因果元数据和完整配置，补F/PF固定选择器，公开可独立执行的校验入口 |

对应测试为`test_decision_accounting.py`、`test_structured_admission.py`、`test_structured_training.py`、`test_driver_readiness_preparation.py`及受影响的method/evaluation/budget测试；`scripts/check_review.py`注册了新增轻量检查。README、STATUS和审查入口同步本批完成边界。完整提交文件清单可从本批审查记录查看。

A1新增`compute_accounting_version=toolv2x_compute_accounting_v1`。真正开始决策时先留下未完成事件，成功后才标完整；policy看到的仍是决策前已花费用，不包含当前未结束事件。策略/分发/归档中断保留已知小计，缺失部分不补零。旧明确v2计时继续可读，损坏版本保留无效行。费用只包括已列明的测量阶段，不包含全部网络、执行器管理和归档I/O，`end_to_end_seconds`仍未知。

A2的`audit_structured_episode(episode)`先执行既有numeric episode校验，再导出逐方案审计；评价行新增`structured_audit`，旧GoT路径为None。`known_acquired_remote_refs`、`new_acquired_remote_refs`与`previously_acquired_remote_refs`严格对应provider实际返回；`known/new/previously_receiver_derived_remote_refs`单列本地派生。中性的`known_remote_refs`保留二者并集，不能把它当购买量。

直接使用由实际`use=tensor`、`tensor_locations`与有效mask确定，保留完整字段身份、来源、context、模式和时间覆盖。直接primary、依赖闭包、关联/筛选及prior可以重叠；indirect-only明确扣除直接primary。首次输入/输出变化没有比较基准，记为None；引用复用不新增购得字段。没有新增第二套receipt/ledger，也没有提高64实体/4预测集合容量。

A3完整`toolv2x_structured_training_v2`配置新增`objective`、`row_weighting`、`validation`与`selection`。每行只监督所选阶段和指定同证据修订，前缀仍由当前模型生成并detach，不输入归档数值prior或GT prior。v1前缀均值目标、旧配置和恢复保留。

`frame_condition_weights`从真实primitive receipt集合确定Ego/P/F/PF，按帧、该帧实际存在的条件、组内重复行分配固定权重，在有标签训练行上一次归一化，绝不逐minibatch重新归一化。无标签行不参加v2优化批；缺失条件和失败采集仍应报告。固定系数不意味着随机minibatch/AdamW/末尾小批与全批优化等价。

验证按初始、预定processed-batch周期和计划训练终点执行。中断保存不额外插入验证；验证恢复模型模式/RNG，不更新优化器。所选阶段与额外refinement指标分开；无效轨迹保留标签分母，缺完整六点标签不能进入完整前缀指标。选模依次比较带权无效率、有效方案的`got_prefix_L2_avg`、最早批次；没有合格有限指标就没有best。`fit`返回最后检查点，`selection.json`单独指向所选检查点；完整恢复绑定数据、配置、模型、优化器、RNG、进度和验证记录。离线前缀加refinement可能超过三次forward，不冒充在线三次驾驶尝试的结果。

## 准备数据与配置

唯一打包因果清单共3095帧：2987 train、108 validation，来自8/2个录制组，physical_split均为train。保留已有40个异常训练窗口的排除。该validation是既有开发验证，不能改称独立测试。每组固定首末两个已接纳帧形成20帧验收清单，16train/4validation；没有按标签、收益或模型输出选帧。

准备阶段只读已有JSON，保留sample/scene/g/local_frame、研究角色、当前ego运动与本车读取路径；未打开这些路径指向的数组、标签或权重。路径用于定位/审计，不作为模型或场景语义身份。本地已将3095行与原索引白名单副本精确比较；portable检查只确认结构、计数和选择规则，不能独立认证外部索引来源。

完整配置保留默认observations_only的256hidden/4heads/2layers/64entities/4forecastsets，种子7、AdamW 1e-4/0.01、batch2、主适配20轮、refinement1、每25批保存、每100批及初始/终点验证。CUDA只是后续设备设置，本批未验证CUDA训练一致性或真实速度/容量。P-local继续是独立等信息/计算委托对照。

固定Ego/P-state/F/PF对应`stop/p_current/f_current/p_current_f_current`；原`p_current_f_change`保持原义。单轮委托用真实注册ID`diagnostic_conditional_v1`及episode_aggregate，候选initial/slower/constant_motion，最多3个，slowdown0.5，wrapper2048B、summary最多8字段/1536B，同一最多3次driver额度。在公开合成状态调用真实构造器得到1813 UTF-8 request bytes，加133120B外层返回上限，在196608B总预算内；这不是实际网络或模型耗时。

准备包生命周期为`prepared_not_executed`，实际runtime load identity为空。真实标签覆盖、初始轨迹合法性、P/F覆盖、设备/存储容量和训练能力均未知。公开校验命令与完整B–D验收条件见[准备规格](structured_driver_readiness_2026_09_15.md)。

## 三项实施决定及错误代价

1. v2只监督所选阶段及refinement，按帧/条件冻结权重；v1保留。若这种目标不适合实际分布，必须另存配置/版本并重新训练，不能把旧结果改称新目标结果。
2. 若初始模型合法性阻断采集，条件启动使用20个验收帧固定Ego任务的真实stage0 prepared归档，最多3轮，再冻结并重新采集四条件。不伪造输出或另建采集格式。若仍失败，停止该真实阶段并诊断初始化/数据，不自动延长。标签尚未读取，每轮验证间隔须在未来按实际有标签训练行数另存完整配置。本批未执行启动训练。
3. 原诊断入口无法表达全部四条件，因此只增加F-current-only与固定P-current→F-current两个ID，共用既有执行器/CLI白名单。若这些采集条件不适合，应在采集前更换诊断配置/标识；不改变P/F协议，不引入学习排序器或新方法机制。

启动检查点用于重新采集；完整适配目前按种子从头初始化。没有warm-start API，不把更换数据/config当作精确resume。

## 审查与验证证据

| 范围 | 实际验证与修复 |
|---|---|
| 起点 | 既有method/evaluation 34项通过 |
| A1 | 新计量及受影响query/bundle链路检查；审查修复旧预决策失败费用和损坏版本汇总，修复后25项通过、复审通过 |
| A2 | 初版41项轻量通过；审查抓到acquired/derived混合，修复后23项通过、复审通过 |
| A3 | 38项training/episode通过；恢复测试抓到非周期重验和误复制未来检查点，最终2项恢复定点通过，独立审查通过 |
| A4 | 36项准备/method/budget通过；审查修复重复身份和私有CLI依赖，最终8项准备测试通过、复审通过 |
| 整批 | 独立spec/quality审查通过；轻量346/346（218.261秒）、模型环境448/448（448.310秒），exit均为0 |

一次早期较大定向命令丢失终端回传，不记完整PASS；已确认的失败、修复和终态日志分别保留。上一批322/416不能作为本批通过证据。详细逐项报告、失败日志、复审与决策账本见[证据目录](evidence/driver_readiness_stage_a_2026_09_15/README.md)。

## 完成与后续边界

本批属于计量工程、共享驾驶器训练基础和实验控制，均不包装为论文创新。P仍不执行MTR，P历史继续是causal tracking-state motion proxy；F在完整合法peer context上预测后再排序。旧v1 wire/golden、GoT路径和已有归档保留。

后续B先做预声明真实集成，再适配共同驾驶器；C冻结适配模型后重新执行真实检索与修订；D再拟合和比较查询策略。移除直接P/F而保留含远端信息的prior只测直接边际效应，总体远端依赖需重新产生无远端prior。错配内容只能是标识清楚的诊断输入，不能伪装合法receipt。本批没有方法优越性、论文收益或闭环安全结论，不自动进入B，也不自动推送GitHub。

本地main合并后与受测提交树一致，四份原有未提交文档内容逐字节保留。之后只追加本合并记录等文档，没有重复运行未变化的测试。见[本地集成记录](evidence/driver_readiness_stage_a_2026_09_15/local_integration.json)。
