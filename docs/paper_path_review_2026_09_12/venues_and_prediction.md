# ToolV2X 两车 P/F 开环轨迹方法的论文路线与 CCF B/C 投稿定位

## 结论先行

当前仓库已经形成一个可运行的研究平台，但还不是一篇方法论文：它把冻结的 CMP MotionTransformer（MTR）接成 P 本地重算/F 远端预测能力，把原 V2V-GoT 7B 驾驶模型接成共享 LoRA 与 `mm_projector` 的自车轨迹解码器，准备了 Ego/P/F/PF/rule 五类输入和未来 3 s、6 点文本轨迹任务；全量因果索引为 2,987 个训练帧、108 个开发帧，物理录制按 8/2 组隔离。正式全量训练与生成评价尚未完成，查询策略仍是固定策略或手写 rule。因此，现阶段不能以“拼接两个已有模型”作为论文贡献，也不能把开发集或早期联调结果写成协作收益。仓库状态与证据边界见 [framework_baseline_results.md](../framework_baseline_results.md)、[mtr_adaptation_validation.json](../mtr_adaptation_validation.json) 和 [upstream_sources.md](../upstream_sources.md)。

最先要做的不是训练查询器，而是证明 P/F 确实构成可选择的能力。在当前 full-context 设定里，P 收到状态历史后用本地冻结 MTR 生成未来，F 用同一 MTR 在远端生成未来；同范围、同上下文、同坐标和同时间戳时，两者应接近身份等价。若充分适配后只剩一个固定的字节/算力差，学习 selector 没有研究必要。只有当受限 ROI、通信/计算条件或真实信息差异使 P/F 的质量—成本优势在场景间发生可复现交叉，才进入机制阶段。一次性小 selector 是必须打败的强基线；只有离线反事实显示“看到第一次真实返回后再决定”显著优于只看初态，才选用**结果条件化的 P/F 边际价值查询器**作为那一项针对性机制。序贯不是预设贡献。

按 CCF 官网当前第七版目录核验，较现实的 C 类目标是 **IROS（会议）**、**PRICAI（会议）**、**Neurocomputing（期刊）**；需要更强证据、可作为冲刺目标的是 **ICRA（会议 B）**、**AAMAS（会议 B）**、**IEEE T-ITS（期刊 B）**。其中 T-ITS 是本题最强的交通主题 B 类期刊，ICRA 是最直接的机器人方法 B 类会议；AAMAS 只有在贡献提升为部分可观测多智能体的通信/决策方法时才匹配。CCF 等级本身不规定闭环、第二数据集或固定实验数量；证据由论文主张决定。CCF C 类也不代表容易录用，本报告不预测或承诺录用。

## 目录口径与核验方法

本报告采用 CCF 官网当前显示的**第七版《中国计算机学会推荐国际学术会议和期刊目录》**。官网说明该版于 2026 年 3 月 31 日发布，并于 4 月 9 日勘误；目录评价一般只适用于 full/regular paper，workshop、short、demo 等不能仅凭会议同名推定满足目录要求。[^1] 会议与期刊在下表分列；CCF 类别也不等于 SCI/JCR 分区。学校若执行自己的目录、认定年份或作者/论文类型规则，最终仍须拿本表逐项向学院确认。用户已说明会议、期刊均可且不追赶截止日期，因此本报告按贡献匹配与补实验量排序，不按最近 deadline 排序。

核验时查阅了 CCF 当前十个分类页，并对全称与缩写做精确检索。下列六个候选均在官网当前表中找到明确条目：ICRA、AAMAS、IROS、PRICAI、Neurocomputing 位于人工智能类，T-ITS 位于交叉/综合/新兴类。[^2][^3] CCF 页面中的 AAMAS 会议与同名 Springer 期刊是两个条目，本报告推荐的是 **International Joint Conference on Autonomous Agents and Multi-agent Systems（会议 B）**。

## 六个候选的现实定位

| 类型 | Venue | 当前 CCF | 官方 scope 与本题关系 | 以当前开环基础投稿的判断 | 到达该档的主要补强 |
|---|---|---:|---|---|---|
| 会议 | IROS | C | RAS 关键词直接覆盖 autonomous vehicle navigation、intelligent transportation systems、networked/distributed robots、sensor fusion、planning 与 tracking。[^4] | **最现实的会议主线。** 若先证实 P/F 场景交叉，再用一项清晰机制和严格 held-out 对照，可支撑窄的开环机器人结论。 | 完成正式评价；通过 P/F 可选择性 gate；独立测试；与固定、rule、一次 selector 及适用机制对照。 |
| 期刊 | Neurocomputing | C | 官方 scope 要求对神经计算/学习系统有实质贡献，应用包含视觉、控制与机器人。[^5] | **最现实的期刊主线。** 篇幅适合完整机制分析，但仅把 LoRA、MTR 和规则串起来仍不够。 | 把已验证的 P/F 局限转化为可复用学习问题；做跨录制组稳定性和与主张对应的消融。 |
| 会议 | PRICAI | C | 官方 CFP 包含 AI foundations/planning、机器学习与大模型、intelligent systems/agents/robotics/IoT。[^6] | **现实备选。** 更适合强调能力获取、语言模型决策和部分可观测智能体，而非车端系统工程。 | 证明方法不依赖某个提示模板；容量等价基线；若用序贯机制，先证明首个返回带来额外决策信息。 |
| 期刊 | IEEE T-ITS | B | 官方 scope 明确包含 sensing、communications、controls、planning、coordinated vehicles/infrastructure、connected/autonomous vehicles、AI 与 multi-agent systems，并要求说明对交通系统的实际收益。[^7] | **主题最强的 B 类期刊；当前证据未到位。** 关键缺口是无独立测试结果、无 P/F 可选择性证据和无方法贡献。 | 证明交通场景中的质量—通信/计算收益；独立测试和失败分析。只有声称在线安全、实时部署或闭环交通收益时，才相应补时延、网络或闭环。 |
| 会议 | ICRA | B | ICRA 官方征稿覆盖机器人与自动化全部领域；车辆导航、网络机器人、规划与感知属于 RAS 主题。[^8] | **机器人方法 B 类冲刺。** Co-MTP 说明合作预测开环工作可进入 ICRA，但不替代本稿的证据。 | 给出可复核的机器人方法；先证明 P/F 互补或成本交叉；强简单基线、信息等价控制和独立测试。是否多域/闭环取决于泛化与控制主张。 |
| 会议 | AAMAS | B | 主会征稿要求 autonomous agents / multi-agent systems 的显著原创研究，覆盖 learning/adaptation、communication、robotics/control 与应用。[^9] | **条件性冲刺。** 仅讲 V2V4Real 上两车工具调用会显得过于应用化。 | 抽象成部分可观测多智能体的信息获取；证明方法不是静态成本规则，并与通信决策基线比较。跨域只在声称一般性时需要。 |

AAMAS 有一个有用但不能过度解读的近邻案例：2026 官方 proceedings 将 **UNCAP: Uncertainty-Guided Neurosymbolic Planning Using Natural Language Communication for Cooperative Autonomous Vehicles** 列在 Research Paper Track。[^10] 它说明“不确定性、自然语言通信、协作自动驾驶”属于主会可接受主题，不说明 ToolV2X 以现状态能达到同档。项目页面提到的 Best Paper Nomination 未在本次查到的官方 awards 材料中得到独立确认，因此不作为证据。

### 常见但不能承诺满足当前 CCF 要求的 venue

- **IEEE IV、IEEE ITSC、IEEE Transactions on Intelligent Vehicles（T-IV）、IEEE Robotics and Automation Letters（RA-L）**与智能车/V2X 主题高度相关，但在本次对 CCF 第七版十个分类页的全称和缩写精确检索中均未找到。IEEE ITSS 官方确实把 IV、ITSC 列为 flagship conferences，并设有 T-IV 出版物，这只证明主题归属，不改变 CCF 目录结果。[^7] 如果毕业条件写的是 CCF B/C，不能承诺这些 venue 合格；须让学院书面确认其自有目录。
- **ICME 是 CCF B 会议**，但其核心是 multimedia。当前系统无 RGB，贡献也不是多媒体表示、压缩或跨媒体学习；仅因 B 类而改投会造成 scope 风险。[^11]
- **Pattern Recognition 是 CCF B 期刊**，但官方 scope 明示仅把已有方法用于常规应用的稿件应投更合适的专业刊物。[^12] 除非查询器上升为跨数据集的一般模式学习方法，否则不是当前路线的优先项。

## CMP：可以借骨架，不能把单车 MTR 叫完整 CMP

CMP（*Cooperative Motion Prediction with Multi-Agent Communication*）发表于 IEEE RA-L 2025。其完整链路先由多车 LiDAR/BEV 协作感知与 AB3DMOT 产生跟踪历史，各 CAV 用 MTR 输出多模态他车轨迹，再广播预测；prediction aggregator 结合各车 GMM 轨迹、分数和 ego BEV 上下文，以注意力融合。论文在 OPV2V 与 V2V4Real 上用 1 s 历史预测 5 s，并报告他车多模态的 minADE6/minFDE6。V2V4Real 表中完整 CMP 的 5 s minADE/minFDE 为 3.9099/7.0182，优于同表 cooperative-perception-only 的 4.0857/7.3187。[^13]

代码来源可复核：作者官方仓库为 [tasl-lab/CMP](https://github.com/tasl-lab/CMP)，本地完整仓库在 `/root/autodl-tmp/CMP`，本项目审计见 [cmp_repo_notes.md](../cmp_repo_notes.md)。[^14] ToolV2X 实际借用的是 CMP 的**无预测聚合 MTR 分支**并改造成时间 `t` 固定坐标历史输入；没有运行完整 CoBEVT 感知、完整 CMP prediction aggregator 或原始数据全流程。当前适配还存在原 GT 对齐移动坐标域与因果固定坐标域的分布差异，portable torch 算子与原 CUDA 数值/速度一致性未测，训练 provenance 未完全解决。因此论文可称“基于 CMP 发布的 MTR 预测器构造 P/F 能力”，不能称“复现或改进完整 CMP”。

CMP 对新论文最重要的启发不是再做一次轨迹注意力融合，而是把 **P（共享状态后本地预测）与 F（对端完成预测后共享结果）**设为两个可比较的信息产品。只有在相同源上下文、坐标、时间戳与 ROI 下，P-full 本地同模型重算和 F-full 远端预测才构成身份检查；相同结果是预期现象。真正可研究的是有限 ROI、延迟、字节和第二次调用条件下，两种产品对自车规划损失的边际价值。

## Co-MTP：可借“未来意图交互”，不能借未来标签进入在线输入

Co-MTP（ICRA 2025）面向车路协同的他车轨迹预测：在异构图中编码车辆与路侧历史，通过 STFA/CTCA 融合历史域；未来域同时放入自车规划候选和路侧预测的他车意图，建模未来交互。V2X-Seq 实验使用 3 s 历史、5 s 未来、6 模态，表中 Co-MTP 的 minADE/minFDE/MR 为 0.76/1.15/0.16，比较对象 V2X-Graph 为 1.17/2.03/0.29；另用 0.2 m 噪声与 0.5 s 延迟做稳健性实验。[^15]

官方代码为 [xiaomiaozhang/Co-MTP](https://github.com/xiaomiaozhang/Co-MTP)，项目页给出 V2X-Seq 的预处理、训练与评估入口。[^16] 对发布代码的关键核查是：`preprocess/preprocess_v2x.py` 先从未来 `car_track` 行构造 `label_dic`，随后把切片写入 `other_fut_fea_dic` 并保存为 `other_fut_fea`。[^17] 这说明发布预处理中的该字段是由未来标签构造的代理特征；若 ToolV2X 把这条生成路径直接作为时间 `t` 在线输入，会造成未来泄漏。可以借“让真实获得的 F 影响下一步规划/查询”的思想，不能把未请求的 F、未来 GT 或 Co-MTP 的 `other_fut_fea` 标签切片提前喂给查询器。

Co-MTP 的原表也不能与 ToolV2X 本地指标直接排名：前者是 V2X-Seq 上 5 s、K=6 的**他车预测** minADE/minFDE/MR，后者当前目标是 V2V4Real 派生因果帧上的 3 s、6 点**自车文本轨迹** ADE/FDE。数据、预测对象、坐标、时间长度、模态数和评估实现均不同。若新论文主张改善预测器，必须另外恢复原任务口径或一个规范预测 benchmark；若贡献是查询/规划，则只在同一 ToolV2X 测试集内比较各策略，并把 CMP/Co-MTP 数字留在相关工作，不做排行榜。

## 从组件到机制论文：先做可选择性 gate

第一步是固定驾驶器和 MTR，按同一帧比较 Ego/P/F/PF 的轨迹质量与真实请求/响应字节。必须先通过三项检查：

1. **身份检查：**同范围 full P 与 full F 使用同一上下文和 MTR，应给出相同或可解释的等价预测；差异若来自坐标、时间或实现错误，先修数据契约，不能训练 selector 去利用错误。
2. **交叉检查：**在因果可见的场景特征分层后，确有一部分帧 P 的质量—成本占优，另一部分 F 或 STOP 占优，而且不是某个录制组、静止场景或开发集噪声造成。
3. **可学习上界：**只使用时间 `t` 输入的简单一次 selector 相对最佳固定动作有稳定空间。若 oracle 的优势极小或只有一个固定动作占优，路由问题应停止。

通过 gate 后，一次性小 MLP/树模型 selector 是强基线，并非自动成为论文创新。再用离线反事实回答“它具体缺什么”：

- 若第一次真实返回能把同一初态分成不同的最佳后续动作，才采用**结果条件化边际价值查询器**：第二阶段只在 `(x_t, r_1, 实际成本)` 上判断 STOP 或补查另一能力。
- 若收益主要来自尾部伤害而非第二次信息，再考虑一个带弃权/回退的 regret-aware selector，并用 harm rate 验证；无需同时加入序贯模块。
- 若 P/F 只存在固定通信—计算互换，直接写系统测量或采用固定规则，不应包装成学习方法论文。

离线训练可用同帧固定动作结果构造 `trajectory loss + lambda * communication/compute cost` 的反事实标签；未来 GT 只用于监督和评估。在线审计必须证明没有读取未来姿态、未请求的返回或预计算最优动作。

核心对照保持精简：Ego、Always-P、Always-F、Always-PF、现有 rule、预算匹配随机、一次 selector 与离线 oracle；只有选择了序贯机制才增加结果条件化两阶段基线。核心指标是同一测试集上的自车 ADE3/FDE3、解析失败、实际字节/调用数和按物理录制组的置信区间。仅在机制针对尾部伤害时增加 harm/risk-coverage；仅在论文声称安全或几何可行性时增加碰撞、越界或动力学代理；仅在声称实时系统时测端到端时延与两端计算。

现有 108 个开发帧已参与选择与早停语义，不能同时承担最终无偏测试。应预留未用于模型/机制选择的物理录制组；若做不到，可预注册按录制组交叉验证并说明估计限制。第二数据域不是 CCF B 的硬性条件；只有论文声称跨域一般性，或单一采集无法排除场景特例时，才成为必要证据。

新颖性措辞也必须收窄。通信选择、主动特征获取、uncertainty gating 和 learning-to-communicate 都已有大量工作；不能把“是否通信”本身写成首创。最终贡献只能围绕 gate 实际暴露的局限，例如**结果返回带来的条件化二次价值**或**能力级选择的尾部 regret 控制**，并保留同模型信息等价控制。在完成相关工作扩展与实验前，它仍是研究假设。

## 开环与闭环的取舍

闭环不是机械的投稿门槛，是否需要取决于论文声称什么。

- 若贡献限定为“冻结回放条件下，因果且成本感知的 P/F 获取改善自车开环轨迹生成”，严格的录制组外测试、反事实对照和消息成本可以构成完整证据；若机制针对尾部伤害，再加入 harm rate。题目、摘要和结论应保持在 open-loop trajectory acquisition/generation，不宣称驾驶安全、实时部署或闭环收益。
- 若贡献写成“自适应协作驾驶策略”“提高行车安全/通行效率”或“可实时部署”，则规划输出会改变后续观测，离线固定回放无法验证这种因果反馈；需要闭环仿真，并在实时性主张下加入真实网络时延/丢包与推理时间。
- ICRA/AAMAS/T-ITS 并非形式上都要求闭环；Co-MTP 的 ICRA 先例本身主要是开环预测。但 ToolV2X 的核心若从预测转为**行动相关的查询策略**，审稿人更可能追问分布反馈。最稳妥的路线是先把窄开环机制做严谨；只有结果显示查询会实质改变驾驶交互，或论文希望扩大到安全/控制贡献，再扩闭环。

## 分阶段投稿判断

当前阶段应标记为 **PROBE-GO、PAPER-NOT-READY**。下一道 gate 不是换 venue 或直接训练新查询器，而是完成既有正式评价，并证明 P/F 在充分适配、信息契约正确后仍有稳定的场景化质量—成本交叉。若 full P/F 本质身份等价且一个固定动作支配，路由方法路线停止。若一次 selector 有空间，再根据它的实证局限只选一项机制；若该机制在独立录制组上稳定，IROS/PRICAI/Neurocomputing 可进入成稿评估。ICRA/AAMAS/T-ITS 需要的不是按 CCF 等级机械增加闭环或第二域，而是分别补足可复核机器人方法、多智能体一般性或交通系统收益的主张证据。

学校目录版本仍是选择条件：本报告已经按官网当前第七版核实，但用户尚未给出学校采用的目录年份及论文类型认定细则。投稿前应让学院确认“会议/期刊、full/regular、online-first/正式卷期、学生作者顺序”对应的毕业口径。时间不限意味着没有理由为了某个截稿日跳过独立测试或把 CCF C 当成低门槛目标。

## 查阅范围与未执行边界

本次只读查阅 ToolV2X/CMP 的本地文档与代码、Co-MTP 本地正文、CCF 官网、venue 主办方 scope、论文页面及官方 GitHub 网页；没有 clone 或下载 GitHub/Hugging Face 资源，没有运行 GPU、训练、生成、新模型评测或闭环实验，也没有修改训练代码、数据、权重与既有证据文件。唯一写入是本报告。论文数字来自各自原文，只用于理解其任务和对照强度；未在本机复现，不能视为本地结果。

## CCF 官网证据摘录与可复核限制

以下是 2026-09-12 实际读取到的 CCF 官方页面短片段，保留类别层级以便交叉核对：

| 官方 URL | 实际读取到的短片段 | 用途 |
|---|---|---|
| [当前发布页](https://www.ccf.org.cn/Academic_Evaluation/By_category/) | “第七版……正式发布”；页面日期 `2026-03-31`；“目录于4月9日更新” | 确定当前版本和 full/regular paper 口径。 |
| [AI 当前分类页](https://www.ccf.org.cn/Academic_Evaluation/AI/) | `中国计算机学会推荐国际学术刊物 > 人工智能 > C类 > … > Neurocomputing` | Neurocomputing 为期刊 C；同页分开列会议与期刊。 |
| [ICRA 条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/bl/2017-04-25/592063.shtml) | `人工智能 > 中国计算机学会推荐国际学术会议 > B类 > ICRA` | ICRA 为会议 B。 |
| [IROS 条目](https://www.ccf.org.cn/c/2017-04-25/592077.shtml) | `人工智能 > 中国计算机学会推荐国际学术会议 > C类 > IROS` | IROS 为会议 C。 |
| [PRICAI 条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/cl/2017-04-25/592080.shtml) | `人工智能 > 中国计算机学会推荐国际学术会议 > C类 > PRICAI` | PRICAI 为会议 C。 |
| [交叉/综合/新兴当前分类页](https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/) | `中国计算机学会推荐国际学术刊物 > 交叉/综合/新兴 > B类 > 6 TITS` | T-ITS 为期刊 B。 |
| [图形学与多媒体当前分类页](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT) | `中国计算机学会推荐国际学术会议 > 计算机图形学与多媒体 > B类 > 10 ICME` | ICME 为会议 B，但仅用于说明非优先项。 |

CCF 的直接匿名 HTML 请求在本轮有时返回无条目挑战页，而搜索索引可读取当前聚合页；旧有单条目 URL 又保留 `2017-04-25` 的静态发布日期。所以上表把“第七版当前发布页/当前聚合分类页”和“旧静态条目页的类别面包屑”分开记录，**不把 2017 日期当成第七版发布日期**。AAMAS 会议 B 的判断来自第七版当前 AI 分类表；同页还存在名为 AAMAS 的期刊 B，必须以全称 `International Joint Conference on Autonomous Agents and Multi-agent Systems` 和“国际学术会议”层级区分。由于学校目录版本尚未给出，最终毕业认定仍应以学院保存的第七版正式 PDF 或学校目录逐项核对。本轮遵守“只写本报告”，没有另存 CCF 网页/PDF 缓存。

## Sources

[^1]: 中国计算机学会，[按类别查看推荐目录（第七版发布与勘误说明）](https://www.ccf.org.cn/Academic_Evaluation/By_category/)。
[^2]: 中国计算机学会，[人工智能类推荐目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)；具体条目：[ICRA（B）](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/bl/2017-04-25/592063.shtml)、[IROS（C）](https://www.ccf.org.cn/c/2017-04-25/592077.shtml)、[PRICAI（C）](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/cl/2017-04-25/592080.shtml)。AAMAS 会议 B 与 Neurocomputing 期刊 C 可在同一当前分类页核对。
[^3]: 中国计算机学会，[交叉/综合/新兴类推荐目录](https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/)；[T-ITS 条目（期刊 B）](https://www.ccf.org.cn/c/2017-04-25/592170.shtml)。
[^4]: IEEE Robotics and Automation Society，[IROS Keywords](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/iros/keywords-4/)。
[^5]: Elsevier，[Neurocomputing aims and scope](https://shop.elsevier.com/journals/neurocomputing/0925-2312)。
[^6]: PRICAI 2026，[Official Call for Papers](https://2026.pricai.org/calls/call-for-papers)。
[^7]: IEEE Intelligent Transportation Systems Society，[IEEE Transactions on Intelligent Transportation Systems: Scope](https://ieee-itss.org/pub/t-its/)。同页列出 IEEE ITSC、IEEE IV 与 T-IV。
[^8]: IEEE ICRA 2027，[Official Call for Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)；IEEE RAS，[Robotics conference keywords](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/iros/keywords-4/)。
[^9]: AAMAS 2026，[Main Track Call for Papers](https://cyprusconferences.org/aamas2026/call-for-papers-main-track/)。
[^10]: International Foundation for Autonomous Agents and Multiagent Systems，[AAMAS 2026 Proceedings contents](https://www.ifaamas.org/Proceedings/aamas2026/forms/contents.htm)。
[^11]: 中国计算机学会，[ICME 条目（会议 B）](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/zgjsjxhtjgjxshy/bl/2017-03-17/588306.shtml)。
[^12]: Elsevier，[Pattern Recognition aims and scope](https://shop.elsevier.com/journals/pattern-recognition/0031-3203)。
[^13]: Wang et al., [*CMP: Cooperative Motion Prediction with Multi-Agent Communication*](https://arxiv.org/html/2403.17916), IEEE Robotics and Automation Letters, 2025, DOI [10.1109/LRA.2025.3546862](https://doi.org/10.1109/LRA.2025.3546862).
[^14]: TASL Lab，[CMP official code](https://github.com/tasl-lab/CMP)；本地代码与数据流核查见 [cmp_repo_notes.md](../cmp_repo_notes.md)。
[^15]: Zhang et al., [*Co-MTP: A Cooperative Trajectory Prediction Framework with Multi-Temporal Fusion for Autonomous Vehicles*](https://xiaomiaozhang.github.io/Co-MTP/static/pdfs/ICRA_cooperative_prediction.pdf), ICRA 2025；本地提取正文为 [05_co_mtp.txt](../papers/text/05_co_mtp.txt)。
[^16]: Zhang et al., [Co-MTP official code](https://github.com/xiaomiaozhang/Co-MTP)。
[^17]: Co-MTP official code, [`preprocess/preprocess_v2x.py`](https://github.com/xiaomiaozhang/Co-MTP/blob/main/preprocess/preprocess_v2x.py)：检索 `label_dic`、`other_fut_fea_dic` 与 `other_fut_fea` 可复核未来标签切片的数据流。
