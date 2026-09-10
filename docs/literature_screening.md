# ToolV2X 阶段 A：初步文献筛查与精读清单

日期：2026-09-07。输入：`../start.md`。状态：下载与正文提取已完成；首轮关键正文分析见 [analysis_report.md](analysis_report.md)。本文保留初查记录，不是已确认的 idea_report Part 1，也不是实验结果。

## 初步判断

可以继续做前提验证，暂不能接受原方案“已可直接实现完整方法”或 METHOD-GO 的结论。不以硬件、数据规模或取得成本阻止推进。科学输入契约、方法差异及验证能力仍必须成立。

本轮阅读了 V2V-GoT 在线正文的方法与实验段、官方仓库说明，以及下列文献的官方摘要/出版页面。没有声称已经逐篇全文精读；没有下载论文、运行模型或验证检查点。正式机制调整应在关键论文精读后进行。

## 首轮文献清单

标题链接均指向本轮检索命中的作者、arXiv 或出版方页面。仅核实到 arXiv 的条目不推定其正式发表状态。优先级表示精读顺序，不表示后面的论文可以忽略。

|编号|论文与来源|发表/版本信息|相关内容与精读原因|优先级|
|---|---|---|---|---|
|1|[V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts](https://arxiv.org/abs/2509.18053)|2025 arXiv；官方仓库标注 ICRA 2026|候选后端；核对各 QA 输入、依赖、未来轨迹来源和成本边界|最高|
|2|[Select2Drive: Pragmatic Communications for Real-Time Collaborative Autonomous Driving](https://arxiv.org/abs/2501.12040)|arXiv，2025，页面最新 v4|决策相关区域通信和预测感知；检验 planning-conditioned 是否已有近邻|最高|
|3|[EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://proceedings.mlr.press/v97/ma19c.html)|ICML 2019；arXiv 1809.11142|按已获得证据继续获取信息；序贯机制的跨领域近邻|最高|
|4|[Defer to Plan: Adaptive Multi-Agent Fusion for End-to-End V2X Driving](https://arxiv.org/abs/2607.19774)|2026；页面注明 ICME 2026 接收|直接优化规划的自适应融合；补足较新的竞争工作|最高|
|5|[Co-MTP: A Cooperative Trajectory Prediction Framework with Multi-Temporal Fusion for Autonomous Driving](https://arxiv.org/abs/2502.16589)|2025；官方仓库标注 ICRA 2025|历史和未来信息协作预测；检验 M 的信息来源和增量|高|
|6|[V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_V2XPnP_Vehicle-to-Everything_Spatio-Temporal_Fusion_for_Multi-Agent_Perception_and_Prediction_ICCV_2025_paper.html)|ICCV 2025；arXiv 编号待核实|协同感知与预测联合任务，比较能力拆分是否有实质差异|高|
|7|[Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System](https://arxiv.org/abs/2404.09496)|arXiv 2024|V2XVerse 平台及系统；确认闭环评价入口|高|
|8|[COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles](https://arxiv.org/abs/2205.02222)|CVPR 2022|协作信息到驾驶行为的已有连接；防止夸大问题首次性|高|
|9|[DiFA: Differentiable Feature Acquisition](https://ojs.aaai.org/index.php/AAAI/article/view/25934)|AAAI 2023；arXiv 编号待核实|非 RL 的动态获取策略；检验不用 RL 是否属于已有方法选择|高|
|10|[Learning to Acquire Information](https://arxiv.org/abs/1704.06131)|arXiv 2017|根据前序观测选下一项信息；序贯获取背景|高|
|11|[When2com: Multi-Agent Perception via Communication Graph Grouping](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_When2com_Multi-Agent_Perception_via_Communication_Graph_Grouping_CVPR_2020_paper.html)|CVPR 2020；arXiv 编号待核实|通信时机和分组选择的基础边界|中|
|12|[Who2com: Collaborative Perception via Learnable Handshake Communication](https://arxiv.org/abs/2003.09575)|arXiv 2020|请求与伙伴选择；检查请求式接口本身是否新颖|中|
|13|[Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps](https://arxiv.org/abs/2209.12836)|NeurIPS 2022|空间稀疏通信的代表对照；不能代替规划任务近邻|中|
|14|[How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perception](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f31327e046913c7238d5b671f5d820e-Abstract.html)|NeurIPS 2023；arXiv 编号待核实|通信、延迟与融合相关近邻|中|
|15|[INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_INSTINCT_Instance-Level_Interaction_Architecture_for_Query-Based_Collaborative_Perception_ICCV_2025_paper.html)|ICCV 2025；arXiv 编号待核实|实例查询与路由；结构化对象接口本身不足以构成贡献|中|
|16|[Task-Oriented Communication for Vehicle-to-Infrastructure Cooperative Perception](https://arxiv.org/abs/2407.20748)|arXiv 2024|任务相关通信；区分任务导向与函数调用|中|
|17|[V2V4Real: A Real-world Large-scale Dataset for Vehicle-to-Vehicle Cooperative Perception](https://arxiv.org/abs/2303.07601)|CVPR 2023|真实数据任务、时间序列与标注边界|高|
|18|[DriveAgent-R1: Advancing VLM-based Autonomous Driving with Active Perception and Hybrid Thinking](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbb776e737ec3ea5925887f8740c68b4-Abstract-Conference.html)|ICLR 2026；arXiv 2507.20879；用户指定|主动调用视觉工具辅助行为规划，自适应切换推理模式；核查工具调用机制与 ToolV2X 的直接重叠|最高|

以上 18 篇已下载并提取正文，详见 [下载清单](papers/download_manifest.md)。首轮分析优先覆盖 1、18、2、3，再对相关近邻作重点核查。

### 用户补充：DriveAgent-R1

已通过 ICLR 2026 正式论文集、作者项目页和官方仓库核实标题及发表信息。其主动视觉工具调用服务于高层行为规划，并依据场景复杂度切换文本推理与工具增强视觉推理。因此，它属于工具调用机制的直接近邻，不能只列作通用驾驶 VLM 背景。这里是摘要及项目说明层面的筛查，尚未全文精读或审计代码。

后续重点对照：工具能访问的是本车已有观测还是远端独有证据；后续调用是否依赖前次返回；停止与成本如何进入学习目标；评价是高层行为决策还是轨迹与闭环驾驶。不能仅凭 ToolV2X 使用远端节点或轻量控制器就认定创新成立，也不能据摘要直接认定两者同题。

来源：[ICLR 论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbb776e737ec3ea5925887f8740c68b4-Abstract-Conference.html)、[arXiv](https://arxiv.org/abs/2507.20879)、[作者项目页](https://tsinghua-mars-lab.github.io/DriveAgent-R1/)、[官方仓库](https://github.com/wczheng/DriveAgent-R1)。

## 已发现的关键核查事项

### 1. V2V-GoT 不是现成的两个独立远端服务

官方正文 III-A、III-E、IV-A 表明：模型融合多车当前及历史特征；Q5 依赖 Q4，Q7 合并 Q5/Q6，Q6 还使用其他 CAV 的计划。官方仓库也明确各车向 MLLM 共享特征。因此仅跳过 QA 不自动减少前置通信。将它改为 P/M 服务需要重新核对可见输入及训练分布。这是对原方案可复用性判断的限制，尚非代码级审计结论。[论文正文](https://arxiv.org/html/2509.18053v1)、[官方仓库](https://github.com/eddyhkchiu/V2V-GoT)。

### 2. 新颖性不能仅建立在 planning-conditioned 或 sequential 两个词上

Select2Drive 已研究决策相关区域通信和计算通信效率；Defer to Plan 直接优化规划并自适应融合。EDDI 和 Learning to Acquire Information 已有根据已获得证据继续获取信息的机制。由这些文献推断：原方案需要证明具体远端能力与规划损失之间的独特问题，而不能用“首次考虑规划/首次序贯获取”概括。是否存在完整同题工作仍需专项全文核查。[Select2Drive](https://arxiv.org/abs/2501.12040)、[Defer to Plan](https://arxiv.org/abs/2607.19774)、[EDDI](https://proceedings.mlr.press/v97/ma19c.html)、[Learning to Acquire Information](https://arxiv.org/abs/1704.06131)。

### 3. Case C 仅能支持 M 在 P 之后有增量，不能单独证明序贯必要性

这是对原方案逻辑的分析：若所有样本都应调用 M，固定 P→M 即可；若调用前状态 s0 已能预测 M 是否有益，one-shot 也可能足够。需另外检验 P 返回结果是否在 s0 之外改善下一步效用预测和实际策略收益。事后知道真实损失的 oracle 也不是可部署策略。

### 4. P 与 M 的信息等价性需明确

若 M 仅用 P 已返回的状态、相同地图及同等预测器，远端预测优势可能只是额外计算或模型差异。若 M 使用远端独有历史，则应比较“传同样历史到 ego 后本地预测”。这不否认远端计算价值，而是决定论文到底证明信息价值、计算委托价值，还是序贯决策价值。Co-MTP 与 V2XPnP 是核查这一边界的重要文献。[Co-MTP](https://arxiv.org/abs/2502.16589)、[V2XPnP](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_V2XPnP_Vehicle-to-Everything_Spatio-Temporal_Fusion_for_Multi-Agent_Perception_and_Prediction_ICCV_2025_paper.html)。

### 5. 工具路径不能只按排列枚举

原方案 M(target) 如果依赖 P 才发现的目标 ID，则 M-only 路径未必可用。如果改为 M(ROI)，服务内部的检测、跟踪、目标发现均应纳入调用成本。PM 与 MP 只有证据依赖或时延影响不同才是不同策略，不能把同一信息集合机械重复计算为序贯收益。

### 6. 因果输入与评价需要逐字段核查

reference trajectory 与 partner plan 应追踪到决策时刻真实可得的规划输出，不能将日志中的实际未来轨迹默认为在线计划。离线未来标签可用于训练效用监督及评价，但不能进入部署策略输入；训练/验证/测试应按驾驶序列隔离。原数据 Q3 的 invisible 依据包括 ego detector 未检出，不能直接等同物理遮挡。在线计划来源目前待代码和样本核验。[V2V-GoT III-C—III-F](https://arxiv.org/html/2509.18053v1)。

### 7. 成本与延迟属于研究目标，即使不考虑硬件限制也必须计量

请求响应之外，预先上传的特征、工具内部推理和最终 planner 重算都需要归属；离线缓存不能当成在线零成本。按请求→服务→返回→重规划记录时序。离线轨迹误差、离线碰撞代理与闭环驾驶指标分别报告。闭环验证能力可从 COOPERNAUT 和 V2XVerse 的平台设计核查。[COOPERNAUT](https://arxiv.org/abs/2205.02222)、[V2XVerse](https://arxiv.org/abs/2404.09496)。

## 后续需要回答的前提，不作为本轮已确认 RQ

1. 后端是否能提供输入隔离、时间因果有效的 P/M，且前置通信不绕过控制器？
2. M 的额外价值来自哪里，信息等价本地预测后是否仍存在？
3. P 返回后是否增加可学习的决策信息，并在相同预算下超过 one-shot 和简单规则？
4. 这些结果是否跨驾驶序列成立，而非少数 oracle 样本或随机生成波动？

原文中损坏的引用标记与标题不作为事实依据；本文给出可访问的一手链接。没有据此确认完整方法创新性或实验成功。
