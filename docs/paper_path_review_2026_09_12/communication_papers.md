# ToolV2X 通信论文路径评审

## 结论先行

现有五条件共享驾驶训练是形成论文证据的必要底座，还不是论文方法。最小且有机会与近邻拉开距离的主张应收缩为：**在因果的时刻 `t` 输入下，针对同一远端观测，学习是否通信、传 P（状态与 1 秒历史、接收端冻结 MTR）还是传 F（发送端冻结 MTR 预测），并仅在第一条返回证据确实改变后续价值时进行第二次获取；用实际序列化字节、两端计算和下游轨迹损失共同约束。** 关键差异是“能力类型与计算位置的任务条件化选择”，不是笼统的规划感知通信，也不是把 `P/F/PF` 名称写进提示词。

这是一条可检验的投稿路径，不构成 CCF-B/C 录用保证；论文强度最终取决于未见录制组上的质量—成本优势、等信息对照和统计稳定性。

Select2Drive 已覆盖规划路线约束的关键区域通信、预测感知、时延模型和闭环驾驶；Where2comm 已覆盖动态空间稀疏通信和多轮请求；How2comm 已覆盖空间—通道压缩、特征保真和时延补偿。新增强近邻 UNCAP 还覆盖了先选通信车辆、再发送带不确定性的自然语言消息，以及按规划互信息筛选消息。因此，“任务条件化”“多轮”“自然语言”“互信息”任何一个词单独都不构成 ToolV2X 的充分新颖性。[^1][^4][^6][^8]

当前代码中的 `rule` 确实是固定条件分支：先按速度决定 STOP/P，再按 P 是否返回目标及几何候选决定 F/STOP（`src/planning/episode.py:8-33`）。训练的共享 GoT 只学习在五种已执行证据条件下输出六点轨迹；查询器没有学习。真实查询、证据更新与最多两次调用已接通（`src/planning/episode.py:36-118`），这使后续研究可做，但不能把现有规则包装成学习路由。

## 当前 ToolV2X 的可写证据边界

截至本次只读查阅，正式重跑仍在第一轮训练，状态文件显示 `epoch=0`、`957` 次更新、`7656` 条样本，尚无轮末选模、540 条开发集生成或全量质量结果；这是会继续变化的运行快照。已完成且可写的是 2987 个训练帧、108 个开发帧、8/2 个录制组隔离、每帧五条件因果任务准备，以及冻结 MTR、无真实 RGB、直接 3 秒六点轨迹的明确契约。联调的每条件 4 帧 ADE/FDE 明显劣于简单运动学外推，只能证明链路可运行，不能支持协同驾驶收益（`docs/framework_baseline_results.md:36-64,94-114`）。

本地实现已经具备三项值得保留的论文基础：P 与 F 返回同一时刻、同一坐标系的可审计报文（`src/tools/vehicle.py:58-102`）；P 在接收端用同一 MTR 重算、F 在提供端计算并保留来源标签（`src/tools/vehicle.py:162-205`）；评价同时保存 ADE3/FDE3、失败率、请求/响应字节和两端计算时间，并做逐帧 Ego 配对伤害率（`src/evaluation/framework.py:38-68,94-117`）。其限制也必须原样写出：应用层 JSON 字节不是无线带宽或传输时延；计算时间不含模型加载、真实网络和原感知；当前只是开放环模仿评价。

这里还有一个必须先证伪的重复信息问题：默认 `predict_p=True` 时，P 的历史一到接收端就已经由同一冻结 MTR 变成未来预测；在 full ROI、相同目标和相同上下文下，它应与 F 的语义信息等价。因此 P 后再取 F 通常只是换计算位置或消息表示，不是新增未来信息。若有限 ROI 下出现差异，应拆分成目标集合、MTR 上下文截断、数值/序列化和计算位置效应，不能笼统归因于“第二次工具推理”。

## 三篇深读

### Select2Drive

**正式身份与科学叙事。** 论文已正式发表于 *IEEE T-ITS* 26(12), 21939–21953，DOI `10.1109/TITS.2025.3611377`；本地是 arXiv v4/接收稿。DPP 以低维热图预测和运动仿射重建补偿累计时延；APC 用先前规划路点形成 Area-of-Importance，再结合发送端置信图裁剪 BEV 特征。主张是任务无关宽视野会扰动行为克隆规划，因而只传驾驶关键区域。[^1][^2]

**任务、成本与最低消融。** V2Xverse/DAIR-V2X 报 AP30/50/70；CARLA Town05 31 条路线报 route completion、infraction、driving score 和碰撞。成本含稀疏 BEV 理论字节，以及 1–20 MHz 信道下消息、编码、队列和决策时延。最低对照是 full、去 APC、无融合和 Where2comm/Select2Col/SiCP。20 MHz 时去 APC 使 driving score 46.904→40.991，但 route completion 略升，故不能只报综合分。

**代码边界与复用。** 论文脚注指向 ZJUNICE；其公开仓库是 `a75929928/Select2Drive` 的 fork，含四个子系统和训练/评价入口，但示例保留作者绝对路径，页面仅见 3 次提交。本次未下载、运行或验权重，只能写“代码页公开”。[^3] 可复用的是规划无关 ROI、带宽档位、闭环分项和 `w/o task conditioning`；DPP 会改变当前冻结 MTR 的 F 定义，不宜直接并入。

### Where2comm

**正式身份与科学叙事。** Where2comm 是 NeurIPS 2022 Spotlight。检测置信图决定首轮稀疏发送，`1-confidence` 形成请求；后续发送再乘上一轮请求图，并以是否有激活 patch 建通信图，最后做带距离编码的逐位置注意力。其 “critical” 由检测任务定义，多轮补的是感知空间区域，并非规划轨迹价值。[^4]

**任务、成本与最低消融。** 四个数据集覆盖相机/LiDAR与车/无人机，报 AP@0.5/0.7；成本是非零 feature+index 的理论字节，不含额外压缩。训练随机预算/轮数。最低消融为 confidence mask、位置编码、confidence-aware MHA，以及一轮/多轮和预算分配。后轮留预算更优只证明请求图定向区域有效，不能证明 P 返回后选择 F 有效。

**代码边界与复用。** 官方 README 当前只勾选 DAIR-V2X，不能据此声称四数据集均可复现。[^5] 本地 CMP 移植的训练 top-k/推理阈值 mask 返回激活比例（`where2comm_fuse.py:39-80`）；后续只是 mask+逐位置注意力（`:127-208`），无跨轮 request-map；`point_pillar_where2comm.py:88-120` 的 `com` 也不是序列化字节。它可作置信 ROI 基线，不能标作完整复现或与 JSON 字节直比。

### How2comm（第三篇选择）

**为何选它。** How2comm 是 NeurIPS 2023 主会论文，比 When2com/CoDriving 更直接解释通信压缩后的质量。MIC 做空间/通道筛选，以 JSD 互信息约束原特征和稀疏消息；FDC 用历史 flow/scale 对齐延迟消息；STCFormer 融合共有/独有空间和时间线索。该互信息衡量**表示保真**，不是规划边际价值。[^6]

**任务、成本与最低消融。** 三个 LiDAR 数据集报 AP@0.5/0.7；成本沿用 `log2(bytes)`、默认约 1 MB，并测 0–400 ms 时延及位姿噪声。最低对照是空间/空间+通道/去 MI、无/有延迟补偿、No Fusion 和等预算曲线。去 MI 后三数据集 AP 都下降，但只支持感知特征保真。

**代码边界与复用。** 官方仓库有 train/test、OPV2V/V2XSet 配置和权重链接，无 DAIR 入口；示例配置名含 `v2xset`，根目录却指 OPV2V，默认 `frame:1, delay:0`，不能直接对应论文时序实验。本次只读网页、未运行。[^7] 可复用“双重质量”：先验 P/F 表示，再验下游规划；特征相似不能代替安全。

## 横向对照

| 工作 | 选择变量 | 条件信号 | 质量目标 | 成本口径 | 是否证明序贯能力选择 |
|---|---|---|---|---|---|
| Select2Drive | BEV 驾驶关键区域 | 先前路点、检测置信、时延 | AP + 闭环驾驶 | 理论特征字节、MHz→时延、计算时延 | 否；无 P/F 能力类型选择 |
| Where2comm | 区域、协作者、轮次预算 | 检测/请求置信图 | AP | 理论稀疏 feature+index 字节 | 否；证明的是感知请求图多轮 |
| How2comm | 空间/通道与延迟补偿 | 特征重要性、历史 flow | AP、特征互信息 | 理论消息字节 | 否；无下游任务选择 |
| UNCAP | 车辆、语言消息、融合消息 | 几何规则、感知不确定性、规划互信息 | safety score、计划不确定性 | 文本带宽缩减 | 两阶段但选车为启发式；不是 P→F 结果依赖策略 |
| 当前 ToolV2X | STOP/P/F/PF/rule | 速度、P 返回对象与几何关系 | 共享驾驶六点轨迹 | 实际 JSON 字节、两端计算 | 接口已具备；学习性与收益均未证明 |

When2com 仍应作为 `who/when` 经典基线引用（CVPR 2020），但其 AirSim 多视角语义分割和 ModelNet 形状分类距当前 V2V4Real 驾驶任务较远；CoDriving 适合作为 V2Xverse 端到端平台来源。二者不值得在这份“质量—通信成本选择”报告中占用第三篇深读名额。

## 强近邻短对照：UNCAP

UNCAP 的最新正式身份是 AAMAS 2026 Research Paper Track，会议论文题名已更新为 *Uncertainty-Guided Neurosymbolic Planning Using Natural Language Communication for Cooperative Autonomous Vehicles*，页 2609–2617，DOI `10.65109/WHRE3513`；不能再写成仅 2025 预印本。它先让所有车辆广播位置/速度（BARE），再用距离与朝向目标的几何条件选车（SPARE），随后发送带检测不确定性的语言消息，并按规划点互信息筛选。论文自己明确 SPARE 是 heuristic，说明“有 if/else”不必然妨碍发表；前提是它服务于一个独立、可检验的整体机制，并有不确定性、通信量和安全指标。[^8]

这篇对 ToolV2X 的压力很直接：两阶段通信、自然语言、规划条件、互信息和低带宽都已出现。ToolV2X 可保留的差异是 P/F **能力语义与计算位置**、相同远端信息的接收端/提供端预测对照、以及真实返回后是否继续调用。还要指出 UNCAP 的边界：论文目标式把通信成本简化为二元连边；计划置信/熵下降不自动等于轨迹更安全。公开“Code”链接指向项目网站仓库，仓库提供 OPV2V/OpenCOOD/仿真目录与 demo notebook；仅凭 README/demo 不能视为端到端完整复现。[^9]

## 可检验的最小论文贡献候选

先做 full P-local 与 full F-peer 的质量—字节—两端计算交叉，再决定是否存在可选择的能力差异。一次性小选择器本身只是强基线，不能直接当原创方法；可写贡献必须来自下列问题设定、反事实监督和可审计评价的组合，并在未见录制组上形成稳定优势。

建议只保留一个主贡献和两个支撑贡献：

1. **主贡献候选：因果反事实的能力/计算位置决策问题。** 在时刻 `t` 的同一远端信息上，把“传历史到本地算”和“远端算后传预测”设成信息等价或近等价动作；以逐帧反事实轨迹代价或 cooperative regret 加实际字节和两端计算定义监督。轻量选择器只是该问题的一种实现，必须与 MLP/树等简单模型比较；语言模型 token 概率、检测置信或特征互信息不能代替下游价值。
2. **条件性支撑贡献：证据后更新。** 只有在第一步 P 的对象、覆盖、不确定性或规划风险能预测一个真正不同的后续动作价值时才继续。以 `s0` 一次选择器对比 `s0+eP` 证据更新选择器，证明收益来自返回内容，不是固定 P→F 次序；若 full P 已产生等价 F，第二步动作必须改为新增 ROI/协作者/模态，否则没有该贡献。
3. **支撑贡献：可审计的成本—质量分解。** 报告 payload bytes、调用数、发送端 MTR 计算、接收端 MTR 计算、驾驶推理和失败；若没有真实网络，只把字节映射到若干假设吞吐率画敏感性曲线，并明确这不是实测延迟。

这一路由不是 if/else 的最低证明条件是：在未见录制组上，同一初始本车状态会因真实 P 返回内容不同而选择不同且信息不重复的第二步；学习选择器在相同预算下优于固定 P、固定 F、固定 PF、当前 rule、随机/频率匹配策略和只看 `s0` 的一次选择器；去掉任务输入或打乱 P 返回后优势消失；预测的增益/风险与真实逐帧反事实改善有校准关系。若这些条件不成立，就应诚实把 `rule` 和一次小选择器保留为工程基线，不把它们升格为方法。

## 必须完成的对照与顺序决策门

| 问题 | 最少对照 | 通过标准 |
|---|---|---|
| 通信本身有无价值 | Ego、P、F、PF；逐帧配对 | 平均改善之外，同时报告 improved/harmed fraction 与录制组置信区间 |
| 收益是否来自额外信息 | P-full 本地 MTR vs F-full 远端同 MTR；统一 ROI/目标集合 | 质量差解释为上下文、截断或计算位置，不能混成“预测工具更智能” |
| 任务条件化是否必要 | 规划无关置信/距离 ROI、含任务选择器、去任务输入 | 含任务信息在相同字节预算上稳定更优 |
| 学习是否必要 | 当前 rule、随机、频率匹配、轻量 MLP/树、所提选择器 | 所提方法超过简单模型；否则选最简单可解释模型 |
| 第二步是否必要 | 一次 `s0`、固定 P→F、`s0+eP`；分别给允许动作集的 oracle 上界 | 在线 `s0+eP` 在同预算胜过一次与固定序列，且决策确实依赖 eP；oracle 只解释上界，不作硬门 |
| 成本权衡是否真实 | 多个预算/λ、实际 bytes+compute，失败计入 | 给出 Pareto 曲线，不只报单点加权分 |
| 泛化是否可信 | 8/2 录制组之外增加冻结测试组或交叉录制评估，多种 selector seed | 调参只用开发组；按录制组 bootstrap/统计，不把帧当独立样本 |

**现在没有必要先实现顺序决策。** 先让本轮共享驾驶模型完成并通过基本轨迹质量门，再回答：F 或 P 是否在足够多帧改善 Ego；P/F 的场景相关质量—成本优势是否交叉；P 返回是否能预测停止或一个不重复动作的增量。随后才值得比较在线 `s0` 与 `s0+eP` 的同预算结果。一次选择若可直接选 PF，它与两步的终态动作集相同，知晓全部反事实的 oracle 上界可以相同；这不否定在线证据更新可能通过 P 后停止来节省成本，故 oracle 只描述动作集和上界。若在线 `s0+eP` 没有增益，论文应收缩为单步研究；若 full P/F 质量相同，主线只能研究计算卸载与消息表示的成本选择，不能声称信息质量路由。

## 结果出来后的写法

### 情形 A：学习选择器形成稳定 Pareto 优势

摘要按“问题—机制—公平性—结果”写：固定稀疏通信只选区域，现有语言通信多选协作者；ToolV2X 在同一因果远端信息上选择原始历史或远端预测及计算位置。方法先用共享驾驶器得到动作条件反事实监督，再训练轻量选择器；仅在 P 返回可改变后续价值时追加 F。结果至少同时给出相对 Ego 的 ADE/FDE 或安全代理改善、harm rate、平均 bytes、两端计算、与一次选择器的差值及置信区间。不要使用“首次任务条件化/首次序贯/首次语言 V2X”。

### 情形 B：一次选择有效，顺序选择无额外收益

删除“序贯”主张，把研究写成**预算约束的能力与计算位置选择**。第二步负结果放在消融：真实 P 返回没有提供足够的条件增量，因此一次轻量选择器更简洁。若它在未见录制组上稳定超过固定动作、规则与预算匹配随机，并且 P/F 等信息对照成立，已经形成可检验主张；是否需要闭环、安全指标或第二场景取决于最终主题与主张范围，应作为增强证据，而不是按 CCF 等级机械设门。

### 情形 C：只有固定 F/PF 好，选择器不优于简单规则

不能写路由方法论文。可把结果作为“远端预测证据对共享驾驶的作用”实验，继续寻找可预测的异质性；若无异质性，固定 F 就是正确基线。若只比 Ego 平均好但 harm rate 高，先做 do-no-harm 选择，不宣称普遍协同收益。

### 情形 D：正式模型仍不能超过运动学基线或轨迹不物理

停止方法收益写作。可以报告框架、因果契约和失败诊断，但这不足以支撑目标论文。解析成功、训练损失下降、MTR 冻结或接口检查通过都不能替代下游任务质量。

## 透明实现差异的写法

允许与参考框架有合理差异，但表述要逐项可核验：Select2Drive/Where2comm/How2comm 传 BEV 特征，ToolV2X 传对象历史或预测的结构化 JSON；它们多优化检测 AP，ToolV2X 用共享 GoT 输出轨迹；它们的通信量主要是理论 feature bytes，ToolV2X 是实际应用层 payload；Select2Drive 有 CARLA 闭环，当前 ToolV2X 没有；UNCAP 用视觉/语言与不确定性，当前 ToolV2X 无真实 RGB。建议在论文中单列 “Implementation differences from reference systems”，说明每项差异的目的、公平控制和结论限制，不使用“复现 Select2Drive/Where2comm/How2comm”。

## 查阅状态与来源

查阅日期：2026-09-12。

| 项目 | 正式来源 | 代码 | 本次状态 |
|---|---|---|---|
| Select2Drive | [IEEE DOI](https://doi.org/10.1109/TITS.2025.3611377)；[arXiv v4](https://arxiv.org/abs/2501.12040) | [ZJUNICE fork](https://github.com/ZJUNICE/Select2Drive)；[源仓库](https://github.com/a75929928/Select2Drive) | 正文已本地深读；代码网页只读；未下载/运行 |
| Where2comm | [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html) | [官方仓库](https://github.com/MediaBrain-SJTU/Where2comm) | 正文已本地深读；官方网页与本地 CMP 移植只读；未运行 |
| How2comm | [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f31327e046913c7238d5b671f5d820e-Abstract.html) | [官方仓库](https://github.com/ydk122024/How2comm) | 正文已本地深读；代码/配置网页只读；未下载/运行 |
| When2com | [CVPR 2020 official page](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_When2com_Multi-Agent_Perception_via_Communication_Graph_Grouping_CVPR_2020_paper.html) | [官方仓库](https://github.com/GT-RIPL/MultiAgentPerception) | 用于第三篇取舍和经典基线定位；未深度复现 |
| CoDriving | [论文](https://arxiv.org/abs/2404.09496) | [V2Xverse 官方仓库](https://github.com/CollaborativePerception/V2Xverse) | 用于第三篇取舍和闭环平台定位；未深度复现 |
| UNCAP | [AAMAS 2026 正式论文](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WHRE3513.pdf)；[DOI](https://doi.org/10.65109/WHRE3513) | [项目页](https://uncap-project.github.io/)；[公开仓库](https://github.com/uncap-project/uncap-project.github.io) | 正式身份、方法和代码页面已核对；仅作强近邻短对照，未运行 |

[^1]: Jiahao Huang et al., “Select2Drive: Pragmatic Communications for Real-Time Collaborative Autonomous Driving,” *IEEE T-ITS*, 2025, [DOI](https://doi.org/10.1109/TITS.2025.3611377).
[^2]: Select2Drive [arXiv:2501.12040v4](https://arxiv.org/abs/2501.12040)；本地提取正文 `docs/papers/text/02_select2drive.txt`，重点核对系统/成本 `:328-501`、DPP/APC `:477-785`、评价 `:790-949`。
[^3]: ZJUNICE 的[组织页](https://github.com/ZJUNICE)明确列出其 Select2Drive fork 来源；[公开源码页](https://github.com/a75929928/Select2Drive)给出模块与运行入口。
[^4]: Yue Hu et al., “Where2comm,” [NeurIPS 2022 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)；本地正文 `docs/papers/text/13_where2comm.txt:185-376,372-440,964-1018,1106-1139`。
[^5]: Where2comm [官方仓库 README](https://github.com/MediaBrain-SJTU/Where2comm/blob/main/README.md?plain=1)，查阅时数据支持清单与训练/测试入口如文中所述。
[^6]: Dingkang Yang et al., “How2comm,” [NeurIPS 2023 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f31327e046913c7238d5b671f5d820e-Abstract.html)；本地正文 `docs/papers/text/14_how2comm.txt:200-329,484-520,586-664`。
[^7]: How2comm [官方仓库](https://github.com/ydk122024/How2comm)及[公开示例配置](https://github.com/ydk122024/How2comm/blob/main/v2xvit/hypes_yaml/how2comm/v2xset_how2comm_stcformer.yaml)。
[^8]: Neel P. Bhatt et al., “UNCAP,” [AAMAS 2026 正式会议论文](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WHRE3513.pdf)，DOI `10.65109/WHRE3513`；正式论文明确给出 BARE/SPARE、几何启发式和规划互信息。
[^9]: UNCAP [项目页](https://uncap-project.github.io/)与其链接的[公开仓库](https://github.com/uncap-project/uncap-project.github.io)。
