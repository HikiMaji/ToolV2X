# ToolV2X 主线远端能力调用候选

日期：2026-09-12。性质：方法候选与反方核查；不是实现完成、实验结果、最终方案批准或录用承诺。本次只读项目代码、已有论文正文及论文/出版方网页，没有运行模型、GPU 或训练，没有下载代码、权重或数据，也没有修改运行配置。

## 1. 结论先行

本轮不再把“固定先 F、再补 P”作为主线。较有辨识度的主候选是 **任务对比远端执行（Task-Contrast Remote Execution, TCRE）**：Ego 以模型实际生成的初始轨迹和一个因果构造的受限对比探针描述当前纵向驾驶问题，连同预算发给邻车；邻车不是按固定 ROI 搬运数据，而是在自己的私有、截至时刻 `t` 的上下文中执行 P 或 F 任务，找出对两条探针具有不同关系的目标，并返回原始证据、探针—目标关系和覆盖账本。Ego 在看到真实返回后再选 `STOP/P/F`，首版最多两次能力执行。当前没有 map、GoT 候选分数或两轨迹可行性证明，因此这两条轨迹不能表述成排序不稳的两个可执行方案。

TCRE 的可能增量不是“两条轨迹”“规划相关性”“主动查询”中的任何一项。这些都已有直接近邻。可检验的新组合是：

1. 请求参数是一个受限、机器可读的**驾驶决策对比探针**，而不是固定矩形、自然语言包装或未约束的 CoT；
2. 邻车利用 Ego 尚未拥有的私有上下文执行能力特定的目标检索，P 与 F 的执行语义不同；
3. 返回既含可供原 GoT 使用的原始证据，也含可审计的选择、覆盖和成本账本；实际返回再进入第二次 `STOP/P/F` 决策。

第二候选是 **条件化远端能力委托（Conditional Remote Capability Delegation, CRCD）**：把一段受限的 `P -> 条件 STOP/F` 程序随驾驶问题一次发给邻车，由邻车在私有 P 结果上决定是否执行 F。P、F 仍各计一次能力执行，只把两次网络往返合并为一次。它值得作为部署候选和强基线；单独看更像 RPC/workflow batching 或 code shipping，不能据此声称算法创新。若 TCRE 的任务条件确实改变邻车的目标选择和续调用，CRCD 可成为同一方法的低往返实现；否则只能报告时延优化。

## 2. 当前主线与接口事实

项目主线要求本车根据当前驾驶任务及已购证据，在 `STOP/P/F` 中决定是否调用远端能力、请求什么，并依据返回调整下一步；P→F、F→P 都可以，最多两次，RSU/I 后置。P 是当前状态加 1 秒因果 tracker 历史；`history_valid` 表示 tracker 状态可用，包括漏检时因果维护的状态，并非逐帧检测匹配位。F 是邻车用完整私有上下文和冻结 MTR 计算的每目标六模态未来。完整 P 传到 Ego 后用同一 MTR 重算，可与完整 F 信息等价；这不取消调用前 F 的远端能力属性，但要求把**信息获取**与**计算委托**分开核算。

现有请求只有 `tool/provider/scene/g/roi`，服务按 ROI 在完整 P 或完整上下文 F 的输出端筛选；F 当前仍为所有目标运行预测后再筛返回目标（[`src/tools/vehicle.py:58`](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:58)，[`src/tools/vehicle.py:75`](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:75)）。P/F 的已有字段和因果校验见 [`src/tools/vehicle.py:12`](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:12) 与 [`src/tools/vehicle.py:105`](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:105)。当前策略只有固定序列和一个 P 后按目标关系决定 F 的简单规则，没有驾驶问题参数、目标预算或学习价值（[`src/planning/episode.py:8`](/root/autodl-tmp/ToolV2X/src/planning/episode.py:8)）。因此下面两项都需要显式扩展请求协议；它们不是对现有代码行为的重命名。

原框架已有合理的共同前置：只读本地证据先生成 Ego 初始 Q8/Q9，STOP 直接复用该方案；查询策略可以参考这个模型生成的方案，而不能读取 GT 未来（[`docs/framework_design.md:127`](/root/autodl-tmp/ToolV2X/docs/framework_design.md:127)）。TCRE 沿用这一步，查询后最终 GoT 再运行一次；第一次返回与第二次调用之间用小型数值策略，不再插入一次 GoT。所有策略都必须计入共同初始规划成本和最终规划成本。

| 路径 | Ego 新获得的信息 | 远端执行 | 必须保留的分离对照 |
|---|---|---|---|
| `P_contrast` | 被 peer 任务检索选中的原始私有观测/历史 | 因果目标检索，不运行 MTR | 传 full P 后在 Ego 运行相同检索；完整 P 后本地 MTR |
| `F_contrast` | 由私有上下文产生的预测及任务关系 | 完整上下文 MTR + 目标检索 | 完整 P 本地同 MTR；传 full F 后只在 Ego 做相同检索 |
| `P->F` | 先观测，再把后续预测计算留在远端 | 两次能力执行 | 完整 P 一次传输后本地重算，分别核算字节和两端计算 |

这样才能判断收益来自 peer 独有信息、任务检索，还是计算位置；“远端执行了”本身不等于新增信息。

## 3. 直接近邻核查：方法实际做了什么

### 3.1 Select2Drive：规划给出空间请求图，尚不是能力任务委托

Select2Drive 的 APC 用先前时刻的 waypoint plan，在最近 waypoint 周围构造高斯 request map；邻车进行延迟补偿的特征预测，按 request/confidence map 打包特征，再由 Ego 融合并产生下一计划。它已经覆盖“先有规划、再让通信集中到规划关键空间”以及闭环质量—通信评价。因此，若 TCRE 只把两条轨迹变成并集/差集 mask，再返回该 mask 内的 P/F，它就是 APC/Plan2comm 风格动态 ROI 的变体。TCRE 必须比较同字节的单轨迹走廊、两轨迹并集、两轨迹差异 mask；只有邻车以**两条问题探针的不同作用关系**执行目标级 P/F 任务后仍有增量，才留下方法空间。[arXiv v4 与作者正文](https://arxiv.org/abs/2501.12040)；[APC 方法段](https://arxiv.org/html/2501.12040#S4.SS2)；[正式 DOI](https://doi.org/10.1109/TITS.2025.3611377)。网页与本地 `02_select2drive.txt` 方法段已交叉核验。

### 3.2 DriveAgent-R1：返回驱动的多轮工具调用已经成立

DriveAgent-R1 先在 text/tool 两种思考模式间选择；tool 模式中，LM 交替产生思考和工具请求，Vision Toolkit 返回历史/其他相机视图、ROI 放大、深度或 3D 检测，结果进入上下文，直至输出动作或达到调用上限。其三阶段训练先 SFT 两种模式，再分别强化模式，最后学习自适应选择；工具奖励相对 text-only 结果强调有用调用。它已经覆盖“驾驶歧义触发工具、读返回后继续调用、STOP 与工具模式切换”。TCRE 的边界只能来自**跨节点私有上下文、由请求者定义的受限驾驶问题、peer 端能力执行和真实通信/远端计算账本**，不能来自多轮工具推理本身。[arXiv v3](https://arxiv.org/abs/2507.20879) 明示 Accepted to ICLR 2026；[HTML 方法段](https://arxiv.org/html/2507.20879v3#S2.SS1)。网页与本地 `18_driveagent_r1.txt:159-230,248-374` 已交叉核验。

### 3.3 EDDI：结果条件化的成本敏感获取是通用成熟问题

EDDI 用 Partial VAE 对任意已观测变量子集建模，并以目标变量的期望信息增益逐项选择下一变量，直到预算/停止条件满足。它已经给出“当前已有值不同，下一项获取也不同”的通用框架。将 P/F 当成两个缺失特征，训练一个 `STOP/P/F` 选择器，本质上是主动特征获取的驾驶应用。TCRE 要求 peer 执行一个由驾驶对比定义的任务，并显式验证它胜过得到相同在线输入、动作和预算的 EDDI 风格或小 MLP 获取器。[ICML 2019 PMLR 正式页](https://proceedings.mlr.press/v97/ma19c.html)；方法在本地 `03_eddi.txt:184-304` 已核验。

### 3.4 UNCAP：选车、传不确定语义、按信息量融合也已很近

UNCAP v2 先以 BARE 广播车辆位置/航向，再由 SPARE 通过几何规则筛协作车；选中车辆发送带感知置信度的自然语言对象消息，Ego 按对象不确定性和点互信息选择融合，再交给 VLM 规划。它已经覆盖两阶段选通信方、任务/计划相关的信息量和语言证据。它没有让 Ego 把一对数值轨迹探针交给 peer，也没有在 P/F 两种远端能力间选择。TCRE 不能靠“文字请求”与它区分，主协议应为数值结构体，文本只作可读日志。[arXiv v2](https://arxiv.org/abs/2510.12992) 与 [HTML 方法 3.3–3.5](https://arxiv.org/html/2510.12992v2#S3.S3) 已核验；arXiv 标注 journal reference AAMAS 2026。

### 3.5 规划影响、任务通信和渐进查询进一步压缩了表述空间

CAPO 已用替换单个参与者预测后引起的控制变化来给预测目标加权；TIP 已以期望效用形式评价上游误差对规划的影响。所以“两方案效用差”“找改变规划的目标”不单独构成创新。[CAPO 作者正式条目](https://rowanmcallister.github.io/publication/capo/)；[TIP 的 ICML 2023 PMLR 正式页](https://proceedings.mlr.press/v202/li23al.html)。TOCOM-V2I 已按空间关系和感知先验选择任务相关特征，再做熵编码与注意力融合；这说明“任务相关传输”也不能泛称新。[作者预印本](https://arxiv.org/abs/2407.20748)。Collaborative Edge-to-Server Inference 已将问题与低清全图先发到服务器，服务器按输出 min-entropy 和内部注意力请求高分辨 ROI 再推理；这直接覆盖“远端执行问题、依据首次推理要求补充、再选答案”的通用骨架。[arXiv v2](https://arxiv.org/abs/2512.16349)；正文仅说明初稿曾投 GLOBECOM 2026，未核到正式录用页。

LLMCompiler 更直接否决“把条件工具程序发去一次执行”本身的新颖性：它的 Task Fetching Unit 用前序输出替换后续任务变量，3.4 节对简单 if-else 静态编译执行流并按中间结果选分支，复杂情形再回 Planner 动态重规划。传统 function shipping 也早已把传数据还是传计算、在哪端执行视为成本决策。因此 CRCD 从一开始就降为部署候选/强对照，不能独立列论文贡献。[LLMCompiler 的 ICML 2024 PMLR 正式页](https://proceedings.mlr.press/v235/kim24y.html)与[作者方法正文](https://arxiv.org/html/2312.04511#S3.SS4)；[To Ship or Not to (Function) Ship](https://arxiv.org/abs/1807.11149)。

## 4. 候选一：任务对比远端执行（TCRE）

### 4.1 请求中的驾驶问题从哪里来

时间 `t` 先由当前共享 GoT 仅用 Ego 合法输入生成初始 Q8/Q9 与六点轨迹 `tau_a`。构造器沿 `tau_a` 的空间曲线生成一个“较早减速/让行”的时间参数化版本 `tau_b`：只收缩沿原曲线的累计进度，不创造新车道或转向意图，并用预注册的最大减速度/速度界约束构造。这个检查只排除明显越界的探针，不证明 `tau_a/tau_b` 在地图、交通规则或动力学上可执行，也不表示 GoT 认为两者排序接近。若 `tau_a` 解析失败，不发对比请求，按已有失败/回退规则处理。`tau_b` 不是 GT、导航或最终计划，也不作为新规划贡献；它只给远端目标检索提供一组问题坐标，最终任务效用仍由 actual GoT 输出和隔离 evaluator 决定。

第一版把问题限制为纵向“继续当前方案还是更早让行”；项目没有导航路线，不能伪造左右绕行候选。若以后任务端能稳定生成多候选，才把 `tau_b` 换成模型第二候选。请求为结构体：

```text
problem = {
  as_of_g, coordinate_frame,
  candidate_a: six points + times,
  candidate_b: six points + times,
  decision_axis: "progress-vs-yield",
  search_scope: "all provider-observable targets",
  max_response_bytes, max_targets
}
```

它不预先给 ROI。邻车必须在自己的可观测范围内检索；请求和响应字节均计费。

### 4.2 P 执行器：只读当前与因果历史

`P_contrast(problem)` 只能读取邻车截至 `t` 的当前框、分数、11 帧 tracker 历史、`history_valid/history_scores/times` 和传感器覆盖元数据；禁止调用 MTR、读取 GT future、未来检测或离线轨迹标签。对每个源内 track：

1. 从有效历史估计当前速度方向和观测残差；短历史保持缺失特征。
2. 用预注册的速度/加速度界构造保守可达包络，分别计算它与 `tau_a/tau_b` 扫掠包络的最小间隔、首次可能相交时刻和间隔差。该包络只用于目标检索，不作为 F 返回，也不称为预测准确度。
3. 一个小型 `rank_P` 读取这些因果几何量、当前框/分数、历史长度及问题 embedding，估计“返回该目标 P 后的规划损失改变量”；按分数选到字节预算为止。
4. 返回所选目标的完整原始 P 字段、用于排序的数值、provider coverage、候选数、截断状态和执行版本。空结果只能写成 `no_counterexample_in_examined_tracks` 或 `coverage_unknown/truncated`，不能写“安全”。

关键严格对照是去掉历史与可达包络，仅按当前框到两轨迹的距离选择。若 `rank_P` 不超过这个动态 corridor 基线，它就是更复杂的 ROI 检索。

### 4.3 F 执行器：远端 MTR 计算与任务回答

`F_contrast(problem)` 让邻车用其完整私有上下文运行冻结 MTR。现实现会为所有当前目标计算 F；即使最后只返回少量目标，也按全量远端计算计费。对每个目标保留六个互斥 mode、原 score、六个 0.5 秒采样时刻和源内 handle，并计算“同一 mode 相对 `tau_a/tau_b` 的有符号接近/暴露差”。mode 身份不跨目标组成联合世界；NMS 后 score 可和小于 1，不能当校准概率或据此声称期望风险。

`rank_F` 读取每 mode 相对两条探针的绝对暴露、差值、原 score、目标锚点及问题 embedding，估计返回该目标 F 的任务边际值；按真实字节预算选目标。响应同时给原始 F 和 mode 对比表，不能只发一个“风险分”。若以后实现仅预测被选目标，仍须保留完整邻居上下文，并单独报告减少的目标 head 计算；当前版本不能预记该收益。

### 4.4 Ego 的初始与返回驱动策略

小型策略头 `pi_q` 的初始状态只含 Ego 已有证据、`tau_a/tau_b`、动作成本和服务目录，输出 `STOP/P/F`。STOP 直接复用初始方案。P/F 返回后，策略读取实际响应中的原始证据、目标/模式对比、覆盖、截断和已付成本，在剩余预算内再输出 `STOP/P/F`；首版第二次只能 STOP 或调用另一能力，并可用第一条返回的 source-local handle 寻址，避免同时扩张同能力多轮搜索。最后共享 GoT 读取所有已购证据生成最终轨迹。

这并不预设 P→F 或 F→P：

- P 首次可能发现一条历史朝向两探针差异区域的远端轨迹，随后对该 handle 请求 F；也可能空答/低值而 STOP。
- F 首次可能显示某目标的 mode 对两条探针具有不同暴露关系，随后请求该目标 P 历史作为上游上下文；P 只能补有损 F 未保留的历史证据，不能揭示未观测意图或消除固有多模态。
- 如果某一工具全局占优，或第一次真实返回不能改变第二动作，删除相应分支或序贯主张。

### 4.5 训练信号

能力模块和共享 GoT 先固定。训练记录上离线执行 `STOP/P/F/P->F/F->P`，保存每一步真实响应、最终轨迹、请求/响应字节、MTR 计算和调用次数。未来 Ego 标签只进入离线 evaluator，形成终态效用 `-trajectory_loss - lambda_bytes*bytes - lambda_compute*compute - lambda_call*calls`，不进入在线特征。

- `rank_P/rank_F`：对每个可返回目标做单目标或贪心集合反事实，标签是把该目标原始证据交给固定接收器/GoT 后相对当前证据的终态效用增量。使用 pairwise ranking，按录制组交叉拟合；训练标签可以看未来 Ego GT，部署排序器只能看 `t` 时可用的 peer 私有证据和问题。
- 第二步 `pi_q`：在每个真实第一响应状态上，以三种允许动作的实测终态效用作 cost-sensitive value regression/ranking 标签。
- 第一步 `pi_q`：先按录制组 cross-fit 第二步 value/策略，在当前首步训练折之外拟合并冻结 continuation；对每个首动作执行真实返回，再让这个冻结且只读合法在线状态的 continuation 选择第二动作，由其实际终态回报监督首步 value。逐帧对所有第二动作取 GT 最优只能作为 hindsight oracle 上界或另列的 imitation 诊断，不能称为可部署策略 value。

不需要先上 RL。若简单 pairwise/监督 value head 已能表达两步有限动作集，再用 GRPO 只会把 DriveAgent-R1 的训练复杂性搬来。策略头若不超过同输入、同参数量的普通 MLP，则论文贡献只能落在 peer 任务接口/执行器；执行器也不超过 matched ROI 后，候选整体失败。

### 4.6 最强简洁反例与可分离验证

只保留四组决定性比较：

1. **问题是否超越 ROI**：单轨迹 corridor、两轨迹并集/差异 mask、TCRE-P、TCRE-F；相同返回字节、相同 target 数和能力计算。
2. **peer 执行是否必要**：一次性 `STOP/P/F` MLP 路由 + full/ROI 返回；同一个路由器 + peer 任务执行器。前者若相同，任务委托无增量。
3. **返回是否改变后续**：同平均总字节/能力次数比较一次性动作、固定 P→F/F→P、实际返回驱动策略；一步 oracle 和两步 oracle 只描述各自动作集上界，不要求后者必然更高。
4. **关系表是否只是数值 adapter**：原始证据文本、原始证据+同容量普通数值 adapter、TCRE 对比表+排序训练。普通 adapter 若相同，不能把对比表示列为贡献。

必须同时报告全帧六点 ADE/FDE、有效输出率、逐录制配对恶化比例、真实请求/响应字节、P/F 能力执行数、MTR 目标数和串行等待。开放环轨迹与几何暴露不能外推为闭环安全。

TCRE 的主要失败条件是：保守候选始终不改变检索；peer 私有信息与 Ego 高度重合；规划器不使用返回证据；P/F 一个动作全局支配；两轨迹 mask 已达到相同前沿；或目标级反事实标签跨录制不稳定。任一情况都应收缩主张，而不是再加更多模块。

## 5. 候选二：条件化远端能力委托（CRCD）

### 5.1 协议

CRCD 复用同一个 `problem`，但 Ego 可以把一个版本化、无循环、最多两次能力执行的有限程序附在首次 P 请求上：

```text
P_contrast(problem, budget_P)
if coverage_ok and selected_targets > 0 and max_task_score >= theta:
    F_contrast(problem, handles=top_handles, budget_F)
else:
    STOP
return every raw response + branch condition + execution/cost ledger
```

邻车先执行 P；条件只读实际 P 结果、coverage/truncation、任务对比分数和请求预算，不读未来标签。触发时再以 P 返回 handle 在同一私有上下文执行 F。Ego 也可直接 STOP、直接 P、直接 F，程序不把所有帧强制成 P-first。程序必须来自允许指令白名单，不能发送任意 Python/LLM 文本；否则难以验证因果输入、成本和执行安全。

即使只有一轮网络请求，只要执行了 P 与 F，账本就是两次能力执行，并分别记录 P 检索、MTR 计算、各段响应字节和总服务时延。返回必须含 P/F 原始证据；只返回 peer 的最终“建议动作”会把正确性完全交给远端，无法判断改善来自证据还是隐藏规划器。

### 5.2 训练、价值与边界

continuation gate 可用 TCRE 的训练记录做监督：输入实际 P 状态和任务问题，标签比较 `STOP` 与对选中 handles 执行 F 的成本后效用。它与交互式 P→Ego→F 使用同一训练标签和执行器，唯一系统差别是分支在 peer 还是 Ego 执行。若 peer 缺少决定续调用所需的 Ego 私有状态，应把允许摘要明确写入初始请求并计费；不能让 peer 暗读 Ego 全场景。

CRCD 最强反例是“一次固定 PF bundle”：peer 总是一次返回同预算 P+F；另一个反例是“一次 learned bundle”，peer 看同一 `problem` 后直接选最有用的 P/F 组合。还需与普通两轮 P→Ego→F、直接 P/F、去掉 `problem` 的同程序比较。LLMCompiler 已覆盖带依赖的函数执行、以中间输出替换下游变量和简单 if-else 分支；传统 function shipping 已覆盖按通信/计算成本选择执行位置。因而即便这些对照中 CRCD 有 RTT 优势，也只能说明本驾驶协议的部署取舍，不能声称条件程序或远端编排的新颖性。

若 CRCD 只减少 request header 或网络 RTT，而目标选择、质量和能力成本与 fixed/learned bundle 相同，它是部署优化。若未建立可信到达时延模型，当前冻结快照只能报告模拟请求轮数和字节，不能宣称实时收益。它只有在任务条件令 peer 在不同样本选择不同目标/是否执行 F，并在同能力成本下超过 learned bundle 时，才可与 TCRE 合为“可部署的条件能力委托”贡献。单独的有限状态 RPC、batching 和 function shipping 不新，本文未检到足以支持“首创”的证据。

## 6. 两个候选如何取舍

| 判断项 | TCRE | CRCD |
|---|---|---|
| 是否直接服从 ToolV2X 主线 | 是：初始与第二步都为 STOP/P/F | 是：直接 P/F 保留，P 可带受限 continuation |
| peer 是否真正执行驾驶任务 | 是：按方案对比从私有上下文检索并计算 | 是，但任务执行器来自 TCRE |
| 相对近邻的潜在辨识度 | 中等；必须胜过规划 ROI、CAPO/TIP 排序和 AFA 路由 | 低；LLMCompiler 和 function shipping 已直接覆盖条件编排/计算位置原则 |
| 首版工程改动 | 请求 schema、P/F target ranker、策略头、证据/成本账本 | 另加有限程序解释器和单往返封装 |
| 推荐定位 | 主候选 | 部署变体兼强基线，不单列第一贡献 |

因此建议下一设计轮只围绕 TCRE 形成最小原型规格，把 CRCD 保留为同一执行器的部署比较。TCRE 也不应直接写成已成立创新：先离线生成匹配对照所需的反事实响应和目标标签，检查两轨迹 mask、普通 MLP 和 learned one-shot bundle 是否已经解释全部收益。通过以后，论文贡献才可收为“驾驶问题参数化的远端 P/F 执行器”与“返回条件化的能力价值策略”两点；不通过则回到简单一次 STOP/P/F 路由。

## 7. 查阅状态与链接层级

- **项目文件，已读**：`docs/user_requirements.md` 顶部主线修订、`start.md` 原始 Tool 定义、`docs/framework_design.md` 完整设计；当前 P/F 和 episode 源码逐行检查。旧文档是设计/候选，不当作实验结果。
- **正式出版页已核**：EDDI（ICML 2019 PMLR）、TIP（ICML 2023 PMLR）、LLMCompiler（ICML 2024 PMLR）、CAPO（作者页标 ICRA 2022）、Select2Drive DOI（IEEE T-ITS 2025）。
- **作者预印本正文已核**：DriveAgent-R1 v3（arXiv 作者标注 ICLR 2026 accepted）、UNCAP v2（arXiv journal reference AAMAS 2026）、TOCOM-V2I、Collaborative Edge-to-Server Inference v2 和 To Ship or Not to (Function) Ship。TOCOM-V2I 与 Edge-to-Server 未从正式会议目录核到录用，不升级表述。
- **本次没有做**：没有浏览或 clone GitHub/Hugging Face 资源，没有核称任何代码完整复现，没有运行模型/训练/数据生成，也没有修改接口或训练文件。
