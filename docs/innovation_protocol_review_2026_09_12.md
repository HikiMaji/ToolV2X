# ToolV2X 查询协议创新候选

查阅日期：2026-09-12。本文只提出可证伪的研究候选，不保证文献穷尽、新颖性、CCF 分级或录用。

## 结论

优先研究一个协议：**预测寻址的因果澄清（Forecast-Addressed Causal Clarification, FACC）**。第一次调用让远端基于其完整因果上下文运行冻结 MTR，返回可寻址的多模态预测目录 `F_cat`；接收端用少量因果运动学探查轨迹、自己的本地目标证据与实际 F，找出“同一远端模态会改变两个 ego 候选之间比较”的目标；只有这时才按 `track_id` 请求该目标及其交互邻域的原始 1 秒历史 `P_id`。`F_cat` 是计算委托结果，`P_id` 是当前 P 的按目标/邻域扩展，返回第一次响应中没有的源观测。

这不是把现有 P→F 倒过来。现有 full P 到达后会在 ego 端运行同一 MTR，已经生成与 full F 近等价的未来（`src/tools/vehicle.py:162-191`）；因此 P→F 往往重复。FACC 的第二问不再请求预测，而是检查 MTR 有损输出是否遗漏了当前规划需要的历史/交互证据。若 full F 经普通同信息数值适配器已达到同样结果，或当前配置下 `F+P_id` 对 F 没有稳定增益，就没有补证机制的实验证据；这不构成“F 在所有任务上严格充分”的一般结论。

“紧凑 F”不能作为前提或创新点。当前 F 每个对象含当前 box、6 条模态、6 个时间点、原始 mode scores 和 `model_used`（`src/tools/vehicle.py:34-47`）；其实际 JSON 可能比 P 大，也未证明更准。第一阶段先使用当前 F 验证信息假设，再决定是否压缩。规划条件 token 压缩已有 COMPACT-VA，按质量反馈增量发送已有 progressive semantic communication；只把 F 变短不足以区分。[^1][^2]

第二个较弱候选是**目录后定向委托（Inventory-to-Targeted Forecast, I2TF）**：第一次只取对象目录，第二次请求选定目标的 F。它能消除 full P→F 重复，但很容易被“一次发送 ego 轨迹，由远端直接选目标并返回 F”编译成单轮。除非实验证明接收端私有证据和实际目录值使交互不可替代，否则只应作为 FACC 的简洁反例或工程协议。

## 先把信息获取与计算委托分开

| 操作 | 返回内容 | 新信息来自哪里 | 计算发生在哪里 | 科学含义 |
|---|---|---|---|---|
| 当前 P | ROI 内所有目标的 11 帧状态/tracker 可用位/时序分数 | peer 的因果观测历史 | 随后 ego 再跑 MTR | 信息获取 + 接收端计算，当前实现把两者捆在一起 |
| 当前 F | peer 完整上下文 MTR 的逐目标多模态未来 | 原始信息仍来自 peer 历史，消息是模型变换结果 | peer | 计算委托；不自动比 P 小或准 |
| 扩展 `P_id(T)` | 选定目标 `T` 及固定交互邻域的原始历史，带 target/support 角色 | F 未携带的历史轨迹、tracker 可用位和时序分数 | 主实验不在 ego 重跑 MTR | 纯信息补充，用来检验当前 F 表示是否遗漏规划相关证据 |
| 新 `F_T` | 对目标集合 `T` 的远端预测 | peer 私有完整上下文经 MTR 的结果 | peer | 定向计算委托；若仍全帧推理，只省响应字节，不省 peer 计算 |

当前服务会缓存并全帧计算 F，再按 ROI 过滤输出（`src/tools/vehicle.py:58-102`）。所以“只返回几个目标”目前只改变通信量；若以后做 target batching，必须单独验证它没有改变上下文和预测。再次请求已缓存 F 既没有新观测也没有新计算，应作为零增量控制，而不是第二阶段能力。

## 候选一：FACC

### 协议

**阶段 0：因果运动学探查。** 用时刻 `t` 的 `speed_mps/yaw_rate_rps` 构造两条预注册参考：保持当前运动的 `τ_keep` 与有界减速的 `τ_slow`。它们只是比较坐标，不是最终可执行计划，也不声称满足地图/车道规则。当前输入只有这两个运动量与避碰问题（`src/planning/inputs.py:165-216`），所以不能虚构导航意图。这样查询阶段只增加少量几何运算，最终 GoT 仍生成一次；若以后改用 GoT 草案，必须计入额外推理并与同预算方案比较。

**调用 1：`F_cat(roi)`。** peer 在完整 causal window 上运行冻结 MTR，返回当前 F 的 `track_id/box/forecast/forecast_scores/model_used`。G0 用 full ROI，避免先把压缩误差混入协议价值。接收端为每个目标保留六个模态对 `τ_keep/τ_slow` 的关系、原始 score 和绝对暴露；MTR scores 是筛选后的模式分数，六项和可以小于 1，不应重归一化后冒充完整校准概率。不同目标的第 `k` 模态也不是同一个联合世界假设，不能跨目标拼成 joint future。

**澄清寻址。** 对目标 `i`、模式 `k` 和 ego 候选 `τ`，计算有符号最小间隔

`d(i,k,τ) = min_t distance(forecast(i,k,t), τ(t)) - safety_radius(i,t)`。

若某目标的模式跨越零间隔，或同一模式对两条探查轨迹给出相反的相对暴露排序，它就是候选澄清目标。选择器还读取 ego 已有的目标/预测证据，以排除本车已经充分覆盖的对象；这些本地证据不发送给 peer。用下游轨迹损失变化监督“购入哪段历史”沿用 TIP/CAPO 类任务价值原则，不能另算创新；协议的新假设只在于实际 F 产生可寻址目标后，购买一种先前未交付的同源字段。[^13][^14]

**调用 2：`P_id(track_ids, support_radius_m, max_support, probe_reason)` 或 STOP。** 响应复用 P 字段：所选目标的 11 帧 `history/history_valid/history_scores/history_times`，以及 peer window 中固定半径内、按当前距离截断的 support tracks；每行标明 `target` 或 `support`。`history_valid` 只表示 tracker 状态可用，包含漏检时由因果跟踪外推维持的状态，不是逐帧 detection-match 位（`src/tools/vehicle.py:1-4`）。`probe_reason` 只是写入审计日志的操作理由/待评估假设，可取 `history_dynamics_probe`、`observation_provenance_probe` 或 `interaction_context_probe`；它不作为 GoT 证据，也不是在线已知的真实信息损失原因。support 由可复现几何规则定义，不声称是 MTR 的真实注意力因果解释。最终 GoT 同时读取 F 和这批原始历史，一次输出 `τ_FP`。主实验不对 `P_id` 再跑 MTR。

### 额外信息究竟是什么

F 只有当前 anchor 与预测路径，没有目标过去 1 秒的状态变化、tracker 可用位、时序分数，也没有形成预测时可见的邻车历史。`P_id` 补充这些字段。full P→F 是由历史到预测的确定映射，这说明 F 来源于 P，却不说明可以由 F 逆推出 P；原始历史可能包含被 MTR 压掉的信息。FACC 检验的只是这部分表示损失是否影响当前 GoT。它**不能**声称同一段已观测历史会揭示尚未发生的驾驶意图，也不能把 MTR 固有的未来多解性包装成可由 P 消除的不确定性。若新增字段没有改善，只能说当前配置未观察到补证价值。

当前 P 会返回 ROI 内全部历史，再由 ego MTR 转成预测；FACC 需要显式接口差异：

- 请求从只有 `tool=P, roi` 扩为 `tool=P, track_ids, support_radius_m, max_support, probe_reason`；当前严格 packet schema 不接受这些字段（`src/tools/vehicle.py:12-19,105-145`）。
- 响应从 ROI 全量 P 改为目标可寻址的历史子集，并加入 target/support 角色。
- F 与 P 必按同一 `(provider, scene, g, track_id)` 合并：`P_id` 为已有 F 对象增加 `history/provenance/support`，不能再追加一个被规划器当作独立交通参与者的 P 对象。若做本地 MTR parity，其输出也只能标成同源替代/对照，不能作为第二份独立未来叠加。
- 当前 P-local 合并只保留 history、valid 和 times，未把 `history_scores` 加回规划证据（`src/tools/vehicle.py:173-180`）；若要检验观测来源稳定性，`P_id` 路径必须显式保留该字段，并给所有基线相同接收容量。
- 当前 `choose_action` 只看速度、P 对象和几何关系（`src/planning/episode.py:8-33`）。FACC 需要在 F 返回后运行轻量几何探查再决定 P；最终驾驶生成仍保持一次。现有 episode 循环可容纳返回后动作，但 packet 与证据合并语义需要调整（`src/planning/episode.py:36-118`）。
- 当前 context 按距离截断远端对象（`src/planning/context.py:18-36,52-67`）；FACC 的被请求目标须保留，support 才能按预算截断，否则查询协议会被接收端再次静默改写。

### 为什么普通一次选择器可能不够

在第一次响应前，ego 不知道 peer-only `track_id`、遮挡目标或它们的多模态未来，因而不能直接提出同一个目标级澄清问题。第二问的地址和值由实际 F、探查轨迹和 receiver-only 本地目标共同产生；这比从 `s0` 选择 Ego/P/F/PF 多了一类“先发现远端候选，再向其中一个追问源证据”的动作。

但这只是待证假设。最强的单轮反例是 peer-side bundle：ego 把探查参数及必要本地摘要发给 peer，peer 运行 F、选目标并把 F 与相应历史一次返回。另一个反例是 full F 加小数值 adapter；如果文本容量而非信息缺失才是瓶颈，它会解决大部分增益。还必须加入 F 自身的等字节细化：增加目标、增加保留模式，或把每模态的 6 个采样时刻扩到 MTR 原有的最多 50 个时刻。FACC 必须在相同总字节和总计算下超过这些反例，且同一初始 `s0` 在不同真实 F 返回下会请求不同 `track_id` 或停止；否则交互可被编译掉。

### 四个可分离对照

1. **P 字段是否补证：** F；F + 同信息数值 adapter；F + full P raw history（不重跑 MTR）；F 的等字节目标/模式/时间细化。先排除表示容量与“更多 F”解释。
2. **寻址是否任务相关：** 在相同 `P_id` 对象数和字节下，比较规划歧义目标、最近目标、最大 MTR 模态分散目标和随机目标；所有方法只运行一次最终 GoT。
3. **价值来自哪种信息：** `F+P_target_history`、`F+P_target+support`、`F+shuffled P`。这分开目标自身动态、交互上下文和仅增加 token 的效应。
4. **交互是否不可编译：** 同预算比较单轮 peer-side bundle、单轮 full PF 与 FACC；bundle 请求须计入为复现 receiver 评分而发送的探查参数/本地摘要。同时报告 F 后 STOP 比例和返回值条件化地址变化。oracle 只给允许动作集上界。

主张成立至少需要：`F+P_id` 对 F 有配对质量增益；定向 P 在同字节下胜过简单寻址和 F 自身细化；FACC 在计入握手与几何探查后仍有 Pareto 优势；增益在未见录制组上存在，而不只是 token 截断偶然变化。任何一项失败，都应缩回 F + 数值 adapter 或单轮方案。

## 候选二：I2TF（不进入当前主线）

**调用 1：`I()`。** peer 只返回对象目录：`track_id`、当前 box、当前 score、coverage；可选一阶速度必须明确来自历史差分，但不返回完整 11 帧历史。它不是当前 P，需要新名称，避免把接口差异藏在同一字母下。

**调用 2：`F_T(track_ids)`。** ego 用探查轨迹与实际目录选出相交/遮挡目标；peer 仍以完整私有 window 运行 MTR，只返回这些目标的多模态未来。新增信息来自目录未公开的历史和交互上下文，经 peer MTR 转成 F；这是计算委托，不是第二次原始信息获取。

普通一次 STOP/P/F 不能预先指定未知 `track_id`，这是它的表面优势。然而 peer 已拥有目录和 F，ego 也可以在第一次请求中发送探查参数或 corridor，让 peer 直接返回风险目标 F。Who2com 已用 request–match–connect 握手选择通信方，Where2comm 已用请求图做后轮空间选择；仅把选择单位改成目标还不够。[^3][^4]

I2TF 当前不进入实现范围，只用于检查“先发现地址再委托”是否真的需要两轮。最低思维对照是 full F、单轮 `F_ROI(τ_probe)`/peer-side target selection 与 `I→F_T`。当前 F 先全帧计算再过滤，I2TF 也不能声称节省 MTR 计算。

## 与直接近邻的边界

- **Select2Drive** 已用先前规划路点和检测置信裁剪通信 ROI，并建模带宽/时延；FACC 的区别必须落在“F 返回值产生目标地址，随后跨越预测瓶颈获取原始因果字段”，而不是 planning-aware ROI。[^5]
- **UNCAP** 已有启发式选车、带不确定性的语言消息和规划互信息筛选；因此车辆选择、语言、不确定性或 MI 都不能单独作为创新。FACC 使用下游轨迹反事实价值，并显式区分源观测与远端计算结果。[^6]
- **DriveAgent-R1** 已有返回证据后的多次视觉工具调用、最多三次调用、工具/文本模式选择和成本奖励；“Agent 会追问”已被覆盖。它没有建立 V2X 中同源信息的 P/F 计算位置对照，也不研究预测瓶颈后的目标级原始证据请求。[^7]
- **Collaborative Edge-to-Server Inference for VLMs** 先传低分辨率全图，再用输出 min-entropy 与注意力定位请求高清 ROI，是 F→细节的最近协议骨架；论文也承认第二阶段可能使正确答案变差。FACC 不能复用“高不确定就补细节”的叙事，必须以目标级规划关系寻址，并用反事实轨迹价值验证。该文当前自述为 GLOBECOM 2026 在审的前期版本。[^8]
- **Progressive Semantic Communication** 将有序 latent 分块并在质量不足时只补下一块，优势是增量内容复用；其反馈决定传输层级，而非根据第一次语义值改变下一字段的身份。FACC 传的是异质消息类型 F→原始历史，而非同一表示的前缀细化。[^2]
- **EDDI / Learning to Acquire Information** 已覆盖按已有答案动态选择下一缺失变量、互信息/条件熵和成本；“active acquisition”本身不新。ToolV2X 需要把可购变量具体化为远端目标历史，把目标量定为轨迹损失，并证明 receiver-private returned-value addressing。[^9][^10]
- **TIP / CAPO** 已分别用候选效用比较选择有限预测样本、用预测对控制的影响训练规划相关预测；“比较两个 ego 候选的效用差”不能算新点。这里的关系只用于在 F 返回后给 `P_id` 寻址。[^13][^14]
- **R2T** 已将 sender scene、估计的 receiver information gap 与预算送入小 Transformer 做区域发送决策；其 v1 仅在合成 BEV 感知中验证，且低带宽选择器差距很小、融合贡献更大。这使“多信号轻量 reasoning selector”尤其不能成为主张。[^11]
- **COMPACT-VA / Defer to Plan** 分别覆盖 planning-aligned 历史 token 压缩和规划阶段自适应多车融合。FACC 不应声称首次规划条件表示；它研究的是哪一种尚未传输的源证据值得在预测之后购买。[^1][^12]

## 推荐的论文创新点写法

若上述门槛通过，可收成三个相互依赖的点，而不是三个松散模块：

1. **类型化的非重复协议：** 将远端预测定义为计算委托，将目标/交互历史定义为可购买源证据；第二次响应严格来自第一次消息遗漏的字段。
2. **返回值寻址的规划澄清：** 先由 F 暴露远端目标和行为模态，再用因果探查轨迹与 receiver-only 本地证据选择 `P_id` 的 `track_id` 与 support 范围；监督信号是澄清后的配对轨迹损失下降减实际成本。
3. **可编译性检验：** 把单轮 peer-side bundle 和同信息数值 adapter 当作一等基线，只有交互在同成本下仍优时才主张协议必要性。

可用的主句是：“我们研究远端预测何时不是下游规划的充分统计量，并提出一种预测寻址的因果澄清协议，使接收车只为会改变其计划关系的目标购买原始历史证据。”不要写“首次渐进语义通信”“首次任务条件化查询”或“F 更紧凑/更准确”。

## 主来源与查阅状态

| 工作 | 主来源 | 本次用途与状态 |
|---|---|---|
| COMPACT-VA | [arXiv:2606.07464v3](https://arxiv.org/abs/2606.07464) | 网页正文已读；规划条件 token 压缩近邻；arXiv 作者备注称 RA-L 2026 accepted |
| Progressive Semantic Communication | [arXiv:2604.26508v2](https://arxiv.org/abs/2604.26508) | 网页正文已读；反馈式增量 latent 近邻；arXiv 作者备注称 GLOBECOM 2026 accepted |
| Who2com | [ICRA 2020 DOI](https://doi.org/10.1109/ICRA40945.2020.9197364)；[arXiv](https://arxiv.org/abs/2003.09575) | 本地正文已读；request–match–connect 正式近邻 |
| Where2comm | [NeurIPS 2022 官方页](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html) | 本地正文已读；多轮空间请求正式近邻 |
| Select2Drive | [IEEE DOI](https://doi.org/10.1109/TITS.2025.3611377) | 本地正文已读；规划 ROI、带宽/时延正式近邻 |
| UNCAP | [AAMAS 2026 正式论文](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WHRE3513.pdf) | 正式正文/项目页已核；车辆、语言、规划 MI 近邻 |
| DriveAgent-R1 | [arXiv:2507.20879v3](https://arxiv.org/abs/2507.20879) | 本地正文已读；证据后工具调用近邻；arXiv 作者备注称 ICLR 2026 accepted |
| Collaborative Edge-to-Server Inference for VLMs | [arXiv:2512.16349](https://arxiv.org/abs/2512.16349) | 网页正文已读；粗图→熵/注意力→细 ROI 的强协议近邻；文中称 GLOBECOM 2026 在审 |
| EDDI | [ICML 2019 / PMLR](https://proceedings.mlr.press/v97/ma19c.html) | 本地正文已读；逐实例高价值变量获取正式近邻 |
| Learning to Acquire Information | [arXiv:1704.06131](https://arxiv.org/abs/1704.06131) | 本地正文已读；结果条件化的下一观测选择 |
| R2T | [arXiv:2603.20308v1](https://arxiv.org/abs/2603.20308) | arXiv HTML 已读；receiver gap+预算选择近邻；仅预印本、合成 BEV |
| Defer to Plan | [arXiv:2607.19774v1](https://arxiv.org/abs/2607.19774) | 本地正文已读；规划阶段自适应融合近邻；arXiv 作者备注称 ICME 2026 accepted |
| TIP | [arXiv:2110.08750](https://arxiv.org/abs/2110.08750) | 候选效用比较与任务相关预测近邻；IROS 2022 |
| CAPO | [arXiv:2204.13319](https://arxiv.org/abs/2204.13319) | 预测对控制影响的规划相关训练近邻；ICRA 2022 |

[^1]: [COMPACT-VA, arXiv:2606.07464v3](https://arxiv.org/abs/2606.07464)，规划意图条件的长历史 token 压缩；作者备注称 RA-L 2026 accepted。
[^2]: [Progressive Semantic Communication, arXiv:2604.26508v2](https://arxiv.org/abs/2604.26508)，有序 latent 前缀与反馈增量传输；作者备注称 GLOBECOM 2026 accepted。
[^3]: [Who2com, ICRA 2020](https://doi.org/10.1109/ICRA40945.2020.9197364)；[arXiv:2003.09575](https://arxiv.org/abs/2003.09575)。本地正文 `docs/papers/text/12_who2com.txt:43-77,147-191`。
[^4]: [Where2comm, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)。本地正文 `docs/papers/text/13_where2comm.txt:185-376`。
[^5]: [Select2Drive, IEEE T-ITS](https://doi.org/10.1109/TITS.2025.3611377)。本地正文 `docs/papers/text/02_select2drive.txt:477-785`。
[^6]: [UNCAP, AAMAS 2026](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WHRE3513.pdf)，DOI `10.65109/WHRE3513`。
[^7]: [DriveAgent-R1, arXiv:2507.20879v3](https://arxiv.org/abs/2507.20879)，作者备注称 ICLR 2026 accepted。本地正文 `docs/papers/text/18_driveagent_r1.txt:192-229,256-372,437-454`。
[^8]: [Collaborative Edge-to-Server Inference for Vision-Language Models, arXiv:2512.16349](https://arxiv.org/abs/2512.16349)。
[^9]: [EDDI, ICML 2019](https://proceedings.mlr.press/v97/ma19c.html)。本地正文 `docs/papers/text/03_eddi.txt:26-95,137-186`。
[^10]: [Learning to Acquire Information, arXiv:1704.06131](https://arxiv.org/abs/1704.06131)。本地正文 `docs/papers/text/10_learning_to_acquire_information.txt:20-105,120-241`。
[^11]: [Reason-to-Transmit, arXiv:2603.20308v1](https://arxiv.org/abs/2603.20308)。
[^12]: [Defer to Plan, arXiv:2607.19774v1](https://arxiv.org/abs/2607.19774)，作者备注称 ICME 2026 accepted。本地正文 `docs/papers/text/04_defer_to_plan.txt:64-96,106-175`。
[^13]: [TIP: Task-Informed Motion Prediction for Intelligent Vehicles, arXiv:2110.08750](https://arxiv.org/abs/2110.08750)，IROS 2022。
[^14]: [Control-Aware Prediction Objectives for Autonomous Driving (CAPO), arXiv:2204.13319](https://arxiv.org/abs/2204.13319)，ICRA 2022。
