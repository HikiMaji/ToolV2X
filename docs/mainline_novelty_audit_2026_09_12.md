# ToolV2X 主线创新反方审计：规划条件化的远端能力调用

日期：2026-09-12。范围限于两车、时间 (t) 可得输入、最多两次 P/F 调用和 3 秒 6 点自车开放环轨迹；不讨论 RSU、世界模型、闭环工程或安全保证。本报告核查公开论文正文和公开代码网页，没有下载仓库、运行模型、修改训练配置或执行实验。

## 结论先行

**宽泛表述已不能成立。** “规划感知通信”“根据候选决策差异请求信息”“工具返回后再决定是否继续”“预算下非短视地获取可解锁后续价值的证据”“ego 发请求、peer 用私有观测计算后返回”均已有直接先例。最致命的三组近邻分别是：

1. MATE 已让 ego 广播候选轨迹，由 peer 用自己的私有代价图和不确定性图逐候选计算并回传；
2. DriveAgent-R1 已在驾驶任务中按当前历史调用多种工具、把结果写回历史、继续调用或停止，并学习工具模式的收益；
3. BRiG-AFA、NM-PPG 和 AFABench/CUBE-NM 已覆盖预算条件的非短视价值、STOP/异质成本及“先获取一个本身无即时收益、但能决定下一次查询的上下文”的标准机制。

仍值得做的窄机制是：**双端私有状态下、由规划差异参数化的异质远端能力作业及结果依赖组合**。其区别不在“选 P 还是 F”，而在于请求参数实际改变 peer 端执行的任务；第一次返回在 ego 与其本地规划证据融合后，产生新的第二次作业或 STOP。初始动作仍可为 STOP、P 或 F，后续允许 P→F、F→P。这个交叉点在本次检索的工作中没有被一个方法完整覆盖，但它是待验证的组合创新，不能宣称为某个基础概念的首次提出。若继续冻结现有 MTR，ego 候选轨迹并不会改变 F 的预测分布。请求可以选择 MTR 中心对象，但该中心仍输出 50 个未来时刻；所谓请求“时间段”只能整理返回字段，不能声称减少 decoder 计算。这必须如实表述为 planning-conditioned invocation，而不是 ego-conditioned forecasting。

当前代码还没有这个性质：[vehicle.py](../src/tools/vehicle.py:58) 的请求只有 `tool` 和 `roi`，F 对完整 peer 场景预测一次后才按 ROI 过滤输出；[episode.py](../src/planning/episode.py:11) 只有固定序列或手工规则。缓存全量预测再筛选与按需中心计算在预测相同的对象上给出相同任务质量，是应有的正确性结果；区别要由包含预计算、缓存命中、存储、在线计算和通信的完整成本说明。若缓存本来就为其他任务生成且实验时真实可用，应按其真实边际成本计量，不能为 warm cache 虚构一次额外推理收费。

## 八组最直接近邻及否决范围

| 近邻 | 正文已经做了什么 | 对 ToolV2X 的致命重合 | 尚未覆盖的窄空隙 |
|---|---|---|---|
| [Best et al., Planning-Aware Communication, ICRA 2018](https://ieeexplore.ieee.org/document/8460617/) | 预测滑动时域内消息的信息价值；请求其他机器人的计划分布；用粒子滤波预测价值，再在 belief space 中安排通信时刻。 | planning-aware request、长于一步的信息价值和通信调度都不是新概念。 | 消息是计划分布，未在隐藏 peer 状态上选择并执行 P/F 两种作业。 |
| [MATE, ICRA 2023](https://arxiv.org/html/2303.06080) | ego 广播多个候选轨迹；peer 变换到本地坐标，用私有预测得到的 costmap/entropy map 对候选采样并回传，ego 融合并选最低代价轨迹。 | “ego 候选驱动 peer 私有信息上的规划相关计算”已经被直接实现。 | MATE 的远端服务固定为候选评分，没有在 P/F 异质服务间进行返回结果条件化的第二次调用。 |
| [DriveAgent-R1, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbb776e737ec3ea5925887f8740c68b4-Abstract-Conference.html) | 由当前图文历史选择文本或工具模式；工具可检索视图/历史、ROI、深度和 3D 检测；结果写回历史，最多多轮调用；工具奖励基于相对文本模式的任务增益。 | 驾驶中的主动工具、多工具、结果条件化继续/停止和收益减成本都已存在。 | 工具主要访问本车视觉资源；没有两车各持私有状态的远端 P/F 执行契约。官方仓库已公开训练与多轮框架，但评测脚本、指定数据和权重仍列为待发布：[代码](https://github.com/wczheng/DriveAgent-R1)。 |
| [TIP, IROS 2022](https://arxiv.org/html/2110.08750v2) 与 [CAPO, ICRA 2022](https://rowanmcallister.github.io/publication/capo/capo.pdf) | TIP 把候选决策集和任务效用写入预测训练，并令预测条件化于 ego 计划；CAPO 用控制变化或 attention 给不同对象的预测误差加权，无需对 planner 反传。 | 候选计划效用、两候选相对差异、planning-critical target 及控制敏感度均不能单列创新。 | 它们优化预测内容/损失，没有决定在哪个 peer 上按什么参数执行哪项能力。 |
| [BRiG-AFA, arXiv 2026-08](https://arxiv.org/html/2608.02305) | 针对每个剩余预算拟合 candidate-conditioned Bellman risk-to-go；部署只用已见值、mask、候选和预算。CUBE-NM 中弱 gate 的价值只在后续查询显现。 | “第一步解锁第二步”“按剩余预算监督学习终局风险”已经有非常直接的方法。 | 获取动作揭示预先存在的固定 feature value；不会把 query 发送给另一个 agent 并改变其运行的函数。公开代码：[BRiG-AFA](https://github.com/JIAORONG-FENG/BRiG-AFA)。 |
| [NM-PPG, arXiv 2026-05](https://arxiv.org/html/2605.05511) | 把 AFA 写成带 acquire/STOP、异质成本与终局预测损失的 POMDP；通过连续松弛和 straight-through hard rollout 对整条采集轨迹做 pathwise gradient。 | 非短视、STOP、全轨迹信用分配和异质成本都不是可单列的新点。 | 同样把动作视为读取固定特征，不含 query-conditioned peer computation。arXiv 页面未给出可核验的官方代码链接。 |
| [AFABench, KDD 2026](https://arxiv.org/html/2508.14734) | 统一静态、贪心 CMI、RL 和非 RL 非短视 AFA；CUBE-NM 专门令一个无即时标签信息的 context feature 指向后续有效 feature block；还强调 hard/soft budget、共享 predictor 和公共组件的公平性。 | “上下文先行、然后定向取证”已是公开基准结构，不能用 P/F 改名后声称新机制。 | 分类型固定特征基准不建模远端函数、执行成本和双端私有状态。官方代码与 KDD 条目见 [AFA-Benchmark](https://github.com/Linusaronsson/AFA-Benchmark)。 |
| [Select2Drive, T-ITS 2025](https://doi.org/10.1109/TITS.2025.3611377)，并对照 [Who2com](https://arxiv.org/html/2003.09575)、[Where2comm](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Paper-Conference.pdf) | Select2Drive 用上一时刻 ego 规划生成请求图，并结合预测感知和关键区域打包；Who2com 用 request-key handshake 选 peer；Where2comm 用 request/confidence map 做多轮、稀疏区域通信。 | planning-conditioned ROI、请求握手、who/where 选择、稀疏/多轮通信及质量-带宽权衡都已有。 | 这些方法主要选择 peer/区域/特征，不在 P/F 能力语义及其依赖上做两步远端作业规划。Where2comm 代码：[官方仓库](https://github.com/MediaBrain-SJTU/where2comm)。 |

补充边界：[COMPACT-VA v3](https://arxiv.org/abs/2606.07464v3) 已明确被 IEEE RA-L 2026 接收，并以规划意图训练长历史 token 压缩，所以“规划相关的历史压缩”只能是表示组件。关系型特征获取的 [FAIR](https://www.ml.tu-darmstadt.de/papers/ramanan2023codscomad.pdf) 是训练期向人询问关系属性，并非在线远端驾驶能力执行；它提示 target/group 层级查询并不天然新颖。理性元推理早已把计算当作有成本的动作并按其决策价值选择，见 [Selecting Computations, UAI 2012](https://aima.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf)；因此“value of computation”也只能作为理论背景。[LLMCompiler, ICML 2024](https://www.stat.berkeley.edu/~mmahoney/pubs/icml2024_kim24y.pdf) §3.4 还明确讨论了把简单 if-else 分支静态编译，以及根据中间结果重新规划工具图；所以“一轮内在 peer 端执行条件程序”与“结果后重规划”也不能单列为算法创新。

## 建议保留的具体机制

### 核心：双端私有状态上的规划条件化远端作业

第一步，ego 从当前本地证据和 3 秒候选轨迹中形成一个最小请求：争议的候选轨迹、目标或区域、时间段及所需能力类型。它可直接 STOP，也可首先请求 P 或 F；不预设顺序。

第二步，peer 用自己的完整时间 (t) 私有上下文执行指定作业。P 返回请求对象的 1 秒可用 tracker 状态历史或在请求区域发现的对象；F 可以按请求选择 MTR 中心对象，但现有 decoder 对该中心仍输出全部 50 个未来时刻，时间段参数只控制返回整理，不能计为 decoder 算量节省。按需执行和已有全量缓存都是合法实现，必须在各自真实可用条件下比较完整成本。冻结 MTR 时，不得写成候选轨迹改变了预测模式；若未来另加 query-conditioned prediction head，则 TIP 将成为必须击败的直接基线。

第三步，ego 把第一次响应和未传给 peer 的本地规划证据融合，重新计算仍未解决的候选差异，再生成第二个 P/F 作业或 STOP。P 可以发现此前 ego 不知道的 source-local `target_id`，从而实例化 F(target)；F 也可以暴露争议目标，再实例化 P(target)。这只是在线动作可用性的操作定义，不声称找到了“真实误差原因”。

第四步，用所有合法的一步和两步程序的离线结果训练一个轻量值函数，目标是终局开放环轨迹损失加实际请求字节、响应字节和两端计算成本。Bellman/value head 本身采用现有方法即可，不能作为创新；研究点放在带参数远端作业及两端反馈闭环是否产生可测增量。

这可以组织为“一主两支撑”：主机制是双端反馈闭环中的远端能力执行；支撑一是 typed capability registry，记录 P/F 参数、前置对象身份和动态可用作业；支撑二是 source/time/ancestor ledger，防止把同源 P 与其派生 F 当独立证据。后两项是实现正确性和归因条件，不单独声称创新。

现阶段所谓“两条 ego 候选轨迹”只能作为查询构造探针：项目尚无真实候选规划器及候选效用评分，不能据此声称找到了决策歧义、实现了候选排序翻转或验证了可行性。比较 `F_partial→F_full` 可诊断完整 peer 上下文的委托价值，比较 `F_full→F_full+P` 可诊断 F 有损后历史的增量；二者都是能力价值训练的支撑证据，分别近似上下文扩充与互补特征获取，不能作为主创新。

收敛后的具体实例可改为**驾驶修订驱动再请求**：ego 先由当前完整私有输入得到 `τ0` 并选择 STOP/P/F；收到第一次响应 `r1` 后，把它与未发送的 ego 私有证据融合，再由同一 GoT 生成 `τ1`。`τ0` 与 `τ1` 仅用于提出新增或变化的交互关系，由此参数化第二次 P/F 或 STOP，peer 返回对应原始证据；动作价值由最终实际任务损失和真实成本监督，不能用轨迹改动量代替。这是待验证的 plan-revision-driven requery 归纳偏置，不是已确立的新概念。两项关键风险是：`τ1` 单独已经足以定位请求，使差分没有增量；以及 `τ0→τ1` 只是 GoT 解码噪声或错误修订。应固定解码随机性，并比较 `τ1-only`、旧计划不变、差异 mask 与同-input MLP。

## 信息方向与必须保留的诊断

- peer 的完整 F 是其历史和周围上下文经 MTR 的有损函数，因而 `F → P` 仍可能返回 F 中不存在的历史、目标和上下文；不能由 F 反推 P 无增量。
- 若 P 只返回某目标 1 秒历史，它没有传输 peer MTR 使用的全部周围上下文，因此 `P_target → 本地 MTR` 与远端 F 不等信息。
- 只有 `P_full_context` 精确传输远端 F 的全部时间 (t) 输入时，相同冻结 MTR 的本地重算才应与远端 F 内容一致；此时差异应归到通信与计算位置，而不是预测能力。
- `history_valid` 表示 tracker 状态可用，包括预测维持的状态，不是逐帧检测命中位。
- MTR 的六个 NMS 后原始 score 未校准且和可小于 1，只能作为路由诊断特征，不能解释为概率质量或直接据此计算概率期望后悔。

## 可直接证伪的主张与强基线

建议只预注册一个核心主张：

> 在相同因果输入与匹配的完整通信/计算/缓存成本下，规划修订参数化、第一次返回后由 ego 再决定第二项远端作业的策略，在 held-out 录制组上改善 3 秒 6 点开放环轨迹任务损失；增量不能由固定 P/F、同输入普通 AFA、真实可用的缓存选择、或一次 peer-side 打包解释。

四个最强对照足够：

1. **固定与普通选择器**：STOP、P、F、PF、FP，以及输入和训练监督相同、只选粗粒度动作 ID 的小 MLP/AFA 策略；修订分支另比较 `τ1-only`、旧计划不变和差异 mask，以检验 `τ0→τ1` 关系差分本身。
2. **缓存输出对照**：peer 预先得到同一完整 P/F，策略只选择要发送的目标/字段并匹配最终字节。分别报告冷启动、为本任务预计算并摊销、以及系统本来就有的 warm cache；后者不能重复收取推理成本。相同输出带来相同质量只验证内容一致，研究增量取决于完整质量—成本前沿。
3. **一次 peer-side bundle**：peer 得到同一个初始 ego query，可在本地条件化执行至多同样的 P/F 计算，但只回传一次；按实际平均字节、计算量和可见信息匹配。它检验第二轮 ego 反馈是否必要。离线逐帧最优 bundle 只能列为 oracle 上界，不能同在线策略排名。
4. **同信息本地计算与 MATE-style**：分别比较 `P_target→本地 MTR`、`P_full_context→本地同 MTR`，并加入“发送 ego 候选、peer 直接返回候选代价”的 MATE 风格服务。它们区分额外信息、远端计算位置与 P/F 中间证据形式。

一次打包必须允许 peer 在同一请求内部先做 P 再决定是否做 F，否则会人为削弱对照。若 sequential 的优势只来自 bundle 被禁止条件计算，结论无效。

## 明确失败条件

- 若 query 只影响返回字段/ROI，并且相对于同任务可用的缓存选择在任务质量—完整成本前沿上均无增量，就不能声称新增了远端执行机制；缓存预测与按需预测内容一致本身不是失败。
- 普通粗粒度 MLP/AFA 在同输入、同监督、同成本下持平：typed planner 没有算法增量。
- 一次 peer-side conditional bundle 在同预算下持平：没有证据支持第二轮 ego 反馈；应收缩为单轮任务条件化远端执行。
- `P_full_context→本地同 MTR` 与远端 F 在等信息下产生无法解释的内容差异：数据契约或实现不一致，不能解释为协作收益。
- 固定 P 或固定 F 已达到相同质量-成本前沿，或 P/F 在 held-out 录制组没有互补分支：调用学习没有研究必要性。
- 只在训练帧或随机帧拆分上成立、换到 held-out 录制组消失：不支持场景条件化能力选择。

## 来源与查阅边界

以上事实以论文正文、正式论文集/IEEE 页面和作者公开仓库为准。已实际打开并核对 MATE 的请求、peer 变换/采样与融合段；DriveAgent-R1 的工具循环、工具集合、模式训练和奖励段；TIP 的候选决策、效用 softmax、计划条件预测段；CAPO 的控制变化与 attention 权重段；BRiG-AFA 的预算专用 Bellman 回归和 matched myopic ablation；NM-PPG 的 POMDP、STOP/成本和 full-trajectory straight-through rollout；AFABench 的 CUBE-NM 与公平比较条件；Who2com/Where2comm/Select2Drive 的请求、匹配、区域选择和算法步骤。

只浏览了公开代码网页，没有 clone 或下载 GitHub/Hugging Face 资源。可核验代码包括 [DriveAgent-R1](https://github.com/wczheng/DriveAgent-R1)、[BRiG-AFA](https://github.com/JIAORONG-FENG/BRiG-AFA)、[AFABench](https://github.com/Linusaronsson/AFA-Benchmark) 和 [Where2comm](https://github.com/MediaBrain-SJTU/where2comm)。未找到 NM-PPG、TIP、CAPO、MATE 或 Select2Drive 在论文/作者页面明确链接且与论文同名的官方实现，因此不把搜索到的同名仓库当作来源。本报告是新颖性反方审计和可证伪设计，不是方法已经实现、实验已通过或论文创新已确立的证明。
