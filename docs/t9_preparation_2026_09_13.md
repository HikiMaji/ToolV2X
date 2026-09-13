# T9 准备：冻结价值策略入口与独立单轮终态监督

日期：2026-09-13。范围是本轮获准的代码和契约验证，不是 T9 真实效果实验。已将此前完成的 T6 审查修复、T7、T8 提交并推送至 `origin/main`，远端与该本地提交一致。下面 T9 改动单独保留为本地待审查交付。

本轮未加载真实 GoT/MTR，没有真实分支采集、真实录制上的策略训练或新方法 rollout，没有重建缓存。唯一实际参数更新来自临时目录内的小型 CPU 合成契约测试；其中驾驶和预测都是显式测试替身，不能作为模型效果证据。

## 完成的接口

| 文件 | 本批新增或变更 |
| --- | --- |
| `src/planning/method_run_spec.py` | `freeze_method_run_spec`、`validate_runtime_binding`、`prepare_value_checkpoints`；冻结语义配置、精确样本、录制角色/折、完整 utility 和价值权重快照。 |
| `src/planning/run_framework.py` | `interact` 新增显式 request/shared 和 bundle_terminal checkpoint 选项；新增 `collect-bundle-method`，延迟调用原 runtime。未选择 checkpoint 的旧诊断调用保留。 |
| `src/tools/control_bundle.py` | `capture_bundle_prefix` 与进程内真实首返回 handle 的 `fork`；只给离线采集器使用，普通远端请求仍不能携带该 handle。 |
| `src/planning/bundle_data.py` | `bundle_collection_spec`、`collect_bundle_branches`、`validate_bundle_archive`、`make_bundle_targets`、`load_measured_bundle_targets`。 |
| `src/planning/method_episode.py` | 保存离线采集中已实际发生的首响应费用来源；线上 policy 仍仅接收可见数据。 |
| `src/evaluation/framework.py` | 部署计算费用包含保存的真实首响应时长；与当前物理执行的 suffix attempt 分列记录，再计入总费用。 |
| `src/planning/query_data.py`、`src/planning/query_value.py` | 共享语义绑定新增实际 driver 训练元数据；实测 bundle 表在拟合时重读原始终态、报文和离线标签；不能凭一张自行声明的实测表绕过核验。T8 显式 synthetic_contract 表保留契约用途。 |
| `tests/test_bundle_data.py`、`tests/test_method_run_spec.py`、`scripts/check_review.py` | 新增采集、原始终态绑定、失败/覆盖、真实小型 checkpoint 和运行冻结测试；模型测试仍与轻量套件分开。 |

这些都是工程接入或实验对照代码，没有新增 Method、网络结构创新、工具类型、RSU/I、peer 学习排序或联合 RL。核心机制仍是 current/change 参数化 P/F、真实返回后的同一 GoT 修订、修订影响第二请求/STOP，最多两次远端能力调用。

## 单轮监督到底执行什么

每个固定时间样本先实际执行一次初始 driver，得到初始方案。分别为初始 P、F 构造合法且已固定的外层 envelope；每个首工具实际执行一次 provider 首原语，随后才捕获 provider 的真实返回、receipt、完整合法上下文缓存和已发生费用。

从这个真实边界分叉枚举 STOP 与公开可执行的续动作。每个分支实际执行自己的续原语，并经同一个 receiver/driver 得到最终方案。STOP 也需要把第一次返回交给 receiver，再实际执行最终 driver；不能引用初始方案充当 STOP 终态。provider 的 continuation 可见内容只有已发送 envelope、真实首返回、公开候选，不能获得 ego 的中间修订 τ1。共享初始 driver 和首原语只减少物理采集工作，不消除每个部署分支应支付的费用。

P 首原语只用 causal tracking-state motion proxy，不运行 MTR。F 延续完整合法 peer context 后排序/装包。P→F 的不同续分支各自实际预测，不共享兄弟分支的未购结果；F 首原语已经合法算过的完整预测则可在同一分支继续复用。单轮没有两次原语之间的 ego 生成，保持一轮委托对照的因果边界。

归档保存初始任务、每个第一工具的公开可行性/失败、provider 首返回、每个续动作的实际终态、全部期望样本和进度。在线索引严格使用现有因果白名单，没有 GT 参数。`window_reads` 只保留本分支的读取和共享实际前缀，不把兄弟读取追加进来。

## 标签与成本

`make_bundle_targets` 才读取独立的六点离线标签。每个续动作相对同一首返回下的实际 STOP 终态计算：

```text
value = STOP_terminal_loss - action_terminal_loss
        - weighted(action_terminal_cost - STOP_terminal_cost)
```

quality 使用显式 ADE3/FDE3 权重；失败使用显式 failure_loss。request/response 均按完整外层 wire 的实际 UTF-8 bytes 计费。保留首响应 duration，并从恢复该真实首返回之后单独测 suffix duration；评价时相加。分叉重复的协议初始化仅计入物理采集 attempt，不重复算作部署费用，MTR 嵌套诊断不重复加到总计算费用。最终 driver 耗时可以变化，因此相对 STOP 的计算增量允许为负。未测网络/I/O/加载时间仍不能当成端到端实时延迟。

每个公开可执行续动作与 STOP 都必须有归档。标签缺失或任何候选的成本未知时，该源状态记录在 `coverage.json` 的排除原因中，不拼凑不完整监督；生成失败保留，不用较早有效方案替补。原始样本及失败条目仍留在采集归档和 coverage。

实测目标版本为 `toolv2x_bundle_targets_v2`；目标文件保存相对于自身目录的原始分支和离线标签快照位置。拟合时必须传文件路径，重新验证原始请求/响应、first receipt、候选覆盖、实际 Z、最终 raw answer 和成本，并重算整表。`v1 + measured_bundle_branches` 声明缺少这种核验，不能再直接用于拟合。无原始材料的 `v1 + synthetic_contract` 仅保留为测试接口。

## 运行规格与入口

`toolv2x_method_run_spec_v1` 必须显式给出：完整 limits（内含 ExecutionSpec 和 receiver_spec）、local_provenance、control、utility_spec、recording_folds、recording_roles、runtime_binding、query_spec。运行记录补充精确 sample/scene/g/local_frame/role/fold 列表和 value_checkpoints。路径只表示加载/归档位置，模型和上下文的语义身份继续使用版本化结构。共享绑定也记录实际 GoT 的训练 epoch/step/examples/seed 等非路径元数据、实际 MTR 加载结果的 checkpoint 元数据和参数/键匹配信息。模型路径迁移允许，已记录的训练身份变化拒绝；这不等于证明任意人为改写但元数据相同的大模型文件内容一致。

价值 checkpoint 先复制至本次运行私有的 `value_checkpoints/`，再从该副本加载策略并核对全部绑定。恢复时直接比较快照的完整保存对象，包括网络参数、优化状态、配置和 RNG；不能换一个同路径的新文件继续旧运行。恢复同时逐字节核对复制的全部源码/配置，包括 MTR YAML；只比较 Python 文件不足以冻结执行。大模型保持外部资源，不额外复制整套 7B 权重。价值快照使用直接完整对象相等检查，底座则使用版本及实际加载元数据，二者保证强度不同。

不选择 request checkpoint 时，只允许既有且明确标注的诊断请求策略，避免“加载了 bundle 权重”被误称为完整学习方法。

下面是入口说明，**本轮没有执行这些真实模型命令**：

```bash
PYTHONPATH=src python -m planning.run_framework interact NEW_OUTPUT \
  --data CAUSAL_DATA --checkpoint FROZEN_DRIVER --spec FROZEN_RUN_SPEC \
  --value-checkpoint SHARED_VALUE --role validation

PYTHONPATH=src python -m planning.run_framework collect-bundle-method NEW_BRANCHES \
  --data CAUSAL_DATA --checkpoint FROZEN_DRIVER --spec BUNDLE_COLLECTION_SPEC \
  --role train --per-recording 1
```

one_shot 实际执行时还可显式给出 `--bundle-value-checkpoint BUNDLE_TERMINAL_VALUE`；其 policy ID 必须与 envelope 中注册的 continuation 一致。采集规格版本为 `toolv2x_bundle_collection_v1`，字段与 T7 collection 对齐；本批单轮终态采集限定两次 driver、无额外生成。same-evidence 额外生成仍由原 T6 对照负责，不混入这张独立监督表。

## 验证与下一步边界

定向轻量：bundle/底座身份 8 项通过，另新增重复 setup 计费反例 1 项通过；10 项 run spec/入口检查通过；旧入口 4 项回归通过。专门 CPU 的新实测表→checkpoint 契约及原 9 项 T8 拟合/恢复/教师回归共 10/10（129.453 秒）通过。原 8 个 v1 函数 AST 与本轮发布快照相同；旧 160 条 raw answer、32 帧五策略复算在发布前通过。

完整轻量检查和独立复审的最终结果见本文末尾交付记录。

本批实现独立 one-shot **末步 continuation** 的采集/核验和实际价值策略入口，尚无真实 one-shot 首请求/续问公平拟合。one-shot 首请求仍需在未来执行阶段使用独立、分组冻结 continuation 选择其真实终态，不能把 alternating T7 的修订 τ1 标签移植过来，更不能使用离线最优续分支冒充部署策略。本轮没有宣称 T9 全链实验完成。

下一批应先做获准的极小真实链路验证，确认原模型输入、输出、时间与字节账本，再收集按物理录制冻结的有限分支；在此基础上完成教师/价值拟合与公平对照。继续保留旧 Ego/P/F/PF/rule。是否需要进一步适配底座，应由这些真实链路结果决定，不自动恢复长时间 GoT/MTR 训练。

独立复审：发现并修复共享前缀重复计入协议初始化、实际 GoT/MTR 训练/加载身份遗漏、恢复只核对 Python 而漏掉 MTR YAML 三类问题；最终 18/18 定向检查通过（38.204 秒），无剩余必须修复项。底座元数据绑定仍不提供任意同元数据权重篡改的内容证明，详见上文保证范围。

最终交付：T9 共 17 个文件已同步到本地 `main`，与隔离工作树逐文件一致。主工作区完整轻量 **290/290（145.871 秒）**；专门小型 CPU **10/10（129.453 秒）**；最终时间口径说明补充后的评价回归 **16/16（0.685 秒）**。轻量进程没有导入 torch/transformers。历史 160 条回答复算通过，8 个 v1 AST 不变；用户原审查文档与历史证据未修改。T6 修复/T7/T8 已推送 `origin/main` 并复核，T9 保留本地未提交、未推送。没有运行真实模型/数据实验，停止在本批准备范围。
