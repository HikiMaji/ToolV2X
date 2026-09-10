# ToolV2X 原模型任务链接入记录

核查日期：2026-09-10。当前结论：**单个真实验证决策帧上，Ego / P / F / PF 四个固定动作均实际执行了原 V2V-GoT Q8→Q9，并生成可解析的行为与六点轨迹。原 CMP MTR、P/F 报文和同信息本地重算已串联。** 这完成的是框架接入阶段；适配训练、正式规划质量评价、学习查询策略和闭环尚未完成。

后续更新：紧凑证据、真实同信息对照、3027/108 帧训练准备和 8 条原模型监督前向已完成，见 [evidence_adaptation.md](evidence_adaptation.md)。下文保留 `framework_connection_v3` 接入时的结果与命令；当前入口增加了格式和训练角色参数，默认仍为原 JSON。

## 1. 本轮实际执行的链条

```text
Ego 当前/上一帧原检测特征 → 原 V2V-GoT 点云特征投影
Ego 因果跟踪历史 → 原 CMP MTR → Ego 预测证据
固定动作 Ego / P / F / PF
    P：邻车当前检测框 + 已观测历史 → 真实序列化报文 → 本地原 MTR
    F：邻车全部当前目标及历史 → 邻车原 MTR → 真实序列化预测报文
源分离证据与几何关联候选 → 明确记录的上下文选择
原 LLaVA 7B + 原 V2V-GoT projector / LoRA → Q8 行为
同一模型 + 实际生成的 Q8 回答 → Q9 六点轨迹
```

新入口在 `src/planning/run_connection.py`。输入适配在 `src/planning/inputs.py`，原模型调用在 `src/planning/v2vgot.py`，工具和接收在 `src/tools/vehicle.py`。本轮未改写原 CMP / V2V-GoT 仓库；历史上其他 agent 对原仓库的改动不属于本轮新增内容。

检测与 AB3DMOT 跟踪使用已有落盘结果，本轮没有从原始点云重新运行感知、跟踪。CMP 使用此前已接入的原 MotionTransformer 编码器/解码器与现有单车权重，KNN/索引注意力沿用 portable PyTorch 算子适配；没有执行 CMP 完整的协同感知/协同预测聚合器，也没有执行完整 V2V-GoT Q1–Q9 图。

## 2. 可复现位置与命令

最终运行：`/root/autodl-tmp/ToolV2X/outputs/framework_connection_v3/`。

```bash
cd /root/autodl-tmp/ToolV2X
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -u src/planning/run_connection.py outputs/framework_connection_replay --mode infer
```

输出目录必须不存在，避免覆盖证据。省略 `--mode infer` 时默认为 `prepare`，只执行原 MTR、原投影和输入准备；不能将它报告为 Q8/Q9 已执行。当前 CLI 只覆盖预先列入验证决策帧的样本。

本次确切运行命令与上面相同，输出参数为 `outputs/framework_connection_v3`，stdout/stderr 重定向到 `outputs/logs/framework_connection_v3.log`。主要产物：

- `connection.json`、`execution.json`：动作状态及完成范围，`full_framework_complete=false`。
- `cmp_model_loading.json`、`v2vgot_model_loading.json`、`projection.json`：实际模型、权重、配置与执行信息。
- `input_access.json`、`ego_features.npz`、`ego_point_tokens.npy`：输入路径、因果帧与实际投影。
- `forecasts/*.npz`、`predictor_calls.json`：五次实际预测器调用的原输出、目标数与耗时。
- 各动作目录下的 `P/F_request.json`、`P/F_response.json`、`tool_costs.json`：实际请求、响应与字节成本。
- 各动作目录下的 `evidence_full.json`、`evidence_used.json`、`context_selection.json`、`plan.json`：完整证据、送入模型的证据、舍弃项、Q8/Q9 提示和回答原文。
- `same_information.json`、`independent_verification.json`：同信息对照与独立落盘复核。
- `code_snapshot/`：本次运行的 ToolV2X 代码、测试、CMP 模块副本与被调用的 V2V-GoT 原文件。

## 3. 输入与实际模型边界

研究切分使用已有记录级留出清单。本次来自验证记录 `testoutput_CAV_data_2022-03-17-16-06-11_0`，局部帧 10、全局帧 5526；物理文件在原 train 档案。只验证了这个决策帧的接通，不能据此声明跨记录泛化。

主驾驶模型直接读取的原特征只有 Ego 在 5526、5525 两帧的 `regression_map`、`classification_map`、`detection_box_score`，没有加载远端特征、deep feature、GT QA JSON 或未来位姿。原投影输出为 `[1, 540, 4096]`，数值有限，8 项投影状态严格加载。两帧各包含 220 个场景 token 和 50 个目标槽 token；目标槽包括原布局的填充项，不等于 50 个真实目标。

Ego 当前运动由当前/上一帧定位计算，输入窗口仅含 −1.0 至 0 秒的历史，坐标系为固定当前 Ego 系。归档 pickle 内包含其他决策帧，加载器只把选中的因果窗口交给模型，不把整段序列送入在线计算。在线代码不读取真实未来作为工具响应或任务上下文。

MTR 实际严格加载 880 项状态、68,514,524 个参数。权重为 `/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth`。Ego 21 个目标均执行 MTR；邻车 30 个目标中 27 个执行 MTR，另 3 个只有一次观测，明确标记 `model_used=false` 并使用保持当前位置的回退。不能把这 3 个称为 MTR 预测。原 MTR 输出 6 模态、50 个 0.1 秒预测步，本轮报文选取 0.5–3 秒的六个时刻；原 50 步输出仍保留在 NPZ。

驾驶端在 RTX 4090 上实际加载完整 LLaVA v1.5 7B、原 V2V-GoT `checkpoint-4330` 的 LoRA 和非 LoRA 投影权重，通过原 `LlavaLlamaForCausalLM.generate` 生成回答。主模型接收真实点云检测特征 token；**没有输入真实 RGB 图像**。原加载器仍初始化 CLIP，接口中的全零 image 张量只激活原点云特征分支，不能算作图像理解实验。

本地适配目录仅调整 CLIP 为本地路径及单车输入配置，原权重未训练。本轮还将发布配置的 tokenizer 上限从 2048 显式调整到原 Llama 位置容量内的 4096，并在调用前预留生成空间、拒绝超限。日志中 tokenizer 的 2048 提醒来自计数阶段；本轮实际 Q8/Q9 均经过独立的 4096 容量检查。这一配置变化需要在后续适配训练和对照中保持一致。

## 4. 四个固定动作的真实结果

下表中的“记录”区分来源和能力；PF 的同一邻车目标可以同时保留 P 本地重算、远端 F 两条记录，不代表两个物理目标。

| 动作 | 完整证据记录 | 进入文本的记录 / 其中远端 | 请求字节 | 响应字节 | Q8 | Q9 |
|---|---:|---:|---:|---:|---|---|
| Ego | 21 | 5 / 0 | 0 | 0 | fast / straight | 六点可解析 |
| P | 51 | 4 / 2 | 138 | 58,597 | fast / straight | 六点可解析 |
| F | 51 | 5 / 3 | 138 | 53,071 | fast / straight | 六点可解析 |
| PF | 81 | 4 / 3 | 276 | 111,668 | fast / straight | 六点可解析 |

例如 Ego 的实际 Q9 为：

```text
The suggested future trajectory is [(3.8,0.9),(7.7,1.6),(11.8,2.4),(15.9,3.2),(20.1,3.9),(24.5,4.6)].
```

四个动作的末端点分别为 `(24.5,4.6)`、`(72.2,-0.1)`、`(23.9,2.9)`、`(7.2,3.5)`。差异只说明该输入适配下模型给出了不同输出；没有计算规划误差、碰撞或动态可行性，不能解释成 P/F 带来收益。行为均为 fast，而轨迹差异很大，也提示当前未经适配的模型需要任务一致性检查。

响应成本由实际 UTF-8 JSON 文件字节重算，包含目标锚点、历史/预测和元数据；没有模拟网络传输。当前是未压缩浮点 JSON 格式，不能用这一帧的字节比例概括 P/F 的一般通信优势。实际 Q8 生成约 0.84–0.93 秒、Q9 约 1.89–2.12 秒，只是本机该帧调用耗时，不是系统端到端延迟或实时性评价。

## 5. 同信息对照与隔离检查

P 传输完整邻车历史后，本地使用与 F 相同的 MTR、相同全部邻车目标上下文进行预测。30 个目标的六模态六点预测及分数逐元素完全相同，最大差为 0；其中 MTR 目标和短历史回退均逐目标对齐。

这验证了当前完整 P 可以把 F 的计算迁移到 Ego。它不能证明 F 没有研究价值，也不能证明序贯查询有必要：区域 P 可能提供不同上下文，异构模型/算力、计算位置和成本也需另做明确对照。当前 F 始终以全帧目标/历史计算，然后按请求 ROI 筛输出；没有声称已实现按目标节省计算。

Ego 动作的远端档案读取次数和消息字节均为 0。远端窗口只在 query 后加载。接收器拒绝场景/帧不匹配、未来历史时间、未知 GT 字段、同源 P/F 锚点冲突；空响应明确是“未观察到目标”，不是“已确认自由空间”。不同来源的同号 track ID 不被视为同一目标。关联目前只是当前几何/尺寸候选，尚未实现历史一致性关联或学习融合。

## 6. 失败记录、修复与验证

首轮 `outputs/framework_connection_v1` 已实际执行原语言模型，但 Q8 回答成了目标运动描述，因此未执行 Q9。该失败产物保留。

`outputs/prompt_diagnosis_v1/results.json` 保存了同一个真实 Ego 输入、同一个原模型的三项对照：原版问题单独输入时回答 slow/straight；新增含答案占位符的空证据问题会复制 `<speed>` / `<steering>`；相同长证据末尾重新放置原版任务问题时回答 stop/straight。这将故障定位到新增提示组织方式，而不是模型文件缺失。

修复移除了可直接复制的答案占位符，将原 Q8/Q9 任务问题放在已获得证据和实际 Q8 上下文之后。没有添加 GT、固定答案、解析失败后的替代轨迹或重试选最好的结果。`framework_connection_v2` 首次使四动作 Q8/Q9 均可解析；随后补了提供者场景校验和跨来源 ID 的关联保留检查，最终 `framework_connection_v3` 重跑四动作也均可解析。

最终验证命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python outputs/logs/verify_framework_v3.py
```

结果：48 项测试通过；独立脚本重新检查响应字节、真实 Q8→Q9 上下文、六点解析、token 容量、源分离关联、同信息预测和当前 Python 文件与运行快照一致性，结果 PASS。日志分别为 `outputs/logs/framework_tests_v3.log` 和 `outputs/logs/verify_framework_v3.log`。测试中的替身只用于边界单测；真实运行入口使用原 MTR 和完整原 MLLM。

## 7. 当前缺口与下一步

1. **完善证据表示。** 当前六模态浮点 JSON 很长，4096 上下文只容纳 4–5 条记录。完整账本保留全部目标，文本按同一当前距离规则选择，舍弃项全部记录；主模型仍看到 Ego 原特征，但不能说它看到了完整远端证据。P、F、PF 保留的目标集合不同，尚不是规划侧严格同信息对照。下一步应在保持来源、时间、模态与有效性信息的前提下压缩表示，并固定相同预算和目标选择规则。
2. **整理干净训练样本并适配共享任务模型。** 用记录级 train/validation/test 切分，覆盖 Ego/P/F/PF 和缺失证据，训练模型使用新协议。GT 只能进入独立监督/离线评价，不能复用原 QA 中的 GT notable-object、GT 伙伴未来或 GT 父节点回答作为在线输入。
3. **核实并适配预测器。** 本地 MTR checkpoint 元数据是 epoch=1、it=6，训练来源和检测器/LoRA 的记录独立性未证实；因果固定坐标输入也不同于原 GT 对齐 dataset。不能把现有弱/近重合模态输出视为 CMP 的正式性能，或据此否定 ToolV2X。需在干净输入上训练/验证原 MTR，并测量算子适配与原 CUDA 的一致性。
4. **再做正式对照与查询学习。** 先完成训练后的固定动作、同信息 P 本地重算和共享任务评价，再比较简单规则、一次选择与结果驱动的后续查询。实时截止、完整通信/计算代价、历史关联和闭环仍是后续工作。RSU 与 I 继续后置。

## 8. 资源状态

已按用户授权使用 GPT-5.6-Luna Max 子 agent 盘点资源；最新清单是 [resource_downloads.md](resource_downloads.md)。收到下载域名限制后，主 agent 已停止先前启动的下载，改用本机完整历史缓存建立可读链接，随后离线加载模型成功。目前无需用户补下载。

后续若需要 github.com、githubusercontent.com、githubassets.com、huggingface.co 及相关下载域名的资源，先列明官方地址、具体文件与目标路径，由用户下载；不通过镜像或跳转绕过该分工。
