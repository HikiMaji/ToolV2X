# ToolV2X

ToolV2X 研究车辆如何按驾驶任务需要调用邻车的感知 P 与预测 F。当前代码把 **CMP 原 MotionTransformer**、独立 P/F 报文和 **V2V-GoT 原 LLaVA 驾驶模型**接在一起，先验证因果输入和固定动作对照，再开展能力适配与查询机制研究。

截至 2026-09-11，**原 CMP MTR 六组等预算对照已完成**：full/alternating × 三个种子，每次 9,960 步。按训练前固定规则选择 full，冻结预先指定的 seed20 第 9 轮检查点；两个验证录制组平均 top-1 ADE5 为 21.519 → 4.841 米。完整曲线与门槛见 [六组结果](docs/mtr_stability_results.md)，明显训练波动与验证选模的限制仍保留。[首轮适配](docs/mtr_adaptation_results.md)和[发布时进度快照](outputs/mtr_stability_v1/progress_at_github_update.json)保存历史含义。

发布后进程核对发现第 5 组在第一轮后中断，已保留该尝试并从相同原始初始化重跑剩余两组；六组现均完成训练和独立审计，中断尝试不进入选模或六组总更新数。详见 [中断与重跑记录](outputs/mtr_stability_v1/interruption.json)。

驾驶端已支持实际生成 Q8→Q9 和显式 direct 六点轨迹，两种初始化的真实单帧四动作检查见 [接入诊断](docs/driving_decoder_results.md)。四个组合已分别完成一帧、一次真实参数更新与独立重载生成，见 [训练入口结果](docs/driving_training_readiness_results.md)；正式共享驾驶适配及 P/F 驾驶收益尚未验证。查询策略与闭环未评价，RSU、I 和最终查询机制仍后置。当前发布范围见 [更新说明](docs/github_update_2026_09_11.md)。

## 从哪里读

1. [审查范围、代码入口与可直接使用的审查提示](docs/github_review.md)
2. [当前证据表示与训练准备结果](docs/evidence_adaptation.md)
3. [原模型 JSON 接入结果](docs/planning_connection.md)
4. [已批准的目标框架](docs/framework_design.md)
5. [原框架来源、本地适配与许可证](docs/upstream_sources.md)

本机最新能力实验另见 [MTR 实验结果与重放入口](docs/mtr_adaptation_results.md) 和 [执行计划](docs/mtr_adaptation_plan.md)。它是原预测分支在因果输入上的适配，不是原 CMP 全流程或 ToolV2X 方法收益复现。

[STATUS](docs/STATUS.md) 顶部是最新交接状态，下面保留历史正文。早期的“CMP 单仓库 + 自建规划器”、恒速/岭回归方向判断、旧训练入口和已确定路由说法均不代表当前主线。

## 当前执行路径

```text
当前/过去的自车检测特征 ── 原 V2V-GoT projector ──────────────┐
当前/过去的源内检测历史 ── 原 CMP MTR ── 自车预测证据 ───────┤
调用邻车 P 或 F ── 校验报文 ── 保留来源/模式 ── 预算选择 ──┤
                                                         ▼
                            共享原驾驶模型：Q8 → 生成的 Q8 → Q9
                                  或显式 direct → 六点轨迹

未来自车位姿 ── 独立离线标签准备 ── 仅回答 token 的监督目标
```

P 提供当前跟踪状态及可用的 1 秒历史；`history_valid` 表示有可用跟踪输出，包含漏检时跟踪器维持的因果预测状态，不能当作检测匹配标记。F 在提供方完整上下文上预测，再返回查询范围内目标。无 ROI 的完整 P 历史在接收端用同一 MTR 重算，是 F 的同信息对照；仅返回 ROI 内历史的 P 不具备 F 使用的完整上下文。消息保留来源与多模态假设，当前没有学习融合器或查询控制器。

驾驶端实际输入是原检测器的点云浅层特征及文本证据，未提供真实 RGB。原加载器仍初始化 CLIP；占位图像张量用于激活原点云分支。详细执行边界见接入报告。

## 轻量审查检查

使用 Python 3.8–3.11；本机隔离验证使用 Python 3.8。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-review.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_review.py
```

这个入口只需要 NumPy/SciPy，明确选择输入协议、工具、编码、独立标签、资源路径与已上传真实证据的检查。它不加载 PyTorch、Transformers、模型权重或完整数据集。测试对上传的原始失败回答和监督样本也做检查。

首次仓库准备已在隔离导出的文件树上通过 51 项轻量检查；本机资源就绪的完整集成通过 62 项。配置路径调整后，原 MTR 在归档验证帧的 21 个目标上重算结果也与原数组完全一致。核验范围见 [准备记录](docs/github_preparation_validation.json)。

后续审查修复与本轮检查记录见 [审查修复说明](docs/review_9_10_fixes.md)。该修复为 Q8 父回答增加预算余量，可能减少入模目标数；上述历史模型输出和计数保留原含义。

## 完整模型环境与资源

完整本机测试使用 Python 3.8、PyTorch 2.1.2+cu118、Transformers 4.37.2、PEFT 0.10.0、Accelerate 0.21.0、SentencePiece 0.1.99、PyYAML 6.0，以及原 LLaVA 其余依赖。原环境依赖声明随源码保存在 [vendor/v2vgot_llava/pyproject.toml](vendor/v2vgot_llava/pyproject.toml)；其中包含训练/UI 等额外依赖，不是轻量审查安装清单。

下面这些资源不随仓库发布。核心入口接受环境变量；未设置时，数据/原权重默认来自本仓库的同级 `V2V-GoT` 和 `CMP` 目录。

| 环境变量 | 资源 / 默认位置 |
| --- | --- |
| `TOOLV2X_V2VGOT_ROOT` | 检测缓存与位姿；默认 `../V2V-GoT` |
| `TOOLV2X_CMP_ROOT` | 原 CMP 资源；默认 `../CMP` |
| `TOOLV2X_MTR_CHECKPOINT` | 默认 CMP 内 `MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth` |
| `TOOLV2X_V2VGOT_CHECKPOINT` | 默认 V2V-GoT 内原任务 LoRA 的 `checkpoint-4330`，完整子路径见 `src/planning/v2vgot.py` |
| `TOOLV2X_LLAVA_BASE` | 默认 `models/llava-v1.5-7b` |
| `TOOLV2X_CLIP_ROOT` | 默认 `models/clip-vit-large-patch14-336` |
| `TOOLV2X_LLAVA_ROOT` | 默认仓库内 `vendor/v2vgot_llava`；用于显式选择其他原源码位置 |

原 MTR YAML、1.2 KB intention points 和适配模型源码已在 `vendor/cmp_mtr`。原 LLaVA 源码和离线标签函数的原文件也已随仓库提供。权重文件清单见 [resource_downloads.md](docs/resource_downloads.md)。这些文件在原工作站已有，新的机器仍需自行准备；加载器强制离线，不会自动补下权重。

原始轨迹/检测预处理入口还依赖外部 OpenCOOD、AB3DMOT 和原数据。历史预处理脚本中的旧路径未全部改造，不能直接用作正式训练入口。当前运行器还需要本地生成的 `outputs/causal_windows_v1/`，生成与协议说明见 [protocol_audit.md](docs/protocol_audit.md)。

资源就绪后，运行完整集成测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

原模型运行命令见 [planning_connection.md](docs/planning_connection.md) 与 [evidence_adaptation.md](docs/evidence_adaptation.md)，其中工作站绝对路径按上表替换。每次运行使用新的输出目录。测试通过不代表执行了新的 7B 生成、适配训练或质量评价。

## 仓库里的证据

`outputs/` 只提交明确筛选的文本：场景划分、工具报文、输入访问记录、完整/选中证据、模型原始回答、比较摘要、8 条监督样本及历史前向/独立复核结果。原 JSON 接入和紧凑格式失败都保留。

模型权重、完整数据、检测/跟踪缓存、特征数组、论文 PDF/全文、下载缓存和重复源码快照留在原工作站。历史 JSON 中的绝对路径是执行来源记录；对应文件可能没有上传。旧复核脚本依赖这些外部文件和当时的源码快照，不应在本仓库直接重跑后宣称原实验已完整复现。

较大的紧凑表示收益依赖当前两位小数舍入后接近重合的 MTR 模态；扩充目标后的 P 分支仍有 Q9 五点输出失败。预训练模型/检测缓存的训练来源，以及适配算子与原 CUDA 的一致性尚未证实。这些问题是后续工作和本次审查的上下文。
