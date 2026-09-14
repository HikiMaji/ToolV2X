# 结构化协同驾驶路径实现记录

日期：2026-09-15。状态：四项实现及单项审查通过，最终全分支审查及修复复审通过，轻量 322/322、本地 main 完整模型环境 416/416 通过，代码已本地整合，尚未推送。承接 09-14 架构设计，保留 P/F 的因果协议与真实反馈主线；不启动真实数据训练或新方法效果实验。

## 参考源码

已从官方仓库下载 UniV2X（压缩包 1,598,011 字节）和 VAD（542,442 字节），均小于用户批准的 100 MB。通过 ZIP 完整性与成员检查，没有下载权重或数据集。逐函数来源与实现边界见 [参考源码映射](structured_driver_source_mapping_2026_09_14.md)。这是标准组件借鉴和本项目接口重实现，不是两套原框架复现。

## 实施范围与状态

| 部分 | 代码入口 | 状态 |
|---|---|---|
| 结构化输入、因果历史、目标关联 | `planning/structured_inputs.py`, `planning/evidence.py` | 实现与两轮修复审查通过 |
| 共享数值规划器 | `planning/structured_driver.py` | 实现与 7 项核心行为测试通过，修复后独立审查通过 |
| 既有 P/F 交互、控制与评价 | `planning/method_episode.py` 等 | 接入、定向回归与独立审查通过 |
| 数值监督、初始化与恢复 | `planning/train_structured_driver.py` | 实现、合成回归与修复后独立审查通过 |

输入层沿用上游已经对齐的 ego(t) 坐标并验证契约，不再次变换已经对齐的框。门控一对一关联保留远端独有目标和歧义；自车判定需要当前几何及合法历史佐证。源内 ID、receipt、请求类型和 provider/receiver 执行地点不作为学习特征。

新路径使用完整 `StructuredDriverSpec`，默认 P-state (`observations_only`)；P-local 为明确选项。旧 `new_ledger` 默认仍是原 v1 anchor/forecast 账本，新增历史需显式开启 v2。P 的历史是 causal tracking-state motion proxy，不代表逐帧 detector hit。

P 的任务参数控制合法私有跟踪状态的排序/装包，当前实现没有重新跑 detector/tracker。F 由原 CMP MTR 在完整合法 peer context 上预测后再按任务排序，ego 方案并未进入 MTR，不能称为反应式条件预测；合法预测缓存复用保留。

准备记录区分字段获取、派生、选择及间接父依赖。运行器将历史读取路径保存在 `task.inputs.ego_history_read_paths`；`prepared.ego_history_used` 从实际数值数组重建，包含状态、有效掩码和时间，其 `read_paths` 为空。导出器分别保存读取审计并校验数值输入，路径不进入张量或模型身份。先前方案仍携带其字段父依赖，不能在本轮删除直接 token 后宣称模型已完全遗忘该字段。

## 实际代码修改文件

- [scripts/check_review.py](../scripts/check_review.py)
- [src/evaluation/framework.py](../src/evaluation/framework.py)
- [src/evaluation/planning.py](../src/evaluation/planning.py)
- [src/planning/bundle_data.py](../src/planning/bundle_data.py)
- [src/planning/driver_contract.py](../src/planning/driver_contract.py)
- [src/planning/evidence.py](../src/planning/evidence.py)
- [src/planning/method_episode.py](../src/planning/method_episode.py)
- [src/planning/query_data.py](../src/planning/query_data.py)
- [src/planning/run_framework.py](../src/planning/run_framework.py)
- [src/planning/structured_driver.py](../src/planning/structured_driver.py)
- [src/planning/structured_inputs.py](../src/planning/structured_inputs.py)
- [src/planning/train_structured_driver.py](../src/planning/train_structured_driver.py)
- [tests/test_evidence_ledger.py](../tests/test_evidence_ledger.py)
- [tests/test_structured_driver.py](../tests/test_structured_driver.py)
- [tests/test_structured_episode.py](../tests/test_structured_episode.py)
- [tests/test_structured_inputs.py](../tests/test_structured_inputs.py)
- [tests/test_structured_training.py](../tests/test_structured_training.py)

## 接口和使用方式

交互仍走 `run_task_episode`，数值分支由 `toolv2x_interaction_v2` 明确选择。每次实际驾驶输出为 `toolv2x_numeric_plan_v1` 的六个二维点，独立记录 numeric token / output point 计数，语言生成计数为零。总计算时间累加外层各阶段，内部模型耗时不重复相加。numeric token 计数是最终注意力中未屏蔽的 token 加六个查询，不能当作 padding 后的 FLOPs 或语言 token；实际耗时另记。

精确重跑冻结整个输入（包括旧 prior 槽）；同证据修订只替换为刚刚实际输出的方案及其依赖，冻结实体、张量和已选择字段。单轮条件委托仍是一个 RPC、至多两个原语，可用同证据修订补足三次驾驶预算。旧 Ego/P/F/PF/rule 保留在共同执行器。

评价新增明确命名的 `got_prefix_L2_1s/2s/3s/avg`，分别计算前 2/4/6 点误差均值。每个前缀要求全部标签有效，avg 要求三段均存在。原端点 `L2_1/L2_2` 与有效点 ADE3 不改；相同公式不代表与 GoT 论文相同数据划分、任务和输入。

数值运行器已通过 `planning.structured_driver.load_structured_planner(checkpoint, device=...)` 接入实际序列化格式。监督 API 为 `prepare_training_rows`、`masked_trajectory_loss`、`training_loss`、`initialize`、`fit` 和同名加载器。

导出行引用 `source_task` 和实际阶段 prefix 索引，独立保存完整精度监督，不重复存储 padded tensor。训练按阶段顺序用当前模型生成并 detach 先前方案，再进入较多证据阶段；精确重跑保留原 prior 槽。每个阶段行监督其全部前缀和配置数量的最终同证据修订，因此早期前缀会出现在多个行目标中。该权重方式属于当前训练配置/实现，后续效果比较必须共用且明确报告。

无效 prior 保留原因并屏蔽；无有效标签的 batch 不更新参数。验证行仅计算损失，按物理录制组与训练隔离。任务/特征逐文件复制一份输入快照，训练读取快照；恢复直接比对原任务语义、特征、模型、优化器、随机状态和位置，允许内容一致的目录迁移。

先保存 `checkpoint_000000.pt`；`save_interval` 以处理 batch 为单位（包括无监督 batch），周期保存完整状态，结束或显式停止时另存最终状态。原路径拒绝覆盖；恢复写新目录。参数和恢复状态的直接副本比较增加磁盘占用，不是对抗性防伪机制。`report.json` 记录训练/验证行数、监督前向、无效 prior 原因、P/F 实际回执覆盖和验证损失。

### 已核对的命令模板

以下模板经实际 CLI help 核对，未在真实资源上执行。配置字段必须完整，未知字段拒绝，driver spec 与采集时严格一致。先用既有 `StructuredDriverSpec` 生成完整配置；CPU 示例对应本轮验证环境。

```python
import json
from pathlib import Path
from planning.structured_inputs import StructuredDriverSpec

spec = StructuredDriverSpec().to_dict()
config = dict(
    version='toolv2x_structured_training_v1',
    driver_spec=spec,
    seed=7,
    optimizer=dict(name='AdamW', lr=0.0001, weight_decay=0.01),
    batch_size=2,
    epochs=2,
    refinement_depth=1,
    save_interval=10,
    device='cpu',
)
Path('structured_spec.json').write_text(json.dumps(spec, indent=2))
Path('structured_training.json').write_text(json.dumps(config, indent=2))
```

```bash
PYTHONPATH=src python -m planning.train_structured_driver initialize \
  /path/to/new_untrained.pt --spec structured_spec.json --seed 7

PYTHONPATH=src python -m planning.train_structured_driver prepare \
  --tasks /path/to/numeric_run/tasks.jsonl \
  --labels /path/to/independent_offline_labels.jsonl \
  --out /path/to/new_prepared_rows.jsonl

PYTHONPATH=src python -m planning.train_structured_driver train \
  --rows /path/to/new_prepared_rows.jsonl \
  --config structured_training.json --out /path/to/new_training_run

PYTHONPATH=src python -m planning.train_structured_driver train \
  --rows /path/to/new_prepared_rows.jsonl \
  --config structured_training.json --out /path/to/new_resumed_run \
  --resume /path/to/new_training_run/checkpoint_000010.pt
```

`train` 是正式命令，`fit` 是别名；`--resume` 读取完整训练检查点并写入新目录。初始化文件仅用于启动后续数值任务采集，不含优化器/数据/位置绑定，不能当作训练恢复点。无 resume 时训练从配置中的固定 seed 新建模型。

## 验证

| 检查 | 结果 | 能支持的结论 |
|---|---|---|
| 改动前资源无关回归 | 302 通过 | 原始基线可运行 |
| 输入层新测试 + 指定旧协议回归 | 13 + 75 = 88 通过 | 输入、关联与证据契约 |
| 输入层之后完整轻量回归 | 302 通过 | 原轻量检查未回归，未导入 torch/transformers |
| 输入层修复后定向回归 | 41 通过 | 严格关联校验、反向航向、空本地/纯远端/空场景 |
| 输入层独立审查 | 两轮修复后通过 | 已确认任务范围内没有未解决的重大问题 |
| 数值网络核心行为 | 7 通过 | 实际梯度、mask、分块/置换、先前方案、空/单点输入及输出契约 |
| 数值闭环及定向旧回归 | 43 通过（16 交互 + 7 网络 + 20 旧项） | 真实合成 P/F、先前方案、共同对照、成本、归档和指标 |
| 最后成本修复 / 旧 bundle 回归 | 分别 19 / 8 通过 | 内层耗时与总尝试成本一致，旧单轮归档保留 |
| 交互独立审查 | 通过，无阻断项 | 数值接入范围内的规格与代码质量 |
| 全结构张量同信息核对 | 通过 | 完整 P-local 后追加等价 F 的全部张量、运动、prior 相同；合成契约 |
| 监督/恢复最终定向回归 | 19 通过（12 监督 + 7 网络） | 真实归档格式、独立标签、精确恢复、迁移、失败和单轮覆盖 |
| 监督物理划分修复 | 14 训练测试通过、复审通过 | 拒绝 physical test/缺失 split 经导出和保存行拟合进入 train/validation |
| 最终全分支审查与修复复审 | 通过，两项重要问题关闭 | 实际历史审计路径可导出/拟合，旧 v1 与 opt-in 列表窗口兼容 |
| 最终完整资源无关回归 | 322 通过，199.796 秒 | 旧 v1/协议/归档及新输入，未导入 torch/transformers |
| 完整模型环境首轮 | 416 项，415 通过、1 项测试子进程路径失败 | 失败发生在 import 前；原始失败日志保留 |
| 测试子进程修复 | 清空 PYTHONPATH 后 15 项训练测试通过 | 子进程自行从测试文件定位 src，生产代码不变 |
| main 完整模型复验 | 416/416 通过，390.541 秒，退出码 0 | 清空 PYTHONPATH 后标准完整入口通过；旧模型/协议/归档和数值通路回归 |

最终复验的 [完整日志](structured_driver_validation_2026_09_15/model_environment_main.log) 和 [机器可读结果](structured_driver_validation_2026_09_15/summary.json) 已保存。轻量套件包含于完整模型环境的覆盖范围，两组数字不能相加当作独立用例总数。生产代码自最终审查后未再改变；首轮完整失败仅修改测试子进程定位方式，随后在 main 重跑完整套件。

Task 3 初次 160 项定向运行有 1 项 tokenizer 路径配置错误；纠正外部资源路径后，该项已通过并计入最终 43 项。初次运行不记为完整 PASS。

完整资源无关检查使用普通 review Python 执行 `python scripts/check_review.py`，原始输出见 [轻量日志](structured_driver_validation_2026_09_15/resource_independent.log)。完整模型环境首轮的测试启动失败保存在 [首轮日志](structured_driver_validation_2026_09_15/model_environment_initial.log)，不以修复后的定向通过覆盖这次失败。最终复验从本地 main 使用以下命令，明确清空父进程 `PYTHONPATH`：

```bash
cd /root/autodl-tmp/ToolV2X
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  TOOLV2X_V2VGOT_ROOT=/root/autodl-tmp/V2V-GoT \
  TOOLV2X_CMP_ROOT=/root/autodl-tmp/CMP \
  TOOLV2X_LLAVA_BASE=/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b \
  TOOLV2X_CLIP_ROOT=/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336 \
  /root/autodl-tmp/conda-envs/llava/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

模型环境为 Python 3.8、PyTorch 2.1.2+cu118、NumPy 1.24.2、SciPy 1.10.1；本轮模型行为验证在 CPU 上进行。原 MTR 测试使用合成窗口/标签与内存中的微型更新，原 GoT projector 测试使用合成场景数组；其余包含 tokenizer、微型适配器及既有归档回归。日志保留上游 Transformer nested-tensor 提示、PEFT 缺少基础配置提示和旧 tokenizer 超长序列提示；相关 tokenizer 检查没有把超长提示交给真实 GoT 生成。

本轮合成 CPU 前向/反向与微型恢复检查只验证代码通路和可训练性。没有在真实样本上生成新 GoT/MTR/结构化轨迹，没有更新实际研究检查点。最终代码完整不等于已有有效驾驶权重，更不等于 ToolV2X 的反馈调用收益成立。

## 实施中的依赖调整

增加明确的未训练检查点初始化入口：如果先采集数值任务要求已有训练检查点，而训练又要求数值任务，就无法启动首批数据准备。初始化入口使用固定 seed 和完整规格，并与训练后状态分开标记；本轮仅在临时合成测试目录验证。该接口若与后续数据组织不合适，代价是调整初始化/采集接入，不涉及改变 P/F 方法机制。


独立初始化/训练增加显式 `training_run_id`，保存于 `model_version.training`，复制/加载检查点和同一恢复链保留身份。它弥补同名、同 seed、同步数仍可能对应不同权重的问题；直接参数/优化器/输入比较仍保留。该字段不是内容指纹、防伪证明或网络特征。代价是相同权重的独立训练也会被保守视为不同查询策略绑定，复用需要明确重绑定。

## 当前结论边界

保持开环六点轨迹模仿任务口径，不声称闭环安全、路线执行或道路约束。当前没有合法道路/导航分支；本车检测头场景图不等于地图。标准关联、编码、注意力和数值轨迹头属于共同驾驶基础，创新归因仍需由真实反馈的请求/STOP 机制和公平对照来验证。

## 本地交付

已将审查通过的分支快进整合到本地 `main`，没有推送 GitHub。原五份未提交文档与整合前副本逐字节一致；其中三份已有设计文档按原内容纳入本批记录，`docs/user_requirements.md` 和 `docs/9-14-1.md` 保持原来的未提交状态。权重、数据集和原实验产物未纳入本批提交。临时实现/审查记录已转存到公开文档，保留最初失败及后续修复，不用最终 PASS 覆盖历史。
