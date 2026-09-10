# 车车学习型能力验证 v1

> **用户已纠正本轮推进路线：此自建岭回归实验降为接口/简单基线诊断，不作为 ToolV2X 核心能力验证或路由停止依据。** 后续必须复用 CMP / V2V-GoT 原模块或基于它们修改，不能继续用自建小预测器替代。下文保存这次试跑事实。

预先固定于 2026-09-10，结果产生前写入。本轮承接 vehicle_probe.md，不确定最终 ToolV2X 方法。

## 本轮设计

- 使用已有 recording_holdout_v1：15 个训练片段拟合，两个 validation 片段评估；本轮不读取 test 标签或调试 test 结果。
- P_state 候选：全部当前 peer tracks 的来源内 ID、当前 3D 框、过去 1 秒 OLS 估计的速度、当前检测置信度、有效观测数量与短历史回退标记。包含公共时间、来源、坐标与状态字段。没有原始历史或未来轨迹。
- F：远端基于完整 1 秒历史生成未来 5 秒轨迹。P_history：传回相同完整历史，在本地运行同一模型，作为信息等价强对照。
- 共享预测器：目标自身坐标系中的恒速残差 ridge 回归，53 维特征，50 个未来时刻各回归二维残差。当前特征含速度、尺寸、置信度、观测数量；历史特征为有效位置相对恒速拟合的残差、mask、历史分数及视图标记。不输入来源身份、场景、绝对位置、GT ID 或未来信息。
- 训练时同时使用完整历史和状态摘要视图，权重共享且两个视图等权；P_state 在其训练过的输入视图上推理。每个未来时刻只用该时刻存在的离线标签拟合。短历史在线目标保留并作静止回退，不接受学习残差。
- L2 正则候选固定为 0.001 / 0.01 / 0.1，目标为均方残差加正则，截距不惩罚。按两个验证场景 × 两个来源的完整历史 observed ADE 等权平均选参数，完整记录所有候选及恒速对照。
- 这是单目标、单轨迹的学习基线；不含邻车交互、地图或多模态输出。它不代表成熟预测器上限，也不叫 MTR 复现。CMP 原 dataset 的 GT 交集和筛选仍未适配；现存权重继续保持未验证可用状态。

## 固定对照与判断

1. 每个来源在同目标、同未来标签 mask 上比较 CV 与 learned，报告 ADE、5s FDE、退化比例与逐场景结果。
2. 同一批 peer 目标比较 learned(P_state) 与 learned(P_history)，区分历史信息增量。允许摘要获胜。
3. 枚举 ego、P、F、PF、P_history，以及 ego_cv/F_cv 两个恒速诊断。Ego 使用共享学习型模型；P 使用摘要视图，F/P_history 使用完整历史视图。
4. P_history 本地计算必须等于远端 F；PF 保留同一 peer 一次，因本轮接收器使用 F 预测，预期 PF=F。不能把这项一致性当互补性或顺序查询收益。
5. 对共同 ego 目标保留相同当前 GT 匹配与未来标签，配对评分；新增远端目标与匹配歧义另记，不能直接比较不同目标集合的均值。
6. 相同简单接收器和历史外推参考路径继续使用。速度缩放、中心距离近距离代理仅作诊断，不是闭环规划或车体碰撞实验。
7. 字节按实际 JSON 请求/响应记录；网络、上游检测/跟踪时延未知。未使用 RSU、I、LLM、RL 或路由训练。

本轮只决定该基线是否提供进一步验证的证据。验证场景少且用于模型选择，结果不是最终测试成绩；已有检测缓存的训练来源仍未核实。

## 实测结果

**本轮已完成拟合、两个验证场景回放和独立复核。该线性预测基线没有显示稳定超越恒速的能力，不能据此进入路由训练。**

训练使用 3027 个决策帧，保留 63573 条在线目标—帧记录；其中 44657 条有至少一个未来标签，43890 条同时满足历史至少两个有效状态。未来标签点共 1524981 个。没有按未来标签存在与否删掉在线目标；只有有标签且历史合格的条目进入回归损失。

验证共 108 帧、4340 条在线目标—帧记录，3863 条可评分。三个候选完整历史验证目标依次为 2.738233 / 2.659149 / 2.527796 m，选择 alpha=0.1。模型含 5300 个回归系数，在现有 NumPy 环境实际完成闭式拟合，没有 GPU 推理、MTR、LLM 或路由训练。

以下 ADE 在同一来源的相同目标和相同 future mask 上配对计算，先算每条目标—帧的已观测 ADE，再在该场景内平均。A 为 `2022-03-17-16-06-11_0`（60 帧），B 为 `2022-03-21-09-50-20_10`（48 帧），完整场景名前缀均为 `testoutput_CAV_data_`。

| 场景/来源 | 恒速 ADE (m) | 共享模型完整历史 ADE (m) | 后者减前者 (m) |
|---|---:|---:|---:|
| A / ego | 1.7189 | 1.8483 | +0.1294 |
| A / peer | 1.4763 | 1.6184 | +0.1421 |
| B / ego | 3.3735 | 3.5109 | +0.1373 |
| B / peer | 3.1362 | 3.1337 | −0.0025 |

每场景等权后，ego 从 2.5462 变为 2.6796 m，peer 从 2.3062 变为 2.3760 m，均较差。将恒速协作接收器换成学习型版本，在相同接收目标集合上，A 从 1.3764 变为 1.4846 m，B 从 3.3540 变为 3.4126 m，也较差。不能只挑“协作 learned 优于 ego learned”的比较来宣称学习型 F 有效。

完整历史相对摘要并非每个场景都更好：peer 在 A 的摘要/历史 ADE 为 1.5519 / 1.6184 m，在 B 为 3.3081 / 3.1337 m。说明本基线存在场景差异，但远不足以证明需要结果驱动的工具选择。

### 接收、规划与成本

- 完整历史 P 本地计算与远端 F：108/108 帧完全一致；PF 和 F 的接收结果也全部一致。这两项是预期对照。
- 七个动作在 108 帧都选择 1.0 倍历史外推速度，规划输出没有差别。A 的 3s ego L2 为 0.7244 m，B 为 1.4343 m，已观测近距离事件均为 0；这仍不是闭环安全结论。
- 按几何接收规则，新增 peer-only 目标—帧记录共 1012 条，其中 857 条能在 peer 侧匹配当前 GT，838 条的 GT 身份没有在 ego 匹配集合中出现。并非 838 个独立物体或经确认的物理遮挡。
- 两侧分别几何匹配 GT 后，1101 对合并目标中有 40 对对应的 GT ID 不同，全部来自 A。这个离线歧义标记不能自动判定真实错配，也没有用于在线纠错；它限制了对当前距离融合收益的解释。单来源 CV/learned 比较不经过跨车融合。

| 响应 | 108 帧平均应用层请求+响应字节 |
|---|---:|
| P_state | 6712.8 |
| F | 48354.8 |
| PF | 55067.6 |
| P_history | 36664.2 |

本 JSON 格式下，F 比信息等价的完整历史响应更大；这些字节只代表当前单轨迹编码。计算时间按单次 Python 墙钟记录，上游检测/跟踪与网络时延为 null，不据此声称远端计算或端到端时延优势。

### 复核与结果边界

- 新增 8 个行为测试，覆盖摘要速度、同信息一致性、短历史回退、缺失历史、离线标签掩码、已知回归映射、目标坐标旋转和共同目标评分；连同此前 19 个，共 27 个 unittest 用例全部通过。
- 独立实现使用 `polyfit` 与逐时刻矩阵乘法，复算 4340 条来源预测、19634 条接收轨迹及 432 条消息字节。最大预测差约 1.75e−12 m；回归正规方程最大残差约 1.78e−15。
- 对 17 个训练/验证片段各取首、中、末决策帧，独立关联当前 GT 并用相对位姿变换未来标签，共检查 51 帧、1361 条目标记录。最大标签残差差约 3.78e−6 m，符合 float32 缓存精度。并核对全部 67913 条训练/验证在线目标的数量，没有因未来标签筛除输入。
- 生成阶段禁用 GT 加载函数和 GT ego 未来函数，拒绝标签/其他权重文件读取，检查 1728 次位姿读取均在当前历史窗内。未验证的上游检测器训练历史不在此检查证明范围内。
- validation 用于选择正则参数，且仅有两个场景；本轮没有独立 test 成绩，没有 bootstrap 显著性或正式泛化声明。

本轮结论限于：**接口与学习/评价链路已建立，当前线性残差基线未提供足够的预测能力增量。ToolV2X 方向尚未被证实或否定。**

## 产物与复现

产物根目录：`outputs/learned_capability_v1/`。

- `preregistered_protocol.md`：运行前写下的设计内容副本；归档时才复制，副本修改时间不代表预先写下设计的时间。本节实测结果不包含在该副本中，也没有外部预注册平台记录。
- `data/`：角色 manifest、逐场景训练/验证特征及离线标签，`preparation.json` 记录覆盖。
- `model/`：`model.npz`、完整拟合统计、`training.json` 及所有候选验证分数。
- `replay/{scene}/`：逐场景 `run.json`、消息原文、七动作决策、生成摘要。
- `replay/offline_summary.json`、`offline_frames.csv`：同目标配对和规划代理指标；`input_guard.json`：生成输入检查。
- `artifact_verification.json`：完整独立复核；`replay_verification.json`：先行的纯回放复核。
- `code_snapshot/`：本次使用的源码与测试副本。

在 ToolV2X 根目录运行，输出必须使用新的目录名，脚本拒绝覆盖已有结果。顺序依次为：

```bash
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python src/probe/learned_data.py outputs/learned_recheck/data
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python src/probe/train_learned.py outputs/learned_recheck/data outputs/learned_recheck/model
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python src/probe/run_learned_probe.py outputs/learned_recheck/model/model.npz outputs/learned_recheck/replay
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python src/probe/evaluate_learned_probe.py outputs/learned_recheck/replay
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python tests/verify_learned_probe.py outputs/learned_recheck
OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/conda-envs/dmstrack/bin/python -m unittest discover -s tests -v
```

## 下一步判断

先补足预测能力与关联质量，再讨论路由。后续候选预测器应先在独立单车输入上通过恒速强基线检查，再接回当前 P_state/P_history/F 同信息对照；不能仅靠把简单模型称为 F 推动故事。跨车关联需要基于可观测历史、朝向/尺寸等信息核查上述歧义，GT 只作离线诊断。更换预测器或关联规则应另开版本，保留本次负结果。

上述是后续建议，不是已经确定的新模型、完整架构或新增实验授权。RSU、I 和完整 Agent 仍后置。
