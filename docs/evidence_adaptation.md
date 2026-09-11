# 紧凑证据与适配训练入口

本文记录紧凑表示与驾驶监督前向的历史阶段，文中的“零优化步”只适用于该阶段。后续已完成原 MTR 首轮因果适配，见 [实验结果](mtr_adaptation_results.md) 和 [最新状态](STATUS.md)；本文保存的部分驾驶回答已补做 [基础质量诊断](planning_quality.md)，历史输出保持原内容。

日期：2026-09-10。承接用户“开始”，已实现可还原的紧凑证据格式、实际原模型对照、训练输入/标签准备和原模型监督前向检查。**本轮优化器步数为 0，没有完成适配训练，也没有规划质量结论。**

审查后补充：以下表格保留原运行结果。后续 [审查修复](review_9_10_fixes.md) 为实际 Q8 父回答增加 128 token 余量，并把残余 Q9 超限保存为动作失败；新选择结果可能少于历史记录，不能把旧模型输出视为新预算下的结果。

## 1. 已完成的改动

`src/planning/inputs.py` 增加 `pack_evidence` / `unpack_evidence`：目标使用公共表头和结构化行，重复公共字段只写一次。完全相等的预测路径可以共用存储，但六个有序模态仍通过 `mode_to_path` 保留各自路径与分数；不平均模态、不归一化分数、不合并不同来源的同号目标。历史时间、历史有效位和缺失字段均可还原。

编码器本身不舍入。原始精度输入经过编码/解码后逐字段完全相同；略有差别的六条轨迹仍是六条。当前规划输入继续沿用上一阶段的两位小数舍入，两种格式的精度一致，完整工具报文和原 MTR NPZ 未改变。这是规划文本编码改进，**不是网络报文压缩**。

`src/planning/v2vgot.py` 和 `run_connection.py` 支持 `--evidence-format json|compact`。两种格式使用同一距离优先规则和同一 4096 token 容量，保留至少 256 个生成 token 的预算，记录全部舍弃项。紧凑版当前作为可选训练候选，默认仍为已有 JSON；新格式未经训练，不能因为更短就直接替换默认模型行为。

`run_connection.py` 另支持 `--research-split train|validation`、`--scene-index`，只允许已有决策清单内的帧，旧 `--validation-index` 仍可用。测试数据没有加入此入口。

## 2. 相同信息与增加覆盖的对照

验证帧仍为记录 `testoutput_CAV_data_2022-03-17-16-06-11_0`、局部帧 10、全局帧 5526。相同原 MTR、原 7B + V2V-GoT LoRA、真实 Ego 点云特征；没有新增真实 RGB 输入。

执行了三种条件：

- 旧 JSON：`outputs/framework_connection_v3/` 中已保存的真实原模型运行。
- 相同目标、相同证据，只改紧凑格式：`outputs/evidence_comparison_v1/*_same_set_compact.json`，本轮重新执行原 7B。
- 固定相同 token 容量，紧凑格式容纳更多完整目标记录：`outputs/framework_compact_v1/`，本轮重跑原 MTR、P/F 和原 7B。

已逐元素确认旧/新运行的完整证据相同、实际 P/F 响应文件相同。同信息条件下，送入两种格式的数值、目标集合和关联候选相同，仅序列化改变。

| 动作 | 旧格式记录数 | 紧凑版记录数 / 其中远端 | 相同集合 JSON token 上界 | 相同集合紧凑 token 上界 | 相同集合紧凑 Q8/Q9 | 增加覆盖后紧凑 Q8/Q9 |
|---|---:|---:|---:|---:|---|---|
| Ego | 5 | 19 / 0 | 3384 | 1641 | 均可解析 | 均可解析 |
| P | 4 | 9 / 3 | 3809 | 2540 | 均可解析 | Q8 可解析，Q9 只有五点 |
| F | 5 | 19 / 7 | 3516 | 1815 | 均可解析 | 均可解析 |
| PF | 4 | 8 / 6 | 3477 | 2256 | 均可解析 | 均可解析 |

表中的 token 数含 540 个原点云特征 token，以 Q8 和固定短 Q8 原型条件下的 Q9 取较大值。此前称其为“最长合法 Q8”的上界不准确：解析器允许额外说明，实际生成的父回答可以更长。修复后在原型计数之外另预留完整 128 token 的 Q8 生成余量；解码再编码的实际长度仍在每次生成前检查，不能将估算视为无条件上界。记录数区分来源和能力，PF 中同一邻车目标的 P 本地重算/F 可以占两条记录，不等于两个物理目标。

P 的失败原文完整保存在 `framework_compact_v1/P/plan.json`。没有补造第六点、放宽解析器或替换为 GT。相同信息条件四动作均可解析，只能说明紧凑格式在这个较小目标集合上可用；增加目标后的故障还可能涉及长度、内容及未训练的输入分布，尚未严格定位为单一原因。

Ego 增加覆盖后的回答虽然有六点，但六点全为 `(49.3,1.4)`。这进一步说明格式可解析不代表轨迹可用。没有进行误差、动态可行性或碰撞评价，不能据此比较 P/F 的驾驶收益。

## 3. 压缩收益的边界

在完全不舍入的完整证据上，公共字段/结构压缩的 token 数为：

| 动作 | 原始精度 JSON | 原始精度紧凑版 |
|---|---:|---:|
| Ego | 36,252 | 35,113 |
| P | 130,103 | 123,899 |
| F | 86,440 | 80,520 |
| PF | 179,898 | 169,053 |

这些只在 tokenizer 中计数，未向模型输入超长序列。原始精度下的收益约 3%–7%；大部分数字本身没有可消除的重复。

当前验证帧的 21 个 Ego 目标在原始精度下均有六条不同路径，但沿用两位小数舍入后，其中 18 个只有一条不同路径、3 个有两条。紧凑编码能在不进一步丢失这些已舍入数值的前提下复用路径，所以当前可保留记录数有较大增长。不能把这一增长外推成一般六模态预测器的压缩率，也不能将模态重合解释为预测质量好。

完整 P 历史仍然占用大量文本。当前紧凑输入没有覆盖所有远端目标，P/F/PF 的入模目标集合也仍不相同。若后续因果训练后的 MTR 模态更丰富，需重新测量容量；届时可能需要面向原投影/任务模型训练结构化证据 token，而不应继续通过更粗舍入隐藏信息损失。

## 4. 已落盘的训练准备

入口：`src/planning/adaptation_data.py`，产物：`outputs/adaptation_data_v1/`。

| 产物 | 实际内容 | 不能等同于 |
|---|---|---|
| `online_index/train.jsonl` | 3027 个既定训练决策帧；只含 Ego 当前/历史特征路径、当前运动、窗口引用和查询配方 | 3027 帧完整工具证据已生成 |
| `online_index/validation.jsonl` | 108 个既定验证决策帧，同样的因果字段 | 用验证帧训练 |
| `offline_labels/{train,validation}.jsonl` | 各帧独立的六点 Ego 未来、有效位、Q8/Q9 监督目标及读取路径；本次全部具备完整六点标签 | 在线输入或最优安全规划标签 |
| `examples/train.jsonl` | 一个真实训练帧、Ego/P/F/PF 四条件，共 8 条实际 Q8/Q9 监督样本 | 全量 SFT 数据已物化 |
| `split_manifest.json`、`preparation.json` | 记录级隔离检查、计数、未完成项 | 预训练来源独立性已证明 |

训练/验证沿用已有记录级划分，未读取测试集特征或标签。输入张量实际逐帧加载并检查形状/数值，未来标签存放在独立文件。输入清单只引用邻车窗口，并不提前加载邻车内容；实际工具执行仍由 query 后的服务负责。

离线标签直接执行原 V2V-GoT `inference.py` 中两个纯函数 `get_suggested_speed_steering`、`get_future_trajectory_str`，通过 AST 只加载这两个函数，避开整个 GT QA 图和感知程序初始化。未来 Ego 定位以双精度转换到固定当前 Ego 系，采样 0.5/1/1.5/2/2.5/3 秒；Q9 标签沿用原函数的一位小数输出。Q8 沿用原距离/角度分类，包括阈值边界和低速直行约定。

它们是实际驾驶轨迹的模仿学习标签，不是碰撞最优规划标签。缺失或跨记录的未来点保留无效位，不伪造标签、不据此删除在线决策记录。本次完整标签覆盖源于继承的决策清单本身留有足够片段时长。

## 5. 一个真实训练帧与监督前向

实际训练帧是 `testoutput_CAV_data_2022-03-15-10-09-50_0`、局部帧 10、全局帧 719，产物 `outputs/framework_compact_train_v1/`。使用原 MTR 和原 7B 实际生成四动作，Q8/Q9 都可解析；Ego 和邻车原输入分别有 8、6 个目标。没有用前述验证帧冒充训练样本。

Q9 的输入来自对应动作实际生成的 Q8 回答，GT Q8 只作为 Q8 的监督目标。即便某个 Q9 推理回答格式失败，只要 Q8 父回答合法且标签存在，也允许保留其 Q9 训练样本；不按模型答对与否筛选训练集。Q8 父回答不合法时，不用 GT 回填 Q9。

`encode_supervision` 使用原 Vicuna 提示模板，将全部系统/用户/证据 token 的标签设为 −100；只有 assistant 目标与结束符参与损失。`src/planning/check_adaptation.py` 已加载原完整模型，对 8 条样本分别调用原多模态输入准备和监督 forward，并确认新增的 540 个点云特征位置也被屏蔽。

结果在 `outputs/adaptation_forward_v1/forward_check.json`：8 次 loss 全部有限，序列长度 2191–3864，均不超过 4096。没有 backward、没有 optimizer、没有新 checkpoint。这里只验证监督数据能通过原模型，不能把单次 teacher-forcing loss 当作训练提升或规划成绩；训练用可更新 LoRA/优化器配置仍待接入。

## 6. 复现与验证

在 `/root/autodl-tmp/ToolV2X` 下使用现有 `llava` 环境；所有输出目录须是新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python src/planning/run_connection.py outputs/framework_compact_replay --mode infer --evidence-format compact
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python src/planning/compare_evidence.py outputs/evidence_comparison_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python src/planning/run_connection.py outputs/framework_compact_train_replay --mode infer --evidence-format compact --research-split train --scene-index 0 --local-frame 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python src/planning/adaptation_data.py outputs/adaptation_data_replay --runs outputs/framework_compact_train_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python src/planning/check_adaptation.py outputs/adaptation_data_replay/examples/train.jsonl outputs/adaptation_forward_replay
```

比较入口默认读取本次已保存的 `framework_connection_v3` 和 `framework_compact_v1`；若比较新运行，用 `--baseline` / `--expanded` 显式指定。它会检查真实特征、完整证据与帧一致，再调用原模型做相同目标集合的紧凑对照。

本次日志在 `outputs/logs/`：`framework_compact_v1.log`、`evidence_comparison_v1.log`、`framework_compact_train_v1.log`、`adaptation_data_v1.log`、`adaptation_forward_v1.log`。

58 项测试通过，日志为 `compact_adaptation_tests_v1.log`。独立脚本 `outputs/logs/verify_compact_adaptation_v1.py` 已重新检查完整报文一致、原始精度及已舍入证据可还原、同信息目标集合、3027/108 输入与标签的匹配、因果路径、部分原始定位标签重算、实际 Q8 父上下文、8 次监督前向和当前代码与最后快照一致性；结果见 `outputs/adaptation_data_v1/independent_verification.json`，PASS。

旧运行快照未覆盖。最终代码快照在 `outputs/adaptation_forward_v1/code_snapshot/`；先前实验目录的快照反映各自执行时的版本。

## 7. 接下来做什么

下一步优先把**原 CMP MTR 的因果适配训练**补起来，使用训练记录、独立未来标签、源内目标与缺失历史，验证原结构在当前输入上的预测质量和模态分布；暂不全量物化 3027 帧的当前弱 MTR 证据。

随后用冻结的已验证能力模型生成全量 Ego/P/F/PF 证据，再适配共享 V2V-GoT 驾驶模型，比较新旧表示、目标覆盖、格式失败和任务误差。当前 8 条样本只验证入口。发布权重/检测缓存的训练来源仍未证实，正式未见场景泛化需要独立处理这一问题。查询策略、RSU、I 和闭环继续后置。
