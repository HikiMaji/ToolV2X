# 原驾驶模型：真实更新、保存与重载结果

日期：2026-09-11。**GoT / V2V-LLM × direct / Q8→Q9 四个组合均完成一次真实参数更新、保存及独立重载生成。** 这项结果建立训练入口的可用性；只有一个训练帧，不代表正式适配、骨干选型或驾驶收益。

协议见 [driving_training_readiness_plan.md](driving_training_readiness_plan.md)，汇总与复核入口为 `outputs/driving_training_readiness_v1/summary.json` 和 `summarize.py`。

## 实际执行

使用原 LLaVA 7B、GoT checkpoint-4330 或 V2V-LLM checkpoint-490 的已有 LoRA、原点云浅层 projector。每个组合内部 Ego/P/F/PF 共用一个模型，累积全部监督行的平均损失后只做一次 AdamW 更新；四个组合从各自原初始化开始，不串接上一组合的更新权重。

训练帧为 `testoutput_CAV_data_2022-03-15-10-09-50_0:10`、全局帧 719，已从研究 train 索引核实。输入是真实 ego 当前/上一帧检测特征，共 540 个点云特征 token，没有真实 RGB。GT 只出现在独立监督目标，提示和点云位置的损失均屏蔽。

| 初始化 | 解码 | 监督行 | 实际更新 | 改变的参数张量 | 重载及实际生成 | 峰值分配显存 |
|---|---|---:|---:|---:|---|---:|
| GoT | direct | 4 | 1 | 452 | PASS | 18.878 GiB |
| GoT | Q8→Q9 | 8 | 1 | 452 | PASS | 18.891 GiB |
| V2V-LLM | direct | 4 | 1 | 452 | PASS | 18.878 GiB |
| V2V-LLM | Q8→Q9 | 8 | 1 | 452 | PASS | 18.891 GiB |

只训练已有 LoRA A/B 和实际使用的 `mm_projector`，共 340,795,392 个参数；基础语言模型、CLIP 与未使用的 scene projector 冻结。基础计算使用 BF16，可训练参数和 AdamW 状态为 FP32。所有可训练张量都有有限梯度，冻结参数没有梯度，452 个张量均有非零变化。direct 的实际序列长 2,248–3,839 token，两阶段为 2,191–3,864 token；没有为了通过检查删减旧 P/PF 证据。

## 保存与独立核对

- 保存前，全部 LoRA 和 projector 文件与内存中的待保存张量逐元素一致。保存 adapter、projector、tokenizer/config；没有保存 AdamW 状态，不能将这些目录当作完整续训检查点。
- 每组另启进程，走默认 FP16、SDPA、合并 LoRA 的标准推理入口。抽查三个参数矩阵（第 0 层 q_proj、gate_proj 和第 31 层 v_proj）共 1,572,864 个值，与显式基础权重加 LoRA 的计算一致，最大差为 0；8 个 projector 张量在目标精度下全部一致。这个抽查不覆盖全部基础语言参数。
- 每组真实生成一条保存的 PF 训练提示，输入没有目标回答，原始输出在 `reload_check.json`。两阶段使用旧已生成的 Q8 父回答检查 Q9 重载路径，没有执行更新后模型的全新 Q8→Q9 链条。该生成只用于检查重载，未作为质量评价。
- `summarize.py` 另核对训练归属、两种初始化使用相同监督行、两种解码使用相同轨迹标签、全部参数变化记录，以及运行时关键源码快照与当前源码逐字节一致。

## 此前的反向传播失败与处理

旧 `got_direct/` 中 Ego 行曾完成前后向，但长 P 行在 torch 2.1.2 默认 SDPA 回退路径 OOM；`got_direct_flash_diagnostic/` 强制内置 Flash 后明确报告本机 sm89/head_dim128 训练不受支持。这两次都没有优化器更新，失败文件原样保留。

随后按原 `train_mem.py` 使用外部 FlashAttention 2，从 PyPI 的 2.5.9.post1 源码构建本地 wheel。构建只保留 head_dim128 的 FP16/BF16 前向、反向与 split 内核，API 显式拒绝其他维度；保留 sm80 代码供本机 sm89 执行，省略 sm90。**这是限定范围的本地构建，不是完整上游 wheel。** 主环境只安装该 wheel，torch 2.1.2+cu118、Transformers 4.37.2、PEFT 0.10.0 等版本保持不变。

固定长度/varlen、FP16/BF16、causal/noncausal 的实际 GPU 前后向检查均通过。另用显式 FP32 attention 公式核对 causal 固定长度与 varlen 的输出和梯度：FP16 最大输出差小于 0.00089，BF16 小于 0.0078；梯度差均小于 0.00066，满足脚本中固定容差。它们是有限用例的内核核验；原 7B 长序列真实更新则由上表四组实际运行验证。资源补丁、版本与检查保存在 `outputs/resources_flash_attention_v1/`。

## 结论边界与后续

本次使用的是旧 MTR 生成的历史训练证据；两阶段父回答来自旧 GoT 真实生成，不能冒充各候选自身生成的正式 2×2 训练集。重新使用训练帧生成的四个回答也不能与此前验证帧的零样本失败直接比较。

当前可以使用 [最终冻结 MTR](mtr_stability_results.md) 物化 train/validation 工具证据，再开展相同数据、可比预算的共享驾驶适配与多帧四动作评价。GoT、V2V-LLM 和两种解码方式仍保留为候选。尚无驾驶质量、P/F 增益、查询策略或闭环结论。

## 重放

完整资源环境及本地训练内核就绪后，分别从原 GoT / V2V-LLM 权重运行；每次使用新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/planning/check_training.py NEW_OUTPUT --checkpoint ORIGINAL_CHECKPOINT --decoding direct
python tests/verify_driving_update.py NEW_OUTPUT
```

另一模式使用 `--decoding q8_q9`。四组合实际命令在 `outputs/driving_training_readiness_v1/run_flash_checks.py`；大权重、特征数组和代码快照仍保存在原工作站。
