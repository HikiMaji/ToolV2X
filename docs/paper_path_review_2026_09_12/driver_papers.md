# 三篇驾驶论文的论文—实现—数据对照

核验日期：2026-09-12。结论：ToolV2X 已有可继续研究的原模型底座，但“GoT 驾驶器 + MTR + P/F 查询接口”本身还不足以构成方法贡献。更值得借鉴的是三篇论文如何把问题、机制和排除替代解释的实验连起来。本轮只核验正文、网页与代码，未运行三篇论文的训练、推理或评价；以下“建议”均未成为已确认机制。当前约束按 [user_requirements.md](/root/autodl-tmp/ToolV2X/docs/user_requirements.md)；共享驾驶训练与本次调研分开。

本次实读的训练 manifest 是 2,987 个训练帧、108 个开发验证帧，各展开为 Ego/P/F/PF/rule 五条件，即 14,935/540 行，并明确没有 learned query targets；这不是五倍独立驾驶场景。重跑配置确认 GoT `checkpoint-4330` 初始化、direct、冻结 MTR、无真实 RGB。监督拼接参考答案并只给回答 token 损失，是正常自回归 SFT；低验证 CE 尚不能代替自由生成的六点轨迹评价。当前数据源是已有 GoT 处理缓存、开发划分是本项目协议，不能称复现原论文完整数据与测试集。[训练 manifest](/root/autodl-tmp/ToolV2X/outputs/framework_baseline_resume_v1/training_data/manifest.json:5)、[实际重跑配置](/root/autodl-tmp/ToolV2X/outputs/framework_baseline_restart_v1/training/config.json:2)、[回答 token 监督](/root/autodl-tmp/ToolV2X/src/planning/adaptation_data.py:112)。

## 1. 身份、版本与实际公开状态

| 论文 | 本次核验版本、发表状态 | 官方入口及公开边界 |
| --- | --- | --- |
| DriveAgent-R1 | arXiv v3，2026-04-20；ICLR 2026。现有本地文本已是该版，开头同时标注会议与版本 | [论文](https://arxiv.org/abs/2507.20879v3)、[项目](https://tsinghua-mars-lab.github.io/DriveAgent-R1/)、[真正代码仓 wczheng/DriveAgent-R1](https://github.com/wczheng/DriveAgent-R1)。README 声明已放出 SFT/Cascaded RL 训练代码；完整评价脚本、指定数据/标签和模型权重仍列 upcoming。Tsinghua-MARS-Lab 同名仓是网页托管，不能用它判断源码未公开。 |
| V2V-GoT | arXiv v4，2026-02-16；ICRA 2026 | [论文](https://arxiv.org/abs/2509.18053v4)、[项目](https://eddyhkchiu.github.io/v2vgot.github.io/)、[官方实现](https://github.com/eddyhkchiu/V2V-GoT)、[官方数据入口](https://huggingface.co/datasets/eddyhkchiu/V2V-GoT-QA)。本地 `/root/autodl-tmp/V2V-GoT` 可追训练、QA 构造、图推理与评价；现有缓存不是原始传感器数据的完整复现。 |
| V2V-LLM | arXiv v4，2026-02-16；ICRA 2026 | [论文](https://arxiv.org/abs/2502.09980v4)、[项目](https://eddyhkchiu.github.io/v2vllm.github.io/)、[官方仓](https://github.com/eddyhkchiu/V2V-LLM)。官方仓将实现引向 GoT 仓；本地 `v2vllmq*` 脚本可核验，不应将 GoT 改过的双帧 baseline 当成原版主表配置。 |

DriveAgent 的 [OpenReview PDF](https://openreview.net/pdf?id=r2g8TV4nJy) 本次访问遇浏览器验证，未声称完成两份 PDF 的逐页一致性检查；实际正文依据是已存在的 [本地 v3 文本，第 1–17 行](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:1) 与在线 arXiv。GoT 本地同样标明 v4，见 [正文第 14 行](/root/autodl-tmp/ToolV2X/docs/papers/text/01_v2v_got.txt:14)。

## 2. DriveAgent-R1：借鉴能力与选择分开的训练逻辑

**论文报告。** 问题是固定视觉输入不能随推理需求补证，而全量输入和始终调用工具又有成本。方法把主动视觉取证、文本/工具模式选择、先练两种模式再练选择连在一起；DM-SFT → FCM-RL → AMS-RL 分别对应格式/能力基础、分模式强化、自主选择。主任务是未来 8 秒四步元动作；轨迹 MLP 是额外开环扩展。关键实验是主动/被动输入、等 RL 轮数训练替代路径、强制模式/自适应模式与效率比较。它已有成本感知取证，ToolV2X 不能仅以“有工具、有停止、有序贯”建立新颖性。[正文训练设计](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:252)、[开环扩展](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:498)、[消融与效率](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:648)。

**代码静态确认。** 真正官方仓的 [多轮 GRPO trainer](https://github.com/wczheng/DriveAgent-R1/blob/main/tool-rl/src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py) 有 `_prepare_inputs_with_tool_calls`、`execute_tool_call` 导入、当前/历史视角映射、前视与 no-tool 六视图输入分支，以及 assistant token 掩码构造。这说明有可读实现，尚不等于所有训练阶段和论文表格已独立跑通。README 明确列出的未公开资产限制了精确复现；本次没有下载代码。其多模态处理器确实接收图像，这与 ToolV2X 的点云特征路径不同。[代码仓发布状态及目录](https://github.com/wczheng/DriveAgent-R1#releases)。

**对我们的三点借鉴（分析）。** 第一，当前共享驾驶 SFT 可解释为“先让接收器学会使用不同条件证据”，不能用它证明调用策略已学会。第二，控制器若训练，监督/奖励要对应生成轨迹的增量效用，不能只学回答格式或看到多少目标。第三，必须用相同可获信息、相近训练预算和真实代价对照，防止收益只是图像/历史更多或训练更久。这是把论文的实验逻辑迁移到已有 [direct 训练入口](/root/autodl-tmp/ToolV2X/src/planning/train_driver.py:247) 的建议；无需因此把整套 GRPO、图像工具和元动作任务搬进项目。原文按调用次数设计的代价也不能直接当作 V2X 字节或毫秒。[原文工具奖励说明](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:1300)。

可比的是“按需补证是否优于固定输入/固定调用”的问题及实验结构；不可直接比较其行为准确率、nuScenes 开环轨迹值和我们的六点输出误差。我们没有它的真实 RGB、Drive-Internal 分布、指定标签与训练权重，3B/7B 也不是受控变量。没有证据支持把任一结果称为闭环安全收益。[原文测试集](/root/autodl-tmp/ToolV2X/docs/papers/text/18_driveagent_r1.txt:388)、[ToolV2X 输入与配置](/root/autodl-tmp/ToolV2X/src/planning/train_driver.py:264)。

## 3. V2V-GoT：借鉴结构假设的消融，保留原协议与因果改版的区别

**论文报告。** 它在已共享两车当前/上一帧特征的前提下，把遮挡感知和规划相关预测组织成九节点推理图；训练用 GT 父回答，推理用生成的父回答。最终 Q9 是三秒六点轨迹。Table I 特别把 V2V-LLM 等基线改为双帧输入；Table II 拆感知图与预测图。值得学的是“提出具体结构假设，再拆掉对应结构看下游规划变化”，而非节点越多越好。[正文数据与上下文](/root/autodl-tmp/ToolV2X/docs/papers/text/01_v2v_got.txt:220)、[训练与对照](/root/autodl-tmp/ToolV2X/docs/papers/text/01_v2v_got.txt:364)、[消融](/root/autodl-tmp/ToolV2X/docs/papers/text/01_v2v_got.txt:445)。

**代码静态确认：必须拆开两类 GT。** 发布的 [图推理脚本，第 50–101 行](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/v1_5/inference_v2vgot.sh:50) 为下游节点重新生成问题；[temp_qa_generation.py，第 102–152 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/temp_qa_generation.py:102) 在非训练模式读取父节点 `merge.jsonl`，Q9 的 [生成父回答分支，第 4227–4236 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py:4227) 拼接 `outputs`。因此不能将训练的 teacher forcing 写成“测试直接给 Q8 GT”。模型加载器也只把 human 文本放入 prompt，随后实际 `model.generate`，不是把 JSON 内的参考答案作为生成结果。[prompt 构造](/root/autodl-tmp/V2V-GoT/LLaVA/llava/eval/model_vqa_loader.py:418)、[生成调用](/root/autodl-tmp/V2V-GoT/LLaVA/llava/eval/model_vqa_loader.py:769)。

但另一条路径真实存在：发布推理脚本生成 Q6 时调用原生成器；后者 [第 7694 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py:7694) 读取双车未来轨迹，经 [第 7774–7783 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py:7774) 传给 sample 构造器；[第 4593–4602 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py:4593) 将邻车 GT future 写作 planned trajectory 输入，与随后读取生成的 Q4 父回答是并行的两种上下文。底层函数确实访问 `t+1…t+N` 位姿，见 [第 2988–2990 行](/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py:2988)。这支持“该发布生成路径不满足 ToolV2X 的 time-t 因果契约”，**不支持直接断言论文表格用了哪个具体生成产物、全部成绩因泄漏失效**；尚缺发表运行的输入清单与来源记录。对 GoT 应保留原基准语义，并另列移除 GT planned context 的因果改版，不能悄悄混作同一 baseline。[官方脚本网页](https://github.com/eddyhkchiu/V2V-GoT/blob/main/LLaVA/scripts/v1_5/inference_v2vgot.sh)。

**可借鉴的三点（分析）。** ① 像拆图那样拆“选择依赖哪个证据”，比较只看初始状态与看 P 返回后再决定；② 共享驾驶初始化、输入帧数和训练预算，避免把原版单帧、GoT 双帧改版、ToolV2X direct 混在同一名字下；③ 同时报告节点/格式失败与最终任务损失，不让中间 QA 成功代替规划质量。当前 direct 绕过整个 Q8 链，所以应命名“GoT 初始化的 direct 驾驶器”，不称原 GoT 方法复现。[当前解码分支](/root/autodl-tmp/ToolV2X/src/planning/v2vgot.py:285)。

另有可核验的复现细节：正文写起始学习率 `2e-5`，本地发布脚本的 LoRA/general LR 是 `2e-4`、projector LR 是 `2e-5`。这只是待确认的论文—脚本差异，不能任选其一后声称精确复现。[论文设置](/root/autodl-tmp/ToolV2X/docs/papers/text/01_v2v_got.txt:364)、[脚本第 15、45 行](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh:15)。

## 4. V2V-LLM：借鉴完整基准的支撑，不能只引用旧版

**论文报告。** 最终 v4 已从旧版的 V2V4Real 扩展到 V2V4Real + V2X-Real，约 1.45M QA、48K frames。研究组织是协作问答问题定义 → 数据/任务 → 场景与目标特征的 LLM 融合基线 → 不同融合方式及非 LLM 基线。最终版还含时延/位姿误差实验和三帧扩展；主表仍是单帧。它的贡献包含新基准，不能简化成“旧骨干拼接也能发论文”。[最终正文 I、V-C/D、IX](https://arxiv.org/html/2502.09980v4)。

**代码静态确认。** 本地 [原单帧 Q5 训练脚本，第 13–26 行](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f1.sh:13) 明确 LoRA/projector、`num_input_frames=1`、`ego_only=False`；同仓还单列 `f2` 改版。实际视觉输入是点云特征，零图像仅占位，[loader 第 418–429 行](/root/autodl-tmp/V2V-GoT/LLaVA/llava/eval/model_vqa_loader.py:418)。因此 LLaVA/CLIP 初始化不能被解释为原实现用了真 RGB。这也是 ToolV2X 必须持续明确点云特征与文本输入的原因。

**可借鉴的三点（分析）。** ① 用同一任务出口比较融合/服务选择，减少骨干和损失的混杂；② 若宣称 MLLM 是必要组件，加入使用同等输入的简单非 LLM 任务端，而不只比较多个 7B 配置；③ 将数据构造作为可审查贡献的基础，但我们复用处理缓存并重划录制组，尚没有新大规模基准的贡献。其 V2X-Real 扩展只用于理解最终论文的证据广度，不构成当前加入 RSU 的理由。[官方实现导航](https://github.com/eddyhkchiu/V2V-LLM)、[当前两车执行器](/root/autodl-tmp/ToolV2X/src/planning/episode.py:8)。

三秒六点是我们与这两篇最接近的输出契约，但还需对齐 metric 实现。发布评价将 `1s/2s/3s L2` 分别计算为前 2/4/6 点的平均欧氏误差；不是仅取第 1/2/3 秒末端误差。碰撞使用记录中的未来 GT 盒做离线几何检测，并不执行车辆闭环。ToolV2X 的 ADE/FDE 必须另列，不能只因单位同为米就抄进原表；只有缓存没有足够原 GT 时，也不能伪造相同碰撞指标。[L2 实现第 737–743 行](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/eval_v2v4real_3d_grounding.py:737)、[碰撞实现第 531 行](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/eval_v2v4real_3d_grounding.py:531)。

## 5. 资源复用与最小合理研究增量

| 资源 | 现在可复用什么 | 暂不需要扩张什么 |
| --- | --- | --- |
| DriveAgent-R1 | 在线阅读 trainer 的工具结果回填、模式/输入分支和奖励组织；优先用来设计对照 | 未公开的完整数据、指定标签、权重不能算已具备；当前无须下载整套 RL 工程或更换 Qwen 骨干。[官方发布清单](https://github.com/wczheng/DriveAgent-R1#releases) |
| V2V-GoT | 已有 PointPillars 特征、原 LoRA/projector、样本组织、生成与解析代码；teacher-forced/生成父上下文的区别 | 原九节点图不是独立远端工具，不能直接搬入在线输入；完整原图重跑不是当前 direct 的前置条件。[图推理入口](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/v1_5/inference_v2vgot.sh:1) |
| V2V-LLM | 单帧 direct 风格任务端与多种融合对照的命名/配置；原 Q5 评价函数 | 不为追最终版规模自动增加 V2X-Real/RSU；本地 f2 不能代替原版主表 f1。[单帧训练配置](/root/autodl-tmp/V2V-GoT/LLaVA/scripts/v1_5/train_task_lora_7b_my_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f1.sh:13) |

**当前最容易被审稿人视为工程拼装的原因，是尚未有一个独立成立、可被强对照推翻的方法命题。** 五条件 SFT 只适配驾驶器；规则查询器仍由速度、返回目标和几何关系决定；P/F 都使用同一个 MTR 时，F 的收益可能只是传输形态、上下文范围或计算位置不同。代码正确、协议完整和训练能恢复值得保留，但不是因果方法收益。[规则决策](/root/autodl-tmp/ToolV2X/src/planning/episode.py:11)、[F 完整上下文计算后筛 ROI](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:75)、[当前训练冻结/目标配置](/root/autodl-tmp/ToolV2X/src/planning/train_driver.py:264)。

**建议只增加一个可检验机制：结果条件化的远端能力选择。** 初始状态先决定是否取 P/F；取得 P 后，利用新证据重新估计继续查询 F 的净价值，并允许停止。先以共享冻结驾驶器的配对生成结果离线构造效用监督，再用小选择器检验是否能泛化；这是候选设计，不预设 RL，也不承诺序贯必然有效。最小证据链是：Ego/P/F/PF/rule → 只看初始状态的一次性小选择器 → 读取真实 P 返回的选择器。采用相同预算/可获信息/接收器、保留失败，在独立录制组比较质量—通信/计算代价；离线 oracle 只给上限。若后者不能稳定超过一次性与规则，论文应收缩到被证据支持的能力选择问题，不能靠增加工具或改写故事维持序贯主张。[现有可复用的执行/计费/证据更新链](/root/autodl-tmp/ToolV2X/src/planning/episode.py:36)。

其中最关键的替代解释对照是：**无 ROI 的完整 P 在本地用同一 MTR 重算，与 F 使用相同源上下文**；只给 ROI 内 P 历史却让 F 看完整邻车场景，不能证明远端预测能力独有。还应拆分远端新信息与执行地点、报告真实 request/response bytes、接收端重算成本。已有实现允许这样检验，尚待任务收益实证。[提供方边界](/root/autodl-tmp/ToolV2X/src/tools/vehicle.py:60)、[接收端 MTR 重算及成本](/root/autodl-tmp/ToolV2X/src/planning/episode.py:56)。这条研究增量若成立，贡献是“在哪种已见证据下继续获取哪种远端能力有价值”，而不是“我们把三篇论文接了起来”。
