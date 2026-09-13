# T3：已购 E、派生记录和实际入模 Z

日期：2026-09-13。用户授权实施 T3，完成后停止，不进入 T4，不启动训练、GoT 生成、真实 MTR 实验或数据缓存重建。

T3 已接通真实 v2 服务返回到接收端 ledger 和驾驶输入构造。当前可以解释每次收到哪些字段、哪些预测来自本地已购 P、哪些字段最终进入提示或被容量丢弃；尚未执行真实 GoT 修订，也没有请求价值模型或两轮方法收益结论。

## 1. 文件与 API

| 文件 | 本批改动 |
| --- | --- |
| `src/planning/evidence.py`（新增） | 已购、派生、回执、有效上下文、失败状态、同信息字段归并。 |
| `src/planning/context.py` | 独立 `build_task_plan_input`，共享确定性接收选择、精确字段位置和 token 预算。旧 builder 保留。 |
| `src/planning/inputs.py` | 显式 v2 evidence 白名单，复用原 compact codec 和 direct prompt；旧 v1 分支保持原行为。 |
| `src/tools/task_spec.py` | `FrozenPredictor` 将同一冻结 callable/config 绑定到当前运行；抽出已有 predictor 校验供 provider/receiver 共用。 |
| `src/tools/vehicle.py` | 可选付费完整 P 上下文证明、F 实际 predictor binding；旧 v1 query/decoder/helpers 不改。 |
| `tests/test_evidence_ledger.py`（新增） | 21 项资源无关契约测试。 |
| `tests/test_framework_episode.py` | 新增原 tokenizer 的 v2 接收预算测试，不生成模型回答。 |
| `scripts/check_review.py` | 注册新轻量测试。 |

```python
bound = FrozenPredictor(predictor, model_version, settings)
ledger = new_ledger(local_window, local_prediction,
                    predictor=bound, local_provenance=provenance)
ledger = apply_response(ledger, service.query_task(request), bound)
manifest = known_field_manifest(ledger)
prepared = build_task_plan_input(tokenizer, motion, ledger, feature_tokens, limits)
```

`apply_response` 消费实际 `{request, wire, cost}`，而非自行声明的字段清单；解码后还检查旧引用对应的已收数值、实际字节成本和固定场景/时间/来源。默认 P 处理为 `local_mtr`，`observations_only` 只是显式诊断配置，在 ledger 创建后不能切换。没有新增模型骨干或预测算法。

## 2. E、派生和 Z 的分工

- `acquired_fields` 只存远端实际返回的新身份字段，`origin=remote`，保留全部取得该字段的 receipt IDs。重传继续付费，但不增加 E 的字段集合。
- `derived_fields` 只存 receiver 输出，`origin=receiver_derived`，保留精确 anchor/history 父引用、模型版本和实际输入上下文。新上下文产生新派生记录，旧记录不覆盖。
- `receipts` 保留真实请求、完整响应、wire 文本和实际服务成本；同一已处理 receipt 重放是幂等的，新 receipt 的重传仍是新的付费事件。
- `contexts` 保存重建输入的数组数值、dtype、目标顺序、来源/时间、父字段和 predictor binding；`active_contexts` 指向当前有效派生，旧 partial 结果只留审计，不自动继续进入 Z。
- `known_field_manifest` 从实际 receipt 和对应数值构造远端确认，不输出 derived。调用方可据此形成第二请求；T3 不执行策略决策或提前屏蔽 F。
- `prepared.admission_report` 保存已购/派生/实际入模/丢弃字段引用、丢弃原因、共享 anchor 和等价来源别名、对象行列或 shared 列位置、本车字段位置及付费观察来源。
- `prepared` 保存实际舍入后的本车/远端 evidence、完整 direct prompt、精确输入 token 数、完整 receiver 配置及构造耗时。T4 尚未接通，不应称这些字段已被 GoT 使用或理解。

输入 ledger 和响应不原地修改，返回快照可 JSON 序列化；JSON 保存/读取后能构造相同提示。普通内部数据结构不是任意外部文件的认证系统：接收 API 信任当前 service transport 和受控运行器，检查字段/父链一致性，不构建密码学证明或跨进程 checkpoint 认证。

## 3. 完整 P 与 F 的同信息前提

本批发现：P 按任务排序返回，F 的 MTR 保留 provider 原窗口顺序。相同目标集合不足以证明相同输入，dtype 也会影响下游计算。因此比原 T3 文件清单多修改了两个工具文件，补齐必要契约，而不是重写服务或增加机制。

完整 P 的可选 `context_certificate` 只在当前实际返回 history 加合法旧 history 回执覆盖所有 provider 目标时签发，包含输入目标顺序、数组 dtype、eligible 字段是否存在，以及实际绑定的 predictor 描述。它只列已经交付/确认收到的目标；普通 references 不插入仅用于证明的额外目标。完整证明整体计入最终 UTF-8 cap；装不下时省略证明，保持“未证明完整”。被裁剪且未购完整的 P 不带未知目标清单。

接收端规则：没有完整证明时，对累计已购 P 按 handle 排序，以明确的接收端 dtype 重建；有完整证明时，按 provider 实际顺序和 dtype 无损重建。二者都是已购信息决定的确定顺序。首次收到证明，即使该响应 `no_new_fields`，若有效输入从 subset/sorted 变为 full/provider-order，也会执行并单独计费一次本地预测。仅新增 F 的 anchor 不扩大本地 P 的输入父集。

`FrozenPredictor` 不加载模型，只包装已经注入的 callable、可读模型版本和冻结执行配置，赋予同进程运行绑定 ID。共享同一实例是本批的模型一致性前提；调用方负责冻结权重和执行设置。稳定 `name/revision` 或不同包装实例的相同字符串均不充分。绑定信息属于旁路/付费响应，不作为提示中的字段身份。

只有在完整输入证明成立、实际 predictor 绑定相同，并且**已购 F 自身绑定也匹配**时，才比较按 handle 对齐的全部六模式、六驾驶时刻 forecast、模式分数、时间和 fallback 状态。数值与语义精确相等才在 Z 归为一份数值、多个来源引用；E 仍保留 F 的购买事实。不同上下文或不同绑定即使数值碰巧相等也不去掉。

本批没有复杂 fingerprint、跨实例回执持久化、额外模型调用分支或策略动作屏蔽。真实 MTR 50 步全数组复算仍未新运行；当前比较的是 wire 中合法返回的六模式六时刻字段，fake predictor 只验证契约。

## 4. 共同 receiver 和成本边界

v2 复用 compact_v1 编码及原 direct prompt 尾部任务，显式标记 `toolv2x_driver_evidence_v2` 和 `source_blocks_v2`，不放宽旧 evidence 的字段白名单。

候选以当前 anchor 距离排序，平局按 provider、handle、字段种类和上下文等稳定身份处理。按“共享 anchor + 完整 history/forecast”尝试装入，兼容上下文的 history/forecast 可以共用对象中的 anchor。预算不足时丢掉整 bundle，继续尝试后续目标；E 和历史派生不会被删除。新近目标可以替换此前远目标的入模位置，原因有记录。所有未来新比较臂应使用此共同 receiver；当前距离排序可能抵消任务检索收益，T3 没有改成任务优先接收器。

本车块只由本车输入及公共预算构造，远端到达不改变它。`peer_reserve=0` 可构造 Ego-max-context；旧五策略的 v1 路径未迁移。默认 receiver 配置为 context 4096、generation reserve 256、peer reserve 1536、两位小数，并完整记录在 prepared 中。

请求 ID、工具名、current/change 标签、receipt IDs、价格不进入驾驶文本；数值与已购边界元数据决定 Z。已付费 source observation 保留 coverage=not_established；重传本身不重复追加提示。六模式分数分别显示舍入可能令总和略大于 1，v2 显示校验允许每模式至多半个最小显示单位的总偏差，不归一化或改写原始 E。

`receiver_events.receiver_seconds` 包括接收验证/更新，`model_seconds` 是其子项，不应再重复相加；`prepared.input_build_seconds` 是另外的提示选择/编码阶段。服务自身成本保留原定义。T4 后续需要将这些互不重叠阶段接入 episode 总费用；本批没有三次 GoT 或真实网络时延数据。

模型派生失败抛出带 ledger 的 `EvidenceUpdateError`：合法已购 E、回执和已花成本仍保存，不写入半份 derived。解码/引用/费用验证失败也保留原始 received attempt 和 reported cost，但不将坏响应登记为合法 E；失败状态不能作为成功驾驶输入。旧合法 F 缓存继续保留，不重建外部缓存。

## 5. 验证与实际边界

T3 新增 21 项轻量测试，覆盖完整/部分上下文、原目标顺序/dtype、两次 P 累计补全、证明恰好超预算及 no_new_fields 首次取得证明、重传/幂等、F→P 引用、不同上下文/绑定、准确父链、坏 wire/派生失败留账、相同 E 的提示一致性、容量替换、字段回溯及 codec 白名单。

原工作站 Python 3.8.10 / NumPy 1.24.2 的轻量套件：原 135 项 + T3 21 项 = **156/156**。独立只读审查针对 T3、task_spec、vehicle_tools、compact_evidence 运行 **77/77**，两项发现均补了失败反例并修复。旧 v1 query 与六个 helpers 的 AST 一致。

另在原 llava 环境使用本机原 tokenizer 和 GoT prompt 包装执行共享输入检查，**3/3 通过**：新增 v2 接收预算测试、旧保存提示逐字复现、旧训练/生成提示一致性。没有模型权重加载、真实 MTR/GoT 前向或生成；旧路径生成入口使用现有测试替身。旧保存提示检查触发 tokenizer 默认长度提示（2456 > 2048）；本次按已有执行器显式 4096 上限核对容量，不从 tokenizer 测试推断模型实际长上下文效果。两目标合成窗口、64 个配置的 feature tokens、默认 ExecutionSpec/receiver 预算的实际容量示例：

| 顺序/步骤 | 请求 B | 响应 B | 已购字段 E | 派生字段 | 入模 primary bundles | 输入 tokens |
| --- | --- | --- | --- | --- | --- | --- |
| P→F：P 后 | 614 | 3943 | 4 | 2 | 4 | 2011 |
| P→F：F 后 | 1747 | 4920 | 6 | 2 | 4 | 2011 |
| F→P：F 后 | 614 | 3809 | 4 | 0 | 2 | 1181 |
| F→P：P 后 | 1755 | 5062 | 6 | 2 | 4 | 2011 |

每项另预留 256 个生成 tokens，均在 4096 内。这是原 tokenizer 加合成 predictor 的接收容量检查，不是实际数据分布容量保证、驾驶改善或真实同模型效果实验。将来正式运行还需预设预算档的代表性数据检查，不因此暂停全部方法推进。

**停止点：T3 的代码与契约验证完成。T4 的实际 GoT 交替修订、第二请求/STOP，以及训练和新方法实验均未实施。**
