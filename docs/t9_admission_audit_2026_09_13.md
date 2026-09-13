# T9 两帧 receiver admission 审计

完成时间：2026-09-14（UTC+8）。审计身份沿用 2026-09-13 的 T9 两帧真实联调。本报告只覆盖 `docs/9-13-3.md` 的 B，以及 C 中 receiver v2 候选讨论；A1/A2 由主任务另行实现和汇报。

## 结论

保存的六个任务、12 个 plan stage 全部可复现：用本地原 GoT tokenizer 重新构造 prompt 后，保存的 local/remote evidence、admission field refs、object index/field position、完整 prompt 和 input token 总数均相等；provider 的保存记录也与 ledger receipt 逐项相等。审计只加载 tokenizer 计数，没有构造或调用 GoT、MTR 或其他模型，也没有读取 offline label。

两个远端臂的 provider 排序确实不同，但 receiver v1 不按 provider rank admission。它先按 anchor 距离排列 atomic unit，再把同一近目标的不同 field/context unit 连续尝试。在 4096 context、256 generation reserve、1536 peer reserve、540 feature tokens 下，每个远端阶段只保留 2 个 unit，且都属于最近的同一目标。因此：

- g5526 最终都只保留 target 20；alternating 和 one-shot 的 final input 都是 3573 tokens。
- g7007 最终都只保留 target 0；两臂的 final input 都是 3572 tokens。
- 两帧中，alternating 与 one-shot 的最终 remote Z、完整 prompt 和保存轨迹均相等。该事实说明 provider 差异没有传入最终 GoT 输入；它不证明 provider mode 等价，也不证明任何 receiver v2 会改善驾驶。

## 数据和复算边界

输入固定为：

- `/root/autodl-tmp/ToolV2X/outputs/t9_real_smoke_2026_09_13_v1/tasks/g{5526,7007}_{Ego,alternating,one_shot}.json`
- 各任务对应的 `inputs/<task>/provider_records.json`
- tokenizer `/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b`

复算直接调用现有 `planning.evidence.remote_units()`、`planning.context.build_task_plan_input()`、`planning.inputs.make_prompt()` 和原 `prompt_tokens()`。tokenizer 元数据的旧 warning limit 是 2048；审计只把 warning threshold 提到冻结 receiver 的 4096，不截断、不改变 token ID。每次计数都包含 540 个 feature tokens。

同时修正 request/source-stage 语义并覆盖全部 plan dropped refs 的最新输出位于 `/root/autodl-tmp/ToolV2X/outputs/t9_admission_audit_2026_09_13_v3/`；此前 v1/v2 目录保持原样：

- `remote_units_actual.csv`：64 个实际 remote-unit stage rows；包含 request origin、provider rank、context、距离、standalone 增量、实际 attempt/cumulative tokens、admission/drop、object index/section 和精确 ref。
- `remote_units_counterfactual.csv`：相同 64 个 unit 的 round-robin 模拟顺序及实际 tokenizer token attempt。
- `provider_rankings.csv`：每个 provider ranking item 到 response record/reference 的精确映射。
- `dropped_fields.csv`：12 个保存 stage 的全部 482 个 dropped ref、origin、field group 和原因；其中 local 408、remote/receiver-derived 74，包含 Ego 与所有 initial stages 被 local reserve 裁掉的 ego fields。
- `plan_stages.csv`、`token_group_summary.csv`、`reproducibility.json`、`summary.json`：阶段归约、field group token 成本与断言结果。

上述八个输出文件也完整复制到 `docs/evidence/t9_admission_audit_2026_09_13/`，包括逐 unit actual/counterfactual、全部 dropped fields 和 summary/reproducibility JSON。输入的旧 T9 归档没有写入或覆盖。

### Request stage 与首次可见 stage

`request_stage` 现在只取自 episode 的 `request_started` 事件；`first_visible_plan_stage` 是该 primitive receipt 首次进入保存 ledger plan 的 stage。脚本同时保存 `outer_request_id`，并断言每个完成 receipt 都满足 `first_visible_plan_stage = request_stage + 1`，且首次可见 plan 的 `request_id` 等于外层请求：

| arm | primitive | outer request | request stage | first visible plan stage |
| --- | --- | --- | ---: | ---: |
| alternating | P `q0` | `q0` | 0 | 1 |
| alternating | F `q1` | `q1` | 1 | 2 |
| one-shot | P `q0` | `bundle0` | 0 | 1 |
| one-shot | F `bundle0:second` | `bundle0` | 0 | 1 |

one-shot 的两个内部 primitive 没有伪造独立 episode stage；它们都由 stage 0 的同一个外层 bundle 请求产生，并在 stage 1 一起首次可见。

## Provider ranking 到 receiver ordering

P(current) 的 provider 排序在两种远端臂间相同：g5526 为 `[20,0,4,6]`，g7007 为 `[0,5,11,21]`。F 的排序为：

| frame | arm | F request | provider ranking | final retained target |
| --- | --- | --- | --- | --- |
| g5526 | alternating | F(change) | `[6,20,3,8]` | 20 |
| g5526 | one_shot | F(current) | `[3,6,20,4]` | 20 |
| g7007 | alternating | F(change) | `[0,21,5,11]` | 0 |
| g7007 | one_shot | F(current) | `[0,5,2,21]` | 0 |

receiver v1 的排序键为 `(anchor distance, provider, track_handle, field kind, context identity, field identity)`。稳定 tie rule 全部来自现有 `remote_units()`；审计没有另造 rank。关键 anchor 距离如下：

| frame | receiver target order and distance (m) |
| --- | --- |
| g5526 | `20:0.627452, 6:6.519300, 4:9.585312, 3:15.109988, 0:23.996878, 8:53.349402`（具体可用 target 随 arm 不同） |
| g7007 | `0:0.163538, 5:14.197674, 2:15.951906, 11:16.774911, 21:48.953030`（target 2 只在 one-shot F 中） |

不同 context 不合并。P 后，最近目标的 `receiver_acquired_subset` forecast 位于 remote object index 0，history 位于 index 1。最终阶段，最近目标的 `provider_full_at_t` forecast 位于 index 0，`receiver_acquired_subset` forecast 位于 index 1。两者都在 `remote` prompt section；每个字段在 compact `object_shared` 或 `object_rows` 中的精确列位置见 `remote_units_actual.csv`。所有未保留的 active remote unit 原因均为 `context_budget`。

forecast 行保留协议中的精确 `context_scope`；history 协议本身没有该字段，因此 CSV 中该列为空，同时保留完整 `context_version`，没有把 history 虚构为某种 forecast scope。

## 实际 admission 与覆盖

所有 Ego plan 和四个远端臂的初始 plan 均为 `remote_units_total=0`。有远端证据的六个 stage 如表所示。group 顺序为 `history / receiver-subset forecast / provider-full forecast`。

| task stage | available units / targets | available units per target | actual retained / targets | actual groups | round-robin retained / targets | simulated groups | extra targets |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| g5526 alternating, P 后 | 8 / 4 | `0:2,4:2,6:2,20:2` | 2 / 1 (`20:2`) | `1/1/0` | 2 / 2 (`20,6`) | `0/2/0` | +1 |
| g5526 alternating, final | 12 / 6 | `0:2,3:1,4:2,6:3,8:1,20:3` | 2 / 1 (`20:2`) | `0/1/1` | 2 / 2 (`20,6`) | `0/0/2` | +1 |
| g5526 one-shot, final | 12 / 5 | `0:2,3:1,4:3,6:3,20:3` | 2 / 1 (`20:2`) | `0/1/1` | 2 / 2 (`20,6`) | `0/0/2` | +1 |
| g7007 alternating, P 后 | 8 / 4 | `0:2,5:2,11:2,21:2` | 2 / 1 (`0:2`) | `1/1/0` | 2 / 2 (`0,5`) | `0/2/0` | +1 |
| g7007 alternating, final | 12 / 4 | `0:3,5:3,11:3,21:3` | 2 / 1 (`0:2`) | `0/1/1` | 2 / 2 (`0,5`) | `0/0/2` | +1 |
| g7007 one-shot, final | 12 / 5 | `0:3,2:1,5:3,11:2,21:3` | 2 / 1 (`0:2`) | `0/1/1` | 2 / 2 (`0,5`) | `0/0/2` | +1 |

因此，“禁止同一 target 连续占多个 unit”的稳定 round-robin 在每个已购证据 stage 都能在相同总 budget 下多覆盖 1 个 target；保留 unit 数仍为 2。这里是 prompt admission 模拟，不含 GoT 输出，不能推导 ADE/FDE 或安全收益。

## Token 成本分解

`standalone_increment_vs_ego` 的定义是：同一 stage 的 local-only prompt 加上“remote observation header + 仅该 unit”后，与 local-only prompt 的 token 差。它不是实际顺序中的 marginal cost；compact columns/shared values 和同 object 合并使成本不可简单相加。实际 `attempt_input_tokens` 则是在当时已经 admitted 的 unit 上再尝试该 unit 后的完整输入，失败时 `cumulative_input_tokens` 保持不变。

| frame | ego-only | empty remote header | history standalone | receiver-subset forecast standalone | provider-full forecast standalone |
| --- | ---: | ---: | ---: | ---: | ---: |
| g5526 | 2292 | 2405 | 777–794 | 714–737 | 705–743 |
| g7007 | 2300 | 2413 | 789–805 | 726–746 | 711–727 |

实际关键 token growth 为：

| stage | admitted chain (`attempt_input_tokens`) | with generation reserve | first rejected attempt |
| --- | --- | ---: | --- |
| g5526 P 后 | target20 subset forecast `3026` → target20 history `3701` | 3957 | target6 subset forecast `4286 + 256 > 4096` |
| g5526 final，两臂相同 | target20 provider-full `3016` → target20 subset `3573` | 3829 | target20 history `4194 + 256 > 4096` |
| g7007 P 后 | target0 subset forecast `3033` → target0 history `3702` | 3958 | target5 subset forecast `4300 + 256 > 4096` |
| g7007 final，两臂相同 | target0 provider-full `3017` → target0 subset `3572` | 3828 | target0 history `4191 + 256 > 4096` |

在第一个失败后，receiver v1 仍继续尝试后续 unit；完整表显示它们也全部超限。round-robin 的第二个已接纳 unit 分别得到 g5526 P `3477`、g5526 final `3472`、g7007 P `3496`、g7007 final `3480` input tokens，均未扩大 context/generation budget。

## Counterfactual 算法

本次只模拟 `stable_target_round_robin_v1`：输入仍是保存 stage 中 `remote_units()` 给出的 exact paid/receiver-derived units。目标顺序按其在 receiver-v1 顺序中的首次出现固定；每个目标内部顺序不变。第一轮每目标取第 1 个 unit，第二轮每目标取第 2 个，依次进行。每次使用原 tokenizer 重新渲染完整 prompt；`input + 256 <= 4096` 时接纳，否则跳过该 unit 并继续。相同 provider/track 但 context 不同的 unit 全部保留为不同候选。

这不是 hard unique-target cap。round-robin 在所有目标各尝试一轮后允许第二轮；它不会永久禁止同 target 多 field。模拟不读 GT、offline label、未购买 peer 信息或 alternating 独有的 `tau1`，所有 arm 使用同一顺序和预算。

## Receiver v2 候选（仅提案，不实现）

1. **Stable target round-robin。** 直接采用上面的共同规则，完成一轮目标覆盖后才尝试同 target 的第二 field。优点是本次两帧同预算都从 1 个目标扩到 2 个。公平性风险是 target diversity 本身未必等于任务相关性，且大 unit/compact merge 的顺序效应仍可能影响后续容量；本次没有模型输出证据证明收益。
2. **共同的 per-target field quota。** 例如先对所有 target 固定 `max_fields_per_target=1`，所有目标尝试完后再以固定 `max_fields_per_target=2` 上限做第二轮。这是 hard quota：即使仍有空间也拒绝超过上限的同 target unit，和 round-robin 的“延后但不永久禁止”不同。公平性风险是 track 碎片或场景中只有一个关键目标时会浪费预算，也可能压掉 context 不同且确有互补性的 subset/full forecast；quota 必须对 Ego/P/F/PF/one-shot/alternating 完全相同。
3. **共同的 task-relevance / marginal-token-cost score。** 只使用各方法当时已经合法拥有的 anchor、已购/receiver-derived field、共同初始 `tau0` 和真实 marginal token cost；固定 score 公式与 tie rule 后统一排序。禁止使用 GT、未购买字段或 alternating 的修订 `tau1`。公平性风险是 score 权重可能系统偏向某个 field/tool，compact merge 使 marginal cost 依赖先前选择，且不同方法拥有的合法候选集仍不同；需要预注册并用强 common controls 审核。

这些候选都不删除 receiver-subset 或 provider-full forecast，不扩大 window，不给 alternating 特权，也不改变上游 provider ranking。当前证据只支持“receiver admission 是这两帧的差异消失点”，不支持它是全部场景的主要瓶颈。

## 验证与停止

本次生成命令如下。脚本对非空 output directory fail closed，因此独立复核时应指定新的目录。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python scripts/audit_t9_admission.py \
  --output-dir /root/autodl-tmp/ToolV2X/outputs/t9_admission_audit_2026_09_13_v3 \
  --evidence-dir docs/evidence/t9_admission_audit_2026_09_13
```

可直接运行的独立复核命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python scripts/audit_t9_admission.py \
  --output-dir /root/autodl-tmp/ToolV2X/outputs/t9_admission_audit_2026_09_14_recheck_v3
```

专门测试为 `4/4 PASS`：覆盖 context-distinct unit 不被 round-robin 合并、超限 unit 跳过后的精确 cumulative tokens、receipt 首次可见 stage，以及 alternating/one-shot 从真实外层 request event 到 primitive receipt 的 stage 映射。真实审计为六任务、12 plan stages、64 remote-unit rows，全部 reproducibility assertions PASS；482 个 dropped CSV rows 的 ref、reason 和 origin 与各原始 plan admission report 逐项相等。

本批停在离线 tokenizer 审计和 admission counterfactual。没有新训练、新采集、GoT/MTR inference、trajectory 生成、receiver 修改或旧 T9 结果修改。
