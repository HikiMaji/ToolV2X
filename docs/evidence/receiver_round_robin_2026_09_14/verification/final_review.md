# Receiver round-robin final whole-branch review

Date: 2026-09-14. Reviewer scope: read-only, approved two-step plan in `docs/superpowers/plans/2026-09-14-receiver-round-robin.md`; source/test diff `/tmp/toolv2x-receiver-final-review.diff`; current worktree `/root/autodl-tmp/ToolV2X/.worktrees/receiver-round-robin`; actual run `/root/autodl-tmp/ToolV2X/outputs/receiver_round_robin_smoke_2026_09_14_v2` (RUN below). No model/tokenizer calls, training, downloads, additional frames, or source changes were performed by this reviewer. Only this review file was written.

## Technical verdict

Spec compliance **PASS**. Code quality **PASS**. No must-fix findings. Final documentation and linked evidence also PASS. The branch satisfies the approved common receiver plus bounded real two-frame integration scope.

## Source and common receiver

- The supplied five-file source/test diff exactly equals the worktree diff. `git diff --check` passes and all five changed Python files parse.
- `src/planning/context.py:91` introduces one pure stable ordering helper reused by the audit. The v2 path applies it after the existing `remote_units()` provenance validation and legacy sorting, using `(source, track_id)`. It preserves target first appearance, within-target order, and every existing unit, including different forecast contexts.
- v1 remains the default at `context.py:113`. The existing local construction, rounding, prompt packing, field provenance, full-token capacity check, oversize-unit continuation, and budgets are unchanged. v2 uses no policy/method name, tau1, GT, unknown peer field, or new input.
- `src/planning/v2vgot.py:281` only expands the accepted declared receiver versions. Original feature count, prompt equality, actual tokenizer, generation reserve, context and evidence checks remain in force.
- New tests exercise coverage under the same token capacity, continued rounds/context retention, oversize skipping, local invariance, ledger immutability, supported v2 reaching the driver and unknown version rejection. No dependency or speculative abstraction was added.

## Independent actual-artifact audit

The reviewer ran one read-only Python/NumPy assertion audit over raw tasks, provider records, ledger snapshots, raw wire, frozen sources and offline labels. All assertions passed:

1. All **158** frozen source/vendor files equal current source bytes; all five frozen wrapper/config/index files equal their executed copies. The selected two-row index equals the prior archive. Snapshot time precedes inference, which ends before offline label access.
2. Exactly **12** completed tasks, **0** task failures, **24** real GoT/Q9 attempts: Ego 1, alternating 3, one-shot 2 per frame/receiver. All 24 plans have real Q9/language-model execution, parsed trajectories and Q8 disabled. No collection-reuse record occurs.
3. All **4** alternating F(change) requests have `tau_new` exactly equal to their actual stage-1 output and `tau_old` equal to stage 0. Each equals the corresponding current provider request. These are fresh second requests after actual generation.
4. All **16** provider primitives completed. F model-plus-fallback target counts equal the full provider context size. All **16** final provider receipts are distinct across tasks; ledger response and decoded receipt wire equal that task's actual provider response. They are not old archived replies.
5. All **24** prepared inputs declare the actual original GoT token counter, 540 feature tokens, and input plus 256 generation reserve within 4096. Their output cost token counts agree. Every admitted field ref occurs in that plan's selected ledger snapshot; observation receipts match that snapshot. Local evidence is unchanged across matched v1/v2 stages.
6. Actual encoded request and response wire lengths independently reproduce task and offline byte totals. All driver counts, input/output token totals and non-overlapping measured compute totals independently match offline evaluation. Nested generation/provider/receiver MTR durations are not added twice.
7. Old v1's **12** stages exactly reproduce prior prompt, local/remote evidence, selection, raw answer and trajectory. All 12 tasks' freshly generated feature/local-forecast arrays exactly match the corresponding prior inputs. This checks reproducibility without importing old task responses into execution.
8. Independent six-waypoint Euclidean errors from the two offline observed-trajectory labels reproduce ADE3/FDE3 for all 12 tasks to numerical tolerance.

Final v1 retains two forecast units for one target (g5526: 20; g7007: 0). Final v2 retains two targets (g5526: 20 and 6; g7007: 0 and 5). Same target/kind membership does not establish identical Z: g5526 v2 alternating and one-shot have different final evidence/prompts (3555 versus 3472 input tokens).

| Frame | Receiver | Arm | ADE3 (m) | FDE3 (m) |
| --- | --- | --- | ---: | ---: |
| 5526 | v1/v2 | Ego | 0.948969 | 1.854059 |
| 5526 | v1 | alternating / one-shot | 0.700788 | 1.292016 |
| 5526 | v2 | alternating | 0.936881 | 1.778120 |
| 5526 | v2 | one-shot | 0.849000 | 1.643314 |
| 7007 | v1/v2 | Ego | 0.410510 | 1.189432 |
| 7007 | v1 | alternating / one-shot | 1.623345 | 4.309533 |
| 7007 | v2 | alternating / one-shot | 1.398179 | 2.019000 |

This is a mixed two-frame imitation result: v2 worsens g5526 relative to v1; g7007 improves relative to v1 but remains worse than Ego. It does not support a general driving-benefit or policy-quality claim.

## Validation evidence inspected

- `/tmp/toolv2x-receiver-full-light.log`: 302 tests OK.
- `/tmp/toolv2x-receiver-full-model-env-final.log`: 358 tests OK in 332.041 seconds.
- `/tmp/toolv2x-receiver-resource-regression.log`: the two initially missing-resource fixture tests now OK.
- `/tmp/toolv2x-receiver-fixed-controls.json`: Ego/P/F/PF/rule common-v2 synthetic contracts all PASS; these are not five-policy real-model evaluations.
- `/tmp/toolv2x-receiver-r1-review.md`: independent R1 spec/quality PASS, supported by unchanged final source diff.
- Initial full-suite fixture path errors are environment setup failures, retained in their original log. The complete corrected suite passes; no regression was hidden by only rerunning the failing tests.

## Claim and execution boundaries

The initial RUN sibling `_v1` failed all 12 tasks before episode creation because of a wrong V2V-GoT data root; it had zero GoT/provider calls and is preserved as an environment failure. The corrected `_v2` is a new actual run with explicit roots and frozen snapshots.

The real input uses point-cloud feature tensors and text, not RGB images. These are the existing adapted frozen GoT epoch01 and causal CMP MTR epoch09; Q8 is skipped, portable MTR operators are used, native CUDA parity and historical training provenance are not newly verified. The receiver smoke neither trains nor establishes an official full-pipeline reproduction.

All 24 GoT attempts and all actual service costs must be reported. Existing one-shot per-RPC versus alternating primitive response-cap differences remain; this is diagnostic, not a fair strong-one-shot/value-policy comparison. Labels are read only after inference for observed-trajectory imitation, not online selection or collision/safety/optimal-planning evaluation. No new training, frames, policy fitting, or branch enumeration was performed.

## Final report check

**PASS**, after the writer confirmed the final report at `docs/receiver_round_robin_smoke_2026_09_14.md` and evidence/figures were ready.

The final report accurately states 12 successful tasks, all 24 actual GoT calls, mixed errors, causal stage-1 F binding, different g5526 forecast contexts, unchanged g7007 final trajectories between the two v2 arms, and the two-receiver cost totals. Independent raw checks reproduced all eight F ranking/returned-field/context rows and both receiver aggregate cost rows. All Markdown artifact/figure links resolve. Copied JSON evidence equals RUN data; repository CSV records equal the raw CSV after the declared line-ending normalization.

The report removes pending-run wording and discloses the preserved environment failure, no RGB, point-cloud inputs, direct Q9/Q8 skip, adapted frozen model versions, unverified historical training provenance/native-operator parity, offline-only labels, diagnostic wire-cap mismatch, timing exclusions/measurement limitations, and lack of general policy/driving-benefit claims. No report correction is required.

Final result: **spec PASS; quality PASS; bounded real integration PASS; must-fix none**. This verdict authorizes no broader scientific claim or additional experiment.
