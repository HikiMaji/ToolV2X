# ToolV2X receiver R1 spec and quality review

Date: 2026-09-14

Scope: read-only review of `/tmp/toolv2x-receiver-r1-review.diff` against worktree `/root/autodl-tmp/ToolV2X/.worktrees/receiver-round-robin`, before real-model smoke. No model, training, new frame, cache rebuild, or git mutation was performed.

## Verdict

- Spec compliance: **PASS**.
- Code quality: **PASS**.
- Must-fix findings: **none**.
- Gate: **GO for the bounded two-frame real-model smoke**. This is a code/contract readiness verdict, not evidence of trajectory benefit or value-policy quality.

The worktree diff exactly matches the supplied review diff. `git diff --check` and AST parsing of all five changed Python files pass.

## Spec compliance evidence

1. **Explicit opt-in with unchanged v1 default.** `build_task_plan_input` still initializes `version='toolv2x_receiver_v1'` at `src/planning/context.py:113`; v2 is only selected when explicitly supplied. Relative to `HEAD`, the v1 construction, local selection, token calculation, remote rendering, admission, and result schema are unchanged. The only v1-path semantic change is that configuration validation now recognizes one additional declared version.

2. **Old order first, then stable target round robin.** `remote_units(ledger)` still performs the provenance validation and deterministic legacy sort (`src/planning/evidence.py:257-311`). Only after that, v2 calls `target_round_robin` (`src/planning/context.py:154-156`). The helper uses first target appearance and retains append order within each target (`src/planning/context.py:91-99`).

3. **Correct target identity and context preservation.** The production grouping key is `(unit['object']['source'], unit['object']['track_id'])`, so identical track handles from different providers remain different targets (`src/planning/context.py:155-156`). The helper neither copies, merges, filters, nor deduplicates units. Context-distinct forecast units therefore remain separate and retain legacy within-target order. Existing receiver rendering continues to merge only compatible different field kinds while preserving same-kind forecast units (`src/planning/context.py:178-201`). A separate read-only probe with `p1/track7`, `p2/track7`, and `p1/track9` passed the expected source-aware order and object-identity preservation.

4. **Continue after an oversize unit.** The admission loop appends only fitting trials and never breaks on a rejection (`src/planning/context.py:203-210`). The new contract test demonstrates rejection of an oversized `track7/forecast` followed by admission of `track9/forecast` and later `track7/history`.

5. **Budgets and ego/local block unchanged.** v2 reorders only `units`; local construction and `local_limit = context_limit - generation_reserve - peer_reserve - extra_prompt_reserve` are unchanged (`src/planning/context.py:157-170`). Each remote trial still uses the same exact prompt counter and `context_limit`/generation reserve check (`src/planning/context.py:203-210`). Numeric precision remains fixed at two decimals. Tests confirm the v1/v2 local evidence is identical, retained-token total is bounded, and ego-only output is unchanged.

6. **Common receiver with no method, tau1, GT, or unknown-peer dependency.** The receiver helper accepts only units and a target-key function. Production use derives that key solely from validated visible unit source and track ID. The common `run_task_episode` path passes any method/control's declared `receiver_spec` into the same builder (`src/planning/method_episode.py:242-250`); there is no policy/method branch in the new implementation. No new data access, label access, future access, service call, model call, or filesystem access was introduced.

7. **GoT execution gate preserved.** In `V2VGoTPlanner.plan_prepared`, the sole changed condition expands the receiver-version allowlist from v1 to `(v1, v2)` (`src/planning/v2vgot.py:275-289`). Prompt equality, feature-token count, context limit, 256-token generation reserve, peer reserve, original tokenizer counting, exact input token total, overflow prevention, prepared/execution flags, admission report, and evidence version checks remain intact. The new driver test confirms declared v2 reaches `_generate` once with the original 256-token budget and an unknown version is rejected before a second call.

8. **Audit uses the production ordering helper.** `scripts/audit_t9_admission.py:18-19` imports `target_round_robin`; its frozen-v1 counterfactual simulation uses the same source-plus-track key (`scripts/audit_t9_admission.py:296-322`). The audit's actual saved-v1 replay logic remains unchanged.

## Test and review evidence

- Baseline log: 25 tests PASS (`/tmp/toolv2x-receiver-baseline.log`).
- Pre-implementation receiver RED: both new receiver tests fail because v2 is rejected (`/tmp/toolv2x-receiver-red-valid.log`).
- Pre-implementation driver RED: the new driver-gate test fails because v2 is rejected (`/tmp/toolv2x-receiver-driver-red.log`).
- Receiver GREEN: 27 tests PASS, including both new receiver tests and the existing audit helper checks (`/tmp/toolv2x-receiver-green.log`).
- Driver GREEN: 4 focused driver/tokenizer tests PASS (`/tmp/toolv2x-receiver-driver-green-final.log`).
- Independent static checks: supplied diff equals worktree diff; whitespace/error check passes; all five changed Python files parse; direct multi-provider same-track helper probe passes.

## Code quality assessment

The implementation is minimal and localized: one 9-line pure ordering helper, one guarded v2 application point, one extra accepted version in the existing GoT gate, and audit reuse of the helper. It preserves stable ordering through Python insertion-ordered dictionaries, does not mutate input units, and avoids duplicating receiver logic in the audit. Tests cover the material failure modes: v1 versus v2 selection, context-distinct units, full-order cycling, oversized-unit continuation, local/ego invariance, ledger immutability, unknown-version rejection, and preservation of the original execution budget checks.

No blocking correctness, contract, or maintainability issue was found in the reviewed diff.
