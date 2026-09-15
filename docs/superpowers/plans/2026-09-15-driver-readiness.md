# Shared Driver Readiness — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. No nested subagents. The user approved the Stage A proposal with “开始”.

**Goal:** Complete measurement, evidence-use auditing, explicit training/validation objectives, and a frozen preparation package before real shared-driver collection or training.

**Architecture:** Keep one existing P/F executor and one shared numeric driver. Correct accounting at the common executor/evaluator boundary; derive evidence-use statistics from existing validated archives. Add an opt-in training-config version for stage supervision and periodic validation, preserve old configuration/resume semantics. Freeze a metadata-only preparation package using existing recording roles and causal sample identities.

**Tech Stack:** Existing Python, NumPy/SciPy, PyTorch; standard unittest. Ordinary Python for torch-free contracts; `/root/autodl-tmp/conda-envs/llava/bin/python` for synthetic CPU training checks. No new dependencies.

**Spec:** `docs/9-15-1.md`, `docs/review_9_15_1_response.md` (Stage A accepted). This plan resolves implementation choices without extending the P/F method.

## Global Constraints

- Physical time-t observations only. P never runs MTR; P history is causal tracking-state motion proxy. F uses complete legal peer context before task ranking; ego trajectories do not condition MTR prediction. Two cars, P/F/STOP, maximum two remote calls and three actual driver attempts.
- Existing query/decode/receipt/ExecutionSpec semantics unchanged. No forged manifests, provider replies or numeric driver outputs. Receipt/derived/admitted dependencies remain traceable; path is not semantic identity.
- Preserve old v1 wire/golden behavior, old GoT, old training config/resume, all original artifacts. Correcting historical unknown timing interpretation may change derived cost evaluation; never rewrite archived measurements as zero.
- New training protocol and default preparation use the same observations_only StructuredDriverSpec for Ego/P-state/F/PF. P-local is a separately recorded equivalence/compute-delegation control. Do not enlarge 64/4 capacity.
- No real dataset training, real model generation, new method runs, real provider inference or dataset/cache rebuilding. Synthetic CPU forwards/gradients/temporary optimizer/resume checks allowed; fake predictor is contract evidence only.
- Read existing metadata to freeze roles and indices; no future labels read during preparation/selection. No performance-based frame choice. No new detector, prediction backbone, RL, peer sorter, RSU/I or GoT rebinding.
- Full configuration records experimental numbers. No content hashes or SHA256; use stable versions, readable source paths, direct comparison and existing load identities.
- No automatic push. Integrate reviewed code into local main while preserving all pre-existing dirty documents; publish logs/reviews before cleaning this task workspace.
- Baseline is 322 lightweight /416 model-environment tests. Run changed-area checks after each fix; required complete regressions once after final review, with explicit external resource environment. No real generate scripts.

## Task 1: Complete request-decision cost accounting

**Files:** `src/planning/method_episode.py`, `src/evaluation/framework.py`; narrowly update archive consumers only if required. Tests in existing method episode/evaluation/controls files or `tests/test_decision_accounting.py`; add new torch-free test module to `scripts/check_review.py` if used.

**Interfaces:** `run_task_episode` and `_method_cost(task)` stay public/common paths. New episodes carry an explicit compute-accounting version independent of control kind. Every actually started decision/preflight interval has an auditable stage identity and either a complete duration or unknown/incomplete accounting. Keep `control_seconds`, `known_cost`, `cost_complete`, `total_compute_seconds` and `end_to_end_seconds` meanings; no duplicate nested timing.

- [x] RED: with a patched monotonic clock, policy advances 5 seconds then STOP. None and explicit feedback both report control_seconds >=5; previous None path yields 0. Assert evaluation rejects missing duration as unknown, retains known partial costs, and never labels it complete.

```python
clock = [0.]
def policy(state):
    clock[0] += 5.
    return dict(tool='STOP', mode=None, reason='timed_fixture')
# Existing real executor + fixture driver; no sleep or actual model execution.
assert evaluated['control_seconds'] >= 5.
```

- [x] Cover no-call budget STOP, policy exceptions before decision append, dispatch failures, progress interruption between decision and completed accounting, valid prefix reuse, exact-repeat with only final actual decision, and old archives without timing. Do not charge stages which never attempted a decision; distinguish missing from actual zero. Existing explicit timing records remain usable. Persistence callbacks stay outside timed computation.
- [x] Implement one shared accounting contract; keep role/control configuration separate. Document which management/loading/network/I/O work remains excluded. Bind or validate the new version in affected consumers if they otherwise misinterpret its completeness.
- [x] Run covering tests and relevant existing method control/evaluation/query/bundle tests. Report commands and actual results. Avoid full suite until final integration.
- [x] Commit only owned files, subject `Account for every actual request decision`; write task report and get independent spec/quality review.

## Task 2: Derived structured evidence-use audit

**Files:** Create `src/evaluation/structured.py`, `tests/test_structured_admission.py`; minimal integration into `src/evaluation/framework.py` and `scripts/check_review.py`. Do not modify provider, receiver capacity, field_groups or ledger semantics.

**Interfaces:** `audit_structured_episode(episode) -> list[dict]` returns JSON-native per-plan audit from a validated numeric episode. Integrate into ordinary method evaluation row output under an explicitly named structured audit field; old GoT rows remain compatible. No model, tokenizer, labels or filesystem reads in this helper.

- [x] RED using existing real synthetic P/F service fixtures: local x=10 /peer x=30, max_entities=1 gives acquired remote anchor+history, both dependency-closed, zero direct remote primary fields. Capacity2 gives a direct remote history at actual observation slot. Required module assertion fails before implementation.

```python
audit = audit_structured_episode(ep)
assert audit[stage]['direct_remote_primary_refs'] == []  # dropped by capacity
# Exact published output keys additionally documented in task report.
```

- [x] Report new versus previously acquired remote refs at each ledger stage; remote/local direct primary refs from `use='tensor'`, valid mask and tensor_locations; observations/forecast sets/modes/timepoints coverage; capacities/ego filter; association/selection dependency closure; actual prior dependencies; indirect-only refs subtract direct refs. Preserve full field identities, allow overlapping causal roles, do not fabricate counterfactual usefulness.
- [x] Include actual tensor and prior changes versus preceding plan and actual output change/validity when available, without running a model. First-stage changes are undefined, not a comparison with fabricated zeros. Same-evidence repeat/refinement and reference-only receipts produce correct differences.
- [x] Cover remote-only/ambiguous, F multimodal sources/contexts, capacity and ego filtering, empty/masked inputs, full-P/local-derived equivalent F, prior-only dependencies, corrupt tensor locations/receipts rejected and old GoT dispatch unchanged. Reuse upstream validation rather than second receipt registry.
- [x] Run new torch-free tests plus structured input/method evaluation regression; commit `Audit direct and indirect structured evidence use`, report exact API and review.

## Task 3: Versioned stage objective, periodic validation and selection

**Files:** `src/planning/train_structured_driver.py`, `tests/test_structured_training.py`; a focused `src/planning/structured_validation.py` is allowed if needed to isolate shared evaluation/metric aggregation. No changes to network, provider or old `train_driver.py`.

**Interfaces:** Preserve all old `toolv2x_structured_training_v1` calls, loss behavior and resume. Add full `toolv2x_structured_training_v2` config accepted by `_config` and `fit`, with explicit objective, row weighting, periodic validation and checkpoint selection. `training_loss` gets explicit optional objective config and returns measurements used by validation; old invocation remains identical. Report exact normalized keys/default-free JSON for Task4.

The new objective supervises the row's selected stage plus its final configured same-evidence refinements, averaging those losses. Prior prefixes are actual current-model detached forwards but not additional supervised losses. No target-as-prior, unchanged exact-repeat behavior. Old prefix supervision remains old config only.

New row weighting is `frame_condition_equal`: assign equal total weight to each unique frame, then its represented Ego/P/F/PF evidence conditions, then duplicate stage rows within that frame-condition. Derive conditions from authentic primitive receipts, including bundles, not filenames or labels. Normalize weights once over eligible labeled training rows to mean one; apply fixed weights before minibatch reduction, never renormalize per minibatch. Explain usual minibatch/last-batch optimization effects and report actual group counts/weights. Frames/conditions absent due to failed collection remain explicit coverage gaps, not fabricated rows. Validation uses the same declared frame-condition weighting and separate train/validation recording roles.

```python
# With no refinements, each stage row gets only its own loss:
# [L0, L1, L2], not [L0, (L0+L1)/2, (L0+L1+L2)/3].
# Repeated initial-stage rows for the same frame/Ego condition do not multiply
# that frame-condition's total frozen dataset weight.
assert sum(weights_for_one_frame_condition) == expected_group_weight
```

Validation intervals are explicit processed-batch counts. V2 validates initial, periodic and final checkpoints with no optimizer updates and restored training RNG/mode. Preserve history and best selection across exact resume. Report stage/condition loss, full label/valid-output denominators, invalid outputs, ADE/FDE and existing got_prefix metrics; first-stage and final/refinement identities clear. Strict invalid plans follow ExecutionSpec; no silently dropping failures in selection.

Checkpoint selection is lexicographic: lowest weighted invalid-plan rate among rows with valid labels, then lowest weighted valid-plan prefix-L2 average, then earliest processed batch. No eligible finite valid metric => checkpoint not eligible; no fabricated best path. Compare a common frozen validation set, one selected driver for all arms. Document that missing label horizons cannot enter the complete-prefix metric, though remaining labels can contribute other diagnostics. Preserve all interval checkpoints; selected checkpoint is a reference/record, not deletion or moving weights.

- [x] RED: hand-computed stage loss and gradient prove prior-prefix losses excluded; legacy11:5:2 unchanged; frame-condition duplicate balancing; invalid labels generate no updates; actual detached current-model prior used.
- [x] RED: periodic validation occurs before training completion, groups match literal trajectories, invalid plans keep denominator, selection ties choose earliest, all-invalid cannot invent an eligible checkpoint; validation preserves optimizer/RNG; interrupted+resumed equals continuous training weights/progress/validation history/selection with relocated paths.
- [x] Implement minimal shared objective/evaluation code, validate complete versioned config, bind exact resume including validation/weights. Add accurate help/examples; no actual resource training.
- [x] Run actual tiny CPU network/optimizer tests plus old training/episode tests. Commit `Add stage-weighted training and periodic validation`, report/review.

## Task 4: Frozen preparation package and bootstrap contract

**Files:** `configs/structured_driver_readiness_v1/` JSON config/manifest files, `docs/structured_driver_readiness_2026_09_15.md`; one focused metadata-only preparation/validation helper and tests if executable validation is required. Consume exact APIs from tasks1–3. No generic experiment manager. Verified preparation gap: extend the existing diagnostic selector and CLI whitelist with F-current-only and fixed P-current then F-current IDs, preserving the old IDs; use one shared diagnostic-ID constant and focused contract tests. This adds `src/planning/method_episode.py`, `src/planning/run_framework.py` and the focused diagnostic tests to Task4 ownership. It only makes all four promised collection conditions expressible, not a learned policy or new mechanism.

**Preparation source:** Read the existing `outputs/framework_baseline_resume_v1` causal selected/index metadata and existing split manifest on main. Freeze all currently accepted physical-train frames and their existing train/validation recording identities, retain known existing discontinuity exclusions; never choose by labels or quality. Keep only causal identity/motion/path audit fields in frame manifest. Paths aid audit only. Do not read point clouds, images, labels or checkpoints to generate the preparation package. Controller supplies verified concrete source file paths and metadata counts before this task.

**Run settings:** Common full default StructuredDriverSpec observations_only (256hidden,4heads,2layers,64entities,4forecastsets), complete ExecutionSpec from prior reviewed two-frame spec, seed7, AdamW lr1e-4/decay.01, batch_size2, initial adaptation20epochs, refinement_depth1, checkpoint every25processedbatches, validation every100processedbatches plus initial/final, devicecuda as future execution config only. New v2 objective/weighting/selection from Task3. Freeze exact values as experiment choices, not protocol constants. Actual performance adjustments need a new saved config; no runtime auto-tuning now.

Collection/acceptance selection uses two deterministic evenly spread eligible frames per physical recording for coverage acceptance, plus the full accepted metadata population for subsequent adaptation. Include all failures in denominator; these are integration checks, not two-frame effect validation. All four Ego/P-state/F/PF conditions requested by fixed diagnostic controls, shared driver/inputs; no query-policy training. Explicit feedback control for method runs; one-shot comparison config uses episode_aggregate, same total bidirectional ExecutionSpec and actual wrapper/candidate costs, candidate_sources initial/slower/constant_motion, max_candidates3, slowdown.5, wrapper reserve2048, summary maxfields8/maxbytes1536, extra_generation self_refinement with3driver quota. Verify configs fit actual request budget construction before publishing as valid.

**Bootstrap decision:** Do not assume random output will buy P/F. Phase B first runs declared numeric integration set and audits failures; if initial validity prevents required evidence coverage, first train only from retained initial Ego prepared inputs until a predeclared acceptance gate, then freeze and recollect all four conditions. This uses existing genuine numeric archives/exporter and no forged plans or second collection schema. Declare this is a conditional bootstrap substage, never completed now or sufficient by itself. Acceptance gate: all declared acceptance frames produce finite ExecutionSpec-valid initial outputs; no performance-based frame exclusion, all four intended condition tasks reach their requested acquisition boundary or an explicit budget-empty result. History/receipt/direct-use audits must validate, no hidden remote access; zero returned fields stays a reported valid empty service result, not evidence consumption success. Unexpected failures block long training and trigger bounded correction; no automatic repeated training or silent schedule extension.

Freeze bootstrap budget as a maximum of3epochs of v2 Ego rows, sameoptimizer/seed/caps, with acceptance after each epoch via explicit later commands; if gate remains unmet, stop real phase and diagnose. This is a contingency specification only, no training authorization this batch. Shortening/lengthening requires a new configuration record.

- [x] Validate literal files against StructuredDriverSpec, ExecutionSpec, Task3 _config and frozen metadata rules; invalid role, duplicate IDs, future/label fields, mismatched p_processing, invalid budgets or altered frame selection fail. CPU contract only.
- [x] Save exact source metadata counts/selection recipe, complete configs and copied causal identity manifests. Mark lifecycle `prepared_not_executed`, no fabricated runtime load identity/checkpoint, no fake ready-to-run real method spec before actual loading. Preflight chooses exact output paths later and refuses overwrite.
- [x] Write diagnostic/selection metric definitions and B/C/D gates; controls share weights and evidence rules, prior ablations distinguish direct versus historical total effects. Ordinary training/probe metrics are not paper superiority or closed-loop safety.
- [x] Commit `Freeze shared driver preparation and acceptance settings`, report/review. No real resource execution.

## Final delivery

- [x] Independent whole-branch review; one consolidated fix wave with scoped re-review if needed. No open important findings silently waived.
- [x] Required complete lightweight and model-environment regressions on final source, external paths explicit; publish first failures and resolution separately. No real model generation/training.
- [x] Update README/STATUS/review entry and detailed report with actual APIs/files/configs/tests/review history/limitations. Archive decision ledger before private-workspace cleanup.
- [x] Locally integrate to main preserving original dirty documents byte-for-byte, no push. Report Stage A complete and real stages B–D not executed.
