# Stage A independent whole-branch final review

## Verdicts

- **Spec compliance: PASS for the implemented Stage A scope.** No unresolved Critical or Important discrepancy was found against `docs/review_9_15_1_response.md`, the approved implementation plan, and repository review rules.
- **Code quality: PASS.** No change is requested from this review. The changes reuse the common executor, validated receipt ledger, structured collator/network, existing metric functions and diagnostic runtime instead of introducing another execution or provenance system.
- **Ready for final verification: YES.** Local integration remains conditional on the controller completing the required full lightweight and model-environment regressions on final source and updating the delivery records. This review does not claim those suites have already passed.

## Review scope and strengths

I reviewed the packaged whole-branch net diff, the current touched source/test/config/document files, the approved spec/plan, the decision ledger and prior findings/fix reports. I applied the SDD final-review contract, requesting-code-review and Ponytail. No subagents, git mutations, implementation fixes, real training, real inference, provider collection, arrays or model resources were used. This report is the only intentional file written by the reviewer.

1. **Decision accounting and policy-visible prefixes.** `src/planning/method_episode.py:364` creates a pending cost record for every actual decision/preflight interval regardless of explicit control, including forced STOP. `:370` supplies the policy only the preceding cost ledger, preserving the archive reconstruction boundary; `:391-412` excludes the persistence callback while retaining known elapsed time if interruption occurs. Prefix reuse requires the explicit accounting version and retains previous measured costs. Exact-repeat stages bypass decision work until their actual final decision. `src/evaluation/framework.py:177-195` detects missing decision costs, legacy preflight/policy failures without a decision event, duplicate stages, unsupported versions and incomplete measurements; recognized old explicit timing remains usable. Known partial subtotals survive without inventing a complete total. The v1 evaluation functions and wire format are unchanged. Non-string version aggregation retains an invalid row instead of crashing the run summary.

2. **Evidence audit semantics and integration.** `src/evaluation/structured.py:75` validates the numeric episode before deriving its per-plan output. Provider-acquired and receiver-derived known/new/previous sets are distinct at `:87-105`; the neutral union is not mislabeled as purchase count. Direct primary refs follow existing tensor groups, and coverage uses the validated tensor locations/masks. Source, context, mode/time coverage, entity/forecast capacity, ego filtering, association dependencies and prior parents remain explicit. Direct/indirect roles may overlap as specified; indirect-only subtracts direct primary refs. Prior-only and repeat/refinement cases reuse existing dependency identities rather than creating receipts. The ordinary evaluator attaches this audit only to the numeric path and retains old GoT dispatch. First-stage changes remain undefined rather than being compared with fabricated zeros.

3. **Versioned objective and fixed dataset weighting.** `src/planning/train_structured_driver.py:167-235` keeps legacy prefix supervision for v1 and limits explicit v2 losses to the selected stage and configured refinements. Earlier current-model outputs are detached and archive numeric priors are cleared before every forward; labels enter only the masked loss. `src/planning/structured_validation.py:13-56` derives Ego/P/F/PF from authentic primitive receipts, including bundles, then freezes equal frame/represented-condition/duplicate-row weights over eligible labeled rows. `train_structured_driver.py:494-566` excludes zero-label optimization rows and applies the fixed weight before minibatch reduction. This is the disclosed weighted minibatch objective, not a claim that AdamW or a smaller final batch reproduces a full-batch optimization step.

4. **Validation, selection and checkpoint continuation.** `src/planning/structured_validation.py:59-120` retains labeled invalid-plan denominators, metric-specific eligible weights and incomplete-label limits, and separates selected-stage from refinement diagnostics. Selection requires finite invalid-rate and complete-prefix metrics, breaks ties by earliest processed batch, and cannot invent a best checkpoint for all-invalid/no-eligible-metric data. `train_structured_driver.py:509-580` validates initial, scheduled processed batches and normal training end; an interruption does not insert an extra validation point. It restores model mode and Python/NumPy/Torch/CUDA RNG and performs no optimizer update. Full config, row semantics, frozen weights, source/feature snapshots, optimizer/RNG, progress, validation history and selection bind resume. Earlier checkpoint copies exclude future batches, retained history references are checked, and `fit` returns the final checkpoint while `selection.json` names the chosen one. V1 retains its original config and final-validation path.

5. **Preparation and real interfaces.** `scripts/prepare_driver_readiness.py` validates literal complete configs using `StructuredDriverSpec`, `ExecutionSpec`, Task 3 `_config`, and the actual bundle constructor. Stable sample, local-frame and scene/global-frame identities are separately unique. The public validate-only CLI has no dependency on private preparation files. The shared `DIAGNOSTIC_POLICY_IDS` is consumed by the selector and both ordinary/bundle-only runtime whitelist checks; F-current-only and fixed P-current/F-current are expressible without repurposing the old change-dependent policy. The registered one-shot continuation, aggregate response mode, finite candidate set and full bidirectional budget are concrete. The package and reports accurately leave runtime identities, labels, coverage, CUDA performance, actual collection/training and method effect unverified. Conditional bootstrap uses genuine retained Ego stage-0 archives, has the declared three-epoch cap, and discloses both actual label-count-dependent cadence and fresh full-adaptation initialization rather than promising a nonexistent warm-start API.

## Independent metadata evidence

I read only the disclosed existing JSON metadata, not the paths to arrays, labels or weights contained in it. Programmatic checks established:

- Every one of the **3,095** packaged rows is exactly the approved nine-field projection of the disclosed original `outputs/framework_baseline_resume_v1/episodes/selected_index.jsonl`; its disclosed byte count also matches.
- Role counts are **2,987 train / 108 validation**; sample IDs, scene/global-frame IDs and scene/local-frame IDs are individually unique.
- The package recording-role sets equal the original split manifest after the repository's `recording(scene)` normalization: **8 train / 2 validation recordings**. Original exclusion config counts are **3,027 to 2,987 (40 excluded)** for train and **108 to 108 (0 excluded)** for validation.
- All **20** acceptance identities equal independently computed per-recording source-order endpoints. This agrees with the existing two-per-recording selection interface and does not use labels or model outcomes.

The first role-set assertion in this independent probe incorrectly compared normalized recording names with raw scene names and failed. Applying `recording` to the split-manifest scene names resolved that reviewer-script comparison error; no package or implementation change was needed. The check output was compact and the full metadata was not printed.

## Findings and prior-review triage

### Critical

None.

### Important

None.

### Minor

None requiring a change. The task ledger contains no unresolved deferred finding to waive. The previous legacy timing, invalid version aggregation, acquired/derived mixing, duplicate identity and private CLI dependency findings are addressed in the final source. No consolidated fix wave is needed.

## Verification and conclusion boundary

I read the targeted tests and existing logs instead of rerunning passed suites. In particular, the recorded 38-test training/episode run, two final resume checks and eight final preparation checks have terminal OK and exit 0 evidence. Earlier failures and the lost-output qualification remain in the task reports; they are not reclassified as successful full runs. The independent metadata probes above are additional read-only source comparisons, not training or model tests.

The required final complete lightweight/model-environment regressions, final delivery documentation and local integration are still controller work. Once those pass, this review presents no code/spec blocker to local integration. Contract correctness, preparation completeness and synthetic CPU behavior do not establish learned driver capability, fair measured quality/cost superiority, feedback benefit or closed-loop safety; real stages B-D remain unexecuted.
