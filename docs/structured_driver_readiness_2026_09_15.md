# Shared structured-driver preparation contract (2026-09-15)

This package freezes Stage A metadata and experimental choices. Its lifecycle is `prepared_not_executed`: no real collection, training, inference, provider call, checkpoint/model load or runtime output-path allocation occurred while producing it. It is a preparation record, not evidence that the shared driver is ready, useful, safe or superior.

## Frozen files and source boundary

The package is under `configs/structured_driver_readiness_v1/`:

- `frames.jsonl` is the only packaged copy of the 3,095 accepted causal frame rows: 2,987 train and 108 validation. Each row contains only sample/scene/frame identity, the existing research role and physical split, time-t ego motion, and audit-only feature/motion read paths.
- `acceptance_frames.json` contains 20 identities: the first and last eligible source-order frame from each physical recording, yielding 16 train and 4 validation identities. This is deterministic coverage selection. It never reads labels, quality, gains or model outputs.
- `source_metadata.json` records the controller-supplied source paths, recording roles, counts and the existing 40/0 train/validation exclusion counts. Paths support a local audit only; they are not semantic identity.
- `training_v2.json` is the complete Task3 v2 full-adaptation config: the default observations-only 256/4-head/2-layer/64-entity/4-forecast-set driver, seed 7, AdamW 1e-4/0.01, batch size 2, 20 epochs, refinement depth 1, checkpoint interval 25, validation interval 100 plus initial/final, CUDA as a future execution choice, frame-condition weighting and selected-stage/refinement objective.
- `bootstrap_training_expected_v2.json` freezes the conditional maximum-three-epoch recipe with the expected 16 eligible train rows and interval 8. Metadata cannot certify label eligibility. Before any real bootstrap, calculate the actual eligible train-row count and save a new complete concrete config with `validation.interval_batches = ceil(eligible_train_rows / 2)`. Never rewrite a running config.
- `readiness.json` binds the common execution/driver specs, fixed collection recipes, controls, metrics, stage gates and all unexecuted flags. `runtime_load_identity` is null. A runnable `method_run_spec` is intentionally absent until the actual driver and predictor loads provide their required identities.

The local validator can compare all 3,095 rows directly with the controller-whitelisted JSON source. Away from that workstation source, it still performs strict structural, count, role and deterministic-selection validation, but cannot independently prove the package came from the external 4.6 MB selected index. The original index, split manifest and exclusion config paths remain disclosed for that local audit.

## Collection and budget contract

Ego/P-state/F/PF use one shared observations-only driver and the fixed diagnostic policies `stop`, `p_current`, `f_current` and `p_current_f_current`. The old `p_current_f_change` remains a distinct actual-revision diagnostic. Fixed policies use actual receipt count, select only currently legal actions and never inspect forecast previews. They express collection conditions only; they are not a learned query policy.

Each episode allows at most two remote calls and three actual driver attempts. All 20 × 4 acceptance tasks stay in the denominator, including invalid initial plans, provider/service failures, STOP and explicit budget-empty outcomes. Zero returned fields are a valid empty service result, but do not count as successful evidence consumption.

The explicit feedback control uses the existing `feedback` control. The one-shot comparison uses the existing registered `diagnostic_conditional_v1` continuation, `episode_aggregate`, initial/slower/constant-motion candidates, at most three candidates, slowdown 0.5, 2,048 B wrapper reserve, at most eight summary fields/1,536 B, and self-refinement within the same three-driver-attempt quota. Both use the same complete `real_smoke_v1` bidirectional `ExecutionSpec`.

The helper calls the actual `normalize_control` and `episode_bundle` constructors and encodes the complete public request as UTF-8. On its public synthetic state the request is 1,813 B, the permitted aggregate outer response is 133,120 B, and their sum fits the 196,608 B episode limit; the request also fits the 16,384 B request limit. This is a constructor/budget contract probe without a provider or model invocation. Runtime bytes and cost have not been measured.

## Conditional bootstrap and later gates

Phase B first attempts all declared numeric acceptance tasks and audits every failure. Bootstrap is permitted only if invalid initial output prevents the required acquisition coverage. It exports one genuine retained Ego stage-0 prepared row per selected frame from the initial attempt, including a failed numeric task when its prepared input and independent label are valid. It does not duplicate stage-0 rows from the four conditions or create a new collection schema. A missing label, invalid prepared input or feature failure remains in the coverage report and is diagnosed before training; it is never silently omitted.

The bootstrap gate is checked after each of at most three epochs: every acceptance frame must have a finite `ExecutionSpec`-valid initial output; every intended condition must reach its fixed acquisition boundary or an explicit budget-empty outcome; and history, receipt and direct-use audits must pass with no hidden remote access. Failure after three epochs stops the real phase for diagnosis. There is no automatic repeat, epoch extension or frame filtering.

A bootstrap checkpoint is only for recollecting valid four-condition episodes. Full adaptation starts from fresh seed-7 initialization on recollected data. The current fit API has no warm-start contract, and changing rows or config is not exact resume. Shared-driver capability is evaluated only after actual adaptation and recollection.

Stage C recollects all four conditions with the one frozen adapted driver, reports initial STOP, first-return STOP, two-call and failure rates, and retains direct/derived/admitted evidence accounting. Stage D compares feedback, frozen feedback, fixed-evidence refinements and conditional one-shot under the same driver weights, inputs, evidence rules and total bidirectional budget.

## Metrics and interpretation

Integration diagnostics are finite and execution-valid initial-output rate, requested acquisition-boundary rate, explicit budget-empty rate, and history/receipt/direct-use audit outcomes. Validation selects one common driver lexicographically by weighted invalid-plan rate, complete `got_prefix_L2_avg`, then earliest processed batch. Every condition uses the same weights and failure rules.

Removing direct P/F while retaining a remote-derived prior measures only the marginal direct-input effect. Measuring historical total remote effect requires regenerating a legal no-remote prefix before comparison. Mismatched evidence is a labeled diagnostic input and never a forged provider receipt. These integration and offline-imitation metrics do not establish paper-level superiority or closed-loop safety.

Before any future real command, verify actual eligible labels, runtime driver/predictor load identities, target-device capacity and fresh non-existing output paths. Preserve every interval checkpoint; no storage/runtime cost or retention policy was measured in Stage A.

Resource-free validation:

```bash
PYTHONPATH=src python scripts/prepare_driver_readiness.py \
  configs/structured_driver_readiness_v1 \
  --source-frames .superpowers/sdd/2026-09-15-driver-readiness/source_causal_frames.jsonl \
  --acceptance-preview .superpowers/sdd/2026-09-15-driver-readiness/acceptance-identity-preview.json \
  --source-metadata .superpowers/sdd/2026-09-15-driver-readiness/task-4-source-metadata.json \
  --validate-only
```
