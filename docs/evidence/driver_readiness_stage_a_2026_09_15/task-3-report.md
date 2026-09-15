# Task3 incremental implementation report

Status: implementation committed locally; targeted checks pass; independent review pending. Scope limited to `src/planning/train_structured_driver.py`, `src/planning/structured_validation.py`, `tests/test_structured_training.py`. Ponytail + TDD applied; no provider/network architecture change, no real dataset training, no real provider inference.

## RED evidence

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/autodl-tmp/conda-envs/llava/bin/python -m unittest` with the four `StructuredTrainingTests.test_v2_*` tests initially added. `task-3-red.log` / `task-3-red.exit`: exit 1, four expected missing API/config errors. Existing legacy literal 11:5:2 gradient portion passed before the missing v2 keyword failed. No output values were forged into archived driver records.

Initial GREEN command `... -m unittest test_structured_training` running under exec session 64102; terminal completion must be polled exactly, not rerun because of transport uncertainty. Logs: `task-3-green-1.log`, exit file `task-3-green-1.exit`.

## Exact config surface for Task4

All v1 keys retain their old required values/behavior. Full v2 config keys (no defaults):

```json
{
  "version": "toolv2x_structured_training_v2",
  "driver_spec": {
    "version": "toolv2x_structured_driver_v1",
    "hidden_dim": 256,
    "attention_heads": 4,
    "interaction_layers": 2,
    "max_entities": 64,
    "max_forecast_sets_per_entity": 4,
    "p_processing": "observations_only",
    "position_scale_m": 50.0,
    "size_scale_m": 10.0,
    "time_scale_s": 3.0,
    "association_distance_m": 2.0,
    "association_history_distance_m": 2.0,
    "association_size_ratio_min": 0.5,
    "association_size_ratio_max": 2.0,
    "association_heading_rad": 1.0471975511965976,
    "association_ambiguity_margin_m": 0.25,
    "ego_position_threshold_m": 1.5,
    "ego_history_distance_m": 0.5,
    "ego_heading_rad": 0.5235987755982988,
    "ego_min_common_history": 2,
    "ego_length_min_m": 3.0,
    "ego_length_max_m": 6.0,
    "ego_width_min_m": 1.4,
    "ego_width_max_m": 3.0,
    "ego_height_min_m": 1.0,
    "ego_height_max_m": 3.0
  },
  "seed": 7,
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.01
  },
  "batch_size": 2,
  "epochs": 20,
  "refinement_depth": 1,
  "save_interval": 25,
  "device": "cuda",
  "objective": {
    "name": "selected_stage_and_refinement",
    "reduction": "mean"
  },
  "row_weighting": "frame_condition_equal",
  "validation": {
    "interval_batches": 100,
    "initial": true,
    "final": true
  },
  "selection": {
    "name": "invalid_rate_prefix_l2_earliest",
    "output": "selected_stage"
  }
}
```

Driver spec must be full explicit existing observations_only 64/4 spec; no capacity change. Bootstrap changes only epochs=3 and validation.interval_batches to prepared eligible training batch count (ceil eligible rows / 2); no actual bootstrap execution here.

New APIs:
- `training_loss(model, features, row, *, refinement_depth=0, task=None, objective=None, evaluation=False)`: unchanged v1 when objective absent; v2 report adds measurements with output_kind selected_stage/refinement, refinement number, actual detached predicted waypoints and finite loss. Earlier prefix forwards are detached actual current-model inputs, not supervised.
- `structured_validation.evidence_condition(row, task=None)` canonical set of ledger primitive receipt request.tool, including decoded bundle receipts. Caller first validates genuine source episode.
- `frame_condition_weights(rows, role)` returns list aligned with all supplied rows and coverage dict; zero weight for other roles/no labels. Mean-one normalization is fixed across eligible role rows, never repeated per minibatch.
- `summarize_measurements(records, execution_spec)` strict ExecutionSpec validation plus existing trajectory_metrics; selected_stage/refinements and stage/condition/generation groups have explicit rows/weights/label/output/metric denominators.
- `select_checkpoint(best, summary, batches, checkpoint)` compares invalid rate, complete-prefix metric, earliest processed batch. No finite eligible metric returns existing best/None.

V2 returns final saved checkpoint from fit (not automatically selected checkpoint). `report.json` and checkpoint report hold validation_history and selection; `selection.json` is the selected reference or null. Checkpoint references are filenames plus processed batch identity. Resume copies retained checkpoint files into new out, never deletes prior weights.

## Important boundaries

Validation selection uses selected-stage output per row; same-evidence refinements are separately named diagnostics. Prefix plus refinement offline forwards can exceed online three-attempt count and are not episode results. Partial labels can support ADE and eligible short prefixes but cannot enter complete-prefix average. Invalid plans remain in labeled denominator even when their distance metrics are unavailable. Missing conditions for represented frames are explicit; entirely absent frames require collection-manifest coverage, not fabrication.

Fixed dataset weights do not make minibatch AdamW identical to a single full-batch weighted update. Shuffle, parameter changes and smaller last batches still matter. Eligible zero-label rows are excluded from v2 optimizer batches. Recording roles remain isolated and bound across resume.

## Iteration 1 completion

Session 64102 completed with exit 1: 19 tests, 18 passing; one expected resume-history mismatch exposed validation added at a nonperiodic resumed batch. Fixed initial validation to run only for fresh runs. `task-3-green-1.exit` records 1. Added no-optimizer/RNG cadence comparison, v2 no-label skip, partial-horizon denominators, and actual detached current-model priors. Session 48894 runs training + episode tests with `task-3-green-2.log`/`.exit`.

## Final verification and exact process completion

All commands run in `/root/autodl-tmp/ToolV2X/.worktrees/driver-readiness`. Environment prefix for model tests:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests /root/autodl-tmp/conda-envs/llava/bin/python -m unittest
```

Append these exact test arguments for each recorded command:

| Run | Arguments | Process completion | Logs / exit files |
|---|---|---|---|
| Initial RED | `test_structured_training.StructuredTrainingTests.test_v2_excludes_prefix_gradients_and_preserves_legacy_ratio test_structured_training.StructuredTrainingTests.test_v2_fixed_frame_condition_weights_include_bundle_and_gaps test_structured_training.StructuredTrainingTests.test_v2_periodic_validation_exact_resume_and_relocation test_structured_training.StructuredTrainingTests.test_v2_metrics_invalid_denominators_and_selection` | exit 1, 4 missing API/config errors | `task-3-red.log`, `task-3-red.exit` |
| Initial implementation | `test_structured_training` | session 64102 completed, exit 1; 19 tests / 1 failure, 24.534 s | `task-3-green-1.log`, `task-3-green-1.exit` |
| Training + episode regression | `test_structured_training test_structured_episode` | session 48894 completed, exit 0; 38/38, 62.213 s | `task-3-green-2.log`, `task-3-green-2.exit` |
| Earlier-checkpoint RED | `test_structured_training.StructuredTrainingTests.test_v2_resume_earlier_checkpoint_does_not_import_future_checkpoints` | session 14576 completed, exit 1, 2.308 s; future checkpoint copy caused expected FileExistsError | `task-3-red-resume.log`, `task-3-red-resume.exit` |
| Resume fix GREEN | `test_structured_training.StructuredTrainingTests.test_v2_resume_earlier_checkpoint_does_not_import_future_checkpoints test_structured_training.StructuredTrainingTests.test_v2_periodic_validation_exact_resume_and_relocation` | session 44925 completed, exit 0; 2/2, 8.773 s | `task-3-green-resume.log`, `task-3-green-resume.exit` |

Paths above are relative to this report directory. All sessions reached confirmed terminal completion; none were rerun because transport lost output. The last targeted run exercises the final resume code. Full repository suites are left to root after review as instructed. `git diff --check` also passed with no output.

Additional configuration-only check: `PYTHONPATH=src /root/miniconda3/bin/python` constructed the full explicit JSON above and asserted `_config(config) == config`, exit 0, without importing/creating a model. This report's full driver_spec is actual `StructuredDriverSpec().to_dict()` output, not a placeholder.

## Verified behavior

- Actual StructuredPlannerNetwork forward outputs at zero residual prove loss=1.5 with the literal target [2,2], v1 cumulative prefix gradient weights 11:5:2 and v2 selected-stage weights 1:1:1. Additional same-evidence refinement averages only selected+refinement losses.
- Actual current-model prefix outputs are supplied detached as the next prior. Earlier v2 prefix forwards do not build gradient graphs; old exact-repeat slot semantics and old v1 tests remain passing.
- Genuine feedback and one-shot bundle archives yield canonical Ego/P/PF groups. In the test's first frame with 6 stage rows, represented groups of 2/1/3 rows each sum to frozen total weight 1.5; 9 eligible rows over 2 frames sum to mean-one total weight 9. Absent F is recorded.
- Actual tiny CPU AdamW training is unchanged by additional validation cadence: identical weights, optimizer states and Python/NumPy/Torch RNG. Zero-label v2 rows produce no optimizer update and no invented selection.
- Literal distance diagnostics verify invalid-plan labeled denominator retention, ADE/FDE/prefix values, incomplete horizon exclusion, no-label exclusion, lexicographic earliest tie and all-invalid ineligibility.
- Interrupted vs continuous training preserves exact weights, progress, optimizer, RNG, history and selection. Source data and checkpoint directory are both relocated in the test. The final version uses an instrumented constructor that scales the actual network's final layer down, ensuring valid outputs and a non-null selected checkpoint; no synthetic numeric output replaces a network forward. Selected checkpoint and every historical validation checkpoint exist in the resumed output.
- A resumed nonperiodic stop checkpoint triggers no extra validation; final validation occurs only at the scheduled epoch endpoint. Resume from an earlier checkpoint of a finished run copies only preceding saved checkpoints and avoids importing future checkpoints.
- Resume remains exact continuation only; initialization-only bootstrap files cannot satisfy fit's data/config/optimizer binding. No warm-start was added.

## Remaining review concerns / declared limits

- The selected-stage metric is deliberately used for one common driver selection; extra offline refinement metrics are separate, and their additional forwards are not billed online episode attempts. Main trajectory metrics use the common frozen validation role rows.
- No eligible finite complete-prefix distance means `selection.json` is null and no best path is invented, including no validation rows or all-invalid outputs. A v2 corpus with no eligible train labels yields only its initial checkpoint and zero processed batches/optimizer steps.
- Resume portability assumes the retained run directory, including `inputs` and earlier checkpoints, is moved together. A lone checkpoint without its bound input snapshots/history is not a complete exact-resume bundle. Files are copied, never moved/deleted; old archives are untouched.
- Weights are normalized separately within each role's eligible labeled rows. Partial labels count as eligible for training and diagnostics; their missing horizons never enter complete-prefix selection. Per-minibatch AdamW and smaller last batches retain their normal optimization effects.
- Completely absent frames cannot be inferred from training rows. Their coverage must be supplied by collection preparation; the code reports observed-frame missing conditions without fabricating data. Receipt-derived condition is distinct from direct/admitted coverage (Task2 audit).
- The metric helpers accept literal trajectory fixtures for contract checks; these are not archived provider replies or model-generated planning results. No actual dataset training, provider inference, external-model generation or performance claim occurred.
- Independent Task3 review is pending root dispatch. No outstanding failing test is known. No architecture/provider/legacy train_driver.py change, new dependency, full suite run or remote push was performed.

## Commit

Subject: `Add stage-weighted training and periodic validation`.
Own files only: `src/planning/train_structured_driver.py`, `src/planning/structured_validation.py`, `tests/test_structured_training.py`. Root-owned plan/docs and this ignored orchestration report are not staged.

Commit command completed with exit 0; output was suppressed. Post-commit status contains only root-owned plan/docs changes. No commit identifier was printed.
