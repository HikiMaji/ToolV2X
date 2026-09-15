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

- [ ] RED: hand-computed stage loss and gradient prove prior-prefix losses excluded; legacy11:5:2 unchanged; frame-condition duplicate balancing; invalid labels generate no updates; actual detached current-model prior used.
- [ ] RED: periodic validation occurs before training completion, groups match literal trajectories, invalid plans keep denominator, selection ties choose earliest, all-invalid cannot invent an eligible checkpoint; validation preserves optimizer/RNG; interrupted+resumed equals continuous training weights/progress/validation history/selection with relocated paths.
- [ ] Implement minimal shared objective/evaluation code, validate complete versioned config, bind exact resume including validation/weights. Add accurate help/examples; no actual resource training.
- [ ] Run actual tiny CPU network/optimizer tests plus old training/episode tests. Commit `Add stage-weighted training and periodic validation`, report/review.
