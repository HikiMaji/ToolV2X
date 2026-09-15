## Task 2 report: complete

### Scope and boundary

Implemented a derived audit over existing validated numeric episode archives. The production helper performs no model/tokenizer/label/filesystem access and creates no receipt or evidence registry. It first calls the existing `validate_numeric_episode`, which reconstructs structured inputs from the actual ledger snapshots and paid response wires. All field identities in the output are the original full reference dictionaries.

No real dataset, training, model generation, provider inference, cache rebuild, or method-effect experiment ran. The test-only numeric driver creates schema-valid deterministic plans so the actual P/F service, response codec, receiver, ledger, structured preparation, and numeric episode validator can be exercised without torch. Its outputs are contract evidence only.

### Public signatures and integration

- `evaluation.structured.audit_structured_episode(episode) -> list[dict]`
- `evaluation.framework.evaluate_method_task(...)` publishes the per-plan list as `structured_audit` for `toolv2x_interaction_v2`; old GoT rows publish `structured_audit=None` and retain their original evaluation dispatch.
- `scripts/check_review.py` registers `test_structured_admission` in the torch-free review suite.

Each returned plan dictionary has these exact top-level keys:

`plan_id`, `stage`, `ledger_snapshot`, `receipt_ids`, `new_receipt_ids`, `previous_receipt_ids`, `known_remote_refs`, `known_local_refs`, `new_remote_refs`, `previously_known_remote_refs`, `known_acquired_remote_refs`, `new_acquired_remote_refs`, `previously_acquired_remote_refs`, `known_receiver_derived_remote_refs`, `new_receiver_derived_remote_refs`, `previously_receiver_derived_remote_refs`, `admitted_refs`, `dropped_refs`, `direct_primary_refs`, `direct_remote_primary_refs`, `direct_local_primary_refs`, `direct_dependency_closure_refs`, `association_selection_dependency_refs`, `prior_dependency_refs`, `prior_dependency_closure_refs`, `indirect_only_refs`, `indirect_only_remote_refs`, `indirect_only_local_refs`, `prior_only_refs`, `counts`, `capacities`, `association_status_counts`, `observation_coverage`, `forecast_coverage`, `tensor_changed_from_previous`, `prior_changed_from_previous`, `output_available`, `output_valid`, `output_changed_from_previous`.

`counts` repeats the cardinality for every published reference collection: `known_remote_refs`, `known_local_refs`, `new_remote_refs`, `previously_known_remote_refs`, `known_acquired_remote_refs`, `new_acquired_remote_refs`, `previously_acquired_remote_refs`, `known_receiver_derived_remote_refs`, `new_receiver_derived_remote_refs`, `previously_receiver_derived_remote_refs`, `admitted_refs`, `dropped_refs`, `direct_primary_refs`, `direct_remote_primary_refs`, `direct_local_primary_refs`, `direct_dependency_closure_refs`, `association_selection_dependency_refs`, `prior_dependency_refs`, `prior_dependency_closure_refs`, `indirect_only_refs`, `indirect_only_remote_refs`, `indirect_only_local_refs`, `prior_only_refs`.

`capacities` publishes `max_entities`, `max_forecast_sets_per_entity`, `tensor_entities`, `ego_filtered_entities`, `entity_capacity_filtered_entities`, and `structured_capacity_filtered_forecast_sets`.

Each `observation_coverage` item publishes `entity_id`, `source`, `source_role`, `primary_refs`, `anchor_ref`, `tensor_location`, `valid_steps`, and `timepoints_seconds`. Each `forecast_coverage` item publishes `entity_id`, `source`, `source_role`, `context_scope`, `model_used`, `primary_refs`, `anchor_ref`, `tensor_location`, `valid_modes`, `valid_timepoints_per_mode`, `valid_mode_timepoints`, and `timepoints_seconds`.

First-plan tensor/prior/output changes are `None`; no zero baseline is fabricated. Later values compare the archived tensors, prior plan plus parent refs, and waypoints against the immediately preceding plan. Missing outputs have `output_available=False`, `output_valid=None`; available outputs are checked with the existing numeric output validator.

### Coverage

- Local x=10 and peer x=30 with `max_entities=1`: remote anchor/history remain known and dependency-closed, while `direct_remote_primary_refs=[]` and no remote observation coverage is claimed.
- The same archive with `max_entities=2`: remote history is direct at the actual `observations/entity=1/source_slot=1` location with 11 valid causal timepoints.
- Remote-only F retains one provider-full forecast set with six modes and six valid timepoints per mode.
- Ambiguous local/remote association, empty scenes, masked observation histories, entity capacity, forecast capacity, and ego-filter accounting are read from validated structured metadata and masks.
- Full P local-derived forecast and information-equivalent direct F preserve two full forecast identities in one actual tensor set and retain the derived parent closure.
- Exact-repeat and self-refinement distinguish unchanged tensors from actual prior/output changes.
- Two-stage F covers a forecast admitted only through actual prior dependencies after capacity reorder.
- A paid reference-only second P receipt adds a receipt but no new remote reference or tensor change.
- Corrupt tensor locations and receipt wire text are rejected by upstream validation before audit output.
- Old GoT evaluation remains on its original branch with `structured_audit=None`.

### Commands and results

1. Baseline before implementation:

   `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python -m unittest -v test_structured_inputs test_method_evaluation`

   Result: 35 tests, PASS, exit 0.

2. Initial RED:

   `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python -m unittest -v test_structured_admission`

   Result: expected module assertion failure because `src/evaluation/structured.py` was absent; 1 failure, exit 1.

3. Behavior RED after adding the API shell:

   Same command as above.

   Result: expected empty-audit and missing-validation failures; 3 failures and 5 errors, exit 1. Two subsequent fixture-only corrections used a recognized recording name and allowed both ambiguous remote targets through the real execution capacity.

4. Core GREEN:

   Same command as above.

   Result: 6 tests, PASS, exit 0.

5. Final targeted torch-free regression:

   `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python - <<'PY' ... loadTestsFromNames(['test_structured_admission', 'test_structured_inputs', 'test_method_evaluation']) ... assert torch/transformers absent ... PY`

   Result: 41 tests, PASS, exit 0; neither torch nor transformers was imported.

6. Static and registration checks:

   `PYTHONPATH=src:tests python -m py_compile src/evaluation/structured.py src/evaluation/framework.py tests/test_structured_admission.py scripts/check_review.py`

   `PYTHONPATH=src:tests python - <<'PY' ... assert 'test_structured_admission' in check_review.names ... assert torch/transformers absent ... PY`

   `git diff --check`

   Result: all PASS, exit 0. The saved Task 2 report is present. Only the four Task 2 code/test/review-registration files are included in the Task 2 commit; the pre-existing root plan edit and driver-readiness draft remain unstaged.

Per Task 2 instructions, no full 416-test model-environment suite was run in this task. The complete `check_review` suite was registered but not rerun; final verification was limited to the specified changed-area tests.

### Review concerns

- `structured_audit` is descriptive provenance and input-use evidence. It does not establish counterfactual usefulness, task quality, training success, CUDA parity, or closed-loop benefit.
- A field may appear in multiple causal-role collections by design. Only `indirect_only_*` subtracts direct primary refs; `prior_only_refs` additionally subtracts association/selection dependencies.
- Neutral `known_remote_refs`/`new_remote_refs`/`previously_known_remote_refs` include both provider-acquired and receiver-derived full identities. The six provenance-specific collections keep provider receipts separate from receiver derivations, and `previously_acquired_remote_refs` is acquired-only. Repeated plans and reference-only receipts correctly report no new field in either provenance class.
- Task 1 compute accounting code and semantics were not changed.

Commit subject: `Audit direct and indirect structured evidence use`.

### Fix round 1: complete

Independent review found one Important issue: the neutral remote union was also used for `previously_acquired_remote_refs`, which conflated provider-receipt fields with receiver-derived forecasts. The bounded fix reuses the existing `ledger['acquired_fields']` / `ledger['derived_fields']` split and adds explicit known/new/previous collections and counts for each provenance class. No ledger or protocol change is involved.

RED command:

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python -m unittest -v test_structured_admission.StructuredAdmissionContractTests.test_full_p_and_equivalent_f_keep_multirole_dependency_closure test_structured_admission.StructuredAdmissionContractTests.test_repeat_refinement_prior_only_and_reference_only_differences`

Result: 2 expected errors from missing `known_acquired_remote_refs`, exit 1. The fixtures exercised actual P-local followed by F, and actual P-local followed by a paid reference-only P receipt.

GREEN implementation:

- Neutral union: `known_remote_refs`, `new_remote_refs`, `previously_known_remote_refs`.
- Provider receipt fields: `known_acquired_remote_refs`, `new_acquired_remote_refs`, `previously_acquired_remote_refs`.
- Receiver derivations: `known_receiver_derived_remote_refs`, `new_receiver_derived_remote_refs`, `previously_receiver_derived_remote_refs`.
- `counts` contains matching entries for all nine collections. All collections reuse the original ledger lists and retain full field-reference dictionaries.

Focused GREEN command:

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python -m unittest -v test_structured_admission.StructuredAdmissionContractTests.test_full_p_and_equivalent_f_keep_multirole_dependency_closure test_structured_admission.StructuredAdmissionContractTests.test_repeat_refinement_prior_only_and_reference_only_differences`

Result: 2 tests, PASS, exit 0.

Final fix-round regression:

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests python - <<'PY' ... loadTestsFromNames(['test_structured_admission', 'test_method_evaluation']) ... assert torch/transformers absent ... PY`

Result: 23 tests, PASS, exit 0; neither torch nor transformers was imported. No complete review or model-environment suite was run, per fix-round scope.

Fix commit subject: `Separate purchased and receiver-derived audit fields`.
