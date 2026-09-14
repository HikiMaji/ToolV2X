# Structured driver implementation and review records

Batch: 2026-09-14 to 2026-09-15. All task and final fix reviews are complete. Final validation: 322 lightweight tests and 416 model-environment tests passed; reviewed code integrated locally to main, no push or real dataset training/generation. Chronological reports preserve initial failures and later resolutions. Paths/line numbers refer to the reviewed isolated checkout at each recorded stage; source is now available under the main checkout. Final command, scope and raw logs are in structured_driver_implementation_2026_09_15.md and structured_driver_validation_2026_09_15/.

---

## Record: progress.md

# SDD ledger — plan: docs/superpowers/plans/2026-09-14-structured-driver.md

## Authorization and limits
User: downloads below 100 MB may be made directly, then continue the proposed complete structured-driver implementation. No real training or new method experiments. Preserve dirty main docs. Scope follows reviewed design plus 9-14-1 semantic response.

## Source preparation
UniV2X main archive 1598011 bytes, unpacked 4070704 bytes; VAD main archive 542442 bytes, unpacked 1602485 bytes. CRC validation PASS. Official codeload, no weights. Local source paths /root/autodl-tmp/UniV2X and /root/autodl-tmp/VAD. Sources have Apache 2.0 license. Repo APIs/HEAD did not provide size; enforced streaming limit.

## Preflight interface scan
| Tasks | Shared surface | Result |
|---|---|---|
| 1 / 2 | StructuredDriverSpec, prepared arrays | Exact tensor names/shapes fixed; task 1 report supplies complete spec defaults |
| 1 / 3 | opt-in local-history ledger and receiver config | Old defaults unchanged; numeric interaction v2 activates both |
| 2 / 3 | prepare_input, plan_prepared, numeric output | No fake Q9; output validation and costing dispatch explicitly |
| 2 / 4 | same collator/network, checkpoint loader | Loader added by task 4, runtime execution disabled until complete |
| 3 / 4 | numeric task archive and runtime load | Exporter consumes actual archived numeric prepared input; real task collection deferred |
| 1 | inputs/spec/association tests | Explicit geometric fixtures and provenance checks, no driver or labels |
| 2 | gradient/mask vs architecture | Synthetic real network only; no GT-assisted prior |
| 3 | common episode vs controls | Keep one loop, numeric exact-repeat freezes input, same-evidence only updates prior |
| 4 | fit/export vs no training | Tiny synthetic optimizer/resume tests allowed; no real dataset training |

## Execution
Task 1: implementation and verification complete; 13 new + 75 legacy targeted tests passed; 302 lightweight tests passed; committed as Add causal structured evidence and entity receiver; independent review found 2 Important fixes, round 1 in progress
Task 2: pending
Task 3: pending
Task 4: pending

Baseline: 302 resource-independent tests passed in isolated workspace before implementation.

Task 4 preflight Ruling: Add an explicit seeded untrained checkpoint initialization command, tested only in temporary synthetic fixtures. Runtime collection otherwise requires a checkpoint while checkpoint training requires collected numeric tasks. This resolves the bootstrap dependency without new method logic or real execution; if wrong, the initialization interface would need replacement.

Task 1: minor (deferred): 841-line input module combines loader/association/tensor projection/validation; final review should triage a focused split.

Task 1: fix round 1/5 (2 original findings addressed, 1 new open: zero-local/empty-scene validation regression; commit subject Tighten structured evidence validation). Fix round 2 assigned.

Task 1: fix round 2/5 (1 addressed, 0 open; Support empty structured evidence scenes; 41 tests passed).
Task 1: complete (subjects Add causal structured evidence and entity receiver through Support empty structured evidence scenes; review clean; one module-size Minor deferred).
Task 2: in progress; base ref sdd/structured-network-base.

Full-suite resource preparation: copied three unchanged ignored JSON fixtures for train_g0719 from main outputs into worktree (ego_input, F_input, sample; 90010 bytes total); direct byte comparison passed. No cache/model generation, no new experiment.

Task 2: implementation committed as Add shared numeric cooperative planner; 6 core synthetic CPU tests plus compile/import/diff checks passed; independent review in progress. No real-data inference/training.

Task 2 review: 2 Important fixes assigned (JSON-native output, canonical-input permutation coverage), 1 narrow Minor included (empty training status). Red chronology exists in implementer report/status messages; final diff cannot independently establish execution order, and no code defect was identified on that point. Final tests will be freshly verified.

Task 2: fix round 1 awaiting scoped review; JSON-native output, canonical authentic-response permutation coverage and nonempty training status implemented; 7 targeted tests passed; subject Fix structured output validation and permutation coverage.

Task 2: complete (fix round 1/5: 3 addressed, 0 open; independent review clean; 7 synthetic CPU tests).
Task 3: in progress; base sdd/structured-integration-base.

Task 3 first integration cases green per implementer: actual tiny numeric network, real P/F fixture service, 3driver/2query/prior/exact-refine/cost/prefix metrics. Concrete shared-interface fix authorized: readonly NumPy views passed by episode into torch.as_tensor warn and share unsafe storage; copy only nonwriteable arrays in structured_driver collator, with covering test. Archive/control/runtime still in progress.

Task 3 progress: 11 numeric integration cases passed per implementer (actual P/F/prior, fixed and prefix controls, 1RPC/2primitive/3driver bundle, archives/state_features, corruption, failures, readonly buffers, metric formulas). Runtime lazy loader signature load_structured_planner(checkpoint,device='cuda') in planning.structured_driver. Saves three ego pose arrays plus task.inputs.ego_history_read_paths. Remaining: targeted legacy suite, bundle archive/old checkpoint coverage, summary fields, report/review.

Task 3: committed as Integrate numeric driver with tool episodes and controls; final 43 covering checks passed (16 integration, 7 network, 20 legacy); final nested-cost 19 and bundle 8 passed. Initial 160 run had one tokenizer path configuration error, corrected check passed. Independent task review pending; T4 lazy loader remains explicit dependency.

Task 3 cross-task evidence: checked Task1 full-P local / equivalent-F fixture plus tensor encoding fields; dedicated read-only direct comparison confirmed ALL tensor_inputs, ego_motion and previous_plan identical after equivalent F addition. Initial standalone script used wrong test class name (ImportError before execution); corrected StructuredReceiverTests run passed. Fake predictor contract evidence only. Loader dependency remains assigned T4; final complete suites deferred until integration.

Task 3: complete (Integrate numeric driver with tool episodes and controls; independent spec and quality review clean; 0 open findings). Cross-task provider/time-t contracts use prior Task1 tests and unchanged provider code; direct all-tensor equivalence PASS. Task4 loader and final full regression checks remain tracked dependencies, not completed claims.
Task 4: in progress; base sdd/structured-training-base.

Task 4 design/RED update: literal source_task path plus ordered prefix indices; validate archives on demand; shared feature files snapshot once for direct resume comparison. Avoid repeating full padded prepared/scene JSON in every stage. Failed prepared inputs retained. No real resources executed.

Task 4 initial 6 tests GREEN per implementer (5.43s). Additional behavioral cases in progress: real distinct recording validation, all-invalid labels zero updates, invalid model prior masked, checkpoint/progress/optimizer mutation and stable per-file task/feature snapshots. No actual training resources.

Task 4 Ruling: Add an explicit random training_run_id in model_version.training for initialization/new independent training, preserved on checkpoint copy/load/resume, alongside direct parameter/state comparisons. Name/seed/steps can coincide for different actual runs; the ID distinguishes their policy binding without path identity or content digests. It is audit identity, not a learned feature or content/authenticity proof. Cost if wrong: identical weights from independent runs are conservatively incompatible for policy reuse and may require explicit rebinding; causal P/F/model inputs do not change.

Task 4 latest covering suite: 9 tests passed per implementer (11.418s). Added exact-repeat regression exposed incorrect latest-output prior usage; fixed to preserve its existing prior slot, with separate refinement using detached current-model output. Added run-ID copy/reload/resume stability and independent-run separation. Final invalid-prior reason reporting/CLI/report pending.

Task 4 pre-submit coverage per implementer: 34 passed (11 training, 7 network, 16 numeric episode; 43.876s) plus 18 structured input tests with torch/transformers absent. Self-review found real bundle requests lack top-level tool; coverage must use authentic primitive receipts. One bounded regression/fix in progress, then task report/review; no scope expansion.

Task 4: committed as Add structured driver supervision and resumable training; final 19 tests passed (12 training+7 network) after real bundle coverage fix; prior 34 integrated passed. Torch-free input 18 passed. Report/CLI present, independent task review pending. No real data training/generation.

Task 4 review in progress: concrete exporter physical_split guard gap identified. Controller verified existing run_framework.select_rows:114 requires physical_split=train for both allowed research roles. Awaiting reviewer full findings before one scoped original-worker fix round; no data changes/rebuild/training. Fix base sdd/structured-training-fix-base.

Task 4 review: spec needs fixes, one Important only. Focused synthetic probe confirmed physical_split=test role=train yielded 3 exported train rows. Fix round 1/5 assigned to original worker: shared explicit physical train guard, actual fixture field, test/missing split for train and validation and fit bypass. No new data or experiment.

Task 4: fix round 1/5 (1 addressed, 0 open; Require physical train provenance for structured supervision; 14 training tests passed; scoped review clean).
Task 4: complete (Add structured driver supervision and resumable training through Require physical train provenance for structured supervision; independent spec and quality review clean). All task gates complete; final whole-branch review and full integrated validation remain.

Final whole-branch review in progress: real runtime/exporter history audit-path mismatch identified. Runtime preserves nonempty task.inputs.ego_history_read_paths; wrapper intentionally prepared.ego_history_used.read_paths=[] from numeric arrays; Task4 wrongly requires equality. Concrete cross-task integration gap, pending full findings and one consolidated final fix wave. Base sdd/structured-final-fix-base.

Final whole-branch review: 2 Important confirmed (runtime history audit/export mismatch; default-v1 list track_ids regression). One consolidated fresh-worker fix wave assigned; no new mechanism/data/model work. Carried structured_inputs module-size Minor explicitly triaged nonblocking by final reviewer: bounded shared receiver contract, no concrete duplicate/correctness defect; no split requested, extraction only when an actual additional caller/lifecycle requires it. Both initialization/run-identity Rulings accepted.

Final consolidated fix wave committed as Fix numeric history export and legacy window compatibility. Both original findings reproduced RED then corrected; 58 covering tests passed (19.299s), compile/diff checks passed. Runtime history audit preserved in snapshots; v1 and opt-in full list windows match array contracts. Scoped final re-review pending; no second broad review or full suite yet.

Final scoped re-review: both original Important findings ADDRESSED, no new blockers. Final full validation started via run-final-validation.sh (exec session 70867), ordinary review Python then model environment. Code remains unchanged while tests run; root documentation archival may continue.

Final resource-independent full suite: 322 passed in 199.796s (OK). Model-environment full discovery now running under same frozen code. Light raw log persisted to docs/structured_driver_validation_2026_09_15/resource_independent.log.

Full-model log observations while running: upstream torch Transformer nested-tensor-disabled UserWarning, PEFT missing-config/vocabulary-assumption UserWarning, and old tokenizer overlength notices (2456/2048 and 2087/2048). No failure yet. Focused source checks confirm original tokenizer context/fit_evidence checks and method_controls ControlModelTests use tokenizer only or replace _generate; numeric path has no tokenizer/PEFT. These log notices will be retained, not suppressed or presented as real GoT generation.

Required full model validation completed: 416 tests / 1 failure in 398.695s; only child-process clean-import/help test lacked source resolution when parent PYTHONPATH absent. Other 415 passed, production code unchanged. Task4 worker reproduced under env -u PYTHONPATH, fixed child cwd from test file path, added transformers absence assertion; target test and 15-test training module passed (17.326s). Subject Make structured training subprocess checks self-contained. This is a new required-validation portability repair, not reopening closed final-review findings. Scoped test-only review pending; plan final model recheck on integrated main.

Test portability scoped review APPROVED: child source cwd resolves independently of parent PYTHONPATH and all import/help assertions retained. Ready for local main integration, then final full model recheck on main; no production change since final code review.

Local main integration: fast-forwarded reviewed feature branch; all five original main documents match their pre-integration bytes. Three already identical design documents are now tracked; user_requirements.md and 9-14-1.md remain untouched and uncommitted. No push. Complete model recheck started from main with PYTHONPATH unset (exec session 73513), independent log final-model-main.log.

Final main model recheck: 416/416 passed in 390.541s, exit 0, PYTHONPATH explicitly unset. Full light 322/322 already passed; no production edits since reviewed code. Public logs preserve the first model-suite test-environment failure separately from final success. All required plan checks complete.

Final delivery: reviewed implementation locally integrated to main, original five document bytes preserved, only user_requirements.md and 9-14-1.md retain their original uncommitted status. No real dataset training, new real model generation, method experiment or GitHub push. Reports/ledger/rulings and final validation are archived publicly before removing only this plan private workspace and its owned temporary worktree.


---

## Record: task-1-report.md

# Task 1 report: causal structured evidence and entity receiver

Status: DONE_WITH_CONCERNS. The implementation and synthetic contract checks are complete. No training, real model generation, dataset expansion, or method-quality experiment was run.

## Files owned

- `src/planning/structured_inputs.py` (new)
- `src/planning/evidence.py` (only the opt-in local-history extension)
- `tests/test_structured_inputs.py` (new)

The report itself is stored at the requested SDD path and is ignored by the repository. Concurrent changes in `AGENTS.md` and `docs/` were not edited or staged by Task 1.

## Public API and defaults

`StructuredDriverSpec` is a frozen dataclass. `to_dict()` emits every field and `from_dict()` accepts exactly that full field set. It rejects unknown/missing keys, unsupported versions or P modes, nonpositive capacities/scales/gates, invalid ratio/dimension intervals, and hidden-dimension/head mismatch.

```text
version = toolv2x_structured_driver_v1
hidden_dim = 256
attention_heads = 4
interaction_layers = 2
max_entities = 64
max_forecast_sets_per_entity = 4
p_processing = observations_only
position_scale_m = 50.0
size_scale_m = 10.0
time_scale_s = 3.0
association_distance_m = 2.0
association_history_distance_m = 2.0
association_size_ratio_min = 0.5
association_size_ratio_max = 2.0
association_heading_rad = 1.0471975511965976
association_ambiguity_margin_m = 0.25
ego_position_threshold_m = 1.5
ego_history_distance_m = 0.5
ego_heading_rad = 0.5235987755982988
ego_min_common_history = 2
ego_length_min_m = 3.0
ego_length_max_m = 6.0
ego_width_min_m = 1.4
ego_width_max_m = 3.0
ego_height_min_m = 1.0
ego_height_max_m = 3.0
```

These are explicit engineering defaults for later controlled evaluation, not fitted statistics or calibrated association/ego-identification claims.

The callable interfaces are:

```python
load_ego_history(root, split, g)
build_structured_plan_input(motion, ledger, spec, *, ego_history=None,
                            previous_plan=None, previous_parent_refs=())
validate_structured_prepared(prepared)
```

`load_ego_history` reads only `g-10..g` within the physical recording boundary. It returns `states[11,3]`, `valid[11]`, `times[-1..0]`, and attempted `read_paths`. Valid poses are converted once into ego(t); missing, malformed, non-rigid, or unusable poses remain zero and masked. It never reads future poses.

`build_structured_plan_input` has no filesystem, predictor, tokenizer, torch, label, or driver access. It calls `planning.evidence.remote_units` before projection, so paid receipts, remote values, derived parent chains, active contexts, and P-local/F equivalence are validated by the existing receiver. It requires `ledger.p_processing == spec.p_processing`.

The prepared record has exactly:

```text
input_layout, decoding, driver_spec, scene, g, ego_motion, entities,
tensor_inputs, admission_report, ego_history_used, previous_plan,
previous_parent_refs
```

`ego_history_used` is the complete validated history dictionary or `None`; `read_paths` remain audit metadata and never enter numeric tensors or semantic identity. `previous_plan` is `None` or finite `6x2`; parent refs are forbidden without an actual previous plan and must resolve within the cumulative ledger. Derived parent closure is propagated independently of direct tensor admission.

`tensor_inputs` is JSON-native and padded according to the spec:

```text
observations        [E,2,11,10]
observation_mask    [E,2,11]
observation_sources [E,2]       (local=0, remote=1)
forecasts           [E,C,6,6,5]
forecast_mask       [E,C,6,6]
forecast_context    [E,C,2]
entity_mask         [E]
```

Only position, size and time use the explicit spec scales. Track handles, receipt strings, request labels, provider/receiver execution origin, and mode-index identity do not enter learned tensors. Source role and actual full/subset context do. Missing/padded values are zero under false masks.

Association uses source-local aliases, deterministic local-first canonical aliases, current distance, common valid-history RMS, size ratios, and heading gates, followed by SciPy one-to-one Hungarian assignment. Any row/column near tie within the configured ambiguity margin stays separate. Remote-only entities and same-number handles from different sources are retained. A representative anchor is an original measurement chosen by local source, then score, then canonical alias; coordinates are never variance-averaged. The source boxes are already in ego(t), so no peer transform is applied.

Near-origin alone is `unresolved`. Ego exclusion additionally requires supplied causal ego history, enough common valid history, configured size bounds, history residual, and heading agreement. Confirmed ego metadata stays in `entities` and admission groups while its obstacle tensor slot remains absent.

`validate_structured_prepared` is the direct downstream validation helper. It strictly checks the top-level and tensor schemas, JSON-native finite values, masks and zero padding, fixed source slots, entity capacity/order, nested source-local aliases and refs, forecast/history values, exact tensor reproduction from entity metadata, admission partition, previous/derived parent closure, and structured tensor locations.

`planning.evidence.new_ledger` now has:

```python
new_ledger(..., p_processing='local_mtr', include_local_history=False)
```

The default remains `toolv2x_evidence_ledger_v1` with the existing anchor/forecast fields and ordering. `include_local_history=True` selects `toolv2x_evidence_ledger_v2` and adds each real local `window.states/valid/scores/time` history with the tracking producer version and `_validate_task_value`-compatible proxy status. The existing default `p_processing='local_mtr'` is unchanged; the new structured route explicitly constructs it with the spec default `observations_only` unless P-local is selected.

## TDD and validation evidence

RED was observed before production creation:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs.StructuredModuleContractTests.test_module_and_opt_in_history_api_exist
Result: expected failure, structured numeric input module is missing (1 test, 1 failure).
```

Additional red/green checks were observed for previous-plan parent refs without a plan, local-history tracking provenance, and invalid singular ego poses. Each failed for the intended absent behavior before its minimal production change and passed afterward.

Final targeted and requested legacy command:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs test_evidence_ledger test_task_spec test_vehicle_tools
Result: 88 tests passed, 0 failures/errors.
```

The 13 new tests cover strict/default config, exact default/v2 ledgers, causal pose loading and invalid masks, source-local identity, gated match and ambiguity, row permutation invariance, P-state without predictor calls, absent-value masks, P-local/F equal forecast encoding, paid-receipt rejection, parent closure, ego corroboration/exclusion, and standalone corruption rejection. The requested legacy subset contributes 75 tests.

Repository lightweight review command:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_review.py
Result: 302 tests passed, 0 failures/errors. The guard also confirmed that torch and transformers were not imported.
```

Other checks:

```text
python -m py_compile src/planning/structured_inputs.py src/planning/evidence.py tests/test_structured_inputs.py
Result: passed.

git diff --check
Result: passed.
```

## Concerns and boundaries

- Association and ego-role behavior is verified with hand-derived synthetic fixtures, not measured association accuracy on a labelled corpus. All thresholds are versioned so later experiments can state the exact rule.
- Capacity is explicit and deterministic, but `max_entities=64` and `max_forecast_sets_per_entity=4` have not been audited over the full dataset in this task. Dropped forecasts remain in entity/group audit metadata and `dropped` refs.
- This task establishes causal input/provenance and numeric-shape contracts only. It does not establish trained-driver quality, P/F utility, closed-loop safety, or a method gain.

## Fix round 1: strict semantic validation and directed ego heading

The standalone validator now reconstructs every source-local track from the entity observations and reruns the configured global association. It requires the recomputed alias grouping, status, metrics, representative original anchor, and ego/obstacle/unresolved role to equal the prepared metadata. Matched status requires exactly the local/remote alias pair; single-source ambiguous and unmatched entities stay separate. It also regenerates `field_groups` and `local_field_groups` from the entities and spec and requires exact equality, which binds each location to the correct tensor, entity, local/remote source slot, forecast-set index, capacity, and empty-location case.

Association retains axis-equivalent box yaw distance because an unoriented 3D box can be geometrically equivalent after a 180-degree rotation. Ego corroboration now uses directed angle distance wrapped over 2 pi, so a near-origin track facing opposite the causal ego history remains unresolved and stays in the obstacle tensor.

RED was observed before these fixes:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs.StructuredReceiverTests.test_opposite_ego_heading_remains_unresolved_and_in_tensor test_structured_inputs.StructuredReceiverTests.test_validator_recomputes_association_representative_and_role test_structured_inputs.StructuredReceiverTests.test_validator_requires_exact_bounded_tensor_locations
Result: 3 tests run; 7 expected assertion failures. Opposite heading was labelled ego, and validator accepted all mutated status, metrics, representative box, role, source_slot=99 and forecast_set=99 records.
```

Exact covering command after the fixes:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs test_evidence_ledger
Result:
.......................................
----------------------------------------------------------------------
Ran 39 tests in 2.208s

OK
```

No full-suite rerun, model execution, training, or data experiment was performed in this fix round.

## Fix round 2: empty local detection sets

The semantic validator now accepts zero local observations while still rejecting multiple distinct local sources. When the local set is empty, association reconstruction uses `None` only as an internal canonical-order sentinel; each observation retains its existing `source_role`, so paid peer objects remain remote. A remote-only scene therefore retains its entity in the remote observation slot, and a fully empty legal window produces empty entities with all tensor masks false. No prepared schema field or mechanism was added.

RED was observed before the fix:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs.StructuredReceiverTests.test_empty_local_detections_retain_paid_remote_entity test_structured_inputs.StructuredReceiverTests.test_fully_empty_scene_builds_all_masked_structured_input
Result: 2 tests run; both errored at `structured entities require one local source`.
```

Exact covering command after the fix:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_structured_inputs test_evidence_ledger
Result:
.........................................
----------------------------------------------------------------------
Ran 41 tests in 2.156s

OK
```

No broad-suite rerun, model execution, training, or data experiment was performed in this fix round.


---

## Record: task-1-review.md

### Spec Compliance

- ❌ Issues found: `validate_structured_prepared` does not strictly validate association/provenance metadata or tensor-location consistency (`src/planning/structured_inputs.py:695`, `src/planning/structured_inputs.py:807`), and the ego heading gate treats opposite headings as agreement (`src/planning/structured_inputs.py:236`, `src/planning/structured_inputs.py:358`). These violate the strict-schema requirement and the requirement to exclude only a sufficiently corroborated ego.

### Strengths

- `load_ego_history` restricts reads to `t-10..t`, respects the recording boundary, masks missing/invalid poses, and transforms each valid pose once into ego(t) (`src/planning/structured_inputs.py:98`).
- The builder calls the existing `remote_units` receiver before projection, retains source-local aliases and remote-only/ambiguous entities, prioritizes full-context forecasts before capacity cropping, and keeps identity strings out of numeric feature values (`src/planning/structured_inputs.py:153`, `src/planning/structured_inputs.py:443`, `src/planning/structured_inputs.py:466`, `src/planning/structured_inputs.py:539`).
- The opt-in history path preserves the v1 default output and records the exact causal tracking window only in ledger v2 (`src/planning/evidence.py:34`, `src/planning/evidence.py:45`, `src/planning/evidence.py:57`). Tests cover the key hand-derived association cases, paid-receipt rejection, P-state predictor isolation, P-local/F forecast equality, causal history loading, masks, and the clarified dictionary-or-None `ego_history_used` contract (`tests/test_structured_inputs.py:39`, `tests/test_structured_inputs.py:185`, `tests/test_structured_inputs.py:212`, `tests/test_structured_inputs.py:272`).

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `src/planning/structured_inputs.py:695`: the standalone validator checks only allowed association labels and the representative anchor's basic shape; it never recomputes or cross-checks the association status/metrics, representative anchor, or role against the entity observations. It also accepts arbitrary integer `source_slot`/`forecast_set` values in admission locations at `src/planning/structured_inputs.py:807`. A focused synthetic check changed a matched entity to `unmatched`, changed its representative box, and changed a valid observation location to `source_slot=99`; all three corrupted records were accepted. This makes the promised strict provenance/shape/capacity validation unreliable for downstream consumers and audits. Recompute these fields from entity observations/spec where possible, require status/alias cardinality and representative-source equality, validate exact tensor locations and bounds, and add corruption tests alongside `tests/test_structured_inputs.py:283`.
- `src/planning/structured_inputs.py:236`: `_angle_distance` reduces differences modulo pi, so headings separated by 180 degrees have distance zero. `_role` uses that result to confirm and remove ego at `src/planning/structured_inputs.py:358`; a near-origin track moving/facing opposite to the causal ego history can therefore be labeled `ego` instead of retained as `unresolved`. Use directed wrapped angular distance for ego corroboration (and document separately if box association intentionally uses axis-equivalent orientation), then add an opposite-heading ego-filter test near `tests/test_structured_inputs.py:272`.

#### Minor (Nice to Have)

- `src/planning/structured_inputs.py:1`: the new 841-line module combines filesystem history loading, association, entity construction, tensor projection, admission reporting, and a large standalone validator. The functions are individually clear, but this is already costly to review and maintain. Move the association/entity helpers into the brief's optional `entities.py`, or isolate validation in a focused module without changing the public API.

### Checks

- Focused unchanged-interface check: `src/planning/evidence.py:268` validates the cumulative ledger through `known_field_manifest`, exact acquired parents, active context, and equivalent full-context forecasts before returning `remote_units`; the builder's call at `src/planning/structured_inputs.py:539` therefore uses the required existing receiver verification path.
- Focused executable check: one synthetic matched record was mutated independently in association status, representative geometry, and admission source slot; `validate_structured_prepared` accepted every mutation. The reported suites were not rerun.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The causal construction path and test coverage are strong, but the public validator accepts materially false audit metadata and the ego heading rule can delete an insufficiently corroborated object. Those two defects block approval of Task 1's contract.


---

## Record: task-1-fix-review.md

### Original Important Findings

- **ADDRESSED — strict standalone validation:** the fix reconstructs source-local tracks, recomputes global association/role/representative metadata, compares those semantics exactly, and regenerates admission groups so tensor locations must match their entity, source slot, forecast column, and capacity state (`src/planning/structured_inputs.py:728`, `src/planning/structured_inputs.py:802`, `src/planning/structured_inputs.py:839`). The added corruption tests cover status, metrics, representative geometry, role, `source_slot=99`, and `forecast_set=99` (`tests/test_structured_inputs.py:298`, `tests/test_structured_inputs.py:319`).
- **ADDRESSED — directed ego heading:** association keeps its documented axis-equivalent box-yaw comparison, while ego corroboration now uses a directed angle wrapped over 2 pi (`src/planning/structured_inputs.py:236`, `src/planning/structured_inputs.py:241`, `src/planning/structured_inputs.py:375`). The new opposite-heading test verifies the object remains unresolved and present in the tensor (`tests/test_structured_inputs.py:283`).

### New Critical/Important Breakage

#### Critical

- None.

#### Important

- `src/planning/structured_inputs.py:802`: semantic validation now requires exactly one source represented by a local observation. A valid scene with zero local detections has no local fields, which is permitted by the unchanged window/prediction contracts (`src/tools/task_spec.py:247`, `src/tools/task_spec.py:297`) and can still contain purchased peer objects. The builder creates those remote-only entities, then calls this validator and fails at `len(local_sources) != 1`. It likewise rejects a valid completely empty scene. This regresses Task 1's requirement to retain remote-only objects and support masked absent values. Recompute with an explicit prepared local-source field, or make `_entities` validation support an empty local-track set using a sentinel/order rule that cannot classify a remote source as local; add zero-local/remote-present and fully-empty builder tests.

### Fix-Round Verdict

**Spec compliance:** ❌ Needs fixes — both reviewed findings are addressed, but the fix introduces a remote-only/empty-scene rejection.

**Task quality:** Needs fixes

**Check:** Read the unchanged zero-length window and aligned prediction contracts to resolve the named regression risk. The reported 39 tests were not rerun. The prior Minor file-size concern is deferred to final triage as instructed.


---

## Record: task-1-fix2-review.md

### Regression Verdict

- **ADDRESSED — zero-local remote-only and empty scenes:** validation now rejects only multiple represented local sources and uses `None` solely as the internal canonical-order sentinel when no local observation exists (`src/planning/structured_inputs.py:802`). Actual aliases remain validated source strings, and `_entities` already handles empty association sides, so a paid peer-only entity remains remote while a fully empty scene produces no entities and all-false masks (`src/planning/structured_inputs.py:268`, `src/planning/structured_inputs.py:457`). The two new builder-level tests cover both cases, including remote-slot placement and masked local/empty tensors (`tests/test_structured_inputs.py:178`, `tests/test_structured_inputs.py:190`).

### New Critical/Important Breakage

#### Critical

- None.

#### Important

- None.

### Fix-Round Verdict

**Spec compliance:** ✅ Spec compliant for the scoped round-two regression.

**Task quality:** Approved

**Reasoning:** The seven-line production change directly removes the invalid empty-local assumption while preserving the round-one semantic reconstruction for scenes that contain local observations. The supplied 41-test run covers the targeted regression; it was not rerun.


---

## Record: task-2-report.md

# Task 2 report: shared numeric network and authentic output

Status: COMPLETE. This task implemented and exercised an untrained numeric network on synthetic CPU inputs only. It did not run real-data inference, dataset training, a frozen MTR forward, or any method-effect experiment.

## Owned files

- `src/planning/structured_driver.py`
- `src/planning/driver_contract.py`
- `tests/test_structured_driver.py`

Task 1 files were not modified. Existing dirty `AGENTS.md` and untracked `docs/` files were not edited or staged.

## Public interfaces

```python
StructuredPlannerNetwork(spec)
StructuredPlannerNetwork.forward(self, batch)

collate_structured_inputs(features_list, prepared_list, device='cpu')

StructuredPlanner(spec, *, model=None, device='cpu', model_version=None)
StructuredPlanner.prepare_input(
    self, features, motion, ledger, receiver_spec,
    previous_plan=None, previous_parent_refs=())
StructuredPlanner.plan_prepared(self, features, prepared)

validate_numeric_output(output, prepared, execution_spec)
```

`validate_numeric_output` is defined once in `planning.driver_contract` and re-exported by `planning.structured_driver`. Importing the contract module leaves `torch` unloaded, so later offline evaluation can validate numeric records without importing the network runtime.

The planner provenance is exactly:

```text
driver_kind: structured
driver_spec: full strict StructuredDriverSpec dictionary
decoding: numeric
model_version: stable name/revision plus actual training status and optimizer step metadata
```

The default identity says `initialized_untrained` with zero optimizer steps. Version name/revision values use semantic identifiers rather than filesystem paths. A future loader may keep checkpoint paths as separate audit metadata; they are not part of the default semantic model identity.

## Implemented representation and network

The collator calls Task 1's standalone prepared-input validator and requires identical complete driver specifications across a batch. It accepts the original single-ego feature shapes:

```text
regression_map      [1,2,1,14,50,88]
classification_map  [1,2,1,2,50,88]
active_agent_mask   [1,2,1]
```

It concatenates regression then classification channels, flattens frame/source exactly as the original shallow implementation, and applies non-overlapping `unfold(kernel_size=(5,4), stride=(5,4))`. The result is `[B,2,220,320]`. Inactive previous-frame patches are zero and masked. Extra feature audit fields are ignored rather than embedded.

Optional ego history is supplied as the three separate causal arrays `ego_pose_history [11,3]`, `ego_pose_history_valid [11]`, and `ego_pose_history_times [11]`. `prepare_input` converts them to Task 1's full `ego_history_used` dictionary with `read_paths=[]`; it never reads the filesystem. The collator checks those arrays against the prepared record. No track handle, receipt string, tool kind, request kind, mode index, or provider/receiver execution identity enters a tensor.

The actual shared PyTorch network contains:

- a 320-to-hidden shallow-scene projection with learned two-frame and 10-by-22 grid codes;
- a shared LayerNorm MLP, masked max, post-pool MLP and shared multi-head attention for the two local/remote observation-source histories per entity;
- a shared LayerNorm point encoder for every forecast mode independently, with numeric mode score/model-used values and allowed source/context fields, but no mode-index embedding or coordinate averaging;
- causal ego-history and ego-motion tokens;
- six learned waypoint queries and ordinary zero-dropout `TransformerDecoderLayer` interactions over state, forecast-mode and scene tokens;
- an actual six-point previous-plan condition and mask, with the output expressed as a learned residual in meters when the prior is present;
- learned null tokens so an empty entity/forecast/scene batch remains finite in training mode.

All point encoders use LayerNorm and contain no BatchNorm, so a single valid point is legal. The model produces one `[B,6,2]` trajectory and does not contain a traffic predictor.

## Output and cost contract

`plan_prepared` runs a real forward pass before creating an output with exactly these fields:

```text
output_version=toolv2x_numeric_plan_v1
driver_kind=structured
status=valid
waypoints
prepared_input
driver_cost
parent_refs
```

`prepared_input` is the complete exact prepared record. `parent_refs` equals its admitted evidence closure, which already includes the previous-plan closure. The cost contains only `seconds`, `numeric_token_count`, `output_points`, and `model_executed`. `model_executed` becomes true only after the forward returns.

`numeric_token_count` is defined as the number of unmasked tokens passed to the final waypoint cross-attention, plus its six waypoint queries: active scene patches, admitted entity-state tokens, valid forecast-mode tokens, ego-history condition, ego-motion condition, the safe null-memory token, and six queries. It is a numeric sequence count and is not a tokenizer or generated-language count.

The torch-free validator requires exact prepared-input and parent bindings, a successful real-model cost, six finite points, and the existing ExecutionSpec speed/acceleration bounds. No Q9 text, generated-language token count, tokenizer field, or language execution flag is created.

## TDD and validation evidence

Before implementation, the targeted API test failed with four explicit failures for the missing `StructuredPlannerNetwork`, `collate_structured_inputs`, `StructuredPlanner`, and `validate_numeric_output` interfaces. The earlier module-existence probe also failed while the module was absent; that placeholder test was removed from the final behavior suite as directed.

The first complete behavior run exposed an unpacking error in network batch validation. After that root cause was fixed, the reduced Task 2 suite exposed one strict float32-versus-JSON equality error on the same ego-history timeline. It now uses finite-value `allclose` for numeric history arrays while retaining exact boolean masks, matching Task 1's numeric validation policy.

Final command:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_structured_driver
Result: 6 tests passed, 0 failures/errors.
```

The six tests cover exact patch ordering and inactive-frame masking; strict feature/full-spec incompatibility; deterministic real forward; source and forecast-mode permutation invariance; masked-value invariance; gradients from both local and paid observation/forecast sources plus scene, ego motion/history and prior; zero gradients under masks; empty and one-valid-point training forwards; exact history/provenance/output/cost/parent contracts; numeric output validation; and incompatible runtime/model versions.

Additional checks:

```text
/root/autodl-tmp/conda-envs/llava/bin/python -m py_compile \
  src/planning/structured_driver.py src/planning/driver_contract.py \
  tests/test_structured_driver.py
Result: passed.

Fresh Python import of planning.driver_contract
Result: torch remained unloaded.

Forbidden dependency/identity scan over owned files
Result: no 7B, LLaVA, CLIP, tokenizer, Q9, track-handle, receipt or mode-index path in production.
```

The tiny test specification has 13,026 parameters and runs entirely on CPU synthetic inputs. These checks establish implementation and gradient contracts only; they provide no trained-driver quality, P/F utility, closed-loop safety, or scientific-effect evidence.

## Fix round 1: JSON-native archive boundary and canonical permutation coverage

The shared torch-free output validator now applies the repository JSON-native check to the complete output before inspecting its fields or normalizing waypoints. A NumPy waypoint array therefore fails at the archive boundary instead of being accepted and converted by `validate_plan`.

The model-version check now also rejects an empty `training.status`; name and revision remain semantic identifiers and optimizer steps remain a nonnegative exact integer.

The new end-to-end permutation fixture uses two distinct objects and six numerically distinct forecast modes. It reverses both local and remote raw window rows, permutes the predictor modes before any ledger or service call, obtains an authentic P receipt, acknowledges its known-field manifest, obtains an authentic F receipt, and rebuilds both prepared records with the unchanged Task 1 canonical builder. The test first proves that both raw source orders changed, that canonical entity IDs and observation tensors agree, and that the valid forecast arrays differ due to the real mode permutation. It then collates both records and verifies the same shared network output. The earlier direct tensor-level source/mode permutation check remains as a narrower model test.

RED evidence was observed before production changes:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python \
  -m unittest test_structured_driver
Result: 7 tests run, 1 failure: NumPy waypoints were accepted instead of rejected.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python \
  -m unittest test_structured_driver.StructuredDriverTests.test_wrapper_preserves_history_provenance_cost_and_numeric_validation
Result: 1 test run, 1 failure: empty training.status was accepted instead of rejected.
```

Final targeted command:

```text
cd tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_structured_driver
Result:
.......
----------------------------------------------------------------------
Ran 7 tests in 0.409s

OK
```

No full suite, real-data inference, model training, checkpoint creation, or scientific experiment was run in this fix round. Task 1 files and validators were not changed.


---

## Record: task-2-review.md

### Spec Compliance

- ❌ Issues found. The network, collator, wrapper, provenance, real numeric forward, cost record, parent closure, and shared torch-free validator are present, but the validator does not enforce the required JSON-native output contract (`src/planning/driver_contract.py:8`), and the required canonical-builder permutation test was replaced by direct post-collation tensor mutation (`tests/test_structured_driver.py:102`, `task-2-preflight.md:12`).
- ⚠️ Cannot verify from diff: the chronological RED-before-GREEN claim is recorded only in `task-2-report.md:91`; the supplied final six-test result is complete and clean, so I did not rerun it.

### Strengths

- `src/planning/structured_driver.py:83` validates every prepared sample through Task 1's strict validator and requires one complete identical `StructuredDriverSpec`; `src/planning/structured_driver.py:110` reproduces the regression-then-classification 5x4 patch layout and masks an inactive past frame. The focused Task 1 interface check confirmed fixed source slots, separate forecast modes, source/full-context fields, exact zero padding, and full-spec reconstruction in `src/planning/structured_inputs.py:486` and `src/planning/structured_inputs.py:637`.
- `src/planning/structured_driver.py:170` is a real shared numeric model: it uses frame/grid-coded scene patches, shared PointNet-style masked history encoders, per-entity source attention, separate forecast-mode tokens, ego history/motion, learned waypoint queries, and a residual prior condition. The masking paths at `src/planning/structured_driver.py:245`, `src/planning/structured_driver.py:259`, and `src/planning/structured_driver.py:274` keep invalid values out of final attention while retaining a safe null token.
- No entity identity, receipt, request/tool label, audit path, provider execution identity, language model, tokenizer, or road-map field is embedded. The only learned source/context signals are the Task 1 numeric source slots and forecast context (`src/planning/structured_driver.py:247`, `src/planning/structured_driver.py:261`).
- `src/planning/structured_driver.py:351` performs an actual model forward before emitting exactly the requested numeric output and cost fields; `model_executed=True` is set only after a finite `[1,6,2]` result. The focused output-interface check confirmed `validate_plan` applies the established six-point speed/acceleration contract (`src/tools/task_spec.py:202`).
- `src/planning/driver_contract.py:1` contains the single shared validator and imports no network module. The supplied fresh-import evidence says torch remained unloaded (`task-2-report.md:113`); no duplicate validation implementation exists.
- The synthetic tests exercise deterministic inference, active P/F observation and forecast gradients, zero gradients under the tested scene/P/F masks, prior/history/motion gradients, empty inputs, and the one-valid-point training case (`tests/test_structured_driver.py:91`, `tests/test_structured_driver.py:117`, `tests/test_structured_driver.py:151`). These are implementation checks only; neither code nor report makes a model-effect or real-data claim.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `src/planning/driver_contract.py:8`: `validate_numeric_output` checks the exact field set but never checks that the complete output is JSON-native. Its final call to `validate_plan` accepts NumPy arrays, so an output with `waypoints=np.ndarray((6,2))` can be accepted and normalized even though the Task 2 contract requires the output itself to be JSON-native. This weakens the shared offline/archive boundary that justified moving this validator forward. Apply the repository's JSON-native check before field validation and add a focused rejection test for a non-native waypoint/container value.
- `tests/test_structured_driver.py:102`: the permutation test flips observation-source and forecast-mode tensor dimensions after collation and sends the modified batch directly to the model. It therefore bypasses Task 1 semantic validation, does not rebuild matching entity metadata through the canonical builder as explicitly required by `task-2-preflight.md:12`, and never tests entity-row permutation. Keep the direct model-level check if useful, but add the required end-to-end fixtures by permuting authentic source/entity/mode inputs, rebuilding prepared records through the canonical builder, collating them, and comparing model outputs.

#### Minor (Nice to Have)

- `src/planning/structured_driver.py:291`: model-version validation accepts an empty `training.status` because it checks only `isinstance(..., str)`. That is weaker than the stated stable, actual training metadata contract and leaves an avoidable provenance hole. Require a nonempty semantic status and add the corresponding incompatible-version assertion; Task 4 can still define the later checkpoint-to-weights binding.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The model architecture and its main P/F, mask, prior, output, cost, and provenance behavior fit Task 2 and remain appropriately synthetic. The two Important gaps are narrow but contract-bearing: the shared validator accepts a non-native output representation, and the permutation evidence does not exercise the canonical prepared-input path that the preflight explicitly required.


---

## Record: task-2-fix-review.md

### Finding Verdicts

- **Torch-free validator accepts non-JSON-native output containers** — ADDRESSED. `src/planning/driver_contract.py:10` now applies `_check_json_native` to the complete output before field inspection or waypoint normalization, and `tests/test_structured_driver.py:263` verifies that NumPy waypoints are rejected.
- **Permutation coverage bypasses the canonical builder and omits entity-row permutation** — ADDRESSED. `tests/test_structured_driver.py:117` now reverses distinct local and remote raw rows and permutes six numerically distinct forecast modes before ledger/service construction; it obtains real P/F responses, rebuilds both records through the Task 1 builder, collates them, and compares shared-model outputs at `tests/test_structured_driver.py:170`. The assertions at `tests/test_structured_driver.py:162` also prove the raw row order changed while canonical entity IDs and observations agree and valid forecasts retain the intended mode permutation.
- **Model-version validation accepts an empty training status** — ADDRESSED. `src/planning/structured_driver.py:292` rejects an empty status, and `tests/test_structured_driver.py:260` covers that failure.

### New Breakage in the Fix Diff

- None. The changes are confined to the three findings and preserve the single torch-free validator, existing numeric model path, and prior tests. The appended report records the exact covering command with 7 tests passing and clean output at `task-2-report.md:142`; I did not rerun the suite.

### Out-of-Scope Observations

- None.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.


---

## Record: task-3-report.md

# Task 3 report: numeric driver integration

Status: COMPLETE, with the explicit Task 4 loader dependency below. No real dataset training, real GoT/MTR generation, real numeric task collection, or scientific experiment was executed. All new network execution used a tiny untrained CPU StructuredPlanner and synthetic windows through the real P/F service. Existing query-value regression tests include their own small synthetic fits.

## Owned changes

- `src/planning/method_episode.py`: one existing `run_task_episode` loop, numeric dispatch and actual prior slot, numeric failure/cost handling, validated live prefix replay.
- `src/planning/driver_contract.py`: shared model-free runtime/output/cost/episode contracts.
- `src/planning/query_data.py`: full structured semantic binding and numeric archive validation.
- `src/planning/bundle_data.py`: numeric one-shot archive grammar allows the common third same-evidence driver attempt; v1 keeps its original two-attempt grammar.
- `src/planning/run_framework.py`: lazy numeric checkpoint loader hookup and causal local pose history features/audit.
- `src/evaluation/framework.py`, `src/evaluation/planning.py`: numeric outputs, authentic costs and explicit cumulative-prefix metrics.
- `tests/test_structured_episode.py`: 16 real synthetic CPU integration tests.
- `src/planning/structured_driver.py`: parent-authorized interface fix only; numeric and boolean conversion copy a non-writable NumPy array before creating a writable torch tensor. A regression proves tensor mutation cannot mutate the frozen source buffer.

No changes were necessary in `method_controls.py`, `method_run_spec.py`, or `query_value.py`: their existing control, semantic-binding, and state-feature callers use the shared contracts. Parent-owned AGENTS/docs were not staged.

## Final APIs and archive fields

`validate_limits(limits)` now accepts `toolv2x_interaction_v2` with the same required top-level keys as v1. `receiver_spec` must be the complete strict `StructuredDriverSpec.to_dict()`; numeric defaults explicitly select `p_processing=observations_only`. `driver_version` is the name/revision pair from the actual structured `model_version`.

The shared loop validates the driver kind/spec/model binding before ledger construction or private service reads. Numeric ledgers opt into real local history and `toolv2x_evidence_ledger_v2`. P-local is independently selectable through the full structured spec and is billed by existing receiver events.

Every numeric evidence update calls:

```python
driver.prepare_input(features, motion, ledger, limits['receiver_spec'],
    previous_plan=prior_output['waypoints'] if prior_output else None,
    previous_parent_refs=prior_output['parent_refs'] if prior_output else ())
driver.plan_prepared(features, prepared)
```

All controls have this same numerical prior slot. Exact repeat deep-copies the entire prior prepared input, including its old prior slot. Same-evidence refinement copies the prior entities/tensors/admission and changes only `previous_plan` and `previous_parent_refs`, using the actual immediately preceding output. It does not rebuild or rerank evidence. Every result is checked by the genuine `toolv2x_numeric_plan_v1` contract; no Q9 text, tokenizer, Q8/Q9 execution flags or language-generation flags are invented.

Numeric episode additions:

```text
execution_kind = structured_numeric
feature_tokens = 0
numeric_scene_tokens = 220 or 440 (actual active local feature frames)
```

The existing `plans[].prepared`, `plans[].output`, ledger snapshots, costs, decisions and event ordering remain the archive structure. Policy plan `raw` is `None` for numeric output; numeric waypoints and the exact existing admission schema feed `state_features` successfully. The query semantic identity retains `driver_kind`, full `driver_spec`, `decoding`, and the complete `model_version={name, revision, training}` including every training extension field. Old GoT query checkpoints fail the real loader's expected-binding check before policy construction.

Shared model-free contracts now exported by `planning.driver_contract`:

```python
validate_driver_binding(limits, provenance)  # returns whether route is numeric
validate_numeric_cost(cost)
validate_numeric_output(output, prepared, execution_spec)
validate_numeric_episode(episode)
```

The episode validator checks numeric route identity, at most three driver attempts and two capability allowances, local-input stability, authentic receipt/actual-wire agreement (including bundle primitive decoding), snapshot progression, and exact ledger reconstruction of normal numeric Z. Exact/refinement inputs are instead compared against the actual frozen previous prepared record. Successful outputs must bind that exact Z, its parent closure, finite bounded six-point trajectories, and the actual admitted numeric token count. A coordinated entity/tensor replacement still fails against the untouched source ledger. Live prefix replay uses this validator before continuing, so a coherent but replaced Z cannot reach another private query. Failed attempts with a prepared input and missing output stay auditable.

Cost evaluation retains all original fields and their meanings. Numeric calls report zero `input_tokens`, `output_tokens`, and language `feature_tokens`, plus separate `numeric_token_count` and `numeric_output_points`. The `generation_seconds` diagnostic reads the actual numeric model duration; `driver_seconds` remains the outer driver attempt duration. Total compute adds outer stages once and does not add nested generation/MTR durations again. Stage cost must match the actual output cost; numeric model duration cannot exceed its enclosing attempt. Missing/invalid cost remains incomplete, never zero. Actual invalid numeric answers remain recorded failures, preserving raw waypoints and spent attempts.

One-shot numeric archive configurations support either two driver calls with no extra generation, or three with explicit `exact_repeat`/`self_refinement`. Both keep a single external RPC and no more than two primitives. Runtime feedback, frozen feedback, current-old/union, evidence refinement, exact/self repeats, ego-only maximum capacity, and existing Ego/P/F/PF/rule controls all use the common loop.

## Runtime hookup required in Task 4

Inside `_load_interaction_runtime`, explicit numeric limits lazily import:

```python
from planning.structured_driver import load_structured_planner
planner = load_structured_planner(Path(config['checkpoint']), device='cuda')
```

Task 4 must provide that exact function (or a re-export there) and preserve the actual loaded full model/training provenance. This function did not exist and was not invoked during Task 3. The existing original frozen CMP loader still runs only when the user explicitly executes the runtime, and the legacy GoT planner is imported only on its original branch. No placeholder model loader was created.

For numeric runtime samples, `load_ego_history` reads legal local past poses only and adds `ego_pose_history`, `ego_pose_history_valid`, `ego_pose_history_times` as separate NumPy feature arrays. They are saved in the existing `inputs.feature_path` NPZ. The attempted physical pose paths are saved separately in `task.inputs.ego_history_read_paths`. Task 2 deliberately places `read_paths=[]` inside `prepared.ego_history_used`; physical audit paths do not enter learning or semantic identity. Existing `task.row`, `task.inputs.artifact_root`, and `task.inputs.feature_path` remain the exporter interfaces. No future-bearing fields were added to the online index.

## Metrics

Existing endpoint `L2_1`, `L2_2`, valid-point `ADE3`, and `FDE3` definitions remain intact. New metrics are:

- `got_prefix_L2_1s`: mean point error over the first two points, only if both labels are valid.
- `got_prefix_L2_2s`: mean over the first four, only if all four are valid.
- `got_prefix_L2_3s`: mean over all six, only if all six are valid.
- `got_prefix_L2_avg`: mean of the three prefix means, only if all three exist.

The new metrics appear in trajectory quality and method task/group summaries. The formula matches the previously inspected original GoT prefix calculation; this establishes no equivalence of task, splits, supervision, model inputs or paper results. Collision and closed-loop performance remain unevaluated. Formal `test` research-role support remains deferred; existing train/validation role rules were not relabeled or expanded.

## TDD and validation evidence

RED was observed for full numeric limits/flow and cumulative-prefix behavior before integration. Further explicit red/green checks covered semantic model identity, coordinated prepared replacement, authentic receipt wire mismatch, readonly-buffer aliasing, numeric bundle refinement archive admission, invalid output/failure preservation and invalid cost handling, live-prefix coordinated Z replacement, and coordinated nested-cost inflation. Subsequent implementation fixes were limited to those observed contracts.

Final integration count is **16**, not the earlier intermediate 15: the last added test rejects coordinated Z replacement in a live captured prefix before any peer read. Coverage includes real P/F feedback and actual priors/second request/STOP, all control arms, shared paid prefixes, one-shot and one-shot archives, ordinary numeric branch archives and query features, P-local billing and local pose arrays, old-policy rejection, exact/refinement differences, output/field/receipt/cost/model tampering, failures, numerical metrics, and readonly buffers.

### Initial broad targeted regression (resource path error, not a code failure)

Working directory: `tests`.

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_method_episode test_method_controls test_method_evaluation test_query_data test_bundle_data test_method_run_spec test_query_value test_control_bundle test_planning_evaluation test_structured_driver test_structured_episode
```

Result: **160 tests, 159 passed, 1 error**, 352.053 seconds. The single error was `test_method_controls.ControlModelTests.test_original_planner_accepts_explicit_refinement_and_rejects_tampering`: the worktree-default tokenizer path incorrectly resolved under `.worktrees/V2V-GoT`. No model was loaded or generated; this existing test loads only the local original tokenizer and replaces `_generate`. The then-current integration suite had 11 tests. Do not describe this initial run as a clean PASS.

### Corrected final numeric/network/evaluation and failed-resource coverage

Working directory: repository root.

```text
PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOOLV2X_V2VGOT_ROOT=/root/autodl-tmp/V2V-GoT TOOLV2X_CMP_ROOT=/root/autodl-tmp/CMP TOOLV2X_LLAVA_BASE=/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b TOOLV2X_CLIP_ROOT=/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_structured_episode test_structured_driver test_method_evaluation test_planning_evaluation test_method_controls.ControlModelTests
```

Result: **43 tests passed**, 35.075 seconds: 16 numeric integration + 7 network + 19 old evaluation + the one corrected original-tokenizer/planner contract. No warnings/errors.

After the final numeric nested-cost guard was added, the exact affected checks were rerun:

```text
PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_structured_episode.StructuredEpisodeTests.test_paid_receipt_must_match_real_wire_and_all_stages_keep_binding test_structured_episode.StructuredEpisodeTests.test_numeric_feedback_uses_paid_evidence_actual_prior_and_cost test_structured_episode.StructuredEpisodeTests.test_failed_driver_attempt_keeps_prepared_and_paid_cost test_method_evaluation
```

Result: **19 tests passed**, 2.261 seconds. Includes coordinated output/event nested-cost inflation and unchanged old method evaluation.

After the numeric-only bundle archive extension:

```text
PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_bundle_data.BundleDataTests
```

Result: **8 old bundle-data tests passed**, 53.534 seconds.

Additional checks passed: `python -m py_compile` over all modified Python files and the new test; `git diff --check`; `PYTHONPATH=src python -m planning.run_framework --help`; fresh imports of `planning.run_framework`, `planning.driver_contract`, and `evaluation.framework` asserting neither torch nor transformers was imported. No full lightweight or full model-environment suite was rerun here; the parent will perform the final integrated suites after Task 4/review.

## Limitations

This is contract/integration evidence for an untrained network on synthetic inputs. It establishes no planning quality, P/F scientific utility, calibration, closed-loop safety or paper gain. The actual numeric checkpoint initializer/serializer/loader and real task-training export are Task 4 work, not executed reproduction. No push or external publication was performed.


---

## Record: task-3-review.md

# Task 3 review

## Spec Compliance

- ✅ Spec compliant within Task 3 scope. Numeric limits retain the same top-level contract, require the structured receiver specification, and reject driver-kind mismatch before ledger construction/private reads (`src/planning/method_episode.py:22`, `:140`; `src/planning/driver_contract.py:39`). The original text branch keeps its tokenizer, context, raw-answer and direct-generation checks (`src/planning/method_episode.py:273`, `:326`).
- ✅ One shared executor handles numeric preparation, actual prior coordinates/parent refs, output validation and failed-attempt recording; no second episode runner or fabricated Q9 fields were introduced (`src/planning/method_episode.py:256`, `:298`, `:322`; `tests/test_structured_episode.py:55`, `:188`, `:315`).
- ✅ Exact repeat copies the whole preceding prepared record. Numeric same-evidence refinement only updates its prior coordinates/refs; archive validation compares with the frozen predecessor rather than reranking evidence (`src/planning/method_episode.py:259`; `src/planning/driver_contract.py:115`; `tests/test_structured_episode.py:87`).
- ✅ Full model/training identity enters query semantic binding, numeric archive validation replaces the Q9-specific path, and numeric one-shot archives accept only the explicit two/three-attempt grammar (`src/planning/query_data.py:65`, `:263`; `src/planning/bundle_data.py:29`, `:233`; `tests/test_structured_episode.py:216`, `:271`, `:298`).
- ✅ Numeric raw trajectory, prepared-input identity, parent closure, measured output cost and admitted token count are checked; evaluation separates language tokens, numeric tokens/points and nested model duration (`src/planning/driver_contract.py:8`, `:28`, `:136`; `src/evaluation/framework.py:207`, `:279`).
- ✅ Cumulative prefix metrics require every label in the relevant prefix, and the all-prefix average requires all six labels. Existing endpoint L2 and valid-point ADE/FDE formulas remain (`src/evaluation/planning.py:43`; `tests/test_structured_episode.py:111`).
- ✅ Runtime changes are a lazy numeric-loader hook and separate causal local history arrays/read audit; diagnostic ego capacity handling preserves the strict numeric specification (`src/planning/run_framework.py:396`, `:427`, `:472`).
- ✅ No edits were required in conditionally listed `method_controls.py`, `method_run_spec.py` or `query_value.py`: existing contracts dispatch through shared functions. Focused checks below establish the control/run-spec interfaces; the real state-feature and old-policy rejection tests cover the unchanged value caller (`tests/test_structured_episode.py:216`, `:298`). The small readonly-buffer fix in `structured_driver.py` is explicitly authorized and has a mutation-isolation regression (`src/planning/structured_driver.py:34`, `:51`; `tests/test_structured_episode.py:208`).
- ⚠️ Cannot verify from diff: Task 4's `load_structured_planner` is explicitly unimplemented at this task boundary. The hook at `src/planning/run_framework.py:398` is a future dependency, not a usable checkpoint loading route. Controller must verify the exact loader signature, actual model/training provenance, initializer/serializer/resume behavior and feature/history exporter binding in Task 4 (`task-3-report.md:60-71`, `:130`).
- ⚠️ Cannot verify from diff: end-to-end time-t data provenance, full-context F inference before ranking, P never invoking MTR, and identical full-source P-local/F numeric tensors are cross-task/provider contracts. This task calls existing services and explicitly selects P processing (`src/planning/method_episode.py:149`); its synthetic P-local test establishes billing and history use, not the full information-equivalence theorem (`tests/test_structured_episode.py:337`). Controller should use Task 1/2 evidence for these constraints.
- ⚠️ Cannot verify from diff: all historical v1 golden behavior and complete lightweight/model-environment suites. The changed original branches and supplied targeted regression results support this task gate; final integrated suites remain the controller's responsibility (`task-3-report.md:98-126`). Formal research `test` role remains deferred, without relabeling validation (`task-3-report.md:82`).

## Strengths

- Receipt-wire decoding and normal-input reconstruction bind numeric values to paid source records; coordinated entity/tensor replacement cannot pass simply by preserving ref lists. Live captured prefixes use this validation before continuation (`src/planning/driver_contract.py:93-135`; `src/planning/method_episode.py:184`; `tests/test_structured_episode.py:152`, `:239`, `:353`).
- Integration tests execute a real tiny CPU StructuredPlanner through synthetic real P/F services, with the predictor substituted only at the MTR fixture boundary. They check actual changed refinement outputs, no peer reads before the first plan, control prefixes, archives and failure charges (`tests/test_structured_episode.py:27-52`, `:55`, `:87`, `:124`, `:188`).
- Cost evaluation retains outer attempt timing and does not add nested model durations to total compute; coordinated event/output duration inflation is explicitly rejected (`src/evaluation/framework.py:216-227`, `:260-267`; `tests/test_structured_episode.py:239`).
- The additional shared contract module remains model-free, and runtime model loading remains inside the explicit runtime function (`src/planning/driver_contract.py:1-5`, `:63-74`; `src/planning/run_framework.py:371-405`).

## Issues

### Critical (Must Fix)

- None found in the reviewed Task 3 change.

### Important (Should Fix)

- None found in the reviewed Task 3 change. The declared Task 4 loader dependency is tracked above, not assumed complete or misclassified as a skipped Task 3 implementation.

### Minor (Nice to Have)

- None requiring a change for this task gate.

## Focused checks and validation evidence

- Reviewed the supplied task brief, preflight, global constraints and implementation report. Read the supplied diff once logically; the first tool result was truncated, so the omitted middle was recovered in bounded slices of the same package. No Git commands or broad repository scan were run.
- Named risk: numeric Z reconstruction might trust unverified acquired/derived values. Checked the unchanged builder and its `remote_units` boundary: `src/planning/structured_inputs.py:565-629` calls `remote_units`, and `src/planning/evidence.py:268-331` calls the existing receipt manifest validator and checks derived context/parent chains. Did not re-audit the underlying provider or manifest implementation.
- Named risk: unchanged repeat/prefix APIs could silently rebuild evidence, alter the prior slot, or reject NumPy history fields. Checked `src/planning/method_controls.py:12-46`, `:113-119`, `:285-298`: controls cap attempts at three, repetition is driven by the declared control, and live prefix matching compares arrays by dtype/content. The new loop handles prior copying explicitly.
- Named risk: one-shot could spend more primitive allowances than the common remaining budget. Checked existing bundle construction at `src/planning/method_controls.py:163-187`: it uses `min(2, remaining calls)` and passes that allowance into bundle limits. No provider implementation was re-audited.
- Named risk: unchanged run-spec freezing could strip structured identity. Checked `src/planning/method_run_spec.py:66-85`: it validates limits and derives saved settings through the changed `_semantic_binding`.
- Diff context cut off shared functions needed to judge compatibility/failure behavior. Read only the missing relevant portions of `method_plan_row`, `_method_cost`, and the post-generation episode loop (`src/evaluation/framework.py:299-324`, `:143-197`, `:240-267`; `src/planning/method_episode.py:344-480`). Original outputs populate `quality`; total compute sums outer stages; STOP/service/receiver transitions retain the shared runner.
- No suite or focused test was rerun: code inspection raised no concrete unresolved doubt requiring execution. Exact reported evidence: initial broad run **160 tests, 159 passed, 1 resource-path error, 352.053 s** (`task-3-report.md:95-98`); corrected numeric/network/evaluation plus failed-resource coverage **43 passed, 35.075 s**, no warnings/errors (`:105-108`); final affected cost checks **19 passed, 2.261 s** (`:113-116`); existing bundle-data checks **8 passed, 53.534 s** (`:121-124`). The initial run is not a clean PASS. Compilation, help and model-free import checks are reported at `:126`; these were not independently rerun.
- The report's RED/green chronology is a supplied claim (`task-3-report.md:86`), not independently reconstructed history. New test assertions were reviewed directly. No real data training, model generation, runtime execution, external publication, or push was performed by this review.
- Controller-supplied cross-task update: `tests/test_structured_inputs.py:test_full_p_local_and_equivalent_f_have_identical_forecast_encoding` and a focused real-receipt fixture check establish equality of all `tensor_inputs`, `ego_motion` and `previous_plan` after adding equivalent F to P-local. The controller reports PASS after correcting an initial fixture-class import error. This is fake-predictor contract evidence, not actual prediction-effect evidence, and was not rerun by this reviewer.

## Assessment

**Task quality: Approved.**

The numeric route shares the existing execution/control machinery while providing separate authentic output and archive contracts. Approval is the Task 3 scope gate; the explicit Task 4 loading/export dependency and controller's cross-task/integrated checks remain outstanding.


---

## Record: task-4-report.md

# Task 4 report: numeric supervision and resumable training

Status: COMPLETE within the authorized implementation and synthetic verification scope. Independent task/branch review and final full regression suites belong to the parent. No real dataset training, numeric task collection, actual resource initialization, GoT/MTR generation, push, or research experiment was performed.

## Owned files

- `src/planning/train_structured_driver.py`: offline exporter, explicit initializer, masked numeric supervision, training/resume, checkpoint loading, CLI.
- `src/planning/structured_driver.py`: six-line `load_structured_planner` re-export with the exact runtime signature. Existing network/collator behavior was preserved.
- `tests/test_structured_training.py`: 12 tiny synthetic CPU tests built from the actual numeric episode and task archive schemas, including real synthetic P/F and one-shot bundle service execution.
- `scripts/check_review.py`: register only `test_structured_inputs`; no torch tests added to the lightweight runner.

Parent-owned dirty documentation and AGENTS were preserved and not staged. This report is intentionally retained in the existing ignored `.superpowers` working record directory.

## APIs and representation

```python
prepare_training_rows(tasks, labels)
masked_trajectory_loss(predicted, target, valid)
training_loss(model, features, row, *, refinement_depth=0, task=None)
initialize(checkpoint, driver_spec, *, seed=0)
fit(rows, out, config, *, resume=None, stop_after_steps=None)
load_structured_planner(checkpoint, *, device='cpu')
```

`planning.structured_driver.load_structured_planner(checkpoint, *, device='cpu')` is the runtime entry point already imported lazily by Task 3. It restores the actual state into `StructuredPlannerNetwork`, switches to eval mode, and supplies the saved complete `model_version` to `StructuredPlanner`.

`prepare_training_rows` accepts a numeric run directory, its `tasks.jsonl`, or actual task dictionaries whose archived `task.json` lives at `inputs.artifact_root`. Labels can be an independent JSONL path or label dictionaries. A manifest entry is the existing `{sample_id, path}`; the task is the existing `{row, episode, inputs: {feature_path, artifact_root, ego_history_read_paths}, ...}`. No alternate simplified task schema was invented.

Exported stage rows are small JSON-native references:

```text
version, sample_id, scene, g, role, recording, stage, source_task
inputs: {feature_path, prefix: [0, ..., stage]}
supervision: {waypoints, valid, times_seconds, label_read_paths, scope}
```

Neither padded prepared tensors nor whole feature arrays are duplicated into each exported row. The ordered prefix consists of indices into one actual source task. Offline labels retain native full-precision waypoints and null/mask pairs; Q8/Q9 target strings are ignored. Sample identity must establish the exact scene/local-frame join, with matching g and research role. Duplicate/ambiguous label identities, recording-role overlap, inconsistent same-frame targets, malformed numeric source records, obsolete GoT episodes, incompatible features/history/masks, and row/source disagreement fail closed. The actual feature mask must match the archive's numeric scene-token count.

A usable prepared input remains eligible when the source output failed, including both a driver exception and finite output rejected by motion admissibility. The original failure and output are retained unchanged. Records without a usable numeric prepared input are rejected; this exporter does not silently synthesize one.

## Training objective and checkpoint semantics

The exact inference collator and `StructuredPlannerNetwork` are reused. Each selected stage row supervises every ordered prefix forward plus `refinement_depth` extra forwards at the final evidence stage. Per-forward masked coordinate SmoothL1 losses are averaged within each row; row objectives are averaged within each batch. Thus earlier prefixes deliberately occur in multiple stage-row objectives. Validation uses the same objective on validation-role rows with no optimizer update. This is offline observed-trajectory imitation.

All numeric prior slots are replaced before model forward. A smaller-evidence prefix provides its own detached actual prediction to the next enriched stage. Same-evidence refinement uses the preceding detached prediction. Archived exact-repeat stages retain the preceding model input's original prior slot, including its missing-prior mask. Each supervised forward uses the same network weights. Invalid model priors are masked and counted with the actual admissibility reason; targets never fill that slot. Rows with no valid labels return no loss and cause no optimizer update when the batch contains no supervised rows.

Feature loading is bounded by the selected batch. Input preparation/snapshotting reads one shared task or feature file at a time. Each training output has one `inputs/task_NNNNNN.json` and `inputs/feature_NNNNNN.npz` per shared source. Training reads these saved snapshots, preventing later live-source changes from entering the run. Resume compares source task semantics and feature arrays directly against the preceding run snapshots. Read paths remain audit references; a tested input-directory relocation with identical content resumes successfully.

Checkpoints include actual model parameters, optimizer state, Python/NumPy/torch/CUDA RNG states, current epoch, optimizer steps, processed batches, permutation, row position, complete config, recording/row binding, shared-input layout, and training counters. A direct duplicate binding of model/config/version and continuation state is checked before load/resume. This detects ordinary state/config/data mismatch; it is not an adversarial authenticity mechanism. The duplication trades checkpoint size for straightforward direct comparison.

`checkpoint_000000.pt` is saved before the first training batch. Subsequent checkpoint names count processed batches, including wholly unsupervised batches; this also bounds saving when optimizer steps remain zero. Save interval is measured in batches, so lost optimizer work is bounded by at most that many batches rather than an entire epoch. Final/explicitly bounded-stop state is saved. Saves write and flush a temporary file, then atomically publish with a no-overwrite hard link. Existing output directories/checkpoint paths are refused. Resume writes a new output directory and carries forward full state.

The parent's explicit ruling is implemented as `model_version.training.training_run_id`, a standard UUID text assigned to each initialization/new independent training run and preserved through reload/resume. It represents run lineage, is not computed from contents or paths, and never enters the numeric model. Two independent runs with identical numerical parameters may conservatively have different policy bindings. Direct state comparisons remain in place. Nested training metadata also carries status, optimizer steps, seed, full configuration, and recording/sample-stage binding. Zero-step checkpoints are explicitly `initialized_untrained`.

`report.json` records observed training/validation row counts, supervised forwards, masked-prior reasons, unsupervised rows, progress, validation loss, and evidence-stage counts. P/F coverage comes from actual ledger primitive receipts, including bundled P/F. These are observed counts, not a claim of complete scenario coverage or benefit.

## Exact CLI and config examples

These are documentation templates, not commands executed on real resources during this task. All subcommand signatures were checked using actual `--help` output. `train` is the canonical training command; `fit` is an alias. Resume is `train --resume` into a new directory.

The existing complete spec constructs the config without another schema class:

```python
import json
from pathlib import Path
from planning.structured_inputs import StructuredDriverSpec

spec = StructuredDriverSpec().to_dict()
config = dict(
    version='toolv2x_structured_training_v1',
    driver_spec=spec,
    seed=7,
    optimizer=dict(name='AdamW', lr=0.0001, weight_decay=0.01),
    batch_size=2,
    epochs=2,
    refinement_depth=1,
    save_interval=10,
    device='cpu',
)
Path('structured_spec.json').write_text(json.dumps(spec, indent=2))
Path('structured_training.json').write_text(json.dumps(config, indent=2))
```

All config fields shown above are required; unknown fields are rejected. The collected receiver spec and configured driver spec must agree exactly. CPU is shown for clarity; only tiny CPU execution was verified in this task.

```bash
PYTHONPATH=src python -m planning.train_structured_driver initialize \
  /path/to/new_untrained.pt --spec structured_spec.json --seed 7

PYTHONPATH=src python -m planning.train_structured_driver prepare \
  --tasks /path/to/numeric_run/tasks.jsonl \
  --labels /path/to/independent_offline_labels.jsonl \
  --out /path/to/new_prepared_rows.jsonl

PYTHONPATH=src python -m planning.train_structured_driver train \
  --rows /path/to/new_prepared_rows.jsonl \
  --config structured_training.json --out /path/to/new_training_run

PYTHONPATH=src python -m planning.train_structured_driver train \
  --rows /path/to/new_prepared_rows.jsonl \
  --config structured_training.json --out /path/to/new_resumed_run \
  --resume /path/to/new_training_run/checkpoint_000010.pt
```

Initialize is an explicit bootstrap for later numeric task collection. `fit` starts a seeded new training run unless `resume` is supplied; an initializer file is not a training-resume checkpoint. Collection itself remains the existing runtime entry point and was not executed here.

## RED/GREEN chronology and exact verification

The initial test import accidentally exposed the imported `StructuredEpisodeTests` class to unittest discovery: 22 tests ran, comprising 16 existing passes and six deliberate missing-module failures. The fixture import was corrected to a module import. The authoritative initial RED was then **6 failures in 6 tests**, all explicit `numeric training module is missing` assertions.

After implementation, the first six-test run had one fixture expectation mismatch: actual failed-driver status is `driver_error`, not the test's `failed`. Correcting that assertion produced **6 tests passed, 5.430 seconds**.

Meaningful follow-on RED cases were observed before their fixes:

- Mutated continuation progress was accepted instead of rejected. Added direct complete-continuation binding, also covering optimizer/RNG state.
- `training_run_id` was absent. Added the parent-approved lineage metadata and reload/resume/new-run assertions.
- Exact-repeat prior masks were `[False, True, True, True]` instead of `[False, False, False, True]`. Preserved the original repeated input's prior slot before an explicit extra refinement.
- Invalid-prior reason was absent despite a positive masked count. Added reason counters from the real admissibility failure.
- Modified feature active mask was accepted despite different archived numeric tokens. Bound the actual mask to the archive count.
- Actual one-shot bundle training failed because top-level bundle requests lack `tool`. Switched coverage reporting to the actual primitive receipt requests. The final test expects exactly `{'Ego': 1, 'PF': 2}`.

An intermediate 8-test suite passed in 12.122 seconds. Nine training tests passed in 11.418 seconds, and a 16-test training/network run passed in 11.803 seconds. These precede the final feature/relocation/bundle checks.

Working directory for final focused commands: repository root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_structured_training test_structured_driver test_structured_episode
```

**34 tests passed, 43.876 seconds**: 11 training + 7 network + 16 numeric episode. This covers actual-schema exporting, failure retention, literal masked loss/gradient, recording isolation, validation not updating weights, all-missing labels, invalid-prior masking, direct state tampering, exact CPU interruption/resume, directory relocation, shared prefix/refinement behavior, and the existing numeric integration. This run preceded only the final bundle-coverage correction.

After the bundle RED and fix:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_structured_training test_structured_driver
```

**19 tests passed, 15.063 seconds**: all final 12 training tests plus 7 network tests. No errors or warnings. Existing numeric episode code was unchanged by the bundle-coverage correction.

A separate fresh-process `test_structured_inputs` run passed **18 tests, 1.636 seconds**, followed by assertions that neither `torch` nor `transformers` was imported. This checks the sole new lightweight-suite registration without rerunning the full lightweight suite.

Additional checks passed: actual top-level/initialize/prepare/train CLI help, `py_compile` of owned Python files, `git diff --check`, and fresh imports of both training and runtime asserting no torch/transformers import. The initializer/reloader tests used only seeded tiny temporary checkpoints. All optimizer states, task archives, and feature snapshots created by tests were temporary.

## Limits and deviations

- No actual dataset optimizer run, trained research checkpoint, resource-backed numeric collection, real MTR/GoT generation, GPU parity test, or scientific/closed-loop claim.
- Full lightweight and model-environment suites intentionally remain for the parent's final integrated validation.
- Direct binding and readable snapshots intentionally increase checkpoint/disk size; they avoid deriving semantic identity from artifact location and do not introduce another storage dependency.
- Fresh training optimizes the configured seeded model using causal generated prefixes; loading an untrained initializer as a resume state is intentionally unsupported because it has no optimizer/data/permutation binding. Seeded initializer weights are available to bootstrap task collection.
- No upstream UniV2X/VAD reproduction claim. This implements the approved shared numeric planner path with existing standard components.

## Review fix round 1: require explicit physical train provenance

Addressed the Important finding in `task-4-review.md`. The real runtime's `select_rows` admits both research roles only from explicit physical `train` data. The common `_task_rows` boundary now rejects any `physical_split` other than literal `train`, including an omitted field. Both export and `fit` saved-row revalidation reuse this two-line guard. The actual archive fixture now carries `row.physical_split='train'`. No schema redesign or additional feature was introduced.

Two tests cover all eight combinations of entry point (export / fit), research role (train / validation), and invalid physical provenance (test / missing). The fit test first saves legitimately prepared JSONL rows, then changes the source task's physical split; a separate valid train recording accompanies validation rows so absence of a training role cannot accidentally satisfy the rejection test.

RED command, repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_structured_training.StructuredTrainingTests.test_export_requires_physical_train_for_both_research_roles \
  test_structured_training.StructuredTrainingTests.test_saved_rows_cannot_bypass_physical_split_guard_during_fit
```

Observed RED: **2 tests, 8 subtest failures, 6.543 seconds**, each `ValueError not raised`. This demonstrates both the export hole and the saved-row fit bypass on tiny synthetic archives.

After the shared guard, GREEN command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest test_structured_training
```

Observed GREEN: **14 tests passed, 16.512 seconds**, `OK`; no warnings/errors. `git diff --check` passed. No full suite or real data run was executed. Only `src/planning/train_structured_driver.py` and `tests/test_structured_training.py` are staged for the fix; parent documentation changes remain untouched.

## Final-validation portability fix: self-contained subprocess source resolution

The parent's full discovery exposed a test-environment failure: the README command does not set `PYTHONPATH`, while the focused command did. The clean-import child therefore could not import `planning`. No production defect or closed review finding was reopened.

Changed only `tests/test_structured_training.py`: both child subprocesses now run with `cwd` set to the repository `src` directory derived from `Path(__file__).resolve()`. Their source resolution no longer depends on the parent's `PYTHONPATH`. The clean-import child asserts that neither torch nor transformers was imported; CLI help is still executed in a separate clean child process.

Reproduced RED and verified GREEN with exactly:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest discover \
  -s tests -p 'test_structured_training.py' \
  -k test_seeded_initializer_reload_and_clean_help
```

RED: **1 test failed, 0.222 seconds**, child `ModuleNotFoundError: No module named 'planning'`. After the test-only fix: **1 test passed, 0.290 seconds**, `OK`.

Then ran the current complete training test module under the same unset environment:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest discover \
  -s tests -p 'test_structured_training.py'
```

Result: **15 tests passed, 17.326 seconds**, `OK`. `git diff --check` passed. No full suite, real model/data training, or production edit was performed. Only the test file is committed; parent documentation and validation logs remain untouched.


---

## Record: task-4-review.md

# Task 4 review

## Spec Compliance

- ❌ Issues found: physical test rows can be relabeled as train/validation at the exporter boundary. `src/planning/train_structured_driver.py:65` checks research role but omits the literal archive's `physical_split`; `src/planning/train_structured_driver.py:358` reuses the same incomplete check during fitting. This violates the binding requirement that train/validation must not be relabeled test data. One Important finding below.
- ✅ Requested files are present in the supplied diff: training/export/initializer/CLI at `src/planning/train_structured_driver.py:1`, runtime loader re-export at `src/planning/structured_driver.py:371`, synthetic tests at `tests/test_structured_training.py:1`, and torch-free suite registration at `scripts/check_review.py:12`.
- ⚠️ Cannot verify from diff: full branch compatibility and required final lightweight/model-environment suites. The implementer's task report records focused synthetic results; the controller must run the final integrated suites after fixes, as planned.
- ⚠️ Cannot verify from diff: real-resource numeric collection, CUDA exact resume, real data trainability at scale, and scientific benefit. These were explicitly excluded from this task; no additional run is requested by this review and no such claim is supported.

## Strengths

- Exact inference representation is reused through `collate_structured_inputs` and `StructuredPlannerNetwork` (`src/planning/train_structured_driver.py:169`, `src/planning/train_structured_driver.py:431`). Targets are built separately, archived numeric priors are cleared before forward, and smaller-evidence/refinement priors come from detached model predictions (`src/planning/train_structured_driver.py:176`, `src/planning/train_structured_driver.py:186`, `src/planning/train_structured_driver.py:211`). The actual forward hooks and nonzero prior-encoder gradient test inspect these behaviors (`tests/test_structured_training.py:277`).
- Independent labels retain six full-precision points, missing-point masks, scene/sample/g/role agreement, and offline label scope; recording overlap and inconsistent same-frame supervision are rejected (`src/planning/train_structured_driver.py:34`, `src/planning/train_structured_driver.py:118`). Invalid source output is retained when prepared input remains usable (`src/planning/train_structured_driver.py:68`, `tests/test_structured_training.py:82`).
- Literal masked SmoothL1 value and gradient are asserted, including masked NaNs; validation has no optimizer update and fully missing labels produce no optimizer state (`tests/test_structured_training.py:71`, `tests/test_structured_training.py:116`).
- Shared source/task snapshots and per-row indices avoid duplicating full feature arrays into every stage row (`src/planning/train_structured_driver.py:382`); training reads bounded selected-row snapshots (`src/planning/train_structured_driver.py:465`). Resume compares readable source semantics and actual feature arrays while allowing relocation (`tests/test_structured_training.py:224`).
- Checkpoints duplicate state/config/continuation for direct mismatch checks, capture optimizer/RNG/permutation/position, save before training and at bounded batch intervals, and publish without overwrite (`src/planning/train_structured_driver.py:216`, `src/planning/train_structured_driver.py:278`, `src/planning/train_structured_driver.py:298`, `src/planning/train_structured_driver.py:449`). Exact CPU interrupted/resumed parameters and optimizer state are tested (`tests/test_structured_training.py:186`).
- Seeded initialization is explicit and labeled untrained. The parent-approved random training lineage ID is assigned to independent runs and retained on resume (`src/planning/train_structured_driver.py:286`, `src/planning/train_structured_driver.py:446`), consistent with `.superpowers/sdd/2026-09-14-structured-driver/progress.md:30` and `progress.md:66`.
- Runtime loading restores actual state strictly, sets evaluation mode, and passes the saved full model version into the existing planner (`src/planning/train_structured_driver.py:320`; `src/planning/structured_driver.py:371`).

## Issues

### Critical (Must Fix)

- None found in the reviewed scope.

### Important (Should Fix)

1. **Reject physical test data and missing physical split before export or fitting.** `src/planning/train_structured_driver.py:65` accepts any task whose `row.role` is train/validation, even when its explicit `row.physical_split` is test. `_verify_rows` calls this same path at `src/planning/train_structured_driver.py:358`, so it does not close the hole. The real runtime already requires physical train provenance at `src/planning/run_framework.py:114`; both train and validation research roles originate there. The synthetic archive fixture changes role but omits physical split (`tests/test_structured_training.py:54`), masking the missing literal-schema constraint. A focused temporary probe changed only the archived row's physical split to test and obtained `{'source_physical_split': 'test', 'source_role': 'train', 'accepted_rows': 3, 'exported_roles': ['train']}`. Require explicit `physical_split == 'train'` in the common source-task validator, update the fixture to preserve this actual runtime field, and test rejection of physical test/missing split for both allowed research roles, including the fit entry point so saved rows cannot bypass it.

### Minor (Nice to Have)

- None requiring a separate change.

## Focused Checks and Evidence Boundary

- Read the supplied review package once. Its initial tool output was truncated mid-file; read only the missing middle/test-setup segment to complete that pass. No changed source file was reopened to rereview its implementation; source line references were derived from the supplied diff.
- Named risk: whether the imported numeric validator actually reconstructs causal evidence and permits failed prepared attempts. Checked only its implementation at `src/planning/driver_contract.py:63`; it reconstructs prepared inputs from paid receipt ledgers and requires valid outputs only for attempts claiming success.
- Named risk: whether label shape/role and recording semantics match existing producers. Checked `src/planning/adaptation_data.py:37`, its label/index output fields at `src/planning/adaptation_data.py:162`, and `src/common/audit_protocol.py:21`. This confirmed full-precision labels, optional separate scene, physical train index provenance, and whole-recording grouping.
- Named risk: whether feature history and masks are validated by the reused inference collator, and whether the loader binds model/spec/provenance. Checked the unchanged collator at `src/planning/structured_driver.py:83`, feature-history helper at `src/planning/structured_driver.py:57`, and planner constructor at `src/planning/structured_driver.py:308`.
- Named risk: literal runtime archive and loader compatibility, including physical split. Checked relevant producer/call-site fields in `src/planning/run_framework.py:114`, `src/planning/run_framework.py:397`, and `src/planning/run_framework.py:558`. The physical split omission above is a new concrete failure, not an inferred authenticity requirement.
- Executed one focused physical-split rejection probe using the existing synthetic episode fixture, a temporary directory, and `/root/autodl-tmp/conda-envs/llava/bin/python -B` with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests`. It completed in 2.156 seconds and reproduced acceptance of the invalid physical-test archive. No suite was rerun, no actual resource was initialized/trained, and no persistent model artifact was produced.
- Implementer-reported final focused evidence is readable: 19 training/network tests passed after the bundle fix; prior 34 training/network/episode tests passed; 18 torch-free input tests passed. These reported runs were inspected rather than regenerated. Their stated final output contains no warnings.

## Assessment

**Task quality: Needs fixes.**

The supervision, causal prior handling, shared snapshots, explicit initialization, and CPU exact-resume implementation are coherent and meaningfully tested. The exporter currently drops an existing physical-split invariant, allowing test data into train/validation despite otherwise strict source/label checks; fix and cover that boundary before approval.


---

## Record: task-4-fix-review.md

# Task 4 fix round 1 review

**Original Important finding: ADDRESSED.**

**Spec compliance: Approved within Task 4 review scope. Task quality: Approved.**

- `src/planning/train_structured_driver.py:65` now requires literal `row.physical_split == 'train'` in `_task_rows`; missing fields and physical test rows raise a descriptive `ValueError`. Both the exporter and saved-row fitting reuse this existing common boundary, closing the originally demonstrated bypass without introducing a second validation path.
- `tests/test_structured_training.py:54` now preserves the real runtime archive's explicit physical train field in the synthetic fixture.
- `tests/test_structured_training.py:107` exercises export rejection for research roles train/validation with physical test/missing provenance. The assertions match the physical-train error, so unrelated rejection cannot satisfy them.
- `tests/test_structured_training.py:121` first prepares valid rows, saves actual JSONL, then modifies their source archive and calls `fit`. The validation case includes a separate valid training recording, preventing the no-training-rows guard from masking the original hole. Together the two tests cover all eight entry-point/role/provenance combinations.
- No new Critical or Important issue found in the supplied fix diff. The production change is a two-line guard in the existing source-task validator; the remaining changes are fixture accuracy and focused regression coverage.
- Read `task-4-fix-review-package.diff` once and the appended fix evidence in `task-4-report.md`. The report records RED as two tests with eight subtest failures (`ValueError not raised`), followed by 14 passing training tests in 16.512 seconds without warnings/errors. No tests or suites were rerun during this re-review, and no source/index/branch state was changed.
- Final integrated lightweight/model-environment validation remains the controller's planned check; this scoped approval does not add any real-resource training, CUDA parity, or scientific-benefit claim.


---

## Record: final-review.md

# Structured driver whole-branch review

Reviewed 2026-09-15. Base: `sdd/structured-driver-base`; head after **Document structured driver implementation and review boundaries**. Read-only source review; this report is the only checkout file written. No branch/index changes, subagents, real-resource initialization, dataset training, model experiments, downloads, or push.

## Strengths

- The numeric path is a real shared PyTorch network, with the same collator in inference and training. Observations, independent forecast modes, local detector patches, motion/history, and the actual previous numeric plan enter the network. It does not synthesize a Q9 answer or claim language execution (`src/planning/structured_driver.py:82`, `:233`, `:347`; `src/planning/train_structured_driver.py:168`).
- Evidence acquisition stays behind the actual P/F service. P does not invoke the predictor; F predicts the full legal source window before ranking. The receiver calls `remote_units` and the archive validator reconstructs paid receipts from the actual wire before rebuilding Z (`src/tools/vehicle.py:169`, `:182`; `src/planning/structured_inputs.py:585`; `src/planning/driver_contract.py:93`). Local history is opt-in; received observations, derived forecasts, admitted dependencies and prior-plan parents remain separately represented.
- Matching uses source-local identities, explicit geometric/history gates and ambiguity handling. Remote-only/empty scenes have real paths through the builder, and the prior dependency closure prevents a removed direct token from being incorrectly declared forgotten (`src/planning/structured_inputs.py:268`, `:456`, `:597`). Equivalent full-context P-local/F forecasts merge through the existing verified equivalence mechanism; execution location is absent from learned feature encoding (`src/planning/evidence.py:268`; `src/planning/structured_inputs.py:514`).
- The common episode runner keeps exact-repeat prepared inputs fixed and only updates the prior for same-evidence refinement. The one-shot control retains actual primitive/RPC accounting and may use the shared three-attempt driver budget. Numeric failure attempts and nested model costs remain visible without adding nested durations twice (`src/planning/method_episode.py:262`, `:294`; `src/planning/method_controls.py:113`; `src/evaluation/framework.py:207`, `:260`).
- Offline training uses full-precision label records, explicit physical train provenance and recording-level train/validation separation. Ordered evidence prefixes and detached model-generated refinements use the same weights; invalid priors are masked and unsupervised batches do not update. Initial, bounded-batch and terminal checkpoints retain optimizer/RNG/permutation position and snapshot data for direct comparison (`src/planning/train_structured_driver.py:34`, `:65`, `:168`, `:384`, `:451`).

## Issues

### Critical — none found

### Important — must fix before final integration

1. **Actual runtime history audit prevents numeric training export.**
   - **Location:** `src/planning/train_structured_driver.py:84`; producing callers at `src/planning/run_framework.py:427` and `:443`, and `src/planning/structured_driver.py:342`.
   - **Trigger:** Use the literal numeric runtime loader, which loads ego history and records nonempty `task.inputs.ego_history_read_paths`. `StructuredPlanner.prepare_input` reconstructs history from the three arrays and deliberately sets `prepared.ego_history_used.read_paths=[]`.
   - **Consequence:** `_task_rows` requires these two different audit fields to be equal and rejects otherwise valid real runtime task archives with `ValueError: ego history audit paths differ from actual task`. Both `prepare_training_rows` and saved-row verification in `fit` use this boundary, so the actual collection-to-training path is blocked. The current training fixture supplies empty paths and normally no history, hiding the mismatch.
   - **Evidence:** A focused synthetic check used the actual tiny network, common episode runner, real synthetic P/F receipts, and valid history arrays. The fixture exported **3 rows** with an empty task audit list. Changing only the saved task audit list to a nonempty runtime-style pose path preserved `prepared` history paths as `[]` and produced the exact error above. The first probe lacked history arrays and stopped at a diagnostic `NoneType` access; the corrected check explicitly supplied all three arrays and reproduced the exporter failure. No real data was used.
   - **Minimal fix:** Respect the established Task 3 separation: retain read paths in `task.inputs`, retain array-derived prepared history, and remove the invalid cross-field equality requirement. Preserve history shape/value/mask validation through `_check_features`/the shared collator and preserve the actual task audit in snapshots. Add a focused export-and-fit verification using nonempty task history audit paths and the exact array-derived prepared history emitted by the runtime/planner. Do not erase real task audit paths merely to make the comparison pass.

2. **The new unconditional row lookup breaks the original v1 accepted window representation.**
   - **Location:** `src/planning/evidence.py:47` (also opt-in history array assumptions at `:50`–`:53`).
   - **Trigger:** A causal window with list-valued `track_ids`, accepted by `_check_window` and `validate_prediction`, is passed to default `new_ledger(..., include_local_history=False)`. The new comparison `w['track_ids'] == obj['track_id']` is then a scalar false comparison, and indexing the empty `flatnonzero` result fails. The lookup executes even though history was not requested.
   - **Consequence:** A formerly valid original v1 call now raises `IndexError` instead of returning its unchanged anchor/forecast ledger. This violates the explicit old-default/old-contract requirement independently of the new numeric branch.
   - **Evidence:** Focused base/current comparison with the same valid synthetic window, prediction and frozen predictor, changing only `track_ids` to a list: both validators accepted it; baseline `new_ledger` returned `toolv2x_evidence_ledger_v1` with **4 fields**; current `new_ledger` raised `IndexError: index 0 is out of bounds for axis 0 with size 0`. Baseline source was read without changing checkout state.
   - **Minimal fix:** Enumerate `prediction_objects` in its already validated source order, removing the redundant ID search, or normalize the indexing arrays once. When constructing opt-in history, also normalize `valid` and `time_seconds` before using NumPy-only methods so the common window validator's accepted representations remain supported. Add a default-v1 list-window regression against the equivalent array-window result, plus the opt-in history case.

### Minor — carried finding explicitly triaged

- **`src/planning/structured_inputs.py:98`, `:268`, `:486`, `:637`: module size/cohesion.** The approximately 900-line module includes pose loading, association, tensor projection and strict reconstruction validation. These sections serve one bounded receiver contract, keep predictor/label/torch access out, and share the same geometry and projection helpers. No concrete correctness defect or forced duplicated implementation follows from keeping them together. **Nonblocking; no split requested in this fix wave.** Extract a section only when an actual second caller or conflicting lifecycle makes a separation useful. This closes the earlier deferred Minor as an accepted maintenance tradeoff, rather than silently dropping it.

## Rulings and completion boundaries

- **Explicit seeded untrained initialization: accepted.** `initialize` writes a clearly `initialized_untrained` checkpoint, and the exact runtime loader restores that serialization format. Ordinary training starts from the declared seed; the initializer is not falsely accepted as a full training-resume checkpoint. This resolves the collection bootstrap without adding new method logic (`src/planning/train_structured_driver.py:288`, `:322`, `:425`; `src/planning/run_framework.py:397`). Only synthetic initialization tests are reported; no actual model resource was initialized here.
- **Random training run identity plus direct state comparisons: accepted.** Independent initialize/fit calls create a `training_run_id`; resume preserves it. Query semantic binding retains the full `model_version.training` object (`src/planning/train_structured_driver.py:259`, `:448`; `src/planning/query_data.py:79`). Direct model and continuation-state comparisons remain in `_load`, and resume compares data/feature snapshots. The run ID is audit identity, not a content fingerprint or authenticity proof. Independent identical-weight runs being conservatively incompatible for policy reuse is accurately documented.
- **Initial/bounded/full resume:** source inspection and the supplied synthetic tests cover initial save, bounded interruption, full continuation, model/optimizer/RNG equality and input relocation. I did not rerun those suites. The new history-export defect above must be fixed before describing the real archived-data training path as complete.
- **Common foundation versus effect claims:** source mapping and current implementation/review docs correctly distinguish standard component reimplementation from wholesale VAD/UniV2X reproduction, and contract/trainability evidence from quality, CUDA parity, closed-loop safety or ToolV2X effect evidence. No new claim requiring actual experiments was verified by this review.
- **v1 compatibility:** explicit branch dispatch generally preserves the old Q9 route and limits; finding 2 is a concrete exception. The parent still needs its planned fresh full lightweight and model-environment regressions after fixes. Previously reported task suite counts do not establish a fresh whole-branch PASS.

## Review and verification scope

Read the repository instructions, final brief/global constraints/progress, approved implementation plan and both controlling specs, current implementation/source mapping/review documentation, supplied final-review package, and task reports/reviews. Inspected all new production modules and the changed integration paths, plus actual relevant unchanged provider, receiver, control, query-binding and run-spec callers. Reviewed corresponding synthetic tests and per-task evidence; did not repeat reported suites.

Executed only two narrow new-doubt checks: (1) valid history arrays plus runtime-style nonempty audit paths at the export boundary; (2) accepted list track IDs through baseline/current default v1 ledgers. Both used the local model Python environment with `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src:tests`, and one BLAS/OpenMP thread. Synthetic episode files lived in automatically removed temporary directories. The first check runs the real tiny CPU numeric network; the second is protocol-only. Neither invokes real GoT/MTR, dataset training, or actual collection.

Exact successful reproduction commands, from the reviewed worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python - <<'PY'
import json, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_structured_training import StructuredTrainingTests
from test_structured_episode import StructuredEpisodeTests
from planning.train_structured_driver import prepare_training_rows
original=StructuredEpisodeTests.inputs
def history_inputs(self,*args,**kwargs):
    value=original(self,*args,**kwargs)
    value['features'].update(ego_pose_history=np.zeros((11,3)),ego_pose_history_valid=np.ones(11,dtype=bool),ego_pose_history_times=np.arange(-10,1)/10.)
    return value
with tempfile.TemporaryDirectory() as directory, patch.object(StructuredEpisodeTests,'inputs',history_inputs):
    root=Path(directory)
    task,label=StructuredTrainingTests().archive(root)
    print('fixture export rows:',len(prepare_training_rows(root/'tasks.jsonl',[label])))
    task['inputs']['ego_history_read_paths']=['/synthetic/ego/0000_lidar_pose.npy']
    (root/'task.json').write_text(json.dumps(task))
    print('prepared paths:',task['episode']['plans'][0]['prepared']['ego_history_used']['read_paths'])
    try:prepare_training_rows(root/'tasks.jsonl',[label])
    except ValueError as exc:print('runtime-style nonempty audit:',type(exc).__name__,str(exc))
    else:raise AssertionError('expected mismatch not reproduced')
PY
```

Output: `fixture export rows: 3`; `prepared paths: []`; `runtime-style nonempty audit: ValueError ego history audit paths differ from actual task`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/llava/bin/python - <<'PY'
import subprocess, types
from test_evidence_ledger import LedgerFixture
from test_vehicle_tools import window
from test_task_spec import provenance
from tools.task_spec import _check_window, validate_prediction
from planning.evidence import new_ledger
fixture=LedgerFixture();fixture.setUp()
w=window('ego');prediction=fixture.predictor(w)
w['track_ids']=w['track_ids'].tolist()
_check_window(w);validate_prediction(w,prediction)
print('list track_ids accepted by causal-window and prediction validators')
base=types.ModuleType('planning.evidence_review_base')
exec(subprocess.check_output(['git','show','sdd/structured-driver-base:src/planning/evidence.py'],text=True),base.__dict__)
for name,call in [('base v1',base.new_ledger),('current v1',new_ledger)]:
    try:
        result=call(w,prediction,predictor=fixture.predictor,local_provenance=provenance())
        print(name, result['version'], len(result['local_fields']))
    except Exception as exc:print(name,type(exc).__name__,str(exc))
PY
```

Output: both validators accept; `base v1 toolv2x_evidence_ledger_v1 4`; `current v1 IndexError index 0 is out of bounds for axis 0 with size 0`.

## Assessment

**Ready to merge? With fixes.**

The shared numeric architecture, causal receipt path, controls and resumable training design are coherent, but two concrete integration regressions remain: real history audit blocks export, and valid default-v1 list inputs now fail. Fix those two boundaries, obtain scoped re-review, and run the planned final integrated checks before marking the branch complete.


---

## Record: final-fix-report.md

# Structured driver consolidated final fix report

Date: 2026-09-15

Status: COMPLETE for the two Important findings in `final-review.md`. This wave used only synthetic CPU fixtures. It did not run real training, initialize real resources, execute real GoT/MTR, download data, run the repository-wide suites, push, or change the accepted `structured_inputs.py` module-size ruling.

## Changes

### Runtime history audit and numeric training

`src/planning/train_structured_driver.py::_task_rows` no longer compares `task.inputs.ego_history_read_paths` with `plans[0].prepared.ego_history_used.read_paths`. They serve different contracts: the task field preserves the physical files attempted by the runtime loader, while the prepared field is reconstructed solely from the three numeric feature arrays and intentionally has `read_paths=[]` so paths do not enter model input or semantic identity.

No history validation was removed. Export still calls `_check_features`, and saved-row validation in `fit` still reaches the same check. The shared collator verifies that `ego_pose_history`, `ego_pose_history_valid`, and `ego_pose_history_times` are all present together, have the required shapes/dtypes/finite values, and equal the prepared states, mask, and times. Input snapshots continue to copy the full source task, including its actual nonempty history audit paths.

`tests/test_structured_training.py` now has a runtime-style fixture with eleven nonempty attempted pose paths and all three real numeric history arrays. Its focused test verifies the intended empty prepared-path field, exact array/mask/value agreement, successful three-row export, a real tiny-network one-step optimizer fit, and preservation of the nonempty task audit list in `fit/inputs/task_000000.json`.

### Legacy list-window compatibility

`src/planning/evidence.py::new_ledger` now enumerates `prediction_objects` in the source order already checked by `validate_prediction`, instead of redundantly finding each object ID with a NumPy-only comparison. This preserves the original v1 object/field order and removes the invalid assumption that `track_ids` is an array.

The opt-in history branch converts each selected `valid` row and `time_seconds` to NumPy only where `.tolist()` and `.sum()` are needed. List-valued `track_ids`, `states`, `valid`, `scores`, and `time_seconds`, all accepted by the causal-window validator, therefore work in both default v1 and history-enabled v2. Paths remain absent from the ledger and model features.

`tests/test_evidence_ledger.py` now checks that a fully list-valued default window returns exactly the same v1 ledger as its array-valued equivalent. A separate test checks full list-valued opt-in history content, masks, and times.

## TDD evidence

Focused RED before production changes:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_evidence_ledger.EvidenceLedgerTests.test_default_v1_list_window_matches_array_window \
  test_evidence_ledger.EvidenceLedgerTests.test_opt_in_history_accepts_full_list_window \
  test_structured_training.StructuredTrainingTests.test_runtime_history_audit_exports_and_fits_from_numeric_arrays
```

Observed result: 3 tests, 3 errors. Both ledger tests failed at `np.flatnonzero(w['track_ids'] == obj['track_id'])[0]` with `IndexError`. The training test failed at `_task_rows` with `ValueError: ego history audit paths differ from actual task`.

The same command after the minimal production changes:

```text
...
----------------------------------------------------------------------
Ran 3 tests in 1.880s

OK
```

## Scoped verification

Directly covering suites, using the required model environment:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/conda-envs/llava/bin/python -m unittest \
  test_evidence_ledger test_structured_inputs test_structured_training
```

Result:

```text
..........................................................
----------------------------------------------------------------------
Ran 58 tests in 19.299s

OK
```

Syntax compilation:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  /root/autodl-tmp/conda-envs/llava/bin/python -m py_compile \
  src/planning/evidence.py src/planning/train_structured_driver.py \
  tests/test_evidence_ledger.py tests/test_structured_training.py
```

Result: exit 0, no output.

Repository whitespace check:

```text
git diff --check
```

Result: exit 0, no output.

The parent agent retains responsibility for the single planned fresh repository-wide lightweight and model-environment suite run after scoped review.


---

## Record: final-fix-review.md

# Final consolidated fix review

Scope: only the two original Important findings and new breakage in the supplied fix diff. Base: `sdd/structured-final-fix-base`; head after **Fix numeric history export and legacy window compatibility**. Read `final-fix-report.md`, the complete `final-fix-review-package.diff`, and the relevant current source lines. No suites, experiments, model execution, or nested agents were run. This report is the only file written by this re-review.

## Original finding verdicts

1. **Runtime history audit blocks numeric training export — ADDRESSED.**
   - `src/planning/train_structured_driver.py:83` now proceeds from the feature-file check to stage-row export without equating the task's real read audit with the prepared history's intentionally empty path list.
   - Actual numeric history validation remains: export calls `_check_features` at `:129`; saved-row verification calls it at `:374`; `_check_features` invokes the same structured collator at `:160`. The deleted code compared only the two path lists and did not validate history values, masks or timestamps. The full task is still copied into the input snapshot at `:395`, retaining its audit data.
   - `tests/test_structured_training.py:45` adds the three numeric history arrays and eleven nonempty attempted pose paths to the actual episode fixture. The regression at `:115` checks the empty prepared paths, matching numeric arrays/mask/times, successful export and a real tiny-network one-step fit, and preservation of the nonempty task audit in the snapshot. It reaches both originally affected entry boundaries without altering the runtime/planner contract.

2. **Default v1 list-window compatibility regression — ADDRESSED.**
   - `src/planning/evidence.py:46` enumerates the already aligned `prediction_objects` output instead of comparing list-valued IDs with a scalar. This preserves existing object/field order and removes the failing unconditional NumPy lookup from the default v1 route.
   - The opt-in history branch normalizes each valid row and the time axis before NumPy-specific operations at `:49` and `:52`; states/scores were already normalized. This also resolves the fully list-valued window case without changing array semantics or the old default ledger version.
   - `tests/test_evidence_ledger.py:49` compares the complete default v1 ledger for fully list-valued versus array-valued inputs. The separate `:60` test checks the opt-in v2 history values, masks and times. Both tests exercise the actual common ledger API with aligned synthetic predictions.

## New breakage in the fix diff

No new Critical or Important issue found. The production changes are confined to removal of the invalid audit-path equality and a source-order iteration/normalization correction. No new driver, training objective, service behavior, or checkpoint semantics were introduced.

The previously deferred `structured_inputs.py` module-size Minor remains explicitly accepted and nonblocking; this wave does not reopen or alter that ruling.

## Verification evidence and limits

The supplied fix report records the three new regressions failing before the production changes with the exact original errors: two list-ID `IndexError` failures and one history-audit `ValueError`. It records the same three tests passing afterward in 1.880 seconds, followed by **58 passing covering tests in 19.299 seconds** across evidence ledger, structured inputs and structured training. Syntax compilation and whitespace checks are reported clean. These results were reviewed, not rerun, and their execution chronology is the implementer's supplied evidence.

## Assessment

**Both original Important findings: ADDRESSED. Scoped fix review: Approved.**

No unresolved blocker remains from the whole-branch findings. The parent's planned fresh full lightweight and model-environment suites remain required before declaring final integrated completion. This approval establishes the scoped code corrections, not real-data training, model quality, CUDA parity, closed-loop results, or method effects.


---

## Record: portability-review.md

# Required-validation portability repair review

**Verdict: APPROVED.**

- `tests/test_structured_training.py:244` derives the repository `src` directory from the test file's resolved location. Both fresh Python subprocesses use this explicit working directory (`tests/test_structured_training.py:245`, `tests/test_structured_training.py:247`), making local package resolution independent of the caller's working directory and the presence of inherited `PYTHONPATH`.
- The clean-import assertion remains and is strengthened to prohibit both torch and transformers imports (`tests/test_structured_training.py:245`). The separate CLI `--help` invocation remains, and both zero-exit assertions still include child stderr on failure (`tests/test_structured_training.py:246`, `tests/test_structured_training.py:248`). Seeded initialization, reload identity, parameter equality, and no-overwrite assertions remain unchanged in the supplied context (`tests/test_structured_training.py:234`).
- The supplied diff changes only test subprocess setup/assertions; no production behavior changes. No Critical or Important issue found in this repair.
- Read the appended portability evidence in `task-4-report.md`: with `PYTHONPATH` explicitly unset, the target initially failed with the expected child import error, then passed after this fix; the complete 15-test training module subsequently passed in 17.326 seconds. These are implementer-reported runs, not reviewer reruns.
- Scope retained: reviewed only `portability-review-package.diff` and the appended report. No tests rerun, no old findings reopened, and no source/index changes made. Fresh final full model validation remains with the controller.
