# Structured Cooperative Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. No nested subagents. User authorized download then continuation on 2026-09-14.

**Goal:** Deliver a complete trainable numeric cooperative driver using actual causal P/F services, with original GoT/v1 behavior preserved; do not execute real training or new method experiments.

**Architecture:** Versioned structured receiver builds entities from local/paid/derived ledger fields, separately encodes observations and multimodal forecasts with local detector scene patches, and uses one shared numeric trajectory network. The existing interaction/control/branch/evaluation flow dispatches by driver kind, preserving P/F protocol and archived GoT meaning. Training shares the same input builder and supports actual prior-plan conditioning plus restorable checkpoints.

**Tech Stack:** Existing Python, NumPy/SciPy, PyTorch; no new model dependency. Model tests use `/root/autodl-tmp/conda-envs/llava/bin/python` on CPU. Source references downloaded to `/root/autodl-tmp/UniV2X` and `/root/autodl-tmp/VAD` (both Apache 2.0); no wholesale MMDetection import.

**Spec:** `docs/structured_cooperative_planner_design_2026_09_14.md`, constrained by `docs/review_9_14_1_response.md` section 3.

## Global Constraints

- Physical time t inputs only, two cars, P/F and STOP, at most two remote calls and three shared-driver attempts. P never runs MTR; F runs full legal peer context before ranking/cropping; task plans do not condition the MTR model. Legitimate prediction cache reuse remains.
- Existing `query/decode_response`, task wire, receipts, ExecutionSpec and old golden behavior remain unchanged. Add a new receiver/driver/interaction version instead of relabeling Q9 or fabricating language generation.
- Local raw inputs fixed; remote evidence only after authentic response. Acquired/derived/admitted and indirect prior-plan parent refs remain distinguishable. Do not trust requester manifests or use GT for association.
- Explicit P-state (`observations_only`) default for new numeric route; P-local selectable, separately billed. Identical full-source P-local/F forecasts must produce identical numeric model input when other admitted information matches.
- Retain remote-only and ambiguous targets, avoid blind origin deletion, preserve mode sets and contexts. No score-as-covariance or new road/route/coverage claims.
- No new prediction backbone, RSU/I, RL, peer learned ranking, hidden peer features, joint predictor training, real GoT/MTR generation, or dataset training in this batch. CPU synthetic forward/backward and tiny optimizer/resume unit tests are verification only; no research effect claims.
- Keep all model weights/output evidence external, preserve existing artifacts, no automatic GitHub push. Semantic identity is version/config plus direct parameter comparison; never write hashes or SHA256.
- Latest user authorizes reference downloads <100 MB; old GitHub restriction does not block these explicitly authorized small sources.
- Prior reviewed main has 302 lightweight/358 model-env tests. Re-run required suites after integration with explicit external resource paths, not after every small edit.

## Task 1: Structured evidence, causal ego history, entity association

**Files:** Create `src/planning/structured_inputs.py`, optionally `src/planning/entities.py`; modify `src/planning/evidence.py` only for opt-in local history; create `tests/test_structured_inputs.py`. Do not edit other tasks' files.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class StructuredDriverSpec:
    # to_dict / from_dict strictly validate full versioned configuration.
    version: str = 'toolv2x_structured_driver_v1'
    hidden_dim: int = 256
    attention_heads: int = 4
    interaction_layers: int = 2
    max_entities: int = 64
    max_forecast_sets_per_entity: int = 4

def load_ego_history(root, split, g):
    # {states: [11,3] xy/yaw in ego(t), valid: [11], times: [-1..0], read_paths: [...]}

def build_structured_plan_input(motion, ledger, spec, *, ego_history=None,
                                previous_plan=None, previous_parent_refs=()):
    # JSON-native prepared record, no filesystem, predictor or torch access.

def validate_structured_prepared(prepared):
    # Strict schema, finite values, masks, provenance/shape/capacity consistency.

def new_ledger(..., p_processing='local_mtr', include_local_history=False):
    # Preserve exact default v1 output. Opt-in ledger_v2 adds real local history.
```

Prepared record fields: `input_layout='structured_evidence_v1'`, `decoding='numeric'`, `driver_spec`, `scene`, `g`, `ego_motion`, `entities`, `tensor_inputs`, `admission_report`, `ego_history_used`, `previous_plan` (None or actual 6x2), `previous_parent_refs`. No q9 prompt or execution flag. `tensor_inputs` uses padding to E=max_entities, S=2 source observation slots, C=max_forecast_sets_per_entity:

```text
observations [E,2,11,10]: x,y,z,l,w,h,sin(yaw),cos(yaw),score,time
observation_mask [E,2,11] bool; observation_sources [E,2] local=0,remote=1
forecasts [E,C,6,6,5]: x,y,time,mode_score,model_used
forecast_mask [E,C,6,6] bool; forecast_context [E,C,2]: source_is_local,full_context
entity_mask [E] bool
```

Use scale parameters in full spec (position/size/time) for these arrays; no data-derived normalization in this batch. Encoding must not embed track IDs, receipt strings, request kinds or remote-versus-local execution origin; source observation role and actual context scope are allowed. mode order must not create artificial semantic identity.

Use `remote_units` to validate receipts, derived parents and equivalent forecasts; do not bypass its verification. New local history is exactly window.states/valid/scores/time, with `_validate_task_value`-compatible proxy status. Association follows UniV2X gated one-to-one matching, extended with common valid history, dimensions and heading; SciPy Hungarian, deterministic input order and documented ties; ambiguous competing candidates remain separate. Gate thresholds, ambiguity margin and ego geometry/history thresholds live in StructuredDriverSpec. Fix representative anchor by deterministic source quality, not variance averaging. Entity IDs follow canonical alias; source slots are local vs peer. Preserve remote-only objects and ego-role evidence in reports; exclude only a sufficiently corroborated ego from obstacle tensors, otherwise keep unresolved. Ego history optional: without it never declare near-origin object certainly ego. `load_ego_history` reads t-10..t within recording boundary only and masks missing/invalid history.

Admission report retains existing keys from `context.build_task_plan_input`: acquired_field_refs, derived_field_refs, admitted_field_refs, dropped_field_refs, dropped, field_groups, local_field_groups, observation_receipt_ids. This enables value-feature reuse. Structured locations belong in groups; association/filter influences count as admitted dependencies even when no standalone token; prior parent dependencies propagate. Audit-only details may live inside entities/groups, not as flattened learned features. No unknown parents accepted.

- [x] Write tests first, assert missing module/opt-in API via an explicit unittest assertion; run red.
- [x] Hand-derived fixtures: local track 7 at x=10 and remote track 7 at x=30 remain two; remote x=10.2 matched to local x=10 within configured gates; equal competing matches remain ambiguous; a lone near-origin anchor remains unresolved without ego history. Permuting source rows cannot change tensor values.
- [x] Test paid receipt validation and same-information full-P/F numeric forecast equality; history-only P works without calling predictor; all absent values masked; source-local ID never treated as global.
- [x] Implement spec, optional history and entity/tensor builder; use explicit ordinary geometry and source/context-preserving arrays as above.
- [x] Run targeted lightweight tests and old `test_evidence_ledger.py`, `test_task_spec.py`, `test_vehicle_tools.py`; report full commands/results. No model execution.
- [x] Commit only task files with subject `Add causal structured evidence and entity receiver`; return report and discovered API facts. Never print commit identifiers.

## Task 2: Shared numeric network and authentic output

**Files:** Create `src/planning/structured_driver.py`, `tests/test_structured_driver.py`; do not modify Task 1 files without reporting a concrete interface defect. Read Task 1 report for finalized spec fields.

**Interfaces consumed:** Task 1 builder/spec/prepared schema. Features retain `regression_map`, `classification_map`, `active_agent_mask`, optional `ego_pose_history` / `ego_pose_history_valid` / `ego_pose_history_times`; original shapes from `inputs.load_ego_features`.

**Interfaces produced:**

```python
class StructuredPlannerNetwork(torch.nn.Module):
    def __init__(self, spec): ...
    def forward(self, batch): ...  # returns [B,6,2] in meters

def collate_structured_inputs(features_list, prepared_list, device='cpu'):
    # tensors consumed by network; validates same spec and exact feature layout.

class StructuredPlanner:
    def __init__(self, spec, *, model=None, device='cpu', model_version=None): ...
    def prepare_input(self, features, motion, ledger, receiver_spec,
                      previous_plan=None, previous_parent_refs=()): ...
    def plan_prepared(self, features, prepared): ...
    # provenance: stable driver_kind='structured', full spec, numeric decoding,
    # actual model version/training metadata; context_limit is not a language cap.

def validate_numeric_output(output, prepared, execution_spec):
    # returns actual validated numeric waypoints, never parses synthetic Q9 text.
```

Output is JSON-native: `output_version='toolv2x_numeric_plan_v1'`, `driver_kind='structured'`, `status='valid'`, `waypoints`, `prepared_input` (exact prepared), `driver_cost` (seconds, numeric_token_count, output_points, model_executed), `parent_refs` including current admission and prior closure. No fake q9_raw, q9_executed, language cost or generated-language token counts.

Network: original shallow GoT 5x4 detector-map patch layout -> Linear(320,d) + frame/grid position codes; use both current and legal previous scene grids, not road maps. Each source history uses masked PointNet-style MLP/global max/MLP with LayerNorm (derived from local MTR polyline structure, no frozen parameter changes). Fuse observation source tokens per entity with shared attention. Each forecast mode is encoded separately using time/score/model_used + source/context; no mode-index embeddings or coordinate averaging. Six learned waypoint queries attend to state, forecast-mode and scene tokens using ordinary PyTorch attention, then MLP outputs coordinates. Prior actual plan and its validity are numeric conditions for a residual refinement; initialization has false prior mask. All-masked inputs have safe null tokens. Dropout off for deterministic deployment. Network outputs one trajectory and no new traffic predictor.

- [x] Red tests for real model output, missing branch gradient and incompatible input/version failures.
- [x] Hand-check patch layout and masked invalid values; assert deterministic inference; mode/source row permutation after canonical builder is invariant; moving valid P/F data can reach gradients, masked fields cannot. Use tiny spec for CPU tests, not new test-only production methods.
- [x] Implement network, shared collator/wrapper/output validation; no 7B/tokenizer/CLIP imports. Empty objects and one valid history point must train-forward without BatchNorm failure.
- [x] Tests use untrained network on synthetic inputs only; document not effect evidence. No real cached data model generation and no dataset optimizer run.
- [x] Commit only task files with subject `Add shared numeric cooperative planner`; report exact new signatures and tests.

## Task 3: Existing method loop, controls, runtime and offline evaluation

**Files:** Modify `method_episode.py`, `method_controls.py`, `method_run_spec.py`, `run_framework.py`, `query_data.py`, `bundle_data.py`, `query_value.py` only where real callers require numeric dispatch; `src/evaluation/framework.py`, `src/evaluation/planning.py`. Add a small `driver_contract.py` only if shared output/input access removes real duplication. Extend existing relevant tests and add `tests/test_structured_episode.py`.

**Interfaces:** `run_task_episode` stays the shared public executor. Add limits version `toolv2x_interaction_v2`: same top-level fields as v1, receiver_spec is the full StructuredDriverSpec dictionary. Validate driver kind against limits before reading remote content. New ledger opts into local history, P processing is explicit in new spec. Wrapper `prepare_input` gets actual prior coordinates and prior output parent refs. Keep literal v1 output/behavior unchanged.

Network is invoked through same `plan_prepared` API, but numeric output/provenance/cost/shape are checked by actual output version; no GoT flag spoofing. Prefix capture/replay, decision_state, archived Z validation and query semantic binding dispatch to correct schema. Policy `raw` can be None for numeric plans; exact existing admitted-ref schema supports state_features. Bind full structured spec/model/training identity so old GoT query checkpoints cannot load as new policies.

All numeric calls use the prior slot, independent of control. Exact repeat freezes the whole prepared input. Same-evidence refinement copies E/Z tensors/entities/admission and only replaces prior plan and its dependency closure; no rebuild allowing different admission. One-shot may use remaining driver calls for same-evidence refinement, preserving one RPC/two primitives and actual budget accounting. Existing fixed Ego/P/F/PF/rule, frozen feedback/current_old/union and full-P controls remain.

Runtime `_load_interaction_runtime` dispatches from explicit new limits to StructuredPlanner checkpoint loader (Task 4 adds load function); it still loads original frozen CMP only when the CLI is explicitly executed. `load_ego_history` adds only legal past-local pose fields to numeric features and read audit. Never execute real runtime now.

Evaluation branches on numeric output; validates raw actual 6x2, finite motion bounds, output-to-prepared identity, parent refs and stage cost. Numeric calls have zero language token generation, separate numeric-token/point counts, actual driver_seconds; sum stages without nested-cost double counting. Add `got_prefix_L2_1s/2s/3s/avg` from cumulative point errors without silently changing existing endpoint L2_1/L2_2.

- [x] Red tests for numeric loop with actual synthetic network and real P/F service fixtures: initial/response/prior/second-query flow; numeric plan not misreported as Q9; no hidden peer reads; failures billed.
- [x] Test exact-repeat vs meaningful same-evidence prior update, common adapter, full cost, real prefix controls, one-shot bundle, query/branch archive acceptance and old-checkpoint rejection.
- [x] Implement dispatch through common code, avoid copying entire episode runner. Preserve all original limits/GoT checks on original branch.
- [x] Run relevant existing method/controls/query/bundle/value/evaluation tests; no scientific branches or outputs on actual data. Use synthetic network for numeric integration, fake predictor only at MTR boundary.
- [x] Commit with subject `Integrate numeric driver with tool episodes and controls`; report tested interfaces, known deferred formal-test role addition (do not relabel validation).

## Task 4: Trainable dataset path, masked supervision and resume

**Files:** Create `src/planning/train_structured_driver.py`, `tests/test_structured_training.py`; add `load_structured_planner` in `structured_driver.py` if its serialization boundary belongs there. Modify structured runtime call site only if needed to bind exact loader. Do not rewrite old train_driver.py.

**Interfaces:**

```python
def prepare_training_rows(tasks, labels):
    # validate numeric episode/prepared/feature references and offline label role,
    # sample, scene, g; JSON-native rows with separate input vs supervision.
def masked_trajectory_loss(predicted, target, valid): ...
def fit(rows, out, config): ...
def load_structured_planner(checkpoint, *, device='cpu'): ...
```

CLI supports preparation from saved numeric method task records + independent offline label JSONL, and explicit training from prepared rows with a versioned config. Add an explicit `initialize` command to create a seeded, clearly untrained checkpoint from the full driver spec, loadable by the same runtime: initial causal task collection must not require an already trained checkpoint. Do not execute it on actual resources in this batch. Existing labels are full-precision six points from ego_label, not rounded Q9; scene may be verified through the exact unique sample_id/task join when the existing label schema omits a separate scene field. No automatic extraction of future-derived navigation/prior. Source task references and masks are validated; same source frame with multiple evidence stages retains one reference trajectory. Hold out whole recordings, not adjacent rows, using explicit train/validation row role. Refuse disagreement between task research role and label/row role.

Training uses same collator/network; explicit evidence-stage rows and repeat/refinement steps share weights. Previous plans come only from causal model forward or archived actual same-input plan with provenance, never target coordinates; use model-generated detached prior for unrolled refinement with each step supervised. Invalid model prior is masked, not copied from GT. Samples with no valid labels do not fabricate loss. P/F mix/coverage is reported, not implicitly claimed complete.

Config versions save architecture, seed, optimizer settings, batch size, epoch/step, data role binding, refinement depth and save interval. Full checkpoints include model/optimizer/random state and row/permutation position; save initial state and bounded step intervals so interruption loses no entire long epoch. Atomic writes, no destructive deletion of external artifacts, refuse output overwrite. Resume verifies exact model/data/optimizer binding; path is not semantic identity. Standalone planner checkpoint load binds actual trained state/version and produces numeric output with provenance.

- [x] Write red tests: literal masked SmoothL1 value and gradient; train/validation leakage rejected; targets never enter online collator; missing-point loss handled; interrupted resume equals uninterrupted small synthetic updates.
- [x] Implement exporter, loss, explicit fit and loader with clean CLI help. Tests may perform tiny CPU synthetic optimizer steps in temporary directories; no real dataset training/checkpoints in outputs.
- [x] Test reload output equals saved model; exact resume RNG/optimizer; source record tampering/old GoT row rejection; train/infer same representation. Runtime import and `--help` work without loading models or reading labels.
- [x] Run task tests, then required full lightweight and model-environment regressions once after branch review fixes. Record which full tests are CPU/frozen model validation versus generation; do not run old generate scripts.
- [x] Commit with subject `Add structured driver supervision and resumable training`.

## Final delivery

- [x] Independent whole-branch source review, fix concrete correctness issues, preserve v1 golden behavior. No evidence claim beyond contracts and trainability.
- [x] Update README/STATUS succinctly with actual APIs/tests/no-training boundary; persist source module map, downloads manifest, task reviews and final validation. Update AGENTS small-download exception and numeric driver branch guidance; do not weaken causal review rules.
- [x] Bring reviewed code into the user's main workspace preserving dirty docs; no push. Report files/APIs/results and material deviations (standard components rather than wholesale VAD/UniV2X reproduction).
