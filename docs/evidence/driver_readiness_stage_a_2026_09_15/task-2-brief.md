## Task 2: Derived structured evidence-use audit

**Files:** Create `src/evaluation/structured.py`, `tests/test_structured_admission.py`; minimal integration into `src/evaluation/framework.py` and `scripts/check_review.py`. Do not modify provider, receiver capacity, field_groups or ledger semantics.

**Interfaces:** `audit_structured_episode(episode) -> list[dict]` returns JSON-native per-plan audit from a validated numeric episode. Integrate into ordinary method evaluation row output under an explicitly named structured audit field; old GoT rows remain compatible. No model, tokenizer, labels or filesystem reads in this helper.

- [ ] RED using existing real synthetic P/F service fixtures: local x=10 /peer x=30, max_entities=1 gives acquired remote anchor+history, both dependency-closed, zero direct remote primary fields. Capacity2 gives a direct remote history at actual observation slot. Required module assertion fails before implementation.

```python
audit = audit_structured_episode(ep)
assert audit[stage]['direct_remote_primary_refs'] == []  # dropped by capacity
# Exact published output keys additionally documented in task report.
```

- [ ] Report new versus previously acquired remote refs at each ledger stage; remote/local direct primary refs from `use='tensor'`, valid mask and tensor_locations; observations/forecast sets/modes/timepoints coverage; capacities/ego filter; association/selection dependency closure; actual prior dependencies; indirect-only refs subtract direct refs. Preserve full field identities, allow overlapping causal roles, do not fabricate counterfactual usefulness.
- [ ] Include actual tensor and prior changes versus preceding plan and actual output change/validity when available, without running a model. First-stage changes are undefined, not a comparison with fabricated zeros. Same-evidence repeat/refinement and reference-only receipts produce correct differences.
- [ ] Cover remote-only/ambiguous, F multimodal sources/contexts, capacity and ego filtering, empty/masked inputs, full-P/local-derived equivalent F, prior-only dependencies, corrupt tensor locations/receipts rejected and old GoT dispatch unchanged. Reuse upstream validation rather than second receipt registry.
- [ ] Run new torch-free tests plus structured input/method evaluation regression; commit `Audit direct and indirect structured evidence use`, report exact API and review.
