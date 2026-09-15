## Task 1: Complete request-decision cost accounting

**Files:** `src/planning/method_episode.py`, `src/evaluation/framework.py`; narrowly update archive consumers only if required. Tests in existing method episode/evaluation/controls files or `tests/test_decision_accounting.py`; add new torch-free test module to `scripts/check_review.py` if used.

**Interfaces:** `run_task_episode` and `_method_cost(task)` stay public/common paths. New episodes carry an explicit compute-accounting version independent of control kind. Every actually started decision/preflight interval has an auditable stage identity and either a complete duration or unknown/incomplete accounting. Keep `control_seconds`, `known_cost`, `cost_complete`, `total_compute_seconds` and `end_to_end_seconds` meanings; no duplicate nested timing.

- [ ] RED: with a patched monotonic clock, policy advances 5 seconds then STOP. None and explicit feedback both report control_seconds >=5; previous None path yields 0. Assert evaluation rejects missing duration as unknown, retains known partial costs, and never labels it complete.

```python
clock = [0.]
def policy(state):
    clock[0] += 5.
    return dict(tool='STOP', mode=None, reason='timed_fixture')
# Existing real executor + fixture driver; no sleep or actual model execution.
assert evaluated['control_seconds'] >= 5.
```

- [ ] Cover no-call budget STOP, policy exceptions before decision append, dispatch failures, progress interruption between decision and completed accounting, valid prefix reuse, exact-repeat with only final actual decision, and old archives without timing. Do not charge stages which never attempted a decision; distinguish missing from actual zero. Existing explicit timing records remain usable. Persistence callbacks stay outside timed computation.
- [ ] Implement one shared accounting contract; keep role/control configuration separate. Document which management/loading/network/I/O work remains excluded. Bind or validate the new version in affected consumers if they otherwise misinterpret its completeness.
- [ ] Run covering tests and relevant existing method control/evaluation/query/bundle tests. Report commands and actual results. Avoid full suite until final integration.
- [ ] Commit only owned files, subject `Account for every actual request decision`; write task report and get independent spec/quality review.
