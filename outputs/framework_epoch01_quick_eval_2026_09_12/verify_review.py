"""Recompute the published 32-frame trajectory results with Python stdlib only."""
import ast
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (root / 'review_predictions.jsonl').read_text().splitlines()]
summary = json.loads((root / 'evaluation/summary.json').read_text())
assert len(rows) == 160
assert len({(r['sample_id'], r['policy']) for r in rows}) == 160
assert len({r['sample_id'] for r in rows}) == 32
for row in rows:
    assert row['valid'] and row['role'] == 'validation'
    assert row['truth_valid'] == [True] * 6
    assert row['times_seconds'] == [.5, 1., 1.5, 2., 2.5, 3.]
    raw = row['raw_answer']
    points = ast.literal_eval(raw[raw.index('['):raw.rindex(']') + 1])
    assert [list(p) for p in points] == row['waypoints']
    assert len(points) == len(row['truth']) == 6
    distances = []
    for predicted, truth in zip(points, row['truth']):
        assert len(predicted) == len(truth) == 2
        assert all(math.isfinite(v) for v in (*predicted, *truth))
        distances.append(math.hypot(predicted[0] - truth[0], predicted[1] - truth[1]))
    assert abs(sum(distances) / 6 - row['ADE3']) < 1e-10
    assert abs(distances[-1] - row['FDE3']) < 1e-10
assert set(summary['policies']) == {'Ego', 'P', 'F', 'PF', 'rule'}
for policy, report in summary['policies'].items():
    selected = [r for r in rows if r['policy'] == policy]
    assert len(selected) == report['attempts'] == 32 and report['failures'] == 0
    for metric in ('ADE3', 'FDE3'):
        assert abs(sum(r[metric] for r in selected) / 32 - report['mean_' + metric]) < 1e-10
print('PASS: 160 raw answers, 32 matched frames, all ADE/FDE rows and policy means.')
