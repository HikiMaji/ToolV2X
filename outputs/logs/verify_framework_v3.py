"""Recompute saved connection invariants without loading models or GT."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path('/root/autodl-tmp/ToolV2X')
sys.path.insert(0, str(root / 'src'))
from planning.inputs import make_prompt, parse_q8, parse_q9

out = root / 'outputs/framework_connection_v3'
def read(path):
    return json.loads(path.read_text())

meta = read(out / 'connection.json')
execution = read(out / 'execution.json')
assert execution['status'] == 'completed' and execution['all_actions_parsed'] is True
assert not any(meta[k] for k in ('training', 'quality_evaluation', 'closed_loop', 'full_framework_complete'))
assert meta['g'] == 5526 and meta['research_split'] == 'validation'
projection = read(out / 'projection.json')
assert projection['input_frames'] == [5526, 5525]
assert all('/co_llm/ego/' in p for p in projection['feature_read_paths'])
tokens = np.load(out / 'ego_point_tokens.npy', allow_pickle=False)
assert tokens.shape == (1, 540, 4096) and np.isfinite(tokens).all()
assert meta['actions']['Ego']['remote_archive_reads'] == 0

summary = dict(status='PASS', scope='saved real-frame connection invariants, no quality assessment',
               scene=meta['scene'], global_frame=meta['g'], actions={})
for action, row in meta['actions'].items():
    directory = out / action
    plan = read(directory / 'plan.json')
    full = read(directory / 'evidence_full.json')
    selected = read(directory / 'evidence_used.json')
    assert selected == plan['evidence_used']
    assert plan['status'] == 'parsed' and plan['q9_executed']
    assert parse_q8(plan['q8_raw']) == plan['action']
    np.testing.assert_array_equal(parse_q9(plan['q9_raw']), plan['waypoints'])
    assert plan['q9_prompt'] == make_prompt('Q9', meta['ego_motion'], selected, plan['q8_raw'])
    assert len(full['objects']) == row['evidence_objects']
    full_ids = {(o['source'], o['track_id']) for o in full['objects']}
    assert all((o['source'], o['track_id']) in full_ids for o in selected['objects'])
    ego_ids = {o['track_id'] for o in selected['objects'] if o['source'] == 'ego'}
    peer_ids = {o['track_id'] for o in selected['objects'] if o['source'] != 'ego'}
    assert all(r['ego_id'] in ego_ids and r['peer_id'] in peer_ids for r in selected.get('relations', []))
    costs = read(directory / 'tool_costs.json')
    measured = sum((directory / (c['tool'] + '_response.json')).stat().st_size for c in costs)
    assert measured == row['response_bytes'] == sum(c['response_bytes'] for c in costs)
    for task, reserve in (('q8', 128), ('q9', 256)):
        assert plan[task + '_cost']['input_tokens'] + reserve <= 4096
    summary['actions'][action] = dict(
        status=plan['status'], q8=plan['action'], final_waypoint=plan['waypoints'][-1],
        full_records=len(full['objects']), prompt_records=len(selected['objects']),
        remote_prompt_records=sum(o['source'] != 'ego' for o in selected['objects']),
        request_bytes=row['request_bytes'], response_bytes=measured,
        q8_seconds=plan['q8_cost']['seconds'], q9_seconds=plan['q9_cost']['seconds'],
        q8_input_tokens=plan['q8_cost']['input_tokens'], q9_input_tokens=plan['q9_cost']['input_tokens'])

objects = read(out / 'PF/evidence_full.json')['objects']
left = {o['track_id']: o for o in objects if o['source'].endswith(':P_local')}
right = {o['track_id']: o for o in objects if o['source'].endswith(':F')}
assert left.keys() == right.keys() and len(left) == 30
for key in left:
    for field in ('forecast', 'forecast_scores', 'box'):
        np.testing.assert_array_equal(left[key][field], right[key][field])
summary['same_information_targets'] = len(left)
summary['same_information_max_difference'] = 0.

for folder in ('src', 'tests', 'vendor/cmp_mtr'):
    for saved in (out / 'code_snapshot' / folder).rglob('*.py'):
        current = root / saved.relative_to(out / 'code_snapshot')
        assert saved.read_bytes() == current.read_bytes(), str(current)
summary['saved_python_matches_current'] = True
(out / 'independent_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
