"""Independent artifact checks; no fitting, test-set input or test-set GT reads."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

root = Path('/root/autodl-tmp/ToolV2X')
sys.path.insert(0, str(root / 'src'))
from planning.inputs import pack_evidence, unpack_evidence, parse_q9
from planning.v2vgot import rounded

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

baseline = root / 'outputs/framework_connection_v3'
expanded = root / 'outputs/framework_compact_v1'
comparison = root / 'outputs/evidence_comparison_v1'
data = root / 'outputs/adaptation_data_v1'
forward = root / 'outputs/adaptation_forward_v1'
report = dict(status='PASS', scope='encoding, provenance, separated labels and native forward; no quality claim', actions={}, data={})
for action in ('Ego', 'P', 'F', 'PF'):
    full = read(baseline / action / 'evidence_full.json')
    assert full == read(expanded / action / 'evidence_full.json')
    selected = read(baseline / action / 'evidence_used.json')
    same = read(comparison / (action + '_same_set_compact.json'))
    assert same['evidence_used'] == selected
    assert same['status'] == 'parsed'
    for evidence in (full, rounded(full), selected):
        assert unpack_evidence(pack_evidence(evidence)) == evidence
    for tool in action if action != 'Ego' else ():
        assert (baseline / action / (tool + '_response.json')).read_bytes() == (expanded / action / (tool + '_response.json')).read_bytes()
    raw_modes, rounded_modes = Counter(), Counter()
    for obj, quantized in zip(full['objects'], rounded(full)['objects']):
        raw_modes[len({json.dumps(path) for path in obj['forecast']})] += 1
        rounded_modes[len({json.dumps(path) for path in quantized['forecast']})] += 1
    report['actions'][action] = dict(raw_distinct_mode_counts=dict(raw_modes), rounded_distinct_mode_counts=dict(rounded_modes))

all_labels = {}
for role in ('train', 'validation'):
    online = lines(data / 'online_index' / (role + '.jsonl'))
    labels = lines(data / 'offline_labels' / (role + '.jsonl'))
    assert [r['sample_id'] for r in online] == [r['sample_id'] for r in labels]
    assert len(online) == (3027 if role == 'train' else 108)
    for row, label in zip(online, labels):
        assert row['role'] == label['role'] == role and row['g'] == label['g']
        assert set(row) == {'sample_id', 'role', 'scene', 'local_frame', 'g', 'physical_split',
                           'feature_read_paths', 'feature_frames', 'ego_motion', 'motion_read_paths',
                           'window_archive_refs', 'action_queries', 'tool_outputs_materialized'}
        assert row['feature_frames'] == [row['g'], row['g'] - 1]
        assert all(int(Path(p).name.split('_')[0]) <= row['g'] for p in row['feature_read_paths'] + row['motion_read_paths'])
        assert all('/co_llm/ego/' in p for p in row['feature_read_paths'])
        assert row['action_queries']['Ego'] == []
        assert all(label['valid'])
        np.testing.assert_allclose(parse_q9(label['target_q9']), label['waypoints'], atol=.051, rtol=0)
        all_labels[row['sample_id']] = label
    # Independently recompute first/last labels from raw poses in double precision.
    for label in (labels[0], labels[-1]):
        paths = label['label_read_paths']
        inverse = np.linalg.inv(np.load(paths[0]).astype(float))
        expected = [(inverse @ np.load(path).astype(float))[:2, 3] for path in paths[1:]]
        np.testing.assert_allclose(expected, label['waypoints'], atol=1e-9, rtol=0)
    report['data'][role] = dict(frames=len(online), complete_labels=len(labels),
                              q8_distribution=dict(Counter(r['target_q8'] for r in labels)))

examples = lines(data / 'examples/train.jsonl')
assert len(examples) == 8 and len({(r['sample_id'], r['action'], r['task']) for r in examples}) == 8
for example in examples:
    label = all_labels[example['sample_id']]
    assert label['role'] == 'train'
    assert example['target'] == label['target_' + example['task'].lower()]
    if example['task'] == 'Q9':
        plan = read(Path(example['source_run']) / example['action'] / 'plan.json')
        assert plan['q8_raw'] in example['prompt'] and example['prompt'] == plan['q9_prompt']

checked = read(forward / 'forward_check.json')
assert checked['optimizer_steps'] == 0 and checked['all_losses_finite'] and len(checked['rows']) == 8
assert all(np.isfinite(row['loss']) and row['feature_labels_masked'] and row['sequence_tokens'] <= 4096 for row in checked['rows'])
for folder in ('src', 'tests', 'vendor/cmp_mtr'):
    for saved in (forward / 'code_snapshot' / folder).rglob('*.py'):
        assert saved.read_bytes() == (root / saved.relative_to(forward / 'code_snapshot')).read_bytes()
report.update(materialized_examples=8, native_supervised_forward_rows=8, optimizer_steps=0, final_snapshot_matches_current=True)
(data / 'independent_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
