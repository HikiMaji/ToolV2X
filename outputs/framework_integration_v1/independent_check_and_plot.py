"""Independent arithmetic check of the saved integration run; all future reads are offline."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path('/root/autodl-tmp/ToolV2X')
OUT = ROOT/'outputs/framework_integration_v1'
def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def size(packet):
    return len(json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(',', ':'), sort_keys=True).encode())

tasks = read(ROOT/'outputs/framework_debug_v2/tasks.jsonl')
groups, roles, coverage = defaultdict(list), defaultdict(set), defaultdict(list)
for record in tasks:
    task = json.loads(Path(record['path']).read_text())
    groups[task['sample_id']].append(task)
    roles[task['role']].add(task['scene'].rsplit('_', 1)[0])
    if task['policy'] == 'Ego':
        assert task['remote_reads'] == []
    episode = task['episode']
    actual = [s for s in episode['steps'] if 'request' in s]
    assert episode['cost']['calls'] == len(actual)
    assert episode['cost']['request_bytes'] == sum(size(s['request']) for s in actual)
    assert episode['cost']['response_bytes'] == sum(size(s['response']) for s in actual)
    prepared = task['prepared']
    assert prepared['q8_executed'] is False and prepared['q8_raw'] is None
    assert prepared['evidence_selection']['input_tokens']+256 <= 4096
    remote = prepared['remote_evidence_used']
    coverage[task['policy']].append((len(prepared['evidence_used']['objects']), len(remote['objects']) if remote else 0))
assert not roles['train'] & roles['validation']
for values in groups.values():
    assert {t['policy'] for t in values} == {'Ego', 'P', 'F', 'PF', 'rule'}
    assert len(values) == 5
    ego = next(t for t in values if t['policy'] == 'Ego')
    assert all(t['prepared']['evidence_used'] == ego['prepared']['evidence_used'] for t in values)
    assert all(t['feature_path'] == ego['feature_path'] for t in values)

labels = {r['sample_id']:r for r in read(ROOT/'outputs/paired_driving_data_v1/offline_labels/validation.jsonl')}
reported = {(r['sample_id'], r['policy']):r for r in read(ROOT/'outputs/framework_evaluation_v1/rows.jsonl')}
predictions, recomputed, repeated = defaultdict(dict), defaultdict(list), []
for record in read(ROOT/'outputs/framework_generation_v1/generations.jsonl'):
    saved = json.loads(Path(record['generation_path']).read_text())
    task = json.loads(Path(saved['task_path']).read_text())
    points = np.asarray(saved['plan']['waypoints'])
    truth = np.asarray(labels[task['sample_id']]['waypoints'])
    delta = np.sqrt(((points-truth)**2).sum(axis=1))
    expected = reported[(task['sample_id'], task['policy'])]
    assert abs(delta.mean()-expected['ADE3']) < 1e-9
    assert abs(delta[-1]-expected['FDE3']) < 1e-9
    assert abs(sum(expected[k] for k in ('local_model_seconds','service_seconds','receiver_seconds','driver_seconds'))-expected['total_compute_seconds']) < 1e-9
    predictions[task['sample_id']][task['policy']] = points
    recomputed[task['policy']].append(dict(ADE3=float(delta.mean()), FDE3=float(delta[-1])))
    if np.max(np.linalg.norm(points-points[0], axis=1)) < 1e-6:
        repeated.append(dict(sample_id=task['sample_id'], policy=task['policy']))
summary = json.loads((ROOT/'outputs/framework_evaluation_v1/summary.json').read_text())
for policy, rows in recomputed.items():
    for key in ('ADE3','FDE3'):
        assert abs(np.mean([r[key] for r in rows])-summary['policies'][policy]['mean_'+key]) < 1e-9
result = dict(status='passed', tasks_checked=len(tasks), frames=len(groups),
    recording_groups={k:len(v) for k,v in roles.items()}, generated_trajectories_checked=len(reported),
    exact_shared_ego_input=True, actual_wire_accounting=True, independent_ADE_FDE=True,
    timing_addition=True, repeated_point_trajectories=repeated,
    mean_retained_records={p:np.mean(v,axis=0).tolist() for p,v in coverage.items()},
    scope='integration arithmetic and input audit; parsed trajectories are not necessarily physically useful')
(OUT/'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n')

font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
if font.is_file():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
fig, axes = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)
colors = {'Ego':'#3269ae','P':'#e79b26','F':'#18866b','PF':'#9659ac','rule':'#d9534f'}
# Time versus forward displacement makes repeated six-point outputs visible; no case selection.
for ax, (sample, policies) in zip(axes.flat, sorted(predictions.items())):
    label = labels[sample]
    times = np.arange(7)*.5
    ax.plot(times, [0]+[p[0] for p in label['waypoints']], 'k--', linewidth=2.4, label='离线真实轨迹')
    for policy, points in policies.items():
        ax.plot(times, np.r_[0,points[:,0]], marker='o', markersize=3, color=colors[policy], label=policy)
    ax.set_title('g%d · %s'%(label['g'], sample.replace('testoutput_CAV_data_', '').split(':')[0]))
    ax.set_xlabel('未来时间 (s)')
    ax.set_ylabel('自车前向位移 (m)')
    ax.grid(alpha=.18)
axes[0,0].legend(ncol=3, fontsize=8)
fig.suptitle('完整流程联调：全部 4 个预选验证帧\n只训练 16 帧 × 5 策略 × 1 轮；可解析输出不代表驾驶质量合格', fontsize=14)
figure = ROOT/'docs/figures/framework_integration_trajectories.png'
fig.savefig(figure, dpi=150)
print(json.dumps(result, indent=2))
