"""Independent state, label and archive checks after the bounded real fit."""
from pathlib import Path
from collections import Counter
import json, math
import numpy as np
import torch
from planning import train_structured_driver as T
from common import v2v4real_meta as M
BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]
def read(name):return json.loads((BASE/name).read_text())
def lines(name):return [json.loads(x) for x in (BASE/name).read_text().splitlines()]
config=read('training_config.json')
a=T._load(BASE/'continuous/checkpoint_000004.pt')
b=T._load(BASE/'resumed/checkpoint_000004.pt')
initial=T._load(BASE/'continuous/checkpoint_000000.pt')
checks={key:T._same(a[key],b[key]) for key in ('model_state','optimizer_state','rng','progress','report','data_binding','input_layout','config','driver_spec')}
assert all(checks.values()),checks
assert a['progress']['optimizer_steps']==b['progress']['optimizer_steps']==4
assert initial['progress']['optimizer_steps']==0
acceptance_init=T._load(ROOT/'outputs/structured_driver_fix_2026_09_15_v1/acceptance/initialized_seed7.pt')
assert T._same(initial['model_state'],acceptance_init['model_state'])
changed=[k for k in a['model_state'] if not torch.equal(a['model_state'][k],initial['model_state'][k])]
assert changed
retained={}
for mode,end in (('continuous',4),('interrupted',2),('resumed',4)):
    retained[mode]=[]
    for step in range(end+1):
        path=BASE/mode/('checkpoint_%06d.pt'%step)
        state=T._load(path)
        assert state['progress']['optimizer_steps']==step
        assert state['driver_spec']==config['driver_spec']
        retained[mode].append(dict(step=step,bytes=path.stat().st_size))
print('All checkpoints load; exact resume state equality verified.',flush=True)
# Recompute all 120 reference points with solve, independently of ego_label implementation.
max_error=0.
for label in lines('labels.jsonl'):
    origin=M.load_pose('train',label['g'])
    for offset,expected in zip((5,10,15,20,25,30),label['waypoints']):
        actual=np.linalg.solve(origin.astype(float),M.load_pose('train',label['g']+offset).astype(float))[:2,3]
        max_error=max(max_error,float(np.abs(actual-np.asarray(expected)).max()))
assert max_error<1e-9
snapshots=list((BASE/'source_snapshot').rglob('*'))
source_count=0
for p in snapshots:
    if p.is_file():
        assert p.read_bytes()==(ROOT/p.relative_to(BASE/'source_snapshot')).read_bytes()
        source_count+=1
for path,stat in read('source_archive_stat.json').items():
    p=Path(path)
    assert p.stat().st_size==stat['size'] and p.stat().st_mtime_ns==stat['mtime_ns']
# Check full task semantic contents and arrays against the fit snapshots, not only stats.
rows=lines('rows.jsonl');sources=list(dict.fromkeys(r['source_task'] for r in rows))
features=list(dict.fromkeys(r['inputs']['feature_path'] for r in rows))
for i,path in enumerate(sources):
    actual=json.loads(Path(path).read_text())
    expected=json.loads((BASE/'continuous/inputs'/('task_%06d.json'%i)).read_text())
    assert actual==expected
for i,path in enumerate(features):
    actual=T._read_features(path);expected=T._read_features(BASE/'continuous/inputs'/('feature_%06d.npz'%i))
    assert actual.keys()==expected.keys()
    assert all(np.array_equal(actual[k],expected[k]) for k in actual)
print('Labels, source files and all real archive inputs verified.',flush=True)
traces={m:lines(m+'_trace.jsonl') for m in ('continuous','interrupted','resumed')}
steps=[r for m in traces for r in traces[m] if r['kind']=='step']
assert len(steps)==8
assert all(math.isfinite(v) for r in steps for v in r['losses'])
train=[r for r in traces['continuous'] if r['kind']=='train']
assert all(r['role']=='train' for r in train)
# Reproduce loss details through the public checkpoint loader, with current-model causal prefixes.
row=next(r for r in rows if r['role']=='train' and r['stage']==2)
features=T._read_features(row['inputs']['feature_path'])
measurements=[]
for mode in ('continuous','resumed'):
    planner=T.load_structured_planner(BASE/mode/'checkpoint_000004.pt',device='cuda')
    with torch.no_grad():
        loss,detail=T.training_loss(planner.model,features,row,refinement_depth=1,
            objective=config['objective'],evaluation=True)
    measurements.append(detail)
assert measurements[0]==measurements[1]
results={m:read(m+'_result.json') for m in traces}
summary=dict(version='toolv2x_structured_training_preflight_result_v1',status='PASS',
    frame_count=20,archived_tasks=80,exported_stage_rows=160,train_rows=128,validation_rows=32,
    total_actual_optimizer_steps=8,steps_per_lineage=4,training_conditions_seen=dict(Counter(r['condition'] for r in train)),
    finite_losses=True,exact_resume_checks=checks,changed_parameter_tensors=len(changed),
    changed_encoder_tensors={p:sum(k.startswith(p+'.') for k in changed) for p in ('observation_encoder','forecast_encoder','output_head')},
    checkpoints=retained,offline_label_points=120,label_max_abs_error=max_error,
    source_files_byte_identical=source_count,archive_tasks_content_equal=len(sources),feature_files_equal=len(features),
    reloaded_pf_measurements_exact=True,validation_history=a['report']['validation_history'],
    invalid_prior_masks=a['report']['invalid_prior_masks'],run_results=results,
    remote_gradients=read('remote_gradients.json'),storage_preflight=read('storage_preflight.json'),
    schedule_projection=read('schedule_projection.json'),
    new_provider_calls=0,new_got_generations=0,full_collection_started=False,formal_adaptation_started=False,
    limitations=['Four optimization steps are training mechanics evidence, not method quality.',
      'Fixed archived evidence; current-model prefixes recomputed offline; no trained online recollection.',
      'Four development frames are already used, not independent paper evaluation.',
      'Single-device exact resume is not cross-device determinism.',
      'Small-batch preflight does not establish full-collection runtime or storage feasibility.'])
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('validation_history','checkpoints','run_results')},ensure_ascii=False),flush=True)
