"""Independent verification of the bounded real twenty-epoch fit."""
from pathlib import Path
from collections import Counter
import json, math
import torch
from planning import train_structured_driver as T
from planning.structured_validation import evidence_condition, frame_condition_weights, summarize_measurements
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
def read(p):return json.loads(Path(p).read_text())
rows=[json.loads(x) for x in (BASE/'rows.jsonl').read_text().splitlines()]
trace=[json.loads(x) for x in (BASE/'training_trace.jsonl').read_text().splitlines()]
steps=[r for r in trace if r['kind']=='step'];train=[r for r in trace if r['kind']=='train']
assert len(steps)==1280 and [r['optimizer_steps'] for r in steps]==list(range(1,1281))
assert len(train)==2560 and all(r['role']=='train' and math.isfinite(r['loss']) for r in train)
expected=Counter((r['sample_id'],r['stage'],evidence_condition(r)) for r in rows if r['role']=='train')
actual=Counter((r['sample_id'],r['stage'],r['condition']) for r in train)
assert Counter({k:v*20 for k,v in expected.items()})==actual
config=read(BASE/'training_config.json')
initial=T._load(BASE/'training/checkpoint_000000.pt');final=T._load(BASE/'training/checkpoint_001280.pt')
assert final['progress']['optimizer_steps']==1280 and final['progress']['epoch']==20
assert final['config']==config and config['save_interval']==config['validation']['interval_batches']==500
assert [x['batches'] for x in final['report']['validation_history']]==[0,500,1000,1280]
original=T._load(ROOT/'outputs/structured_driver_fix_2026_09_15_v1/acceptance/initialized_seed7.pt')
assert T._same(initial['model_state'],original['model_state'])
changed=sum(not torch.equal(initial['model_state'][k],v) for k,v in final['model_state'].items())
assert changed>0
selected=read(BASE/'training/selection.json');assert selected is not None
T._load(BASE/'training'/selected['checkpoint'])
for path in sorted((BASE/'training').glob('checkpoint_*.pt')):
    T._load(path)
# Re-evaluate final held-out stages using the public checkpoint loader, no optimization.
planner=T.load_structured_planner(BASE/'training/checkpoint_001280.pt',device='cuda')
weights,_=frame_condition_weights(rows,'validation');records=[]
with torch.no_grad():
    for i,row in enumerate(rows):
        if row['role']!='validation':continue
        task=read(row['source_task'])
        _,detail=T.training_loss(planner.model,T._read_features(row['inputs']['feature_path']),row,task=task,
            refinement_depth=config['refinement_depth'],objective=config['objective'],evaluation=True)
        records.extend(dict(m,stage=row['stage'],condition=evidence_condition(row,task),weight=weights[i],
                            label=row['supervision'],execution_spec=task['episode']['limits']['execution_spec'])
                       for m in detail['measurements'])
recomputed=summarize_measurements(records,None)
archived=final['report']['validation_history'][-1]['metrics']
# Preserve any tiny CUDA difference; do not recast allclose as bitwise equivalence.
def max_numeric_diff(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max([max_numeric_diff(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,list):
        assert len(a)==len(b);return max([max_numeric_diff(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)
        return abs(a-b)
    assert a==b;return 0.
difference=max_numeric_diff(archived,recomputed)
for kind in ('selected_stage','refinements'):
    for key in ('rows','valid_plans','invalid_plans','valid_label_points'):
        assert archived[kind][key]==recomputed[kind][key]
unchanged=0
for p in (BASE/'source_snapshot').rglob('*.py'):
    assert p.read_bytes()==(ROOT/p.relative_to(BASE/'source_snapshot')).read_bytes();unchanged+=1
# Full source-task content comparison against retained fit input snapshots.
sources=list(dict.fromkeys(r['source_task'] for r in rows))
for i,p in enumerate(sources):assert read(p)==read(BASE/'training/inputs'/('task_%06d.json'%i))
result=dict(version='toolv2x_structured_driver_small_fit_result_v1',status='completed_twenty_epochs',
    optimizer_steps=1280,epochs_completed=20,train_rows_consumed=2560,validation_rows=32,
    train_conditions=dict(Counter(r['condition'] for r in train)),changed_parameter_tensors=changed,
    initial_equals_previous_seed7=True,checkpoint_full_load_checks=True,selected_checkpoint=selected,
    validation_history=final['report']['validation_history'],reloaded_final_validation=recomputed,
    reloaded_validation_exact=recomputed==archived,reloaded_validation_max_numeric_difference=difference,
    invalid_prior_masks=final['report']['invalid_prior_masks'],invalid_prior_reasons=final['report']['invalid_prior_reasons'],
    train_mean_raw_row_loss=sum(r['loss'] for r in train)/len(train),
    source_files_byte_identical=unchanged,task_snapshots_equal=len(sources),
    runtime=read(BASE/'training_result.json'),new_provider_calls=0,new_got_calls=0,new_mtr_calls=0,
    scope='16 train and 4 already-used development frames; fixed archived evidence; not full dataset adaptation, online recollection or method superiority.')
(BASE/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('validation_history','reloaded_final_validation')},ensure_ascii=False),flush=True)
