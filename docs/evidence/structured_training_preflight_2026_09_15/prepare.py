"""Bounded offline export from the unchanged real v2 acceptance archive."""
from collections import Counter
from pathlib import Path
import json, random, shutil
from planning.adaptation_data import ego_label
from planning.train_structured_driver import prepare_training_rows, _config
from planning.structured_validation import frame_condition_weights, evidence_condition
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE = ROOT/'outputs/structured_driver_fix_2026_09_15_v1/acceptance'
def save(name, obj):
    (BASE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
assert not (BASE/'rows.jsonl').exists()
# Freeze relevant source and historical configuration without changing either.
for folder in ('src', 'configs/structured_driver_readiness_v1'):
    for path in (ROOT/folder).rglob('*'):
        if path.is_file() and path.suffix in ('.py','.json','.jsonl'):
            target=BASE/'source_snapshot'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path,target)
launch=json.loads((SOURCE/'launch.json').read_text())
labels=[dict(ego_label('train', r['g']), **{k:r[k] for k in ('sample_id','scene','g','role')}) for r in launch['samples']]
assert len(labels)==20 and all(all(x['valid']) for x in labels)
(BASE/'labels.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in labels))
rows=prepare_training_rows(SOURCE/'tasks.jsonl', labels)
assert len(rows)==160
(BASE/'rows.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
config=json.loads((ROOT/'configs/structured_driver_readiness_v1/training_v2.json').read_text())
config['driver_spec']['version']='toolv2x_structured_driver_v2'
# Same architecture, objective, optimizer, refinement and batch size; only bounded schedule differs.
config.update(epochs=1,save_interval=1)
config['validation']['interval_batches']=4
_config(config)
save('training_config.json',config)
coverage={}
conditions=[evidence_condition(row) for row in rows]
for role in ('train','validation'):
    weights, detail=frame_condition_weights(rows,role)
    coverage[role]=detail
    assert all(not x['conditions'] for x in detail['missing_conditions'])
    coverage[role]['stage_conditions']=dict(Counter(c for c,r in zip(conditions,rows) if r['role']==role))
    coverage[role]['recordings']=sorted({r['recording'] for r in rows if r['role']==role})
assert not set(coverage['train']['recordings']) & set(coverage['validation']['recordings'])
order=random.Random(config['seed']).sample([i for i,r in enumerate(rows) if r['role']=='train'],128)
first=[dict(index=i,sample_id=rows[i]['sample_id'],stage=rows[i]['stage'],condition=conditions[i]) for i in order[:8]]
save('coverage.json',coverage)
save('plan.json',dict(version='toolv2x_structured_training_preflight_v1',source=str(SOURCE),
    lineage_steps=4,interrupt_after=2,total_optimizer_steps_across_two_lineages=8,
    config='training_config.json',rows='rows.jsonl',first_eight_training_rows=first,
    offline_fixed_evidence=True,new_provider_calls=0,new_got_generations=0,
    validation_scope='four already-used development frames; not independent test',
    role_filter_control=dict(status='declared_not_executed',
      definition='Future separate fixed lawful evidence keep-versus-filter inferred ego entities; same shared driver and generation budget; no free PF history in F-only'),
    stop='No full collection, full adaptation, policy fitting or online recollection.'))
save('source_archive_stat.json',{str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    for p in SOURCE.rglob('*') if p.is_file()})
print(json.dumps(dict(rows=len(rows),label_frames=len(labels),train_rows=128,validation_rows=32,
    train_recordings=8,validation_recordings=2,first_eight_conditions=[r['condition'] for r in first]),ensure_ascii=False),flush=True)
