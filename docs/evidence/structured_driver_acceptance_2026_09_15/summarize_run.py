"""Summarize retained real attempts and verify the bounded checkpoint lineage."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
from planning.train_structured_driver import _load

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
result=json.loads((BASE/'bounded_bootstrap_result.json').read_text())
assert result['status']!='running' and result['formal_adaptation_started'] is False
runs=[];physics=[]
labels={r['g']:r for r in map(json.loads,(BASE/'offline_bootstrap/labels.jsonl').read_text().splitlines())}
def dynamics(points):
    if points is None:return None
    velocity=np.diff(np.vstack(([0.,0.],np.asarray(points))),axis=0)/.5
    return dict(max_speed_mps=float(np.linalg.norm(velocity,axis=1).max()),
        max_acceleration_mps2=float(np.linalg.norm(np.diff(velocity,axis=0)/.5,axis=1).max()))

names=['initial','initial_recheck']+['acceptance_epoch%d'%e['epoch'] for e in result['epochs']]
for name in names:
    root=BASE/name
    progress=json.loads((root/'progress.json').read_text())
    audit=json.loads((root/'acceptance_audit.json').read_text())
    assert progress['status']=='completed' and progress['attempted_tasks']==80
    tasks=[json.loads((root/r['path']).read_text()) for r in
        map(json.loads,(root/'tasks.jsonl').read_text().splitlines())]
    issued=Counter(request['tool'] for task in tasks for request in task['episode']['requests'])
    runs.append(dict(name=name,wall_seconds=progress['wall_seconds'],attempted_tasks=80,
        initial_valid_frames=audit['initial_valid_frames'],passing_tasks=audit['passing_tasks'],
        audit_failures=audit['audit_failures'],issued_requests=dict(issued),
        pass_gate=audit['pass_gate'],per_condition=[dict(condition=arm,
            attempted=sum(t['arm']==arm for t in tasks),
            completed=sum(t['arm']==arm and t['status']=='completed' for t in tasks)) for arm in ('Ego','P','F','PF')]))
    for task in tasks:
        if task['arm']!='Ego':continue
        output=task['episode']['plans'][0].get('output') or {}
        physics.append(dict(run=name,g=task['row']['g'],role=task['row']['role'],
            output=dynamics(output.get('waypoints')),offline_label=dynamics(labels[task['row']['g']]['waypoints'])))

latest=Path(result['epochs'][-1]['checkpoint']).parent
checkpoints=[];lineage=set()
for path in sorted(latest.glob('checkpoint_*.pt')):
    saved=_load(path)
    lineage.add(saved['model_version']['training']['training_run_id'])
    progress=saved['progress']
    assert progress['optimizer_steps']<=24 and progress['epoch']<=3
    assert saved['config']==json.loads((BASE/'offline_bootstrap/concrete_training_v2.json').read_text())
    checkpoints.append(dict(path=str(path),epoch=progress['epoch'],optimizer_steps=progress['optimizer_steps'],
        bytes=path.stat().st_size,complete_binding_verified=True))
assert len(lineage)==1
validations=[]
for epoch in result['epochs']:
    report=json.loads((Path(epoch['checkpoint']).parent/'report.json').read_text())
    metrics=report['validation_history'][-1]['metrics']['selected_stage']
    validations.append(dict(epoch=epoch['epoch'],optimizer_steps=epoch['optimizer_steps'],
        loss=metrics['loss'],valid_plans=metrics['valid_plans'],total_plans=metrics['rows'],
        selected_checkpoint=report['selection']))
summary=dict(status=result['status'],formal_adaptation_started=False,
    full_four_condition_collection_started=False,production_code_changed=False,
    train_frames=16,validation_frames=4,all_labels_complete=True,
    total_fixed_tasks=sum(r['attempted_tasks'] for r in runs),runs=runs,
    epochs=result['epochs'],validation=validations,retained_lineage_checkpoints=checkpoints,
    code_state='main with per-run literal source snapshots; no production code edits',
    historical_inconsistency='Three initial runtime tensor reconstruction errors and one pretraining verification failure retained. Later full recheck and strict training completed. Root cause remains unlocalized; no validator bypass or tolerance change.',
    conclusions_allowed=['Current frozen initialization and bounded bootstrap readiness result only'],
    conclusions_not_allowed=['P/F effectiveness','feedback method benefit','paper comparison','closed-loop safety'],
    summarized_utc=datetime.now(timezone.utc).isoformat())
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(BASE/'physics_diagnosis.json').write_text(json.dumps(physics,indent=2,allow_nan=False)+'\n')
evidence=ROOT/'docs/evidence/structured_driver_acceptance_2026_09_15'
evidence.mkdir(parents=True,exist_ok=False)
for name in ('summary.json','physics_diagnosis.json','execution_record.json','bounded_bootstrap_result.json'):
    shutil.copy2(BASE/name,evidence/name)
for name in ('coverage.json','preflight.json','concrete_training_v2.json'):
    shutil.copy2(BASE/'offline_bootstrap'/name,evidence/name)
# The small operational wrappers are retained as run evidence, not production APIs.
for path in BASE.glob('*.py'):
    shutil.copy2(path,evidence/path.name)
print(json.dumps(dict(status=summary['status'],tasks=summary['total_fixed_tasks'],
    optimizer_steps=checkpoints[-1]['optimizer_steps'],checkpoint_bindings_verified=len(checkpoints)),indent=2))
