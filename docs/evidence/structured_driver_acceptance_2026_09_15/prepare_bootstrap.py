"""Offline-only label join and coverage audit; never starts training."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from planning.adaptation_data import ego_label
from planning.train_structured_driver import prepare_training_rows
from planning.structured_validation import frame_condition_weights
from evaluation.structured import audit_structured_episode

BASE = Path(__file__).resolve().parent
initial = BASE / 'initial'
out = BASE / 'offline_bootstrap'
assert json.loads((initial / 'progress.json').read_text())['status'] == 'completed'
out.mkdir(exist_ok=False)
def save(name, data):
    (out / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
save('label_access.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
    scope='independent offline ego_label after initial real acceptance; no online model invocation'))
rows = json.loads((initial / 'launch.json').read_text())['samples']
labels = [dict(ego_label('train', row['g']), **{k:row[k] for k in ('sample_id','scene','g','role')}) for row in rows]
(out / 'labels.jsonl').write_text(''.join(json.dumps(label,allow_nan=False)+'\n' for label in labels))
manifest = [dict(sample_id=row['sample_id'], path=str(initial/'tasks'/('g%d_Ego.json'%row['g']))) for row in rows]
(out / 'ego_tasks.jsonl').write_text(''.join(json.dumps(ref)+'\n' for ref in manifest))
coverage = []
for ref,label in zip(manifest,labels):
    task=json.loads(Path(ref['path']).read_text())
    try:
        audit_structured_episode(task['episode'])
        audit_error=None
    except Exception as exc:
        audit_error=str(exc)
    coverage.append(dict(sample_id=ref['sample_id'], role=task['row']['role'],
        archive_status=task['status'], valid_label_points=sum(label['valid']),
        prepared_stages=len(task['episode']['plans']), audit_error=audit_error))
save('coverage.json',coverage)
assert all(x['audit_error'] is None and x['prepared_stages']==1 and x['valid_label_points']==6 for x in coverage)
training = prepare_training_rows(out/'ego_tasks.jsonl', labels)
assert len(training)==20 and all(r['stage']==0 for r in training)
train_weights,train_coverage=frame_condition_weights(training,'train')
_,validation_coverage=frame_condition_weights(training,'validation')
assert train_coverage['eligible_rows']==16 and validation_coverage['eligible_rows']==4
(out/'rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in training))
config=json.loads((initial/'predeclared_package/bootstrap_training_expected_v2.json').read_text())
config['validation']['interval_batches']=math.ceil(train_coverage['eligible_rows']/config['batch_size'])
assert config['epochs']==3 and config['validation']['interval_batches']==8
save('concrete_training_v2.json',config)
save('preflight.json',dict(train=train_coverage,validation=validation_coverage,
    eligible_labels_certified=True,training_started=False,epochs_cap=3,
    note='Initial 80 outcomes remain failures; this validates only retained Ego prepared inputs and offline labels.'))
print(json.dumps(dict(rows=len(training),train=16,validation=4,batches_per_epoch=8,training_started=False)))
