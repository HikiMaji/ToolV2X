"""Re-audit existing episodes; no model inference, training or input rewriting."""
from collections import Counter
import json
from pathlib import Path
import sys
from evaluation.structured import audit_structured_episode

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
RUN=ROOT/'outputs/structured_driver_fix_2026_09_15_v1/acceptance'
refs=[json.loads(x) for x in (RUN/'tasks.jsonl').read_text().splitlines()]
assert len(refs)==80
out=BASE/'audits';out.mkdir(exist_ok=False)
summary=dict(tasks=0,plans=0,all_old_audit_fields_unchanged=True,raw_task_files_unchanged=True,
    final_remote_exclusions={},details=[],scope='replay of existing saved evidence only')
for ref in refs:
 path=RUN/ref['path'];raw=path.read_bytes();task=json.loads(raw)
 audited=audit_structured_episode(task['episode'])
 old=json.loads((Path(task['inputs']['artifact_root'])/'structured_audit.json').read_text())
 assert len(old)==len(audited)
 for before,after in zip(old,audited):
  assert {k:v for k,v in after.items() if k not in ('exclusion_reason_version','non_direct_primary_fields')}==before
 assert path.read_bytes()==raw
 (out/path.name).write_text(json.dumps(audited,indent=2,allow_nan=False)+'\n')
 summary['tasks']+=1;summary['plans']+=len(audited)
 for item in audited[-1]['non_direct_primary_fields']:
  if item['source_role']=='remote':
   summary['details'].append(dict(g=task['row']['g'],arm=ref['arm'],**item))
 if summary['tasks']%20==0:print('Replayed',summary['tasks'],flush=True)
for arm in ('Ego','P','F','PF'):
 rows=[x for x in summary['details'] if x['arm']==arm]
 summary['final_remote_exclusions'][arm]=dict(by_reason=dict(Counter(x['reason'] for x in rows)),
   by_kind=dict(Counter(x['ref']['field_kind'] for x in rows)),
   admitted_dependency=sum(x['admitted_dependency'] for x in rows),dropped=sum(x['dropped'] for x in rows))
assert summary['plans']==160
assert len(summary['details'])==30
assert all(x['reason']=='ego_filter' for x in summary['details'])
assert sum(x['dropped'] for x in summary['details'])==10
assert all(x['legacy_reason']=='structured_capacity' for x in summary['details'] if x['dropped'])
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
summary['model_modules_imported']=False
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='details'},indent=2),flush=True)
