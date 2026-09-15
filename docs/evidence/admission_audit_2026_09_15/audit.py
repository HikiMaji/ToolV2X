"""Read saved fields and replay receiver validation; no driver/predictor or GT."""
from collections import Counter
import copy,json,math
from pathlib import Path
import numpy as np
from tools.task_spec import field_key
from planning import structured_inputs as S
from planning.evidence import remote_units

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
RUN=ROOT/'outputs/structured_driver_fix_2026_09_15_v1/acceptance'
load=lambda path:json.loads(path.read_text())
refs=[json.loads(x) for x in (RUN/'tasks.jsonl').read_text().splitlines()]
frames=sorted({x['sample_id'] for x in refs})
totals={};missing=[];comparisons=[];validated=0
for sample in frames:
 arms={}
 for arm in ('P','F','PF'):
  ref=next(x for x in refs if x['sample_id']==sample and x['arm']==arm)
  task=load(RUN/ref['path']);ep=task['episode'];plan=ep['plans'][-1];p=plan['prepared']
  S.validate_structured_prepared(p);validated+=1
  ledger=ep['ledger_snapshots'][plan['ledger_snapshot']]
  acquired={field_key(x['ref']):x for x in ledger['acquired_fields']}
  groups={field_key(ref):g for g in p['admission_report']['field_groups'] for ref in g['primary_refs']}
  entities={e['entity_id']:e for e in p['entities']}
  local,remote=S._tracks(ledger,remote_units(ledger));tracks={t['alias']:t for t in local+remote}
  spec=S.StructuredDriverSpec.from_dict(p['driver_spec'])
  stat=totals.setdefault(arm,dict(acquired=Counter(),direct=Counter(),non_direct=Counter()))
  for key,record in acquired.items():
   kind=record['ref']['field_kind']
   if kind not in ('history','forecast'):continue
   stat['acquired'][kind]+=1
   group=groups.get(key)
   if group is not None and group['use']=='tensor':stat['direct'][kind]+=1;continue
   reason=group['use'] if group else 'no_group';stat['non_direct'][reason]+=1
   entity=entities[group['entity_id']]
   ts=[tracks[(a['source'],a['track_handle'])] for a in entity['aliases']]
   recomputed=S._role(ts,p['ego_history_used'],spec);assert recomputed==entity['role']
   representative=S._representative(ts);box=np.asarray(representative['records']['anchor']['value']['box'])
   hist=representative['records'].get('history');metrics={}
   if hist:
    h=hist['value'];mask=np.array(h['history_valid'])&np.array(p['ego_history_used']['valid'])
    observed=np.array(h['history'])[mask];ego=np.array(p['ego_history_used']['states'])[mask]
    metrics=dict(common_steps=int(mask.sum()),history_rmse_m=float(np.sqrt(np.mean(np.sum((observed[:,:2]-ego[:,:2])**2,axis=1)))),heading_max_rad=max(S._directed_angle_distance(a,b) for a,b in zip(observed[:,6],ego[:,2])))
   without=copy.deepcopy(ts)
   for track in without:track['records'].pop('history',None)
   missing.append(dict(sample_id=sample,g=ep['g'],arm=arm,ref=record['ref'],reason=reason,
    admitted_dependency=record['ref'] in p['admission_report']['admitted_field_refs'],
    reported_dropped=record['ref'] in p['admission_report']['dropped_field_refs'],
    raw_drop_reason=next((x['reason'] for x in p['admission_report']['dropped'] if x['ref']==record['ref']),None),
    entity_id=entity['entity_id'],aliases=entity['aliases'],role=entity['role'],association=entity['association'],
    representative_source=representative['source_role'],representative_alias=list(representative['alias']),
    current_distance_m=float(np.linalg.norm(box[:2])),dimensions=box[3:6].tolist(),
    role_without_history_diagnostic=S._role(without,p['ego_history_used'],spec),
    tensor_index=entity['tensor_index'],entity_slots_used=sum(p['tensor_inputs']['entity_mask']),
    max_entity_slots=spec.max_entities,**metrics))
  arms[arm]=(acquired,groups,entities,ep)
 f,fg,fe,_=arms['F'];pf,pfg,pfe,_=arms['PF']
 fkeys={k for k,v in f.items() if v['ref']['field_kind']=='forecast'}
 pfkeys={k for k,v in pf.items() if v['ref']['field_kind']=='forecast'}
 changed=[]
 for key in sorted(fkeys&pfkeys):
  assert f[key]['value']==pf[key]['value'],'forecast content drift at same field identity'
  a,b=fg[key],pfg[key]
  if a['use']!=b['use']:
   assert fe[a['entity_id']]['aliases']==pfe[b['entity_id']]['aliases']
   assert fe[a['entity_id']]['representative_anchor']==pfe[b['entity_id']]['representative_anchor']
   changed.append(dict(ref=f[key]['ref'],f_use=a['use'],pf_use=b['use'],f_entity=fe[a['entity_id']],pf_entity=pfe[b['entity_id']]))
 comparisons.append(dict(sample_id=sample,g=arms['F'][3]['g'],same_forecast_ref_set=fkeys==pfkeys,
  shared_forecasts=len(fkeys&pfkeys),f_only_forecasts=len(fkeys-pfkeys),pf_only_forecasts=len(pfkeys-fkeys),changed=changed))
 print(sample,flush=True)
import sys
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
summary=dict(scope='offline saved receiver audit only; no model generation, no labels, no training',validated_final_prepared=validated,totals=totals,missing=missing,comparisons=comparisons)
(BASE/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(validated=validated,totals=totals,missing_count=len(missing),changed_forecasts=sum(len(x['changed']) for x in comparisons),same_sets=all(x['same_forecast_ref_set'] for x in comparisons)),indent=2))
