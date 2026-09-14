from pathlib import Path
import json,numpy as np,ast
out=Path('/root/autodl-tmp/ToolV2X/outputs/receiver_round_robin_smoke_2026_09_14_v2')
prior=out.parent/'t9_real_smoke_2026_09_13_v1'
labels={r['sample_id']:r for r in json.loads((out/'offline/selected_labels.json').read_text())}
reported={(r['g'],r['receiver'],r['arm']):r for r in json.loads((out/'offline/summary.json').read_text())['rows']}
rows=[];stages=[];binding=[]
for path in sorted((out/'tasks').glob('*.json')):
 task=json.loads(path.read_text());ep=task['episode'];label=labels[task['row']['sample_id']]
 assert label['role']==task['row']['role']=='validation' and label['g']==task['row']['g'] and all(label['valid'])
 assert not ep['gt_labels_read'] and task['status']=='completed'
 assert ep['limits']['receiver_spec']['version']==task['receiver']
 for plan in ep['plans']:
  raw=plan['output']['q9_raw'];points=ast.literal_eval(raw[raw.index('['):raw.index(']')+1])
  assert points==[tuple(x) for x in plan['output']['waypoints']] or points==plan['output']['waypoints']
  pred=np.asarray(plan['output']['waypoints']);gt=np.asarray(label['waypoints']);err=np.linalg.norm(pred-gt,axis=1)
  prepared=plan['prepared'];c=plan['output']['q9_cost']
  assert c['input_tokens']==prepared['evidence_selection']['input_tokens']
  assert c['input_tokens']+prepared['receiver_spec']['generation_reserve']<=prepared['receiver_spec']['context_limit']==4096
  assert c['feature_tokens']==540 and prepared['q8_executed'] is False
  groups=prepared['admission_report']['field_groups']
  stages.append(dict(task=path.stem,stage=plan['stage'],ADE3=float(err.mean()),FDE3=float(err[-1]),input_tokens=c['input_tokens'],targets=[g['anchor_ref']['track_handle'] for g in groups]))
 report=reported[(task['row']['g'],task['receiver'],task['arm'])]
 assert abs(report['ADE3']-err.mean())<1e-12 and abs(report['FDE3']-err[-1])<1e-12
 assert report['driver_calls']==len(ep['plans']) and report['input_tokens']==sum(p['output']['q9_cost']['input_tokens'] for p in ep['plans'])
 assert report['output_tokens']==sum(p['output']['q9_cost']['output_tokens'] for p in ep['plans'])
 assert report['rpc_rounds']==len(ep['requests'])
 provider=json.loads((out/'inputs'/path.stem/'provider_records.json').read_text())
 receipts=ep['ledger_snapshots'][-1]['receipts']
 assert len(provider['primitive_records'])==len(receipts)==report['calls']
 for a,b in zip(provider['primitive_records'],receipts):
  assert all(a[k]==b[k] for k in ('request','response','cost'))
  if a['request']['tool']=='P':assert a['cost']['model_seconds']==0
 if task['arm']=='alternating':
  q=ep['requests'][1];assert q['tau_new']==ep['plans'][1]['output']['waypoints'] and q['tau_old']==ep['plans'][0]['output']['waypoints']
  assert q==provider['primitive_records'][1]['request']
  scopes=[]
  for obj in ep['plans'][-1]['prepared']['remote_evidence_used']['objects']:
   scopes.extend((obj['track_id'],m.get('context_scope')) for m in obj['field_metadata'].values())
  binding.append(dict(task=path.stem,actual_second_request_bound=True,F_ranking=[r['track_handle'] for r in provider['primitive_records'][1]['response']['ranking']],final_contexts=scopes))
 if task['receiver']=='toolv2x_receiver_v1':
  old=json.loads((prior/'tasks'/('g%d_%s.json'%(task['row']['g'],task['arm']))).read_text())['episode']
  assert len(old['plans'])==len(ep['plans'])
  assert all(a['output']['q9_raw']==b['output']['q9_raw'] and a['prepared']['q9_prompt']==b['prepared']['q9_prompt'] for a,b in zip(old['plans'],ep['plans']))
 rows.append(dict(task=path.stem,ADE3=float(err.mean()),FDE3=float(err[-1]),driver_calls=len(ep['plans']),rpc_rounds=len(ep['requests']),primitive_calls=len(receipts),status='PASS'))
assert len(rows)==12 and len(stages)==24
result=dict(status='PASS',method='independent raw answer parsing and numpy Euclidean errors; direct saved event/receipt/field comparisons, no evaluator calls',tasks=rows,stages=stages,alternating_binding=binding,old_v1_all_stage_prompts_and_raw_answers_equal_original=True,online_labels_read=False,new_model_calls=0)
Path('/tmp/toolv2x-receiver-root-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS:12 final metrics and24 raw plans independently recomputed; receipt/Fbinding/token budgets verified; freshv1 all stages reproduce prior prompts/raw answers.')
for value in binding:print(value)
