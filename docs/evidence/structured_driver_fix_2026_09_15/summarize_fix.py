"""Summarize saved integration evidence only; never trains or reads labels."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parent
root=base/'acceptance'
load=lambda p:json.loads(p.read_text())
audit=load(root/'acceptance_audit.json')
progress=load(root/'progress.json')
assert progress['status']=='completed' and progress['attempted_tasks']==80
refs=[json.loads(line) for line in (root/'tasks.jsonl').read_text().splitlines()]
rows=[]
for item in refs:
 task=load(root/item['path']);ep=task['episode'];plans=ep['plans']
 records=load(Path(task['inputs']['artifact_root'])/'provider_records.json')['primitive_records']
 stages=load(Path(task['inputs']['artifact_root'])/'structured_audit.json')
 end=stages[-1]
 assert all(record['cost']['model_targets_computed']==0 and record['cost']['model_seconds']==0. for record in records if record['request']['tool']=='P')
 acquired=Counter(x['field_kind'] for x in end['known_acquired_remote_refs'])
 direct=Counter(x['field_kind'] for x in end['direct_remote_primary_refs'])
 paths=[np.asarray(x['output']['waypoints']) for x in plans if x.get('output')]
 acceleration=[]
 for path in paths:
  velocity=np.diff(np.vstack([np.zeros((1,2)),path]),axis=0)/.5
  acceleration.append(float(np.linalg.norm(np.diff(velocity,axis=0)/.5,axis=1).max()))
 remote_forecasts=[x for x in end['forecast_coverage'] if x['source_role']=='remote']
 rows.append(dict(g=task['row']['g'],arm=task['arm'],status=task['status'],
  requests=[x['tool'] for x in ep['requests']],driver_attempts=len(plans),
  all_output_valid=all(x['output_valid'] for x in stages),
  acquired_fields_by_kind=dict(acquired),direct_fields_by_kind=dict(direct),
  receiver_derived_fields=end['counts']['known_receiver_derived_remote_refs'],
  remote_forecast_sets=len(remote_forecasts),
  all_remote_forecasts_full_context=all(x['context_scope']=='provider_full_at_t' for x in remote_forecasts),
  remote_forecast_sets_model_used=sum(x['model_used'] for x in remote_forecasts),
  revisions_changed=sum(x['output_changed_from_previous'] is True for x in stages),
  max_acceleration_mps2=max(acceleration),
  provider_model_targets=sum(x['cost']['model_targets_computed'] for x in records),
  request_bytes=sum(x['cost']['request_bytes'] for x in records),
  response_bytes=sum(x['cost']['response_bytes'] for x in records),
  wall_seconds=task['wall_seconds'],provider_completed=sum(x['status']=='completed' for x in records)))
by_arm={}
for arm in ('Ego','P','F','PF'):
 selected=[x for x in rows if x['arm']==arm]
 acquired=Counter();direct=Counter();requests=Counter()
 for row in selected:
  acquired.update(row['acquired_fields_by_kind']);direct.update(row['direct_fields_by_kind']);requests.update(row['requests'])
 by_arm[arm]=dict(tasks=len(selected),completed=sum(x['status']=='completed' for x in selected),
  driver_attempts=sum(x['driver_attempts'] for x in selected),requests=dict(requests),
  tasks_with_direct_remote_fields=sum(bool(x['direct_fields_by_kind']) for x in selected),
  acquired_fields_by_kind=dict(acquired),direct_fields_by_kind=dict(direct),
  provider_model_targets=sum(x['provider_model_targets'] for x in selected),
  request_bytes=sum(x['request_bytes'] for x in selected),
  response_bytes=sum(x['response_bytes'] for x in selected),
  revisions_changed=sum(x['revisions_changed'] for x in selected))
summary=dict(version='toolv2x_structured_driver_fix_result_v1',status='bounded_acceptance_finished',
 initialization='seed7 untrained structured_planner_network v2',optimizer_steps=0,
 real_training_started=False,full_collection_started=False,
 initial_valid_frames=audit['initial_valid_frames'],passing_tasks=audit['passing_tasks'],
 audit_failures=audit['audit_failures'],pass_gate=audit['pass_gate'],
 all_four_arm_initial_outputs_equal=all(x['initial_outputs_equal'] for x in audit['shared_inputs']),
 wall_seconds=progress['wall_seconds'],by_arm=by_arm,rows=rows,
 source_comparison=load(base/'source_comparison.json'),
 legacy_tensor_status='Raw real-angle ufunc variation reproduced; original failure locals not captured; 300 strict replays did not fail. v1 encoding retained; v2 uses canonical scalar trigonometry.',
 conclusion_boundary='Integration and evidence consumption only; fixed policies and untrained driver do not establish causal P/F benefit or sequential value.')
(base/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ('initial_valid_frames','passing_tasks','audit_failures','pass_gate','by_arm','wall_seconds')},indent=2))
