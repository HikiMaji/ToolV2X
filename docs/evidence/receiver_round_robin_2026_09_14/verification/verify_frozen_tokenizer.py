import csv,json
from pathlib import Path
from transformers import AutoTokenizer
from planning.context import build_task_plan_input
main=Path('/root/autodl-tmp/ToolV2X')
tokenizer=AutoTokenizer.from_pretrained(str(main/'models/llava-v1.5-7b'),use_fast=False,local_files_only=True)
tokenizer.model_max_length=4096
sim=list(csv.DictReader((main/'docs/evidence/t9_admission_audit_2026_09_13/remote_units_counterfactual.csv').open()))
checks=[]
for path in sorted((main/'outputs/t9_real_smoke_2026_09_13_v1/tasks').glob('*.json')):
 ep=json.loads(path.read_text())['episode']
 for i,plan in enumerate(ep['plans']):
  ledger=ep['ledger_snapshots'][plan['ledger_snapshot']];saved=plan['prepared']
  old=build_task_plan_input(tokenizer,ep['ego_motion'],ledger,ep['feature_tokens'],limits=ep['limits']['receiver_spec'])
  for name in ('input_layout','decoding','ego_motion','evidence_used','remote_evidence_used','receiver_spec','admission_report','evidence_selection','q9_prompt'):
   assert old[name]==saved[name],(path.stem,i,name)
  new=build_task_plan_input(tokenizer,ep['ego_motion'],ledger,ep['feature_tokens'],limits=dict(ep['limits']['receiver_spec'],version='toolv2x_receiver_v2'))
  assert new['evidence_used']==old['evidence_used']
  rows=sorted([r for r in sim if r['task']==path.stem and int(r['plan_index'])==i],key=lambda r:int(r['simulated_order']))
  selected=[r for r in rows if r['admitted']=='True']
  groups=new['admission_report']['field_groups']
  assert len(groups)==len(selected)
  for group,row in zip(groups,selected):
   assert group['primary_refs']==json.loads(row['primary_refs']) and group['anchor_ref']==json.loads(row['anchor_ref'])
   assert group['object_index']==int(row['object_index'])
  expected=int(rows[-1]['cumulative_input_tokens']) if rows else saved['evidence_selection']['input_tokens']
  assert new['evidence_selection']['input_tokens']==expected
  checks.append(dict(task=path.stem,stage=i,old_input_tokens=old['evidence_selection']['input_tokens'],new_input_tokens=expected,old_targets=sorted({g['anchor_ref']['track_handle'] for g in old['admission_report']['field_groups']}),new_targets=sorted({g['anchor_ref']['track_handle'] for g in groups}),v1_exact_saved_match=True,v2_exact_simulation_match=True))
assert len(checks)==12
Path('/tmp/toolv2x-receiver-tokenizer-check.json').write_text(json.dumps(dict(status='PASS',stages=checks,model_generation=False),indent=2)+'\n')
print('PASS:12 saved stages exact v1 prompt/Z/token regression; new v2 refs/order/positions/tokens equal frozen counterfactual; local block unchanged.')
