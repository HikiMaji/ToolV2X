"""Audit real saved coverage and evidence, without labels or model access."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from evaluation.structured import audit_structured_episode
from tools.task_spec import validate_plan

EXPECTED=dict(Ego=[],P=['P'],F=['F'],PF=['P','F'])
def check(root):
    root=Path(root)
    progress=json.loads((root/'progress.json').read_text())
    assert progress['status']=='completed' and progress['attempted_tasks']==80
    refs=[json.loads(line) for line in (root/'tasks.jsonl').read_text().splitlines()]
    assert len(refs)==80 and len({(r['sample_id'],r['arm']) for r in refs})==80
    rows=[];by_frame={}
    for ref in refs:
        task=json.loads((root/ref['path']).read_text());ep=task.get('episode') or {}
        errors=[];audits=None
        try:
            audits=audit_structured_episode(ep)
        except Exception as exc:errors.append('audit: '+str(exc))
        plans=ep.get('plans',[]);initial_valid=False
        try:
            validate_plan(plans[0]['output']['waypoints'],ep['limits']['execution_spec'])
            initial_valid=plans[0]['status']=='valid'
        except Exception:pass
        inputs=task.get('inputs',{});provider_path=Path(inputs.get('artifact_root',''))/'provider_records.json'
        provider=json.loads(provider_path.read_text()) if provider_path.is_file() else {}
        requests=ep.get('requests',[]);tools=[r['tool'] for r in requests]
        records=provider.get('primitive_records',[])
        if [record.get('request') for record in records]!=requests:
            errors.append('provider records differ from actual issued requests')
        if any(record.get('status')!='completed' for record in records):
            errors.append('provider primitive failure')
        if not requests and (provider.get('peer_context_loaded') or
                any(read['source']!='no_fusion' for read in inputs.get('window_reads',[]))):
            errors.append('peer access without issued request')
        if any(a['counts']['known_receiver_derived_remote_refs'] for a in audits or []):
            errors.append('observations-only unexpectedly derived remote forecasts')
        if audits and audits[0]['counts']['known_remote_refs']:
            errors.append('remote fields before purchase')
        stop=ep.get('stop_reason')
        budget_empty=stop in ('byte_budget_exhausted','request_byte_limit')
        acquired_boundary=(tools==EXPECTED[ref['arm']] or budget_empty)
        passed=(task['status']=='completed' and initial_valid and acquired_boundary and not errors)
        row=dict(g=task['row']['g'],arm=ref['arm'],task_status=task['status'],
            initial_valid=initial_valid,requests=tools,stop_reason=stop,budget_empty=budget_empty,
            acquisition_boundary=acquired_boundary,passed=passed,errors=errors,
            driver_attempts=len(plans),wall_seconds=task.get('wall_seconds'),
            acquired_fields=(audits[-1]['counts']['known_acquired_remote_refs'] if audits else None),
            direct_remote_fields=(audits[-1]['counts']['direct_remote_primary_refs'] if audits else None),
            empty_service=sum(not record.get('response',{}).get('records',[]) for record in records),
            episode_error=ep.get('error'))
        rows.append(row);by_frame.setdefault(task['row']['sample_id'],[]).append(task)
    shared=[]
    for key,tasks in by_frame.items():
        baseline=next(t for t in tasks if t['arm']=='Ego')
        feature_equal=True;forecast_equal=True;initial_equal=True
        for task in tasks:
            for name in ('ego_features.npz','local_forecast.npz'):
                try:
                    with np.load(Path(baseline['inputs']['artifact_root'])/name) as a,np.load(Path(task['inputs']['artifact_root'])/name) as b:
                        equal=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
                except Exception:equal=False
                if name=='ego_features.npz':feature_equal &= equal
                else:forecast_equal &= equal
            a=(baseline.get('episode') or {}).get('plans',[]);b=(task.get('episode') or {}).get('plans',[])
            initial_equal &= bool(a and b) and (a[0].get('output') or {}).get('waypoints')==(b[0].get('output') or {}).get('waypoints')
        shared.append(dict(sample_id=key,features_equal=feature_equal,local_forecasts_equal=forecast_equal,
            initial_outputs_equal=initial_equal))
    result=dict(expected_tasks=80,rows=rows,shared_inputs=shared,
        initial_valid_frames=sum(x['initial_valid'] for x in rows if x['arm']=='Ego'),
        passing_tasks=sum(x['passed'] for x in rows),
        audit_failures=sum(bool(x['errors']) for x in rows),
        pass_gate=all(x['passed'] for x in rows) and all(x['features_equal'] and x['local_forecasts_equal'] for x in shared),
        scope='real integration and evidence coverage only; zero returned fields are not consumption success')
    target=root/'acceptance_audit.json'
    assert not target.exists()
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('initial_valid_frames','passing_tasks','audit_failures','pass_gate')}),flush=True)
    return result

if __name__=='__main__':check(sys.argv[1])
