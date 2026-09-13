"""Independent one-shot supervision: real first-return forks and audited terminals.

Online collection has no label argument. Offline labelling alone reads GT. This
is control infrastructure; a fixture runtime never establishes model performance.
"""
import copy
import json
import os
from pathlib import Path

from common.audit_protocol import recording
from planning.method_controls import normalize_control, capture_control_prefix
from planning.method_episode import decision_state, run_task_episode, validate_limits
from planning.query_data import (_online_rows, _semantic_binding, _write_rows, _read,
    _check_episode, _labels, validate_utility_spec)
from planning.run_framework import _save_method_json, read_jsonl
from tools.control_bundle import capture_bundle_prefix, decode_bundle_response
from tools.task_spec import _check_json_native, validate_provenance
from probe.kinematic_tools import encode


def bundle_collection_spec(value):
    value=copy.deepcopy(value);_check_json_native(value)
    if set(value)!={'version','limits','local_provenance','control','recording_folds'} or value['version']!='toolv2x_bundle_collection_v1':
        raise ValueError('complete versioned bundle collection specification required')
    value['limits']=validate_limits(value['limits']);validate_provenance(value['local_provenance'])
    control=value['control']=normalize_control(value['control'])
    if (control is None or control['name']!='one_shot' or control['driver_calls']!=2 or
            control['bundle']['extra_generation'] is not None):
        raise ValueError('bundle supervision requires initial and final driver only')
    folds=value['recording_folds']
    if not isinstance(folds,dict) or not folds or any(recording(k)!=k or not isinstance(v,str) or not v for k,v in folds.items()):
        raise ValueError('folds must map physical recordings to explicit fold names')
    return value


def _action_key(action):
    return (action['tool'],action['mode'],action['candidate_id'])


def collect_bundle_branches(online_index, out, runtime, controls_spec):
    """Enumerate legal first P/F and independent STOP/continuation terminals.

    One initial GoT is physically shared. For each first tool, the real provider
    first return is captured once; siblings fork only this state. Each suffix and
    final GoT really executes. Deployed charges include all retained prefix costs.
    """
    from planning.query_value import _bundle_state
    spec=bundle_collection_spec(controls_spec)
    rows=_online_rows(read_jsonl(online_index) if isinstance(online_index,(str,Path)) else online_index,spec)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in ('tasks','prefixes','inputs'):(out/name).mkdir()
    progress=dict(status='initializing',expected_samples=len(rows),completed_samples=0,failed_samples=0,
        physical_driver_attempts=0,physical_first_primitives=0)
    _save_method_json(out/'progress.json',progress)
    runtime=runtime() if callable(runtime) else runtime
    runtime_binding=dict(driver=copy.deepcopy(runtime.driver.provenance),predictor=copy.deepcopy(runtime.predictor.descriptor))
    config=dict(version='toolv2x_bundle_branches_v1',spec=spec,binding=_semantic_binding(runtime_binding,spec),
        runtime_binding=runtime_binding,runtime=copy.deepcopy(runtime.provenance),gt_labels_read=False,
        reuse='same live initial driver and actual first provider return; independent suffixes')
    _save_method_json(out/'config.json',config);_write_rows(out/'selected_index.jsonl',rows)
    states,branches,initials=[],[],[]
    def save():
        _write_rows(out/'online_states.jsonl',states);_write_rows(out/'branches.jsonl',branches)
        _write_rows(out/'initials.jsonl',initials);_save_method_json(out/'progress.json',progress)
    progress['status']='running';save()
    for index,row in enumerate(rows):
        if runtime.driver.provenance!=runtime_binding['driver'] or runtime.predictor.descriptor!=runtime_binding['predictor']:
            raise ValueError('collection runtime changed between samples')
        sample='s%06d'%index;directory=out/'inputs'/sample;directory.mkdir()
        entry=dict(sample_id=row['sample_id'],path='prefixes/'+sample+'.json',status='started',first_actions=[])
        initials.append(entry)
        task=dict(row=copy.deepcopy(row),status='started',episode=None)
        try:
            inputs=runtime.load_inputs(copy.deepcopy(row),directory);metadata=inputs.pop('metadata')
            task['inputs']=dict(copy.deepcopy(metadata),artifact_root=str(directory))
        except Exception as exc:
            task.update(status='failed',error=dict(stage='local_preparation',type=type(exc).__name__,message=str(exc)))
            entry['status']='failed';_save_method_json(out/entry['path'],task)
            progress['failed_samples']+=1;progress['completed_samples']+=1;save();continue
        common=dict(inputs,predictor=runtime.predictor,driver=runtime.driver,limits=spec['limits'],
            local_provenance=spec['local_provenance'],sample_id=row['sample_id'],control_spec=spec['control'])
        def persist(ep):
            task.update(episode=ep,status='running');_save_method_json(out/entry['path'],task)
        try:
            initial=capture_control_prefix(**common,after_calls=0,branch_id=sample,
                policy=lambda state:dict(tool='STOP',mode=None,reason='capture_initial'),on_progress=persist)
        except ValueError:
            if task['episode'] is None or task['episode']['status']=='running':raise
            initial=None
        progress['physical_driver_attempts']+=len((task.get('episode') or {}).get('plans',[]))
        if initial is None:
            task['status']=entry['status']='failed';progress['failed_samples']+=1
            _save_method_json(out/entry['path'],task)
        else:
            task['status']=entry['status']='prefix_ready';_save_method_json(out/entry['path'],task)
            _,envelopes,preflight=decision_state(initial['episode'])
            for check in preflight:
                action=check['action'];state_id=sample+'__'+action['tool']
                first=dict(action=action,feasible=check['feasible'],status='infeasible',state_id=None,path=None,reason=check['reason'])
                entry['first_actions'].append(first)
                if not check['feasible']:continue
                env=envelopes[(action['tool'],action['mode'])]
                service=inputs['service'].fork_task()
                service.register_bundle_policy(env['continuation_policy_id'],lambda state:dict(tool='STOP',mode=None,candidate_id=None,reason='capture'))
                read_start=len(metadata.get('window_reads',[]))
                provider_path='prefixes/'+state_id+'_provider.json';first['path']=provider_path
                try:
                    prefix=capture_bundle_prefix(service,env)
                except Exception as exc:
                    first.update(status='failed',reason='first_primitive_failed')
                    _save_method_json(out/provider_path,dict(request=env,records=service.bundle_records,
                        error=dict(type=type(exc).__name__,message=str(exc))))
                    progress['physical_first_primitives']+=len(service.task_records);save();continue
                progress['physical_first_primitives']+=1
                prefix_inputs=copy.deepcopy(task['inputs'])
                if 'window_reads' in metadata:
                    prefix_inputs['window_reads']+=metadata['window_reads'][read_start:]
                state=prefix.visible_state;feasible=_bundle_state(state)[2]
                _save_method_json(out/provider_path,dict(record=prefix.record,inputs=prefix_inputs))
                first.update(status='prefix_ready',state_id=state_id)
                states.append(dict(state_id=state_id,sample_id=row['sample_id'],physical_recording=recording(row['scene']),
                    fold=spec['recording_folds'][recording(row['scene'])],initial_path=entry['path'],provider_prefix_path=provider_path,state=state))
                for action_index,continuation in enumerate(state['available_actions']):
                    branch_id=state_id+'__a%03d'%action_index
                    branch=dict(state_id=state_id,branch_id=branch_id,action=copy.deepcopy(continuation),
                        feasible=continuation in feasible,status='pending' if continuation in feasible else 'infeasible',path=None)
                    branches.append(branch)
                    if not branch['feasible']:continue
                    branch['path']='tasks/'+branch_id+'.json'
                    terminal=dict(row=copy.deepcopy(row),inputs=copy.deepcopy(prefix_inputs),status='started',episode=None)
                    child=prefix.fork(lambda visible,a=continuation:dict(a,reason='enumerated_bundle_suffix'))
                    read_start=len(metadata.get('window_reads',[]))
                    def persist_terminal(ep):
                        terminal.update(episode=ep,status='running')
                        if 'window_reads' in metadata:
                            terminal['inputs']['window_reads']=prefix_inputs['window_reads']+metadata['window_reads'][read_start:]
                        _save_method_json(out/branch['path'],terminal)
                    ep=run_task_episode(**dict(common,service=child),prefix=initial,branch_id=branch_id,
                        policy=lambda visible,t=env['first_request']['tool']:dict(tool=t,mode='current',reason='enumerated_first_tool'),
                        on_progress=persist_terminal)
                    terminal['status']=branch['status']='completed' if ep['status']=='completed' else 'failed'
                    progress['physical_driver_attempts']+=len(ep['plans'])-len(initial['episode']['plans'])
                    _save_method_json(out/branch['path'],terminal);save()
        progress['completed_samples']+=1;save()
    progress['status']='completed';save()
    return progress


def validate_bundle_archive(root):
    """Rebind sent candidates, real receipts, exact prefix, driver Z and terminal costs."""
    from planning.query_value import _bundle_state
    from evaluation.framework import _method_cost
    root=Path(root);config=_read(root,'config.json');spec=bundle_collection_spec(config['spec'])
    progress=_read(root,'progress.json');rows=_online_rows(read_jsonl(root/'selected_index.jsonl'),spec)
    if (config['version']!='toolv2x_bundle_branches_v1' or config['gt_labels_read'] is not False or
            config['binding']!=_semantic_binding(config['runtime_binding'],spec) or
            progress['status']!='completed' or progress['expected_samples']!=len(rows) or progress['completed_samples']!=len(rows)):
        raise ValueError('incomplete or rebound bundle archive')
    initials=read_jsonl(root/'initials.jsonl');states=read_jsonl(root/'online_states.jsonl');branches=read_jsonl(root/'branches.jsonl')
    identities={r['sample_id']:r for r in rows};state_index={s['state_id']:s for s in states}
    if (len(initials)!=len(rows) or {r['sample_id'] for r in initials}!=set(identities) or len(state_index)!=len(states) or
            len({b['branch_id'] for b in branches})!=len(branches)):
        raise ValueError('missing or duplicated bundle sample/state/branch')
    referenced=[]
    for entry in initials:
        initial=_read(root,entry['path'])
        if initial['row']!=identities[entry['sample_id']]:raise ValueError('initial sample identity drift')
        if entry['status']=='failed':
            if initial['status']!='failed' or entry['first_actions'] or not (initial.get('error') or (initial.get('episode') or {}).get('error')):
                raise ValueError('invalid initial failure archive')
            continue
        _check_episode(initial,config);ep=initial['episode']
        if entry['status']!='prefix_ready' or ep['requests'] or len(ep['plans'])!=1 or ep['events'][-1]['kind']!='driver_completed':
            raise ValueError('bundle source must be the actual initial GoT only')
        _,envelopes,preflight=decision_state(ep)
        if len(entry['first_actions'])!=len(preflight):raise ValueError('missing first-tool candidate')
        for first,check in zip(entry['first_actions'],preflight):
            if first['action']!=check['action'] or first['feasible']!=check['feasible']:
                raise ValueError('first-tool public feasibility changed')
            if not first['feasible']:
                if first['status']!='infeasible' or first['path'] is not None or first['state_id'] is not None:
                    raise ValueError('infeasible first tool has an outcome')
                continue
            env=envelopes[(first['action']['tool'],first['action']['mode'])]
            saved=_read(root,first['path'])
            if first['status']=='failed':
                if first['state_id'] is not None or saved.get('request')!=env or not saved.get('error'):
                    raise ValueError('first primitive failure lacks its actual record')
                continue
            state=state_index.get(first['state_id']);referenced.append(first['state_id'])
            if state is None or state['initial_path']!=entry['path'] or state['provider_prefix_path']!=first['path']:
                raise ValueError('missing or rebound first-return state')
            actual_env,packet,feasible=_bundle_state(state['state']);record=saved['record']
            if (actual_env!=env or state['sample_id']!=entry['sample_id'] or
                    state['physical_recording']!=recording(initial['row']['scene']) or
                    state['fold']!=spec['recording_folds'][state['physical_recording']] or
                    record['request']!=env or bytes.fromhex(record['request_wire_hex'])!=encode(env) or
                    record['status']!='prefix_ready' or len(record['primitive_responses'])!=1 or len(record['primitive_records'])!=1 or
                    json.loads(bytes.fromhex(record['primitive_responses'][0]['wire_hex']))!=packet or
                    record['primitive_records'][0]['response']!=packet):
                raise ValueError('conditional source differs from the actual sent first return')
            own=[b for b in branches if b['state_id']==state['state_id']]
            expected=state['state']['available_actions']
            if len(own)!=len(expected) or [_action_key(b['action']) for b in own]!=[_action_key(a) for a in expected]:
                raise ValueError('missing or reordered conditional candidate coverage')
            for branch in own:
                if branch['feasible']!=(branch['action'] in feasible):raise ValueError('conditional feasibility changed')
                if not branch['feasible']:
                    if branch['status']!='infeasible' or branch['path'] is not None:raise ValueError('infeasible branch has a terminal')
                    continue
                terminal=_read(root,branch['path']);_check_episode(terminal,config);tail=terminal['episode']
                if (terminal['row']!=initial['row'] or terminal['status']!=branch['status'] or tail['branch_id']!=branch['branch_id'] or
                        tail['requests']!=[env] or branch['status'] not in ('completed','failed')):
                    raise ValueError('bundle terminal identity/request/status mismatch')
                for key in ('plans','responses','ledger_snapshots','cost_events'):
                    if tail[key][:len(ep[key])]!=ep[key]:raise ValueError('terminal changed its actual initial prefix')
                event=next(e for e in tail['cost_events'] if e['kind']=='service')
                reuse=dict(kind='actual_bundle_first_return',prefix_service_seconds=record['cost']['service_seconds'],
                    suffix_service_seconds=event['service_cost']['service_seconds'])
                if event.get('collection_reuse')!=reuse:raise ValueError('retained first-return deployment cost changed')
                if tail['responses']:
                    response=tail['responses'][0]
                    decoded=decode_bundle_response(bytes.fromhex(response['wire_hex']),env)
                    actual=decoded['packet']
                    if (actual['responses'][0]['packet']!=packet or actual['responses'][0]['cost']!=record['primitive_responses'][0]['cost'] or
                            _action_key(decoded['continuation'])!=_action_key(branch['action']) or response.get('collection_reuse')!=reuse):
                        raise ValueError('terminal action or first receipt differs from its source')
                    if tail['status']=='completed' and len(tail['plans'])!=2:
                        raise ValueError('one-shot requires exactly one actual final driver')
                elif tail['status']!='service_error':raise ValueError('missing actual bundle transport')
                cost=_method_cost(terminal)
                if cost['cost_issues']:raise ValueError('inconsistent raw terminal costs: '+str(cost['cost_issues']))
    if (len(referenced)!=len(states) or set(referenced)!=set(state_index) or
            any(b['state_id'] not in state_index for b in branches)):
        raise ValueError('orphan or duplicated bundle state')
    return config,states,branches


def _terminal(task,label,utility):
    from evaluation.framework import evaluate_method_task
    result=evaluate_method_task(task,label)
    if result['label_status']!='complete':return None,'labels_'+result['label_status']
    if result['artifact_status'] not in ('completed','recorded_failure'):
        raise ValueError('terminal raw driver artifact is inconsistent')
    if any(result[k] is None for k in utility['cost_weights']) or result['cost_issues']:
        return None,'unknown_terminal_cost'
    loss=(sum(utility['quality_weights'][k]*result[k] for k in ('ADE3','FDE3'))
          if result['task_success'] else utility['failure_loss'])
    return (loss,{k:result[k] for k in utility['cost_weights']}),None


def _build_targets(root,labels,utility,*,train_recordings=()):
    config,states,branches=validate_bundle_archive(root);rows=[];excluded=[]
    # validate_bundle_archive already binds every state's sample, recording and
    # fold to this original index and its actual initial/terminal task rows.
    for identity in read_jsonl(root/'selected_index.jsonl'):
        if recording(identity['scene']) in train_recordings and identity['role']!='train':
            raise ValueError('validation recording cannot train a bundle continuation')
    for state in states:
        own=[b for b in branches if b['state_id']==state['state_id'] and b['feasible']]
        terminals={};reason=None
        for branch in own:
            task=_read(root,branch['path']);label=labels.get((task['row']['sample_id'],task['row']['role']))
            terminal,problem=_terminal(task,label,utility)
            if problem:reason=problem
            terminals[_action_key(branch['action'])]=terminal
        if reason:
            excluded.append(dict(state_id=state['state_id'],reason=reason));continue
        stop_loss,stop_cost=terminals[('STOP',None,None)]
        for branch in own:
            loss,cost=terminals[_action_key(branch['action'])]
            delta={k:cost[k]-stop_cost[k] for k in stop_cost}
            target=stop_loss-loss-sum(utility['cost_weights'][k]*delta[k] for k in delta)
            rows.append(dict(state_id=state['state_id'],physical_recording=state['physical_recording'],fold=state['fold'],
                state=copy.deepcopy(state['state']),action=copy.deepcopy(branch['action']),stop_loss=stop_loss,
                terminal_loss=loss,incremental_cost=delta,target=target))
    table=dict(version='toolv2x_bundle_targets_v2',kind='bundle_terminal',binding=config['binding'],utility_spec=utility,
        recording_folds=config['spec']['recording_folds'],rows=rows,supervision=dict(version='toolv2x_bundle_supervision_v1',
        origin='measured_bundle_branches',stop_reference='first_return_then_final_driver',cost_scope='outer_wire_and_complete_terminal_compute'))
    coverage=dict(expected_samples=_read(root,'progress.json')['expected_samples'],source_states=len(states),
        target_rows=len(rows),excluded_states=excluded,initials=read_jsonl(root/'initials.jsonl'),runtime=config['runtime'])
    return table,coverage


def make_bundle_targets(branch_dir, labels_root, utility_spec, out):
    """Separate offline labels, bound to STOP's real final driver and full outer cost."""
    root=Path(branch_dir);utility=validate_utility_spec(utility_spec)
    labels=_labels(labels_root,{r['role'] for r in read_jsonl(root/'selected_index.jsonl')})
    table,coverage=_build_targets(root,labels,utility)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _write_rows(out/'labels.jsonl',list(labels.values()))
    table['archive']=dict(version='toolv2x_bundle_target_archive_v1',branches=os.path.relpath(root.resolve(),out.resolve()),labels='labels.jsonl')
    _save_method_json(out/'targets.json',table);_save_method_json(out/'coverage.json',coverage)
    return table


def load_measured_bundle_targets(path,*,train_recordings=()):
    """Training reloads raw sources and independently recomputes every target."""
    path=Path(path);table=json.loads(path.read_text());source=table.get('archive',{})
    if (set(source)!={'version','branches','labels'} or source['version']!='toolv2x_bundle_target_archive_v1' or
            Path(source['branches']).is_absolute() or Path(source['labels']).is_absolute()):
        raise ValueError('measured bundle table requires portable raw source references')
    labels={}
    for label in read_jsonl(path.parent/source['labels']):
        key=(label['sample_id'],label['role'])
        if key in labels:raise ValueError('duplicate offline label identity')
        labels[key]=label
    expected,_=_build_targets(path.parent/source['branches'],labels,validate_utility_spec(table['utility_spec']),
        train_recordings=train_recordings)
    expected['archive']=source
    if table!=expected:raise ValueError('bundle labels differ from their actual archived terminals')
    return table
