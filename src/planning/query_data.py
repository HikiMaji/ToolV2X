"""Finite causal branch collection and separate offline supervision; no training."""
import copy
import json
import math
from pathlib import Path

from common.audit_protocol import recording
from planning.method_controls import normalize_control
from planning.method_episode import run_task_episode, validate_limits
from planning.run_framework import _save_method_json, read_jsonl
from tools.task_spec import _check_json_native, validate_provenance, _version, _version_pair


def _write_rows(path, rows):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    temporary.replace(path)


def collection_spec(value):
    """Freeze one branch grammar and recording folds before any model access."""
    value=copy.deepcopy(value)
    _check_json_native(value)
    if set(value)!={'version','limits','local_provenance','control','recording_folds'} or value['version']!='toolv2x_query_collection_v1':
        raise ValueError('complete versioned collection specification required')
    value['limits']=validate_limits(value['limits'])
    validate_provenance(value['local_provenance'])
    value['control']=normalize_control(value['control'])
    if value['control'] is None or value['control']['name'] not in (
            'feedback','frozen_feedback','current_old','current_union','evidence_refinement'):
        raise ValueError('collection requires an alternating task control, not a fixed-generation or bundle graph')
    folds=value['recording_folds']
    if not isinstance(folds,dict) or not folds or any(recording(k)!=k or not isinstance(v,str) or not v for k,v in folds.items()):
        raise ValueError('folds must map whole physical recordings to explicit fold names')
    return value


def _online_rows(rows, spec):
    rows=copy.deepcopy(list(rows))
    required={'sample_id','scene','g','role','physical_split','local_frame','ego_motion','feature_read_paths','motion_read_paths'}
    if not rows or any(set(r)!=required for r in rows):
        raise ValueError('collection accepts only the causal online index schema')
    if len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate sample identity')
    roles={}
    for row in rows:
        _check_json_native(row)
        group=recording(row['scene'])
        if group not in spec['recording_folds'] or row['role'] not in ('train','validation'):
            raise ValueError('missing physical recording fold or unsupported research role')
        if group in roles and roles[group]!=row['role']:
            raise ValueError('one physical recording cannot cross research roles')
        roles[group]=row['role']
        if any(type(row[k]) is not int or row[k]<0 for k in ('g','local_frame')):
            raise ValueError('invalid causal frame identity')
    return rows


def _action_id(action):
    return action['tool']+('_'+action['mode'] if action['mode'] is not None else '')


def _semantic_binding(runtime_binding,spec):
    """Versioned model/settings identity, separate from load paths and run handles."""
    driver=runtime_binding['driver']
    settings={k:copy.deepcopy(driver[k]) for k in ('model_class','decoding','my_model_config','context_limit',
        'evidence_format','released_tokenizer_limit','attention_implementation','actual_rgb_input',
        'point_cloud_feature_input') if k in driver}
    predictor=runtime_binding['predictor']
    return dict(state_version='toolv2x_query_state_v1',driver=dict(version=spec['limits']['driver_version'],settings=settings),
        predictor={k:copy.deepcopy(predictor[k]) for k in ('model_version','settings')},
        limits=copy.deepcopy(spec['limits']),control=copy.deepcopy(spec['control']),local_provenance=copy.deepcopy(spec['local_provenance']))


def collect_branches(online_index, out, runtime, controls_spec):
    """Collect actual legal counterfactuals; never open labels or expose siblings.

    runtime has driver, predictor, load_inputs(row, directory), provenance;
    a zero-argument factory can initialize it after archive creation.
    Shared prefixes retain their deployed costs. Only collection work avoids
    rerunning them. All action suffixes execute; no cross-sibling result cache.
    """
    spec=collection_spec(controls_spec)
    rows=_online_rows(read_jsonl(online_index) if isinstance(online_index,(str,Path)) else online_index,spec)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in ('tasks','prefixes','inputs'):(out/name).mkdir()
    _save_method_json(out/'progress.json',dict(status='initializing',expected_samples=len(rows)))
    runtime=runtime() if callable(runtime) else runtime
    runtime_binding=dict(driver=copy.deepcopy(runtime.driver.provenance),predictor=copy.deepcopy(runtime.predictor.descriptor))
    binding=_semantic_binding(runtime_binding,spec)
    _save_method_json(out/'config.json',dict(version='toolv2x_query_branches_v1',spec=spec,binding=binding,
        runtime_binding=runtime_binding,runtime=runtime.provenance,gt_labels_read=False,
        reuse='actual prefixes only; no equivalent sibling memoization'))
    _write_rows(out/'selected_index.jsonl',rows)
    states,branches=[],[]
    progress=dict(status='running',expected_samples=len(rows),completed_samples=0,failed_samples=0,physical_driver_attempts=0)

    def save():
        _write_rows(out/'online_states.jsonl',states)
        _write_rows(out/'branches.jsonl',branches)
        _save_method_json(out/'progress.json',progress)
    save()

    class Boundary(Exception):
        pass

    for index,row in enumerate(rows):
        if runtime.driver.provenance!=runtime_binding['driver'] or runtime.predictor.descriptor!=runtime_binding['predictor']:
            raise ValueError('collection runtime changed between samples')
        sample='s%06d'%index
        directory=out/'inputs'/sample;directory.mkdir()
        group=recording(row['scene']);fold=spec['recording_folds'][group]
        try:
            inputs=runtime.load_inputs(copy.deepcopy(row),directory)
            metadata=inputs.pop('metadata')
        except Exception as exc:
            path='tasks/'+sample+'_preparation.json'
            _save_method_json(out/path,dict(row=row,status='failed',episode=None,
                error=dict(stage='local_preparation',type=type(exc).__name__,message=str(exc))))
            branches.append(dict(branch_id=sample+'_preparation',state_id=None,sample_id=row['sample_id'],
                physical_recording=group,fold=fold,action=None,feasible=True,status='failed',reason='local_preparation',
                path=path,next_state_id=None,reuse=None))
            progress['failed_samples']+=1;progress['completed_samples']+=1;save()
            continue
        common=dict(inputs,predictor=runtime.predictor,driver=runtime.driver,limits=spec['limits'],
            local_provenance=spec['local_provenance'],sample_id=row['sample_id'],control_spec=spec['control'])
        initial_metadata=copy.deepcopy(metadata)

        def execute(branch_id, service, prefix=None, action=None, capture_after=None, prefix_inputs=None):
            path='tasks/'+branch_id+'.json'
            source_inputs=copy.deepcopy(initial_metadata if prefix_inputs is None else prefix_inputs)
            read_start=len(metadata.get('window_reads',[]))
            task=dict(row=copy.deepcopy(row),inputs=dict(source_inputs,artifact_root=str(directory)),status='started',episode=None)
            def update_reads():
                if 'window_reads' in metadata:
                    task['inputs']['window_reads']=copy.deepcopy(source_inputs['window_reads']+metadata['window_reads'][read_start:])
            _save_method_json(out/path,task)
            before=len(prefix['episode']['plans']) if prefix else 0
            called=False
            def policy(state):
                nonlocal called
                chosen=action if action is not None and not called else dict(tool='STOP',mode=None)
                called=True
                return dict(chosen,reason='enumerated_legal_branch')
            def persist(ep):
                task.update(episode=ep,status='running')
                update_reads()
                _save_method_json(out/path,task)
                if (capture_after is not None and ep['events'][-1]['kind']=='driver_completed' and
                        len(ep['requests'])==capture_after and ep['plans'][-1]['status']=='valid'):
                    raise Boundary()
            captured=None
            try:
                ep=run_task_episode(**dict(common,service=service),policy=policy,prefix=prefix,
                    branch_id=branch_id,on_progress=persist)
                task.update(episode=ep,status='completed' if ep['status']=='completed' else 'failed')
            except Boundary:
                task['status']='prefix_ready'
                captured=dict(episode=copy.deepcopy(task['episode']),features=copy.deepcopy(inputs['features']),driver=runtime.driver)
            finally:
                progress['physical_driver_attempts']+=max(0,len((task.get('episode') or {}).get('plans',[]))-before)
                update_reads()
                _save_method_json(out/path,task)
                save()
            return path,task,captured

        def expand(state_id, prefix, service, depth, prefix_inputs):
            prefix_path='prefixes/'+state_id+'.json'
            _save_method_json(out/prefix_path,dict(row=row,inputs=prefix_inputs,episode=prefix['episode']))
            stop_id=state_id+'__STOP'
            stop_path,stop_task,_=execute(stop_id,service.fork_task(),prefix,prefix_inputs=prefix_inputs)
            ep=stop_task['episode']
            if ep['status']!='completed':
                raise ValueError('valid source prefix could not execute STOP')
            decision=ep['decisions'][-1]
            visible=copy.deepcopy(decision['state'])
            states.append(dict(version='toolv2x_query_state_v1',state_id=state_id,sample_id=row['sample_id'],physical_recording=group,fold=fold,
                depth=depth,prefix_path=prefix_path,source_plan_id=prefix['episode']['last_valid_plan_id'],
                cost_event_count=len(prefix['episode']['cost_events']),state=visible))
            base=dict(state_id=state_id,sample_id=row['sample_id'],physical_recording=group,fold=fold,
                next_state_id=None,reuse=dict(kind='shared_actual_prefix',prefix_path=prefix_path,
                    deployed_cost='retain all recorded prefix plus actual suffix costs'))
            branches.append(dict(base,branch_id=stop_id,action=dict(tool='STOP',mode=None),feasible=True,
                status='completed',path=stop_path,reason='STOP_references_current_plan'))
            save()
            # Public preflight includes rejected actions; no unseen peer outcomes are consulted.
            for candidate in decision['request_preflight']:
                action=candidate['action'];branch_id=state_id+'__'+_action_id(action)
                branch=dict(copy.deepcopy(base),branch_id=branch_id,action=copy.deepcopy(action),
                    feasible=candidate['feasible'],status='pending' if candidate['feasible'] else 'infeasible',
                    path=None,reason=candidate['reason'])
                branches.append(branch);save()
                if not candidate['feasible']:continue
                branch_service=service.fork_task()
                path,task,child=execute(branch_id,branch_service,prefix,action,capture_after=1 if depth==0 else None,prefix_inputs=prefix_inputs)
                branch.update(path=path,status=task['status'])
                if child is not None:
                    child_id=branch_id
                    branch.update(next_state_id=child_id,status='expanding')
                    save()
                    # The action's immediate terminal is its child's STOP, using the same actual plan.
                    branch.update(path=expand(child_id,child,branch_service,depth+1,task['inputs']),status='completed')
                save()
            return stop_path

        path,task,prefix=execute(sample+'_initial',inputs['service'],capture_after=0)
        if prefix is None:
            branches.append(dict(branch_id=sample+'_initial',state_id=None,sample_id=row['sample_id'],
                physical_recording=group,fold=fold,action=None,feasible=True,status='failed',reason='initial_plan_failed',
                path=path,next_state_id=None,reuse=None))
            progress['failed_samples']+=1
        else:
            expand(sample,prefix,inputs['service'],0,task['inputs'])
        progress['completed_samples']+=1;save()
    progress['status']='completed';save()
    return progress


def validate_utility_spec(value):
    """Explicit loss units and disjoint measured costs; no fitted defaults."""
    value=copy.deepcopy(value)
    _check_json_native(value)
    if (set(value)!={'version','label_coverage','quality_weights','cost_weights','failure_loss'} or
            value['version']!='toolv2x_query_utility_v1' or value['label_coverage']!='all_six' or
            set(value['quality_weights'])!={'ADE3','FDE3'} or
            set(value['cost_weights'])!={'request_bytes','response_bytes','total_compute_seconds'}):
        raise ValueError('complete versioned utility weights and six-point coverage required')
    numbers=[value['failure_loss'],*value['quality_weights'].values(),*value['cost_weights'].values()]
    if (any(type(n) not in (int,float) or not math.isfinite(n) or n<0 for n in numbers) or
            value['failure_loss']<=0 or not any(value['quality_weights'].values())):
        raise ValueError('utility weights must be finite nonnegative numbers and failure loss positive')
    return value


def _read(root,path):
    target=(root/path).resolve()
    if root.resolve() not in target.parents:
        raise ValueError('branch artifact escapes its archive')
    return json.loads(target.read_text())


def _check_episode(task,config):
    """Require the actual driver input and its ledger; empty legal Z is valid."""
    ep=task['episode'];binding=config['binding'];actual=config['runtime_binding']
    if (ep['limits']!=binding['limits'] or ep.get('control_spec')!=binding['control'] or
            ep['driver_provenance']!=actual['driver'] or ep['predictor_binding']!=actual['predictor'] or
            any(ep[k]!=task['row'][k] for k in ('sample_id','scene','g'))):
        raise ValueError('branch model/query/receiver/identity binding changed')
    for plan in ep['plans']:
        prepared=plan.get('prepared') or {}
        if not {'evidence_used','remote_evidence_used','admission_report','q9_prompt'}<=set(prepared):
            raise ValueError('branch is missing its actual driver Z')
        snapshot=plan['ledger_snapshot']
        if type(snapshot) is not int or not 0<=snapshot<len(ep['ledger_snapshots']):
            raise ValueError('driver Z has no actual ledger snapshot')
        ledger=ep['ledger_snapshots'][snapshot];report=prepared['admission_report']
        if (report['acquired_field_refs']!=[r['ref'] for r in ledger['acquired_fields']] or
                report['derived_field_refs']!=[r['ref'] for r in ledger['derived_fields']] or
                report['observation_receipt_ids']!=[r['receipt_id'] for r in ledger['receipts']]):
            raise ValueError('driver Z and acquired/derived/receipt ledger disagree')
        if plan.get('output') is not None and any(
                plan['output'].get(k)!=prepared[k] for k in ('evidence_used','remote_evidence_used','admission_report','q9_prompt')):
            raise ValueError('saved output was not generated from the archived Z')


def _branch_archive(root):
    root=Path(root);config=_read(root,'config.json');spec=collection_spec(config['spec'])
    progress=_read(root,'progress.json')
    if config['version']!='toolv2x_query_branches_v1' or progress['status']!='completed':
        raise ValueError('branch collection is incomplete')
    if config['binding']!=_semantic_binding(config['runtime_binding'],spec):
        raise ValueError('collection specification and runtime binding disagree')
    selected=_online_rows(read_jsonl(root/'selected_index.jsonl'),spec)
    if progress['expected_samples']!=len(selected) or progress['completed_samples']!=len(selected):
        raise ValueError('branch sample coverage mismatch')
    identities={r['sample_id']:r for r in selected}
    states=read_jsonl(root/'online_states.jsonl');branches=read_jsonl(root/'branches.jsonl')
    state_index={s['state_id']:s for s in states}
    if len(state_index)!=len(states) or len({b['branch_id'] for b in branches})!=len(branches):
        raise ValueError('duplicate state or branch identity')
    if any(b['state_id'] is not None and b['state_id'] not in state_index for b in branches):
        raise ValueError('orphan branch has no source state')
    for record in states+branches:
        row=identities[record['sample_id']];group=recording(row['scene'])
        if record['physical_recording']!=group or record['fold']!=spec['recording_folds'][group]:
            raise ValueError('a physical recording crossed collection folds')
    edges={}
    for state in states:
        prefix=_read(root,state['prefix_path']);_check_episode(prefix,config)
        ep=prefix['episode']
        if (state.get('version')!=config['binding']['state_version'] or
                prefix['row']!=identities[state['sample_id']] or state['depth'] not in (0,1) or
                len(ep['requests'])!=state['depth'] or len(ep['plans'])!=state['depth']+1 or
                ep['status']!='running' or ep['events'][-1]['kind']!='driver_completed' or
                any(p['status']!='valid' for p in ep['plans']) or
                state['source_plan_id']!=ep['last_valid_plan_id'] or state['cost_event_count']!=len(ep['cost_events'])):
            raise ValueError('state is not its actual pre-action prefix')
        own=[b for b in branches if b['state_id']==state['state_id']]
        expected={_action_id(a):True for a in state['state']['available_actions']}
        expected.update({_action_id(p['action']):p['feasible'] for p in state['state']['action_feasibility']})
        if len(own)!=len(expected) or {_action_id(b['action']):b['feasible'] for b in own}!=expected:
            raise ValueError('missing, duplicated or relabeled action branch')
        for branch in own:
            key=(state['state_id'],_action_id(branch['action']));edges[key]=branch
            if branch['branch_id']!=state['state_id']+'__'+_action_id(branch['action']):
                raise ValueError('action and branch identity disagree')
            if branch['feasible']:
                if branch['status'] not in ('completed','failed') or branch['path'] is None:
                    raise ValueError('action branch is unfinished')
                if branch['next_state_id'] is not None:
                    child=state_index.get(branch['next_state_id'])
                    if child is None or child['depth']!=1 or state['depth']!=0 or child['sample_id']!=state['sample_id']:
                        raise ValueError('invalid first-response child state')
                requires_child=state['depth']==0 and branch['action']['tool']!='STOP' and branch['status']=='completed'
                if requires_child!=(branch['next_state_id'] is not None):
                    raise ValueError('missing or unexpected continuation tree')
            elif branch['status']!='infeasible' or branch['path'] is not None or branch['next_state_id'] is not None:
                raise ValueError('infeasible action masquerades as an executed outcome')
        stop=edges[(state['state_id'],'STOP')]
        stop_task=_read(root,stop['path'])
        from planning.method_episode import decision_state
        source_ep=dict(ep,branch_id=stop['branch_id'])
        reconstructed,_,_=decision_state(source_ep)
        if reconstructed!=state['state'] or stop_task['episode']['decisions'][-1]['state']!=state['state']:
            raise ValueError('online state differs from its recorded decision boundary')
    for state in states:
        if state['depth']==1 and sum(b['next_state_id']==state['state_id'] for b in branches)!=1:
            raise ValueError('first-response state must have exactly one actual parent')
    for branch in branches:
        if branch['state_id'] is None or not branch['feasible']:continue
        state=state_index[branch['state_id']]
        task=_read(root,branch['path']);_check_episode(task,config)
        prefix=_read(root,state['prefix_path']);ep=task['episode']
        actual_id=branch['next_state_id']+'__STOP' if branch['next_state_id'] else branch['branch_id']
        if (ep['branch_id']!=actual_id or task['row']!=prefix['row'] or
                task['status']!=branch['status']):
            raise ValueError('declared branch does not match its actual task')
        for key in ('plans','requests','responses','ledger_snapshots','cost_events'):
            if ep[key][:len(prefix['episode'][key])]!=prefix['episode'][key]:
                raise ValueError('branch changed its parent prefix: '+key)
        offset=state['depth']
        if branch['action']['tool']=='STOP':
            if len(ep['requests'])!=offset or len(ep['plans'])!=offset+1:
                raise ValueError('STOP acquired information or regenerated a plan')
        else:
            from planning.method_episode import prepare_request
            from planning.evidence import known_field_manifest
            expected_request=prepare_request(state['state'],branch['action'],'q%d'%offset,
                known_field_manifest(prefix['episode']['ledger_snapshots'][-1]),config['spec']['control'])
            if len(ep['requests'])!=offset+1 or ep['requests'][-1]!=expected_request:
                raise ValueError('declared action differs from the actual bound request')
        if branch['next_state_id']:
            child=_read(root,state_index[branch['next_state_id']]['prefix_path'])['episode']
            if any(ep[k][:len(child[k])]!=child[k] for k in ('plans','requests','responses','ledger_snapshots','cost_events')):
                raise ValueError('child state differs from its actual paid first-response prefix')
    root_samples=[s['sample_id'] for s in states if s['depth']==0]
    failed_initial=[b['sample_id'] for b in branches if b['state_id'] is None and b['status']=='failed']
    if len(root_samples+failed_initial)!=len(selected) or set(root_samples+failed_initial)!=set(identities):
        raise ValueError('missing or duplicated initial sample result')
    for branch in branches:
        if branch['state_id'] is not None:continue
        task=_read(root,branch['path'])
        if (task['row']!=identities[branch['sample_id']] or task['status']!='failed' or branch['next_state_id'] is not None or
                branch['action'] is not None or branch['reason'] not in ('local_preparation','initial_plan_failed')):
            raise ValueError('initial failure entry lacks its actual failed task')
        ep=task.get('episode')
        if ep is None:
            if branch['reason']!='local_preparation' or not task.get('error'):
                raise ValueError('missing preparation failure record')
        elif (ep['status'] in ('completed','running') or ep['requests'] or ep['final_plan_id'] is not None or
                ep['events'][-1]['kind']!='failed'):
            raise ValueError('invalid initial failure artifact')
    return root,config,states,state_index,branches,edges


def _labels(labels_root, roles):
    result={}
    for role in sorted(roles):
        for label in read_jsonl(Path(labels_root)/(role+'.jsonl')):
            key=(label['sample_id'],label['role'])
            if key in result:raise ValueError('duplicate offline label identity')
            result[key]=label
    return result


def _source(root,state,config,label,utility):
    from evaluation.framework import _method_cost,_method_label,method_plan_row
    task=_read(root,state['prefix_path'])
    usable,status=_method_label(label,task['row'])
    if status!='complete':raise ValueError('supervision requires complete matching six-point labels: '+status)
    ep=task['episode'];metric=method_plan_row(ep['plans'][-1],usable,ep['ego_motion'],ep['limits']['execution_spec'])
    if not metric['record_consistent'] or not metric['parse_valid'] or metric['admissibility'] is not True:
        raise ValueError('source plan is not a valid actually generated plan')
    cost=_method_cost(task)
    # A captured prefix is intentionally running. All completed stages must still be known.
    if (cost['cost_issues'] or ep['cost']['complete'] is not True or
            any(e.get('complete') is not True for e in ep['cost_events']) or
            any(cost[k] is None for k in utility['cost_weights'])):
        raise ValueError('source prefix has missing or inconsistent costs')
    loss=sum(utility['quality_weights'][k]*metric['raw_'+k] for k in ('ADE3','FDE3'))
    return task,cost,loss,usable


def _target(root,config,state,branch,terminal,label,utility):
    from evaluation.framework import evaluate_method_task
    source,source_cost,source_loss,usable=_source(root,state,config,label,utility)
    result=dict(version='toolv2x_query_target_v1',sample_id=state['sample_id'],physical_recording=state['physical_recording'],
        fold=state['fold'],state_id=state['state_id'],action=copy.deepcopy(branch['action']),
        source_plan_id=state['source_plan_id'],source_loss=source_loss,utility_version=utility['version'],
        terminal_branch_id=None,terminal_plan_id=None,terminal_loss=None,terminal_status=None,
        incremental_cost=None,cost_penalty=None,target=None,status='infeasible' if not branch['feasible'] else 'labeled',
        reason=branch['reason'])
    if not branch['feasible']:return result
    task=_read(root,terminal['path']);_check_episode(task,config)
    ep=task['episode'];prefix=source['episode']
    if task['row']!=source['row'] or task['inputs'].get('local_prediction_seconds')!=source['inputs'].get('local_prediction_seconds'):
        raise ValueError('branch source inputs changed')
    for field in ('plans','requests','responses','ledger_snapshots','cost_events'):
        if ep[field][:len(prefix[field])]!=prefix[field]:
            raise ValueError('branch did not retain its actual shared prefix: '+field)
    report=evaluate_method_task(task,usable)
    if report['artifact_status'] not in ('completed','recorded_failure') or not report['cost_complete']:
        raise ValueError('terminal branch has invalid/missing outcome or costs')
    stop=branch['action']['tool']=='STOP'
    loss=(sum(utility['quality_weights'][k]*report[k] for k in ('ADE3','FDE3'))
          if report['task_success'] else utility['failure_loss'])
    delta={k:0. if stop else report[k]-source_cost[k] for k in utility['cost_weights']}
    if any(v<0 for v in delta.values()):raise ValueError('negative incremental cost from a mismatched prefix')
    penalty=sum(utility['cost_weights'][k]*delta[k] for k in delta)
    result.update(terminal_branch_id=terminal['branch_id'],terminal_plan_id=report['final_plan_id'],
        terminal_loss=loss,terminal_status=report['artifact_status'],incremental_cost=delta,cost_penalty=penalty,
        target=0. if stop else source_loss-loss-penalty)
    return result


def _save_targets(out,root,config,utility,rows,kind,teachers=None):
    out=Path(out)
    if out.resolve()==root.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError('offline targets must be outside the online branch archive')
    out.mkdir(parents=True,exist_ok=False)
    _save_method_json(out/'config.json',dict(version='toolv2x_query_targets_v1',kind=kind,source=str(root),
        binding=config['binding'],recording_folds=config['spec']['recording_folds'],utility_spec=utility,teachers=teachers,
        cost_scope='toolv2x_measured_stages_v3; delta after actual source prefix; STOP=0'))
    _write_rows(out/'targets.jsonl',rows)
    initial_failures=[b for b in read_jsonl(root/'branches.jsonl') if b['state_id'] is None]
    _save_method_json(out/'coverage.json',dict(expected_source_samples=len(read_jsonl(root/'selected_index.jsonl')),
        expected_targets=len(rows),labeled=sum(r['status']=='labeled' for r in rows),
        infeasible=sum(r['status']=='infeasible' for r in rows),failed_terminals=sum(r['terminal_status']=='recorded_failure' for r in rows),
        unlabelled_initial_samples=initial_failures))
    return rows


def make_terminal_targets(branch_root,labels_root,utility_spec,out):
    """Last-step net improvement relative to the actual current plan; STOP=0."""
    utility=validate_utility_spec(utility_spec)
    root,config,states,_,_,edges=_branch_archive(branch_root)
    labels=_labels(labels_root,{_read(root,s['prefix_path'])['row']['role'] for s in states})
    rows=[]
    for state in states:
        if state['depth']!=1:continue
        identity=_read(root,state['prefix_path'])['row'];label=labels.get((identity['sample_id'],identity['role']))
        for (state_id,_),branch in edges.items():
            if state_id==state['state_id']:
                rows.append(_target(root,config,state,branch,branch,label,utility))
    return _save_targets(out,root,config,utility,rows,'terminal')


def _teacher_provenance(teacher,fold,config,utility):
    p=copy.deepcopy(teacher.provenance);_check_json_native(p)
    required={'version','teacher_id','held_out_fold','training_recordings','binding','utility_spec','feature_spec','model_version'}
    if p.get('version')=='toolv2x_continuation_v2':
        required.add('frozen_source')
        from planning.query_value import validate_frozen_source
        validate_frozen_source(p)
    if (set(p)!=required or p['version'] not in ('toolv2x_continuation_v1','toolv2x_continuation_v2') or p['held_out_fold']!=fold or
            p['binding']!=config['binding'] or p['utility_spec']!=utility or not callable(teacher) or
            not isinstance(p['feature_spec'],dict) or not p['feature_spec'].get('version') or
            set(p['model_version'])!={'name','revision'} or not all(isinstance(v,str) and v and not v.startswith('/') for v in p['model_version'].values())):
        raise ValueError('frozen teacher binding/utility/version mismatch')
    _version(p['teacher_id']);_version(p['feature_spec']['version']);_version_pair(p['model_version'])
    groups=p['training_recordings'];folds=config['spec']['recording_folds']
    if (not isinstance(groups,list) or not groups or len(set(groups))!=len(groups) or
            any(recording(g)!=g or g not in folds or folds[g]==fold for g in groups)):
        raise ValueError('teacher training recordings overlap the held-out fold or lack provenance')
    return p


def make_first_targets(branch_root,labels_root,frozen_continuation,utility_spec,out):
    """Use a held-out frozen teacher's actual continuation, never GT argmax."""
    utility=validate_utility_spec(utility_spec)
    root,config,states,index,_,edges=_branch_archive(branch_root)
    chosen=[];teachers={}
    # Decide all continuations from saved visible states before opening labels.
    for state in states:
        if state['depth']!=0:continue
        fold=state['fold']
        if fold not in frozen_continuation:raise ValueError('missing held-out continuation teacher')
        teacher=frozen_continuation[fold]
        provenance=_teacher_provenance(teacher,fold,config,utility);teachers[fold]=provenance
        for (state_id,_),branch in edges.items():
            if state_id!=state['state_id']:continue
            terminal=branch;action=None
            if branch['feasible'] and branch['next_state_id'] is not None:
                child=index[branch['next_state_id']]
                action=teacher(copy.deepcopy(child['state']))
                _check_json_native(action)
                if (set(action)!={'tool','mode','reason'} or not isinstance(action['reason'],str) or
                        dict(tool=action['tool'],mode=action['mode']) not in child['state']['available_actions']):
                    raise ValueError('teacher selected an infeasible or unknown action')
                if teacher.provenance!=provenance:raise ValueError('teacher configuration changed during target construction')
                terminal=edges[(child['state_id'],_action_id(action))]
            chosen.append((state,branch,terminal,action,provenance))
    labels=_labels(labels_root,{_read(root,s['prefix_path'])['row']['role'] for s in states})
    rows=[]
    for state,branch,terminal,action,provenance in chosen:
        identity=_read(root,state['prefix_path'])['row'];label=labels.get((identity['sample_id'],identity['role']))
        row=_target(root,config,state,branch,terminal,label,utility)
        row.update(continuation_action=action,teacher_id=provenance['teacher_id'],
            teacher_training_recordings=provenance['training_recordings'])
        rows.append(row)
    return _save_targets(out,root,config,utility,rows,'first',teachers)
