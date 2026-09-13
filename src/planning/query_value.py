"""T8 ordinary supervised request values. No model/data loading at import time.

STOP has value zero. One shared action-conditioned MLP scores only public legal
requests. This is a standard solver, not an additional research mechanism.
"""
import copy
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from tools.task_spec import (ExecutionSpec, _keys, _check_json_native, _version,
                             _version_pair, field_key, validate_plan)

MODES = ('current', 'change', 'old', 'union')
ACTION_ORDER = ('STOP',) + tuple(t+'_'+m for m in MODES for t in ('P','F'))
STATE_KEYS = {'sample_id','branch_id','scene','g','provider','policy_id','ego_motion',
    'local_evidence','current_plan','previous_plan','acquired_fields','derived_fields',
    'admission_report','response_receipts','cost_ledger','remaining_budget',
    'available_actions','execution_spec','action_feasibility'}
ADMISSION_KEYS = {'acquired_field_refs','derived_field_refs','admitted_field_refs',
    'dropped_field_refs','dropped','field_groups','local_field_groups','observation_receipt_ids'}


def feature_spec():
    return dict(version='toolv2x_query_features_v1',plans='old_current_xy_6',
        motion=['speed_mps','yaw_rate_rps'],field_kinds=['anchor','history','forecast'],
        relations='known_anchor_and_aligned_forecast_min_distance',missing='zero_with_indicator')


def _reject_outcomes(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if (key in ('gt','labels','label','target','ground_truth','future','future_poses','features','provider_state') or
                    key.startswith(('gt_','unbought','unexecuted','candidate_actual'))):
                raise ValueError('offline or unpurchased content in visible state: '+key)
            _reject_outcomes(item)
    elif isinstance(value,list):
        for item in value:_reject_outcomes(item)


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite numeric feature/target required')
    return float(value)


def _records(records,origin,state):
    from tools.vehicle import _validate_task_value
    keys=set()
    for r in records:
        extra={'receipt_ids'} if origin=='remote' else {'parent_refs'} if origin=='receiver_derived' else set()
        _keys(r,{'ref','value','origin'}|extra,'visible field')
        key=field_key(r['ref'])
        if key in keys or r['origin']!=origin or key[1:3]!=(state['scene'],state['g']):
            raise ValueError('duplicate or mismatched visible field')
        if (origin=='local')==(key[0]==state['provider']):
            raise ValueError('local and remote evidence source mismatch')
        keys.add(key)
        value=r['value']
        if r['ref']['field_kind']=='forecast':
            if value.get('context_scope') not in ('ego_full_at_t','provider_full_at_t','receiver_acquired_subset'):
                raise ValueError('unknown forecast context')
            value=dict(value,context_scope='provider_full_at_t')
        _validate_task_value(r['ref']['field_kind'],value)
    return keys


def _validate_state(state,spec):
    if spec!=feature_spec():raise ValueError('unsupported feature specification')
    _check_json_native(state);_keys(state,STATE_KEYS,'query visible state');_reject_outcomes(state)
    execution=ExecutionSpec.from_dict(state['execution_spec'])
    _keys(state['ego_motion'],spec['motion'],'ego motion')
    for name in ('current_plan','previous_plan'):
        plan=state[name]
        if plan is None:
            if name=='current_plan':raise ValueError('current plan required')
            continue
        _keys(plan,('plan_id','waypoints','raw'),'visible plan')
        validate_plan(plan['waypoints'],execution)
    _keys(state['remaining_budget'],('calls','bytes'),'remaining budget')
    if any(type(v) is not int or v<0 for v in state['remaining_budget'].values()):
        raise ValueError('invalid remaining resource')
    for a in state['available_actions']:
        _keys(a,('tool','mode'),'available action');_action_id(a)
    for p in state['action_feasibility']:
        _keys(p,('action','request_bytes','response_reserve_bytes','feasible','reason'),'public preflight')
        _keys(p['action'],('tool','mode'),'preflight action');_action_id(p['action'])
        if type(p['feasible']) is not bool or any(type(p[k]) is not int or p[k]<0 for k in ('request_bytes','response_reserve_bytes')):
            raise ValueError('invalid public preflight')
    local=_records(state['local_evidence'],'local',state)
    acquired=_records(state['acquired_fields'],'remote',state)
    derived=_records(state['derived_fields'],'receiver_derived',state)
    from planning.evidence import known_field_manifest
    from tools.vehicle import decode_task_response
    from probe.kinematic_tools import encode
    for receipt in state['response_receipts']:
        _keys(receipt,('receipt_id','request','response','record_refs','wire_text','cost'),'receipt')
        packet=decode_task_response(receipt['wire_text'].encode('utf-8'),receipt['request'])
        if (packet!=receipt['response'] or packet['receipt_id']!=receipt['receipt_id'] or
                receipt['record_refs']!=[r['ref'] for r in packet['records']]):
            raise ValueError('visible receipt does not match its real response')
    known_field_manifest(dict(acquired_fields=state['acquired_fields'],receipts=state['response_receipts']))
    for record in state['derived_fields']:
        if not record['parent_refs'] or any(field_key(r) not in acquired for r in record['parent_refs']):
            raise ValueError('derived field lacks purchased parents')
    admission=state['admission_report'];_keys(admission,ADMISSION_KEYS,'admission report')
    if ({field_key(r) for r in admission['acquired_field_refs']}!=acquired or
            {field_key(r) for r in admission['derived_field_refs']}!=derived):
        raise ValueError('admission report does not match purchased/derived evidence')
    for name in ('admitted_field_refs','dropped_field_refs'):
        if any(field_key(r) not in local|acquired|derived for r in admission[name]):
            raise ValueError('admission names unknown evidence')
    return state


def _summaries(current,previous,motion,groups,admission,remaining,first_tool,stage,spec,*,bundle=False,truncated=False):
    names=[];values=[];missing=[]
    def add(name,value,nullable=False):
        absent=value is None
        names.append(name);values.append(0. if absent else _number(value))
        if nullable:
            names.append(name+'.missing');values.append(float(absent))
            if absent:missing.append(name)
    for label,plan in (('old',previous),('current',current)):
        for i,point in enumerate(plan if plan is not None else [[0.,0.]]*6):
            for axis,value in zip('xy',point):add('plan.%s.%d.%s'%(label,i,axis),value)
    add('previous_plan.missing',int(previous is None))
    if previous is None:missing.append('previous_plan')
    for name in spec['motion']:add('motion.'+name,motion[name],True)
    distances=None if previous is None else np.linalg.norm(np.asarray(current)-previous,axis=1)
    for name,value in (('mean',None if distances is None else float(distances.mean())),
                       ('max',None if distances is None else float(distances.max())),
                       ('last',None if distances is None else float(distances[-1]))):
        add('plan_difference.'+name,value,True)
    for name,records in list(groups.items())+[(k,admission[k+'_field_refs']) for k in ('admitted','dropped')]:
        for kind in spec['field_kinds']:
            add(name+'.'+kind+'.count',sum((r.get('ref') or r)['field_kind']==kind for r in records))
    for group,records in groups.items():
        for label,plan in (('current',current),('old',previous)):
            candidates=[]
            if plan is not None:
                for record in records:
                    kind=record['ref']['field_kind'];v=record['value']
                    if kind=='anchor':candidates.append(float(np.linalg.norm(np.asarray(plan)-v['box'][:2],axis=1).min()))
                    if kind=='forecast':candidates.append(float(np.linalg.norm(np.asarray(v['forecast'])-plan,axis=-1).min()))
            add(group+'.'+label+'.min_distance',min(candidates) if candidates else None,True)
    add('remaining.calls',remaining['calls']);add('remaining.bytes',remaining['bytes'])
    for tool in ('none','P','F'):add('first_tool.'+tool,int((first_tool or 'none')==tool))
    add('stage',stage)
    add('admission.missing',int(bundle));add('view.bundle',int(bundle))
    add('local_summary.truncated',int(truncated))
    return dict(version=spec['version'],names=names,values=values,missing=missing)


def state_features(visible_state, feature_spec):
    """Closed consumed schemas; audit metadata is never flattened into features."""
    s=_validate_state(visible_state,feature_spec)
    return _summaries(s['current_plan']['waypoints'],None if s['previous_plan'] is None else s['previous_plan']['waypoints'],
        s['ego_motion'],dict(local=s['local_evidence'],acquired=s['acquired_fields'],derived=s['derived_fields']),
        s['admission_report'],s['remaining_budget'],s['response_receipts'][0]['request']['tool'] if s['response_receipts'] else None,
        len(s['response_receipts']),feature_spec)


def _action_id(action):
    result=action['tool']+('_'+action['mode'] if action['mode'] is not None else '')
    if result not in ACTION_ORDER:raise ValueError('unknown value-policy action')
    return result


def feasible_queries(state):
    """Intersect the executor mask with causal stage and public reservations."""
    remaining=state['remaining_budget'];preflight={_action_id(p['action']):p for p in state['action_feasibility']}
    result=['STOP']
    for a in state['available_actions']:
        name=_action_id(a)
        if name=='STOP' or remaining['calls']<=0:continue
        if a['mode']!='current' and (state['previous_plan'] is None or
                a['mode']=='change' and state['previous_plan']['waypoints']==state['current_plan']['waypoints']):continue
        p=preflight.get(name)
        if p and p['feasible'] and p['request_bytes']+p['response_reserve_bytes']<=remaining['bytes']:
            result.append(name)
    return sorted(set(result),key=ACTION_ORDER.index)


def choose_query(values, feasible_actions):
    """STOP wins all zero ties; positive ties use the documented action order."""
    if 'STOP' not in feasible_actions or any(a not in ACTION_ORDER for a in feasible_actions):
        raise ValueError('STOP and known feasible actions required')
    best='STOP';value=0.
    for action in ACTION_ORDER[1:]:
        if action in feasible_actions:
            candidate=_number(values[action])
            if candidate>value:best,value=action,candidate
    return best


def _action_features(action):
    _action_id(action)
    return [float(action['tool']==t) for t in ('P','F')]+[float(action['mode']==m) for m in MODES]


def training_config(*,checkpoint,train_recordings,binding,utility_spec,**options):
    """Versioned experimental settings; paths are locations, never model identity."""
    result=dict(version='toolv2x_query_training_v1',feature_spec=feature_spec(),
        model_version=dict(name='action_conditioned_mlp',revision='v1'),hidden_sizes=[64,64],
        action_spec=dict(version='toolv2x_value_actions_v1',tools=['P','F'],modes=list(MODES),stop_value=0.,
            tie_order=list(ACTION_ORDER),candidate_ties='sent_order'),
        policy_id='query_value_v1',train_recordings=list(train_recordings),held_out_fold=None,
        binding=copy.deepcopy(binding),utility_spec=copy.deepcopy(utility_spec),seed=0,steps=200,
        batch_size_per_stage=32,learning_rate=.001,checkpoint_every=25,
        checkpoint=str(checkpoint),resume_from=None)
    if set(options)-set(result):raise ValueError('unknown training setting')
    result.update(options)
    return _training_config(result)


def _training_config(config):
    from common.audit_protocol import recording
    from planning.query_data import validate_utility_spec
    c=copy.deepcopy(config);_check_json_native(c)
    expected={'version','feature_spec','model_version','hidden_sizes','action_spec','policy_id',
        'train_recordings','held_out_fold','binding','utility_spec','seed','steps','batch_size_per_stage',
        'learning_rate','checkpoint_every','checkpoint','resume_from'}
    _keys(c,expected,'training configuration')
    if c['version']!='toolv2x_query_training_v1' or c['feature_spec']!=feature_spec():
        raise ValueError('unsupported training/feature version')
    _version_pair(c['model_version']);_version(c['policy_id'])
    if c['model_version']!={'name':'action_conditioned_mlp','revision':'v1'}:
        raise ValueError('unsupported ordinary value model')
    if c['action_spec']!=dict(version='toolv2x_value_actions_v1',tools=['P','F'],modes=list(MODES),
            stop_value=0.,tie_order=list(ACTION_ORDER),candidate_ties='sent_order'):
        raise ValueError('unsupported action semantics')
    if len(c['hidden_sizes'])!=2 or any(type(n) is not int or n<1 for n in c['hidden_sizes']):
        raise ValueError('two positive ordinary hidden widths required')
    if any(type(c[k]) is not int or c[k]<(0 if k=='seed' else 1) for k in
           ('seed','steps','batch_size_per_stage','checkpoint_every')) or _number(c['learning_rate'])<=0:
        raise ValueError('invalid optimizer/update budget')
    groups=c['train_recordings']
    if not isinstance(groups,list) or not groups or len(set(groups))!=len(groups) or any(recording(g)!=g for g in groups):
        raise ValueError('unique whole physical training recordings required')
    c['train_recordings']=sorted(groups)
    if c['held_out_fold'] is not None:_version(c['held_out_fold'])
    c['utility_spec']=validate_utility_spec(c['utility_spec'])
    return c


def _training_groups(c,folds):
    if any(g not in folds or folds[g]==c['held_out_fold'] for g in c['train_recordings']):
        raise ValueError('training recordings overlap held-out fold or are unknown')


def _target_terminal(row,branch,index,edges,source,utility,cache):
    """Rebind supervision to the actual chosen suffix and measured prefix costs."""
    from evaluation.framework import _method_cost
    terminal=branch
    if 'continuation_action' in row:
        action=row['continuation_action'];child=branch['next_state_id']
        if child is None:
            if action is not None:raise ValueError('unexpected continuation without a child state')
        else:
            _keys(action,('tool','mode','reason'),'frozen continuation')
            if (not isinstance(action['reason'],str) or
                    {k:action[k] for k in ('tool','mode')} not in index[child]['state']['available_actions']):
                raise ValueError('first label continuation is not feasible in its actual child')
            terminal=edges[(child,_action_id(action))]
    def task(path):
        if path not in cache:
            value=json.loads((source/path).read_text());cache[path]=(value,_method_cost(value))
        return cache[path]
    end,end_cost=task(terminal['path']);_,start_cost=task(index[row['state_id']]['prefix_path'])
    if (row['terminal_branch_id']!=terminal['branch_id'] or
            row['terminal_plan_id']!=end['episode']['final_plan_id'] or
            row['terminal_status']!=('completed' if end['status']=='completed' else 'recorded_failure')):
        raise ValueError('label terminal does not match its actual continuation branch')
    stop=row['action']['tool']=='STOP'
    _keys(row['incremental_cost'],utility['cost_weights'],'measured incremental cost')
    delta={k:0. if stop else end_cost[k]-start_cost[k] for k in utility['cost_weights']}
    if any(not math.isclose(_number(row['incremental_cost'][k]),v,abs_tol=1e-9) for k,v in delta.items()):
        raise ValueError('label cost differs from its actual source and terminal')
    penalty=sum(utility['cost_weights'][k]*v for k,v in delta.items())
    if not math.isclose(_number(row['cost_penalty']),penalty,abs_tol=1e-9):
        raise ValueError('label penalty differs from its frozen utility')


def training_examples(targets,kind,config):
    """Join labels to verified T7 online states; labels never enter state_features.

    Whole recording selection precedes normalization. First-target teachers must
    exclude both their own fold and any outer held-out training recordings.
    """
    from planning.query_data import _branch_archive, _teacher_provenance
    from planning.run_framework import read_jsonl
    c=_training_config(config);root=Path(targets)
    meta=json.loads((root/'config.json').read_text())
    if (meta['version']!='toolv2x_query_targets_v1' or kind not in ('first','terminal') or meta['kind']!=kind or
            meta['binding']!=c['binding'] or meta['utility_spec']!=c['utility_spec']):
        raise ValueError('target kind/binding/utility does not match this policy')
    source,archive,states,index,_,edges=_branch_archive(meta['source'])
    folds=archive['spec']['recording_folds'];_training_groups(c,folds)
    if meta['binding']!=archive['binding'] or meta['recording_folds']!=folds:
        raise ValueError('target source and configuration drift')
    selected={r['sample_id']:r for r in read_jsonl(source/'selected_index.jsonl')}
    stage=0 if kind=='first' else 1
    expected={(sid,aid) for (sid,aid) in edges if index[sid]['depth']==stage}
    rows=read_jsonl(root/'targets.jsonl');seen=set();examples=[];teachers={};names=None;cost_cache={}
    fields={'version','sample_id','physical_recording','fold','state_id','action','source_plan_id','source_loss',
        'utility_version','terminal_branch_id','terminal_plan_id','terminal_loss','terminal_status','incremental_cost',
        'cost_penalty','target','status','reason'}
    if kind=='first':fields|={'continuation_action','teacher_id','teacher_training_recordings'}
    for row in rows:
        _keys(row,fields,'offline target');_keys(row['action'],('tool','mode'),'target action')
        key=(row['state_id'],_action_id(row['action']))
        if key not in expected or key in seen:raise ValueError('duplicate/unknown target action or state')
        seen.add(key);s=index[row['state_id']];branch=edges[key]
        if (row['version']!='toolv2x_query_target_v1' or row['utility_version']!=c['utility_spec']['version'] or
                any(row[k]!=s[k] for k in ('sample_id','physical_recording','fold','source_plan_id')) or
                row['status']!=('labeled' if branch['feasible'] else 'infeasible')):
            raise ValueError('target does not match its actual source state')
        train=row['physical_recording'] in c['train_recordings']
        if train and selected[row['sample_id']]['role']!='train':raise ValueError('validation recording cannot train a value policy')
        if kind=='first' and train:
            p=meta['teachers'].get(row['fold'])
            if p is None:raise ValueError('first target lacks a frozen teacher manifest')
            teacher=lambda _:None
            teacher.provenance=p
            _teacher_provenance(teacher,row['fold'],archive,c['utility_spec'])
            if (p['feature_spec']!=c['feature_spec'] or row['teacher_id']!=p['teacher_id'] or
                    row['teacher_training_recordings']!=p['training_recordings'] or
                    not set(p['training_recordings'])<=set(c['train_recordings'])):
                raise ValueError('first teacher feature/training provenance leaks an outer group or drifts')
            teachers[row['fold']]=copy.deepcopy(p)
        if row['status']=='infeasible':
            if row['target'] is not None:raise ValueError('infeasible action has an invented value')
            continue
        _target_terminal(row,branch,index,edges,source,c['utility_spec'],cost_cache)
        value=_number(row['target'])
        if row['action']['tool']=='STOP':
            if value!=0.:raise ValueError('STOP target must be zero')
            continue
        if not math.isclose(value,_number(row['source_loss'])-_number(row['terminal_loss'])-_number(row['cost_penalty']),abs_tol=1e-9):
            raise ValueError('target is inconsistent with recorded utility')
        if not train:continue
        f=state_features(s['state'],c['feature_spec']);names=f['names']
        if key[1] not in feasible_queries(s['state']):raise ValueError('labeled request is not publicly feasible')
        examples.append(dict(state_id=s['state_id'],sample_id=s['sample_id'],physical_recording=s['physical_recording'],
            action=copy.deepcopy(row['action']),stage=kind,features=f['values'],target=value))
    if seen!=expected:raise ValueError('incomplete candidate supervision coverage')
    if {e['physical_recording'] for e in examples}!=set(c['train_recordings']):
        raise ValueError('each declared training recording requires actual feasible supervision')
    return dict(examples=examples,feature_names=names,teachers=teachers,
        source=dict(location=str(root),kind=kind,binding=meta['binding'],utility_spec=meta['utility_spec']),
        coverage=dict(available_targets=len(rows),training_remote_targets=len(examples),
            available_by_recording={g:sum(r['physical_recording']==g for r in rows) for g in folds}))


def _network(torch,dimension,c):
    nn=torch.nn;h=c['hidden_sizes']
    return nn.Sequential(nn.Linear(dimension+6,h[0]),nn.ReLU(),nn.Linear(h[0],h[1]),nn.ReLU(),nn.Linear(h[1],1))


def _torch_load(path):
    import torch
    # Only local trusted checkpoints produced here. No external download loader.
    return torch.load(str(path),map_location='cpu',weights_only=False)


def _save_checkpoint(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,str(temporary));temporary.replace(path)


def _fit(tables,kind,config):
    """CPU ordinary MSE regression; every shared update draws both stages."""
    import torch
    c=_training_config(config)
    examples=[e for table in tables for e in table['examples']]
    names=tables[0]['feature_names']
    if any(t['feature_names']!=names for t in tables):raise ValueError('feature order differs between supervision stages')
    path=Path(c['checkpoint'])
    if path.exists() and (c['resume_from'] is None or path.resolve()!=Path(c['resume_from']).resolve()):
        raise FileExistsError('choose a new checkpoint or explicitly resume this one')
    x=np.asarray([e['features'] for e in examples],dtype=np.float64)
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1.
    normalization=dict(mean=mean.tolist(),scale=scale.tolist())
    rng=np.random.RandomState(c['seed'])
    # Model initialization and training do not alter the caller's global torch RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        model=_network(torch,len(names),c);optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
        fit_id=uuid4().hex
        progress=dict(step=0,loss_history=[],stage_draws={e['stage']:0 for e in examples},
            actual_training_recordings=sorted({e['physical_recording'] for e in examples}),
            normalization_recordings=c['train_recordings'],supervision_counts={
                stage:sum(e['stage']==stage for e in examples) for stage in sorted({e['stage'] for e in examples})})
        teachers={fold:p for table in tables for fold,p in table['teachers'].items()}
        if c['resume_from'] is not None:
            previous=_torch_load(c['resume_from'])
            ignored={'steps','checkpoint','resume_from'}
            if (previous['kind']!=kind or any(c[k]!=previous['config'][k] for k in c if k not in ignored) or
                    previous['examples']!=examples or previous['normalization']!=normalization or
                    previous['feature_names']!=names or previous['teachers']!=teachers):
                raise ValueError('resume requires the same data, features, groups and optimizer contract')
            model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
            fit_id=previous['fit_id']
            progress=copy.deepcopy(previous['training']);rng.set_state(previous['numpy_rng'])
            torch.set_rng_state(previous['torch_rng'])
            if progress['step']>c['steps']:raise ValueError('resume update budget precedes saved progress')
        features=torch.tensor(np.concatenate(((x-mean)/scale,
            np.asarray([_action_features(e['action']) for e in examples])),axis=1),dtype=torch.float32)
        targets=torch.tensor([e['target'] for e in examples],dtype=torch.float32)
        stages={stage:np.asarray([i for i,e in enumerate(examples) if e['stage']==stage]) for stage in progress['stage_draws']}
        def save():
            _save_checkpoint(path,dict(version='toolv2x_query_checkpoint_v1',kind=kind,config=c,fit_id=fit_id,
                model=model.state_dict(),optimizer=optimizer.state_dict(),training=progress,
                normalization=normalization,feature_names=names,examples=examples,teachers=teachers,
                source_tables=[dict(source=t['source'],coverage=t['coverage']) for t in tables],
                numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state()))
        save()  # Step zero is durable before the first optimizer update.
        model.train()
        while progress['step']<c['steps']:
            batch=[]
            for stage,indices in stages.items():
                draw=rng.choice(indices,size=c['batch_size_per_stage'],replace=True)
                batch.extend(draw.tolist());progress['stage_draws'][stage]+=len(draw)
            optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(model(features[batch]).squeeze(-1),targets[batch])
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite supervised regression loss')
            loss.backward();optimizer.step()
            progress['step']+=1;progress['loss_history'].append(float(loss.detach()))
            if progress['step']%c['checkpoint_every']==0:save()
        save()
    return load_query_policy(path)


def fit_terminal_policy(targets,train_recordings,config):
    c=_training_config(config)
    if sorted(train_recordings)!=c['train_recordings'] or c['held_out_fold'] is None:
        raise ValueError('terminal teacher needs explicit training groups and held-out fold')
    return _fit([training_examples(targets,'terminal',c)],'terminal',c)


def fit_shared_policy(first_targets,terminal_targets,config):
    c=_training_config(config)
    return _fit([training_examples(first_targets,'first',c),training_examples(terminal_targets,'terminal',c)],'shared',c)


class QueryPolicy:
    """Frozen single network with callable executor interface and auditable origin."""
    def __init__(self,saved):
        import torch
        self.config=copy.deepcopy(saved['config']);self.kind=saved['kind']
        with torch.random.fork_rng(devices=[]):
            self.model=_network(torch,len(saved['feature_names']),self.config)
        self.model.load_state_dict(saved['model']);self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.feature_names=copy.deepcopy(saved['feature_names'])
        self.normalization=copy.deepcopy(saved['normalization']);self.training=copy.deepcopy(saved['training'])
        self.source_tables=copy.deepcopy(saved['source_tables'])
        c=self.config
        source=dict(version='toolv2x_frozen_value_source_v1',fit_id=saved['fit_id'],step=self.training['step'],
            training_spec={k:v for k,v in c.items() if k not in ('checkpoint','resume_from')},
            normalization=self.normalization,feature_names=self.feature_names)
        self._provenance=dict(version='toolv2x_continuation_v2' if self.kind=='terminal' else 'toolv2x_query_policy_v2',
            **({'teacher_id':c['policy_id']+'.fit.'+saved['fit_id']+'.step.'+str(self.training['step'])}
               if self.kind=='terminal' else {'policy_id':c['policy_id']}),
            held_out_fold=c['held_out_fold'],training_recordings=self.training['actual_training_recordings'],
            binding=c['binding'],utility_spec=c['utility_spec'],feature_spec=c['feature_spec'],model_version=c['model_version'],
            frozen_source=source)

    @property
    def provenance(self):return copy.deepcopy(self._provenance)

    @property
    def training_contract(self):
        c=self.config
        limits=copy.deepcopy(c['binding']['limits']);limits.pop('policy_id')
        candidates={g:sum(t['coverage']['available_by_recording'].get(g,0) for t in self.source_tables)
                    for g in c['train_recordings']}
        return dict(feature_spec=c['feature_spec'],hidden_sizes=c['hidden_sizes'],action_spec=c['action_spec'],
            train_recordings=c['train_recordings'],optimizer=dict(learning_rate=c['learning_rate'],seed=c['seed']),
            updates=self.training['step'],stage_draws=self.training['stage_draws'],
            supervision_counts=self.training['supervision_counts'],candidates_by_recording=candidates,
            receiver_query_spec=limits,utility_spec=c['utility_spec'],
            base_binding={k:c['binding'][k] for k in ('state_version','driver','predictor','local_provenance')})

    def _score(self,features,actions):
        import torch
        if not features:return []
        if any(f['names']!=self.feature_names for f in features):raise ValueError('feature order drift')
        mean=np.asarray(self.normalization['mean']);scale=np.asarray(self.normalization['scale'])
        x=(np.asarray([f['values'] for f in features])-mean)/scale
        x=np.concatenate((x,np.asarray([_action_features(a) for a in actions])),axis=1)
        with torch.no_grad():return self.model(torch.tensor(x,dtype=torch.float32)).squeeze(-1).tolist()

    def values(self,state):
        f=state_features(state,self.config['feature_spec'])
        if state['execution_spec']!=self.config['binding']['limits']['execution_spec']:
            raise ValueError('policy execution specification drift')
        ids=feasible_queries(state)[1:];actions=[dict(zip(('tool','mode'),name.split('_',1))) for name in ids]
        return dict(STOP=0.,**dict(zip(ids,self._score([f]*len(actions),actions))))

    def __call__(self,state):
        values=self.values(state);choice=choose_query(values,list(values))
        return dict(tool='STOP' if choice=='STOP' else choice.split('_',1)[0],
            mode=None if choice=='STOP' else choice.split('_',1)[1],reason='frozen_request_value')

    def validate_runtime(self,*,driver_provenance,predictor_descriptor,limits,local_provenance,control_spec):
        from planning.query_data import _semantic_binding
        actual=_semantic_binding(dict(driver=driver_provenance,predictor=predictor_descriptor),
            dict(limits=limits,control=control_spec,local_provenance=local_provenance))
        expected=copy.deepcopy(self.config['binding'])
        # policy_id is a dispatch label, not a receiver/model/query capability.
        actual['limits'].pop('policy_id');expected['limits'].pop('policy_id')
        if actual!=expected:raise ValueError('loaded value policy runtime binding drift')


def load_query_policy(checkpoint,*,expected_binding=None,policy_kind=None):
    saved=_torch_load(checkpoint)
    if saved['version']!='toolv2x_query_checkpoint_v1':raise ValueError('unsupported value checkpoint')
    _training_config(saved['config'])
    if expected_binding is not None and saved['config']['binding']!=expected_binding:
        raise ValueError('checkpoint model/query/receiver/utility binding drift')
    if policy_kind is not None and saved['kind']!=policy_kind:raise ValueError('wrong checkpoint policy kind')
    if saved['kind'] not in ('terminal','shared','bundle_terminal'):raise ValueError('unknown policy kind')
    return (BundlePolicy if saved['kind']=='bundle_terminal' else QueryPolicy)(saved)


def _bundle_state(state):
    from tools.control_bundle import validate_bundle_envelope,bundle_actions,continuation_request
    from tools.vehicle import decode_task_response
    from tools.task_spec import _request_spec
    from probe.kinematic_tools import encode
    _keys(state,('envelope','first_response','available_actions'),'conditional visible state')
    _reject_outcomes(state)
    env=state['envelope'];validate_bundle_envelope(env)
    packet=decode_task_response(encode(state['first_response']),env['first_request'])
    if packet!=state['first_response'] or state['available_actions']!=bundle_actions(env):
        raise ValueError('conditional actions must come from the actual sent envelope')
    summary=env['public_summary']
    _keys(summary,('ego_motion','local_evidence','summary_status'),'transmitted local summary')
    _keys(summary['ego_motion'],('speed_mps','yaw_rate_rps'),'transmitted motion')
    _keys(summary['summary_status'],('version','selected_fields','truncated'),'summary availability')
    for value in summary['ego_motion'].values():
        if value is not None:_number(value)
    first=env['first_request']
    _records(summary['local_evidence'],'local',dict(scene=first['scene'],g=first['g'],provider=first['provider']))
    if (summary['summary_status']['version']!='toolv2x_local_summary_v1' or
            summary['summary_status']['selected_fields']!=len(summary['local_evidence']) or
            type(summary['summary_status']['truncated']) is not bool):
        raise ValueError('invalid transmitted summary availability')
    feasible=[state['available_actions'][0]]
    for action in state['available_actions'][1:]:
        try:_request_spec(continuation_request(env,packet,action))
        except ValueError:continue  # Only sent data and public request capacity are inspected.
        feasible.append(action)
    return env,packet,feasible


def bundle_features(visible_state,action,feature_spec):
    """Per sent candidate, using first tau and actual returned fields; no ego tau1."""
    if feature_spec!=globals()['feature_spec']():raise ValueError('unsupported conditional features')
    env,packet,feasible=_bundle_state(visible_state)
    _keys(action,('tool','mode','candidate_id'),'conditional candidate')
    if action not in feasible or action['tool']=='STOP':raise ValueError('conditional feature action is infeasible')
    candidate=next(c for c in env['candidates'] if c['candidate_id']==action['candidate_id'])
    from probe.kinematic_tools import encode
    from tools.control_bundle import bundle_response_cap
    remaining=max(0,min(env['limits']['remaining_bytes']-len(encode(env)),bundle_response_cap(env))-
        len(encode(packet))-env['limits']['wrapper_reserve_bytes'])
    summary=env['public_summary']
    return _summaries(candidate['waypoints'],env['first_request']['tau_new'],summary['ego_motion'],
        dict(local=summary['local_evidence'],acquired=packet['records'],derived=[]),
        dict(admitted_field_refs=[],dropped_field_refs=[]),dict(calls=env['limits']['max_calls']-1,bytes=remaining),
        env['first_request']['tool'],1,feature_spec,bundle=True,truncated=summary['summary_status']['truncated'])


def _bundle_examples(targets,c):
    """Independent offline bundle table. Alternating T7 labels are not convertible.

    The producer must measure STOP as first-return -> final driver, and measure
    candidate suffixes against the same actual first response. This adapter
    validates schema/coverage/utility; it does not produce or certify experiments.
    """
    from common.audit_protocol import recording
    table=copy.deepcopy(targets) if isinstance(targets,dict) else json.loads(Path(targets).read_text())
    _keys(table,('version','kind','binding','utility_spec','recording_folds','rows','supervision'),'bundle target table')
    if (table['version']!='toolv2x_bundle_targets_v1' or table['kind']!='bundle_terminal' or
            table['binding']!=c['binding'] or table['utility_spec']!=c['utility_spec'] or
            c['binding']['control']['name']!='one_shot'):
        raise ValueError('conditional training requires separately bound bundle terminal targets')
    _training_groups(c,table['recording_folds'])
    supervision=table['supervision']
    _keys(supervision,('version','origin','stop_reference','cost_scope'),'bundle supervision')
    if (supervision['version']!='toolv2x_bundle_supervision_v1' or
            supervision['origin'] not in ('synthetic_contract','measured_bundle_branches') or
            supervision['stop_reference']!='first_return_then_final_driver' or
            supervision['cost_scope']!='outer_wire_and_complete_terminal_compute'):
        raise ValueError('bundle supervision must use its own STOP and outer transport costs')
    states={};seen={};examples=[];names=None
    def action_key(a):return (a['tool'],a['mode'],a['candidate_id'])
    for row in table['rows']:
        _keys(row,('state_id','physical_recording','fold','state','action','stop_loss','terminal_loss','incremental_cost','target'),
            'bundle supervision row')
        env,packet,feasible=_bundle_state(row['state'])
        _validate_bundle_binding(env,c)
        if env['limits']['execution_spec']!=c['binding']['limits']['execution_spec']:
            raise ValueError('conditional execution specification drift')
        group=recording(env['first_request']['scene'])
        if row['physical_recording']!=group or table['recording_folds'].get(group)!=row['fold']:
            raise ValueError('conditional physical recording mismatch')
        key=row['state_id']
        state_identity=(row['state'],row['physical_recording'],row['fold'],row['stop_loss'])
        if states.setdefault(key,state_identity)!=state_identity:raise ValueError('bundle sibling prefix or STOP loss changed')
        choices=seen.setdefault(key,set());action=row['action'];_keys(action,('tool','mode','candidate_id'),'bundle label action')
        if action not in feasible or action_key(action) in choices:raise ValueError('duplicate or infeasible conditional label')
        choices.add(action_key(action))
        _keys(row['incremental_cost'],c['utility_spec']['cost_weights'],'incremental bundle cost')
        penalty=sum(_number(row['incremental_cost'][k])*w for k,w in c['utility_spec']['cost_weights'].items())
        # A bundle continuation can reduce the final driver's compute time, so
        # signed differences are legal here. Absolute stage costs stay nonnegative.
        target=0. if action['tool']=='STOP' else _number(row['stop_loss'])-_number(row['terminal_loss'])-penalty
        if not math.isclose(_number(row['target']),target,abs_tol=1e-9):raise ValueError('inconsistent bundle target utility')
        if action['tool']=='STOP':
            if row['stop_loss']!=row['terminal_loss'] or any(v!=0 for v in row['incremental_cost'].values()):
                raise ValueError('conditional STOP must reference its own unchanged terminal')
            continue
        if group not in c['train_recordings']:continue
        f=bundle_features(row['state'],action,c['feature_spec']);names=f['names']
        examples.append(dict(state_id=key,physical_recording=group,action=copy.deepcopy(action),
            stage='bundle_terminal',features=f['values'],target=target))
    for key,(state,_,_,_) in states.items():
        if seen[key]!={action_key(a) for a in _bundle_state(state)[2]}:
            raise ValueError('incomplete conditional candidate supervision coverage')
    if {e['physical_recording'] for e in examples}!=set(c['train_recordings']):
        raise ValueError('missing conditional training recordings')
    return dict(examples=examples,feature_names=names,teachers={},source=dict(kind='bundle_terminal',supervision=supervision),
        coverage=dict(available_targets=len(table['rows']),training_remote_targets=len(examples),
            available_by_recording={g:sum(r['physical_recording']==g for r in table['rows']) for g in table['recording_folds']}))


def fit_bundle_continuation(targets,train_recordings,config):
    """Fit an independent control copy with the same features/capacity/optimizer."""
    c=_training_config(config)
    if sorted(train_recordings)!=c['train_recordings']:raise ValueError('conditional training groups differ')
    return _fit([_bundle_examples(targets,c)],'bundle_terminal',c)


def _validate_bundle_binding(envelope,c):
    from planning.method_controls import episode_bundle
    from probe.kinematic_tools import encode
    first=envelope['first_request'];summary=envelope['public_summary']
    _keys(summary,('ego_motion','local_evidence','summary_status'),'transmitted local summary')
    state=dict(provider=first['provider'],scene=first['scene'],g=first['g'],
        current_plan=dict(waypoints=first['tau_new']),ego_motion=summary['ego_motion'],
        local_evidence=summary['local_evidence'],response_receipts=[],
        remaining_budget=dict(calls=c['binding']['limits']['max_calls'],
            bytes=c['binding']['limits']['execution_spec']['max_episode_bytes']),
        execution_spec=c['binding']['limits']['execution_spec'])
    expected=episode_bundle(state,c['binding']['control'],first['tool'])
    summary_spec=c['binding']['control']['bundle']['summary_spec']
    if (envelope['continuation_policy_id']!=c['policy_id'] or
            any(envelope[k]!=expected[k] for k in ('limits','candidates','continuation_policy_id')) or
            any(first[k]!=expected['first_request'][k] for k in first if k!='request_id') or
            len(encode(summary))>summary_spec['max_bytes'] or len(summary['local_evidence'])>summary_spec['max_fields'] or
            any(r['ref']['field_kind'] not in summary_spec['field_kinds'] for r in summary['local_evidence'])):
        raise ValueError('conditional checkpoint bundle budget/candidate binding drift')


class BundlePolicy(QueryPolicy):
    def validate_provider(self,descriptor,provenance):
        expected=self.config['binding']
        if (descriptor is None or {k:descriptor[k] for k in ('model_version','settings')}!=expected['predictor'] or
                provenance!=expected['local_provenance']):
            raise ValueError('conditional policy actual provider binding drift')

    def validate_envelope(self,envelope):
        _validate_bundle_binding(envelope,self.config)

    def __call__(self,state):
        env,_,actions=_bundle_state(state)
        self.validate_envelope(env)
        scores=self._score([bundle_features(state,a,self.config['feature_spec']) for a in actions[1:]],actions[1:])
        best=actions[0];value=0.
        # Main action order, then pretransmitted candidate order; STOP wins zero.
        candidates={c['candidate_id']:i for i,c in enumerate(env['candidates'])}
        ordered=sorted(zip(actions[1:],scores),key=lambda pair:(ACTION_ORDER.index(_action_id(pair[0])),candidates[pair[0]['candidate_id']]))
        for action,score in ordered:
            if _number(score)>value:best,value=action,score
        return dict(best,reason='frozen_conditional_request_value')


def compare_training_contracts(left,right):
    """Report comparability; this neither balances data nor certifies performance."""
    required={'feature_spec','hidden_sizes','action_spec','train_recordings','optimizer','updates',
        'stage_draws','supervision_counts','candidates_by_recording','receiver_query_spec','utility_spec','base_binding'}
    _keys(left,required,'training comparison');_keys(right,required,'training comparison')
    differences={k:dict(left=left[k],right=right[k]) for k in sorted(required) if left[k]!=right[k]}
    return dict(version='toolv2x_training_comparison_v1',matched=not differences,differences=differences)


def validate_frozen_source(provenance):
    """A persistent fit/step identity plus full numeric configuration, no path IDs.

    The fit ID identifies a saved local training process. It is not a content
    fingerprint or proof against deliberate edits of a trusted checkpoint.
    """
    p=provenance;s=p['frozen_source']
    _keys(s,('version','fit_id','step','training_spec','normalization','feature_names'),'frozen value source')
    _version(s['fit_id'])
    if s['version']!='toolv2x_frozen_value_source_v1' or type(s['step']) is not int or s['step']<1:
        raise ValueError('teacher requires a saved completed optimizer step')
    c=_training_config(dict(s['training_spec'],checkpoint='audit_location',resume_from=None))
    if (s['step']>c['steps'] or p['teacher_id']!=c['policy_id']+'.fit.'+s['fit_id']+'.step.'+str(s['step']) or
            any(p[k]!=c[k] for k in ('held_out_fold','binding','utility_spec','feature_spec','model_version')) or
            p['training_recordings']!=c['train_recordings']):
        raise ValueError('teacher identity/configuration differs from its frozen source')
    _keys(s['normalization'],('mean','scale'),'frozen normalization')
    if (len(set(s['feature_names']))!=len(s['feature_names']) or
            any(len(v)!=len(s['feature_names']) for v in s['normalization'].values()) or
            any(_number(v)<=0 for v in s['normalization']['scale'])):
        raise ValueError('invalid frozen feature normalization')
    for value in s['normalization']['mean']:_number(value)
