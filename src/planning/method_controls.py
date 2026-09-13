"""Explicit T6 experiment arms. These are controls, not a learned request policy."""
import copy
import numpy as np

from tools.task_spec import _check_json_native

CONTROL_NAMES = ('feedback', 'frozen_feedback', 'current_old', 'current_union',
    'exact_repeat', 'self_refinement', 'evidence_refinement', 'legacy_v2',
    'ego_max_context', 'one_shot')


def control_spec(name, **options):
    value = dict(version='toolv2x_controls_v1', name=name, driver_calls=3,
                 repeat_after_calls=0, refinement_slot_tokens=256, baseline='Ego', bundle=None)
    if set(options) - set(value):
        raise ValueError('unknown control parameter')
    value.update(options)
    _check_json_native(value)
    if (value['version'] != 'toolv2x_controls_v1' or name not in CONTROL_NAMES or
            type(value['driver_calls']) is not int or not 1 <= value['driver_calls'] <= 3 or
            type(value['repeat_after_calls']) is not int or value['repeat_after_calls'] not in (0,1) or
            type(value['refinement_slot_tokens']) is not int or value['refinement_slot_tokens'] <= 0 or
            value['baseline'] not in ('Ego','P','F','PF','rule')):
        raise ValueError('invalid versioned control specification')
    if name == 'one_shot' and not isinstance(value['bundle'], dict):
        raise ValueError('one-shot requires an explicit registered bundle specification')
    return copy.deepcopy(value)


def normalize_control(value):
    if value is None:
        return None
    return control_spec(**value)


def uses_refinement(spec):
    return spec is not None and (spec['name'] in ('self_refinement','evidence_refinement') or
        (spec['name']=='one_shot' and spec['bundle'].get('extra_generation')=='self_refinement'))


def control_state(state, spec, initial_plan):
    """Detach and consistently hide revised plan features in the frozen arm."""
    result = copy.deepcopy(state)
    if spec is None:
        return result
    name = spec['name']
    if name == 'frozen_feedback' and result['previous_plan'] is not None:
        result['current_plan'] = copy.deepcopy(initial_plan)
        result['previous_plan'] = copy.deepcopy(initial_plan)
        result['available_actions'] = [a for a in result['available_actions'] if a['mode'] != 'change']
    if name in ('current_old','current_union'):
        for action in result['available_actions']:
            if action['mode'] == 'change':
                action['mode'] = 'old' if name=='current_old' else 'union'
    if name == 'legacy_v2':
        for action in result['available_actions']:
            if action['tool'] != 'STOP':
                action['mode'] = 'roi'
        result['available_actions'] = [dict(tool='STOP',mode=None)] + (
            [dict(tool=t,mode='roi') for t in ('P','F')] if state['remaining_budget']['calls'] else [])
    return result


def legacy_decision(state, baseline):
    """Reuse v1 choose_action with only received anchors and its original geometry rule."""
    from planning.episode import choose_action
    anchors = {r['ref']['track_handle']:r['value'] for r in state['acquired_fields'] if r['ref']['field_kind']=='anchor'}
    packets=[]
    for receipt in state['response_receipts']:
        req=receipt['request']
        ids={r['track_handle'] for r in receipt['record_refs']}
        packets.append(dict(tool=req['tool'],roi=req.get('roi'),objects=[dict(track_id=i,**anchors[i]) for i in sorted(ids) if i in anchors]))
    relations=[]
    for ego in state['local_evidence']:
        if ego['ref']['field_kind']!='anchor': continue
        box=ego['value']['box']
        for i,peer in anchors.items():
            distance=np.linalg.norm(np.asarray(box[:2])-peer['box'][:2])
            ratios=np.asarray(peer['box'][3:6])/box[3:6]
            if distance<=3. and (ratios>=.5).all() and (ratios<=2.).all():
                relations.append(dict(peer_id=i))
    return choose_action(dict(motion=state['ego_motion'],responses=packets,evidence=dict(relations=relations)),baseline)


def make_control_policy(control_spec, value_policy):
    """Reuse the caller's policy; T8 can inject independently fitted frozen policies."""
    spec=normalize_control(control_spec)
    def policy(state):
        name=spec['name']
        if name=='ego_max_context':
            return dict(tool='STOP',mode=None,reason='ego_max_context')
        if name in ('exact_repeat','self_refinement'):
            if len(state['response_receipts'])>=spec['repeat_after_calls']:
                return dict(tool='STOP',mode=None,reason='fixed_evidence_complete')
        if name=='legacy_v2':
            decision=legacy_decision(state,spec['baseline'])
            return dict(tool=decision['tool'],mode=None if decision['tool']=='STOP' else 'roi',reason=decision['reason'])
        return value_policy(copy.deepcopy(state))
    return policy


def repeat_generation(spec, episode):
    if spec is None or len(episode['plans'])>=spec['driver_calls']:
        return False
    if spec['name'] in ('exact_repeat','self_refinement'):
        return len(episode['requests'])>=spec['repeat_after_calls']
    return spec['name']=='one_shot' and bool(episode['responses']) and spec['bundle'].get('extra_generation') in ('exact_repeat','self_refinement')


def make_bundle(state, public_summary, candidates, continuation_policy_id, limits):
    """Build one external request from causal sent data; never inspect a provider."""
    from tools.task_spec import TASK_VERSION, TIMES
    # Fields are already verified received references; no derived forecast is a receipt.
    manifest=[dict(ref=copy.deepcopy(ref),receipt_id=r['receipt_id'])
              for r in state['response_receipts'] for ref in r['record_refs']]
    first=dict(version=TASK_VERSION,request_id='q0',provider=state['provider'],scene=state['scene'],g=state['g'],
        coordinate_frame='ego_at_t',tool=limits['first_tool'],mode='current',times=list(TIMES),
        tau_new=copy.deepcopy(state['current_plan']['waypoints']),tau_old=None,
        execution_spec=copy.deepcopy(limits['execution_spec']),acquired_field_manifest=manifest)
    config={k:copy.deepcopy(v) for k,v in limits.items() if k!='first_tool'}
    return dict(version='toolv2x_bundle_v1',request_id='bundle0',bundle_id='bundle0',first_request=first,
        public_summary=copy.deepcopy(public_summary),candidates=copy.deepcopy(candidates),
        continuation_policy_id=continuation_policy_id,limits=config)


def episode_bundle(state, spec, tool):
    """Cheap initial/slowdown/constant-motion candidates; all construction is timed by caller."""
    from tools.task_spec import TIMES, validate_plan
    bundle=spec['bundle']
    required={'continuation_policy_id','candidate_sources','slowdown_scale','max_candidates','wrapper_reserve_bytes','extra_generation'}
    if set(bundle)!=required or bundle['extra_generation'] not in (None,'exact_repeat','self_refinement'):
        raise ValueError('invalid bundle control configuration')
    if (not isinstance(bundle['candidate_sources'],list) or not bundle['candidate_sources'] or
            bundle['candidate_sources'][0]!='initial' or len(set(bundle['candidate_sources']))!=len(bundle['candidate_sources']) or
            type(bundle['slowdown_scale']) not in (int,float) or not 0<bundle['slowdown_scale']<=1 or
            type(bundle['max_candidates']) is not int or bundle['max_candidates']<len(bundle['candidate_sources']) or
            type(bundle['wrapper_reserve_bytes']) is not int or bundle['wrapper_reserve_bytes']<=0):
        raise ValueError('invalid predeclared candidate/capacity configuration')
    tau=np.asarray(state['current_plan']['waypoints'])
    motion=state['ego_motion']
    candidates=[]
    for name in bundle['candidate_sources']:
        if name=='initial': points=tau
        elif name=='slower': points=tau*bundle['slowdown_scale']
        elif name=='constant_motion':
            speed,yaw=motion['speed_mps'],motion['yaw_rate_rps']
            if speed is None or yaw is None:
                continue  # Missing causal state is not an invented speed or turn rate.
            t=np.asarray(TIMES)
            points=np.column_stack((speed*t,np.zeros(6))) if abs(yaw)<1e-8 else np.column_stack((speed*np.sin(yaw*t)/yaw,speed*(1-np.cos(yaw*t))/yaw))
        else: raise ValueError('unknown causal candidate source')
        candidates.append(dict(candidate_id=name,waypoints=validate_plan(points.tolist(),state['execution_spec'])))
    cap=state['execution_spec']['max_response_bytes']-bundle['wrapper_reserve_bytes']
    calls=min(2,state['remaining_budget']['calls'])
    if calls<1 or cap<calls:
        raise ValueError('bundle response budget cannot fund primitives')
    caps=[cap//calls]*calls
    caps[-1]+=cap-sum(caps)
    limits=dict(execution_spec=state['execution_spec'],max_calls=calls,max_candidates=bundle['max_candidates'],
        remaining_bytes=state['remaining_budget']['bytes'],primitive_response_caps=caps,
        wrapper_reserve_bytes=bundle['wrapper_reserve_bytes'],first_tool=tool)
    summary={key:copy.deepcopy(state[key]) for key in ('ego_motion','local_evidence','current_plan')}
    return make_bundle(state,summary,candidates,bundle['continuation_policy_id'],limits)


def diagnostic_bundle_continuation(state):
    """Wiring-only conditional rule; not a claim of a fitted strong one-shot policy."""
    has_fields=bool(state['first_response']['records'])
    action=next((a for a in state['available_actions'] if a['tool']=='F' and a['mode']=='current' and a['candidate_id']=='initial'),None)
    if has_fields and action is not None:
        return dict(action,reason='diagnostic_paid_fields')
    return dict(tool='STOP',mode=None,candidate_id=None,reason='diagnostic_no_second_action')


def capture_control_prefix(**episode_inputs):
    """Capture live first-response revision for paired controls, not disk resumption.

    Bind copies of the actual feature values and the same live driver. This
    in-memory handle is not serialized; branch archives keep the real prefix.
    """
    from planning.method_episode import run_task_episode
    class PrefixReady(Exception):
        pass
    episode_inputs=dict(episode_inputs,features=copy.deepcopy(episode_inputs['features']))
    captured={}
    original=episode_inputs.get('on_progress')
    def capture(episode):
        if original is not None:
            original(episode)
        if episode['events'][-1]['kind']=='driver_completed' and len(episode['plans'])==2 and episode['plans'][-1]['status']=='valid':
            captured.update(episode=copy.deepcopy(episode),features=copy.deepcopy(episode_inputs['features']),driver=episode_inputs['driver'])
            raise PrefixReady()
    try:
        run_task_episode(**dict(episode_inputs,on_progress=capture))
    except PrefixReady:
        return captured
    raise ValueError('execution did not reach a valid first-response revision')


def validate_control_prefix(prefix, features, driver):
    if (not isinstance(prefix,dict) or set(prefix)!={'episode','features','driver'} or prefix['driver'] is not driver or
            set(features)!=set(prefix['features'])):
        raise ValueError('control branches require the same live driver and captured input')
    for key,value in features.items():
        saved=prefix['features'][key]
        if isinstance(value,np.ndarray):
            equal=isinstance(saved,np.ndarray) and saved.dtype==value.dtype and np.array_equal(saved,value)
        else:
            equal=value==saved
        if not equal:
            raise ValueError('control branch feature content changed: '+key)
    return copy.deepcopy(prefix['episode'])
