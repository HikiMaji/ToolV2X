"""Fixed-time P/F delegation alternating with the same driver. No offline labels.

Policies are trusted local callables receiving detached visible data, not sandboxed
Python. A callable never receives the service, unpurchased peer data or features.
"""
import copy
from time import perf_counter

import numpy as np

from planning.context import build_task_plan_input
from planning.evidence import new_ledger, apply_response, known_field_manifest, EvidenceUpdateError
from planning.inputs import parse_q9
from planning.driver_contract import (validate_driver_binding, validate_numeric_output,
                                      validate_numeric_cost, validate_numeric_episode)
from probe.kinematic_tools import encode
from tools.task_spec import (ExecutionSpec, TASK_VERSION, TIMES, validate_plan,
                             validate_task_request, _check_json_native, _version_pair)


COMPUTE_ACCOUNTING_VERSION = 'toolv2x_compute_accounting_v1'
DECISION_TIMING_VERSION = 'toolv2x_control_compute_v2'
DIAGNOSTIC_POLICY_IDS = ('stop', 'p_current', 'p_current_f_change',
                         'f_current', 'p_current_f_current')


def validate_limits(limits):
    _check_json_native(limits)
    required = {'version', 'execution_spec', 'receiver_spec', 'max_calls', 'policy_id', 'driver_version'}
    if (set(limits) != required or limits['version'] not in ('toolv2x_interaction_v1', 'toolv2x_interaction_v2') or
            type(limits['max_calls']) is not int or not 0 <= limits['max_calls'] <= 2 or
            not isinstance(limits['policy_id'], str) or not limits['policy_id']):
        raise ValueError('invalid interaction limits')
    ExecutionSpec.from_dict(limits['execution_spec'])
    _version_pair(limits['driver_version'])
    r = limits['receiver_spec']
    if limits['version'] == 'toolv2x_interaction_v2':
        from planning.structured_inputs import StructuredDriverSpec
        StructuredDriverSpec.from_dict(r)
        return copy.deepcopy(limits)
    if (set(r) != {'version', 'context_limit', 'generation_reserve', 'peer_reserve', 'numeric_decimal_places'} or
            r['generation_reserve'] != 256):
        raise ValueError('explicit receiver specification with original 256-token generation required')
    return copy.deepcopy(limits)


def diagnostic_policy(state):
    """T4 CLI wiring check only; not the T8 learned request policy or a result claim."""
    name = state['policy_id']
    if name not in DIAGNOSTIC_POLICY_IDS:
        raise ValueError('unknown diagnostic policy')
    if name == 'stop':
        return dict(tool='STOP', mode=None, reason='diagnostic_stop')
    calls = len(state['response_receipts'])
    if calls == 0:
        tool = 'F' if name == 'f_current' else 'P'
        if dict(tool=tool, mode='current') in state['available_actions']:
            return dict(tool=tool, mode='current', reason='diagnostic_first_request')
    if name == 'p_current_f_current' and calls == 1 and dict(tool='F', mode='current') in state['available_actions']:
        return dict(tool='F', mode='current', reason='diagnostic_fixed_second_request')
    if name == 'p_current_f_change' and dict(tool='F', mode='change') in state['available_actions']:
        return dict(tool='F', mode='change', reason='diagnostic_actual_revision')
    return dict(tool='STOP', mode=None, reason='diagnostic_done_or_unchanged')


def prepare_request(state, action, request_id, manifest, control_spec=None):
    """Bind only visible time-t inputs; no service or prediction access."""
    from planning.method_controls import episode_bundle, legacy_decision
    if control_spec and control_spec['name']=='one_shot':
        return episode_bundle(state,control_spec,action['tool'])
    request=dict(version=TASK_VERSION if action['mode'] in ('current','change') else 'toolv2x_control_task_v1',
        request_id=request_id,provider=state['provider'],scene=state['scene'],g=state['g'],coordinate_frame='ego_at_t',
        tool=action['tool'],mode=action['mode'],times=list(TIMES),
        tau_new=copy.deepcopy(state['current_plan']['waypoints']),
        tau_old=copy.deepcopy(state['previous_plan']['waypoints']) if action['mode'] in ('change','old','union') else None,
        execution_spec=copy.deepcopy(state['execution_spec']),acquired_field_manifest=copy.deepcopy(manifest))
    if action['mode']=='roi':
        request['roi']=legacy_decision(state,control_spec['baseline'])['roi']
    return request


def request_budget(request, remaining_bytes):
    """Public reservations, not a preview of an unknown actual response."""
    spec=request.get('execution_spec',request.get('limits',{}).get('execution_spec'))
    size=len(encode(request))
    response_cap=request.get('limits',{}).get('outer_response_cap',spec['max_response_bytes'])
    reason=('request_byte_limit' if size>spec['max_request_bytes'] else
            'byte_budget_exhausted' if size+response_cap>remaining_bytes else None)
    return dict(request_bytes=size,response_reserve_bytes=response_cap,feasible=reason is None,reason=reason)


def decision_state(episode):
    """Build the policy whitelist and request mask from an actual known prefix.

    Shared with offline archive validation; no service, model, labels or child
    outcomes are arguments. The returned requests remain local until selected.
    """
    from planning.method_controls import control_state
    limits=episode['limits'];control=episode.get('control_spec')
    ledger=episode['ledger_snapshots'][-1];plans=episode['plans'];prepared=plans[-1]['prepared']
    def plan(record):
        return dict(plan_id=record['plan_id'],waypoints=copy.deepcopy(record['output']['waypoints']),raw=record['output'].get('q9_raw'))
    current=plan(plans[-1]);previous=plan(plans[-2]) if len(plans)>1 else None
    remaining=limits['max_calls']-len(episode['requests'])
    if remaining<0:raise ValueError('prefix already exceeds the call allowance')
    if control and (len(plans)>=control['driver_calls'] or control['name']=='one_shot' and episode['requests']):
        remaining=0
    actions=[dict(tool='STOP',mode=None)]
    if remaining:
        actions += [dict(tool=t,mode='current') for t in ('P','F')]
        if previous is not None and previous['waypoints']!=current['waypoints']:
            actions += [dict(tool=t,mode='change') for t in ('P','F')]
    state=dict(sample_id=episode['sample_id'],branch_id=episode['branch_id'],scene=episode['scene'],g=episode['g'],
        provider=episode['provider'],policy_id=limits['policy_id'],ego_motion=copy.deepcopy(episode['ego_motion']),
        local_evidence=copy.deepcopy(ledger['local_fields']),current_plan=current,previous_plan=previous,
        acquired_fields=copy.deepcopy(ledger['acquired_fields']),derived_fields=copy.deepcopy(ledger['derived_fields']),
        admission_report=copy.deepcopy(prepared['admission_report']),response_receipts=copy.deepcopy(ledger['receipts']),
        cost_ledger=copy.deepcopy(episode['cost_events']),remaining_budget=dict(calls=remaining,
            bytes=limits['execution_spec']['max_episode_bytes']-episode['cost']['request_bytes']-episode['cost']['response_bytes']),
        available_actions=actions,execution_spec=copy.deepcopy(limits['execution_spec']))
    state=control_state(state,control,plan(plans[0]))
    candidates={};preflight=[]
    for action in state['available_actions']:
        if action['tool']=='STOP':continue
        request=prepare_request(state,action,'q%d'%len(episode['requests']),known_field_manifest(ledger),control)
        budget=request_budget(request,state['remaining_budget']['bytes'])
        preflight.append(dict(action=copy.deepcopy(action),**budget))
        if budget['feasible']:candidates[(action['tool'],action['mode'])]=request
    state['available_actions']=[a for a in state['available_actions'] if a['tool']=='STOP' or (a['tool'],a['mode']) in candidates]
    state['action_feasibility']=copy.deepcopy(preflight)
    return state,candidates,preflight


def run_task_episode(local_window, local_prediction, motion, features,
                     service, predictor, driver, policy, limits, on_progress=None, *,
                     local_provenance, sample_id=None, branch_id='main', token_counter=None, control_spec=None, prefix=None):
    """Run 1--3 direct driver attempts and 0--2 remote calls; persist every boundary.

    A progress callback failure propagates: execution cannot continue without its
    durable record. Partial episodes are never resumed by inventing service state.
    token_counter is only for resource-independent contracts; actual GoT rejects it.
    """
    from planning.method_controls import normalize_control, uses_refinement, repeat_generation, validate_control_prefix
    from planning.inputs import build_refinement_input
    control_spec = normalize_control(control_spec)
    limits = validate_limits(limits)
    numeric = validate_driver_binding(limits, driver.provenance)
    if numeric and token_counter is not None:
        raise ValueError('numeric driver does not accept a language token counter')
    if not numeric and control_spec and control_spec['name'] == 'ego_max_context':
        limits['receiver_spec']['peer_reserve'] = 0
    if hasattr(policy, 'validate_runtime'):
        policy.validate_runtime(driver_provenance=copy.deepcopy(driver.provenance),
            predictor_descriptor=copy.deepcopy(predictor.descriptor), limits=copy.deepcopy(limits),
            local_provenance=copy.deepcopy(local_provenance), control_spec=copy.deepcopy(control_spec))
    ledger = new_ledger(local_window, local_prediction, predictor=predictor,
                        local_provenance=local_provenance,
                        **(dict(p_processing=limits['receiver_spec']['p_processing'], include_local_history=True) if numeric else {}))
    scene, g = ledger['scene'], ledger['g']
    if (service.scene, service.g) != (scene, g) or service.provider == ledger['local_source'] or (service.task_records and prefix is None):
        raise ValueError('one fresh same-time remote service is required per episode')
    features = copy.deepcopy(features)
    for value in features.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    mask = np.asarray(features['active_agent_mask'])
    if mask.shape != (1, 2, 1) or mask.dtype != bool or not mask[0, 0, 0]:
        raise ValueError('fixed single-ego feature mask required')
    feature_tokens = 0 if numeric else int(mask.sum()) * 270
    motion = copy.deepcopy(motion)
    driver_config = copy.deepcopy(driver.provenance)
    driver_context = None if numeric else driver.context_limit
    costs = dict(driver_attempts=0, query_attempts=0, request_bytes=0, response_bytes=0,
                 driver_attempt_seconds=0., generation_seconds=0., service_seconds=0.,
                 receiver_seconds=0., input_build_seconds=0., complete=True)
    episode = dict(version='toolv2x_episode_v2', compute_accounting_version=COMPUTE_ACCOUNTING_VERSION,
        sample_id=sample_id or '%s_g%d' % (scene, g),
        branch_id=branch_id, policy_id=limits['policy_id'], scene=scene, g=g, provider=service.provider,
        coordinate_frame='ego_at_t', times=list(TIMES), limits=limits, ego_motion=motion,
        predictor_binding=predictor.descriptor, driver_provenance=driver_config,
        feature_tokens=feature_tokens, status='running', plans=[], requests=[], responses=[],
        decisions=[], ledger_snapshots=[copy.deepcopy(ledger)], events=[], cost_events=[], cost=costs,
        final_plan_id=None, last_valid_plan_id=None, stop_reason=None,
        execution_kind='structured_numeric' if numeric else 'injected_contract' if token_counter is not None else 'original_got',
        gt_labels_read=False)
    if numeric:
        episode['numeric_scene_tokens'] = int(mask.sum()) * 220
    if control_spec is not None:
        episode['control_spec'] = copy.deepcopy(control_spec)
    stage, current, previous = 0, None, None
    reuse_plan = False
    fixed_generation = False
    if prefix is not None:
        prefix=validate_control_prefix(prefix,features,driver)
        if numeric:
            validate_numeric_episode(prefix)
        # Only live initial/first-response driver prefixes, not process recovery.
        prefix_calls=len(prefix.get('requests', []))
        if (prefix.get('version') != 'toolv2x_episode_v2' or
                prefix.get('compute_accounting_version') != COMPUTE_ACCOUNTING_VERSION or prefix.get('status') != 'running' or
                prefix_calls not in (0,1) or len(prefix.get('plans', [])) != prefix_calls+1 or
                len(prefix.get('responses', [])) != prefix_calls or prefix['events'][-1]['kind'] != 'driver_completed' or
                any(prefix.get(k) != episode[k] for k in ('sample_id','scene','g','provider','driver_provenance','predictor_binding','ego_motion','feature_tokens')) or
                any(prefix['limits'][k] != limits[k] for k in limits if k != 'policy_id') or
                prefix['ledger_snapshots'][0] != ledger):
            raise ValueError('control branch requires the same actual driver prefix')
        if not numeric and uses_refinement(control_spec) and any(
                p['prepared'].get('refinement',{}).get('slot_tokens') != control_spec['refinement_slot_tokens']
                for p in prefix['plans']):
            raise ValueError('paired refinement requires the same slot reserved in the real prefix')
        for plan in prefix['plans']:
            output=plan.get('output') or {}
            if numeric:
                validate_numeric_output(output, plan['prepared'], limits['execution_spec'])
            elif parse_q9(output['q9_raw']).tolist()!=output['waypoints']:
                raise ValueError('invalid driver prefix')
            if plan['status']!='valid':
                raise ValueError('invalid driver prefix')
            validate_plan(output['waypoints'], limits['execution_spec'])
        records=service.task_records
        if len(records)!=prefix_calls or any(
                record['request']!=request or encode(record.get('response'))!=bytes.fromhex(response['wire_hex'])
                for record,request,response in zip(records,prefix['requests'],prefix['responses'])):
            raise ValueError('prefix service receipts must come from the same actual response')
        episode=copy.deepcopy(prefix)
        episode.update(branch_id=branch_id,policy_id=limits['policy_id'],limits=limits,
            control_spec=copy.deepcopy(control_spec),prefix_origin=dict(branch_id=prefix['branch_id'],plan_id='plan_%d'%prefix_calls))
        costs=episode['cost']
        ledger=copy.deepcopy(episode['ledger_snapshots'][-1])
        stage=prefix_calls
        for i in range(prefix_calls+1):
            out=episode['plans'][i]['output']
            plan=dict(plan_id=episode['plans'][i]['plan_id'],waypoints=out['waypoints'],raw=out.get('q9_raw'))
            previous,current=current,plan
        reuse_plan=True

    if hasattr(policy, 'provenance'):
        episode['value_policy'] = copy.deepcopy(policy.provenance)

    def emit(kind, plan_id=None, request_id=None):
        events = episode['events']
        events.append(dict(event_id=len(events), kind=kind, sample_id=episode['sample_id'],
            branch_id=branch_id, stage=stage, plan_id=plan_id, request_id=request_id,
            caused_by_event_id=events[-1]['event_id'] if events else None))
        if on_progress is not None:
            on_progress(copy.deepcopy(episode))

    def fail(status, exc, phase, plan_id=None, request_id=None):
        episode.update(status=status, stop_reason=status, final_plan_id=None,
                       error=dict(stage=phase, error_type=type(exc).__name__, message=str(exc)))
        emit('failed', plan_id, request_id)
        return episode

    def frozen_driver():
        if driver.provenance != driver_config or (not numeric and driver.context_limit != driver_context):
            raise ValueError('driver configuration changed within episode')
        validate_driver_binding(limits, driver.provenance)
        if not numeric and driver_context != limits['receiver_spec']['context_limit']:
            raise ValueError('driver and receiver context capacities differ')
        if predictor.descriptor != episode['predictor_binding']:
            raise ValueError('predictor binding changed within episode')

    if prefix is None:
        emit('episode_started')
    while True:
        plan_id = 'plan_%d' % stage
        caused_by_request = episode['requests'][-1]['request_id'] if episode['requests'] else None
        if not reuse_plan:
            emit('input_started', plan_id, caused_by_request)
            begin = perf_counter()
            try:
                frozen_driver()
                fixed = fixed_generation
                fixed_generation = False
                if fixed and episode['plans']:
                    prepared = copy.deepcopy(episode['plans'][-1]['prepared'])
                    if numeric and uses_refinement(control_spec):
                        prepared['previous_plan'] = copy.deepcopy(current['waypoints'])
                        prepared['previous_parent_refs'] = copy.deepcopy(episode['plans'][-1]['output']['parent_refs'])
                elif numeric:
                    prior = episode['plans'][-1]['output'] if episode['plans'] else None
                    prepared = driver.prepare_input(features, motion, ledger, limits['receiver_spec'],
                        previous_plan=prior['waypoints'] if prior else None,
                        previous_parent_refs=prior['parent_refs'] if prior else ())
                else:
                    reserved = control_spec['refinement_slot_tokens'] if uses_refinement(control_spec) else 0
                    prepared = build_task_plan_input(driver.tokenizer, motion, ledger, feature_tokens,
                        limits['receiver_spec'], token_counter=token_counter, extra_prompt_reserve=reserved)
                if not numeric and uses_refinement(control_spec):
                    prepared = build_refinement_input(prepared, current, tokenizer=driver.tokenizer,
                        token_counter=token_counter, slot_tokens=control_spec['refinement_slot_tokens'])
            except Exception as exc:
                seconds = perf_counter() - begin
                costs['input_build_seconds'] += seconds
                episode['cost_events'].append(dict(kind='input_build', stage=stage, seconds=seconds, complete=False))
                costs['complete'] = False
                return fail('driver_error', exc, 'input_build', plan_id, caused_by_request)
            build_seconds = perf_counter() - begin
            costs['input_build_seconds'] += build_seconds
            episode['cost_events'].append(dict(kind='input_build', stage=stage, seconds=build_seconds, complete=True))
            plan = dict(plan_id=plan_id, stage=stage, request_id=caused_by_request,
                        ledger_snapshot=len(episode['ledger_snapshots']) - 1, prepared=copy.deepcopy(prepared),
                        status='started', output=None, driver_attempt_seconds=None)
            episode['plans'].append(plan)
            costs['driver_attempts'] += 1
            emit('driver_started', plan_id, caused_by_request)
            begin = perf_counter()
            try:
                output = driver.plan_prepared(features, prepared)
            except Exception as exc:
                seconds = perf_counter() - begin
                plan.update(status='driver_error', driver_attempt_seconds=seconds)
                costs['driver_attempt_seconds'] += seconds
                costs.update(complete=False, generation_seconds=None)
                episode['cost_events'].append(dict(kind='driver', stage=stage, plan_id=plan_id,
                    attempt_seconds=seconds, generation_cost=None, complete=False))
                return fail('driver_error', exc, 'driver_generation', plan_id, caused_by_request)
            seconds = perf_counter() - begin
            plan.update(output=copy.deepcopy(output), driver_attempt_seconds=seconds)
            costs['driver_attempt_seconds'] += seconds
            generation_cost = output.get('driver_cost' if numeric else 'q9_cost')
            complete = isinstance(generation_cost, dict) and all(generation_cost.get(k) is not None
                       for k in (('seconds', 'numeric_token_count', 'output_points', 'model_executed') if numeric
                                 else ('seconds', 'input_tokens', 'output_tokens', 'feature_tokens')))
            if numeric:
                try:
                    validate_numeric_cost(generation_cost)
                except (ValueError, TypeError, KeyError):
                    complete = False
            episode['cost_events'].append(dict(kind='driver', stage=stage, plan_id=plan_id,
                attempt_seconds=seconds, generation_cost=copy.deepcopy(generation_cost), complete=complete))
            if complete and costs['generation_seconds'] is not None:
                costs['generation_seconds'] += generation_cost['seconds']
            else:
                costs.update(complete=False, generation_seconds=None)
            try:
                frozen_driver()
                if numeric:
                    waypoints = validate_numeric_output(output, prepared, limits['execution_spec'])
                else:
                    if (output.get('status') != 'parsed' or output.get('q9_executed') is not True or
                            output.get('q8_executed') is not False or output.get('q8_raw')):
                        raise ValueError('driver did not produce a valid direct answer')
                    actual = parse_q9(output['q9_raw']).tolist()
                    if actual != output['waypoints']:
                        raise ValueError('parsed waypoints differ from actual raw driver output')
                    waypoints = validate_plan(actual, limits['execution_spec'])
            except Exception as exc:
                plan['status'] = 'invalid_plan'
                emit('driver_completed', plan_id, caused_by_request)
                return fail('invalid_plan', exc, 'driver_validation', plan_id, caused_by_request)
            plan['status'] = 'valid'
            previous, current = current, dict(plan_id=plan_id, waypoints=waypoints, raw=output.get('q9_raw'))
            episode['last_valid_plan_id'] = plan_id
            emit('driver_completed', plan_id, caused_by_request)
        else:
            prepared=episode['plans'][-1]['prepared']
            reuse_plan=False
        if repeat_generation(control_spec, episode):
            fixed_generation = True
            emit('same_evidence_generation', plan_id, caused_by_request)
            stage += 1
            continue
        control_event = dict(kind='control', stage=stage, timing_version=DECISION_TIMING_VERSION,
                             seconds=None, complete=False)
        episode['cost_events'].append(control_event)
        control_elapsed = 0.
        control_begin = perf_counter()
        try:
            state,candidates,preflight=decision_state(dict(episode,cost_events=episode['cost_events'][:-1]))
            actions=state['available_actions'];remaining=state['remaining_budget']['calls']
            forced_reason=None
            if not remaining:
                forced_reason='call_budget_exhausted'
            elif not candidates:
                forced_reason=('request_byte_limit' if preflight and all(p['reason']=='request_byte_limit' for p in preflight)
                               else 'byte_budget_exhausted')
            decision=(dict(tool='STOP',mode=None,reason=forced_reason) if forced_reason else policy(copy.deepcopy(state)))
            _check_json_native(decision)
            if (set(decision) != {'tool', 'mode', 'reason'} or not isinstance(decision['reason'], str) or
                    dict(tool=decision['tool'], mode=decision['mode']) not in actions):
                raise ValueError('policy must select an available action, not request content')
        except Exception as exc:
            control_event['seconds'] = control_elapsed + perf_counter() - control_begin
            costs['complete'] = False
            return fail('policy_error',exc,'request_preflight_or_policy',plan_id)
        recorded=dict(state=state,action=copy.deepcopy(decision),executed_action=None,
            policy_called=forced_reason is None,forced_reason=forced_reason,
            request_preflight=preflight,infeasible_actions=[p for p in preflight if not p['feasible']],execution_override=None)
        episode['decisions'].append(recorded)
        control_elapsed += perf_counter()-control_begin
        control_event['seconds'] = control_elapsed
        emit('decision',plan_id)
        control_begin=perf_counter()
        if decision['tool']!='STOP':
            try:
                frozen_driver()
                request=candidates[(decision['tool'],decision['mode'])]
                final_budget=request_budget(request,state['remaining_budget']['bytes'])
                size=final_budget['request_bytes']
                is_bundle=request['version'] in ('toolv2x_bundle_v1','toolv2x_bundle_v2')
                if not final_budget['feasible']:
                    recorded['execution_override']=final_budget['reason']
                    decision=dict(tool='STOP',mode=None,reason=final_budget['reason'])
                elif not is_bundle:
                    validate_task_request(request,{r['receipt_id']:r['record_refs'] for r in ledger['receipts']})
            except Exception as exc:
                control_event['seconds'] = control_elapsed + perf_counter() - control_begin
                costs['complete'] = False
                return fail('driver_error',exc,'frozen_configuration_or_dispatch',plan_id)
        recorded['executed_action']=copy.deepcopy(decision)
        control_event.update(seconds=control_elapsed+perf_counter()-control_begin, complete=True)
        if decision['tool'] == 'STOP':
            episode.update(status='completed', final_plan_id=current['plan_id'], stop_reason=decision['reason'])
            emit('STOP', plan_id)
            return episode

        episode['requests'].append(request)
        costs['query_attempts'] += 1
        emit('request_started', plan_id, request['request_id'])
        begin = perf_counter()
        try:
            response = service.query_bundle(request) if is_bundle else service.query_task(request)
        except Exception as exc:
            seconds = perf_counter() - begin
            record = next((r for r in (service.bundle_records if is_bundle else service.task_records) if r['request'] == request), None)
            charges = copy.deepcopy(record['cost']) if record is not None else None
            if charges:
                costs['request_bytes'] += charges['request_bytes']
                costs['response_bytes'] += charges['response_bytes']
            costs['service_seconds'] += seconds
            costs['complete'] = False
            episode['cost_events'].append(dict(kind='service', stage=stage, request_id=request['request_id'],
                attempt_seconds=seconds, service_cost=charges, provider_record=record, complete=False))
            if record and record.get('collection_reuse'):
                episode['cost_events'][-1]['collection_reuse']=copy.deepcopy(record['collection_reuse'])
            return fail('service_error', exc, 'service_query', plan_id, request['request_id'])
        seconds = perf_counter() - begin
        costs['service_seconds'] += seconds
        # Preserve the actual returned bytes before decoding or receiver execution.
        reply = response if isinstance(response, dict) else {}
        wire, charges = reply.get('wire'), reply.get('cost')
        saved_response = dict(request=copy.deepcopy(reply.get('request')), wire_type=type(wire).__name__,
            wire_hex=wire.hex() if isinstance(wire, bytes) else None, cost=copy.deepcopy(charges))
        if reply.get('collection_reuse'):
            saved_response['collection_reuse']=copy.deepcopy(reply['collection_reuse'])
        episode['responses'].append(saved_response)
        costs['request_bytes'] += size
        if isinstance(wire, bytes):
            costs['response_bytes'] += len(wire)
        else:
            costs['response_bytes'] = None  # A returned non-wire value is not a zero-byte response.
        complete = isinstance(charges, dict) and charges.get('complete') is True and isinstance(wire, bytes)
        costs['complete'] = costs['complete'] and complete
        episode['cost_events'].append(dict(kind='service', stage=stage, request_id=request['request_id'],
            attempt_seconds=seconds, service_cost=copy.deepcopy(charges), complete=complete))
        if reply.get('collection_reuse'):
            episode['cost_events'][-1]['collection_reuse']=copy.deepcopy(reply['collection_reuse'])
        emit('response_received', plan_id, request['request_id'])
        begin = perf_counter()
        try:
            if reply.get('request') != request:
                raise ValueError('transport returned a different request')
            if not isinstance(wire, bytes) or not isinstance(charges, dict):
                raise ValueError('transport must return actual wire bytes and a service cost record')
            if is_bundle:
                from tools.control_bundle import decode_bundle_response
                decoded=decode_bundle_response(wire,request)
                for primitive in decoded['primitive_responses']:
                    ledger=apply_response(ledger,primitive,predictor)
            else:
                ledger = apply_response(ledger, response, predictor)
        except Exception as exc:
            seconds = perf_counter() - begin
            costs['receiver_seconds'] += seconds
            costs['complete'] = False
            if isinstance(exc, EvidenceUpdateError):
                ledger = exc.ledger
            episode['ledger_snapshots'].append(copy.deepcopy(ledger))
            episode['cost_events'].append(dict(kind='receiver', stage=stage, request_id=request['request_id'],
                seconds=seconds, complete=False))
            return fail('receiver_error', exc, 'receiver_update', plan_id, request['request_id'])
        seconds = perf_counter() - begin
        costs['receiver_seconds'] += seconds
        episode['ledger_snapshots'].append(copy.deepcopy(ledger))
        episode['cost_events'].append(dict(kind='receiver', stage=stage, request_id=request['request_id'],
                                          seconds=seconds, complete=True))
        emit('receiver_completed', plan_id, request['request_id'])
        stage += 1
