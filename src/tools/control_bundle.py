"""Finite controls-only delegation over one actual canonical UTF-8 envelope.

The registered callable is trusted local configuration (as is the ego policy),
not a Python sandbox. It receives no executor, predictor, window or ego driver.
Primitive request/response sizes are construction/archive costs; only the outer
request and response sizes are network charges. Receiver APIs retain inner sizes.
"""
import copy
import json
from time import perf_counter

from probe.kinematic_tools import encode
from tools.task_spec import (CONTROL_TASK_VERSION, ExecutionSpec, _check_json_native,
                             _keys, _integer, _text, _version, field_key,
                             validate_plan, validate_task_request, _request_spec)


def _public_summary(summary, request):
    allowed = {'ego_motion', 'local_evidence', 'admission_report', 'current_plan',
               'previous_plan', 'acquired_fields', 'derived_fields', 'remaining_budget'}
    if not isinstance(summary, dict) or set(summary) - allowed:
        raise ValueError('unsupported causal public-summary fields')
    known = {field_key(x['ref']) for x in request['acquired_field_manifest']}
    def check(value):
        if isinstance(value, dict):
            if any(k in ('features', 'feature', 'gt', 'labels', 'ground_truth', 'future',
                         'future_poses', 'ego_private', 'provider_state', 'unexecuted_forecast')
                   or k.startswith('gt_') or k.endswith('_features') for k in value):
                raise ValueError('private features or future labels cannot enter a bundle summary')
            if 'track_handle' in value:
                key = field_key(value)
                if (key[1:3] != (request['scene'], request['g']) or
                        key[0] == request['provider'] and key not in known):
                    raise ValueError('summary contains an unreturned peer handle or wrong time')
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(summary)


def validate_bundle_envelope(envelope):
    """Validate only sent content and public caps, without any private access.

    Receipt authentication is separately performed against VehicleTools' registry.
    wrapper_reserve_bytes is run configuration; final actual bytes are always
    checked, even when the selected reserve proves insufficient.
    """
    _check_json_native(envelope)
    _keys(envelope, ('version', 'request_id', 'bundle_id', 'first_request', 'public_summary',
                     'candidates', 'continuation_policy_id', 'limits'), 'bundle envelope')
    if (envelope['version'] != 'toolv2x_bundle_v1' or
            _text(envelope['request_id']) != _text(envelope['bundle_id'])):
        raise ValueError('invalid bundle protocol or request identity')
    _version(envelope['continuation_policy_id'])
    limits = envelope['limits']
    _keys(limits, ('execution_spec', 'max_calls', 'max_candidates', 'remaining_bytes',
                   'primitive_response_caps', 'wrapper_reserve_bytes'), 'bundle limits')
    spec = ExecutionSpec.from_dict(limits['execution_spec'])
    if _integer(limits['max_calls'], 1) > 2:
        raise ValueError('bundle permits at most two primitive attempts')
    _integer(limits['max_candidates'], 1)
    _integer(limits['remaining_bytes'])
    _integer(limits['wrapper_reserve_bytes'])
    caps = limits['primitive_response_caps']
    if (not isinstance(caps, list) or len(caps) != limits['max_calls'] or
            any(_integer(cap, 1) > spec.max_response_bytes for cap in caps)):
        raise ValueError('one frozen response capacity per possible primitive is required')
    if sum(caps) + limits['wrapper_reserve_bytes'] > spec.max_response_bytes:
        raise ValueError('primitive capacities and outer wrapper exceed the external response cap')
    first = envelope['first_request']
    _request_spec(first)
    if first['execution_spec'] != limits['execution_spec']:
        raise ValueError('bundle and primitive execution specifications differ')
    candidates = envelope['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= limits['max_candidates']:
        raise ValueError('candidate capacity must be fixed before transmission')
    ids = set()
    for candidate in candidates:
        _keys(candidate, ('candidate_id', 'waypoints'), 'bundle candidate')
        name = _text(candidate['candidate_id'])
        if name in ids:
            raise ValueError('duplicate bundle candidate identity')
        ids.add(name)
        validate_plan(candidate['waypoints'], spec)
    if not any(c['waypoints'] == first['tau_new'] for c in candidates):
        raise ValueError('first task must use a sent candidate')
    _public_summary(envelope['public_summary'], first)
    size = len(encode(envelope))
    if (size > spec.max_request_bytes or
            size + spec.max_response_bytes > min(limits['remaining_bytes'], spec.max_episode_bytes)):
        raise ValueError('complete external envelope or reserved response exceeds the remaining budget')
    return spec


def bundle_actions(envelope):
    actions = [dict(tool='STOP', mode=None, candidate_id=None)]
    if envelope['limits']['max_calls'] < 2:
        return actions
    first = envelope['first_request']
    for candidate in envelope['candidates']:
        modes = ['current', 'old', 'union']
        if candidate['waypoints'] != first['tau_new']:
            modes.append('change')
        for tool in ('P', 'F'):
            actions.extend(dict(tool=tool, mode=mode, candidate_id=candidate['candidate_id'])
                           for mode in modes)
    return actions


def _decision(value, actions):
    _check_json_native(value)
    _keys(value, ('tool', 'mode', 'candidate_id', 'reason'), 'bundle continuation')
    if (not isinstance(value['reason'], str) or
            {k: value[k] for k in ('tool', 'mode', 'candidate_id')} not in actions):
        raise ValueError('continuation must select a sent candidate and an available task or STOP')
    return copy.deepcopy(value)


def continuation_request(envelope, first_packet, decision):
    """Construct the second task solely from sent candidates and issued fields."""
    first = envelope['first_request']
    candidate = next(c for c in envelope['candidates'] if c['candidate_id'] == decision['candidate_id'])
    request = copy.deepcopy(first)
    request.pop('roi', None)
    request.update(version=CONTROL_TASK_VERSION, request_id=envelope['bundle_id'] + ':second',
        tool=decision['tool'], mode=decision['mode'], tau_new=copy.deepcopy(candidate['waypoints']),
        tau_old=copy.deepcopy(first['tau_new']) if decision['mode'] in ('change', 'old', 'union') else None)
    request['acquired_field_manifest'] += [dict(receipt_id=first_packet['receipt_id'], ref=copy.deepcopy(r['ref']))
                                          for r in first_packet['records']]
    return request


def decode_bundle_response(wire, envelope):
    """Rebuild receiver inputs from the actual outer wire, never a return sidecar."""
    from tools.vehicle import decode_task_response
    spec = validate_bundle_envelope(envelope)
    if not isinstance(wire, bytes) or len(wire) > spec.max_response_bytes:
        raise ValueError('bundle response must be UTF-8 bytes within the full response cap')
    packet = json.loads(wire.decode('utf-8'))
    # Canonical encoding rejects duplicate keys, non-finite numbers, trailing JSON
    # and alternative representations that would lose exact inner-wire recovery.
    if encode(packet) != wire:
        raise ValueError('bundle wire must use canonical JSON encoding')
    _keys(packet, ('version', 'request_id', 'bundle_id', 'continuation', 'responses'), 'bundle response')
    if (packet['version'] != 'toolv2x_bundle_response_v1' or
            packet['bundle_id'] != envelope['bundle_id'] or packet['request_id'] != envelope['request_id']):
        raise ValueError('bundle response identity mismatch')
    decision = _decision(packet['continuation'], bundle_actions(envelope))
    count = 1 if decision['tool'] == 'STOP' else 2
    if not isinstance(packet['responses'], list) or len(packet['responses']) != count:
        raise ValueError('bundle continuation and actual primitive count disagree')
    responses, expected, first_packet = [], envelope['first_request'], None
    known = {}
    for i, item in enumerate(packet['responses']):
        _keys(item, ('packet', 'cost'), 'bundle primitive response')
        if i:
            expected = continuation_request(envelope, first_packet, decision)
        primitive_wire = encode(item['packet'])
        if len(primitive_wire) > envelope['limits']['primitive_response_caps'][i]:
            raise ValueError('primitive exceeded its pre-send response capacity')
        decoded = decode_task_response(primitive_wire, expected)
        # The first manifest is externally authenticated by the provider; every
        # new continuation reference must be backed by the actual first packet.
        if i:
            previous_refs = envelope['first_request']['acquired_field_manifest']
            for entry in previous_refs:
                known.setdefault(entry['receipt_id'], []).append(entry['ref'])
            validate_task_request(expected, known)
        cost = item['cost']
        _keys(cost, ('request_bytes', 'response_bytes', 'service_seconds', 'model_seconds',
                     'model_targets_computed', 'fallback_targets_computed', 'returned_targets',
                     'within_decision_forecast_cache_hit', 'complete'), 'primitive cost')
        if (cost.get('complete') is not True or
                cost.get('request_bytes') != len(encode(expected)) or
                cost.get('response_bytes') != len(primitive_wire)):
            raise ValueError('primitive archive bytes and reported costs disagree')
        for name in ('request_bytes', 'response_bytes', 'model_targets_computed',
                     'fallback_targets_computed', 'returned_targets'):
            _integer(cost[name])
        if (cost['returned_targets'] != len(decoded['ranking']) or
                type(cost['within_decision_forecast_cache_hit']) is not bool):
            raise ValueError('primitive target counts or cache status are invalid')
        for name in ('service_seconds', 'model_seconds'):
            value = cost.get(name)
            if type(value) not in (int, float) or value < 0:
                raise ValueError('invalid primitive execution costs')
        responses.append(dict(request=copy.deepcopy(expected), wire=primitive_wire, cost=copy.deepcopy(cost)))
        known[decoded['receipt_id']] = [r['ref'] for r in decoded['records']]
        first_packet = decoded if i == 0 else first_packet
    return dict(packet=packet, primitive_responses=responses, continuation=decision)


def execute_bundle(service, envelope):
    """VehicleTools implementation; private state never crosses into policy input."""
    from tools.vehicle import decode_task_response
    begin = perf_counter()
    envelope = copy.deepcopy(envelope)
    spec = validate_bundle_envelope(envelope)
    if service._task_records or service._bundle_records:
        raise ValueError('one fresh provider episode is required for one external bundle')
    if envelope['continuation_policy_id'] not in service._bundle_policies:
        raise ValueError('unregistered continuation policy')
    first = envelope['first_request']
    validate_task_request(first, service._task_receipts)
    if (first['provider'], first['scene'], first['g']) != (service.provider, service.scene, service.g):
        raise ValueError('bundle provider/source/time mismatch')
    policy = service._bundle_policies[envelope['continuation_policy_id']]
    request_wire = encode(envelope)
    # The actual transport boundary is one serialization/deserialization. No
    # later ego callback or arbitrary request code is invoked by the provider.
    envelope = json.loads(request_wire.decode('utf-8'))
    cost = dict(request_bytes=len(request_wire), response_bytes=0, rpc_rounds=1,
        capability_calls=0, service_seconds=0., continuation_seconds=0.,
        internal_request_seconds=0., model_seconds=0., model_targets_computed=0,
        fallback_targets_computed=0, returned_targets=0, complete=False)
    record = dict(request=copy.deepcopy(envelope), request_wire_hex=request_wire.hex(),
        wire_hex=None, status='started', cost=cost, primitive_responses=[], primitive_records=[])
    service._bundle_records.append(record)
    service._task_bytes += len(request_wire)
    decision = dict(tool='STOP', mode=None, candidate_id=None, reason='call_budget_exhausted')
    packet = dict(version='toolv2x_bundle_response_v1', request_id=envelope['request_id'],
                  bundle_id=envelope['bundle_id'], continuation=decision, responses=[])
    phase = 'first_primitive'
    def run(request, i):
        # Failed invocation still spends a capability attempt; its executor record
        # carries partial model timing/counts when prediction was actually started.
        cost['capability_calls'] += 1
        response = service._execute_task(request, response_cap=envelope['limits']['primitive_response_caps'][i],
                                         internal=True)
        decoded = decode_task_response(response['wire'], request)
        if encode(decoded) != response['wire']:
            raise ValueError('primitive executor returned a noncanonical wire')
        record['primitive_responses'].append(dict(request=copy.deepcopy(response['request']),
            wire_hex=response['wire'].hex(), cost=copy.deepcopy(response['cost'])))
        packet['responses'].append(dict(packet=decoded, cost=copy.deepcopy(response['cost'])))
        return decoded
    try:
        first_packet = run(first, 0)
        if envelope['limits']['max_calls'] > 1:
            phase = 'continuation'
            visible = dict(envelope=copy.deepcopy(envelope), first_response=copy.deepcopy(first_packet),
                           available_actions=bundle_actions(envelope))
            start = perf_counter()
            try:
                decision = _decision(policy(visible), bundle_actions(envelope))
            finally:
                cost['continuation_seconds'] = perf_counter() - start
            if decision['tool'] != 'STOP':
                phase = 'second_request_construction'
                start = perf_counter()
                try:
                    second = continuation_request(envelope, first_packet, decision)
                    validate_task_request(second, service._task_receipts)
                finally:
                    cost['internal_request_seconds'] = perf_counter() - start
                phase = 'second_primitive'
                run(second, 1)
        phase = 'outer_response_packing'
        packet['continuation'] = decision
        wire = encode(packet)
        # Check every wrapper and inner cost field, rather than assuming the
        # chosen run reserve covered them. Failure charges executed computations.
        if (len(wire) > spec.max_response_bytes or
                len(request_wire) + len(wire) > min(envelope['limits']['remaining_bytes'], spec.max_episode_bytes)):
            raise ValueError('complete aggregate bundle response exceeds the external byte cap')
        decoded = decode_bundle_response(wire, envelope)
        record.update(status='completed', wire_hex=wire.hex(), response=copy.deepcopy(packet))
        service._task_bytes += len(wire)
        cost.update(response_bytes=len(wire), complete=True)
    except Exception as exc:
        record.update(status='error', error=dict(stage=phase, error_type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        record['primitive_records'] = service.task_records
        charged = [r['cost'] for r in record['primitive_records']]
        for name in ('model_seconds', 'model_targets_computed', 'fallback_targets_computed', 'returned_targets'):
            values = [r.get(name) for r in charged]
            cost[name] = sum(values) if all(v is not None for v in values) else None
        cost['service_seconds'] = perf_counter() - begin
    return dict(request=copy.deepcopy(envelope), request_wire=request_wire, wire=wire,
                cost=copy.deepcopy(cost), primitive_responses=decoded['primitive_responses'],
                provider_record=copy.deepcopy(record))
