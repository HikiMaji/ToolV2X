"""Torch-free validation for numeric structured-driver outputs."""
import math

from planning.structured_inputs import validate_structured_prepared
from tools.task_spec import _check_json_native, _keys, validate_plan


def validate_numeric_output(output, prepared, execution_spec):
    """Return the six actual numeric waypoints after strict output validation."""
    _check_json_native(output)
    _keys(output, ('output_version', 'driver_kind', 'status', 'waypoints',
                   'prepared_input', 'driver_cost', 'parent_refs'),
          'numeric driver output')
    if (output['output_version'] != 'toolv2x_numeric_plan_v1' or
            output['driver_kind'] != 'structured' or output['status'] != 'valid'):
        raise ValueError('invalid numeric driver output identity')
    validate_structured_prepared(prepared)
    validate_structured_prepared(output['prepared_input'])
    if output['prepared_input'] != prepared:
        raise ValueError('numeric output does not bind the exact prepared input')
    expected_refs = prepared['admission_report']['admitted_field_refs']
    if output['parent_refs'] != expected_refs:
        raise ValueError('numeric output parent closure changed')
    validate_numeric_cost(output['driver_cost'])
    return validate_plan(output['waypoints'], execution_spec)


def validate_numeric_cost(cost):
    """Validate measured numeric computation without interpreting trajectory validity."""
    _keys(cost, ('seconds', 'numeric_token_count', 'output_points', 'model_executed'),
          'numeric driver cost')
    seconds = cost['seconds']
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or
            type(cost['numeric_token_count']) is not int or cost['numeric_token_count'] <= 0 or
            cost['output_points'] != 6 or cost['model_executed'] is not True):
        raise ValueError('invalid numeric driver cost')


def validate_driver_binding(limits, provenance):
    """Reject a mismatched driver before local preparation or private service access."""
    from planning.structured_inputs import StructuredDriverSpec
    from tools.task_spec import _version_pair
    numeric = limits['version'] == 'toolv2x_interaction_v2'
    if numeric != (provenance.get('driver_kind') == 'structured'):
        raise ValueError('driver kind and interaction version differ')
    if numeric:
        _check_json_native(provenance)
        spec = StructuredDriverSpec.from_dict(provenance.get('driver_spec'))
        version = provenance.get('model_version')
        if (spec.to_dict() != limits['receiver_spec'] or provenance.get('decoding') != 'numeric' or
                not isinstance(version, dict) or set(version) != {'name', 'revision', 'training'}):
            raise ValueError('numeric driver specification or model identity differs')
        identity = {k: version[k] for k in ('name', 'revision')}
        _version_pair(identity)
        training = version['training']
        if (identity != limits['driver_version'] or not isinstance(training, dict) or
                not isinstance(training.get('status'), str) or not training['status'] or
                type(training.get('optimizer_steps')) is not int or training['optimizer_steps'] < 0):
            raise ValueError('numeric driver version and actual training metadata required')
    return numeric


def validate_numeric_episode(ep):
    """Rebuild each numeric Z from its paid ledger, or verify its frozen predecessor.

    No model, feature file, predictor, or future label is read here. Failed attempts
    retain their prepared input; only records claiming success require valid output.
    """
    import copy
    import numpy as np
    from planning.structured_inputs import StructuredDriverSpec, build_structured_plan_input
    from planning.method_controls import repeat_generation, uses_refinement
    validate_driver_binding(ep['limits'], ep['driver_provenance'])
    if ep.get('execution_kind') != 'structured_numeric' or ep.get('feature_tokens') != 0:
        raise ValueError('numeric episode execution identity changed')
    spec = StructuredDriverSpec.from_dict(ep['limits']['receiver_spec'])
    plans = ep['plans']
    if len(plans) > 3 or len(ep['requests']) > ep['limits']['max_calls']:
        raise ValueError('numeric episode exceeds shared call limits')
    history = plans[0]['prepared']['ego_history_used'] if plans else None
    local = ep['ledger_snapshots'][0]['local_fields']
    if ep.get('numeric_scene_tokens') not in (220, 440):
        raise ValueError('invalid numeric scene token count')
    receipts_by_snapshot = {0: []}
    for i, plan in enumerate(plans):
        prepared = plan['prepared']
        validate_structured_prepared(prepared)
        snapshot = plan['ledger_snapshot']
        if type(snapshot) is not int or not 0 <= snapshot < len(ep['ledger_snapshots']):
            raise ValueError('numeric Z has no actual ledger snapshot')
        ledger = ep['ledger_snapshots'][snapshot]
        if (ledger['local_fields'] != local or ledger['version'] != 'toolv2x_evidence_ledger_v2' or
                prepared['ego_history_used'] != history):
            raise ValueError('fixed numeric local inputs changed')
        if snapshot not in receipts_by_snapshot:
            from tools.vehicle import decode_task_response
            receipts = []
            for request, saved in zip(ep['requests'][:snapshot], ep['responses'][:snapshot]):
                wire = bytes.fromhex(saved['wire_hex'])
                if saved['request'] != request:
                    raise ValueError('numeric paid response request changed')
                if request['version'] in ('toolv2x_bundle_v1', 'toolv2x_bundle_v2'):
                    from tools.control_bundle import decode_bundle_response
                    primitives = decode_bundle_response(wire, request)['primitive_responses']
                else:
                    primitives = [dict(request=request, wire=wire, cost=saved['cost'])]
                for primitive in primitives:
                    packet = decode_task_response(primitive['wire'], primitive['request'])
                    receipts.append(dict(receipt_id=packet['receipt_id'], request=primitive['request'],
                        response=packet, record_refs=[r['ref'] for r in packet['records']],
                        wire_text=primitive['wire'].decode('utf-8'), cost=primitive['cost']))
            receipts_by_snapshot[snapshot] = receipts
        if ledger['receipts'] != receipts_by_snapshot[snapshot]:
            raise ValueError('numeric receipt ledger differs from the actual paid wire')
        prior = plans[i-1]['output'] if i else None
        if i and snapshot == plans[i-1]['ledger_snapshot']:
            prefix = dict(ep, plans=plans[:i], requests=ep['requests'][:snapshot], responses=ep['responses'][:snapshot])
            if not repeat_generation(ep.get('control_spec'), prefix):
                raise ValueError('unregistered repeated numeric attempt')
            expected = copy.deepcopy(plans[i-1]['prepared'])
            if uses_refinement(ep.get('control_spec')):
                expected['previous_plan'] = copy.deepcopy(prior['waypoints'])
                expected['previous_parent_refs'] = copy.deepcopy(prior['parent_refs'])
        else:
            if snapshot != (0 if i == 0 else plans[i-1]['ledger_snapshot'] + 1):
                raise ValueError('numeric evidence snapshot skipped or reordered')
            expected = build_structured_plan_input(ep['ego_motion'], ledger, spec,
                ego_history=history, previous_plan=prior['waypoints'] if prior else None,
                previous_parent_refs=prior['parent_refs'] if prior else ())
        if prepared != expected:
            raise ValueError('numeric Z differs from its actual ledger or frozen prior')
        output = plan.get('output')
        if output is not None and (output.get('prepared_input') != prepared or
                output.get('parent_refs') != prepared['admission_report']['admitted_field_refs']):
            raise ValueError('numeric output and archived Z differ')
        if plan['status'] == 'valid':
            validate_numeric_output(output, prepared, ep['limits']['execution_spec'])
            tensors = prepared['tensor_inputs']
            count = (ep['numeric_scene_tokens'] + sum(tensors['entity_mask']) +
                     int(np.asarray(tensors['forecast_mask'], bool).any(axis=-1).sum()) + 9)
            if output['driver_cost']['numeric_token_count'] != count:
                raise ValueError('numeric token count differs from admitted model tokens')
