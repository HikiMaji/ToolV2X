"""Derived evidence-use audit for validated structured numeric episodes."""
import copy
from collections import Counter

import numpy as np

from tools.task_spec import field_key


def _unique_refs(refs):
    result = []
    seen = set()
    for ref in refs:
        key = field_key(ref)
        if key not in seen:
            seen.add(key)
            result.append(copy.deepcopy(ref))
    return result


def _record_index(ledger):
    records = ledger['local_fields'] + ledger['acquired_fields'] + ledger['derived_fields']
    return {field_key(record['ref']): record for record in records}, records


def _closure(refs, records, order):
    keys = {field_key(ref) for ref in refs}
    pending = list(keys)
    while pending:
        for parent in records[pending.pop()].get('parent_refs', []):
            key = field_key(parent)
            if key not in keys:
                keys.add(key)
                pending.append(key)
    return [copy.deepcopy(record['ref']) for record in order if field_key(record['ref']) in keys]


def _coverage(prepared):
    report = prepared['admission_report']
    tensors = prepared['tensor_inputs']
    entities = {entity['entity_id']: entity for entity in prepared['entities']}
    observations, forecasts = [], []
    for group in report['field_groups'] + report['local_field_groups']:
        if group['use'] != 'tensor':
            continue
        location = group['tensor_locations'][0]
        entity = entities[group['entity_id']]
        if group['kind'] == 'observation':
            value = next(item for item in entity['observations']
                         if item['source_role'] == group['source_role'])
            mask = tensors['observation_mask'][location['entity']][location['source_slot']]
            steps = [index for index, valid in enumerate(mask) if valid]
            observations.append(dict(entity_id=group['entity_id'], source=value['source'],
                source_role=group['source_role'], primary_refs=copy.deepcopy(group['primary_refs']),
                anchor_ref=copy.deepcopy(group['anchor_ref']), tensor_location=copy.deepcopy(location),
                valid_steps=steps, timepoints_seconds=[value['value']['times'][index] for index in steps]))
        else:
            value = next(item for item in entity['forecast_sets']
                if item['source_role'] == group['source_role'] and
                item['primary_refs'] == group['primary_refs'])
            mask = np.asarray(tensors['forecast_mask'][location['entity']][location['forecast_set']], bool)
            modes = [index for index in range(mask.shape[0]) if mask[index].any()]
            per_mode = [int(mask[index].sum()) for index in modes]
            forecasts.append(dict(entity_id=group['entity_id'], source=value['source'],
                source_role=group['source_role'], context_scope=value['context_scope'],
                model_used=value['model_used'], primary_refs=copy.deepcopy(group['primary_refs']),
                anchor_ref=copy.deepcopy(group['anchor_ref']), tensor_location=copy.deepcopy(location),
                valid_modes=modes, valid_timepoints_per_mode=per_mode,
                valid_mode_timepoints=sum(per_mode),
                timepoints_seconds=copy.deepcopy(value['forecast_times'])))
    return observations, forecasts


def audit_structured_episode(episode):
    """Return one JSON-native evidence-use audit row per validated numeric plan."""
    from planning.driver_contract import validate_numeric_episode, validate_numeric_output
    validate_numeric_episode(episode)
    plans = episode['plans']
    rows = []
    previous_ledger = None
    previous_plan = None
    for plan in plans:
        prepared = plan['prepared']
        report = prepared['admission_report']
        ledger = episode['ledger_snapshots'][plan['ledger_snapshot']]
        records, record_order = _record_index(ledger)
        local_source = ledger['local_source']
        acquired_records = ledger['acquired_fields']
        derived_records = ledger['derived_fields']
        known_acquired = _unique_refs(record['ref'] for record in acquired_records)
        known_derived = _unique_refs(record['ref'] for record in derived_records)
        known_remote = _unique_refs(known_acquired + known_derived)
        known_local = _unique_refs(record['ref'] for record in ledger['local_fields'])
        previous_acquired_keys = (set() if previous_ledger is None else
            {field_key(record['ref']) for record in previous_ledger['acquired_fields']})
        previous_derived_keys = (set() if previous_ledger is None else
            {field_key(record['ref']) for record in previous_ledger['derived_fields']})
        previous_remote_keys = previous_acquired_keys | previous_derived_keys
        new_remote = [ref for ref in known_remote if field_key(ref) not in previous_remote_keys]
        previously_known_remote = [ref for ref in known_remote if field_key(ref) in previous_remote_keys]
        new_acquired = [ref for ref in known_acquired if field_key(ref) not in previous_acquired_keys]
        previous_acquired = [ref for ref in known_acquired if field_key(ref) in previous_acquired_keys]
        new_derived = [ref for ref in known_derived if field_key(ref) not in previous_derived_keys]
        previous_derived = [ref for ref in known_derived if field_key(ref) in previous_derived_keys]

        remote_groups = report['field_groups']
        local_groups = report['local_field_groups']
        direct_remote = _unique_refs(ref for group in remote_groups if group['use'] == 'tensor'
                                     for ref in group['primary_refs'])
        direct_local = _unique_refs(ref for group in local_groups if group['use'] == 'tensor'
                                    for ref in group['primary_refs'])
        direct = _unique_refs(direct_remote + direct_local)
        direct_keys = {field_key(ref) for ref in direct}
        direct_closure = _closure(direct, records, record_order)

        association = _unique_refs(ref for group in remote_groups + local_groups
            if group['kind'] == 'observation'
            for ref in [group['anchor_ref']] + group['primary_refs'])
        admitted_keys = {field_key(ref) for ref in report['admitted_field_refs']}
        association = [ref for ref in association if field_key(ref) in admitted_keys]
        association_keys = {field_key(ref) for ref in association}
        prior = _unique_refs(prepared['previous_parent_refs'])
        prior_closure = _closure(prior, records, record_order)
        indirect = [copy.deepcopy(ref) for ref in report['admitted_field_refs']
                    if field_key(ref) not in direct_keys]
        indirect_remote = [ref for ref in indirect if ref['provider'] != local_source]
        indirect_local = [ref for ref in indirect if ref['provider'] == local_source]
        prior_only = [ref for ref in prior_closure
                      if field_key(ref) not in direct_keys | association_keys]

        observation_coverage, forecast_coverage = _coverage(prepared)
        groups = remote_groups + local_groups
        # Derive reasons from validated admission decisions, not the legacy
        # catch-all dropped.reason. Excluded primary inputs may still feed a prior.
        dropped_keys = {field_key(ref) for ref in report['dropped_field_refs']}
        legacy_reasons = {field_key(item['ref']): item['reason'] for item in report['dropped']}
        non_direct = []
        for group in groups:
            if group['use'] == 'tensor':
                continue
            reason = ('ego_filter' if group['use'] == 'ego_filter' else
                      'entity_capacity' if group['tensor_index'] is None else
                      'forecast_set_capacity')
            for ref in group['primary_refs']:
                key = field_key(ref)
                non_direct.append(dict(ref=copy.deepcopy(ref), entity_id=group['entity_id'],
                    source_role=group['source_role'], reason=reason,
                    admitted_dependency=key in admitted_keys, dropped=key in dropped_keys,
                    legacy_reason=legacy_reasons.get(key)))
        entity_mask = prepared['tensor_inputs']['entity_mask']
        capacities = dict(max_entities=prepared['driver_spec']['max_entities'],
            max_forecast_sets_per_entity=prepared['driver_spec']['max_forecast_sets_per_entity'],
            tensor_entities=sum(entity_mask),
            ego_filtered_entities=len({g['entity_id'] for g in groups if g['use'] == 'ego_filter'}),
            entity_capacity_filtered_entities=len({g['entity_id'] for g in groups
                                                     if g['use'] == 'entity_capacity'}),
            structured_capacity_filtered_forecast_sets=sum(
                g['kind'] == 'forecast' and g['use'] == 'structured_capacity' for g in groups))
        receipts = [receipt['receipt_id'] for receipt in ledger['receipts']]
        previous_receipts = ([] if previous_ledger is None else
                             [receipt['receipt_id'] for receipt in previous_ledger['receipts']])
        previous_receipt_set = set(previous_receipts)
        output = plan.get('output')
        output_valid = None
        if output is not None:
            try:
                validate_numeric_output(output, prepared, episode['limits']['execution_spec'])
            except (ValueError, TypeError, KeyError, IndexError):
                output_valid = False
            else:
                output_valid = True
        first = previous_plan is None
        output_changed = None
        if not first and output is not None and previous_plan.get('output') is not None:
            output_changed = output.get('waypoints') != previous_plan['output'].get('waypoints')
        row = dict(plan_id=plan['plan_id'], stage=plan['stage'],
            ledger_snapshot=plan['ledger_snapshot'], receipt_ids=receipts,
            new_receipt_ids=[receipt for receipt in receipts if receipt not in previous_receipt_set],
            previous_receipt_ids=previous_receipts,
            known_remote_refs=known_remote, known_local_refs=known_local,
            new_remote_refs=new_remote, previously_known_remote_refs=previously_known_remote,
            known_acquired_remote_refs=known_acquired, new_acquired_remote_refs=new_acquired,
            previously_acquired_remote_refs=previous_acquired,
            known_receiver_derived_remote_refs=known_derived,
            new_receiver_derived_remote_refs=new_derived,
            previously_receiver_derived_remote_refs=previous_derived,
            admitted_refs=copy.deepcopy(report['admitted_field_refs']),
            dropped_refs=copy.deepcopy(report['dropped_field_refs']),
            exclusion_reason_version='toolv2x_structured_exclusion_reasons_v1',
            non_direct_primary_fields=non_direct,
            direct_primary_refs=direct, direct_remote_primary_refs=direct_remote,
            direct_local_primary_refs=direct_local,
            direct_dependency_closure_refs=direct_closure,
            association_selection_dependency_refs=association,
            prior_dependency_refs=prior, prior_dependency_closure_refs=prior_closure,
            indirect_only_refs=indirect, indirect_only_remote_refs=indirect_remote,
            indirect_only_local_refs=indirect_local, prior_only_refs=prior_only,
            counts=dict(known_remote_refs=len(known_remote), known_local_refs=len(known_local),
                new_remote_refs=len(new_remote),
                previously_known_remote_refs=len(previously_known_remote),
                known_acquired_remote_refs=len(known_acquired),
                new_acquired_remote_refs=len(new_acquired),
                previously_acquired_remote_refs=len(previous_acquired),
                known_receiver_derived_remote_refs=len(known_derived),
                new_receiver_derived_remote_refs=len(new_derived),
                previously_receiver_derived_remote_refs=len(previous_derived),
                admitted_refs=len(report['admitted_field_refs']),
                dropped_refs=len(report['dropped_field_refs']), direct_primary_refs=len(direct),
                direct_remote_primary_refs=len(direct_remote),
                direct_local_primary_refs=len(direct_local),
                direct_dependency_closure_refs=len(direct_closure),
                association_selection_dependency_refs=len(association),
                prior_dependency_refs=len(prior), prior_dependency_closure_refs=len(prior_closure),
                indirect_only_refs=len(indirect), indirect_only_remote_refs=len(indirect_remote),
                indirect_only_local_refs=len(indirect_local), prior_only_refs=len(prior_only)),
            capacities=capacities,
            association_status_counts=dict(sorted(Counter(
                entity['association']['status'] for entity in prepared['entities']).items())),
            observation_coverage=observation_coverage, forecast_coverage=forecast_coverage,
            tensor_changed_from_previous=None if first else
                prepared['tensor_inputs'] != previous_plan['prepared']['tensor_inputs'],
            prior_changed_from_previous=None if first else
                (prepared['previous_plan'], prepared['previous_parent_refs']) !=
                (previous_plan['prepared']['previous_plan'],
                 previous_plan['prepared']['previous_parent_refs']),
            output_available=output is not None, output_valid=output_valid,
            output_changed_from_previous=output_changed)
        rows.append(row)
        previous_ledger = ledger
        previous_plan = plan
    return rows
