#!/usr/bin/env python3
"""Read-only token/admission audit for the frozen T9 two-frame artifacts.

This loads a tokenizer only. It never constructs or calls GoT, MTR, or any
other model. The receiver itself remains unchanged; this wrapper reuses its
unit projection and prompt codec, then exposes every greedy token attempt.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from planning.context import target_round_robin


EXPECTED_TASKS = {
    'g5526_Ego', 'g5526_alternating', 'g5526_one_shot',
    'g7007_Ego', 'g7007_alternating', 'g7007_one_shot',
}
GROUPS = ('history', 'receiver_subset_forecast', 'provider_full_forecast')


def admit_in_order(units, input_tokens_for, max_input_tokens):
    """Greedily admit in the supplied order and expose exact attempted totals."""
    selected = []
    rows = []
    cumulative = input_tokens_for(selected)
    for order, unit in enumerate(units, 1):
        attempt = input_tokens_for(selected + [unit])
        admitted = attempt <= max_input_tokens
        if admitted:
            selected.append(unit)
            cumulative = attempt
        rows.append(dict(order=order, attempt_input_tokens=attempt, admitted=admitted,
                         cumulative_input_tokens=cumulative))
    return selected, rows


def receipt_stages(episode):
    """Map each receipt to the first saved plan stage that contains it."""
    result = {}
    for plan in episode['plans']:
        snapshot = episode['ledger_snapshots'][plan['ledger_snapshot']]
        for receipt in snapshot['receipts']:
            result.setdefault(receipt['receipt_id'], plan['stage'])
    return result


def receipt_stage_metadata(episode, provider_records):
    """Bind primitive receipts to outer request events and later visible plans."""
    started = {}
    for event in episode['events']:
        if event['kind'] == 'request_started':
            request_id = event['request_id']
            if request_id in started:
                raise ValueError('duplicate outer request_started event')
            started[request_id] = event['stage']
    request_ids = {request['request_id'] for request in episode['requests']}
    if set(started) != request_ids:
        raise ValueError('outer request events differ from saved episode requests')

    visible = receipt_stages(episode)
    bundle_outer = {}
    for bundle in provider_records['bundle_records']:
        outer_id = bundle['request']['request_id']
        if outer_id not in started:
            raise ValueError('provider bundle has no outer request event')
        for primitive in bundle['primitive_records']:
            receipt_id = primitive['response']['receipt_id']
            if receipt_id in bundle_outer and bundle_outer[receipt_id] != outer_id:
                raise ValueError('primitive receipt belongs to multiple bundles')
            bundle_outer[receipt_id] = outer_id

    result = {}
    for primitive in provider_records['primitive_records']:
        receipt_id = primitive['response']['receipt_id']
        primitive_id = primitive['request']['request_id']
        outer_id = primitive_id if primitive_id in started else bundle_outer.get(receipt_id)
        if outer_id is None or receipt_id not in visible:
            raise ValueError('primitive receipt lacks request/visibility provenance')
        request_stage = started[outer_id]
        first_visible = visible[receipt_id]
        visible_plans = [plan for plan in episode['plans']
                         if plan['stage'] == first_visible and plan['request_id'] == outer_id]
        if first_visible != request_stage + 1 or len(visible_plans) != 1:
            raise ValueError('request event and first visible plan stage are inconsistent')
        result[receipt_id] = dict(request_stage=request_stage,
            first_visible_plan_stage=first_visible, outer_request_id=outer_id)
    if set(result) != set(visible) or set(bundle_outer) - set(result):
        raise ValueError('provider receipts differ from saved ledger visibility')
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError('CSV rows do not share one explicit schema: ' + path.name)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _field_group(unit):
    if unit['kind'] == 'history':
        return 'history'
    scope = unit['object']['field_metadata']['forecast']['context_scope']
    if scope == 'receiver_acquired_subset':
        return 'receiver_subset_forecast'
    if scope == 'provider_full_at_t':
        return 'provider_full_forecast'
    raise ValueError('unknown forecast context scope: ' + repr(scope))


def _unit_id(unit):
    return _json(dict(anchor_ref=unit['anchor_ref'], primary_refs=unit['primary_refs']))


def _remote_view(units, observations, rounded):
    """Render a chosen ordered unit list with the receiver's public merge rule."""
    objects, positions = [], []
    for unit in units:
        obj = rounded(copy.deepcopy(unit['object']))
        kind = unit['kind']
        context = obj['field_metadata'][kind]['context_version']
        compatible = next((i for i, other in enumerate(objects)
            if (other['source'], other['track_id']) == (obj['source'], obj['track_id'])
            and kind not in other['field_metadata']
            and all(meta['context_version'] == context for meta in other['field_metadata'].values())), None)
        if compatible is None:
            positions.append(len(objects))
            objects.append(obj)
        else:
            positions.append(compatible)
            metadata = dict(objects[compatible]['field_metadata'], **obj['field_metadata'])
            objects[compatible].update(obj)
            objects[compatible]['field_metadata'] = metadata
    return dict(evidence_version='toolv2x_driver_evidence_v2',
                as_of_g=units[0]['primary_refs'][0]['g'] if units else None,
                coordinate_frame='ego_at_t', observations=copy.deepcopy(observations),
                objects=objects), positions


def _observations(ledger):
    values = []
    for provider in sorted({receipt['request']['provider'] for receipt in ledger['receipts']}):
        complete = any(receipt['request']['provider'] == provider and
                       'context_certificate' in receipt['response'] for receipt in ledger['receipts'])
        values.append(dict(provider=provider, coverage='not_established', history_complete=complete))
    return values


def _request_origins(ledger, record, field_key, stage_metadata):
    receipts = {receipt['receipt_id']: receipt for receipt in ledger['receipts']}
    acquired = {field_key(item['ref']): item for item in ledger['acquired_fields']}
    if record['origin'] == 'remote':
        receipt_ids = record['receipt_ids']
        parent_refs = []
    else:
        parent_refs = record['parent_refs']
        receipt_ids = sorted({receipt_id for ref in parent_refs
                              for receipt_id in acquired[field_key(ref)]['receipt_ids']})
    origins = []
    target = record['ref']['track_handle']
    for receipt_id in receipt_ids:
        receipt = receipts[receipt_id]
        request = receipt['request']
        ranks = receipt['response']['ranking']
        rank = next((i for i, value in enumerate(ranks, 1)
                     if value['track_handle'] == target), None)
        stages = stage_metadata[receipt_id]
        origins.append(dict(receipt_id=receipt_id, request_id=request['request_id'],
            outer_request_id=stages['outer_request_id'], request_stage=stages['request_stage'],
            first_visible_plan_stage=stages['first_visible_plan_stage'],
            tool=request['tool'], mode=request['mode'],
            provider_rank=rank, provider_ranking=[value['track_handle'] for value in ranks]))
    return origins, parent_refs


def _group_counts(units):
    counts = Counter(_field_group(unit) for unit in units)
    return {name: counts[name] for name in GROUPS}


def _source_records(ledger, field_key):
    return {field_key(record['ref']): record
            for record in ledger['acquired_fields'] + ledger['derived_fields']}


def _provider_rows(task_name, episode, provider_records, field_key, stage_metadata):
    receipts = episode['ledger_snapshots'][-1]['receipts']
    primitive = provider_records['primitive_records']
    if len(primitive) != len(receipts):
        raise AssertionError('%s provider/ledger receipt count differs' % task_name)
    rows = []
    for call_index, (saved, receipt) in enumerate(zip(primitive, receipts), 1):
        for name in ('request', 'response', 'cost'):
            if saved[name] != receipt[name]:
                raise AssertionError('%s provider %s differs from ledger receipt' % (task_name, name))
        request, response = receipt['request'], receipt['response']
        stages = stage_metadata[receipt['receipt_id']]
        records = {field_key(item['ref']): item for item in response['records']}
        references = {field_key(item['ref']): item for item in response['references']}
        for rank, ranking in enumerate(response['ranking'], 1):
            handle = ranking['track_handle']
            target_records = [item for key, item in records.items() if key[3] == handle]
            target_references = [item for key, item in references.items() if key[3] == handle]
            rows.append(dict(task=task_name, frame=episode['g'], arm=episode['branch_id'],
                call_index=call_index, request_id=request['request_id'],
                outer_request_id=stages['outer_request_id'], request_stage=stages['request_stage'],
                first_visible_plan_stage=stages['first_visible_plan_stage'],
                tool=request['tool'], mode=request['mode'], provider=request['provider'],
                provider_rank=rank, track_handle=handle, ranking_score=ranking['score'],
                proxy_status=ranking['proxy_status'],
                response_record_refs=_json([item['ref'] for item in target_records]),
                response_reference_refs=_json([item['ref'] for item in target_references]),
                response_record_kinds=_json([item['ref']['field_kind'] for item in target_records]),
                response_reference_kinds=_json([item['ref']['field_kind'] for item in target_references])))
    return rows


def audit(input_root, tokenizer_path, output_dir, evidence_dir=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'src'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    from planning.context import build_task_plan_input
    from planning.evidence import remote_units
    from planning.inputs import make_prompt
    from planning.v2vgot import prompt_tokens, rounded
    from tools.task_spec import field_key

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False, local_files_only=True)
    released_tokenizer_limit = tokenizer.model_max_length
    # The frozen receiver used 4096; this only silences the legacy 2048 warning
    # while counting and does not truncate or alter token IDs.
    tokenizer.model_max_length = max(released_tokenizer_limit, 4096)
    tasks = {path.stem: json.loads(path.read_text()) for path in sorted((input_root / 'tasks').glob('*.json'))}
    if set(tasks) != EXPECTED_TASKS:
        raise ValueError('expected exactly the frozen six T9 tasks; got ' + repr(sorted(tasks)))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refuse to overwrite non-empty audit output: ' + str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_rows, unit_rows, simulated_rows = [], [], []
    provider_rows, dropped_rows, reproducibility = [], [], []
    final_by_frame = {}
    for task_name, task in sorted(tasks.items()):
        if task['status'] != 'completed':
            raise AssertionError(task_name + ' is not completed')
        episode = task['episode']
        if episode['gt_labels_read'] or episode['feature_tokens'] != 540:
            raise AssertionError(task_name + ' violates frozen no-label/feature-token contract')
        provider_path = input_root / 'inputs' / task_name / 'provider_records.json'
        provider_records = json.loads(provider_path.read_text())
        stage_metadata = receipt_stage_metadata(episode, provider_records)
        provider_rows.extend(_provider_rows(task_name, episode, provider_records, field_key, stage_metadata))
        task_checks = []
        for plan_index, plan in enumerate(episode['plans']):
            ledger = episode['ledger_snapshots'][plan['ledger_snapshot']]
            saved = plan['prepared']
            rebuilt = build_task_plan_input(tokenizer, episode['ego_motion'], ledger,
                episode['feature_tokens'], limits=episode['limits']['receiver_spec'])
            checks = {
                name: rebuilt[name] == saved[name] for name in (
                    'input_layout', 'decoding', 'ego_motion', 'evidence_used',
                    'remote_evidence_used', 'receiver_spec', 'admission_report',
                    'evidence_selection', 'q9_prompt')
            }
            counted = len(prompt_tokens(tokenizer, saved['q9_prompt'])) - 1 + episode['feature_tokens']
            checks['saved_prompt_token_total'] = counted == saved['evidence_selection']['input_tokens']
            if not all(checks.values()):
                raise AssertionError('%s plan %d reproducibility failed: %s' %
                                     (task_name, plan_index, [key for key, value in checks.items() if not value]))
            task_checks.append(dict(plan_index=plan_index, stage=plan['stage'], checks=checks,
                                    input_tokens=counted))

            units = remote_units(ledger)
            actual_group_counts = _group_counts([])
            simulated_group_counts = _group_counts([])
            actual_selected, simulated_selected = [], []
            actual_attempts, simulated_attempts = [], []
            ego_tokens = counted
            empty_remote_tokens = None
            if units:
                observations = _observations(ledger)
                def remote_for(values):
                    view, positions = _remote_view(values, observations, rounded)
                    view['as_of_g'] = ledger['g']
                    return view, positions
                def input_tokens_for(values):
                    remote, _ = remote_for(values)
                    prompt = make_prompt('Trajectory', episode['ego_motion'], saved['evidence_used'],
                                         evidence_format='compact', remote_evidence=remote)
                    return len(prompt_tokens(tokenizer, prompt)) - 1 + episode['feature_tokens']
                ego_prompt = make_prompt('Trajectory', episode['ego_motion'], saved['evidence_used'],
                                         evidence_format='compact')
                ego_tokens = len(prompt_tokens(tokenizer, ego_prompt)) - 1 + episode['feature_tokens']
                empty_remote_tokens = input_tokens_for([])
                max_input = saved['receiver_spec']['context_limit'] - saved['receiver_spec']['generation_reserve']
                actual_selected, actual_attempts = admit_in_order(units, input_tokens_for, max_input)
                simulated_order = target_round_robin(units,
                    target_key=lambda unit: (unit['object']['source'], unit['object']['track_id']))
                simulated_selected, simulated_attempts = admit_in_order(simulated_order, input_tokens_for, max_input)
                saved_groups = saved['admission_report']['field_groups']
                if [_unit_id(unit) for unit in actual_selected] != [
                        _json(dict(anchor_ref=group['anchor_ref'], primary_refs=group['primary_refs']))
                        for group in saved_groups]:
                    raise AssertionError(task_name + ' diagnostic selection differs from saved admission')
                if input_tokens_for(actual_selected) != counted:
                    raise AssertionError(task_name + ' diagnostic final token total differs from saved input')

                records = _source_records(ledger, field_key)
                drop_reasons = {field_key(item['ref']): item['reason']
                                for item in saved['admission_report']['dropped']}
                actual_groups = {_json(dict(anchor_ref=group['anchor_ref'], primary_refs=group['primary_refs'])): group
                                 for group in saved_groups}
                selected_ids = {_unit_id(unit) for unit in actual_selected}
                sim_selected_ids = {_unit_id(unit) for unit in simulated_selected}
                sim_positions = { _unit_id(unit): position for unit, position in
                    zip(simulated_selected, remote_for(simulated_selected)[1]) }
                sim_attempt_by_id = {_unit_id(unit): attempt for unit, attempt in
                                     zip(simulated_order, simulated_attempts)}
                for receiver_order, (unit, attempt) in enumerate(zip(units, actual_attempts), 1):
                    uid = _unit_id(unit)
                    source_record = records[field_key(unit['primary_refs'][0])]
                    origins, parent_refs = _request_origins(ledger, source_record, field_key, stage_metadata)
                    group = actual_groups.get(uid)
                    reasons = sorted({drop_reasons.get(field_key(ref), 'unreported')
                                      for ref in unit['primary_refs']}) if uid not in selected_ids else ['admitted']
                    standalone = input_tokens_for([unit]) - ego_tokens
                    metadata = unit['object']['field_metadata'][unit['kind']]
                    unit_rows.append(dict(task=task_name, frame=episode['g'], arm=episode['branch_id'],
                        plan_index=plan_index, plan_stage=plan['stage'], after_request_id=plan['request_id'],
                        receiver_order=receiver_order, request_origins=_json(origins),
                        outer_request_ids=_json([origin['outer_request_id'] for origin in origins]),
                        request_stages=_json([origin['request_stage'] for origin in origins]),
                        first_visible_plan_stages=_json([origin['first_visible_plan_stage'] for origin in origins]),
                        tool_modes=_json([[origin['tool'], origin['mode']] for origin in origins]),
                        provider_ranks=_json([origin['provider_rank'] for origin in origins]),
                        provider_rankings=_json([origin['provider_ranking'] for origin in origins]),
                        provider=unit['object']['source'], track_handle=unit['object']['track_id'],
                        field_kind=unit['kind'], field_group=_field_group(unit),
                        context_scope=metadata.get('context_scope'), context_version=_json(metadata['context_version']),
                        acquisition_origin=source_record['origin'], anchor_distance_m='%.9f' % math.hypot(*unit['object']['box'][:2]),
                        standalone_increment_vs_ego=standalone, ego_only_input_tokens=ego_tokens,
                        empty_remote_input_tokens=empty_remote_tokens,
                        attempt_input_tokens=attempt['attempt_input_tokens'],
                        attempt_total_with_generation=attempt['attempt_input_tokens'] + saved['receiver_spec']['generation_reserve'],
                        cumulative_input_tokens=attempt['cumulative_input_tokens'],
                        cumulative_total_with_generation=attempt['cumulative_input_tokens'] + saved['receiver_spec']['generation_reserve'],
                        admitted=attempt['admitted'], dropped_reason='|'.join(reasons),
                        prompt_section=group['prompt_section'] if group else None,
                        object_index=group['object_index'] if group else None,
                        field_positions=_json(group['field_positions']) if group else None,
                        anchor_ref=_json(unit['anchor_ref']), primary_refs=_json(unit['primary_refs']),
                        context_parent_refs=_json(parent_refs)))
                    sim_attempt = sim_attempt_by_id[uid]
                    simulated_rows.append(dict(task=task_name, frame=episode['g'], arm=episode['branch_id'],
                        plan_index=plan_index, plan_stage=plan['stage'], simulated_order=sim_attempt['order'],
                        provider=unit['object']['source'], track_handle=unit['object']['track_id'],
                        field_kind=unit['kind'], field_group=_field_group(unit),
                        context_scope=metadata.get('context_scope'), context_version=_json(metadata['context_version']),
                        anchor_distance_m='%.9f' % math.hypot(*unit['object']['box'][:2]),
                        attempt_input_tokens=sim_attempt['attempt_input_tokens'],
                        attempt_total_with_generation=sim_attempt['attempt_input_tokens'] + saved['receiver_spec']['generation_reserve'],
                        cumulative_input_tokens=sim_attempt['cumulative_input_tokens'],
                        cumulative_total_with_generation=sim_attempt['cumulative_input_tokens'] + saved['receiver_spec']['generation_reserve'],
                        admitted=sim_attempt['admitted'], dropped_reason='admitted' if sim_attempt['admitted'] else 'context_budget',
                        prompt_section='remote' if uid in sim_selected_ids else None,
                        object_index=sim_positions.get(uid), anchor_ref=_json(unit['anchor_ref']),
                        primary_refs=_json(unit['primary_refs'])))
                actual_group_counts = _group_counts(actual_selected)
                simulated_group_counts = _group_counts(simulated_selected)
            for item in saved['admission_report']['dropped']:
                dropped_record = next(record for record in
                    ledger['local_fields'] + ledger['acquired_fields'] + ledger['derived_fields']
                    if field_key(record['ref']) == field_key(item['ref']))
                value = dropped_record['value']
                kind = item['ref']['field_kind']
                field_group = ('history' if kind == 'history' else
                    'receiver_subset_forecast' if value.get('context_scope') == 'receiver_acquired_subset' else
                    'provider_full_forecast' if value.get('context_scope') == 'provider_full_at_t' else kind)
                dropped_rows.append(dict(task=task_name, frame=episode['g'], arm=episode['branch_id'],
                    plan_index=plan_index, plan_stage=plan['stage'], provider=item['ref']['provider'],
                    track_handle=item['ref']['track_handle'], field_kind=kind, field_group=field_group,
                    acquisition_origin=dropped_record['origin'],
                    context_scope=value.get('context_scope'), reason=item['reason'], ref=_json(item['ref'])))

            actual_targets = {(unit['object']['source'], unit['object']['track_id']) for unit in actual_selected}
            simulated_targets = {(unit['object']['source'], unit['object']['track_id']) for unit in simulated_selected}
            available_counts = Counter(unit['object']['track_id'] for unit in units)
            actual_counts = Counter(unit['object']['track_id'] for unit in actual_selected)
            simulated_counts = Counter(unit['object']['track_id'] for unit in simulated_selected)
            stage_rows.append(dict(task=task_name, frame=episode['g'], arm=episode['branch_id'],
                plan_index=plan_index, plan_stage=plan['stage'], after_request_id=plan['request_id'],
                remote_units_total=len(units), available_unique_targets=len(available_counts),
                available_units_per_target=_json(dict(sorted(available_counts.items()))),
                remote_units_retained=len(actual_selected), actual_units_retained=len(actual_selected),
                actual_unique_targets=len(actual_targets),
                actual_units_per_target=_json(dict(sorted(actual_counts.items()))),
                actual_max_units_per_target=max(actual_counts.values(), default=0),
                actual_history=actual_group_counts['history'],
                actual_receiver_subset_forecast=actual_group_counts['receiver_subset_forecast'],
                actual_provider_full_forecast=actual_group_counts['provider_full_forecast'],
                counterfactual_units_retained=len(simulated_selected),
                counterfactual_unique_targets=len(simulated_targets),
                additional_unique_targets=len(simulated_targets) - len(actual_targets),
                counterfactual_units_per_target=_json(dict(sorted(simulated_counts.items()))),
                counterfactual_max_units_per_target=max(simulated_counts.values(), default=0),
                counterfactual_history=simulated_group_counts['history'],
                counterfactual_receiver_subset_forecast=simulated_group_counts['receiver_subset_forecast'],
                counterfactual_provider_full_forecast=simulated_group_counts['provider_full_forecast'],
                ego_only_input_tokens=ego_tokens, empty_remote_input_tokens=empty_remote_tokens,
                final_input_tokens=counted, generation_reserve=saved['receiver_spec']['generation_reserve'],
                context_limit=saved['receiver_spec']['context_limit']))
        reproducibility.append(dict(task=task_name, status='PASS', plan_checks=task_checks,
            provider_records_match_ledger=True, request_stage_event_binding=True,
            receipt_stage_metadata=stage_metadata))
        final = episode['plans'][-1]
        final_by_frame.setdefault(episode['g'], {})[episode['branch_id']] = final

    frame_comparisons = []
    for frame, plans in sorted(final_by_frame.items()):
        alternating, one_shot = plans['alternating'], plans['one_shot']
        comparison = dict(frame=frame,
            final_prompt_equal=alternating['prepared']['q9_prompt'] == one_shot['prepared']['q9_prompt'],
            final_remote_evidence_equal=(alternating['prepared']['remote_evidence_used'] ==
                                         one_shot['prepared']['remote_evidence_used']),
            final_input_tokens_alternating=alternating['prepared']['evidence_selection']['input_tokens'],
            final_input_tokens_one_shot=one_shot['prepared']['evidence_selection']['input_tokens'],
            final_waypoints_equal=alternating['output']['waypoints'] == one_shot['output']['waypoints'])
        if not all(comparison[key] for key in ('final_prompt_equal', 'final_remote_evidence_equal', 'final_waypoints_equal')):
            raise AssertionError('saved final arm convergence changed for frame %d' % frame)
        frame_comparisons.append(comparison)

    algorithm = {
        'name': 'stable_target_round_robin_v1',
        'scope': 'counterfactual admission only; no GoT call and no driving output',
        'input_units': 'the exact paid/receiver-derived remote_units from the saved stage; context-distinct units remain distinct',
        'target_order': 'first appearance in frozen receiver-v1 order: anchor distance, provider, track_handle, field kind, context identity, field identity',
        'within_target_order': 'unchanged frozen receiver-v1 order',
        'rounds': 'one unit per target in target order, then the second unit per target, continuing until exhausted',
        'admission': 'greedy; admit when actual original-tokenizer input tokens plus generation reserve fit context_limit; skip a non-fitting unit and continue',
        'budgets': dict(context_limit=4096, generation_reserve=256, peer_reserve=1536, feature_tokens=540),
        'forbidden_inputs': ['GT', 'offline labels', 'unbought peer information', 'alternating-only tau1'],
    }
    summary = dict(version='toolv2x_t9_admission_audit_v3', input_root=str(input_root),
        tokenizer=str(tokenizer_path), model_execution=False, frames=[5526, 7007],
        released_tokenizer_warning_limit=released_tokenizer_limit,
        audit_tokenizer_warning_limit=tokenizer.model_max_length,
        task_count=len(tasks), plan_stage_count=len(stage_rows), remote_unit_row_count=len(unit_rows),
        reproducibility='PASS', frame_comparisons=frame_comparisons,
        counterfactual_algorithm=algorithm)
    token_group_rows = []
    for key in sorted({(row['task'], row['frame'], row['arm'], row['plan_index'], row['plan_stage'], row['field_group'])
                       for row in unit_rows}):
        values = [row for row in unit_rows
                  if (row['task'], row['frame'], row['arm'], row['plan_index'], row['plan_stage'], row['field_group']) == key]
        costs = [int(row['standalone_increment_vs_ego']) for row in values]
        token_group_rows.append(dict(task=key[0], frame=key[1], arm=key[2], plan_index=key[3],
            plan_stage=key[4], field_group=key[5], unit_count=len(values),
            standalone_increment_min=min(costs), standalone_increment_max=max(costs),
            standalone_increments=_json(costs)))
    outputs = {
        'summary.json': summary,
        'reproducibility.json': reproducibility,
        'plan_stages.csv': stage_rows,
        'token_group_summary.csv': token_group_rows,
        'provider_rankings.csv': provider_rows,
        'remote_units_actual.csv': unit_rows,
        'remote_units_counterfactual.csv': simulated_rows,
        'dropped_fields.csv': dropped_rows,
    }
    for name, value in outputs.items():
        path = output_dir / name
        _write_csv(path, value) if name.endswith('.csv') else _write_json(path, value)
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for name in outputs:
            (evidence_dir / name).write_bytes((output_dir / name).read_bytes())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path,
        default=Path('/root/autodl-tmp/ToolV2X/outputs/t9_real_smoke_2026_09_13_v1'))
    parser.add_argument('--tokenizer', type=Path,
        default=Path('/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b'))
    parser.add_argument('--output-dir', type=Path,
        default=Path('/root/autodl-tmp/ToolV2X/outputs/t9_admission_audit_2026_09_13_v3'))
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    summary = audit(args.input_root.resolve(), args.tokenizer.resolve(),
                    args.output_dir.resolve(), args.evidence_dir.resolve() if args.evidence_dir else None)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
