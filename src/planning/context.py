"""One causal source-separated input constructor for training and generation."""
import copy
import numpy as np

from planning.inputs import make_prompt, validate_evidence

CONTEXT = 4096
GENERATION = 256
PEER_RESERVE = 1536


def input_size(tokenizer, motion, local, feature_tokens, remote=None):
    from planning.v2vgot import prompt_tokens
    return len(prompt_tokens(tokenizer, make_prompt('Trajectory', motion, local,
               evidence_format='compact', remote_evidence=remote))) - 1 + feature_tokens


def _select(evidence, fits):
    from planning.v2vgot import rounded
    validate_evidence(evidence)
    view = rounded(copy.deepcopy(evidence))
    objects = sorted(view['objects'], key=lambda o: (np.linalg.norm(o['box'][:2]), o['source'], o['track_id']))
    relations = view.pop('relations', [])
    view['objects'] = []
    if not fits(view):
        raise ValueError('task/header exceeds reserved capacity')
    dropped = []
    for obj in objects:
        trial = dict(view, objects=view['objects'] + [obj])
        if fits(trial):
            view = trial
        else:
            dropped.append(dict(source=obj['source'], track_id=obj['track_id']))
    return view, relations, dict(objects_total=len(objects), objects_retained=len(view['objects']),
        dropped_objects=dropped, numeric_decimal_places=2, evidence_format='compact',
        selection='ascending current anchor distance within each source block')


def fit_local(tokenizer, motion, evidence, feature_tokens, context_limit=CONTEXT, peer_reserve=PEER_RESERVE):
    if (evidence.get('queries') or evidence.get('relations') or
            any(o.get('source') != 'ego' for o in evidence['objects'])):
        raise ValueError('local selection cannot read peer evidence')
    if not 0 <= peer_reserve < context_limit - GENERATION:
        raise ValueError('invalid source block budget')
    view, _, report = _select(evidence, lambda v: input_size(tokenizer, motion, v, feature_tokens)
                             + GENERATION + peer_reserve <= context_limit)
    report.update(input_tokens=input_size(tokenizer, motion, view, feature_tokens),
                  peer_reserved_tokens=peer_reserve, context_limit=context_limit, generation_reserve=GENERATION)
    return view, report


def fit_remote(tokenizer, motion, local, evidence, feature_tokens, context_limit=CONTEXT):
    if (not evidence.get('queries') or any(not o.get('source', '').endswith((':P', ':P_local', ':F'))
                                         for o in evidence['objects'])):
        raise ValueError('remote selector requires only queried P/F objects')
    fits = lambda v: input_size(tokenizer, motion, local, feature_tokens, v) + GENERATION <= context_limit
    view, relations, report = _select(evidence, fits)
    ego_ids = {o['track_id'] for o in local['objects']}
    peer_ids = {o['track_id'] for o in view['objects']}
    for relation in relations:
        if relation['ego_id'] in ego_ids and relation['peer_id'] in peer_ids:
            trial = dict(view, relations=view.get('relations', []) + [relation])
            if fits(trial):
                view = trial
    report.update(input_tokens=input_size(tokenizer, motion, local, feature_tokens, view),
                  local_objects_unchanged=True, context_limit=context_limit, generation_reserve=GENERATION)
    return view, report


def build_plan_input(tokenizer, motion, evidence, feature_tokens, context_limit=CONTEXT, peer_reserve=PEER_RESERVE):
    validate_evidence(evidence)
    local_full = dict(evidence, objects=[o for o in evidence['objects'] if o['source'] == 'ego'],
                      queries=[], relations=[])
    local, local_selection = fit_local(tokenizer, motion, local_full, feature_tokens, context_limit, peer_reserve)
    remote_full = dict(evidence, objects=[o for o in evidence['objects'] if o['source'] != 'ego'])
    if remote_full['objects'] and not remote_full.get('queries'):
        raise ValueError('remote objects arrived without a query')
    remote, remote_selection = None, None
    if evidence.get('queries'):
        remote, remote_selection = fit_remote(tokenizer, motion, local, remote_full, feature_tokens, context_limit)
    prompt = make_prompt('Trajectory', motion, local, evidence_format='compact', remote_evidence=remote)
    return dict(input_layout='source_blocks_v1', decoding='direct', ego_motion=dict(motion),
        evidence_used=local, remote_evidence_used=remote,
        evidence_selection=dict(evidence_format='compact', local=local_selection, remote=remote_selection,
            context_limit=context_limit, peer_reserved_tokens=peer_reserve, generation_reserve=GENERATION,
            feature_tokens=feature_tokens, input_tokens=input_size(tokenizer, motion, local, feature_tokens, remote)),
        q8_executed=False, q8_raw=None, q8_prompt=None, q8_cost=None, q9_prompt=prompt,
        q9_executed=False, language_model_executed=False, status='prepared')


def target_round_robin(units, target_key):
    """Stable target cycling; retain every context-distinct unit exactly once."""
    grouped = {}
    for unit in units:
        grouped.setdefault(target_key(unit), []).append(unit)
    result = []
    for depth in range(max((len(values) for values in grouped.values()), default=0)):
        result.extend(values[depth] for values in grouped.values() if depth < len(values))
    return result


def build_task_plan_input(tokenizer, motion, ledger, feature_tokens, limits=None, *, token_counter=None, extra_prompt_reserve=0):
    """Shared deterministic v2 receiver; never runs a driver or a predictor.

    The optional counter is for resource-independent contract tests, explicitly
    marked in the result. Actual input sizes use the original GoT prompt tokenizer.
    """
    from planning.evidence import remote_units
    from planning.inputs import pack_evidence
    from tools.task_spec import field_key
    from time import perf_counter
    begin = perf_counter()
    config = dict(version='toolv2x_receiver_v1', context_limit=CONTEXT, generation_reserve=GENERATION,
                  peer_reserve=PEER_RESERVE, numeric_decimal_places=2)
    if limits is not None:
        if set(limits) - set(config):
            raise ValueError('unknown receiver limits')
        config.update(limits)
    if (config['version'] not in ('toolv2x_receiver_v1', 'toolv2x_receiver_v2') or
            any(type(config[k]) is not int or config[k] < 0 for k in config if k != 'version') or
            config['generation_reserve'] <= 0 or config['numeric_decimal_places'] != 2 or
            config['peer_reserve'] >= config['context_limit'] - config['generation_reserve'] or
            type(feature_tokens) is not int or feature_tokens < 0):
        raise ValueError('invalid versioned receiver configuration')
    if token_counter is None:
        from planning.v2vgot import prompt_tokens
        token_counter = lambda prompt: len(prompt_tokens(tokenizer, prompt)) - 1
        counting = 'original_got_prompt_tokens'
    else:
        counting = 'injected_contract_counter'

    if type(extra_prompt_reserve) is not int or extra_prompt_reserve < 0:
        raise ValueError('invalid explicit extra prompt reserve')

    def rounded(value):
        # Same two-decimal numeric projection as the existing v1 receiver.
        if isinstance(value, float):
            return round(value, config['numeric_decimal_places'])
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value

    def render(local, remote=None):
        return make_prompt('Trajectory', motion, local, evidence_format='compact', remote_evidence=remote)

    def size(local, remote=None):
        n = token_counter(render(local, remote))
        if type(n) is not int or n < 0:
            raise ValueError('invalid prompt token count')
        return n + feature_tokens

    units = remote_units(ledger)  # Validates paid-field provenance before projecting.
    if config['version'] == 'toolv2x_receiver_v2':
        units = target_round_robin(units, lambda u: (u['object']['source'], u['object']['track_id']))
    local = dict(as_of_g=ledger['g'], coordinate_frame='ego_at_t', objects=[], queries=[], relations=[])
    grouped = {}
    for record in ledger['local_fields']:
        ref = record['ref']
        obj = grouped.setdefault(ref['track_handle'], dict(source='ego', track_id=ref['track_handle']))
        obj.update({k: v for k, v in record['value'].items() if k != 'context_scope'})
    local_objects = sorted(rounded(list(grouped.values())), key=lambda o: (float(np.hypot(*o['box'][:2])), o['track_id']))
    local_limit = config['context_limit'] - config['generation_reserve'] - config['peer_reserve'] - extra_prompt_reserve
    if size(local) > local_limit:
        raise ValueError('local header exceeds configured receiver budget')
    for obj in local_objects:
        trial = dict(local, objects=local['objects'] + [obj])
        if size(trial) <= local_limit:
            local = trial

    observations = []
    for provider in sorted({r['request']['provider'] for r in ledger['receipts']}):
        complete = any(r['request']['provider'] == provider and 'context_certificate' in r['response']
                       for r in ledger['receipts'])
        observations.append(dict(provider=provider, coverage='not_established', history_complete=complete))

    def remote_view(selected):
        if not observations:
            return None
        objects = []
        positions = []
        for unit in selected:
            obj = rounded(copy.deepcopy(unit['object']))
            kind = unit['kind']
            context = obj['field_metadata'][kind]['context_version']
            compatible = next((i for i, other in enumerate(objects)
                if (other['source'], other['track_id']) == (obj['source'], obj['track_id'])
                and kind not in other['field_metadata']
                and all(m['context_version'] == context for m in other['field_metadata'].values())), None)
            if compatible is None:
                positions.append(len(objects))
                objects.append(obj)
            else:
                positions.append(compatible)
                metadata = dict(objects[compatible]['field_metadata'], **obj['field_metadata'])
                objects[compatible].update(obj)
                objects[compatible]['field_metadata'] = metadata
        view = dict(evidence_version='toolv2x_driver_evidence_v2', as_of_g=ledger['g'],
                    coordinate_frame='ego_at_t', observations=copy.deepcopy(observations), objects=objects)
        return view, positions

    selected = []
    empty = remote_view([])
    if size(local, empty[0] if empty else None) + config['generation_reserve'] + extra_prompt_reserve > config['context_limit']:
        raise ValueError('paid remote observation header exceeds receiver budget')
    for unit in units:
        trial, _ = remote_view(selected + [unit])
        if size(local, trial) + config['generation_reserve'] + extra_prompt_reserve <= config['context_limit']:
            selected.append(unit)
    built = remote_view(selected)
    remote, positions = built if built else (None, [])
    admitted = {field_key(r['ref']): r['ref'] for r in ledger['local_fields']
                if r['ref']['track_handle'] in {o['track_id'] for o in local['objects']}}
    groups = []
    packed = pack_evidence(remote) if remote is not None else None
    for unit, row in zip(selected, positions):
        refs = [unit['anchor_ref']] + unit['primary_refs']
        admitted.update({field_key(ref): ref for ref in refs})
        names = sorted(unit['object'])
        locations = {name: dict(area='object_shared') if name in packed['object_shared'] else
                     dict(area='object_rows', row=row, column=packed['object_columns'].index(name)) for name in names}
        groups.append(dict(primary_refs=copy.deepcopy(unit['primary_refs']), anchor_ref=copy.deepcopy(unit['anchor_ref']),
            kind=unit['kind'], object_index=row, prompt_section='remote', field_positions=locations))
    records = ledger['local_fields'] + ledger['acquired_fields'] + ledger['derived_fields']
    dropped = []
    for record in records:
        ref = record['ref']
        if field_key(ref) not in admitted:
            stale = record['origin'] == 'receiver_derived' and ledger['active_contexts'].get(ref['provider']) != ref['context_version']
            dropped.append(dict(ref=copy.deepcopy(ref), reason='superseded_context' if stale else 'context_budget'))
    input_tokens = size(local, remote)
    local_packed = pack_evidence(local)
    local_groups = []
    for row, obj in enumerate(local['objects']):
        for record in ledger['local_fields']:
            ref = record['ref']
            if ref['track_handle'] == obj['track_id']:
                names = [k for k in record['value'] if k != 'context_scope']
                locations = {k: dict(area='object_shared') if k in local_packed['object_shared'] else
                             dict(area='object_rows', row=row, column=local_packed['object_columns'].index(k)) for k in names}
                local_groups.append(dict(ref=copy.deepcopy(ref), object_index=row,
                                         prompt_section='local', field_positions=locations))
    return dict(input_layout='source_blocks_v2', decoding='direct', ego_motion=copy.deepcopy(motion),
        evidence_used=local, remote_evidence_used=remote, q9_prompt=render(local, remote),
        receiver_spec=config, admission_report=dict(
            acquired_field_refs=[copy.deepcopy(r['ref']) for r in ledger['acquired_fields']],
            derived_field_refs=[copy.deepcopy(r['ref']) for r in ledger['derived_fields']],
            admitted_field_refs=[copy.deepcopy(admitted[k]) for k in sorted(admitted)],
            dropped_field_refs=[d['ref'] for d in dropped], dropped=dropped, field_groups=groups,
            local_field_groups=local_groups,
            observation_receipt_ids=[r['receipt_id'] for r in ledger['receipts']]),
        evidence_selection=dict(evidence_format='compact', context_limit=config['context_limit'],
            peer_reserved_tokens=config['peer_reserve'], generation_reserve=config['generation_reserve'],
            feature_tokens=feature_tokens, input_tokens=input_tokens, token_counting=counting,
            local_objects_unchanged=True, remote_units_total=len(units), remote_units_retained=len(selected)),
        input_build_seconds=perf_counter() - begin,
        q8_executed=False, q8_raw=None, q8_prompt=None, q8_cost=None, q9_executed=False,
        language_model_executed=False, status='prepared')
