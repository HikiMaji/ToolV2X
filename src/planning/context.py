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
