"""Actual P/F decisions, responses, evidence updates and termination at time t."""
from time import perf_counter
import numpy as np

from probe.kinematic_tools import encode
from tools.vehicle import VERSION, check_roi, decode_response, make_evidence

POLICIES = ('Ego', 'P', 'F', 'PF', 'rule')


def choose_action(state, policy):
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    queried = len(state['responses'])
    if policy != 'rule':
        sequence = () if policy == 'Ego' else tuple(policy)
        return dict(tool=sequence[queried] if queried < len(sequence) else 'STOP', roi=None,
                    reason='fixed_sequence' if queried < len(sequence) else 'fixed_sequence_completed')
    speed = state['motion']['speed_mps']
    if queried == 0:
        if speed is not None and speed <= .5:
            return dict(tool='STOP', roi=None, reason='stationary_rule')
        roi = [-5., -10., max(20., 3. * (speed if speed is not None else 0.)), 10.]
        return dict(tool='P', roi=roi, reason='inspect_forward_region')
    packet = state['responses'][-1]
    if packet['tool'] != 'P':
        return dict(tool='STOP', roi=None, reason='prediction_received')
    if not packet['objects']:
        return dict(tool='STOP', roi=None, reason='no_returned_targets')
    related = {r['peer_id'] for r in state['evidence']['relations']}
    if any(o['track_id'] not in related for o in packet['objects']):
        return dict(tool='F', roi=packet['roi'], reason='peer_targets_without_local_geometry_candidate')
    return dict(tool='STOP', roi=None, reason='all_returned_targets_have_geometry_candidates')


def run_episode(local_window, local_prediction, motion, service, predictor, policy,
                predict_p=True, max_calls=2, on_progress=None):
    if policy not in POLICIES or isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 0 <= max_calls <= 2:
        raise ValueError('invalid policy/call budget')
    scene, g = local_window['scene'], int(local_window['g'])
    evidence = make_evidence(local_window, local_prediction, [], predictor, predict_p=predict_p)
    state = dict(motion=dict(motion), evidence=evidence, responses=[])
    steps, p_cache = [], {}
    cost = dict(calls=0, request_bytes=0, response_bytes=0, service_seconds=0., receiver_seconds=0.,
                peer_model_seconds=0., peer_model_targets=0, receiver_model_calls=0, receiver_model_targets=0,
                complete=True)
    episode = dict(version='toolv2x_episode_v1', status='tools_running', scene=scene, g=g, policy=policy,
                   p_processing='local_mtr' if predict_p else 'observations', max_calls=max_calls,
                   steps=steps, stop_reason=None, cost=cost, evidence=evidence)

    def persist():
        episode['evidence'] = state['evidence']
        if on_progress is not None:
            on_progress(episode)

    def predict_received(w):
        # At one t, this source's track IDs determine the already received P window.
        key = tuple(map(int, w['track_ids']))
        if key not in p_cache:
            cost['receiver_model_calls'] += 1
            p_cache[key] = predictor(w)
            cost['receiver_model_targets'] += int(np.asarray(p_cache[key]['model_used']).sum())
        return p_cache[key]

    while True:
        decision = (dict(tool='STOP', roi=None, reason='call_budget_exhausted') if cost['calls'] >= max_calls
                    else choose_action(state, policy))
        step = dict(decision=decision, queries_before=cost['calls'], objects_before=len(state['evidence']['objects']))
        steps.append(step)
        if decision['tool'] == 'STOP':
            episode.update(status='tools_completed', stop_reason=decision['reason'])
            persist()
            break
        check_roi(decision['roi'])
        request = dict(version=VERSION, tool=decision['tool'], provider=service.provider, scene=scene, g=g,
                       roi=None if decision['roi'] is None else list(map(float, decision['roi'])))
        cost['calls'] += 1
        cost['request_bytes'] += len(encode(request))
        step.update(request=request, status='request_started')
        persist()
        stage = 'service_query'
        begin = perf_counter()
        try:
            result = service.query(decision['tool'], decision['roi'])
            stage = 'response_validation'
            charges = result['cost']
            cost['response_bytes'] += len(result['wire'])
            cost['service_seconds'] += charges['service_seconds']
            cost['peer_model_seconds'] += charges['model_seconds']
            cost['peer_model_targets'] += charges['model_targets_computed']
            step.update(cost=dict(charges), status='response_received')
            packet = decode_response(result['wire'], scene, g)
            step['response'] = packet
            if result['request'] != request or any(packet[key] != value for key, value in request.items()):
                raise ValueError('response does not match the executed query')
            if charges['request_bytes'] != len(encode(request)) or charges['response_bytes'] != len(result['wire']):
                raise ValueError('wire byte accounting mismatch')
            state['responses'].append(packet)
            persist()  # The actual response and charges survive a failed receiver update.
            stage = 'receiver_update'
            begin = perf_counter()
            try:
                state['evidence'] = make_evidence(local_window, local_prediction, state['responses'],
                                                  predict_received, predict_p=predict_p)
            finally:
                cost['receiver_seconds'] += perf_counter() - begin
            step.update(objects_after=len(state['evidence']['objects']), status='completed')
            persist()
        except Exception as exc:
            if stage == 'service_query':
                cost['service_seconds'] += perf_counter() - begin
            cost['complete'] = False
            error = dict(stage=stage, error_type=type(exc).__name__, message=str(exc))
            step.update(status='error', error=error)
            episode.update(status='tool_error', stop_reason='tool_error', error=error)
            persist()
            break
    return episode
