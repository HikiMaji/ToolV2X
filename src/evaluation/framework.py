"""Offline trajectory, failure and total-query accounting for complete episodes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.planning import evaluate_plan, trajectory_metrics, TIMES
from common.audit_protocol import recording


def validate_generation_run(root):
    from planning.run_framework import read_jsonl
    if not (root/'completion.json').is_file():
        raise ValueError('incomplete generation run: completion record missing')
    completion = json.loads((root/'completion.json').read_text())
    config = json.loads((root/'config.json').read_text())
    if completion.get('status') != 'completed':
        raise ValueError('incomplete generation run')
    expected = read_jsonl(root/'selected_tasks.jsonl')
    rows = read_jsonl(root/'generations.jsonl')
    def key(r):
        return r['sample_id'], r['policy']
    index = {key(r): r for r in expected}
    if (len(index) != len(expected) or len(rows) != len(expected) or len({key(r) for r in rows}) != len(rows) or
            {key(r) for r in rows} != set(index) or config['tasks'] != len(rows) or completion['attempts'] != len(rows)):
        raise ValueError('generation task coverage mismatch')
    for row in rows:
        if any(row[k] != index[key(row)][k] for k in ('path', 'scene', 'g', 'role')):
            raise ValueError('generation task identity mismatch')
    return rows


def summarize_episodes(rows):
    policies = {}
    for policy in sorted({r['policy'] for r in rows}):
        selected = [r for r in rows if r['policy'] == policy]
        valid = [r for r in selected if r['valid']]
        policies[policy] = dict(attempts=len(selected), failures=len(selected)-len(valid),
            failure_rate=1.-len(valid)/len(selected),
            cost_incomplete_attempts=sum(not r.get('cost_complete', True) for r in selected),
            **{'mean_'+key: float(np.mean([r[key] for r in valid])) if valid else None for key in ('ADE3','FDE3')},
            **{'mean_'+key: float(np.mean([r[key] for r in selected])) for key in ('calls','request_bytes','response_bytes')},
            stop_reasons=dict(Counter(r['stop_reason'] for r in selected)))
        for key in ('local_model_seconds', 'service_seconds', 'receiver_seconds', 'driver_seconds', 'total_compute_seconds'):
            values = [r[key] for r in selected if r.get(key) is not None]
            policies[policy]['mean_'+key] = float(np.mean(values)) if values else None
    ego = {r['sample_id']: r for r in rows if r['policy'] == 'Ego'}
    paired = {}
    for policy in policies:
        if policy == 'Ego':
            continue
        matches = [(ego[r['sample_id']], r) for r in rows if r['policy'] == policy and r['sample_id'] in ego]
        valid = [(a,b) for a,b in matches if a['valid'] and b['valid']]
        differences = [b['ADE3']-a['ADE3'] for a,b in valid]
        paired[policy] = dict(matched_attempts=len(matches), both_valid=len(valid),
            ego_valid_peer_failed=sum(a['valid'] and not b['valid'] for a,b in matches),
            ego_failed_peer_valid=sum(not a['valid'] and b['valid'] for a,b in matches),
            both_failed=sum(not a['valid'] and not b['valid'] for a,b in matches),
            mean_ADE3_difference=float(np.mean(differences)) if valid else None,
            mean_FDE3_difference=float(np.mean([b['FDE3']-a['FDE3'] for a,b in valid])) if valid else None,
            improved_fraction=float(np.mean(np.asarray(differences)<-1e-6)) if valid else None,
            harmed_fraction=float(np.mean(np.asarray(differences)>1e-6)) if valid else None)
    return dict(policies=policies, paired_vs_ego=paired)


def evaluate(generations, out, data):
    from planning.run_framework import read_jsonl
    from probe.kinematic_tools import encode
    labels = {r['sample_id']: r for role in ('train','validation')
              for r in read_jsonl(data/'offline_labels'/(role+'.jsonl'))}
    rows, seen, references = [], set(), {}
    for record in validate_generation_run(generations):
        saved = json.loads(Path(record['generation_path']).read_text())
        task = json.loads(Path(saved['task_path']).read_text())
        if (saved['task_path'] != record['path'] or
                any(record[k] != task[k] for k in ('sample_id', 'policy', 'role', 'scene', 'g')) or
                (task['sample_id'],task['policy']) in seen):
            raise ValueError('mismatched or duplicate generated task')
        seen.add((task['sample_id'],task['policy']))
        label = labels[task['sample_id']]
        if label['role'] != task['role'] or label['g'] != task['g']:
            raise ValueError('evaluation role/time mismatch')
        episode = task.get('episode', dict(steps=[], stop_reason='preparation_error',
            cost=dict(calls=0, request_bytes=0, response_bytes=0, complete=False)))
        for step in episode['steps']:
            if 'response' in step and (len(encode(step['response'])) != step['cost']['response_bytes'] or
                    len(encode(step['request'])) != step['cost']['request_bytes']):
                raise ValueError('saved request/response charge mismatch')
        quality = evaluate_plan(saved['plan'], label, task['ego_motion'])
        timing = dict(local_model_seconds=task['local_model_seconds'],
            service_seconds=episode['cost'].get('service_seconds', 0.),
            receiver_seconds=episode['cost'].get('receiver_seconds', 0.),
            driver_seconds=saved.get('driver_attempt_seconds'))
        timing['total_compute_seconds'] = sum(timing.values()) if all(v is not None for v in timing.values()) else None
        row = dict(sample_id=task['sample_id'], scene=task['scene'], role=task['role'], g=task['g'], policy=task['policy'],
            valid=quality['q9_parse_success'], **{k:quality[k] for k in ('ADE3','FDE3')}, quality=quality,
            **{k:episode['cost'][k] for k in ('calls','request_bytes','response_bytes')},
            stop_reason=episode['stop_reason'], cost=episode['cost'], generation_path=record['generation_path'],
            cost_complete=episode['cost'].get('complete', True) and timing['total_compute_seconds'] is not None, **timing)
        rows.append(row)
        speed = task['ego_motion']['speed_mps']
        if task['sample_id'] not in references and speed is not None:
            points = [[speed*t, 0.] for t in TIMES]
            references[task['sample_id']] = dict(sample_id=task['sample_id'], scene=task['scene'],
                **trajectory_metrics(points, label, speed), baseline='current longitudinal speed, zero yaw-rate extrapolation')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='completed', attempts=len(rows), **summarize_episodes(rows),
        by_recording={group:summarize_episodes([r for r in rows if recording(r['scene'])==group])
                      for group in sorted({recording(r['scene']) for r in rows})},
        motion_reference=dict(count=len(references), ADE3=float(np.mean([r['ADE3'] for r in references.values()])) if references else None),
        timing_scope='sum of measured local MTR, service, receiver and driver attempt durations; excludes model loading and network transport; shared local computation charged once per counterfactual task',
        scope='development open-loop imitation and real serialized query costs; not closed-loop or learned-policy benefit')
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'motion_reference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in references.values()))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['policies'],indent=2),flush=True)

# v2 has its own reader and summary. The historical v1 functions above keep their semantics.
METHOD_COST_FIELDS = ('driver_calls', 'calls', 'rpc_rounds', 'control_seconds', 'local_model_seconds', 'input_build_seconds',
    'driver_seconds', 'generation_seconds', 'service_seconds', 'peer_model_seconds',
    'receiver_seconds', 'receiver_model_seconds', 'request_bytes', 'response_bytes',
    'input_tokens', 'output_tokens', 'feature_tokens', 'total_compute_seconds')
METHOD_TIMING_SCOPE = ('sum of measured local MTR, input construction, driver attempts, service attempts and receiver updates; '
    'plus explicitly recorded T6 control decisions/candidate construction; generation and peer/receiver MTR '
    'are nested diagnostics, not added again; excludes model loading, unrecorded executor bookkeeping, '
    'persistence I/O and network transport; old runs have no measured policy stage; shared local MTR charged once per task')


def _number(value):
    return type(value) in (int, float) and np.isfinite(value) and value >= 0


def _method_cost(task):
    """Read non-overlapping measured stages; keep known prefixes when totals are unknown."""
    from probe.kinematic_tools import encode
    ep = task.get('episode') or {}
    events, plans = ep.get('events', []), ep.get('plans', [])
    charges = ep.get('cost_events', [])
    costs, known, issues = {}, {}, []

    def put(key, values, expected=None):
        values = list(values)
        known[key] = sum(v for v in values if _number(v))
        costs[key] = known[key] if (all(_number(v) for v in values) and
                                    (expected is None or len(values) == expected)) else None

    put('local_model_seconds', [task.get('inputs', {}).get('local_prediction_seconds')])
    if not ep:
        for key in METHOD_COST_FIELDS:
            if key not in costs:
                costs[key], known[key] = None, 0
        return dict(costs, cost_complete=False, known_cost=known, cost_issues=issues,
                    policy_seconds=None, end_to_end_seconds=None)
    groups = {kind: [e for e in charges if e['kind'] == kind] for kind in ('driver','service','receiver','input_build')}
    expected = dict(driver=sum(e['kind']=='driver_started' for e in events),
                    service=len(ep.get('requests', [])), receiver=len(ep.get('responses', [])),
                    input_build=sum(e['kind']=='input_started' for e in events))
    put('driver_calls', [expected['driver']])
    put('rpc_rounds', [expected['service']])
    primitive_counts=[]
    for request in ep.get('requests', []):
        if request.get('version') in ('toolv2x_bundle_v1','toolv2x_bundle_v2'):
            event=next((e for e in groups['service'] if e.get('request_id')==request['request_id']),{})
            primitive_counts.append((event.get('service_cost') or {}).get('capability_calls'))
        else:
            primitive_counts.append(1)
    put('calls', primitive_counts)
    control_events=[e for e in charges if e['kind']=='control']
    control_stages={e['stage'] for e in control_events}
    expected_control={e['stage'] for e in events if e['kind']=='decision'} if ep.get('control_spec') else set()
    if len(control_stages)!=len(control_events):
        issues.append('duplicate control cost stage')
    put('control_seconds', [e.get('seconds') if e.get('timing_version')=='toolv2x_control_compute_v2' else None
                            for e in control_events] + [None]*len(expected_control-control_stages))
    if expected['driver'] != len(plans):
        issues.append('driver attempts and plan records disagree')
    for kind, field, value_key in [('driver','driver_seconds','attempt_seconds'),
                                  ('service','service_seconds','attempt_seconds'),
                                  ('receiver','receiver_seconds','seconds'),
                                  ('input_build','input_build_seconds','seconds')]:
        entries = groups[kind]
        if len({e.get('plan_id') if kind=='driver' else e.get('stage') for e in entries}) != len(entries):
            issues.append('duplicate '+kind+' cost stage')
        put(field, [e.get(value_key) for e in entries], expected[kind])
    for key in ('seconds','input_tokens','output_tokens','feature_tokens'):
        put('generation_seconds' if key=='seconds' else key,
            [(e.get('generation_cost') or {}).get(key) for e in groups['driver']], expected['driver'])
    for e in groups['driver']:
        p = next((p for p in plans if p.get('plan_id')==e.get('plan_id')), None)
        if p is None or p.get('driver_attempt_seconds') != e.get('attempt_seconds'):
            issues.append('driver duration differs from its plan record')
        elif (p.get('output') or {}).get('q9_cost') != e.get('generation_cost'):
            issues.append('generation cost differs from its actual output record')
    put('peer_model_seconds', [(e.get('service_cost') or {}).get('model_seconds') for e in groups['service']], expected['service'])
    ledger = (ep.get('ledger_snapshots') or [{}])[-1]
    receiver_models = ledger.get('receiver_events', [])
    missing_receiver = max(0, expected['receiver'] - len(groups['receiver']))
    put('receiver_model_seconds', [e.get('model_seconds') for e in receiver_models] + [None]*missing_receiver)
    request_values, response_values = [], []
    responses = ep.get('responses', [])
    if len(responses) > expected['service']:
        issues.append('more responses than sent requests')
    for i, req in enumerate(ep.get('requests', [])):
        event = next((e for e in groups['service'] if e.get('request_id')==req['request_id']), {})
        claim = event.get('service_cost') or {}
        if i < len(responses):
            reply = responses[i]
            if reply.get('request') != req:
                issues.append('response request identity mismatch')
            try:
                wire = bytes.fromhex(reply['wire_hex'])
            except (KeyError, TypeError, ValueError):
                wire = None
            request_values.append(len(encode(req)))
            response_values.append(len(wire) if wire is not None else None)
            if (claim and (claim.get('request_bytes') != len(encode(req)) or
                          (wire is not None and claim.get('response_bytes') != len(wire)))):
                issues.append('reported service bytes differ from actual wire')
        else:
            # No reply is not proof of free service. A provider's failed-attempt record can prove its spent bytes.
            record = event.get('provider_record')
            if record is not None and record.get('request') == req:
                request_values.append((record.get('cost') or {}).get('request_bytes'))
                response_values.append((record.get('cost') or {}).get('response_bytes'))
            else:
                request_values.append(None)
                response_values.append(None)
    put('request_bytes', request_values)
    put('response_bytes', response_values)
    timing = ('local_model_seconds','input_build_seconds','driver_seconds','service_seconds','receiver_seconds','control_seconds')
    put('total_compute_seconds', [costs[k] for k in timing])
    known['total_compute_seconds'] = sum(known[k] for k in timing)
    complete = (ep.get('status') != 'running' and ep.get('cost', {}).get('complete') is True and not issues and
                all(v is not None for v in costs.values()) and all(e.get('complete') is True for e in charges))
    return dict(costs, cost_complete=complete, known_cost=known, cost_issues=issues,
                reported_control_seconds=sum(e['seconds'] for e in control_events if _number(e.get('seconds'))),
                policy_seconds=None, end_to_end_seconds=None)


def method_plan_row(plan, label, motion, execution_spec):
    """Raw answer quality and protocol admissibility are distinct from episode success."""
    from planning.inputs import parse_q9
    from tools.task_spec import validate_plan
    output = plan.get('output') or {}
    result = dict(plan_id=plan.get('plan_id'), stage=plan.get('stage'), saved_status=plan.get('status'),
        raw_answer=output.get('q9_raw'), parse_valid=False, admissibility=None, record_consistent=True,
        raw_ADE3=None, raw_FDE3=None, valid_label_points=sum(label['valid']), errors=[])
    if output.get('q9_executed') is not True:
        return result
    try:
        points = parse_q9(output.get('q9_raw') or '')
    except (ValueError, TypeError) as exc:
        result['errors'].append(str(exc))
        return result
    result['parse_valid'] = True
    # Reuse the old raw-generation consistency and direct/Q8 contract, with unchanged numerical metrics.
    try:
        evaluate_plan(output, label, motion)
    except (ValueError, TypeError, KeyError) as exc:
        result.update(record_consistent=False)
        result['errors'].append(str(exc))
    quality = trajectory_metrics(points, label, motion.get('speed_mps'))
    result.update(raw_ADE3=quality['ADE3'], raw_FDE3=quality['FDE3'], quality=quality)
    try:
        validate_plan(points, execution_spec)
    except (ValueError, TypeError, KeyError) as exc:
        result['admissibility'] = False
        result['errors'].append(str(exc))
    else:
        result['admissibility'] = True
    return result


def _method_label(label, identity):
    unavailable = dict(valid=[False]*6, times_seconds=list(TIMES), waypoints=[None]*6)
    if label is None:
        return unavailable, 'missing'
    if label.get('_label_error'):
        return unavailable, label['_label_error']
    if (any(label.get(k) != identity[k] for k in ('sample_id','role','g')) or
            ('scene' in label and label['scene'] != identity['scene'])):
        return unavailable, 'identity_mismatch'
    try:
        trajectory_metrics([[0.,0.]]*6, label, None)
    except (ValueError, TypeError, KeyError, IndexError):
        return unavailable, 'invalid'
    return label, 'complete' if all(label['valid']) else 'partial' if any(label['valid']) else 'no_valid_points'


def evaluate_method_task(task, label, *, policy_id=None, branch_id=None):
    """Evaluate one persisted task; a prior valid plan never fills a failed terminal answer."""
    identity = task['row']
    ep = task.get('episode') or {}
    policy_id = policy_id if policy_id is not None else ep.get('policy_id')
    branch_id = branch_id if branch_id is not None else ep.get('branch_id')
    usable_label, label_status = _method_label(label, identity)
    cost = _method_cost(task)
    result = dict(version='toolv2x_method_evaluation_v1', **{k:identity[k] for k in ('sample_id','scene','role','g')},
        policy_id=policy_id, branch_id=branch_id, recording=recording(identity['scene']),
        saved_task_status=task.get('status'), episode_status=ep.get('status'),
        artifact_status='recorded_failure', task_success=False, parse_valid=False, admissibility=None,
        final_plan_id=ep.get('final_plan_id'), raw_plan_id=None, ADE3=None, FDE3=None, raw_ADE3=None, raw_FDE3=None,
        label_status=label_status, valid_label_points=sum(usable_label['valid']), plans=[],
        stop_reason=ep.get('stop_reason'), error=task.get('error') or ep.get('error'),
        execution_kind=ep.get('execution_kind'), control_spec=ep.get('control_spec'), reported_cost=ep.get('cost'), **cost)
    if not ep:
        if task.get('status') != 'failed':
            result['artifact_status'] = 'incomplete'
        return result
    problems = list(cost['cost_issues'])
    if (ep.get('version') != 'toolv2x_episode_v2' or
            any(ep.get(k) != result[k] for k in ('sample_id','scene','g','policy_id','branch_id'))):
        problems.append('episode identity/version mismatch')
    raw_plans = ep.get('plans', [])
    if ([p.get('stage') for p in raw_plans] != list(range(len(raw_plans))) or
            len({p.get('plan_id') for p in raw_plans}) != len(raw_plans)):
        problems.append('duplicate or unordered plan stages')
    for plan in raw_plans:
        result['plans'].append(method_plan_row(plan, usable_label, ep['ego_motion'], ep['limits']['execution_spec']))
    if result['plans']:
        last = result['plans'][-1]
        result.update(parse_valid=last['parse_valid'], admissibility=last['admissibility'],
                      raw_plan_id=last['plan_id'], raw_ADE3=last['raw_ADE3'], raw_FDE3=last['raw_FDE3'])
    if any(not p['record_consistent'] for p in result['plans']):
        problems.append('raw answer and saved plan contract disagree')
    events = ep.get('events', [])
    terminal = events[-1]['kind'] if events else None
    if ep.get('status') == 'completed':
        valid = (result['plans'] and terminal=='STOP' and
                 all(p['parse_valid'] and p['admissibility'] and p['saved_status']=='valid' for p in result['plans']) and
                 ep.get('final_plan_id')==result['plans'][-1]['plan_id'])
        if not valid:
            problems.append('completed episode lacks its actual last valid plan/STOP')
        elif not problems and task.get('status') in ('completed','running'):
            # A STOP snapshot is already terminal even if interrupted before the outer task status update.
            result.update(task_success=True, artifact_status='completed', ADE3=result['raw_ADE3'], FDE3=result['raw_FDE3'])
    elif ep.get('status')=='running' or terminal not in ('STOP','failed'):
        result['artifact_status'] = 'incomplete'
    elif ep.get('final_plan_id') is not None:
        problems.append('failed episode retained a success final pointer')
    if problems:
        result.update(artifact_status='invalid_artifact', task_success=False, ADE3=None, FDE3=None,
                      cost_complete=False, artifact_errors=problems)
    return result


def _known_stats(values):
    selected = [v for v in values if _number(v)]
    return dict(known_count=len(selected), unknown_count=len(values)-len(selected),
                known_sum=sum(selected), mean_known=float(np.mean(selected)) if selected else None)


def summarize_method(rows):
    keys = [(r['sample_id'],r['policy_id'],r['branch_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate method evaluation key')
    groups = []
    for policy, branch in sorted({(r['policy_id'],r['branch_id']) for r in rows}):
        selected = [r for r in rows if (r['policy_id'],r['branch_id'])==(policy,branch)]
        stages = []
        for stage in sorted({p['stage'] for r in selected for p in r['plans'] if type(p['stage']) is int}):
            items = [p for r in selected if r['artifact_status'] != 'invalid_artifact' for p in r['plans'] if p['stage']==stage]
            excluded = sum(p['stage']==stage for r in selected if r['artifact_status']=='invalid_artifact' for p in r['plans'])
            stages.append(dict(stage=stage, recorded_attempts=len(items), excluded_artifact_plans=excluded,
                parse_valid=sum(p['parse_valid'] for p in items),
                raw_metrics={k:_known_stats([p.get('raw_'+k) if p.get('record_consistent') else None for p in items])
                             for k in ('ADE3','FDE3')}))
        groups.append(dict(policy_id=policy, branch_id=branch, attempts=len(selected),
            task_successes=sum(r['task_success'] for r in selected), failures=sum(not r['task_success'] for r in selected),
            parse_valid=sum(r['parse_valid'] and r['artifact_status'] != 'invalid_artifact' for r in selected),
            excluded_artifact_attempts=sum(r['artifact_status']=='invalid_artifact' for r in selected),
            execution_kind_counts=dict(Counter(r['execution_kind'] for r in selected)),
            cost_complete_attempts=sum(r['cost_complete'] for r in selected),
            artifact_status_counts=dict(Counter(r['artifact_status'] for r in selected)),
            label_status_counts=dict(Counter(r['label_status'] for r in selected)),
            metrics={k:_known_stats([r[k] for r in selected]) for k in ('ADE3','FDE3')},
            costs={k:_known_stats([r[k] if r['artifact_status'] != 'invalid_artifact' else None for r in selected]) for k in METHOD_COST_FIELDS}, stages=stages))
    return dict(attempts=len(rows), task_successes=sum(r['task_success'] for r in rows),
                failures=sum(not r['task_success'] for r in rows), groups=groups)


def evaluate_method(episodes_root, out, labels_root):
    """Read T4 archives offline, including incomplete/missing tasks. Never load models or features."""
    from planning.run_framework import read_jsonl
    root, out, labels_root = Path(episodes_root), Path(out), Path(labels_root)
    config = json.loads((root/'config.json').read_text())
    if config.get('version') != 'toolv2x_interact_run_v1':
        raise ValueError('expected explicit T4 interaction run; v1 uses the legacy evaluator')
    policy = config['spec']['limits']['policy_id']
    # Each run declares one arm; future branch trees require their own expected manifest.
    branch=config.get('branch_id',policy)
    expected = read_jsonl(root/'selected_index.jsonl')
    if not expected or len({r['sample_id'] for r in expected}) != len(expected):
        raise ValueError('empty/duplicate expected interaction samples')
    issues, indexed, labels = [], defaultdict(list), {}
    for role in sorted({r['role'] for r in expected}):
        path = labels_root/(role+'.jsonl')
        if not path.is_file():
            issues.append('missing label file: '+path.name)
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                value = json.loads(line)
                key = (role, value['sample_id'])
                if key in labels:
                    labels[key] = dict(_label_error='duplicate')
                else:
                    labels[key] = value
            except (ValueError, KeyError, TypeError):
                issues.append('invalid label line %s:%d'%(path.name,line_number))
    manifest = root/'tasks.jsonl'
    if manifest.is_file():
        for i, line in enumerate(manifest.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                indexed[record['sample_id']].append(record['path'])
            except (ValueError, KeyError, TypeError):
                issues.append('incomplete/invalid task manifest line %d'%i)
    else:
        issues.append('missing task manifest')
    extras = sorted(set(indexed)-{r['sample_id'] for r in expected})
    issues += ['unexpected manifest sample: '+key for key in extras]
    rows = []
    for identity in expected:
        label = labels.get((identity['role'],identity['sample_id']))
        paths = indexed[identity['sample_id']]
        fallback = dict(row=identity, status='failed', episode=None)
        status, message = None, None
        if len(paths) != 1:
            status = 'duplicate_artifact' if paths else 'missing_artifact'
        else:
            try:
                task = json.loads((root/paths[0]).read_text())
                if task['row'] != identity:
                    raise ValueError('task identity or causal index differs from expected row')
                if task.get('episode') and task['episode']['limits'] != config['spec']['limits']:
                    raise ValueError('task execution spec differs from frozen run config')
                if task.get('episode') and task['episode'].get('control_spec') != config['spec'].get('control'):
                    raise ValueError('task control differs from frozen run config')
                row = evaluate_method_task(task, label, policy_id=policy, branch_id=branch)
            except FileNotFoundError:
                status = 'missing_artifact'
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                status, message = 'invalid_artifact', str(exc)
        if status:
            row = evaluate_method_task(fallback, label, policy_id=policy, branch_id=branch)
            row.update(artifact_status=status, artifact_error=message)
        row.update(task_paths=paths, source_run=str(root))
        rows.append(row)
    progress = json.loads((root/'progress.json').read_text()) if (root/'progress.json').is_file() else None
    if not progress or progress.get('status') != 'completed':
        issues.append('source run is incomplete or lacks progress record')
    if progress and progress.get('terminal_tasks') != len(expected):
        issues.append('source terminal count differs from expected samples')
    report = dict(version='toolv2x_method_evaluation_v1', status='evaluation_completed',
        **summarize_method(rows), archive_issues=issues, source_progress=progress,
        by_recording={group:summarize_method([r for r in rows if r['recording']==group])
                      for group in sorted({r['recording'] for r in rows})},
        timing_scope=METHOD_TIMING_SCOPE, cost_scope='toolv2x_measured_stages_v3' if config['spec'].get('control') else 'toolv2x_measured_stages_v1',
        task_success_definition='actual terminal STOP with valid direct answer and configured admissibility; not safety or low-error success',
        raw_metric_scope='last recorded driver attempt may be only a failed episode prefix; never substituted for successful final metrics',
        label_scope='offline only; missing/partial labels change metric coverage, not parsing or execution success',
        new_model_generation=False, collision_evaluated=False, closed_loop_evaluated=False)
    out.mkdir(parents=True, exist_ok=False)
    (out/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    stages = [dict(p, **{k:r[k] for k in ('sample_id','policy_id','branch_id','scene','role','g')}) for r in rows for p in r['plans']]
    (out/'plan_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in stages))
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('generations',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--data',type=Path,default=ROOT/'outputs/paired_driving_data_v1')
    parser.add_argument('--method', action='store_true', help='evaluate v2 interaction archives including missing/failed tasks')
    parser.add_argument('--labels', type=Path, help='offline directory containing role.jsonl labels; required with --method')
    args=parser.parse_args()
    if args.method:
        if args.labels is None:
            parser.error('--method requires an explicit --labels directory')
        result=evaluate_method(args.generations.resolve(),args.out.resolve(),args.labels.resolve())
        print(json.dumps(dict(attempts=result['attempts'], task_successes=result['task_successes'])))
    else:
        evaluate(args.generations.resolve(),args.out.resolve(),args.data.resolve())
