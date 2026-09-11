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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('generations',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--data',type=Path,default=ROOT/'outputs/paired_driving_data_v1')
    args=parser.parse_args()
    evaluate(args.generations.resolve(),args.out.resolve(),args.data.resolve())
