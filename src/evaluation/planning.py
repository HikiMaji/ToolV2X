"""Offline audit of saved original V2V-GoT Q8/Q9 generations, without regeneration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.audit_protocol import recording
from planning.inputs import parse_q8, parse_q9

TIMES = [.5, 1., 1.5, 2., 2.5, 3.]
GOT_PREFIX = ('got_prefix_L2_1s', 'got_prefix_L2_2s', 'got_prefix_L2_3s', 'got_prefix_L2_avg')
QUALITY = ('ADE3', 'FDE3', 'L2_1', 'L2_2', 'first_segment_speed_mps', 'peak_segment_speed_mps',
           'initial_acceleration_estimate_mps2', 'peak_acceleration_estimate_mps2') + GOT_PREFIX


def trajectory_metrics(points, label, speed):
    points = np.asarray(points, float)
    valid = np.asarray(label['valid'])
    if (points.shape != (6, 2) or not np.isfinite(points).all() or valid.shape != (6,) or
            valid.dtype != bool or label['times_seconds'] != TIMES or len(label['waypoints']) != 6):
        raise ValueError('invalid planning prediction or label')
    truth = np.zeros((6, 2))
    for i in np.flatnonzero(valid):
        point = np.asarray(label['waypoints'][i], float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('invalid valid planning label')
        truth[i] = point
    error = np.linalg.norm(points - truth, axis=1)
    # Segment mean velocities lie at .25, .75, ..., 2.75 s, with current v at 0 s.
    velocity = np.diff(np.vstack([np.zeros((1, 2)), points]), axis=0) / .5
    speeds = np.linalg.norm(velocity, axis=1)
    acceleration = np.linalg.norm(np.diff(velocity, axis=0) / .5, axis=1)
    initial = None
    if speed is not None:
        if not np.isfinite(speed) or speed < 0:
            raise ValueError('invalid current speed')
        initial = float(np.linalg.norm((velocity[0] - [speed, 0.]) / .25))
    prefix = {key: float(error[:n].mean()) if valid[:n].all() else None
              for key, n in zip(GOT_PREFIX[:3], (2, 4, 6))}
    prefix[GOT_PREFIX[3]] = float(np.mean(list(prefix.values()))) if valid.all() else None
    return dict(**prefix, valid_label_points=int(valid.sum()), ADE3=float(error[valid].mean()) if valid.any() else None,
        FDE3=float(error[-1]) if valid[-1] else None, L2_1=float(error[1]) if valid[1] else None,
        L2_2=float(error[3]) if valid[3] else None, first_segment_speed_mps=float(speeds[0]),
        peak_segment_speed_mps=float(speeds.max()), initial_acceleration_estimate_mps2=initial,
        peak_acceleration_estimate_mps2=max(float(acceleration.max()), initial if initial is not None else 0.),
        all_future_points_equal=bool(np.max(np.linalg.norm(points - points[0], axis=1)) < 1e-6))


def evaluate_plan(plan, label, motion):
    decoding = plan.get('decoding', 'q8_q9')
    if decoding not in ('direct', 'q8_q9'):
        raise ValueError('unknown saved decoding mode')
    direct = decoding == 'direct'
    if direct and (plan.get('q8_executed') is not False or plan.get('q8_raw') or
                   'Context from the generated action answer:' in plan.get('q9_prompt', '')):
        raise ValueError('direct generation cannot contain or execute a Q8 parent')
    row = dict(saved_status=plan.get('status', 'missing'), decoding=decoding, q8_applicable=not direct,
        q8_parse_success=False,
        q9_executed=bool(plan.get('q9_executed', False)), q9_parse_success=False, errors=[],
        q8_label_available=not direct and label.get('target_q8') is not None,
        q8_speed_correct=None, q8_steering_correct=None, q8_joint_correct=None,
        q8_prediction=None, q8_target=None, q9_generated_parent_verified=False,
        valid_label_points=int(sum(label['valid'])), all_future_points_equal=None,
        **{key: None for key in QUALITY})
    if not direct:
        try:
            row['q8_prediction'] = parse_q8(plan.get('q8_raw') or '')
            row['q8_parse_success'] = True
        except ValueError as exc:
            row['errors'].append(str(exc))
    if row['q8_label_available']:
        target = parse_q8(label['target_q8'])
        row['q8_target'] = target
        for key in ('speed', 'steering'):
            row['q8_' + key + '_correct'] = row['q8_parse_success'] and row['q8_prediction'][key] == target[key]
        row['q8_joint_correct'] = row['q8_speed_correct'] and row['q8_steering_correct']
    if row['q9_executed']:
        if direct:
            row['q9_generated_parent_verified'] = None
        else:
            expected = 'Context from the generated action answer: ' + (plan.get('q8_raw') or '') + '\n'
            row['q9_generated_parent_verified'] = row['q8_parse_success'] and expected in plan.get('q9_prompt', '')
            if not row['q9_generated_parent_verified']:
                raise ValueError('saved Q9 did not retain its generated Q8 parent')
        try:
            points = parse_q9(plan.get('q9_raw', ''))
        except ValueError as exc:
            row['errors'].append(str(exc))
        else:
            if 'waypoints' in plan and not np.array_equal(points, np.asarray(plan['waypoints'])):
                raise ValueError('stored waypoints differ from raw generation')
            row['q9_parse_success'] = True
            row.update(trajectory_metrics(points, label, motion.get('speed_mps')))
    return row


def summarize(rows):
    labelled = sum(r['q8_label_available'] for r in rows)
    result = dict(attempts=len(rows), saved_status_counts=dict(Counter(r['saved_status'] for r in rows)),
        decoding_counts=dict(Counter(r['decoding'] for r in rows)),
        q8_parse_successes=sum(r['q8_parse_success'] for r in rows),
        q9_executed=sum(r['q9_executed'] for r in rows), q9_parse_successes=sum(r['q9_parse_success'] for r in rows),
        q9_failure_fraction=sum(not r['q9_parse_success'] for r in rows) / len(rows) if rows else None,
        q8_labelled_attempts=labelled)
    for kind in ('speed', 'steering', 'joint'):
        key = 'q8_' + kind
        correct = sum(r[key + '_correct'] is True for r in rows)
        result.update({key + '_correct': correct, key + '_accuracy': correct / labelled if labelled else None})
    for key in QUALITY:
        values = [r[key] for r in rows if r[key] is not None]
        result.update({key: float(np.mean(values)) if values else None, key + '_targets': len(values)})
    return result


def run(out, paths, labels_dir):
    out.mkdir(parents=True, exist_ok=False)
    labels = {}
    for role in ('train', 'validation'):
        for line in (labels_dir / (role + '.jsonl')).read_text().splitlines():
            row = json.loads(line)
            key = role, row['sample_id']
            if key in labels or row['role'] != role:
                raise ValueError('duplicate or mislabeled ego supervision')
            labels[key] = row
    rows, sources = [], []
    for path in paths:
        meta = json.loads((path / 'connection.json').read_text())
        sample = meta['scene'] + ':' + str(meta['local_frame'])
        key = meta['research_split'], sample
        if (key not in labels or meta['physical_split'] != 'train' or meta['g'] != labels[key]['g'] or
                meta['language_model_executed'] is not True):
            raise ValueError('run provenance differs from offline labels')
        for action in meta['actions']:
            plan_path = path / action / 'plan.json'
            plan = json.loads(plan_path.read_text())
            row = evaluate_plan(plan, labels[key], meta['ego_motion'])
            row.update(run=path.name, sample_id=sample, role=key[0], g=meta['g'], action=action,
                recording=recording(meta['scene']), plan_path=str(plan_path), current_motion=meta['ego_motion'])
            rows.append(row)
        sources.append(str(path))
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    summary = dict(scope='saved-output diagnostic; no new generation and no assumption about model adaptation status',
        runs={p.name: summarize([r for r in rows if r['run'] == p.name]) for p in paths},
        unique_decision_frames=len({r['sample_id'] for r in rows}), saved_runs=sources,
        labels=str(labels_dir), comparison='keep runs separate; repeated actions/frames are not independent samples',
        dynamics_definition='origin is (0,0); mean segment velocities at midpoint times; initial velocity assumes ego +x heading',
        repeated_points_rule='reported as a pattern only, not automatically invalid or a failure',
        error_definition='metres against observed ego trajectory, only on valid labels; failures reported separately',
        evaluated=['Q8 parse and native-label correctness', 'Q9 parse/coverage and 1/2/3s error', 'initial and segment dynamics'],
        collision_evaluated=False, closed_loop_evaluated=False, new_model_generation=False, test_access=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (out / 'code').mkdir()
    for source in (Path(__file__), ROOT / 'src/planning/inputs.py', ROOT / 'tests/test_planning_evaluation.py'):
        shutil.copy2(source, out / 'code' / source.name)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'outputs/adaptation_data_v1/offline_labels')
    args = parser.parse_args()
    run(args.out.resolve(), [p.resolve() for p in args.runs], args.labels.resolve())
