"""Run causal tool episodes and the original driver; never import future labels."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from planning.episode import POLICIES, run_episode
from common.audit_protocol import recording


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_prefix(root, rows, config):
    """Reuse confirmed complete frames; leave the unfinished tail in its original run."""
    from planning.inputs import make_prompt
    old = json.loads((root/'config.json').read_text())
    keys = ('source_data','per_recording','policies','p_processing','input_layout',
            'mtr_checkpoint','context_limit','peer_reserve')
    if any(old.get(k) != config[k] for k in keys):
        raise ValueError('resume preparation configuration changed')
    if read_jsonl(root/'selected_index.jsonl') != rows:
        raise ValueError('resume causal index changed')
    progress = json.loads((root/'progress.json').read_text())
    frames = progress['frames']
    if isinstance(frames, bool) or not isinstance(frames, int) or not 0 <= frames <= len(rows):
        raise ValueError('invalid completed frame prefix')
    count = frames*len(config['policies'])
    lines = (root/'tasks.jsonl').read_text().splitlines()
    if progress['tasks'] != count or len(lines) < count:
        raise ValueError('incomplete task prefix')
    records = [json.loads(line) for line in lines[:count]]
    for number, record in enumerate(records):
        row = rows[number//len(config['policies'])]
        expected = dict(row, policy=config['policies'][number%len(config['policies'])])
        task = json.loads(Path(record['path']).read_text())
        if any(record[k] != expected[k] or task[k] != expected[k] for k in ('sample_id','scene','role','g','policy')):
            raise ValueError('resume task identity changed')
        if any(task[k] != row[k] for k in ('ego_motion','feature_read_paths','motion_read_paths')):
            raise ValueError('resume task causal input changed')
        if number % len(config['policies']) == 0:
            feature_path = task['feature_path']
        elif task['feature_path'] != feature_path:
            raise ValueError('resume policies no longer share ego features')
        if (task['status'] != 'prepared' or task['episode']['status'] != 'tools_completed' or
                task['episode']['p_processing'] != config['p_processing'] or not Path(task['feature_path']).is_file()):
            raise ValueError('resume prefix contains an unsuccessful or unavailable task')
        prepared = task['prepared']
        prompt = make_prompt('Trajectory',task['ego_motion'],prepared['evidence_used'],evidence_format='compact',
                             remote_evidence=prepared['remote_evidence_used'])
        if prepared['input_layout'] != config['input_layout'] or prompt != prepared['q9_prompt']:
            raise ValueError('resume task prompt changed')
    return records, dict(resume_from=str(root), reused_frames=frames, reused_tasks=count,
                         unreused_task_lines=len(lines)-count, original_progress=progress)


def check_resume_source(root):
    # Only this orchestration module may change when adding preparation resumption.
    snapshot = root/'code_snapshot'
    for name in ('src','vendor/cmp_mtr','v2vgot_original'):
        source = snapshot/name
        if not source.is_dir():
            raise ValueError('missing preparation source snapshot')
        for path in source.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(snapshot)
            if relative == Path('src/planning/run_framework.py'):
                continue
            target = (ROOT/'vendor/v2vgot_llava'/path.relative_to(source)
                      if name == 'v2vgot_original' else ROOT/relative)
            if not target.is_file() or path.read_bytes() != target.read_bytes():
                raise ValueError('preparation dependency changed: '+str(relative))


def completed_tasks(root):
    """Require every selected frame/policy, including failed attempts, exactly once."""
    progress = json.loads((root/'progress.json').read_text())
    config = json.loads((root/'config.json').read_text())
    if progress['status'] != 'completed':
        raise ValueError('incomplete online episode run')
    selected = read_jsonl(root/'selected_index.jsonl')
    expected = {(r['sample_id'], p): r for r in selected for p in config['policies']}
    records = read_jsonl(root/'tasks.jsonl')
    keys = [(r['sample_id'], r['policy']) for r in records]
    if (len(keys) != len(set(keys)) or set(keys) != set(expected) or len(keys) != progress['tasks'] or
            len(selected) != progress['expected_frames']):
        raise ValueError('online task coverage mismatch')
    for record in records:
        source = expected[(record['sample_id'], record['policy'])]
        task = json.loads(Path(record['path']).read_text())
        if (any(record[k] != source[k] for k in ('scene', 'g', 'role')) or
                any(record[k] != task[k] for k in ('sample_id', 'scene', 'g', 'role', 'policy'))):
            raise ValueError('online task identity mismatch')
    return records


def select_rows(data, per_recording=0):
    if per_recording < 0:
        raise ValueError('per-recording count must be nonnegative')
    groups = defaultdict(list)
    for role in ('train', 'validation'):
        for row in read_jsonl(data / 'online_index' / (role + '.jsonl')):
            if row['role'] != role or row['physical_split'] != 'train':
                raise ValueError('invalid causal index role')
            groups[(role, recording(row['scene']))].append(row)
    result = []
    for group in groups.values():
        indices = np.linspace(0, len(group)-1, min(per_recording, len(group)), dtype=int) if per_recording else range(len(group))
        result.extend(group[i] for i in indices)
    return result


def prepare(data, out, per_recording=0, predict_p=True, policies=POLICIES, resume_from=None):
    import torch
    from transformers import AutoTokenizer
    from common import v2v4real_meta as M
    from prediction import cmp_adapter as C
    from planning.context import build_plan_input
    from planning.inputs import load_ego_features, load_ego_motion
    from planning.run_connection import load_window, save_json, snapshot_code
    from planning.v2vgot import CHECKPOINT
    from tools.vehicle import VehicleTools

    rows = select_rows(data, per_recording)
    if not rows or not policies or set(policies) - set(POLICIES) or len(policies) != len(set(policies)):
        raise ValueError('empty or invalid task selection')
    config = json.loads((data / 'config.json').read_text())
    run_config = dict(source_data=str(data), per_recording=per_recording, policies=list(policies),
        p_processing='local_mtr' if predict_p else 'observations', input_layout='source_blocks_v1',
        mtr_checkpoint=config['mtr_checkpoint'], context_limit=4096, peer_reserve=1536,
        real_rgb_input=False, gt_labels_read=False, driver_executed=False, scope='complete tool episodes and shared inputs')
    previous, resume = [], dict(reused_frames=0, reused_tasks=0)
    if resume_from:
        check_resume_source(resume_from)
        previous, resume = completed_prefix(resume_from, rows, run_config)
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    (out / 'frames').mkdir()
    (out / 'selected_index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    save_json(out/'config.json', dict(run_config, resume_from=str(resume_from) if resume_from else None))
    save_json(out/'resumption.json', resume)
    (out/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous))
    progress = dict(status='initializing', frames=resume['reused_frames'], expected_frames=len(rows),
        tasks=len(previous), task_failures=0, reused_frames=resume['reused_frames'], new_frames=0, elapsed_seconds=0.)
    save_json(out/'progress.json', progress)
    if resume['reused_frames'] == len(rows):
        save_json(out/'mtr_loading.json', json.loads((resume_from/'mtr_loading.json').read_text()))
        save_json(out/'progress.json', dict(progress, status='completed'))
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, provenance = C.load_model(checkpoint=Path(config['mtr_checkpoint']), device='cuda')
    save_json(out / 'mtr_loading.json', provenance)
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)

    @lru_cache(maxsize=2)
    def archive(path):
        with path.open('rb') as handle:
            return pickle.load(handle)

    begin = perf_counter()
    completed, failed = len(previous), 0
    with (out / 'tasks.jsonl').open('a') as tasks:
        for index, row in enumerate(rows[resume['reused_frames']:], start=resume['reused_frames']):
            directory = out / 'frames' / ('%s_g%04d' % (row['role'], row['g']))
            directory.mkdir()
            reads = []
            def window(source):
                value = load_window(row['scene'], row['local_frame'], source, reads, archive)
                if int(value['g']) != row['g']:
                    raise ValueError('indexed frame mismatch')
                return value
            ego = window('no_fusion')
            features = load_ego_features(M.V2VGOT_ROOT, 'train', row['g'])
            motion, motion_paths = load_ego_motion(M.V2VGOT_ROOT, 'train', row['g'])
            if (features['read_paths'] != row['feature_read_paths'] or motion != row['ego_motion'] or
                    motion_paths != row['motion_read_paths']):
                raise ValueError('causal input changed since index preparation')
            feature_path = directory / 'ego_features.npz'
            np.savez(str(feature_path), **{k: v for k, v in features.items() if isinstance(v, np.ndarray)})
            start = perf_counter()
            local_prediction = C.predict(model, ego, batch_size=32)
            local_seconds = perf_counter() - start
            np.savez(str(directory / 'local_forecast.npz'), **local_prediction)
            for policy in policies:
                target = directory / policy
                target.mkdir()
                calls = []
                def predict(w):
                    start = perf_counter()
                    result = C.predict(model, w, batch_size=32)
                    path = target / ('forecast_%02d.npz' % len(calls))
                    np.savez(str(path), **result)
                    calls.append(dict(source=w['source'], g=int(w['g']), targets=len(w['track_ids']),
                        model_targets=int(result['model_used'].sum()), seconds=perf_counter()-start, path=str(path)))
                    return result
                read_start = len(reads)
                service = VehicleTools(lambda: window('no_fusion_cav1'), predict, row['scene'], row['g'])
                task = dict(sample_id=row['sample_id'], scene=row['scene'], g=row['g'], role=row['role'], policy=policy,
                    status='started', feature_path=str(feature_path), ego_motion=motion,
                    local_model_seconds=local_seconds, local_model_targets=int(local_prediction['model_used'].sum()),
                    feature_read_paths=features['read_paths'], motion_read_paths=motion_paths,
                    remote_reads=reads[read_start:], predictor_calls=calls, actual_rgb_input=False)
                task_path = target / 'task.json'
                save_json(task_path, task)
                tasks.write(json.dumps(dict(path=str(task_path), sample_id=row['sample_id'], scene=row['scene'],
                                            g=row['g'], role=row['role'], policy=policy)) + '\n')
                tasks.flush()
                def persist(episode):
                    task.update(episode=episode, remote_reads=reads[read_start:])
                    save_json(task_path, task)
                    for number, step in enumerate(episode['steps']):
                        if 'response' in step:
                            from probe.kinematic_tools import encode
                            (target / ('response_%d.json' % number)).write_bytes(encode(step['response']))
                try:
                    episode = run_episode(ego, local_prediction, motion, service, predict, policy,
                                          predict_p=predict_p, on_progress=persist)
                    if episode['status'] != 'tools_completed':
                        task.update(status='tool_error', error=episode['error'])
                    else:
                        prepared = build_plan_input(tokenizer, motion, episode['evidence'],
                                                    int(features['active_agent_mask'].sum())*270)
                        task.update(status='prepared', prepared=prepared)
                    if policy == 'Ego' and reads[read_start:]:
                        raise AssertionError('Ego accessed an unqueried source')
                except Exception as exc:
                    task.update(status='preparation_error', error=dict(error_type=type(exc).__name__, message=str(exc)))
                task['remote_reads'] = reads[read_start:]
                save_json(task_path, task)
                failed += task['status'] != 'prepared'
                completed += 1
            progress = dict(status='running', frames=index+1, expected_frames=len(rows), tasks=completed,
                            task_failures=failed, reused_frames=resume['reused_frames'],
                            new_frames=index+1-resume['reused_frames'], elapsed_seconds=perf_counter()-begin)
            save_json(out / 'progress.json', progress)
            if index % 25 == 0 or index+1 == len(rows):
                print(json.dumps(progress), flush=True)
    save_json(out / 'progress.json', dict(progress, status='completed'))


def generate(prepared_root, out, checkpoint, role='validation', per_recording=0):
    from planning.run_connection import save_json, snapshot_code
    from planning.v2vgot import V2VGoTPlanner
    if per_recording < 0:
        raise ValueError('per-recording count must be nonnegative')
    records = [r for r in completed_tasks(prepared_root) if r['role'] == role]
    if per_recording:
        groups = defaultdict(list)
        for row in records:
            group = recording(row['scene'])
            if row['sample_id'] not in groups[group]:
                groups[group].append(row['sample_id'])
        keep = {samples[i] for samples in groups.values() for i in
                np.linspace(0, len(samples)-1, min(per_recording, len(samples)), dtype=int)}
        records = [r for r in records if r['sample_id'] in keep]
    if not records:
        raise ValueError('no prepared tasks for generation')
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    (out/'selected_tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    save_json(out / 'config.json', dict(prepared_root=str(prepared_root), checkpoint=str(checkpoint), role=role,
        input_layout='source_blocks_v1', tasks=len(records),
        execution='generate from causally completed rule/fixed episodes; no new queries', gt_labels_read=False))
    planner = V2VGoTPlanner(checkpoint=checkpoint, adapter_directory=out / 'loaded_adapter', evidence_format='compact')
    save_json(out/'model.json', planner.provenance)
    with (out / 'generations.jsonl').open('w') as handle:
        for number, record in enumerate(records):
            task = json.loads(Path(record['path']).read_text())
            prepared = task.get('prepared', dict(decoding='direct', q8_executed=False, q9_executed=False))
            start = perf_counter()
            try:
                if task['status'] != 'prepared':
                    plan = dict(prepared, status=task['status'], error=task.get('error'), q9_executed=False)
                else:
                    with np.load(task['feature_path'], allow_pickle=False) as saved:
                        features = dict(saved)
                    plan = planner.plan_prepared(features, prepared)
            except Exception as exc:
                plan = dict(prepared, status='generation_error', error_type=type(exc).__name__, error=str(exc))
            path = out / ('%s_g%04d_%s.json' % (role, record['g'], record['policy']))
            save_json(path, dict(task_path=record['path'], plan=plan, driver_attempt_seconds=perf_counter()-start))
            handle.write(json.dumps(dict(record, generation_path=str(path))) + '\n')
            handle.flush()
            print(json.dumps(dict(completed=number+1, total=len(records), policy=record['policy'], status=plan['status'])), flush=True)
    save_json(out / 'completion.json', dict(status='completed', attempts=len(records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('out', type=Path)
    prepare_parser.add_argument('--data', type=Path, default=ROOT / 'outputs/paired_driving_data_v1')
    prepare_parser.add_argument('--per-recording', type=int, default=0)
    prepare_parser.add_argument('--p-processing', choices=('local_mtr', 'observations'), default='local_mtr')
    prepare_parser.add_argument('--resume-from', type=Path, help='reuse the completed frame prefix from a previous preparation')
    gen_parser = commands.add_parser('generate')
    gen_parser.add_argument('prepared_root', type=Path)
    gen_parser.add_argument('out', type=Path)
    gen_parser.add_argument('--checkpoint', type=Path, required=True)
    gen_parser.add_argument('--role', choices=('train', 'validation'), default='validation')
    gen_parser.add_argument('--per-recording', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.data.resolve(), args.out.resolve(), args.per_recording, args.p_processing == 'local_mtr',
                    resume_from=args.resume_from.resolve() if args.resume_from else None)
        else:
            generate(args.prepared_root.resolve(), args.out.resolve(), args.checkpoint.resolve(), args.role, args.per_recording)
    except Exception as exc:
        if args.out.is_dir() and not isinstance(exc, FileExistsError):
            (args.out / 'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)))+'\n')
        raise
