"""Offline-only preparation: causal input index, separate native labels and SFT rows.

The online planner and tools must never import this module. No model is trained here.
"""
import argparse
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from common.audit_protocol import recording
from planning.inputs import load_ego_features, load_ego_motion, make_prompt, parse_q8, SPEEDS, STEERING

NATIVE = ROOT / 'vendor/v2vgot_opencood/inference.py'
OFFSETS = (5, 10, 15, 20, 25, 30)


@lru_cache(maxsize=1)
def native_labels():
    # Execute only the two pure original functions, avoiding the upstream GT QA
    # graph, OpenCOOD CLI imports and any model/data initialization.
    tree = ast.parse(NATIVE.read_text())
    names = {'get_suggested_speed_steering', 'get_future_trajectory_str'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != 2:
        raise ValueError('native Q8/Q9 label functions missing')
    namespace = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NATIVE), 'exec'), namespace)
    return namespace


def ego_label(split, g, pose_loader=M.load_pose):
    if split not in M.LEN_RECORD or not 0 <= g < M.num_frames(split):
        raise ValueError('invalid offline label frame')
    end = M.seq_of(g, split)[3]
    origin = np.asarray(pose_loader(split, g), dtype=float)
    if origin.shape != (4, 4) or not np.isfinite(origin).all():
        raise ValueError('invalid current localization for label')
    inverse = np.linalg.inv(origin)
    waypoints, valid = [], []
    reads = [str(Path(M.pose_dir(split)) / ('%04d_lidar_pose.npy' % g))]
    trajectory = np.full((30, 2), np.nan)
    for offset in OFFSETS:
        point = None
        if g + offset < end:
            path = str(Path(M.pose_dir(split)) / ('%04d_lidar_pose.npy' % (g + offset)))
            reads.append(path)
            try:
                future = np.asarray(pose_loader(split, g + offset), dtype=float)
                if future.shape == (4, 4) and np.isfinite(future).all():
                    point = (inverse @ future)[:2, 3].tolist()
            except FileNotFoundError:
                pass
        waypoints.append(point)
        valid.append(point is not None)
        if point is not None:
            trajectory[offset - 1] = point
    target8 = target9 = None
    if all(valid):
        native = native_labels()
        speed, steering, _, _ = native['get_suggested_speed_steering'](np.zeros(2), trajectory, 6)
        target8 = 'The suggested speed setting is: %s. The suggested steering setting is: %s.' % (SPEEDS[speed], STEERING[steering])
        target9 = 'The suggested future trajectory is ' + native['get_future_trajectory_str'](trajectory, 6) + '.'
    return dict(waypoints=waypoints, valid=valid, times_seconds=[v / 10 for v in OFFSETS],
                target_q8=target8, target_q9=target9, label_read_paths=reads,
                scope='offline observed ego-trajectory imitation labels, not optimal planning labels')


def supervised_examples(role, sample_id, action, motion, plan, label, feature_path):
    if role != 'train':
        raise ValueError('only train recordings may become SFT examples')
    mode = plan['evidence_selection']['evidence_format']
    evidence = plan['evidence_used']
    common = dict(sample_id=sample_id, action=action, feature_path=feature_path,
                  evidence_format=mode, supervision='assistant target only')
    decoding = plan.get('decoding', 'q8_q9')
    if decoding == 'direct':
        if plan.get('q8_executed') is not False or plan.get('q8_raw'):
            raise ValueError('direct supervision cannot contain a Q8 parent')
        prompt = make_prompt('Trajectory', motion, evidence, evidence_format=mode)
        if prompt != plan['q9_prompt']:
            raise ValueError('direct prompt does not match causal evidence')
        return ([dict(common, task='Trajectory', prompt=prompt, target=label['target_q9'],
                      q8_parent_source='not applicable')] if label['target_q9'] is not None else [])
    if decoding != 'q8_q9':
        raise ValueError('unknown supervision decoding mode')
    prompt8 = make_prompt('Q8', motion, evidence, evidence_format=mode)
    if prompt8 != plan['q8_prompt']:
        raise ValueError('saved prompt does not match causal evidence')
    rows = []
    if label['target_q8'] is not None:
        rows.append(dict(common, task='Q8', prompt=prompt8, target=label['target_q8']))
    try:
        parse_q8(plan['q8_raw'])
    except ValueError:
        return rows
    prompt9 = make_prompt('Q9', motion, evidence, plan['q8_raw'], evidence_format=mode)
    if prompt9 != plan['q9_prompt']:
        raise ValueError('Q9 must use the actual generated Q8 parent')
    if label['target_q9'] is not None:
        rows.append(dict(common, task='Q9', prompt=prompt9, target=label['target_q9'],
                         q8_parent_source='saved model-generated answer'))
    return rows


def encode_supervision(row, tokenizer, feature_tokens, context_limit=4096):
    """Single-example native chat tokens; supervise only the assistant answer."""
    import torch
    from planning.v2vgot import prompt_tokens
    prefix = prompt_tokens(tokenizer, row['prompt'])
    target = torch.tensor(tokenizer(row['target'], add_special_tokens=False).input_ids +
                          [tokenizer.eos_token_id], dtype=torch.long)
    ids = torch.cat([prefix, target])
    if len(ids) - 1 + feature_tokens > context_limit:
        raise ValueError('supervised example exceeds original model context')
    labels = torch.cat([torch.full_like(prefix, -100), target])
    return dict(input_ids=ids[None], labels=labels[None])


def prepare(out, runs=()):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('online_index', 'offline_labels', 'examples', 'code_snapshot'):
        (out / name).mkdir()
    manifest_root = ROOT / 'outputs/protocol_audit'
    manifest = json.loads((manifest_root / 'split_manifest.json').read_text())
    groups = {role: {recording(s) for s in manifest[role + '_scenes']} for role in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('recording holdout overlap')
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'code_snapshot/adaptation_data.py').write_bytes(Path(__file__).read_bytes())
    (out / 'code_snapshot/native_inference.py').write_bytes(NATIVE.read_bytes())
    dump = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
    cached_pose = lru_cache(maxsize=1024)(M.load_pose)
    counts, labels = {}, {}
    for role in ('train', 'validation'):
        frames = json.loads((manifest_root / (role + '_frames.json')).read_text())
        if len({(s, t) for s, t in frames}) != len(frames):
            raise ValueError('duplicate decision frame')
        valid_count = 0
        with (out / 'online_index' / (role + '.jsonl')).open('w') as online, (out / 'offline_labels' / (role + '.jsonl')).open('w') as offline:
            for scene, local_frame in frames:
                if scene not in manifest[role + '_scenes']:
                    raise ValueError('frame does not belong to requested recording role')
                index = manifest['source_scenes']['train'].index(scene)
                _, start, end = M.seq_ranges('train')[index]
                g = start + local_frame
                if not start <= g < end:
                    raise ValueError('decision frame crosses recording')
                sample_id = scene + ':' + str(local_frame)
                features = load_ego_features(M.V2VGOT_ROOT, 'train', g)
                motion, motion_paths = load_ego_motion(M.V2VGOT_ROOT, 'train', g)
                windows = {source: str(ROOT / 'outputs/causal_windows_v1/train' / source / (scene + '.pkl'))
                           for source in ('no_fusion', 'no_fusion_cav1')}
                if not all(Path(path).is_file() for path in windows.values()):
                    raise FileNotFoundError('missing causal history archive')
                record = dict(sample_id=sample_id, role=role, scene=scene, local_frame=local_frame, g=g,
                    physical_split='train', feature_read_paths=features['read_paths'],
                    feature_frames=features['frame_indices'], ego_motion=motion, motion_read_paths=motion_paths,
                    window_archive_refs=windows, action_queries={'Ego': [], 'P': ['P'], 'F': ['F'], 'PF': ['P', 'F']},
                    tool_outputs_materialized=False)
                online.write(dump(record))
                label = dict(ego_label('train', g, cached_pose), sample_id=sample_id, role=role, g=g)
                offline.write(dump(label))
                labels[sample_id] = label
                valid_count += int(all(label['valid']))
        counts[role] = dict(frames=len(frames), complete_ego_labels=valid_count,
                            planned_action_conditions=len(frames) * 4)
        print(json.dumps(dict(role=role, **counts[role])), flush=True)
    generated, accepted = 0, []
    with (out / 'examples/train.jsonl').open('w') as examples:
        for run in runs:
            run = Path(run).resolve()
            meta = json.loads((run / 'connection.json').read_text())
            sample_id = meta['scene'] + ':' + str(meta['local_frame'])
            if (meta['research_split'] != 'train' or sample_id not in labels or labels[sample_id]['role'] != 'train'
                    or meta['g'] != labels[sample_id]['g'] or meta['physical_split'] != 'train'):
                raise ValueError('run does not match an authorized training frame')
            for action in meta['actions']:
                plan = json.loads((run / action / 'plan.json').read_text())
                rows = supervised_examples('train', sample_id, action, meta['ego_motion'], plan,
                                           labels[sample_id], str(run / 'ego_features.npz'))
                for row in rows:
                    examples.write(dump(dict(row, source_run=str(run))))
                generated += len(rows)
            accepted.append(str(run))
    summary = dict(scope='offline training preparation, no optimization performed', roles=counts,
        test_inputs_or_labels_read=False, input_labels_separate=True, materialized_sft_examples=generated,
        materialized_runs=accepted, q9_parent='actual generated Q8 only',
        native_label_source=str(NATIVE), pretrained_provenance_verified=False,
        all_training_tool_outputs_ready=False)
    (out / 'preparation.json').write_text(dump(summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--runs', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    prepare(args.out, args.runs)
