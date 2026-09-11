"""Offline index/label preparation; real online materialization runs separately."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def crossed_jumps(row, jumps):
    return [j['to_g'] for j in jumps if j['scene'] == row['scene'] and row['g'] - 10 < j['to_g'] <= row['g'] + 30]


def initialize(out):
    from common import v2v4real_meta as M
    from common.audit_protocol import recording
    out.mkdir(parents=True, exist_ok=False)
    (out / 'online_index').mkdir()
    source = ROOT / 'outputs/adaptation_data_v1'
    manifest = read(source / 'split_manifest.json')
    groups = {role: {recording(scene) for scene in manifest[role + '_scenes']}
              for role in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('recording split overlap')
    jumps, intervals = [], Counter()
    for role in ('train', 'validation'):
        for scene in manifest[role + '_scenes']:
            seq = manifest['source_scenes']['train'].index(scene)
            _, start, end = M.seq_ranges('train')[seq]
            previous = np.asarray(M.load_pose('train', start), dtype=float)
            for g in range(start + 1, end):
                current = np.asarray(M.load_pose('train', g), dtype=float)
                if current.shape != (4, 4) or not np.isfinite(current).all() or not np.isfinite(previous).all():
                    raise ValueError('invalid localization in continuity scan')
                distance = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
                yaw = np.arctan2(current[1, 0], current[0, 0]) - np.arctan2(previous[1, 0], previous[0, 0])
                angle = float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw))))
                if distance > 20. or abs(angle) > 20.:
                    jumps.append(dict(role=role, scene=scene, from_g=g - 1, to_g=g,
                                      translation_m=distance, yaw_change_deg=angle))
                previous = current
                intervals[role] += 1
    excluded, counts, seen = [], {}, set()
    for role in ('train', 'validation'):
        original = rows(source / 'online_index' / (role + '.jsonl'))
        kept = []
        for row in original:
            if (row['role'] != role or row['physical_split'] != 'train' or
                    row['scene'] not in manifest[role + '_scenes'] or row['sample_id'] in seen):
                raise ValueError('invalid or duplicate causal input index')
            seen.add(row['sample_id'])
            breaks = crossed_jumps(row, jumps)
            if breaks:
                excluded.append(dict(sample_id=row['sample_id'], role=role, scene=row['scene'], g=row['g'],
                    crossed_jump_frames=breaks, reason='one-second history or three-second label crosses pose discontinuity'))
                continue
            if any(g > row['g'] for g in row['feature_frames'] if g is not None):
                raise ValueError('future feature frame')
            kept.append(dict(row, action_queries={'Ego': [], 'F': ['F']},
                             directory='frames/%s_g%04d' % (role, row['g'])))
        counts[role] = dict(original=len(original), kept=len(kept), excluded=len(original) - len(kept),
                            recordings=len({recording(r['scene']) for r in kept}))
        (out / 'online_index' / (role + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in kept))
    frozen = read(ROOT / 'outputs/mtr_stability_v1/frozen_model.json')
    save(out / 'split_manifest.json', manifest)
    save(out / 'config.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), source_index=str(source),
        actions=['Ego', 'F'], decoding='direct', evidence_format='compact', context_limit=4096,
        generation_reserve=256, peer_reserve=1536, mtr_checkpoint=frozen['checkpoint'],
        device='cuda', target_batch_size=32, counts=counts, nominal_dt_seconds=.1,
        actual_sensor_timestamps_verified=False, test_access=False, training=False,
        driver_initialization='existing GoT checkpoint-4330', validation_used_for_development=True))
    save(out / 'continuity.json', dict(translation_threshold_m=20., yaw_threshold_deg=20.,
         scanned_intervals=dict(intervals), jumps=jumps, excluded=excluded,
         rule='g-10 < discontinuity_to_g <= g+30 within the same physical scene',
         scope='offline sample eligibility; underlying source data and frozen MTR unchanged'))
    shutil.copy2(__file__, out / 'preparation_source.py')
    shutil.copy2(ROOT / 'docs/scope_reset_2026_09_11.md', out / 'scope_plan.md')
    print(json.dumps(dict(counts=counts, jumps=jumps)), flush=True)


def finalize(out):
    import torch
    from transformers import AutoTokenizer
    from common import v2v4real_meta as M
    from planning.adaptation_data import ego_label, supervised_examples, encode_supervision
    from planning.inputs import make_prompt, load_ego_features, unpack_evidence, pack_evidence
    from planning.v2vgot import CHECKPOINT, prompt_tokens
    from probe.kinematic_tools import encode
    torch.set_num_threads(1)
    config = read(out / 'config.json')
    status = read(out / 'online_status.json')
    expected = sum(config['counts'][role]['kept'] for role in ('train', 'validation'))
    if status['status'] != 'completed' or status['completed'] != expected:
        raise ValueError('online materialization is incomplete; no partial SFT export')
    if read(out / 'mtr_loading.json')['checkpoint'] != config['mtr_checkpoint']:
        raise ValueError('MTR checkpoint mismatch')
    (out / 'examples').mkdir()
    (out / 'offline_labels').mkdir()
    (out / 'evaluation').mkdir()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    cached_pose = lru_cache(maxsize=1024)(M.load_pose)
    count, coverage, groups = Counter(), [], defaultdict(list)
    with (out / 'examples/train.jsonl').open('w') as training, (out / 'evaluation/validation.jsonl').open('w') as evaluation:
        for role in ('train', 'validation'):
            original_labels = {r['sample_id']: r for r in rows(Path(config['source_index']) / 'offline_labels' / (role + '.jsonl'))}
            with (out / 'offline_labels' / (role + '.jsonl')).open('w') as labels:
                for row in rows(out / 'online_index' / (role + '.jsonl')):
                    directory = out / row['directory']
                    sample, ego, peer = (read(directory / name) for name in ('sample.json', 'ego_input.json', 'F_input.json'))
                    label = original_labels[row['sample_id']]
                    if (sample['sample_id'] != row['sample_id'] or sample['g'] != row['g'] or
                            sample['role'] != role or label['role'] != role or label['g'] != row['g'] or not all(label['valid'])):
                        raise ValueError('sample/label/role mismatch or incomplete trajectory label')
                    recomputed = ego_label('train', row['g'], cached_pose)
                    np.testing.assert_allclose(recomputed['waypoints'], label['waypoints'], rtol=0, atol=1e-6)
                    if recomputed['target_q9'] != label['target_q9']:
                        raise ValueError('native trajectory supervision changed')
                    labels.write(json.dumps(label) + '\n')
                    original_features = load_ego_features(M.V2VGOT_ROOT, 'train', row['g'])
                    with np.load(sample['feature_path'], allow_pickle=False) as arrays:
                        for key in arrays.files:
                            np.testing.assert_array_equal(arrays[key], original_features[key])
                    if ego['remote_archive_reads'] or ego['request_bytes'] or ego['response_bytes']:
                        raise ValueError('Ego accessed peer information')
                    if (peer['remote_archive_reads'] != 1 or len(sample['window_reads']) != 2 or
                            sample['window_reads'][0]['source'] != 'no_fusion' or
                            sample['window_reads'][1]['source'] != 'no_fusion_cav1'):
                        raise ValueError('incorrect query/read count')
                    costs = peer['tool_costs']
                    if (len((directory / 'F_response.json').read_bytes()) != costs['response_bytes'] or
                            len(encode(read(directory / 'F_request.json'))) != costs['request_bytes']):
                        raise ValueError('wire byte count mismatch')
                    if not peer['prompt'].startswith(ego['prompt'].split('Output six numeric')[0]):
                        raise ValueError('literal local block changed')
                    prompts = {}
                    for action, remote in (('Ego', None), ('F', peer['evidence_used'])):
                        for block in [ego['evidence_used']] + ([] if remote is None else [remote]):
                            if unpack_evidence(pack_evidence(block)) != block:
                                raise ValueError('compact evidence did not round trip')
                        prompt = make_prompt('Trajectory', row['ego_motion'], ego['evidence_used'],
                                             evidence_format='compact', remote_evidence=remote)
                        if prompt != (ego if action == 'Ego' else peer)['prompt']:
                            raise ValueError('saved prompt does not match acquired evidence')
                        prompts[action] = len(prompt_tokens(tokenizer, prompt)) - 1 + sample['feature_tokens']
                        if prompts[action] + config['generation_reserve'] > config['context_limit']:
                            raise ValueError('inference context overflow')
                        plan = dict(decoding='direct', q8_executed=False, evidence_used=ego['evidence_used'],
                                    remote_evidence_used=remote, evidence_selection=dict(evidence_format='compact'), q9_prompt=prompt)
                        if role == 'train':
                            example, = supervised_examples(role, row['sample_id'], action, row['ego_motion'], plan, label, sample['feature_path'])
                            batch = encode_supervision(example, tokenizer, sample['feature_tokens'])
                            expected_labels = tokenizer(label['target_q9'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
                            if batch['labels'][batch['labels'] != -100].tolist() != expected_labels:
                                raise ValueError('incorrect assistant-only supervision')
                            training.write(json.dumps(dict(example, source_run=str(directory))) + '\n')
                            count['training_examples'] += 1
                        else:
                            evaluation.write(json.dumps(dict(sample_id=row['sample_id'], action=action, g=row['g'],
                                feature_path=sample['feature_path'], prompt=prompt, source_run=str(directory))) + '\n')
                            count['validation_inputs'] += 1
                    remote_count = len(peer['evidence_used']['objects'])
                    record = dict(sample_id=row['sample_id'], role=role, scene=row['scene'], g=row['g'],
                        ego_full=len(ego['evidence_full']['objects']), ego_retained=len(ego['evidence_used']['objects']),
                        peer_full=len(peer['evidence_full']['objects']), peer_retained=remote_count,
                        ego_input_tokens=prompts['Ego'], F_input_tokens=prompts['F'],
                        request_bytes=costs['request_bytes'], response_bytes=costs['response_bytes'],
                        model_targets=sum(c['model_targets'] for c in sample['predictor_calls']))
                    coverage.append(record)
                    groups[role].append(record)
                    count['frames_verified'] += 1
                    count['no_peer_objects_returned'] += int(not peer['evidence_full']['objects'])
                    count['peer_returned_but_none_retained'] += int(bool(peer['evidence_full']['objects']) and not remote_count)
                    if count['frames_verified'] % 200 == 0:
                        print(json.dumps(dict(phase='offline_verify', **count)), flush=True)
    save(out / 'coverage.json', coverage)
    summaries = {role: dict(frames=len(values), **{
        key: dict(mean=float(np.mean([v[key] for v in values])), minimum=min(v[key] for v in values),
                  maximum=max(v[key] for v in values)) for key in
        ('ego_full', 'ego_retained', 'peer_full', 'peer_retained', 'ego_input_tokens', 'F_input_tokens', 'response_bytes')})
        for role, values in groups.items()}
    result = dict(status='completed', counts=dict(count), coverage=summaries, source_counts=config['counts'],
        checkpoint=config['mtr_checkpoint'], all_ego_feature_arrays_match_source=True,
        all_literal_ego_blocks_preserved=True, all_original_labels_recomputed=True,
        online_gt_labels_read=False, training_parameters_updated=False, driving_model_executed=False,
        scope='paired direct trajectory training samples and validation inputs, not driving quality or method effectiveness')
    save(out / 'preparation_result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('index', 'finalize'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    (initialize if args.stage == 'index' else finalize)(args.out.resolve())
