"""Bounded audit using existing CMP/tool/driver interfaces; no offline label reads."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
ACTIONS = ('Ego', 'P', 'F', 'PF')
FIELDS = ('sample_id', 'scene', 'recording', 'local_frame', 'g', 'ego_count')


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def select_frames(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['recording']].append({k: row[k] for k in FIELDS})
    selected = []
    for group, candidates in sorted(groups.items()):
        if len(candidates) < 8 or len({r['sample_id'] for r in candidates}) != len(candidates):
            raise ValueError('each recording needs at least eight unique candidates')
        temporal = sorted(candidates, key=lambda r: (r['g'], r['scene'], r['local_frame']))
        density = sorted(candidates, key=lambda r: (r['ego_count'], r['g'], r['scene'], r['local_frame']))
        used = set()
        for kind, ordered, quantiles in (('time', temporal, (.125, .375, .625, .875)),
                                        ('density', density, (0., 1 / 3, 2 / 3, 1.))):
            for q in quantiles:
                center = int(q * (len(ordered) - 1) + .5)
                index = min((i for i, r in enumerate(ordered) if r['sample_id'] not in used),
                            key=lambda i: (abs(i - center), i))
                row = dict(ordered[index], selection_reason='%s_quantile_%.6f' % (kind, q),
                           candidate_rank=index, group_candidates=len(ordered))
                used.add(row['sample_id'])
                selected.append(row)
    selected.sort(key=lambda r: (r['recording'], r['g'], r['scene'], r['local_frame']))
    return [dict(row, audit_index=i, directory='frames/%02d_g%d' % (i, row['g']))
            for i, row in enumerate(selected)]


def coverage(full, used):
    objects, retained = full['objects'], used['objects']
    peers = [r for r in objects if r['source'] != 'ego']
    kept = [r for r in retained if r['source'] != 'ego']
    peer_ids = {r['track_id'] for r in peers}
    kept_ids = {r['track_id'] for r in kept}
    related = {r['peer_id'] for r in full.get('relations', [])}
    roi_ids = {r['track_id'] for r in peers if 0 <= r['box'][0] <= 70 and -20 <= r['box'][1] <= 20}
    unassociated = peer_ids - related
    return dict(objects_total=len(objects), objects_retained=len(retained),
                ego_total=len(objects) - len(peers), ego_retained=len(retained) - len(kept),
                remote_records_total=len(peers), remote_records_retained=len(kept),
                remote_unique_total=len(peer_ids), remote_unique_retained=len(kept_ids),
                remote_unassociated_total=len(unassociated),
                remote_unassociated_retained=len(unassociated & kept_ids),
                roi_remote_unique_total=len(roi_ids), roi_remote_unique_retained=len(roi_ids & kept_ids),
                roi_unassociated_total=len(roi_ids & unassociated),
                roi_unassociated_retained=len(roi_ids & unassociated & kept_ids))


def create_manifest(out):
    from common.audit_protocol import recording
    rows = [json.loads(s) for s in (ROOT / 'outputs/adaptation_data_v1/online_index/train.jsonl').read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        if row['role'] != 'train' or row['physical_split'] != 'train':
            raise ValueError('unexpected research split')
        grouped[row['scene']].append(row)
    candidates = []
    for scene, scene_rows in sorted(grouped.items()):
        path = scene_rows[0]['window_archive_refs']['no_fusion']
        with open(path, 'rb') as handle:
            artifact = pickle.load(handle)
        meta = artifact['meta']
        if meta['gt_access'] is not False or meta['scene'] != scene or meta['coordinate_frame'] != 'ego_at_t':
            raise ValueError('invalid causal sampling source')
        for row in scene_rows:
            window = artifact['windows'][row['local_frame']]
            if window['g'] != row['g'] or window['source'] != 'no_fusion':
                raise ValueError('sampling frame/source mismatch')
            candidates.append(dict(sample_id=row['sample_id'], scene=scene, recording=recording(scene),
                local_frame=row['local_frame'], g=row['g'], ego_count=len(window['track_ids'])))
    selected = select_frames(candidates)
    if len(selected) != 64 or len({r['recording'] for r in selected}) != 8:
        raise ValueError('predeclared 64 frame / eight recording scope changed')
    frozen = json.loads((ROOT / 'outputs/mtr_stability_v1/frozen_model.json').read_text())
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), frames=selected,
        candidates=candidates, candidate_count=len(candidates), checkpoint=frozen['checkpoint'],
        selection_fields=list(FIELDS), selection_reads_future_labels=False,
        population_weighting='equal recording diagnostic sample, not natural frame frequency',
        actions=list(ACTIONS), evidence_format='compact', q8_max_new_tokens=128,
        q9_generation_reserve=256, context_limit=4096, training=False, q9_generation=False)
    save(out / 'manifest.json', manifest)
    return manifest


def prepare_frames(out):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    if (out / 'manifest.json').exists():
        raise FileExistsError('new prepare run requires a new directory')
    manifest = create_manifest(out)
    # The model loader's default argument is captured at import time.
    os.environ['TOOLV2X_MTR_CHECKPOINT'] = manifest['checkpoint']
    from planning.run_connection import prepare, snapshot_code, load_window
    from planning.v2vgot import CHECKPOINT, fit_evidence
    snapshot_code(out)
    (out / 'code_snapshot/scripts').mkdir()
    shutil.copy2(__file__, out / 'code_snapshot/scripts' / Path(__file__).name)
    shutil.copy2(ROOT / 'docs/training_evidence_audit_plan.md', out / 'audit_plan.md')
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    scenes = json.loads((ROOT / 'outputs/protocol_audit/split_manifest.json').read_text())['train_scenes']
    statuses = []
    for row in manifest['frames']:
        directory = out / row['directory']
        directory.mkdir(parents=True)
        try:
            features, metadata = prepare(directory, scenes.index(row['scene']), row['local_frame'],
                                         list(ACTIONS), 'compact', 'train')
            if metadata['g'] != row['g']:
                raise AssertionError('selected frame mismatch')
            loading = json.loads((directory / 'cmp_model_loading.json').read_text())
            if loading['checkpoint'] != manifest['checkpoint']:
                raise AssertionError('frozen checkpoint not used')
            for source, short in (('no_fusion', 'ego'), ('no_fusion_cav1', 'peer')):
                window = load_window(row['scene'], row['local_frame'], source, [])
                np.savez(str(directory / (short + '_window.npz')),
                         **{k: v for k, v in window.items() if isinstance(v, np.ndarray)})
            counts = {}
            for action in ACTIONS:
                base = directory / action
                full = json.loads((base / 'evidence_full.json').read_text())
                used = json.loads((base / 'evidence_used.json').read_text())
                json_view, selection = fit_evidence(tokenizer, metadata['ego_motion'], full, 540,
                                                    evidence_format='json')
                save(base / 'evidence_used_json.json', json_view)
                save(base / 'context_selection_json.json', selection)
                counts[action] = dict(compact=coverage(full, used), json=coverage(full, json_view))
            # Compare all original output arrays, beyond the six tool timestamps.
            calls = json.loads((directory / 'predictor_calls.json').read_text())
            remote_calls = [r for r in calls if r['source'] == 'no_fusion_cav1']
            arrays = [np.load(r['path']) for r in remote_calls]
            differences = {}
            try:
                for key in arrays[0].files:
                    ref = arrays[0][key]
                    if any(a[key].shape != ref.shape for a in arrays):
                        raise AssertionError('P/F array shape mismatch')
                    differences[key] = max(float(np.max(np.abs(a[key].astype(float) - ref.astype(float))))
                                           if ref.size else 0. for a in arrays)
                if any(v > 1e-5 for v in differences.values()):
                    raise AssertionError('P/F all-array equivalence failed')
            finally:
                for a in arrays:
                    a.close()
            save(directory / 'audit_capacity.json', counts)
            save(directory / 'audit_array_equivalence.json', dict(calls=len(remote_calls), differences=differences))
            save(directory / 'feature_summary.json', dict(frame_indices=features['frame_indices'],
                truncated_boxes=features['truncated_boxes'], feature_tokens=540, actual_rgb_input=False))
            status = dict(sample_id=row['sample_id'], status='completed')
        except Exception as exc:
            traceback.print_exc()
            status = dict(sample_id=row['sample_id'], status='failed', error_type=type(exc).__name__, error=str(exc))
        save(directory / 'audit_execution.json', status)
        statuses.append(status)
        save(out / 'prepare_status.json', dict(expected=64, attempted=len(statuses), frames=statuses,
            completed=sum(r['status'] == 'completed' for r in statuses)))
        print(json.dumps(dict(phase='prepare', index=row['audit_index'], **status)), flush=True)
    if any(r['status'] != 'completed' for r in statuses):
        raise RuntimeError('prepare failures preserved; do not substitute frames')


def generate_q8(out, initialization):
    import numpy as np
    import torch
    from planning.v2vgot import V2VGoTPlanner, CHECKPOINT, prompt_tokens
    from planning.inputs import make_prompt, parse_q8
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    manifest = json.loads((out / 'manifest.json').read_text())
    checkpoint = CHECKPOINT
    if initialization == 'llm':
        checkpoint = CHECKPOINT.parent.parent / ('llava-v1.5-7b-task-lora_v2v4real_3d_grounding_'
            'v2vllmq5_10ep_both_shallow_f2/checkpoint-490')
    directory = out / ('q8_' + initialization)
    directory.mkdir()
    planner = V2VGoTPlanner(checkpoint=checkpoint, evidence_format='compact',
                           adapter_directory=directory / 'llava-toolv2x-lora-ego')
    save(directory / 'model_loading.json', planner.provenance)
    reference = AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=False, local_files_only=True)
    results = []
    for row in manifest['frames']:
        frame = out / row['directory']
        if json.loads((frame / 'audit_execution.json').read_text())['status'] != 'completed':
            raise RuntimeError('prepare failure must be resolved before generation')
        metadata = json.loads((frame / 'connection.json').read_text())
        with np.load(str(frame / 'ego_features.npz')) as data:
            features = {k: data[k] for k in data.files}
        for action in ACTIONS:
            result = dict(sample_id=row['sample_id'], audit_index=row['audit_index'], action=action,
                initialization=initialization, q8_executed=False, q9_executed=False,
                training=False, offline_label_access=False)
            try:
                prompt = (frame / action / 'q8_prompt.txt').read_text().rstrip('\n')
                if not torch.equal(prompt_tokens(reference, prompt), prompt_tokens(planner.tokenizer, prompt)):
                    raise AssertionError('initialization tokenizers disagree on actual prompt')
                result['tokenizer_matches_reference'] = True
                raw, cost = planner._generate(features, prompt, 128)
                result.update(q8_executed=True, q8_raw=raw, q8_cost=cost)
                try:
                    result['parsed_q8'] = parse_q8(raw)
                except ValueError as exc:
                    result.update(status='invalid_q8', error=str(exc))
                else:
                    evidence = json.loads((frame / action / 'evidence_used.json').read_text())
                    q9 = make_prompt('Q9', metadata['ego_motion'], evidence, raw, evidence_format='compact')
                    count = len(prompt_tokens(planner.tokenizer, q9)) - 1 + 540
                    result.update(status='parsed_q8', q9_prompt=q9, q9_input_tokens=count,
                                  q9_budget_fits=count + 256 <= 4096)
            except Exception as exc:
                traceback.print_exc()
                result.update(status='execution_failed', error_type=type(exc).__name__, error=str(exc))
            save(directory / ('%02d_%s.json' % (row['audit_index'], action)), result)
            results.append(result)
            save(directory / 'status.json', dict(expected=256, attempted=len(results),
                executed=sum(r['q8_executed'] for r in results),
                parsed=sum(r['status'] == 'parsed_q8' for r in results),
                failures=sum(r['status'] == 'execution_failed' for r in results)))
            print(json.dumps(dict(phase='q8', initialization=initialization, index=row['audit_index'],
                                  action=action, status=result['status'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--phase', choices=('prepare', 'q8'), required=True)
    parser.add_argument('--initialization', choices=('got', 'llm'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        prepare_frames(args.out)
    else:
        if args.initialization is None:
            parser.error('q8 requires --initialization')
        generate_q8(args.out, args.initialization)


if __name__ == '__main__':
    main()
