"""Offline label join for completed causal tool episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning.adaptation_data import supervised_examples


def training_row(task, label, task_path):
    if (task['role'] != 'train' or label['role'] != 'train' or task['g'] != label['g'] or
            task['sample_id'] != label['sample_id'] or not all(label['valid']) or task['status'] != 'prepared'):
        raise ValueError('training label does not match this completed train episode')
    rows = supervised_examples('train', task['sample_id'], task['policy'], task['ego_motion'], task['prepared'],
                               label, task['feature_path'])
    if len(rows) != 1:
        raise ValueError('missing direct trajectory supervision')
    return dict(rows[0], role='train', scene=task['scene'], g=task['g'], source_run=task_path,
                input_layout=task['prepared']['input_layout'])


def export(prepared_root, out, data):
    from planning.run_framework import read_jsonl, completed_tasks
    from planning.run_connection import save_json
    progress = json.loads((prepared_root / 'progress.json').read_text())
    if progress['status'] != 'completed':
        raise ValueError('online episodes must finish before offline label joining')
    labels = {role: {r['sample_id']: r for r in read_jsonl(data/'offline_labels'/(role+'.jsonl'))}
              for role in ('train', 'validation')}
    indexes = {role: {r['sample_id']: r for r in read_jsonl(data/'online_index'/(role+'.jsonl'))}
               for role in ('train', 'validation')}
    rows, validation, val_loss, counts, seen = [], [], [], Counter(), set()
    for record in completed_tasks(prepared_root):
        task = json.loads(Path(record['path']).read_text())
        if task['status'] != 'prepared':
            raise ValueError('failed online task must be resolved before adaptation: '+record['path'])
        role, sample_id = task['role'], task['sample_id']
        key = (role, sample_id, task['policy'])
        if key in seen or sample_id not in indexes[role] or indexes[role][sample_id]['g'] != task['g']:
            raise ValueError('duplicate task or role/frame mismatch')
        seen.add(key)
        label = labels[role][sample_id]
        if label['g'] != task['g'] or label['role'] != role or not all(label['valid']):
            raise ValueError('offline labels do not match indexed episode')
        counts[role+'_'+task['policy']] += 1
        if role == 'train':
            rows.append(training_row(task, label, record['path']))
        else:
            validation.append(dict(record, prompt=task['prepared']['q9_prompt'], feature_path=task['feature_path']))
            # Kept separate from all inference inputs and never used by the optimizer.
            val_loss.append(dict(sample_id=sample_id, scene=task['scene'], role='validation', action=task['policy'],
                task='Trajectory', prompt=task['prepared']['q9_prompt'], target=label['target_q9'],
                feature_path=task['feature_path'], source_run=record['path'], input_layout='source_blocks_v1'))
    if len(seen) != progress['tasks']:
        raise ValueError('incomplete episode export')
    out.mkdir(parents=True, exist_ok=False)
    for name, values in (('train.jsonl', rows), ('validation_inputs.jsonl', validation), ('validation_loss.jsonl', val_loss)):
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    save_json(out/'manifest.json', dict(status='completed', prepared_root=str(prepared_root), label_data=str(data),
        counts=dict(counts), input_layout='source_blocks_v1', scope='driver supervision; no learned query targets',
        train_rows=len(rows), validation_rows=len(validation), validation_is_development=True,
        integration_only=json.loads((prepared_root/'config.json').read_text())['per_recording'] > 0))
    print(json.dumps(dict(status='completed', train_rows=len(rows), validation_rows=len(validation))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared_root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/paired_driving_data_v1')
    args = parser.parse_args()
    export(args.prepared_root.resolve(), args.out.resolve(), args.data.resolve())
