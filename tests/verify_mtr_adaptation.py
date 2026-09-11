"""Recompute archived predictions with independent formulas and inspect saved updates."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction import cmp_adapter as C
from prediction.train_mtr import load_examples


def key(row):
    return tuple(row[k] for k in ('context', 'scene', 'source', 't', 'track_id'))


def verify(run):
    config = json.loads((run / 'run_config.json').read_text())
    completion = json.loads((run / 'completion.json').read_text())
    examples = {(r['scene'], r['source'], r['t']): (r, w, y)
                for r, w, y in load_examples(Path(config['data']), 'validation')}
    result = dict(status='PASS', stages={}, test_access=False, metric_formula='independent float64 Euclidean distances')
    for stage in ('before', 'after'):
        rows = [json.loads(line) for line in (run / (stage + '.jsonl')).read_text().splitlines()]
        lookup = {key(r): r for r in rows}
        assert len(lookup) == len(rows)
        with (run / (stage + '_predictions.pkl')).open('rb') as handle:
            archived = pickle.load(handle)
        seen, maximum, values = set(), 0., {}
        for frame in archived:
            index = frame['index']
            frame_key = index['scene'], index['source'], index['t']
            original, window, labels = examples[frame_key]
            assert index == original
            positions = window['states'][:, -1, :2]
            roi = config['roi']
            keep = np.flatnonzero((positions[:, 0] >= roi[0]) & (positions[:, 1] >= roi[1]) &
                                 (positions[:, 0] <= roi[2]) & (positions[:, 1] <= roi[3]))
            for context, prediction, select in (
                    ('full', frame['full'], np.arange(len(positions))),
                    ('roi', frame['roi'], keep), ('full_on_roi', frame['full'], keep)):
                source_ids = {int(t): j for j, t in enumerate(prediction['track_ids'])}
                for i in select:
                    target = int(labels['track_ids'][i])
                    row_key = (context,) + frame_key + (target,)
                    row = lookup[row_key]
                    assert row_key not in seen
                    seen.add(row_key)
                    assert row['matched_gt_id'] == int(labels['matched_gt_ids'][i])
                    j = source_ids[target]
                    means = np.asarray(prediction['means'][j], dtype=np.float64)
                    score = np.asarray(prediction['scores'][j])
                    truth = np.asarray(labels['xy'][i], dtype=np.float64)
                    mask = labels['valid'][i]
                    top = int(np.argmax(score))
                    assert row['model_used'] == bool(window['valid'][i].sum() >= 2)
                    assert np.isfinite(means).all() and np.isfinite(score).all()
                    for horizon, length in ((3, 30), (5, 50)):
                        valid_steps = np.flatnonzero(mask[:length])
                        distance = np.sqrt(((means[:, valid_steps] - truth[None, valid_steps]) ** 2).sum(-1))
                        metrics = {'top1_ADE': float(distance[top].mean()) if len(valid_steps) else None,
                                   'minADE': float(distance.mean(1).min()) if len(valid_steps) else None}
                        end_distance = np.sqrt(((means[:, length - 1] - truth[length - 1]) ** 2).sum(-1))
                        metrics.update(top1_FDE=float(end_distance[top]) if mask[length - 1] else None,
                                       minFDE=float(end_distance.min()) if mask[length - 1] else None)
                        assert row['valid_points' + str(horizon)] == len(valid_steps)
                        for name, expected in metrics.items():
                            metric = name + str(horizon)
                            actual = row[metric]
                            assert (actual is None) == (expected is None)
                            if expected is not None:
                                maximum = max(maximum, abs(expected - actual))
                                assert abs(expected - actual) < 1e-4
                                values.setdefault((context, row['recording'], metric), []).append(expected)
        assert seen == set(lookup)
        summary = json.loads((run / (stage + '_summary.json')).read_text())
        for context in summary:
            for metric, actual in summary[context]['recording_macro'].items():
                group_means = [np.mean(v) for (c, _, m), v in values.items() if c == context and m == metric]
                assert abs(float(np.mean(group_means)) - actual) < 1e-4
        result['stages'][stage] = dict(source_frames=len(archived), target_rows=len(rows),
                                      max_metric_difference_m=maximum, grouping_recomputed=True)
    epochs = [json.loads(p.read_text()) for p in sorted(run.glob('epoch_*.json'))]
    assert len(epochs) == completion['completed_epochs']
    assert sum(r['counts']['source_frames'] for r in epochs) == completion['optimizer_steps']
    scores = [json.loads((run / 'before_summary.json').read_text())['full']['recording_macro']['top1_ADE5']]
    scores += [r['selection_metric'] for r in epochs]
    best_epoch = completion['best_epoch']
    assert abs(scores[best_epoch] - completion['best_metric']) < 1e-8
    assert scores[best_epoch] <= min(scores) + 1e-6
    if config.get('steps_per_epoch') is not None:
        for i, row in enumerate(epochs, 1):
            assert row['counts']['source_frames'] == config['steps_per_epoch']
            assert row['optimizer_steps'] == i * config['steps_per_epoch']
            expected = 'full' if config['context_schedule'] == 'full' or i % 2 else 'roi'
            assert row['context'] == expected
    model, loaded = C.load_model(run / 'best_model.pth')
    original = torch.load(config['source_model']['checkpoint'], mmap=True, map_location='cpu')['model_state']
    changed, total = 0, 0
    for name, parameter in model.named_parameters():
        total += 1
        changed += int(not torch.equal(parameter.detach(), original['motion_transformer.' + name]))
    assert changed > 0 if best_epoch else changed == 0
    assert loaded['checkpoint_metadata']['epoch'] == best_epoch
    assert loaded['checkpoint_metadata']['it'] == (epochs[best_epoch - 1]['optimizer_steps'] if best_epoch else 0)
    result['training'] = dict(optimizer_steps=completion['optimizer_steps'], epochs=len(epochs),
        supervised_center_presentations=sum(r['counts']['supervised_centers'] for r in epochs),
        best_epoch=completion['best_epoch'], saved_parameter_tensors=total, changed_parameter_tensors=changed,
        saved_checkpoint_metadata=loaded['checkpoint_metadata'])
    files = ('src/prediction/train_mtr.py', 'src/prediction/supervision.py', 'src/prediction/cmp_adapter.py',
             'src/evaluation/prediction.py', 'vendor/cmp_mtr/mtr/models_v2v4real/motion_decoder/mtr_decoder.py')
    result['running_code_matches_current'] = {f: (run / 'code' / f).read_bytes() == (ROOT / f).read_bytes() for f in files}
    assert all(result['running_code_matches_current'].values())
    return result


if __name__ == '__main__':
    run, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    result = verify(run)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result))
