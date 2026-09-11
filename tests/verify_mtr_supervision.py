"""Recount all labels and check sampled futures directly with homogeneous poses."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import v2v4real_meta as M
from prediction.train_mtr import load_examples


def verify(data):
    preparation = json.loads((data / 'preparation.json').read_text())
    result = dict(roles={}, sampled_frames=0, sampled_future_frames=0, checked_valid_points=0,
                  maximum_coordinate_error_m=0., test_access=False)
    for role in ('train', 'validation'):
        examples = load_examples(data, role)
        totals, groups = Counter(), {}
        for row, window, labels in examples:
            totals.update(labels['status'].tolist())
            totals['online_targets'] += len(labels['track_ids'])
            totals['future_points'] += int(labels['valid'].sum())
            totals['supervised_model_targets'] += int(((window['valid'].sum(1) >= 2) & labels['valid'].any(1)).sum())
            assert np.all(labels['xy'][~labels['valid']] == 0.)
            assert len(set(labels['matched_gt_ids'][labels['matched_gt_ids'] >= 0].tolist())) == int((labels['matched_gt_ids'] >= 0).sum())
            groups.setdefault((row['scene'], row['source']), []).append((row, window, labels))
        expected = Counter()
        for row in preparation['scenes']:
            if row['role'] == role:
                expected.update(row['counts'])
        assert totals == expected, (role, totals, expected)
        result['roles'][role] = dict(source_frames=len(examples), recording_groups=len({r['recording'] for r, _, _ in examples}),
                                    scenes=len({r['scene'] for r, _, _ in examples}), counts=dict(totals))
        for frames in groups.values():
            for index in sorted({0, len(frames) // 2, len(frames) - 1}):
                row, window, labels = frames[index]
                g = row['g']
                pose = np.asarray(M.load_pose('train', g), float)
                inv = np.linalg.inv(pose)
                end = M.seq_of(g, 'train')[3]
                result['sampled_frames'] += 1
                current, ids = M.load_gt('train', g)
                current_by_id = {int(i): b for i, b in zip(ids, current)}
                for k, target in enumerate(labels['matched_gt_ids']):
                    if target >= 0:
                        distance = np.linalg.norm(window['states'][k, -1, :2] - current_by_id[int(target)][:2])
                        assert distance <= 2.00001
                        assert abs(distance - labels['current_match_distance'][k]) < 1e-4
                for offset in (1, 30, 50):
                    frame = g + offset
                    truth = {}
                    if frame < end:
                        boxes, ids = M.load_gt('train', frame)
                        world = np.asarray(M.load_pose('train', frame), float)
                        for box, target in zip(boxes, ids):
                            point = inv @ world @ np.array([box[0], box[1], box[2], 1.])
                            truth[int(target)] = point[:2]
                    result['sampled_future_frames'] += 1
                    for k, target in enumerate(labels['matched_gt_ids']):
                        expected_valid = target >= 0 and int(target) in truth
                        assert bool(labels['valid'][k, offset - 1]) == expected_valid
                        if expected_valid:
                            error = float(np.linalg.norm(labels['xy'][k, offset - 1] - truth[int(target)]))
                            assert error < 1e-4, (row, target, offset, error)
                            result['maximum_coordinate_error_m'] = max(result['maximum_coordinate_error_m'], error)
                            result['checked_valid_points'] += 1
    result['status'] = 'PASS'
    return result


if __name__ == '__main__':
    data = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    report = verify(data)
    with out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report))
