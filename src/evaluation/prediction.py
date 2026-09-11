"""Target-level quality and coverage, independent of the original training logs."""
import numpy as np

METRICS = tuple(prefix + metric + str(h) for h in (3, 5)
                for prefix in ('top1_', 'min') for metric in ('ADE', 'FDE'))


def target_metrics(means, scores, truth, mask):
    means, scores, truth, mask = map(np.asarray, (means, scores, truth, mask))
    if (means.shape != (6, 50, 2) or scores.shape != (6,) or truth.shape != (50, 2) or
            mask.shape != (50,) or mask.dtype != bool or not np.isfinite(means).all() or
            not np.isfinite(scores).all() or not np.isfinite(truth[mask]).all() or (scores < 0).any()):
        raise ValueError('invalid prediction or target')
    error = np.zeros((6, 50))
    error[:, mask] = np.linalg.norm(means[:, mask] - truth[None, mask], axis=-1)
    top = int(scores.argmax())
    result = {}
    for seconds, end in ((3, 30), (5, 50)):
        count = int(mask[:end].sum())
        ade = error[:, :end].sum(axis=1) / count if count else None
        result.update({
            'valid_points' + str(seconds): count,
            'top1_ADE' + str(seconds): float(ade[top]) if count else None,
            'minADE' + str(seconds): float(ade.min()) if count else None,
            'top1_FDE' + str(seconds): float(error[top, end - 1]) if mask[end - 1] else None,
            'minFDE' + str(seconds): float(error[:, end - 1].min()) if mask[end - 1] else None})
    pair = np.linalg.norm(means[:, None, -1] - means[None, :, -1], axis=-1)
    result['max_mode_endpoint_distance_m'] = float(pair.max())
    return result


def aggregate(rows):
    rows = list(rows)
    result = dict(online_targets=len(rows), model_targets=sum(r['model_used'] for r in rows),
                  fallback_targets=sum(not r['model_used'] for r in rows),
                  matched_targets=sum(r['matched_gt_id'] >= 0 for r in rows))
    for key in METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key + '_targets'] = len(values)
    return result


def summarize(rows):
    result = {}
    for context in sorted({r['context'] for r in rows}):
        part = [r for r in rows if r['context'] == context]
        groups = {group: aggregate(r for r in part if r['recording'] == group)
                  for group in sorted({r['recording'] for r in part})}
        result[context] = dict(all=aggregate(part), recording_groups=groups,
            by_source={source: aggregate(r for r in part if r['source'] == source)
                       for source in sorted({r['source'] for r in part})},
            by_history={history: aggregate(r for r in part if r['history_group'] == history)
                        for history in sorted({r['history_group'] for r in part})},
            by_execution={name: aggregate(r for r in part if r['model_used'] == use)
                          for name, use in (('model', True), ('fallback', False))})
        result[context]['recording_macro'] = {key: float(np.mean([v[key] for v in groups.values() if v[key] is not None]))
            if any(v[key] is not None for v in groups.values()) else None for key in METRICS}
    return result


def row_key(row):
    return tuple(row[k] for k in ('context', 'scene', 'source', 't', 'track_id'))


def paired(before, after):
    left, right = ({row_key(r): r for r in rows} for rows in (before, after))
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError('paired target identities differ or are duplicated')
    for key, a in left.items():
        b = right[key]
        if any(a[k] != b[k] for k in ('recording', 'model_used', 'matched_gt_id')):
            raise ValueError('paired input or label identity differs')
        if any((a[k] is None) != (b[k] is None) for k in METRICS):
            raise ValueError('paired metric coverage differs')
    result = {}
    for context in sorted({r['context'] for r in before}):
        result[context] = {}
        for metric in METRICS:
            keys = [k for k in left if k[0] == context and left[k][metric] is not None]
            delta = np.array([right[k][metric] - left[k][metric] for k in keys])
            groups = sorted({left[k]['recording'] for k in keys})
            group_delta = {g: float(np.mean([right[k][metric] - left[k][metric]
                           for k in keys if left[k]['recording'] == g])) for g in groups}
            result[context][metric] = dict(targets=len(keys),
                after_minus_before_m=float(delta.mean()) if len(keys) else None,
                recording_macro_delta_m=float(np.mean(list(group_delta.values()))) if groups else None,
                recording_group_delta_m=group_delta,
                improved_fraction=float((delta < -1e-6).mean()) if len(keys) else None,
                worsened_fraction=float((delta > 1e-6).mean()) if len(keys) else None,
                unchanged_fraction=float((np.abs(delta) <= 1e-6).mean()) if len(keys) else None)
    return result
