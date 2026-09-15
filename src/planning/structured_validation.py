"""Frozen frame/receipt weighting and strict offline numeric validation metrics."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from evaluation.planning import GOT_PREFIX, trajectory_metrics
from tools.task_spec import validate_plan

CONDITIONS = ('Ego', 'P', 'F', 'PF')


def evidence_condition(row, task=None):
    """Canonical primitive receipt set, including receipts decoded from bundles.

    Callers verify the source episode before using its ledger. No archive label,
    control name, admission mask or receiver-derived forecast defines this key.
    """
    task = json.loads(Path(row['source_task']).read_text()) if task is None else task
    ep = task['episode']
    snapshot = ep['plans'][row['stage']]['ledger_snapshot']
    tools = {receipt['request']['tool'] for receipt in ep['ledger_snapshots'][snapshot]['receipts']}
    if not tools <= {'P', 'F'}:
        raise ValueError('non-primitive tool in validated numeric receipt ledger')
    return ''.join(tool for tool in ('P', 'F') if tool in tools) or 'Ego'


def frame_condition_weights(rows, role):
    """Mean-one weights frozen over eligible labeled rows, separately for each role."""
    groups = defaultdict(list)
    represented = defaultdict(set)
    for index, row in enumerate(rows):
        if row['role'] != role:
            continue
        frame = (row['scene'], row['g'])
        condition = evidence_condition(row)
        represented[frame].add(condition)
        if any(row['supervision']['valid']):
            groups[frame + (condition,)].append(index)
    condition_counts = Counter(key[:2] for key in groups)
    n = sum(len(indices) for indices in groups.values())
    frames = len(condition_counts)
    weights = [0.] * len(rows)
    details = []
    for (scene, g, condition), indices in sorted(groups.items()):
        total = n / frames / condition_counts[(scene, g)]
        for index in indices:
            weights[index] = total / len(indices)
        details.append(dict(scene=scene, g=g, condition=condition, rows=len(indices),
                            row_weight=total/len(indices), total_weight=total))
    return weights, dict(role=role, eligible_rows=n, eligible_frames=frames, groups=details,
        represented_frames=len(represented),
        missing_conditions=[dict(scene=scene, g=g, conditions=[c for c in CONDITIONS if c not in conditions])
                            for (scene, g), conditions in sorted(represented.items())],
        scope='Observed rows only; entirely absent frames require the collection coverage manifest. '
              'Zero-label rows have zero weight; eligible weights have dataset mean one. '
              'Fixed weights precede minibatch mean; shuffle and smaller last batches still affect optimization.')


def _summary(records):
    labeled = [r for r in records if any(r['label']['valid'])]
    valid = [r for r in labeled if r['valid_plan']]
    denominator = sum(r['weight'] for r in labeled)
    invalid = [r for r in labeled if not r['valid_plan']]
    result = dict(rows=len(records), labeled_rows=len(labeled), unlabeled_rows=len(records)-len(labeled),
        complete_label_rows=sum(all(r['label']['valid']) for r in labeled),
        valid_label_points=sum(sum(r['label']['valid']) for r in records),
        valid_plans=len(valid), invalid_plans=len(invalid),
        all_invalid_outputs=sum(not r['valid_plan'] for r in records),
        labeled_weight=denominator, valid_plan_weight=sum(r['weight'] for r in valid),
        invalid_plan_weight=sum(r['weight'] for r in invalid),
        weighted_invalid_plan_rate=sum(r['weight'] for r in invalid)/denominator if denominator else None,
        invalid_reasons=dict(Counter(r['invalid_reason'] for r in records if not r['valid_plan'])))
    for key in ('loss', 'ADE3', 'FDE3') + GOT_PREFIX:
        # Finite loss remains a diagnostic even when the plan violates dynamics.
        values = [(r['weight'], r[key]) for r in records
                  if r.get(key) is not None and math.isfinite(r[key]) and r['weight'] > 0]
        weight = sum(w for w, _ in values)
        result[key] = sum(w*v for w, v in values)/weight if weight else None
        result[key + '_rows'] = len(values)
        result[key + '_weight'] = weight
    return result


def summarize_measurements(records, execution_spec):
    """Strict plan failures retain labeled denominators; prefix metrics require labels."""
    measured = []
    for record in records:
        row = dict(record, valid_plan=True, invalid_reason=None)
        spec = row.get('execution_spec', execution_spec)
        try:
            validate_plan(row['waypoints'], spec)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            row.update(valid_plan=False, invalid_reason=str(exc))
        else:
            row.update(trajectory_metrics(row['waypoints'], row['label'], None))
        measured.append(row)
    selected = [r for r in measured if r['output_kind'] == 'selected_stage']
    groups = defaultdict(list)
    for row in measured:
        key = (row['stage'], row['condition'], row['output_kind'], row['refinement'])
        groups[key].append(row)
    return dict(selected_stage=_summary(selected),
        refinements=_summary([r for r in measured if r['output_kind'] == 'refinement']),
        groups=[dict(stage=stage, condition=condition, output_kind=kind, refinement=refinement,
                     metrics=_summary(group)) for (stage, condition, kind, refinement), group in sorted(groups.items())],
        scope='Selected-stage outputs determine selection; extra offline refinements are separate diagnostics. '
              'Incomplete horizons cannot enter complete got_prefix_L2_avg; invalid outputs retain denominators.')


def select_checkpoint(best, summary, batches, checkpoint):
    metrics = summary['selected_stage']
    rate, error = metrics['weighted_invalid_plan_rate'], metrics['got_prefix_L2_avg']
    if rate is None or error is None or not math.isfinite(rate) or not math.isfinite(error):
        return best
    key = [rate, error, batches]
    if best is None or key < best['key']:
        return dict(key=key, batches=batches, checkpoint=checkpoint)
    return best
