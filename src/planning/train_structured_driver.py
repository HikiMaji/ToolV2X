"""Explicit offline numeric supervision and resumable training; import is model-free."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import uuid

VERSION = 'toolv2x_structured_training_v1'
CHECKPOINT = 'toolv2x_structured_checkpoint_v1'


def _json(path):
    return json.loads(Path(path).read_text())


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _semantic(value):
    """Read paths are audit pointers, never semantic identity or network inputs."""
    if isinstance(value, dict):
        return {k: _semantic(v) for k, v in value.items() if k not in
                ('source_task', 'feature_path', 'artifact_root', 'reused_from') and
                not k.endswith('_paths') and not k.endswith('_path')}
    if isinstance(value, list):
        return [_semantic(v) for v in value]
    return value


def _label(label, row):
    import math
    if (label.get('role') != row['role'] or label.get('sample_id') != row['sample_id'] or
            label.get('g') != row['g'] or label.get('scene', row['scene']) != row['scene']):
        raise ValueError('offline label and task research identity disagree')
    if (not row['sample_id'].startswith(row['scene'] + ':') or
            not row['sample_id'][len(row['scene']) + 1:].isdigit()):
        raise ValueError('sample_id must establish an exact scene/frame join')
    if 'local_frame' in row and row['sample_id'] != row['scene'] + ':' + str(row['local_frame']):
        raise ValueError('sample_id local frame mismatch')
    if (label.get('times_seconds') != [.5, 1., 1.5, 2., 2.5, 3.] or
            not isinstance(label.get('label_read_paths'), list) or
            not label.get('scope', '').startswith('offline observed ego-trajectory imitation labels')):
        raise ValueError('independent ego_label supervision required')
    points, valid = label.get('waypoints'), label.get('valid')
    if not isinstance(points, list) or len(points) != 6 or not isinstance(valid, list) or len(valid) != 6:
        raise ValueError('six full precision points and masks required')
    for point, mask in zip(points, valid):
        if type(mask) is not bool or (not mask and point is not None):
            raise ValueError('missing label points must retain their explicit mask and null')
        if mask and (not isinstance(point, list) or len(point) != 2 or
                     any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError('valid label point must be finite')
    return {key: copy.deepcopy(label[key]) for key in
            ('waypoints', 'valid', 'times_seconds', 'label_read_paths', 'scope')}


def _task_rows(task, label, source):
    from common.audit_protocol import recording
    from planning.driver_contract import validate_numeric_episode
    row, ep = task['row'], task.get('episode')
    if row.get('physical_split') != 'train':
        raise ValueError('explicit physical_split=train is required')
    if row['role'] not in ('train', 'validation'):
        raise ValueError('only explicit train/validation recordings are eligible')
    supervision = _label(label, row)
    if task.get('status') not in ('completed', 'failed') or ep is None:
        raise ValueError('terminal numeric task with usable prepared input required')
    if ep.get('version') != 'toolv2x_episode_v2' or any(ep.get(k) != row[k] for k in ('sample_id', 'scene', 'g')):
        raise ValueError('task/episode identity mismatch')
    validate_numeric_episode(ep)
    if not ep['plans']:
        raise ValueError('numeric task has no usable prepared input')
    inputs = task['inputs']
    feature = Path(inputs['feature_path'])
    if not feature.is_absolute():
        feature = Path(inputs['artifact_root']) / feature
    if not feature.is_file():
        raise ValueError('missing shared numeric feature file')
    return [dict(version=VERSION, sample_id=row['sample_id'], scene=row['scene'], g=row['g'],
        role=row['role'], recording=recording(row['scene']), stage=i, source_task=str(source),
        inputs=dict(feature_path=str(feature), prefix=list(range(i + 1))),
        supervision=copy.deepcopy(supervision)) for i in range(len(ep['plans']))]


def prepare_training_rows(tasks, labels):
    """Read real tasks.jsonl references; keep one shared task/feature reference per stage."""
    labels = _jsonl(labels) if isinstance(labels, (str, Path)) else list(labels)
    label_index = {}
    for label in labels:
        key = label['sample_id']
        if key in label_index:
            raise ValueError('ambiguous duplicate offline label sample_id')
        label_index[key] = label
    if isinstance(tasks, (str, Path)):
        manifest = Path(tasks)
        if manifest.is_dir():
            manifest /= 'tasks.jsonl'
        def task_sources():
            for record in _jsonl(manifest):
                source = manifest.parent / record['path']
                task = _json(source)
                if record['sample_id'] != task['row']['sample_id']:
                    raise ValueError('manifest/task sample mismatch')
                yield task, source
        sources = task_sources()
    else:
        sources = [(task, Path(task['inputs']['artifact_root']) / 'task.json') for task in tasks]
    rows, roles, labels_by_frame = [], {}, {}
    for task, source in sources:
        key = task['row']['sample_id']
        if key not in label_index:
            raise ValueError('numeric task lacks independent offline label')
        exported = _task_rows(task, label_index[key], source)
        group, role = exported[0]['recording'], exported[0]['role']
        if group in roles and roles[group] != role:
            raise ValueError('recording train/validation leakage')
        roles[group] = role
        frame = (task['row']['scene'], task['row']['g'])
        supervision = exported[0]['supervision']
        if frame in labels_by_frame and labels_by_frame[frame] != supervision:
            raise ValueError('same source frame has conflicting reference trajectories')
        labels_by_frame[frame] = supervision
        if not Path(source).is_file() or _semantic(_json(source)) != _semantic(task):
            raise ValueError('task reference does not match actual archived task')
        _check_features(task, exported[0]['inputs']['feature_path'])
        rows.extend(exported)
    if not rows:
        raise ValueError('no usable numeric training rows')
    return rows


def masked_trajectory_loss(predicted, target, valid):
    """Mean SmoothL1 over valid coordinate pairs, or None when unsupervised."""
    import torch
    from torch.nn import functional as F
    if predicted.shape != target.shape or predicted.shape[-1] != 2 or valid.shape != predicted.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('invalid trajectory loss shape or mask')
    if not valid.any().item():
        return None
    if not torch.isfinite(predicted[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('nonfinite supervised trajectory')
    return F.smooth_l1_loss(predicted[valid], target[valid])


def _read_features(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        names = ('regression_map', 'classification_map', 'active_agent_mask',
                 'ego_pose_history', 'ego_pose_history_valid', 'ego_pose_history_times')
        return {key: saved[key] for key in names if key in saved.files}


def _check_features(task, path):
    from planning.structured_driver import collate_structured_inputs
    features = _read_features(path)
    batch = collate_structured_inputs([features], [task['episode']['plans'][0]['prepared']])
    if int(batch['scene_mask'].sum()) != task['episode']['numeric_scene_tokens']:
        raise ValueError('feature active mask differs from actual numeric episode')


def training_loss(model, features, row, *, refinement_depth=0, task=None):
    """Supervise each ordered evidence prefix and detached same-evidence refinement."""
    import torch
    from planning.structured_driver import collate_structured_inputs
    from tools.task_spec import validate_plan
    task = _json(row['source_task']) if task is None else task
    plans = task['episode']['plans']
    indices = row['inputs']['prefix']
    if indices != list(range(row['stage'] + 1)) or indices[-1] >= len(plans):
        raise ValueError('invalid ordered evidence prefix')
    target = torch.tensor([[p if ok else [0., 0.] for p, ok in
        zip(row['supervision']['waypoints'], row['supervision']['valid'])]],
        dtype=torch.float32, device=next(model.parameters()).device)
    valid = torch.tensor([row['supervision']['valid']], dtype=torch.bool, device=target.device)
    from planning.method_controls import uses_refinement
    previous, previous_batch, losses = None, None, []
    report = dict(supervised_forwards=0, invalid_prior_masks=0, invalid_prior_reasons={})
    for position, index in enumerate(indices + [indices[-1]] * refinement_depth):
        prepared = plans[index]['prepared']
        batch = collate_structured_inputs([features], [prepared], device=target.device)
        # Retain the validated archive's dependency closure, but replace its numeric
        # prior before any model forward. No archived prediction reaches the network.
        batch['previous_plan'].zero_()
        batch['previous_plan_valid'].fill_(False)
        exact_repeat = (0 < position < len(indices) and
            plans[index]['ledger_snapshot'] == plans[index - 1]['ledger_snapshot'] and
            not uses_refinement(task['episode'].get('control_spec')))
        if exact_repeat:
            batch['previous_plan'] = previous_batch['previous_plan']
            batch['previous_plan_valid'] = previous_batch['previous_plan_valid']
        elif previous is not None:
            try:
                validate_plan(previous[0].cpu().tolist(), task['episode']['limits']['execution_spec'])
            except ValueError as exc:
                report['invalid_prior_masks'] += 1
                reason = str(exc)
                report['invalid_prior_reasons'][reason] = report['invalid_prior_reasons'].get(reason, 0) + 1
            else:
                batch['previous_plan'] = previous
                batch['previous_plan_valid'].fill_(True)
        predicted = model(batch)
        loss = masked_trajectory_loss(predicted, target, valid)
        if loss is not None:
            losses.append(loss)
            report['supervised_forwards'] += 1
        previous = predicted.detach()
        previous_batch = batch
    return (torch.stack(losses).mean() if losses else None), report


def _atomic_torch(path, value):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses concurrent overwrite.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def _seed(seed):
    import numpy as np
    import torch
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _version(seed, steps, training_run_id, config=None, data=None):
    return dict(name='structured_planner_network', revision='v1', training=dict(
        status='trained' if steps else 'initialized_untrained', optimizer_steps=steps, seed=seed,
        training_run_id=training_run_id,
        config=config, data=data))


def _same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _state(model, version, **extra):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return dict(checkpoint_version=CHECKPOINT, driver_spec=model.spec.to_dict(), model_state=state,
                model_version=version, binding=dict(model_state=copy.deepcopy(state),
                driver_spec=model.spec.to_dict(), model_version=copy.deepcopy(version),
                continuation=copy.deepcopy(extra)), **extra)


def initialize(checkpoint, driver_spec, *, seed=0):
    """Explicitly create untrained seeded weights; never collect tasks implicitly."""
    from planning.structured_inputs import StructuredDriverSpec
    from planning.structured_driver import StructuredPlannerNetwork
    if Path(checkpoint).exists():
        raise FileExistsError(checkpoint)
    _seed(seed)
    model = StructuredPlannerNetwork(StructuredDriverSpec.from_dict(driver_spec))
    _atomic_torch(checkpoint, _state(model, _version(seed, 0, str(uuid.uuid4())), rng=_rng()))
    return Path(checkpoint)


def _load(checkpoint):
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('checkpoint_version') != CHECKPOINT or
            saved.get('binding', {}).get('driver_spec') != saved.get('driver_spec') or
            saved['binding'].get('model_version') != saved.get('model_version')):
        raise ValueError('structured checkpoint binding mismatch')
    actual, bound = saved['model_state'], saved['binding']['model_state']
    if actual.keys() != bound.keys() or any(not torch.equal(actual[k], bound[k]) or
            not torch.isfinite(actual[k]).all() for k in actual):
        raise ValueError('structured checkpoint actual parameter binding mismatch')
    continuation = {k: v for k, v in saved.items() if k not in
                    ('checkpoint_version', 'driver_spec', 'model_state', 'model_version', 'binding')}
    if not _same(continuation, saved['binding'].get('continuation')):
        raise ValueError('checkpoint continuation state binding mismatch')
    if 'progress' in saved and (saved['progress']['optimizer_steps'] !=
            saved['model_version']['training']['optimizer_steps'] or
            saved['config'] != saved['model_version']['training']['config']):
        raise ValueError('checkpoint actual training metadata mismatch')
    return saved


def load_structured_planner(checkpoint, *, device='cpu'):
    from planning.structured_driver import StructuredPlanner, StructuredPlannerNetwork
    from planning.structured_inputs import StructuredDriverSpec
    saved = _load(checkpoint)
    spec = StructuredDriverSpec.from_dict(saved['driver_spec'])
    model = StructuredPlannerNetwork(spec)
    model.load_state_dict(saved['model_state'], strict=True)
    model.eval()
    return StructuredPlanner(spec, model=model, device=device, model_version=saved['model_version'])


def _config(config):
    from planning.structured_inputs import StructuredDriverSpec
    expected = {'version', 'driver_spec', 'seed', 'optimizer', 'batch_size', 'epochs',
                'refinement_depth', 'save_interval', 'device'}
    if set(config) != expected or config['version'] != VERSION:
        raise ValueError('complete versioned structured training config required')
    StructuredDriverSpec.from_dict(config['driver_spec'])
    for key in ('batch_size', 'epochs', 'save_interval', 'refinement_depth'):
        if type(config[key]) is not int or config[key] < (0 if key == 'refinement_depth' else 1):
            raise ValueError('invalid training integer: ' + key)
    opt = config['optimizer']
    import math
    if (set(opt) != {'name', 'lr', 'weight_decay'} or opt['name'] != 'AdamW' or
            any(type(opt[k]) not in (int, float) or not math.isfinite(opt[k]) for k in ('lr', 'weight_decay')) or
            opt['lr'] <= 0 or opt['weight_decay'] < 0):
        raise ValueError('explicit finite AdamW settings required')
    return copy.deepcopy(config)


def _verify_rows(rows):
    roles, frames, seen, checked_features = {}, {}, set(), set()
    for row in rows:
        if row.get('version') != VERSION:
            raise ValueError('old or invalid training row')
        task = _json(row['source_task'])
        label = dict(row['supervision'], sample_id=row['sample_id'], scene=row['scene'],
                     g=row['g'], role=row['role'])
        expected = _task_rows(task, label, row['source_task'])
        if row not in expected:
            raise ValueError('training row differs from source task')
        group, role = row['recording'], row['role']
        if group in roles and roles[group] != role:
            raise ValueError('recording train/validation leakage')
        roles[group] = role
        frame = (row['scene'], row['g'])
        if frame in frames and frames[frame] != row['supervision']:
            raise ValueError('conflicting same-frame targets')
        frames[frame] = row['supervision']
        key = (row['source_task'], row['stage'])
        if key in seen:
            raise ValueError('duplicate stage row')
        seen.add(key)
        feature_key = (row['source_task'], row['inputs']['feature_path'])
        if feature_key not in checked_features:
            _check_features(task, row['inputs']['feature_path'])
            checked_features.add(feature_key)
    if not any(row['role'] == 'train' for row in rows):
        raise ValueError('no training recordings')
    return roles


def _snapshot_inputs(rows, out, resume):
    """One file at a time; shared archives and feature arrays are never corpus-cached."""
    import numpy as np
    sources, features = [], []
    for row in rows:
        if row['source_task'] not in sources:
            sources.append(row['source_task'])
        if row['inputs']['feature_path'] not in features:
            features.append(row['inputs']['feature_path'])
    for index, source in enumerate(sources):
        current = _json(source)
        name = 'task_%06d.json' % index
        if resume is not None and _semantic(current) != _semantic(_json(Path(resume).parent / 'inputs' / name)):
            raise ValueError('resume source task content changed')
        (out / name).write_text(json.dumps(current, allow_nan=False))
    for index, source in enumerate(features):
        current = _read_features(source)
        name = 'feature_%06d.npz' % index
        if resume is not None:
            previous = _read_features(Path(resume).parent / 'inputs' / name)
            if current.keys() != previous.keys() or any(current[k].dtype != previous[k].dtype or
                    not np.array_equal(current[k], previous[k]) for k in current):
                raise ValueError('resume feature content changed')
        np.savez(out / name, **current)
    return dict(tasks=len(sources), features=len(features),
                row_sources=[sources.index(r['source_task']) for r in rows],
                row_features=[features.index(r['inputs']['feature_path']) for r in rows])


def fit(rows, out, config, *, resume=None, stop_after_steps=None):
    """Train explicit rows, saving initial and bounded-step checkpoints to a new directory."""
    import torch
    from planning.structured_driver import StructuredPlannerNetwork
    from planning.structured_inputs import StructuredDriverSpec
    rows = _jsonl(rows) if isinstance(rows, (str, Path)) else list(rows)
    config = _config(config)
    roles = _verify_rows(rows)
    if any(_json(row['source_task'])['episode']['limits']['receiver_spec'] != config['driver_spec'] for row in rows):
        raise ValueError('training architecture differs from collected numeric spec')
    binding = dict(rows=_semantic(rows), recordings=roles)
    saved = _load(resume) if resume is not None else None
    if saved is not None and (saved.get('config') != config or saved.get('data_binding') != binding):
        raise ValueError('resume model/data/optimizer configuration mismatch')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    layout = _snapshot_inputs(rows, out / 'inputs', resume)
    if saved is not None and saved['input_layout'] != layout:
        raise ValueError('resume shared input layout changed')
    _seed(config['seed'])
    model = StructuredPlannerNetwork(StructuredDriverSpec.from_dict(config['driver_spec'])).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    progress = dict(epoch=0, optimizer_steps=0, row_position=0, permutation=[], batches=0)
    report = dict(supervised_forwards=0, invalid_prior_masks=0, invalid_prior_reasons={}, unsupervised_rows=0,
                  train_rows=sum(r['role'] == 'train' for r in rows), validation_rows=sum(r['role'] == 'validation' for r in rows),
                  evidence_stages={})
    for row in rows:
        task = _json(row['source_task'])
        snapshot = task['episode']['plans'][row['stage']]['ledger_snapshot']
        receipts = task['episode']['ledger_snapshots'][snapshot]['receipts']
        tools = ''.join(receipt['request']['tool'] for receipt in receipts) or 'Ego'
        report['evidence_stages'][tools] = report['evidence_stages'].get(tools, 0) + 1
    if saved is not None:
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        progress = saved['progress']; report = saved['report']; _restore_rng(saved['rng'])
    training_run_id = saved['model_version']['training']['training_run_id'] if saved is not None else str(uuid.uuid4())
    train_indices = [i for i, row in enumerate(rows) if row['role'] == 'train']
    last = None
    def save():
        nonlocal last
        version = _version(config['seed'], progress['optimizer_steps'], training_run_id, config,
                           dict(recordings=roles, rows=[{k:r[k] for k in ('sample_id','role','stage')} for r in rows]))
        last = out / ('checkpoint_%06d.pt' % progress['batches'])
        _atomic_torch(last, _state(model, version, config=config, data_binding=binding,
            input_layout=layout, optimizer_state=optimizer.state_dict(), progress=copy.deepcopy(progress),
            rng=_rng(), report=copy.deepcopy(report)))
    save()
    while progress['epoch'] < config['epochs']:
        if not progress['permutation']:
            progress['permutation'] = random.sample(train_indices, len(train_indices))
            progress['row_position'] = 0
        selected = progress['permutation'][progress['row_position']:progress['row_position'] + config['batch_size']]
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in selected:
            row = rows[index]
            features = _read_features(out / 'inputs' / ('feature_%06d.npz' % layout['row_features'][index]))
            task = _json(out / 'inputs' / ('task_%06d.json' % layout['row_sources'][index]))
            loss, detail = training_loss(model, features, row, task=task, refinement_depth=config['refinement_depth'])
            for key in ('supervised_forwards', 'invalid_prior_masks'):
                report[key] += detail[key]
            for reason, count in detail['invalid_prior_reasons'].items():
                report['invalid_prior_reasons'][reason] = report['invalid_prior_reasons'].get(reason, 0) + count
            if loss is None:report['unsupervised_rows'] += 1
            else:losses.append(loss)
        if losses:
            torch.stack(losses).mean().backward(); optimizer.step()
            progress['optimizer_steps'] += 1
        progress['batches'] += 1; progress['row_position'] += len(selected)
        if progress['row_position'] == len(progress['permutation']):
            progress['epoch'] += 1; progress['permutation'] = []; progress['row_position'] = 0
        stopped = stop_after_steps is not None and progress['optimizer_steps'] >= stop_after_steps
        if progress['batches'] % config['save_interval'] == 0 or stopped or progress['epoch'] == config['epochs']:
            save()
        if stopped:break
    # Validation uses held-out rows only and never updates optimizer state.
    model.eval(); validation = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            if row['role'] == 'validation':
                features = _read_features(out / 'inputs' / ('feature_%06d.npz' % layout['row_features'][index]))
                task = _json(out / 'inputs' / ('task_%06d.json' % layout['row_sources'][index]))
                loss, _ = training_loss(model, features, row, task=task,
                                        refinement_depth=config['refinement_depth'])
                if loss is not None:validation.append(loss.item())
    (out / 'report.json').write_text(json.dumps(dict(report, progress=progress,
        validation_loss=sum(validation)/len(validation) if validation else None,
        scope='offline trajectory imitation; coverage is observed, no method benefit established'), indent=2))
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('initialize', help='explicit seeded untrained checkpoint')
    init.add_argument('checkpoint', type=Path); init.add_argument('--spec', required=True, type=Path)
    init.add_argument('--seed', type=int, default=0)
    prepare = commands.add_parser('prepare', help='actual numeric tasks plus independent ego_label JSONL')
    prepare.add_argument('--tasks', required=True, type=Path); prepare.add_argument('--labels', required=True, type=Path)
    prepare.add_argument('--out', required=True, type=Path)
    train = commands.add_parser('train', aliases=['fit'], help='explicit offline training in a new output directory')
    train.add_argument('--rows', required=True, type=Path); train.add_argument('--config', required=True, type=Path)
    train.add_argument('--out', required=True, type=Path); train.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.command == 'initialize':initialize(args.checkpoint, _json(args.spec), seed=args.seed)
    elif args.command == 'prepare':
        rows = prepare_training_rows(args.tasks, args.labels)
        with args.out.open('x') as handle:
            for row in rows:handle.write(json.dumps(row, allow_nan=False) + '\n')
    else:fit(args.rows, args.out, _json(args.config), resume=args.resume)


if __name__ == '__main__':
    main()
