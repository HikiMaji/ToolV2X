"""Adapt original CMP MTR to causal tracks; labels and evaluation stay offline."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.audit_protocol import recording
from prediction import cmp_adapter as C
from prediction.supervision import PROTOCOL, check_labels, subset, training_batch
from evaluation.prediction import METRICS, paired, summarize, target_metrics

ROI = (0., -20., 70., 20.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))


def load_examples(data, role, stride=1):
    if role not in ('train', 'validation') or stride < 1:
        raise ValueError('only train/validation are available to adaptation')
    manifest = json.loads((data / 'split_manifest.json').read_text())
    groups = {r: {recording(s) for s in manifest[r + '_scenes']} for r in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('recording groups overlap')
    windows, labels, examples, seen = {}, {}, [], set()
    for line in (data / (role + '.jsonl')).read_text().splitlines():
        row = json.loads(line)
        key = row['scene'], row['source'], row['t']
        if (key in seen or row['role'] != role or row['scene'] not in manifest[role + '_scenes'] or
                row['recording'] != recording(row['scene']) or row['source'] not in ('no_fusion', 'no_fusion_cav1')):
            raise ValueError('invalid adaptation index')
        seen.add(key)
        if row['t'] % stride:
            continue
        wp, lp = row['window_path'], row['label_path']
        if wp not in windows:
            with open(wp, 'rb') as f:
                windows[wp] = pickle.load(f)
        if lp not in labels:
            with (data / lp).open('rb') as f:
                labels[lp] = pickle.load(f)
        wm, ym = windows[wp]['meta'], labels[lp]
        if (wm['scene'] != row['scene'] or wm['split'] != 'train' or wm['gt_access'] is not False or
                wm['coordinate_frame'] != 'ego_at_t' or wm['dt_seconds'] != .1 or wm['yaw_unit'] != 'radian' or
                ym['protocol'] != PROTOCOL or ym['role'] != role or ym['scene'] != row['scene']):
            raise ValueError('window or supervision provenance differs')
        window, target = windows[wp]['windows'][row['t']], ym['labels'][row['t']]
        check_labels(window, target)
        if window['source'] != row['source'] or window['g'] != row['g']:
            raise ValueError('source or time differs from index')
        examples.append((row, window, target))
    return examples


def metric_rows(index, window, labels, prediction, context):
    if not np.array_equal(window['track_ids'], prediction['track_ids']):
        raise ValueError('prediction target order differs')
    result = []
    for i, track in enumerate(window['track_ids']):
        history = int(window['valid'][i].sum())
        group = '1' if history < 2 else '2-5' if history <= 5 else '6-10' if history <= 10 else '11'
        distance = labels['current_match_distance'][i]
        row = {key: index[key] for key in ('scene', 'source', 't', 'g', 'recording')}
        row.update(context=context, track_id=int(track), matched_gt_id=int(labels['matched_gt_ids'][i]),
            label_status=str(labels['status'][i]), current_match_distance_m=float(distance) if np.isfinite(distance) else None,
            history_points=history, history_group=group, model_used=bool(prediction['model_used'][i]))
        row.update(target_metrics(prediction['means'][i], prediction['scores'][i], labels['xy'][i], labels['valid'][i]))
        result.append(row)
    return result


def evaluate(model, examples, batch_size, archive=None):
    model.eval()
    rows, saved = [], []
    for index, window, labels in examples:
        full = C.predict(model, window, batch_size=batch_size)
        all_rows = metric_rows(index, window, labels, full, 'full')
        rows.extend(all_rows)
        rw, ry = subset(window, labels, ROI)
        roi = C.predict(model, rw, batch_size=batch_size)
        ids = set(rw['track_ids'].tolist())
        rows.extend(dict(r, context='full_on_roi') for r in all_rows if r['track_id'] in ids)
        rows.extend(metric_rows(index, rw, ry, roi, 'roi'))
        if archive is not None:
            saved.append(dict(index=index, full=full, roi=roi))
    if archive is not None:
        with archive.open('wb') as f:
            pickle.dump(saved, f, protocol=4)
    return rows, summarize(rows)


def epoch_examples(examples, context, steps_per_epoch=None):
    """Keep empty-frame accounting while bounding actual optimizer updates."""
    if context not in ('full', 'roi') or (steps_per_epoch is not None and steps_per_epoch < 1):
        raise ValueError('invalid training context or update budget')
    prepared = []
    for index, window, labels in examples:
        if context == 'roi':
            window, labels = subset(window, labels, ROI)
        centers = np.flatnonzero((window['valid'].sum(axis=1) >= 2) & labels['valid'].any(axis=1))
        prepared.append((index, window, labels, centers))
    if not any(len(row[3]) for row in prepared):
        raise ValueError('empty supervised training pool for ' + context)
    updates = 0
    while True:
        for position in np.random.permutation(len(prepared)):
            row = prepared[int(position)]
            yield row
            updates += bool(len(row[3]))
            if steps_per_epoch is not None and updates == steps_per_epoch:
                return
        if steps_per_epoch is None:
            return


def run(args):
    out, data = args.out.resolve(), args.data.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'vendor/cmp_mtr', 'tests'):
        shutil.copytree(ROOT / name, out / 'code' / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('docs/mtr_adaptation_plan.md', 'docs/mtr_stability_plan.md'):
        (out / 'code' / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, out / 'code' / name)
    for name in ('preparation.json', 'split_manifest.json', 'train.jsonl', 'validation.jsonl'):
        shutil.copy2(data / name, out / name)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = load_examples(data, 'train', stride=5)
    validation = load_examples(data, 'validation')
    model, provenance = C.load_model(args.checkpoint, device=args.device)
    config = dict(started_at=datetime.now(timezone.utc).isoformat(), data=str(data), out=str(out),
        source_model=provenance, seed=seed, epochs=args.epochs, patience=args.patience,
        optimizer='original AdamW', learning_rate=1e-4, weight_decay=.01, grad_norm_clip=1000.,
        scheduler=dict(type='original lambdaLR', decay_steps=[k * len(train) for k in (22, 24, 26, 28)],
                       nominal_source_frames_per_epoch=len(train), decay=.5, minimum_lr=1e-6),
        train_source_frames=len(train), validation_source_frames=len(validation), train_stride=5,
        train_frame_batch_size=1, train_center_batch='all eligible centers in the source frame',
        eval_center_batch_size=args.eval_batch_size, roi=list(ROI), context_schedule=args.context,
        contexts='full every epoch' if args.context == 'full' else 'full on odd epochs, roi on even epochs',
        steps_per_epoch=args.steps_per_epoch,
        epoch_sampling='one shuffled traversal' if args.steps_per_epoch is None else
                       'repeat shuffled source-frame traversals until the fixed nonempty update budget',
        selection='full recording-macro top1_ADE5; validation only', training_loss='original decoder + dense prediction loss',
        device=args.device, torch_version=torch.__version__, numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        cudnn_deterministic=True, tf32=False, bitwise_reproducibility_claimed=False,
        native_CUDA_parity_measured=False, pretrained_provenance_verified=False, test_access=False)
    write_json(out / 'run_config.json', config)
    started = time.monotonic()
    before, before_summary = evaluate(model, validation, args.eval_batch_size, out / 'before_predictions.pkl')
    write_rows(out / 'before.jsonl', before)
    write_json(out / 'before_summary.json', before_summary)
    baseline = before_summary['full']['recording_macro']['top1_ADE5']
    if baseline is None:
        raise ValueError('no validation labels for checkpoint selection')
    print(json.dumps(dict(stage='baseline', selection_metric=baseline, elapsed_s=time.monotonic() - started)), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: max(.01, .5 ** sum(step >= k * len(train) for k in (22, 24, 26, 28))))
    best, best_epoch, stale, steps = baseline, 0, 0, 0
    best_rows = before
    first_update = None
    selected_weight = model.motion_decoder.motion_reg_heads[-1][-1].weight
    initial_weight = selected_weight.detach().clone()

    def save_checkpoint(name, epoch, score):
        torch.save(dict(model_state={'motion_transformer.' + k: v for k, v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), epoch=epoch, it=steps,
            version='ToolV2X causal MTR adaptation v1',
            validation_selection_metric=score, adaptation_config=config,
            random_state=random.getstate(), numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state(), cuda_random_state=torch.cuda.get_rng_state_all()), out / name)

    # The original initialization remains eligible if adaptation never improves validation.
    save_checkpoint('best_model.pth', 0, baseline)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        mode = 'full' if args.context == 'full' or epoch % 2 else 'roi'
        lr = optimizer.param_groups[0]['lr']
        count, loss_sum, gradient_max = Counter(), 0., 0.
        for index, window, labels, centers in epoch_examples(train, mode, args.steps_per_epoch):
            if not len(centers):
                count['skipped_empty_source_frames'] += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            _, loss, tb, _ = model(training_batch(window, labels, centers))
            value = float(loss.detach())
            if not np.isfinite(value):
                raise ValueError('nonfinite original MTR training loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1000., error_if_nonfinite=True))
            gradient_max = max(gradient_max, norm)
            if first_update is None:
                gradient_evidence = {}
                for name, module in (('context_encoder', model.context_encoder), ('motion_decoder', model.motion_decoder)):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    gradient_evidence[name] = dict(tensors_with_gradient=len(grads),
                        nonzero_tensors=sum(bool(g.abs().max() > 0) for g in grads),
                        all_finite=all(bool(torch.isfinite(g).all()) for g in grads))
                    if not gradient_evidence[name]['nonzero_tensors'] or not gradient_evidence[name]['all_finite']:
                        raise ValueError('missing or nonfinite original model gradients')
            optimizer.step()
            scheduler.step()
            steps += 1
            count['source_frames'] += 1
            count['supervised_centers'] += len(centers)
            loss_sum += value
            if first_update is None:
                change = float((selected_weight - initial_weight).detach().abs().max())
                if change == 0:
                    raise ValueError('original parameter did not update')
                first_update = dict(step=steps, native_loss=value, dense_loss=float(tb['loss_dense_prediction']),
                    gradient_norm_before_clip=norm, gradients=gradient_evidence,
                    motion_reg_head_max_parameter_change=change)
                write_json(out / 'first_update.json', first_update)
            if count['source_frames'] % 200 == 0:
                print(json.dumps(dict(stage='train', epoch=epoch, context=mode, steps=steps,
                    epoch_source_frames=count['source_frames'], mean_step_loss=loss_sum / count['source_frames'],
                    elapsed_s=time.monotonic() - epoch_start)), flush=True)
        if not count['source_frames']:
            raise ValueError('empty supervised training epoch')
        if args.steps_per_epoch is not None and count['source_frames'] != args.steps_per_epoch:
            raise ValueError('training did not meet the fixed update budget')
        rows, report = evaluate(model, validation, args.eval_batch_size)
        score = report['full']['recording_macro']['top1_ADE5']
        improved = score < best - 1e-6
        if improved:
            best, best_epoch, stale, best_rows = score, epoch, 0, rows
            save_checkpoint('best_model.pth', epoch, score)
            write_rows(out / 'best_validation.jsonl', rows)
        else:
            stale += 1
        row = dict(epoch=epoch, context=mode, optimizer_steps=steps, counts=dict(count), lr_at_start=lr,
            lr_at_end=optimizer.param_groups[0]['lr'],
            mean_step_loss=loss_sum / count['source_frames'], max_gradient_norm_before_clip=gradient_max,
            selection_metric=score, best_epoch=best_epoch, best_metric=best, stale_epochs=stale,
            elapsed_s=time.monotonic() - epoch_start, validation=report)
        write_json(out / ('epoch_%02d.json' % epoch), row)
        print(json.dumps({k: v for k, v in row.items() if k != 'validation'}), flush=True)
        if stale >= args.patience:
            break
    save_checkpoint('last_model.pth', epoch, score)
    last_change = float((selected_weight - initial_weight).detach().abs().max())
    del optimizer, scheduler, selected_weight, initial_weight, model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    best_model, loaded = C.load_model(out / 'best_model.pth', device=args.device)
    after, after_summary = evaluate(best_model, validation, args.eval_batch_size, out / 'after_predictions.pkl')
    reload_difference = max((abs(a[k] - b[k]) for a, b in zip(after, best_rows)
                             for k in METRICS if a[k] is not None), default=0.)
    paired(best_rows, after)  # Fail on any changed target/coverage after checkpoint reload.
    if reload_difference > 1e-5:
        raise ValueError('checkpoint reload changed validation metrics')
    write_rows(out / 'after.jsonl', after)
    write_json(out / 'after_summary.json', after_summary)
    write_json(out / 'paired.json', paired(before, after))
    completion = dict(status='completed', optimizer_steps=steps, completed_epochs=epoch, best_epoch=best_epoch,
        stopped_by_patience=stale >= args.patience, best_metric=best, original_metric=baseline,
        training_changed_parameters=last_change > 0, final_training_head_max_parameter_change=last_change,
        checkpoint_reload_max_metric_difference=reload_difference, checkpoint=loaded,
        elapsed_s=time.monotonic() - started, test_access=False,
        conclusion_scope='conditional native MTR adaptation; validation-selected checkpoint; no policy or closed-loop claim')
    write_json(out / 'completion.json', completion)
    print(json.dumps({k: v for k, v in completion.items() if k != 'checkpoint'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/mtr_supervision_v1')
    parser.add_argument('--checkpoint', type=Path, default=C.CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--context', choices=('alternating', 'full'), default='alternating')
    parser.add_argument('--steps-per-epoch', type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.eval_batch_size < 1:
        parser.error('epochs, patience and batch size must be positive')
    if not 0 <= args.seed < 2 ** 32 or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error('seed must be in [0, 2**32), and an explicit update budget must be positive')
    run(args)
