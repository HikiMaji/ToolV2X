"""Summarize all six archived runs and apply the rule fixed before training."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.prediction import paired


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def identity(row):
    return tuple(row[k] for k in ('context', 'scene', 'source', 't', 'track_id', 'matched_gt_id'))


records, curves, baseline_keys, first_round_checks = [], {}, None, {}
for seed in (20, 21, 22):
    for context in ('full', 'alternating'):
        name = '%s_seed%d' % (context, seed)
        run = OUT / name
        config, completion = read(run / 'run_config.json'), read(run / 'completion.json')
        audit = read(OUT / (name + '_audit.json'))
        assert audit['status'] == 'PASS' and all(audit['running_code_matches_current'].values())
        assert completion['status'] == 'completed' and completion['optimizer_steps'] == 9960
        assert completion['completed_epochs'] == 10 and not completion['stopped_by_patience']
        assert config['seed'] == seed and config['context_schedule'] == context
        assert config['steps_per_epoch'] == 996 and config['train_source_frames'] == 1220
        assert config['validation_source_frames'] == 216 and not config['test_access']
        epochs = [read(run / ('epoch_%02d.json' % n)) for n in range(1, 11)]
        curve = [r['selection_metric'] for r in epochs]
        for n, row in enumerate(epochs, 1):
            assert row['optimizer_steps'] == n * 996 and row['counts']['source_frames'] == 996
            expected = 'full' if context == 'full' or n % 2 else 'roi'
            assert row['context'] == expected
            group_values = [v['top1_ADE5'] for v in row['validation']['full']['recording_groups'].values()]
            assert abs(float(np.mean(group_values)) - row['selection_metric']) < 1e-8
        before, after = read(run / 'before_summary.json'), read(run / 'after_summary.json')
        baseline = before['full']['recording_macro']['top1_ADE5']
        keys = {identity(r) for r in rows(run / 'before.jsonl')}
        if baseline_keys is None:
            baseline_keys = keys
        assert keys == baseline_keys and keys == {identity(r) for r in rows(run / 'after.jsonl')}
        assert abs(baseline - completion['original_metric']) < 1e-8
        record = dict(name=name, seed=seed, context=context, optimizer_steps=9960,
            original_ADE5=baseline, final_ADE5=curve[-1], late_three_mean_ADE5=float(np.mean(curve[-3:])),
            late_three_range_ADE5=float(np.ptp(curve[-3:])),
            best_epoch=completion['best_epoch'], best_ADE5=completion['best_metric'],
            best_full_on_roi_ADE5=after['full_on_roi']['recording_macro']['top1_ADE5'],
            best_roi_ADE5=after['roi']['recording_macro']['top1_ADE5'],
            supervised_center_presentations=sum(r['counts']['supervised_centers'] for r in epochs),
            skipped_empty_frames=sum(r['counts'].get('skipped_empty_source_frames', 0) for r in epochs),
            elapsed_s=completion['elapsed_s'], checkpoint_reload_max_metric_difference=completion['checkpoint_reload_max_metric_difference'])
        records.append(record)
        curves[name] = [baseline] + curve
    first_full = read(OUT / ('full_seed%d' % seed) / 'epoch_01.json')
    first_alternating = read(OUT / ('alternating_seed%d' % seed) / 'epoch_01.json')
    first_round_checks[str(seed)] = dict(
        counts_equal=first_full['counts'] == first_alternating['counts'],
        validation_difference=abs(first_full['selection_metric'] - first_alternating['selection_metric']),
        mean_loss_difference=abs(first_full['mean_step_loss'] - first_alternating['mean_step_loss']))
    assert first_round_checks[str(seed)]['counts_equal']
    assert first_round_checks[str(seed)]['validation_difference'] < 1e-6
    assert first_round_checks[str(seed)]['mean_loss_difference'] < 1e-6
    comparison = paired(rows(OUT / ('full_seed%d' % seed) / 'after.jsonl'),
                        rows(OUT / ('alternating_seed%d' % seed) / 'after.jsonl'))
    (OUT / ('paired_recipes_seed%d.json' % seed)).write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')

families = {}
for context in ('full', 'alternating'):
    group = [r for r in records if r['context'] == context]
    families[context] = dict(eligible=all(r['late_three_mean_ADE5'] < r['original_ADE5'] for r in group),
        late_three_seed_mean=float(np.mean([r['late_three_mean_ADE5'] for r in group])),
        late_three_seed_range=[min(r['late_three_mean_ADE5'] for r in group), max(r['late_three_mean_ADE5'] for r in group)],
        final_seed_mean=float(np.mean([r['final_ADE5'] for r in group])),
        best_seed_mean=float(np.mean([r['best_ADE5'] for r in group])))
eligible = [k for k, v in families.items() if v['eligible']]
selected = min(eligible, key=lambda k: families[k]['late_three_seed_mean']) if eligible else None
if len(eligible) == 2 and abs(families['full']['late_three_seed_mean'] - families['alternating']['late_three_seed_mean']) < .1:
    selected = 'full'
chosen = next((r for r in records if r['context'] == selected and r['seed'] == 20), None)
frozen = dict(status='selected' if chosen else 'no_eligible_recipe', context=selected, seed=20 if chosen else None,
    checkpoint=str(OUT / chosen['name'] / 'best_model.pth') if chosen else None,
    best_epoch=chosen['best_epoch'] if chosen else None,
    selection_protocol=str(ROOT / 'docs/mtr_stability_plan.md'),
    test_access=False, driving_model_evaluated=False, validation_used_for_selection=True)
(OUT / 'frozen_model.json').write_text(json.dumps(frozen, indent=2, allow_nan=False) + '\n')
summary = dict(status='completed', runs=records, families=families, frozen=frozen,
    identical_first_round_controls=first_round_checks,
    optimizer_steps=sum(r['optimizer_steps'] for r in records), validation_identity_rows=len(baseline_keys),
    all_update_budgets_equal=True, all_validation_target_identities_equal=True,
    test_access=False, claim='development stability comparison of training recipes; not independent test or driving utility')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, seed in zip(axes, (20, 21, 22)):
    for context, color in (('full', '#2267a8'), ('alternating', '#cc6d32')):
        name = '%s_seed%d' % (context, seed)
        ax.plot(np.arange(11) * 996, curves[name], marker='o', ms=3, color=color, label=context)
        record = next(r for r in records if r['name'] == name)
        ax.scatter(record['best_epoch'] * 996, record['best_ADE5'], marker='*', s=110, color=color, zorder=3)
    ax.axvspan(8 * 996, 10 * 996, color='grey', alpha=.08)
    ax.set(title='Seed %d' % seed, xlabel='Optimizer updates', xticks=[0, 4980, 9960])
    ax.grid(alpha=.2)
axes[0].set_ylabel('Recording-macro top-1 ADE up to 5 s (m)')
axes[0].legend(frameon=False)
fig.suptitle('Same update budgets; two validation recording groups; stars: validation-selected checkpoints', fontsize=10)
fig.tight_layout()
figure = ROOT / 'docs/figures/mtr_stability_v1.png'
fig.savefig(figure, dpi=170)
plt.close(fig)

lines = ['# MTR 上下文训练稳定性：六次等预算实验', '',
    '本报告来自六次实际原 CMP MTR 训练，每次 10 轮 × 996 次更新，共 59,760 次更新。'
    '训练方式与种子在运行前固定，见 [协议](mtr_stability_plan.md)。原始实验保留。', '',
    '所有运行使用同一原初始化；其完整上下文验证 top-1 ADE5 为 %.3f 米。' % records[0]['original_ADE5'], '',
    '## 结果', '',
    '单位为米。完整上下文 top-1 ADE5 先在各录制组内平均，再对两个验证录制组等权平均。'
    '末三轮均值用于选择训练方案；最佳检查点仍按单次验证指标选取。', '',
    '| 方案 | 种子 | 第 10 轮 | 第 8–10 轮均值 | 最佳 ADE5 | 最佳轮 |',
    '|---|---:|---:|---:|---:|---:|']
for r in records:
    lines.append('| %s | %d | %.3f | %.3f | %.3f | %d |' %
        (r['context'], r['seed'], r['final_ADE5'], r['late_three_mean_ADE5'], r['best_ADE5'], r['best_epoch']))
lines += ['', '| 方案 | 末三轮均值的跨种子平均 | 各种子范围 | 全部种子优于原初始化 |', '|---|---:|---|---|']
for name, family in families.items():
    low, high = family['late_three_seed_range']
    lines.append('| %s | %.3f | %.3f–%.3f | %s |' % (name, family['late_three_seed_mean'], low, high, family['eligible']))
lines += ['', '![三种子的完整训练曲线](figures/mtr_stability_v1.png)', '', '## 冻结选择', '']
if chosen:
    lines += ['按预先固定规则选择 `%s`，下游使用预先指定的 seed=20、第 %d 轮验证最佳检查点：' % (selected, chosen['best_epoch']),
              '', '`%s`' % frozen['checkpoint'], '',
              '冻结记录：`outputs/mtr_stability_v1/frozen_model.json`。此选择使用验证集，不代表独立测试结论。']
else:
    lines += ['两种方案均未满足全部种子末三轮均值优于初始化的门槛，本次未冻结新权重。原结果和全部失败证据保留。']
lines += ['', '## 相同 ROI 目标与实际训练量', '',
    '下表使用各次运行的验证最佳检查点。完整来源上下文后取 ROI 目标，与仅 ROI 历史的输入条件不同；不得将二者混称同信息对照。', '',
    '| 方案 / 种子 | 完整上下文的 ROI 目标 ADE5 | ROI 内历史 ADE5 | 监督中心呈现数 |', '|---|---:|---:|---:|']
for r in records:
    lines.append('| %s / %d | %.3f | %.3f | %d |' % (r['context'], r['seed'], r['best_full_on_roi_ADE5'], r['best_roi_ADE5'], r['supervised_center_presentations']))
lines += ['', '## 复核与限制', '',
    '- 六次原网络训练均保存真实参数更新、best/last 检查点、逐轮报表和选中权重重载结果；各自独立审计在 `outputs/mtr_stability_v1/*_audit.json`。',
    '- 每个种子的两方案第 1 轮均为相同完整上下文；另核对该轮监督计数、平均训练 loss 及验证值一致，差异从第 2 轮开始引入。',
    '- 独立 float64 公式复算原始与所选检查点的预测，核对身份、标签覆盖、录制组、更新预算和检查点元数据。末三轮均值另从逐轮报表的录制组指标复算；未将其表述为每轮原始预测都已独立重算。',
    '- 每次验证均为相同 216 个来源窗口，full/full_on_roi/roi 合计 7,256 条目标/上下文记录，不能当成独立样本数。',
    '- 两方案的更新次数和验证机会相同；监督中心数、ROI 采样重复与实际计算量不同。本次比较训练方案，不能单独识别上下文、中心分布、BatchNorm 或优化噪声的因果作用。',
    '- 部分运行与独立驾驶接入或依赖构建并行，墙钟耗时只是过程记录，不用作公平速度比较。',
    '- 仅三个种子、两个验证录制组；原初始化/检测缓存训练来源以及原 CUDA 算子一致性限制保留。未来 GT 只供离线监督与评价。',
    '- 共享驾驶模型尚未在本阶段更新，本报告不提供 P/F 驾驶收益、查询策略收益或闭环结论。', '',
    '## 重放', '',
    '使用 README 的完整模型环境，对六个组合分别运行：', '', '```bash',
    'python src/prediction/train_mtr.py NEW_OUTPUT --seed SEED --context CONTEXT --steps-per-epoch 996 --epochs 10 --patience 11 --checkpoint /root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth',
    'python tests/verify_mtr_adaptation.py NEW_OUTPUT NEW_AUDIT.json', '```', '',
    '完整运行命令与日志保存在 `outputs/mtr_stability_v1/run_study.py` 和对应日志；汇总脚本为同目录的 `analyze.py`。']
(ROOT / 'docs/mtr_stability_results.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(families=families, frozen=frozen), ensure_ascii=False))
