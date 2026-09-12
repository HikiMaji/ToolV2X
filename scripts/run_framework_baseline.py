"""Run one complete baseline, sequentially, from an isolated source snapshot.

Long stages run as ordinary subprocesses. Launch this script in a detached process
to let preparation, training, generation and evaluation continue between turns.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def latest_checkpoint(training, seen=None):
    seen = set() if seen is None else seen
    training = training.resolve()
    if training in seen:
        raise ValueError('cycle in training resume sources')
    seen.add(training)
    candidates = []
    required = ('training_state.pt', 'adapter_model.safetensors', 'non_lora_trainables.bin')
    for pattern in ('checkpoint-epoch*/training_state.json', 'recovery-step*/training_state.json'):
        markers = list(training.glob(pattern)) + list((training.parent/'first_step').glob(pattern))
        for marker in markers:
            checkpoint = marker.parent
            if not all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size for name in required):
                continue
            state = json.loads(marker.read_text())
            candidates.append(((state['steps'], state['epoch']), checkpoint))
    if not candidates:
        config = training.parent/'config.json'
        if config.is_file():
            source = json.loads(config.read_text()).get('resume_training')
            if source:
                return latest_checkpoint(Path(source)/'training', seen)
        raise ValueError('no complete saved training checkpoint')
    return max(candidates, key=lambda item:item[0])[1]


def run(out, data, per_recording=0, epochs=3, resume_preparation=None, resume_training=None,
        reuse_baseline=None, efficiency_report=None, verify_first_resume=False):
    if per_recording < 0 or not 1 <= epochs <= 3:
        raise ValueError('invalid sampling or epoch budget')
    prepared_root, training_data, checkpoint = out/'episodes', out/'training_data', None
    completed_state = None
    previous_config = None
    previous_run = resume_training or reuse_baseline
    if previous_run:
        if resume_preparation or resume_training and reuse_baseline:
            raise ValueError('preparation and training resume are mutually exclusive')
        previous_config = json.loads((previous_run/'config.json').read_text())
        if (previous_config['data'] != str(data) or previous_config['per_recording'] != per_recording or
                previous_config['epochs'] != epochs):
            raise ValueError('continued baseline configuration changed')
        if resume_training:
            checkpoint = latest_checkpoint(resume_training/'training')
            stored = json.loads((checkpoint/'training_state.json').read_text())
            if stored['epoch'] >= epochs or stored.get('bad_epochs', 0) >= 2:
                completed_state = stored
        prepared_root = Path(previous_config.get('prepared_root', str(previous_run/'episodes')))
        training_data = Path(previous_config.get('training_data', str(previous_run/'training_data')))
        if not prepared_root.is_dir() or not training_data.is_dir():
            raise ValueError('missing previously prepared data')
    out.mkdir(parents=True, exist_ok=False)
    code = out/'code'
    for name in ('src', 'scripts', 'tests', 'vendor'):
        shutil.copytree(ROOT/name, code/name, ignore=shutil.ignore_patterns('__pycache__'))
    (code/'outputs').mkdir()
    (code/'outputs/causal_windows_v1').symlink_to(ROOT/'outputs/causal_windows_v1', target_is_directory=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    resources = dict(TOOLV2X_V2VGOT_ROOT=ROOT.parent/'V2V-GoT', TOOLV2X_CMP_ROOT=ROOT.parent/'CMP',
                     TOOLV2X_LLAVA_BASE=ROOT/'models/llava-v1.5-7b', TOOLV2X_CLIP_ROOT=ROOT/'models/clip-vit-large-patch14-336')
    for key, default in resources.items():
        env[key] = str(Path(env.get(key, str(default))).resolve())
    original = Path(env['TOOLV2X_V2VGOT_ROOT'])/'LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2/checkpoint-4330'
    env['TOOLV2X_V2VGOT_CHECKPOINT'] = str(Path(env.get('TOOLV2X_V2VGOT_CHECKPOINT', str(original))).resolve())
    env['TOOLV2X_LLAVA_ROOT'] = str(code/'vendor/v2vgot_llava')
    if previous_config:
        for key, value in previous_config.get('resources', {}).items():
            if key != 'TOOLV2X_LLAVA_ROOT' and env.get(key) != value:
                raise ValueError('continued baseline resource changed: '+key)
    config = dict(data=str(data), code=str(code), python=sys.executable, epochs=epochs,
        per_recording=per_recording, policies=['Ego', 'P', 'F', 'PF', 'rule'], p_processing='local_mtr',
        integration_only=per_recording > 0, initialization='configured GoT checkpoint without resuming integration weights', mtr_frozen=True,
        resources={k:env[k] for k in list(resources)+['TOOLV2X_LLAVA_ROOT', 'TOOLV2X_V2VGOT_CHECKPOINT']},
        resume_preparation=str(resume_preparation) if resume_preparation else None,
        resume_training=str(resume_training) if resume_training else None,
        reuse_baseline=str(reuse_baseline) if reuse_baseline else None,
        efficiency_report=str(efficiency_report) if efficiency_report else None,
        verify_first_resume=verify_first_resume,
        prepared_root=str(prepared_root), training_data=str(training_data))
    if resume_training:
        config['initialization'] = 'exact saved training state from '+str(checkpoint)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    status = dict(status='running', pid=os.getpid(), started=datetime.now().astimezone().isoformat(), stages=[])

    def save():
        status['updated'] = datetime.now().astimezone().isoformat()
        temporary = out/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(out/'status.json')

    def stage(name, script, *arguments, required=True):
        command = [sys.executable, str(code/script)] + list(map(str, arguments))
        status.update(stage=name, command=command)
        save()
        with (out/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=str(code), env=env, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save()
            returncode = child.wait()
        status.pop('child_pid', None)
        status['stages'].append(dict(name=name, returncode=returncode))
        save()
        if returncode and required:
            raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
        return returncode

    try:
        train_args = []
        if resume_training and not efficiency_report and completed_state is None:
            train_args = ['--resume', checkpoint]
            code_result = stage('benchmark', 'src/planning/train_driver.py', training_data, out/'benchmark',
                '--epochs', epochs, *train_args, '--benchmark-only', required=False)
            decision = dict(enable_answer_head=False, benchmark_returncode=code_result)
            report = out/'benchmark/performance.json'
            if code_result == 0 and report.is_file():
                decision['enable_answer_head'] = json.loads(report.read_text())['enable_answer_head'] is True
            (out/'efficiency_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
            if decision['enable_answer_head']:
                train_args.append('--answer-only-head')
        elif efficiency_report:
            decision = dict(enable_answer_head=json.loads(efficiency_report.read_text())['enable_answer_head'] is True,
                            report=str(efficiency_report))
            (out/'efficiency_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
            if checkpoint:
                train_args = ['--resume', checkpoint]
            if decision['enable_answer_head']:
                train_args.append('--answer-only-head')
        if not previous_run:
            resume_args = ['--resume-from', resume_preparation] if resume_preparation else []
            stage('prepare', 'src/planning/run_framework.py', 'prepare', prepared_root, '--data', data,
                  '--per-recording', per_recording, *resume_args)
            stage('export', 'scripts/prepare_framework_training.py', prepared_root, training_data, '--data', data)
        if verify_first_resume and completed_state is None:
            if checkpoint:
                raise ValueError('first-update resume verification requires original initialization')
            stage('first_step', 'src/planning/train_driver.py', training_data, out/'first_step',
                  '--epochs', epochs, *train_args, '--pause-after-steps', 1)
            saved = json.loads((out/'first_step/result.json').read_text())
            if saved['status'] != 'training_paused' or saved['optimizer_steps'] != 1:
                raise ValueError('first update was not saved for restore verification')
            train_args += ['--resume', saved['checkpoint']]
        if completed_state is None:
            stage('train', 'src/planning/train_driver.py', training_data, out/'training', '--epochs', epochs, *train_args)
        else:
            (out/'training').mkdir()
            restored_result = dict(status='training_completed', checkpoint=completed_state['best_checkpoint'],
                optimizer_steps=completed_state['steps'], epochs=completed_state['epoch'],
                examples_seen=completed_state['examples_seen'], new_training_examples=0,
                validation_loss=completed_state['best_loss'], formal_adaptation_complete=not bool(per_recording),
                generation_evaluated=False, method_effectiveness_established=False,
                recovered_completed_training_state=str(checkpoint))
            (out/'training/result.json').write_text(json.dumps(restored_result, indent=2)+'\n')
        result = json.loads((out/'training/result.json').read_text())
        if result['status'] != 'training_completed' or (not per_recording and not result['formal_adaptation_complete']):
            raise ValueError('training did not complete the configured adaptation')
        stage('generate', 'src/planning/run_framework.py', 'generate', prepared_root, out/'generations',
              '--checkpoint', result['checkpoint'])
        stage('evaluate', 'src/evaluation/framework.py', out/'generations', out/'evaluation', '--data', data)
        status.update(status='completed', method_effectiveness_established=False)
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/paired_driving_data_v1')
    parser.add_argument('--per-recording', type=int, default=0, help='0 uses the full causal index; a subset is integration only')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--resume-preparation',type=Path,help='previous episodes directory; continue the same preparation configuration')
    parser.add_argument('--resume-training', type=Path, help='stopped baseline with a complete recovery/epoch checkpoint')
    parser.add_argument('--reuse-baseline', type=Path, help='reuse completed preparation/export; restart from original weights')
    parser.add_argument('--efficiency-report', type=Path, help='use an existing measured answer-head decision')
    parser.add_argument('--verify-first-resume', action='store_true', help='save first update, exit and restore before long training')
    args = parser.parse_args()
    run(args.out.resolve(), args.data.resolve(), args.per_recording, args.epochs,
        args.resume_preparation.resolve() if args.resume_preparation else None,
        args.resume_training.resolve() if args.resume_training else None,
        args.reuse_baseline.resolve() if args.reuse_baseline else None,
        args.efficiency_report.resolve() if args.efficiency_report else None, args.verify_first_resume)
