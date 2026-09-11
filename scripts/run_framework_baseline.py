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


def run(out, data, per_recording=0, epochs=3):
    if per_recording < 0 or not 1 <= epochs <= 3:
        raise ValueError('invalid sampling or epoch budget')
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
    config = dict(data=str(data), code=str(code), python=sys.executable, epochs=epochs,
        per_recording=per_recording, policies=['Ego', 'P', 'F', 'PF', 'rule'], p_processing='local_mtr',
        integration_only=per_recording > 0, initialization='configured GoT checkpoint without resuming integration weights', mtr_frozen=True,
        resources={k:env[k] for k in list(resources)+['TOOLV2X_LLAVA_ROOT', 'TOOLV2X_V2VGOT_CHECKPOINT']})
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    status = dict(status='running', pid=os.getpid(), started=datetime.now().astimezone().isoformat(), stages=[])

    def save():
        status['updated'] = datetime.now().astimezone().isoformat()
        temporary = out/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(out/'status.json')

    def stage(name, script, *arguments):
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
        if returncode:
            raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))

    try:
        stage('prepare', 'src/planning/run_framework.py', 'prepare', out/'episodes', '--data', data,
              '--per-recording', per_recording)
        stage('export', 'scripts/prepare_framework_training.py', out/'episodes', out/'training_data', '--data', data)
        stage('train', 'src/planning/train_driver.py', out/'training_data', out/'training', '--epochs', epochs)
        result = json.loads((out/'training/result.json').read_text())
        if result['status'] != 'training_completed' or (not per_recording and not result['formal_adaptation_complete']):
            raise ValueError('training did not complete the configured adaptation')
        stage('generate', 'src/planning/run_framework.py', 'generate', out/'episodes', out/'generations',
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
    args = parser.parse_args()
    run(args.out.resolve(), args.data.resolve(), args.per_recording, args.epochs)
