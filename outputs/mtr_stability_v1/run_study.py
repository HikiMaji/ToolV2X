"""Execute the preregistered study sequentially; preserve each failure and log."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/ToolV2X')
OUT = ROOT / 'outputs/mtr_stability_v1'
PYTHON = '/root/autodl-tmp/conda-envs/llava/bin/python'
CHECKPOINT = '/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth'
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
for name in ('full', 'review'):
    shutil.copy2('/tmp/toolv2x_stability_%s_tests.log' % name, OUT / (name + '_tests.log'))
shutil.copy2(ROOT / 'docs/mtr_stability_plan.md', OUT / 'protocol.md')
for seed in (20, 21, 22):
    for context in ('full', 'alternating'):
        name = '%s_seed%d' % (context, seed)
        out = OUT / name
        if out.exists():
            raise FileExistsError('refuse to overwrite study output: ' + str(out))
        command = [PYTHON, str(ROOT / 'src/prediction/train_mtr.py'), str(out),
                   '--checkpoint', CHECKPOINT, '--seed', str(seed), '--context', context,
                   '--steps-per-epoch', '996', '--epochs', '10', '--patience', '11']
        print(json.dumps(dict(event='start', run=name, command=command)), flush=True)
        with (OUT / (name + '.log')).open('x') as log:
            status = subprocess.run(command, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            print(json.dumps(dict(event='failed', run=name, returncode=status.returncode)), flush=True)
            sys.exit(status.returncode)
        completion = json.loads((out / 'completion.json').read_text())
        if completion['optimizer_steps'] != 9960 or completion['completed_epochs'] != 10:
            raise ValueError('unmatched training budget: ' + name)
        audit = subprocess.run([PYTHON, str(ROOT / 'tests/verify_mtr_adaptation.py'),
                                str(out), str(OUT / (name + '_audit.json'))],
                               cwd=str(ROOT), env=environment, capture_output=True, text=True)
        (OUT / (name + '_audit.log')).write_text(audit.stdout + audit.stderr)
        if audit.returncode:
            print(json.dumps(dict(event='audit_failed', run=name, returncode=audit.returncode)), flush=True)
            sys.exit(audit.returncode)
        print(json.dumps(dict(event='completed', run=name, best_epoch=completion['best_epoch'],
                              best_metric=completion['best_metric'], elapsed_s=completion['elapsed_s'])), flush=True)
print(json.dumps(dict(event='all_six_completed')), flush=True)
