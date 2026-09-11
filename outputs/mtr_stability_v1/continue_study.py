"""Run the remaining seed after preserving an interrupted attempt; no partial resume."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/ToolV2X')
OUT = ROOT / 'outputs/mtr_stability_v1'
PYTHON = '/root/autodl-tmp/conda-envs/llava/bin/python'
CHECKPOINT = '/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth'
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def record(**values):
    values['runner_pid'] = os.getpid()
    (OUT / 'continuation_status.json').write_text(json.dumps(values, indent=2) + '\n')
    print(json.dumps(values), flush=True)


for seed in (20, 21):
    for context in ('full', 'alternating'):
        name = '%s_seed%d' % (context, seed)
        completion = json.loads((OUT / name / 'completion.json').read_text())
        audit = json.loads((OUT / (name + '_audit.json')).read_text())
        assert completion['optimizer_steps'] == 9960 and audit['status'] == 'PASS'

for context in ('full', 'alternating'):
    name = context + '_seed22'
    out = OUT / name
    if out.exists():
        raise FileExistsError(str(out))
    command = [PYTHON, str(ROOT / 'src/prediction/train_mtr.py'), str(out),
               '--checkpoint', CHECKPOINT, '--seed', '22', '--context', context,
               '--steps-per-epoch', '996', '--epochs', '10', '--patience', '11']
    record(status='training', run=name, command=command)
    with (OUT / (name + '.log')).open('x') as log:
        process = subprocess.run(command, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        record(status='failed', run=name, returncode=process.returncode)
        sys.exit(process.returncode)
    completion = json.loads((out / 'completion.json').read_text())
    assert completion['optimizer_steps'] == 9960 and completion['completed_epochs'] == 10
    record(status='auditing', run=name)
    audit = subprocess.run([PYTHON, str(ROOT / 'tests/verify_mtr_adaptation.py'), str(out),
                            str(OUT / (name + '_audit.json'))], cwd=str(ROOT), env=environment,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / (name + '_audit.log')).write_text(audit.stdout)
    if audit.returncode:
        record(status='audit_failed', run=name, returncode=audit.returncode)
        sys.exit(audit.returncode)
    record(status='run_completed', run=name, best_epoch=completion['best_epoch'], best_metric=completion['best_metric'])
record(status='all_six_completed')
