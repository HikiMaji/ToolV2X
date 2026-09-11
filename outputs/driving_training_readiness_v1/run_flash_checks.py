"""Run the approved one-update checks sequentially, preserving failures and logs."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/ToolV2X')
OUT = ROOT / 'outputs/driving_training_readiness_v1'
PYTHON = '/root/autodl-tmp/conda-envs/llava/bin/python'
LLM = Path('/root/autodl-tmp/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora') / (
    'llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vllmq5_10ep_both_shallow_f2/checkpoint-490')
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))


def record(**values):
    values['runner_pid'] = os.getpid()
    (OUT / 'flash_checks_status.json').write_text(json.dumps(values, indent=2) + '\n')
    print(json.dumps(values), flush=True)


assert LLM.is_dir()
for initialization in ('got', 'llm'):
    for decoding in ('direct', 'q8_q9'):
        name = '%s_%s_flash' % (initialization, decoding)
        output = OUT / name
        if output.exists():
            raise FileExistsError(str(output))
        command = [PYTHON, str(ROOT / 'src/planning/check_training.py'), str(output), '--decoding', decoding]
        if initialization == 'llm':
            command += ['--checkpoint', str(LLM)]
        for stage, arguments in (('update', command),
                                 ('reload', [PYTHON, str(ROOT / 'tests/verify_driving_update.py'), str(output)])):
            record(status='running', run=name, stage=stage, command=arguments)
            with (OUT / ('%s_%s.log' % (name, stage))).open('x') as log:
                process = subprocess.run(arguments, cwd=str(ROOT), env=environment,
                                         stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                record(status='failed', run=name, stage=stage, returncode=process.returncode)
                sys.exit(process.returncode)
        training = json.loads((output / 'training_check.json').read_text())
        reload = json.loads((output / 'reload_check.json').read_text())
        assert training['optimizer_steps'] == 1 and training['status'] == 'updated_and_saved'
        assert reload['status'] == 'PASS' and reload['actual_generation']
        record(status='run_completed', run=name)
record(status='all_four_completed', formal_adaptation_complete=False)
