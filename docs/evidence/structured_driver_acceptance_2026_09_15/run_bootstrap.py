"""At most three predeclared bootstrap epochs, with full acceptance after each."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import torch

from planning.train_structured_driver import fit, _load
from check_acceptance import check

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
def save(name,value):
    (BASE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    initial=json.loads((BASE/'initial_recheck/acceptance_audit.json').read_text())
    assert initial['audit_failures']==0 and initial['initial_valid_frames']<20
    assert all(x['features_equal'] and x['local_forecasts_equal'] for x in initial['shared_inputs'])
    assert not (BASE/'bootstrap_progress.json').exists()
    rows=BASE/'offline_bootstrap/rows.jsonl'
    config=json.loads((BASE/'offline_bootstrap/concrete_training_v2.json').read_text())
    assert config['epochs']==3 and config['validation']['interval_batches']==8
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    resume=None
    progress=dict(status='running',maximum_epochs=3,full_adaptation_started=False,
        started_utc=datetime.now(timezone.utc).isoformat(),epochs=[])
    save('bootstrap_progress.json',progress)
    for epoch in range(1,4):
        begin=perf_counter()
        directory=BASE/('bootstrap_epoch%d'%epoch)
        print('START bootstrap epoch',epoch,flush=True)
        checkpoint=fit(rows,directory,config,resume=resume,stop_after_steps=epoch*8)
        actual=_load(checkpoint)['progress']
        assert actual['epoch']==epoch and actual['optimizer_steps']==epoch*8
        duration=perf_counter()-begin
        record=dict(epoch=epoch,checkpoint=str(checkpoint),fit_wall_seconds=duration,training_progress=actual)
        progress['epochs'].append(record)
        progress['status']='epoch_acceptance'
        save('bootstrap_progress.json',progress)
        print('SAVED bootstrap epoch',epoch,'seconds',duration,flush=True)
        name='acceptance_epoch%d'%epoch
        with (BASE/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(BASE/'run_acceptance.py'),name,str(checkpoint)],
                cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=check(BASE/name)
        record.update(initial_valid_frames=audit['initial_valid_frames'],passing_tasks=audit['passing_tasks'],
            audit_failures=audit['audit_failures'],pass_gate=audit['pass_gate'])
        save('bootstrap_progress.json',progress)
        if audit['pass_gate']:
            progress.update(status='bootstrap_gate_passed',collection_checkpoint=str(checkpoint),
                frozen_earliest_epoch=epoch,finished_utc=datetime.now(timezone.utc).isoformat())
            save('bootstrap_progress.json',progress)
            return
        if audit['audit_failures']:
            progress.update(status='stopped_on_integration_failure',finished_utc=datetime.now(timezone.utc).isoformat())
            save('bootstrap_progress.json',progress)
            return
        resume=checkpoint
    progress.update(status='stopped_at_three_epoch_cap',finished_utc=datetime.now(timezone.utc).isoformat())
    save('bootstrap_progress.json',progress)

if __name__=='__main__':main()
