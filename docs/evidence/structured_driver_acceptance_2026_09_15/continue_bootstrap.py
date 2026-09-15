"""Continue the one existing bootstrap lineage, never exceed 24 updates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback
import numpy as np
import torch
from planning.train_structured_driver import fit, _load
from check_acceptance import check

BASE=Path(__file__).resolve().parent
def save(name,value):
    (BASE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    assert not (BASE/'bounded_bootstrap_result.json').exists()
    checkpoint=BASE/'captured_bootstrap_epoch1/checkpoint_000008.pt'
    config=json.loads((BASE/'offline_bootstrap/concrete_training_v2.json').read_text())
    assert config['epochs']==3 and config['validation']['interval_batches']==8
    assert _load(checkpoint)['progress']['optimizer_steps']==8
    result=dict(status='running',max_epochs=3,max_optimizer_steps=24,
        formal_adaptation_started=False,epochs=[],started_utc=datetime.now(timezone.utc).isoformat())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    save('bounded_bootstrap_result.json',result)
    try:
        for epoch in range(1,4):
            fit_seconds=None
            if epoch>1:
                print('START_EPOCH',epoch,flush=True)
                start=perf_counter()
                checkpoint=fit(BASE/'offline_bootstrap/rows.jsonl',BASE/('bootstrap_epoch%d'%epoch),
                    config,resume=checkpoint,stop_after_steps=epoch*8)
                fit_seconds=perf_counter()-start
                assert _load(checkpoint)['progress']['optimizer_steps']==epoch*8
                with (BASE/('acceptance_epoch%d.log'%epoch)).open('x') as log:
                    subprocess.run([sys.executable,'-u',str(BASE/'diagnose_acceptance.py'),
                        'acceptance_epoch%d'%epoch,str(checkpoint)],cwd=str(BASE.parents[1]),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            directory=BASE/('acceptance_epoch%d'%epoch)
            audit=check(directory)
            record=dict(epoch=epoch,checkpoint=str(checkpoint),optimizer_steps=epoch*8,
                fit_seconds=fit_seconds,initial_valid_frames=audit['initial_valid_frames'],
                passing_tasks=audit['passing_tasks'],audit_failures=audit['audit_failures'],
                pass_gate=audit['pass_gate'])
            result['epochs'].append(record)
            print(json.dumps(record),flush=True)
            save('bounded_bootstrap_result.json',result)
            if audit['pass_gate']:
                result.update(status='bootstrap_gate_passed',frozen_earliest_epoch=epoch,
                    collection_checkpoint=str(checkpoint))
                break
            if audit['audit_failures']:
                result['status']='stopped_on_integration_failure'
                break
        else:
            result['status']='stopped_at_three_epoch_cap'
    except Exception as exc:
        details=[];tb=exc.__traceback__
        while tb:
            loc=tb.tb_frame.f_locals
            if 'expected_tensors' in loc and 'tensors' in loc:
                for key,a in loc['expected_tensors'].items():
                    left,right=np.asarray(a),np.asarray(loc['tensors'][key]);neq=left!=right
                    if neq.any():
                        at=tuple(np.argwhere(neq)[0]);details.append(dict(g=loc['prepared']['g'],key=key,
                            count=int(neq.sum()),index=list(map(int,at)),expected=float(left[at]),actual=float(right[at]),
                            max_error=float(np.max(abs(left.astype(float)-right.astype(float))))))
            tb=tb.tb_next
        result.update(status='stopped_on_training_or_audit_exception',error=str(exc),
            traceback=traceback.format_exc(),tensor_differences=details)
        print(json.dumps(dict(error=str(exc),tensor_differences=details)),flush=True)
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    save('bounded_bootstrap_result.json',result)

if __name__=='__main__':main()
