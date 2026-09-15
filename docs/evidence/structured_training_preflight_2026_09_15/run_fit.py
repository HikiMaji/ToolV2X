"""Observe existing fit without changing its numerical operations or batching."""
from pathlib import Path
import json, sys, time
import torch
from planning import train_structured_driver as T
from planning.structured_validation import evidence_condition
BASE=Path(__file__).resolve().parent
mode=sys.argv[1]
assert mode in ('continuous','interrupted','resumed')
config=json.loads((BASE/'training_config.json').read_text())
resume=BASE/'interrupted/checkpoint_000002.pt' if mode=='resumed' else None
stop=2 if mode=='interrupted' else 4
original_loss,original_step=T.training_loss,torch.optim.AdamW.step
records=[]; models=[]; pending=[]; step_count=2 if resume else 0
start=time.perf_counter()
trace=(BASE/(mode+'_trace.jsonl')).open('x')
def emit(event):
    trace.write(json.dumps(event,allow_nan=False)+'\n'); trace.flush()
    if event['kind'] in ('step','start'):print(json.dumps(event),flush=True)
def observe_loss(model, features, row, **kwargs):
    begin=time.perf_counter()
    loss,detail=original_loss(model,features,row,**kwargs)
    task=kwargs.get('task')
    item=dict(kind='validation' if kwargs.get('evaluation') else 'train',
        sample_id=row['sample_id'],stage=row['stage'],role=row['role'],
        condition=evidence_condition(row,task),loss=loss.detach().item() if loss is not None else None,
        supervised_forwards=detail['supervised_forwards'],invalid_prior_masks=detail['invalid_prior_masks'],
        seconds=time.perf_counter()-begin)
    emit(item)
    if item['kind']=='train':
        assert row['role']=='train' and loss is not None and torch.isfinite(loss)
        models[:]=[model];pending.append(item)
    return loss,detail
def observe_step(optimizer,*args,**kwargs):
    global step_count
    model=models[0]; gradients={}; tensors=0
    for name,value in model.named_parameters():
        if value.grad is not None:
            assert torch.isfinite(value.grad).all(), name
            tensors+=1
            if name.startswith(('observation_encoder.','forecast_encoder.','output_head.')):
                prefix=name.split('.')[0]
                gradients[prefix]=gradients.get(prefix,0.)+float(value.grad.detach().square().sum())
    begin=time.perf_counter()
    result=original_step(optimizer,*args,**kwargs)
    torch.cuda.synchronize()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    step_count+=1
    emit(dict(kind='step',optimizer_steps=step_count,conditions=[r['condition'] for r in pending],
        rows=[r['sample_id'] for r in pending],losses=[r['loss'] for r in pending],
        finite_gradient_tensors=tensors,gradient_squared_norms=gradients,
        optimizer_seconds=time.perf_counter()-begin))
    pending.clear()
    return result
# Additional read-only observations for the resumed run: exact checkpoint loading.
if resume is not None:
    from planning.structured_driver import StructuredPlannerNetwork
    original_model_load=StructuredPlannerNetwork.load_state_dict
    original_optimizer_load=torch.optim.AdamW.load_state_dict
    original_restore_rng=T._restore_rng
    def model_load(model,state,*args,**kwargs):
        result=original_model_load(model,state,*args,**kwargs)
        exact=T._same(model.state_dict(),state)
        emit(dict(kind='restore_model',exact=exact));assert exact
        return result
    def optimizer_load(optimizer,state):
        result=original_optimizer_load(optimizer,state)
        exact=T._same(optimizer.state_dict(),state)
        emit(dict(kind='restore_optimizer',exact=exact));assert exact
        return result
    def restore_rng(state):
        result=original_restore_rng(state)
        exact=T._same(T._rng(),state)
        emit(dict(kind='restore_rng',exact=exact));assert exact
        return result
    StructuredPlannerNetwork.load_state_dict=model_load
    torch.optim.AdamW.load_state_dict=optimizer_load
    T._restore_rng=restore_rng
T.training_loss=observe_loss
torch.optim.AdamW.step=observe_step
torch.cuda.reset_peak_memory_stats()
emit(dict(kind='start',mode=mode,stop_after_steps=stop,resume=str(resume) if resume else None,
    torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(),
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
try:
    checkpoint=T.fit(BASE/'rows.jsonl',BASE/mode,config,resume=resume,stop_after_steps=stop)
    torch.cuda.synchronize()
    result=dict(mode=mode,checkpoint=str(checkpoint),optimizer_steps=step_count,
        wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),completed=True)
    (BASE/(mode+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    trace.close()
    T.training_loss=original_loss;torch.optim.AdamW.step=original_step
