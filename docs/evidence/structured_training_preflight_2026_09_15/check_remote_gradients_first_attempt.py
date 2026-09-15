"""Four real final-stage losses; remote input gradients, zero optimizer steps."""
from pathlib import Path
import json
import torch
from planning import train_structured_driver as T
from planning import structured_driver as D
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parents[1]/'outputs/structured_driver_fix_2026_09_15_v1/acceptance'
rows=[json.loads(x) for x in (BASE/'rows.jsonl').read_text().splitlines()]
config=json.loads((BASE/'training_config.json').read_text())
first=next(r for r in rows if r['role']=='train')['sample_id']
results=[]
for arm in ('Ego','P','F','PF'):
    candidates=[r for r in rows if r['sample_id']==first and Path(r['source_task']).stem.endswith('_'+arm)]
    row=max(candidates,key=lambda r:r['stage'])
    task=json.loads(Path(row['source_task']).read_text())
    planner=T.load_structured_planner(SOURCE/'initialized_seed7.pt',device='cuda')
    planner.model.train();planner.model.zero_grad(set_to_none=True)
    captured=[];original=D.collate_structured_inputs
    def observed(features,prepared,device='cpu'):
        batch=original(features,prepared,device)
        for name in ('observations','forecasts'):batch[name].requires_grad_(True)
        captured.append((batch,prepared[0]))
        return batch
    D.collate_structured_inputs=observed
    try:
        loss,detail=T.training_loss(planner.model,T._read_features(row['inputs']['feature_path']),row,
            task=task,refinement_depth=1,objective=config['objective'])
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
    finally:D.collate_structured_inputs=original
    norms={'history':0.,'forecast':0.};locations={'history':0,'forecast':0}
    for batch,prepared in captured:
        for group in prepared['admission_report']['field_groups']:
            if group['source_role']!='remote' or group['use']!='tensor':continue
            for location in group['tensor_locations']:
                name=location['array'];value=batch[name]
                if value.grad is None:continue
                e=location['entity']
                if name=='observations':
                    kind='history';s=location['source_slot'];mask=batch['observation_mask'][0,e,s]
                else:
                    kind='forecast';s=location['forecast_set'];mask=batch['forecast_mask'][0,e,s]
                gradient=value.grad[0,e,s][mask]
                assert torch.isfinite(gradient).all()
                norms[kind]+=float(gradient.square().sum());locations[kind]+=1
    assert (norms['history']>0)==('P' in arm)
    assert (norms['forecast']>0)==('F' in arm)
    state=T._load(SOURCE/'initialized_seed7.pt')['model_state']
    assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in planner.model.state_dict().items())
    result=dict(arm=arm,sample_id=first,stage=row['stage'],loss=float(loss.detach()),
        remote_input_gradient_squared_norms=norms,supervised_remote_locations=locations,
        supervised_forwards=detail['supervised_forwards'],invalid_prior_masks=detail['invalid_prior_masks'],
        optimizer_steps=0,weights_unchanged=True)
    results.append(result);print(json.dumps(result),flush=True)
(BASE/'remote_gradients.json').write_text(json.dumps(dict(status='PASS',results=results,
    scope='Actual remote tensor gradients on one fixed train frame; not evidence utility or method quality.'),indent=2)+'\n')
