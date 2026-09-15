"""Read-only v1 tensor replay, with unrelated allocator churn between checks."""
import copy,json,random
from pathlib import Path
import numpy as np
from planning import structured_inputs as S
base=Path(__file__).resolve().parent
p=json.load(open(base.parent/'structured_acceptance_2026_09_15_v1/initial_recheck/tasks/g719_Ego.json'))['episode']['plans'][0]['output']['prepared_input']
spec=S.StructuredDriverSpec.from_dict(p['driver_spec']); rng=random.Random(7)
found=[]
for trial in range(300):
 # Allocation churn changes no input or arithmetic, nor the validator.
 garbage=[np.empty(rng.randrange(4,200)) for _ in range(75)]
 q=copy.deepcopy(p)
 q['tensor_inputs'],_=S._tensors(copy.deepcopy(q['entities']),spec)
 garbage=[np.empty(rng.randrange(4,200)) for _ in range(75)]
 try:S.validate_structured_prepared(q)
 except ValueError as e:
  if str(e)!='entity metadata and structured tensors disagree':raise
  tb=e.__traceback__
  while tb.tb_frame.f_code != S.validate_structured_prepared.__code__:tb=tb.tb_next
  loc=tb.tb_frame.f_locals;diffs=[]
  for k,values in loc['expected_tensors'].items():
   a,b=np.asarray(values),np.asarray(loc['tensors'][k]);at=np.argwhere(a!=b)
   if len(at):
    ix=tuple(at[0]);diffs.append(dict(array=k,index=list(map(int,ix)),expected=float(a[ix]),actual=float(b[ix]),count=len(at)))
  found.append(dict(trial=trial,error=str(e),differences=diffs))
  (base/'legacy_failing_prepared.json').write_text(json.dumps(q)+'\n')
  break
(base/'legacy_validator_capture.json').write_text(json.dumps(dict(attempts=trial+1,failures=found),indent=2)+'\n')
print('attempts',trial+1,'failures',found,flush=True)
