"""Real archived values, unchanged: vary only allocation of float state arrays."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from planning import structured_inputs as S
base=Path(__file__).resolve().parent
p=json.load(open(base.parent/'structured_acceptance_2026_09_15_v1/initial_recheck/tasks/g719_Ego.json'))['episode']['plans'][0]['output']['prepared_input']
original=np.asarray
results=[]
for offset in range(32):
 def allocated(value,*args,**kw):
  a=original(value,*args,**kw)
  if a.shape==(11,7) and a.dtype==np.float64:
   storage=np.empty(a.size+64,dtype=np.float64)
   out=storage[offset:offset+a.size].reshape(a.shape);out[:]=a
   assert np.array_equal(out,a)
   return out
  return a
 canonical=copy.deepcopy(p['entities'])
 with patch.object(S.np,'asarray',allocated):
  expected,_=S._tensors(canonical,S.StructuredDriverSpec.from_dict(p['driver_spec']))
 diffs=[]
 for k in expected:
  a,b=original(expected[k]),original(p['tensor_inputs'][k]);mask=a!=b
  if mask.any():
   idx=tuple(np.argwhere(mask)[0]);diffs.append(dict(array=k,count=int(mask.sum()),index=list(map(int,idx)),expected=float(a[idx]),actual=float(b[idx]),max_abs=float(np.max(abs(a.astype(float)-b.astype(float))))))
 try:
  with patch.object(S.np,'asarray',allocated):S.validate_structured_prepared(p)
  status='pass'
 except ValueError as exc:status=str(exc)
 results.append(dict(offset=offset,differences=diffs,validator=status))
(base/'tensor_reproduction.json').write_text(json.dumps(results,indent=2)+'\n')
print('offsets',len(results),'different',sum(bool(x['differences']) for x in results),'validator_failures',sum(x['validator']!='pass' for x in results))
print(next((x for x in results if x['differences']),None))
