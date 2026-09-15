"""Observational exception capture; leaves validator results and inputs unchanged."""
import json
from pathlib import Path
import sys
import numpy as np
from planning import structured_inputs as S, structured_driver as D, driver_contract as C
import run_acceptance

base=Path(__file__).resolve().parent
original=S.validate_structured_prepared
def checked(prepared):
    try:
        return original(prepared)
    except ValueError as exc:
        if str(exc)=='entity metadata and structured tensors disagree':
            tb=exc.__traceback__
            while tb.tb_frame.f_code!=original.__code__:
                tb=tb.tb_next
            variables=tb.tb_frame.f_locals
            details=[]
            for key,value in variables['expected_tensors'].items():
                left,right=np.asarray(value),np.asarray(variables['tensors'][key])
                different=left!=right
                if different.any():
                    at=tuple(np.argwhere(different)[0])
                    details.append(dict(key=key,count=int(different.sum()),
                        first_index=list(map(int,at)),expected=float(left[at]),actual=float(right[at]),
                        max_error=float(np.max(abs(left.astype(float)-right.astype(float))))))
            details.append(dict(indices=[dict(entity=e['entity_id'],actual=e['tensor_index'],
                expected=variables['expected_indices'][e['entity_id']]) for e in prepared['entities']
                if e['tensor_index']!=variables['expected_indices'][e['entity_id']]]))
            with (base/'exact_tensor_mismatches.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(g=prepared['g'],details=details))+'\n')
            print('CAPTURED_MISMATCH',json.dumps(details),flush=True)
        raise
S.validate_structured_prepared=checked
D.validate_structured_prepared=checked
C.validate_structured_prepared=checked
run_acceptance.main()
