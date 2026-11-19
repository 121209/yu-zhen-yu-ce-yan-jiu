#!/usr/bin/env python3
"""Evaluation-only finite-region LL decomposition; no training/code changes."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ['CUDA_VISIBLE_DEVICES']=''
_HERE=Path(__file__).resolve().parent
ROOT=None
for _c in (_HERE.parent, _HERE, Path('/workspace/ST-NKF-pytorch')):
    if (_c/'models.py').is_file():
        ROOT=_c; break
if ROOT is None:
    ROOT=Path('/workspace/ST-NKF-pytorch')
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
import torch,numpy as np
from models import KernelPointProcess
from run_models import prepare_run
torch.set_num_threads(4)
OUT=Path(os.environ.get('STNKF_REGION_OUT', str(_HERE.parent/'results')))
class RegionEvaluationProcess(KernelPointProcess):
 def _boundary_mass(self,*args,**kwargs):
  mass=super()._boundary_mass(*args,**kwargs)
  self._last_eval_mass=mass
  return mass
 def forward(self,*args,**kwargs):
  out=super().forward(*args,**kwargs)
  if self.use_boundary_correction:
   out['l_t']=self.bac_func.mu*self.area_scalar+(out['kappa']*out['pdf_temporal']*self._last_eval_mass).sum(dim=1,keepdim=True)
  return out

def main():
 data=json.loads((OUT/'four-schemes-ll.json').read_text())
 prep=prepare_run('SanJac',history=128)
 result={'definition':'lambda_t_R=mu*A+sum_i(kappa_i*h_i*mass_i); LL_ts unchanged, evaluation only', 'training_unchanged':True,'schemes':{}}
 for sid,rec in data['schemes'].items():
  if rec.get('status')!='ok':continue
  start=time.perf_counter()
  model=RegionEvaluationProcess(128,128,128,'neural','neural','neural',prep['config']['global_m0'],prep['area'],5,32,seed=1,**rec['flags'],boundary_angles=128)
  for split,setter in [('train',model.set_train_data),('val',model.set_val_data),('test',model.set_test_data)]:
   p=prep[split]; setter(p['t'],p['m'],p['x'],p['y'])
  model.set_model();model.set_region(prep['region_xy'],n_angles=128);model.load_weights(rec['weights'])
  item={}
  for split in ('val','test'):
   getattr(model,'eval_'+split)(batch_size=16)
   item[split]={k:float(getattr(model,k+'_average_'+split)) for k in ('LL_ts','LL_t','LL_s')}
   item[split]['n']=int(getattr(model,'cur_t_'+split).shape[0])
   assert abs(item[split]['LL_ts']-rec[split]['LL_ts'])<1e-6
  item['seconds']=time.perf_counter()-start
  result['schemes'][sid]=item
  print(sid,json.dumps(item),flush=True)
  (OUT/'four-schemes-region-ll.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
