"""Small SanJac prepared-split-prefix smoke, NOT official test split or benchmark."""
import os,sys,json,time,logging
os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir('/workspace/ST-NKF-pytorch'); sys.path.insert(0,os.getcwd())
import numpy as np
import torch
from run_models import prepare_run
from models import KernelPointProcess
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
torch.set_num_threads(2)
p=prepare_run('SanJac'); cat=p['raw_catalog']
H=128; ntrain=256; nval=128; ntest=128
# Reproduce the previous H128 smoke: take the first H+n entries of EACH
# prepared split input. These inputs contain the original H1672 history, so
# small-prefix val/test targets are BEFORE the official target boundary.
# This is a numerical smoke only, not a benchmark evaluation.
counts={'train':ntrain,'val':nval,'test':ntest}
blocks={name:{key:values[:H+counts[name]] for key,values in p[name].items()} for name in counts}
results=[]
for epochs, lr in ((1,.005),(5,.005),(5,.0001)):
 t0=time.time(); np.random.seed(1); torch.manual_seed(1)
 m=KernelPointProcess(H,H,H,'neural','neural','neural',p['config']['global_m0'],p['area'],5,32,seed=1,backbone='transformer')
 for name,setter in [('train',m.set_train_data),('val',m.set_val_data),('test',m.set_test_data)]:
  d=blocks[name]; setter(d['t'],d['m'],d['x'],d['y'])
 m.set_model().compile(lr=lr).fit_eval(epochs=epochs,batch_size=8,patience=20,mu_batch_size=8).eval_test(batch_size=8)
 result={'design':m.temporal_net.design_version,'seed':1,'H':H,'train_targets':ntrain,'val_targets':nval,'test_targets':ntest,'batch':8,'epochs':epochs,'lr':lr,'LL_ts':m.LL_ts_average_test,'LL_t':m.LL_t_average_test,'LL_s':m.LL_s_average_test,'seconds':time.time()-t0,'max_time':m.max_time,'max_mag':m.max_mag,'max_dis':m.max_dis,'domain_note':'prefix of each prepared split input; not official val/test targets'}
 results.append(result);print('CDF_SMOKE '+json.dumps(result),flush=True)
 m.save_weights(f'/workspace/st-nkf-diagrams/cdf-v2-H128-{epochs}epoch'+('' if lr==.005 else '-lr1e-4')+'.pt')
open('/workspace/st-nkf-diagrams/cdf-v2-smoke-results.json','w').write(json.dumps(results,indent=2))
