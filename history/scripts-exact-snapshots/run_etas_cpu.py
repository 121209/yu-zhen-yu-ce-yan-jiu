import os,sys,time,json,logging
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.chdir("/workspace/ST-NKF-pytorch")
sys.path.insert(0,os.getcwd())
import numpy as np
import torch
from run_models import prepare_run
from models import KernelPointProcess
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
torch.set_num_threads(2)
np.random.seed(1); torch.manual_seed(1)
t0=time.time(); p=prepare_run("SanJac")
m=KernelPointProcess(p["input_dim_train"],p["input_dim_val"],p["input_dim_test"],"empirical","empirical","empirical",p["config"]["global_m0"],p["area"],5,32,seed=1)
for split, setter in [("train",m.set_train_data),("val",m.set_val_data),("test",m.set_test_data)]:
 d=p[split]; setter(d["t"],d["m"],d["x"],d["y"])
m.set_model().compile(lr=0.005).fit_eval(epochs=1000,batch_size=128,patience=20,mu_batch_size=512).eval_test(128)
result={"model":"limited-history ETAS", "seed":1,"H":p["input_dim_train"],"epoch_cap":1000,"patience":20,"best_epoch":m.best_epoch,"stopped_early":m.stopped_early,"best_val_loss":m.best_val_loss,"test_LL_ts":m.LL_ts_average_test,"test_LL_t":m.LL_t_average_test,"test_LL_s":m.LL_s_average_test,"seconds":time.time()-t0,"torch_threads":2}
print(json.dumps(result,indent=2),flush=True)
open("/workspace/st-nkf-diagrams/etas-cpu-result.json","w").write(json.dumps(result,indent=2))
m.save_weights("/workspace/st-nkf-diagrams/etas-cpu-converged.pt")
