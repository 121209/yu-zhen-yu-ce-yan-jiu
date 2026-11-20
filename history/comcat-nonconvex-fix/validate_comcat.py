import sys, os, json, time, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/hy-tmp/pack')
from run_four_schemes_gpu import build_model, SCHEMES, RUN_ORDER, BATCH_SIZE
from run_models import prepare_run
from models import KernelPointProcess, _points_in_polygon

torch.set_num_threads(1)
records={}
def fixture():
 m=KernelPointProcess(4,4,4,'neural','neural','neural',1.,100.,2,8,seed=4,backbone='transformer',use_boundary_correction=True,init_mu_from_rate=False).double()
 m.max_dis=1.
 return m

def validate(poly, sources, scale, label):
 m=fixture(); xy=torch.tensor(sources,dtype=torch.float64)[None]
 mags=torch.ones(1,len(sources),1,dtype=torch.float64)
 w=torch.ones(1,len(sources),1,dtype=torch.float64)
 D=torch.full_like(w,scale**2,requires_grad=True); q=torch.full_like(w,1.8)
 values={}
 for a in [128,4096]:
  m.set_region(poly,a)
  mass=m._boundary_mass(xy[...,0:1],xy[...,1:2],mags,(w,D,q))
  values[a]=mass.detach().numpy().ravel()
  assert torch.isfinite(mass).all() and ((mass>=0)&(mass<=1+1e-10)).all()
  if a==128:
   grad=torch.autograd.grad(mass.sum(),D)[0]
   assert torch.isfinite(grad).all()
   eps=scale**2*1e-5
   with torch.no_grad():
    hi=m._boundary_mass(xy[...,0:1],xy[...,1:2],mags,(w,D+eps,q))
    lo=m._boundary_mass(xy[...,0:1],xy[...,1:2],mags,(w,D-eps,q))
   fd=(hi-lo)/(2*eps)
   assert torch.allclose(grad,fd,atol=1e-7,rtol=1e-4),(grad,fd)
 rng=np.random.default_rng(20261114); n=300000
 u=rng.random(n); theta=rng.random(n)*2*np.pi
 radius=scale*np.sqrt((1-u)**(-1/.8)-1)
 delta=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
 mc=np.array([_points_in_polygon(delta+s,poly).mean() for s in sources])
 err=abs(values[128]-mc)
 assert max(err)<0.015,(label,values,mc)
 assert max(abs(values[4096]-mc))<0.005,(label,values,mc)
 # Reversing vertex order must preserve mass, including vertex-aligned rays.
 m.set_region(poly[::-1],128)
 reverse=m._boundary_mass(xy[...,0:1],xy[...,1:2],mags,(w,D,q)).detach().numpy().ravel()
 assert np.allclose(reverse,values[128],atol=1e-12)
 records[label]={'polar128':values[128].tolist(),'polar4096':values[4096].tolist(),'mc300000':mc.tolist(),'max_abs_mc_error128':float(max(err)),'gradient_finite_difference_max_error':float((grad-fd).abs().max()),'orientation_error':float(abs(reverse-values[128]).max())}
 print(label,records[label],flush=True)

# U notch: horizontal rays can traverse two separate interior intervals.
u=np.array([[0,0],[4,0],[4,4],[3,4],[3,1],[1,1],[1,4],[0,4]],float)
validate(u,[[.5,2],[2,2],[-1,2],[2,.5],[0,0],[1,1]],.7,'U_shape')
square=np.array([[-5,-5],[5,-5],[5,5],[-5,5]],float)
validate(square,[[0,0],[3.5,0],[7,0]],2.,'convex_square')
# Compare untouched convex implementation to signed algorithm, exact same rays.
m=fixture();m.set_region(square,128)
x=torch.tensor([[[0.],[3.5],[7.]]],dtype=torch.float64);y=torch.zeros_like(x);mg=torch.ones_like(x)
params=(torch.ones_like(x),torch.full_like(x,4.),torch.full_like(x,1.8))
old=m._boundary_mass(x,y,mg,params);m._region_is_convex=False
new=m._boundary_mass(x,y,mg,params)
assert torch.allclose(old,new,atol=1e-12)
records['convex_old_vs_signed_max_error']=float(abs(old-new).max())
prep=prepare_run('ComCat',history=128)
counts=[int(prep[k]) for k in ['auxiliary_num','training_num','validation_num','testing_num']]
assert counts==[14933,40701,14741,21885],counts
poly=prep['region_xy']; print('COMCAT_POLYGON',poly.tolist(),flush=True)
xy=np.column_stack((prep['train']['x'][:5],prep['train']['y'][:5]))
span=np.ptp(poly,axis=0)
xy=np.vstack((xy,np.mean(poly,axis=0),poly.min(axis=0)-span*.1))
validate(poly,xy.tolist(),float(min(span)*.07),'ComCat_polygon')
records['counts']=counts
records['timing']={}
for sid in RUN_ORDER:
 torch.manual_seed(1);torch.cuda.manual_seed_all(1)
 m=build_model(prep,SCHEMES[sid],torch.device('cuda'))
 opt=torch.optim.Adam(m.adam_parameters(),lr=m.lr)
 t=time.perf_counter(); losses=[]
 for b in range(24):
  idx=np.arange(b*16,(b+1)*16)
  opt.zero_grad(set_to_none=True);loss=m.loss_on('train',idx,create_graph=True)
  assert torch.isfinite(loss)
  loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  opt.step();m.project_constraints();losses.append(float(loss.detach()))
 torch.cuda.synchronize(); sec=(time.perf_counter()-t)/24
 records['timing'][sid]={'train_batch_seconds':sec,'train_epoch_lower_bound_seconds':sec*math.ceil(40701/16),'loss_first':losses[0],'loss_last':losses[-1],'all_gradients_finite':True}
 print('TIMING',sid,records['timing'][sid],flush=True)
 del m,opt;torch.cuda.empty_cache()
Path('/hy-tmp/comcat-validation.json').write_text(json.dumps(records,indent=2))
print('ALL_VALIDATION_PASSED',flush=True)
