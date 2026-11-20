"""Read-only status and checkpoint audit for this rerun only."""
import argparse, hashlib, json, subprocess
from datetime import datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
r=Path('/hy-tmp/st-nkf-comcat-20261114');d=r/'ComCat'
result={'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S Asia/Shanghai'),'instance_online':True,'training_tmux_alive':subprocess.run(['tmux','has-session','-t','st-nkf-comcat'],capture_output=True).returncode==0,'done':(r/'DONE').read_text() if (r/'DONE').exists() else None,'epochs':{},'schemes':{},'files':{},'checkpoint_payloads':{},'alerts':[]}
try:
 if (d/'four-schemes-gpu-ll.json').exists():
  rec=json.loads((d/'four-schemes-gpu-ll.json').read_text())
  result['protocol']=rec.get('protocol',{})
  for sid,v in rec.get('schemes',{}).items():
   v=dict(v)
   n=v.get('test',{}).get('n');ll=v.get('test',{}).get('LL_ts')
   v['full_50_epochs']=v.get('status')=='ok' and v.get('epochs_ran')==50 and v.get('stop_reason') in ('completed','already_finished')
   if n is not None and n!=21885:result['alerts'].append(sid+' unexpected test count '+str(n))
   if ll is not None:v['test_LL_ts_total']=ll*21885
   if v.get('status')=='ok' and not v['full_50_epochs']:result['alerts'].append(sid+' marked ok without completed50')
   result['schemes'][sid]=v
except Exception as e:result['json_read_error']=str(e)
for f in (d/'checkpoints').glob('*_epochs.jsonl'):
 try:result['epochs'][f.name.removesuffix('_epochs.jsonl')]=json.loads(f.read_text().splitlines()[-1])
 except Exception as e:result['alerts'].append(f.name+': '+str(e))
for f in r.rglob('*'):
 if f.is_file():
  st=f.stat();result['files'][str(f.relative_to(r))]={'size':st.st_size,'mtime':st.st_mtime,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
if args.verify:
 import torch,pickle
 for f in (d/'checkpoints').glob('*.pt'):
  obj=torch.load(f,map_location='cpu',weights_only=False)
  meta={'keys':sorted(obj),'has_weights':bool(obj.get('state_dict')),'finite_weights':all(torch.isfinite(v).all().item() for v in obj.get('state_dict',{}).values()),'best_epoch_one_based':int(obj.get('best_epoch',-1))+1}
  if f.name.endswith('_last.pt'):
   meta.update(next_epoch=int(obj['next_epoch']),has_optimizer=bool(obj.get('optimizer',{}).get('state')),shuffle_rng_loaded=pickle.loads(obj['rng']).__class__.__name__,has_best_state=bool(obj.get('best_state')))
  result['checkpoint_payloads'][f.name]=meta
  if not meta['finite_weights']:result['alerts'].append('nonfinite checkpoint '+f.name)
print(json.dumps(result,ensure_ascii=False))
