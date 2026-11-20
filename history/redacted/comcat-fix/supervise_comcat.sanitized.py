"""Deadline-limited read-only supervisor; existing watcher owns all transfers."""
import hashlib,json,os,subprocess,time,zipfile
from datetime import datetime
from pathlib import Path
R=Path('/workspace/st-nkf-diagrams/comcat-fix-20261114')
M=Path('/workspace/st-nkf-diagrams/st-nkf-comcat-20261114')
DEADLINE=datetime.now().replace(hour=12,minute=38,second=0,microsecond=0)
PRECHECK=datetime.now().replace(hour=12,minute=20,second=0,microsecond=0)
ssh=['ssh','-S','<REDACTED_SSH_CTL_OR_IDENTITY>','-p','13953','-o','ConnectTimeout=12','-o','HostKeyAlias=<REDACTED_SSH_CTL_OR_IDENTITY>','-o','UserKnownHostsFile=<REDACTED_SSH_CTL_OR_IDENTITY>','<REDACTED_USER_AT_HOST>']
env=os.environ.copy();env.update(SSH_ASKPASS='<REDACTED_SSH_CTL_OR_IDENTITY>',SSH_ASKPASS_REQUIRE='force',DISPLAY=':0')
last_events={};precheck_passed=False;last={};failures=0

def write_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def mirror_check(remote,final=False):
 issues=[];checked={}
 for rel,meta in remote['files'].items():
  if not rel.endswith('.pt'):continue
  f=M/rel
  info={'exists':f.exists()}
  if f.exists():
   try:
    with zipfile.ZipFile(f) as z:
     info['crc_ok']=z.testzip() is None
     info['members']=len(z.namelist())
    info['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    info['matches_remote_current']=info['sha256']==meta['sha256']
    info['mtime_lag_seconds']=max(0,meta['mtime']-f.stat().st_mtime)
    if not info['crc_ok'] or (final and not info['matches_remote_current']) or info['mtime_lag_seconds']>240:issues.append(rel+' incomplete/stale')
   except Exception as e:info['error']=str(e);issues.append(rel+': '+str(e))
  else:issues.append(rel+' missing')
  checked[rel]=info
 for sid,v in remote.get('epochs',{}).items():
  f=M/'ComCat'/'checkpoints'/(sid+'_epochs.jsonl')
  try:
   n=json.loads(f.read_text().splitlines()[-1])['epoch'];gap=v['epoch']-n
   if gap>3 or (final and gap!=0):issues.append(sid+' epoch backup gap '+str(gap))
  except Exception as e:issues.append(sid+' epoch-log missing/unreadable: '+str(e))
 sync_log=R/'sync.log'
 sync_done=sync_log.exists() and 'DONE marker and final artifacts pulled' in sync_log.read_text()
 sync_alive=subprocess.run(['pgrep','-f','^python3 -u /workspace/st-nkf-diagrams/comcat-fix-20261114/sync_checkpoints.py$'],capture_output=True).returncode==0
 if not sync_alive and not sync_done:issues.append('backup watcher not alive and not finalized')
 if final:
  for rel,meta in remote['files'].items():
   f=M/rel
   if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=meta['sha256']:issues.append('final file not identical: '+rel)
 return {'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S Asia/Shanghai'),'final':final,'ok':not issues,'issues':issues,'checkpoints':checked,'sync_alive':sync_alive,'sync_done':sync_done}

while True:
 now=datetime.now();verify=now>=PRECHECK or bool(last.get('done')) or bool(last and not last.get('training_tmux_alive',True))
 try:
  p=subprocess.run(ssh+['OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /hy-tmp/comcat-supervisor-status.py'+(' --verify' if verify else '')],env=env,capture_output=True,text=True,timeout=60)
  if p.returncode:raise RuntimeError('ssh exit '+str(p.returncode)+': '+p.stderr[-1200:])
  last=json.loads(p.stdout);failures=0
  write_json(R/'supervisor-latest.json',last)
  event={k:(v.get('status'),v.get('epochs_ran'),v.get('best_epoch'),v.get('test_LL_ts_total')) for k,v in last.get('schemes',{}).items()}
  if event!=last_events:
   print(now.isoformat(),'Asia/Shanghai SCHEME_RESULT',json.dumps(event,ensure_ascii=False),flush=True)
   with (R/'scheme-events.jsonl').open('a') as f:f.write(json.dumps({'time':last['time'],'schemes':last['schemes']},ensure_ascii=False)+'\n')
   last_events=event
  print(now.isoformat(),'Asia/Shanghai HEALTH',json.dumps({'online':True,'training_alive':last['training_tmux_alive'],'epochs':{s:v['epoch'] for s,v in last.get('epochs',{}).items()},'done':bool(last.get('done')),'alerts':last.get('alerts',[])},ensure_ascii=False),flush=True)
  finished=bool(last.get('done')) or not last['training_tmux_alive']
  if finished and not verify:
   # Next scheduled iteration performs full trusted checkpoint deserialization.
   time.sleep(5);continue
  if now>=PRECHECK and not precheck_passed:
   check=mirror_check(last,False);write_json(R/'predeadline-backup-verification.json',check)
   print('PREDEADLINE_BACKUP_VERIFICATION',json.dumps(check,ensure_ascii=False),flush=True)
   precheck_passed=check['ok']
  if finished:
   check=mirror_check(last,True);write_json(R/'final-backup-verification.json',check)
   if check['ok']:
    write_json(R/'FINAL-REPORT.json',{'state':last,'backup_verification':check,'predeadline_check_passed':precheck_passed})
    print('FINAL_REPORT',json.dumps({'state':last,'backup_verification':check},ensure_ascii=False),flush=True)
    break
   print('WAITING_FOR_EXISTING_SYNC',json.dumps(check['issues']),flush=True)
 except Exception as e:
  failures+=1
  print(now.isoformat(),'Asia/Shanghai SUPERVISOR_ERROR',type(e).__name__,str(e),flush=True)
  write_json(R/'supervisor-error.json',{'time':now.isoformat()+' Asia/Shanghai','consecutive_failures':failures,'error':str(e)})
 if datetime.now()>=DEADLINE:
  check=mirror_check(last,bool(last.get('done'))) if last.get('files') else {'ok':False,'issues':['no remote status']}
  write_json(R/'FINAL-REPORT.json',{'deadline_reached':True,'state':last,'backup_verification':check,'predeadline_check_passed':precheck_passed})
  print('DEADLINE_REPORT',json.dumps({'state':last,'backup_verification':check},ensure_ascii=False),flush=True)
  break
 time.sleep(min(120,max(1,(DEADLINE-datetime.now()).total_seconds())))
