"""Pull only this ComCat rerun, every 90s, through prepaid expiry; no remote writes."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
DEST=Path('/workspace/st-nkf-diagrams')
ssh=['ssh','-S','<REDACTED_SSH_CTL_OR_IDENTITY>','-p','13953','-o','ConnectTimeout=12','-o','HostKeyAlias=<REDACTED_SSH_CTL_OR_IDENTITY>','-o','UserKnownHostsFile=<REDACTED_SSH_CTL_OR_IDENTITY>','<REDACTED_USER_AT_HOST>']
scp=['scp','-rp','-o','ControlPath=<REDACTED_SSH_CTL_OR_IDENTITY>','-o','ConnectTimeout=12','-o','HostKeyAlias=<REDACTED_SSH_CTL_OR_IDENTITY>','-o','UserKnownHostsFile=<REDACTED_SSH_CTL_OR_IDENTITY>','-P','13953','<REDACTED_USER_AT_HOST>:/hy-tmp/st-nkf-comcat-20261114',str(DEST)]
env=os.environ.copy();env.update(SSH_ASKPASS='<REDACTED_SSH_CTL_OR_IDENTITY>',SSH_ASKPASS_REQUIRE='force',DISPLAY=':0')
deadline=datetime.now().replace(hour=12,minute=39,second=0,microsecond=0)
while True:
 print(datetime.now().isoformat(), 'Asia/Shanghai sync starting',flush=True)
 try:
  p=subprocess.run(scp,env=env,capture_output=True,text=True,timeout=90)
  print('sync exit',p.returncode,p.stderr[-1200:],flush=True)
  done=subprocess.run(ssh+['test -f /hy-tmp/st-nkf-comcat-20261114/DONE'],env=env,capture_output=True,timeout=20).returncode==0
  if done and p.returncode==0:
   print('DONE marker and final artifacts pulled',flush=True);break
 except Exception as e: print(type(e).__name__,str(e),flush=True)
 if datetime.now()>=deadline: print('pre-expiry final sync attempted',flush=True);break
 time.sleep(min(90,max(1,(deadline-datetime.now()).total_seconds())))
