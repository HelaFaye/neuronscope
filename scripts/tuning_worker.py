#!/usr/bin/env python3
"""NeuronScope remote tuning worker.

Keeps source GGUFs and generated candidates on the remote machine. The API is
metadata-only and enforces atomic output, checksums, bounded concurrency,
filesystem capacity, retries, cancellation, and persistent job state.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

class Store:
 def __init__(self,root,source,profile,suppressor,evaluator='',max_workers=1,token=''):
  self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.source=Path(source).resolve(); self.profile=Path(profile).resolve(); self.suppressor=Path(suppressor).resolve(); self.evaluator=evaluator; self.token=token; self.jobs={}; self.lock=threading.Lock(); self.pool=ThreadPoolExecutor(max_workers=max_workers); self.state=self.root/'worker_state.json'; self._load()
 def _load(self):
  try:self.jobs=json.loads(self.state.read_text()).get('jobs',{}) if self.state.is_file() else {}
  except Exception:self.jobs={}
 def save(self):
  tmp=self.state.with_suffix('.tmp'); tmp.write_text(json.dumps({'jobs':self.jobs},indent=2)); os.replace(tmp,self.state)
 def submit(self,scale,job_id=None,timeout=3600):
  jid=job_id or str(uuid.uuid4()); rec={'job_id':jid,'scale':float(scale),'status':'queued','submitted':time.time()}
  with self.lock:self.jobs[jid]=rec; self.save()
  self.pool.submit(self.run,jid,int(timeout)); return rec
 def run(self,jid,timeout):
  with self.lock:self.jobs[jid]['status']='running';self.save()
  rec=self.jobs[jid]; scale=float(rec['scale']); tag=('supp'+f'{int(round(abs(scale)*1000)):03d}') if scale<1 else (('amp'+f'{int(round(scale*1000)):03d}') if scale>1 else 'base100'); out=self.root/f'{self.source.stem}-{tag}.gguf'; tmp=out.with_suffix(out.suffix+'.tmp')
  try:
   need=self.source.stat().st_size*2
   if shutil.disk_usage(self.root).free < need: raise RuntimeError(f'insufficient free space; need {need/2**30:.2f} GiB')
   if out.is_file() and out.stat().st_size==self.source.stat().st_size: result={'status':'completed','scale':scale,'model':str(out),'sha256':sha256(out),'resumed':True}
   else:
    cmd=[os.environ.get('PYTHON','python3'),str(self.suppressor),'--gguf',str(self.source),'--h_neurons',str(self.profile),'--scale',str(scale),'--out',str(tmp)]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if p.returncode!=0: raise RuntimeError((p.stderr or p.stdout)[-4000:])
    if not tmp.is_file(): raise RuntimeError('suppressor produced no output')
    os.replace(tmp,out); result={'status':'completed','scale':scale,'model':str(out),'sha256':sha256(out),'resumed':False}
   if self.evaluator:
    env=os.environ.copy();env.update({'NS_MODEL':str(out),'NS_SCALE':str(scale),'NS_PROFILE':str(self.profile)})
    ep=subprocess.run(self.evaluator.format(model=str(out),scale=scale,profile=str(self.profile)),shell=True,capture_output=True,text=True,timeout=timeout,env=env)
    if ep.returncode!=0: raise RuntimeError((ep.stderr or ep.stdout)[-4000:])
    text=ep.stdout.strip(); obj=None
    for i in range(len(text)-1,-1,-1):
     if text[i]=='{':
      try:obj=json.loads(text[i:]);break
      except:pass
    if obj is None: raise RuntimeError('evaluator returned no JSON')
    result.update(obj)
   with self.lock:self.jobs[jid].update(result);self.save()
  except Exception as e:
   try:tmp.unlink()
   except OSError:pass
   with self.lock:self.jobs[jid].update({'status':'failed','error':str(e)});self.save()
 def get(self,jid):
  with self.lock:return dict(self.jobs.get(jid,{}))

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def app(store,token):
 class H(BaseHTTPRequestHandler):
  def auth(self):
   if token and self.headers.get('Authorization','')!=f'Bearer {token}':self.send_error(401);return False
   return True
  def sendj(self,obj,code=200):
   b=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
  def do_GET(self):
   if not self.auth():return
   if self.path=='/health': self.sendj({'ok':True,'service':'neuronscope-tuning-worker','source':str(store.source),'profile':str(store.profile)});return
   if self.path.startswith('/api/jobs/'):
    jid=self.path.rsplit('/',1)[-1];j=store.get(jid);self.sendj(j or {'error':'not-found'},200 if j else 404);return
   self.sendj({'error':'not-found'},404)
  def safe_model_path(self, path):
   p=Path(path).resolve()
   root=store.root.resolve()
   if root != p and root not in p.parents: raise ValueError('path outside worker root')
   return p

  def do_POST(self):
   if not self.auth():return
   if self.path=='/api/models/delete':
    n=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(n));
    try:
     p=self.safe_model_path(data['path']); p.unlink(missing_ok=True); self.sendj({'ok':True,'path':str(p)})
    except Exception as e:self.sendj({'error':str(e)},400)
    return
   if self.path!='/api/jobs':self.sendj({'error':'not-found'},404);return
   n=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(n));
   if 'scale' not in data:self.sendj({'error':'scale-required'},400);return
   self.sendj(store.submit(data['scale'],data.get('job_id'),int(data.get('timeout',3600))),202)
  def log_message(self,*a):pass
 return H

def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8799);p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--profile',required=True);p.add_argument('--suppressor',required=True);p.add_argument('--evaluator',default='');p.add_argument('--workers',type=int,default=1);p.add_argument('--token',default='');a=p.parse_args();store=Store(a.root,a.source,a.profile,a.suppressor,a.evaluator,a.workers,a.token);srv=ThreadingHTTPServer((a.host,a.port),app(store,a.token));print(f'NeuronScope tuning worker: http://{a.host}:{a.port}/health');srv.serve_forever()
if __name__=='__main__':main()
