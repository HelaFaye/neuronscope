#!/usr/bin/env python3
"""Small browser GUI for adaptive remote/local tuning control."""
from __future__ import annotations
import argparse,json,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_tuner import AdaptiveTuner,WorkerClient
STATE={}
HTML=r"""<!doctype html><meta charset="utf-8"><title>NeuronScope Adaptive Tuning</title><style>body{font:14px system-ui;background:#101216;color:#eee;max-width:1100px;margin:auto;padding:20px}.c{background:#191c22;padding:16px;margin:12px 0;border-radius:12px}input,button{padding:8px;background:#0d1015;color:#eee;border:1px solid #39404a;border-radius:7px;margin:4px}pre{white-space:pre-wrap}canvas{width:100%;height:260px;background:#0c0f13;border-radius:8px}</style><h1>NeuronScope Adaptive Tuning</h1><div class=c><label>Worker <input id=w size=40 placeholder=http://remote:8799></label><label>Token <input id=tok size=28 type=password></label><br><label>Minimum <input id=lo value=.1 type=number step=.01></label><label>Maximum <input id=hi value=.9 type=number step=.01></label><label>Initial step <input id=st value=.1 type=number step=.01></label><label>Auto resolution <input id=res value=.01 type=number step=.001></label><label>Batch size <input id=bs value=2 type=number min=1></label><label>Margin of error <input id=moe value=.03 type=number step=.005></label><label>Baseline score <input id=base placeholder=optional type=number step=.001></label><label><input id=autodel type=checkbox> Auto-delete underperformers</label><br><button onclick=start()>Start</button></div><div class=c><canvas id=plot></canvas><pre id=o></pre></div><script>let state=null;async function start(){let b={worker:document.getElementById('w').value,min:+lo.value,max:+hi.value,initial_step:+st.value,resolution:+res.value,batch_size:+bs.value,margin_of_error:+moe.value,baseline_score:base.value===''?null:+base.value,auto_delete:autodel.checked,token:tok.value,state:'/tmp/neuronscope-adaptive-gui.json'};let r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});document.getElementById('o').textContent=await r.text();poll()}async function poll(){let r=await fetch('/api/state');state=await r.json();document.getElementById('o').textContent=JSON.stringify(state,null,2);draw();if(state.status==='running')setTimeout(poll,1000)}function draw(){let c=document.getElementById('plot'),x=c.getContext('2d'),w=c.width=c.clientWidth*devicePixelRatio,h=c.height=c.clientHeight*devicePixelRatio;x.clearRect(0,0,w,h);let rs=state.results||[];if(!rs.length)return;let min=Math.min(...rs.map(r=>r.scale)),max=Math.max(...rs.map(r=>r.scale)),sc=Math.max(1e-9,max-min),sy=Math.max(...rs.map(r=>r.score||0),1);x.strokeStyle='#65d4ff';x.beginPath();rs.forEach((r,i)=>{let px=(r.scale-min)/sc*w,py=h-(r.score/sy)*h;if(i)x.lineTo(px,py);else x.moveTo(px,py)});x.stroke()} </script>"""
class H(BaseHTTPRequestHandler):
 def sendj(self,o,c=200):
  b=json.dumps(o).encode();self.send_response(c);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  if self.path=='/':
   b=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  if self.path=='/api/state': self.sendj(STATE.get('state',{'status':'idle','results':[]}));return
  self.sendj({'error':'not-found'},404)
 def do_POST(self):
  if self.path!='/api/start':self.sendj({'error':'not-found'},404);return
  n=int(self.headers.get('Content-Length','0'));d=json.loads(self.rfile.read(n));
  if STATE.get('thread') and STATE['thread'].is_alive():self.sendj({'error':'already-running'},409);return
  STATE['state']={'status':'starting','results':[]}
  def run():
   try:
    t=AdaptiveTuner(d['state']);r=t.run(client=WorkerClient(d['worker'],token=d.get('token',''),cafile=d.get('cafile','')),minimum=float(d['min']),maximum=float(d['max']),initial_step=float(d['initial_step']),autotune_resolution=float(d['resolution']),batch_size=int(d['batch_size']),margin_of_error=float(d['margin_of_error']),baseline_score=d.get('baseline_score'),max_batches=40,auto_delete=bool(d.get('auto_delete')));STATE['state']=r
   except Exception as e:STATE['state']={'status':'failed','error':str(e),'results':[]}
  th=threading.Thread(target=run,daemon=True);STATE['thread']=th;th.start();self.sendj({'ok':True})
 def log_message(self,*a):pass
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8800);a=p.parse_args();s=ThreadingHTTPServer((a.host,a.port),H);print(f'NeuronScope Adaptive Tuning: http://{a.host}:{a.port}/');s.serve_forever()
if __name__=='__main__':main()
