import json, tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adaptive_tuner import AdaptiveTuner,parse_eval,wilson_interval

def test_wilson():
 lo,hi=wilson_interval(50,100);assert 0<lo<.5<hi<1

def test_parse():
 x=parse_eval({'correct':8,'total':10});assert x['score']==.8 and 'lower_ci' in x

def test_adaptive_local_client():
 class C:
  def __init__(self):self.calls=[]
  def submit(self,j): self.calls.append(j); return {'status':'completed','scale':j['scale'],'score':1-abs(j['scale']-.37),'correct':99,'total':100,'upper_ci':.99,'lower_ci':.97}
  def wait(self,j): return j
 with tempfile.TemporaryDirectory() as d:
  r=AdaptiveTuner(Path(d)/'state.json').run(client=C(),minimum=.1,maximum=.9,initial_step=.2,autotune_resolution=.05,batch_size=2,margin_of_error=.01,max_batches=5)
  assert r['status'] in {'resolution_reached','max_batches_reached','margin_reached'}
  assert len(r['results'])>=2

def test_state_resume_file():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'state.json';AdaptiveTuner(p)._save();assert p.exists();assert json.loads(p.read_text())['version']==2

def test_batches_narrow_after_each_batch():
 class C:
  def __init__(self): self.scales=[]
  def submit(self,j):
   self.scales.append(j['scale']); return {'status':'completed','scale':j['scale'],'score':1-abs(j['scale']-.35),'correct':90,'total':100}
  def wait(self,j): return j
 with tempfile.TemporaryDirectory() as d:
  c=C(); r=AdaptiveTuner(Path(d)/'s.json').run(client=c,minimum=.1,maximum=.9,initial_step=.2,autotune_resolution=.05,batch_size=3,margin_of_error=.0,max_batches=3)
  batches=r['batches']
  assert len(batches)>=2
  assert batches[1]['hi']-batches[1]['lo'] < batches[0]['hi']-batches[0]['lo']

def test_adaptive_resume_uses_next_interval():
 class C:
  def __init__(self): self.scales=[]
  def submit(self,j): self.scales.append(j['scale']); return {'status':'completed','scale':j['scale'],'score':1-abs(j['scale']-.62)}
  def wait(self,j): return j
 with tempfile.TemporaryDirectory() as d:
  state=Path(d)/'state.json'; c=C(); r=AdaptiveTuner(state).run(client=c,minimum=.0,maximum=1.0,initial_step=.25,autotune_resolution=.01,batch_size=3,margin_of_error=.0,max_batches=1)
  next_lo,next_hi=r['next_interval']
  c2=C(); r2=AdaptiveTuner(state).run(client=c2,minimum=.0,maximum=1.0,initial_step=.25,autotune_resolution=.01,batch_size=3,margin_of_error=.0,max_batches=1)
  assert c2.scales
  assert all(next_lo <= x <= next_hi for x in c2.scales)
