import sys, types
t = types.ModuleType("torch"); t.__path__=[]
class _T:
    def __init__(s, v): s.v=list(v)
    def to(s, *a, **k): return s
    def __iter__(s): return iter(s.v)
    def __len__(s): return len(s.v)
t.long=object()
t.tensor=lambda v, dtype=None, device=None: _T(v)
t.is_tensor=lambda x: isinstance(x,_T)
class Linear:
    def __init__(s, out, inp):
        s.in_features=inp; s.out_features=out
        s.weight=types.SimpleNamespace(device="cpu", shape=(out,inp))
        s._pre=[]
    def register_forward_pre_hook(s, fn):
        s._pre.append(fn)
        return types.SimpleNamespace(remove=lambda: s._pre.remove(fn))
t.nn=types.SimpleNamespace(Linear=Linear, Module=object)
t.norm=lambda *a,**k: None
sys.modules["torch"]=t
tr=types.ModuleType("transformers"); tr.AutoModelForCausalLM=object; tr.AutoTokenizer=object
sys.modules["transformers"]=tr
