import stub, json, os, sys, types, tempfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from profiles import Profile, SuppressionHandle, fingerprint
from gguf_utils import field_value

FAIL=[]
def check(name, fn):
    try:
        fn(); print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name); print(f"  FAIL  {name}: {type(e).__name__}: {e}")

cfg = types.SimpleNamespace(model_type="qwen3", num_hidden_layers=4,
    hidden_size=32, intermediate_size=64, num_attention_heads=4,
    vocab_size=1000, _name_or_path="fake/model")

class FakeModel:
    def __init__(s, cfg, layers=4, inter=64, vision=True):
        s.config=cfg; s._m={}
        for i in range(layers):
            s._m[f"model.layers.{i}.mlp.down_proj"]=torch.nn.Linear(32, inter)
        if vision:  # the trap the regex exists to avoid
            s._m["vision_tower.encoder.layers.0.mlp.down_proj"]=torch.nn.Linear(16,99)
    def named_modules(s): return list(s._m.items())

fp, geom = fingerprint(cfg)
prof = Profile.create(fp, geom, "fake/model", {"0":[1,2],"2":[5]},
                      scale=0.1, n_layers=4, n_neurons=64, config_name="t1")

print("\n[1] Profile carries the dims intervene_model.py reads")
check("n_neurons present", lambda: prof.data["n_neurons"])
check("n_layers present",  lambda: prof.data["n_layers"])
def intervene_contract():
    spec=prof.data
    by_layer={int(k):v for k,v in spec["by_layer"].items()}
    n=spec["n_neurons"]; total=sum(len(v) for v in by_layer.values())
    pct = total/(spec["n_layers"]*n)*100
    assert abs(pct - 3/256*100) < 1e-9, pct
check("intervene_model.py read path", intervene_contract)

print("\n[2] Round-trip through disk")
def roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = prof.save(d)
        assert os.path.basename(p)=="t1.json"
        back = Profile.load(p)
        assert back.by_layer == {0:[1,2],2:[5]}
        assert back.data["n_neurons"]==64
check("save/load", roundtrip)

print("\n[3] Geometry gate")
def rejects_mismatch():
    other = types.SimpleNamespace(**{**vars(cfg), "intermediate_size":128})
    m = FakeModel(other, inter=128)
    try:
        prof.check(m); raise AssertionError("accepted a mismatched model")
    except ValueError: pass
check("rejects wrong geometry", rejects_mismatch)
check("accepts matching geometry", lambda: prof.check(FakeModel(cfg)))

print("\n[4] Hook wiring")
def hooks():
    m = FakeModel(cfg)
    h = SuppressionHandle(m, prof)
    assert len(h._targets)==2, h._targets
    assert {l for l,_ in h._targets}=={0,2}
    # the vision tower must never be touched
    assert m._m["vision_tower.encoder.layers.0.mlp.down_proj"]._pre == []
    assert h.scale==0.1
    h.set_scale(0.5); assert h.scale==0.5
    h.remove()
    assert all(not mod._pre for mod in m._m.values())
check("hooks only text layers, vision untouched", hooks)
def bounds():
    try:
        Profile.create(fp, geom, "fake/model", {"0":[999]}, 0.1, 4, 64)
        raise AssertionError("accepted out-of-range neuron")
    except ValueError as e:
        assert "references neuron 999" in str(e), e
check("rejects out-of-range neuron index", bounds)
def negative_index():
    try:
        Profile.create(fp, geom, "fake/model", {"0":[-1]}, 0.1, 4, 64)
        raise AssertionError("accepted negative neuron index")
    except ValueError as e:
        assert "references neuron -1" in str(e), e
check("rejects negative neuron index", negative_index)
def duplicate_index():
    try:
        Profile.create(fp, geom, "fake/model", {"0":[1,1]}, 0.1, 4, 64)
        raise AssertionError("accepted duplicate neuron index")
    except ValueError as e:
        assert "duplicate neuron 1" in str(e), e
check("rejects duplicate neuron index", duplicate_index)
def bad_profile_dims():
    try:
        Profile.create(fp, geom, "fake/model", {"4":[1]}, 0.1, 4, 64)
        raise AssertionError("accepted out-of-range layer")
    except ValueError as e:
        assert "layer 4 is outside" in str(e), e
check("rejects out-of-range layer", bad_profile_dims)
def missing_layer():
    try:
        Profile.create(fp, geom, "fake/model", {"9":[1]}, 0.1, 4, 64)
        raise AssertionError("accepted a layer that does not exist")
    except ValueError as e:
        assert "layer 9 is outside" in str(e), e
check("rejects nonexistent layer", missing_layer)

print("\n[5] GGUF scalar/string field compatibility")
class FakeField:
    def __init__(self, value):
        self._value = value
        self.parts = {0: [value]}
        self.data = [0]
    def contents(self):
        return self._value
check("GGUF string field", lambda: (_ for _ in ()).throw(AssertionError())
      if field_value(FakeField(b"chat-template")) != "chat-template" else None)
check("GGUF numeric field", lambda: (_ for _ in ()).throw(AssertionError())
      if field_value(FakeField(32)) != 32 else None)

print("\n[6] Single source of truth for the layer regex")
def one_regex():
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "TEXT_LAYER_RE = ", str(ROOT / "scripts")],
        capture_output=True, text=True
    ).stdout
    assert out.count("\n")==1, out
check("TEXT_LAYER_RE defined once", one_regex)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
