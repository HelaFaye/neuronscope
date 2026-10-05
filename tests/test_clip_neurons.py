"""End-to-end CLIP H-Neuron pipeline on a tiny random CLIP (no downloads)."""
import json
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clip_neurons as cn  # noqa: E402


def tiny_clip(path: Path) -> Path:
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor,
                              CLIPTokenizer)
    chars = list(string.ascii_lowercase + string.digits + ".,-'{} ")
    vocab = {}
    for c in chars:
        vocab.setdefault(c, len(vocab))
        vocab.setdefault(c + "</w>", len(vocab))
    for special in ("<|startoftext|>", "<|endoftext|>"):
        vocab[special] = len(vocab)
    path.mkdir(parents=True)
    (path / "vocab.json").write_text(json.dumps(vocab))
    (path / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(path / "vocab.json"), str(path / "merges.txt"))
    img = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         vocab_size=len(vocab), max_position_embeddings=77,
                         bos_token_id=vocab["<|startoftext|>"], eos_token_id=vocab["<|endoftext|>"]),
        vision_config=dict(hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                           image_size=32, patch_size=8),
        projection_dim=16)
    CLIPModel(cfg).save_pretrained(path)
    CLIPProcessor(image_processor=img, tokenizer=tok).save_pretrained(path)
    return path


def make_images(root: Path, n=24):
    rng = np.random.default_rng(0)
    colours = {"red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 200, 30)}
    for name, rgb in colours.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(n):
            arr = np.clip(np.array(rgb) + rng.normal(0, 25, (40, 40, 3)), 0, 255).astype(np.uint8)
            PIL.fromarray(arr).save(d / f"{i}.png")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clip")
    model = tiny_clip(tmp / "model")
    make_images(tmp / "imgs")
    return tmp, model


def test_full_pipeline(setup):
    tmp, model = setup
    run = tmp / "run"
    assert cn.main(["collect", "--model", str(model), "--images", str(tmp / "imgs"), "--out", str(run),
                    "--views", "3", "--min_conf", "0", "--device", "cpu"]) == 0
    meta = json.loads((run / "run.json").read_text())
    assert meta["labels"] == ["blue", "green", "red"]
    assert sum(meta["stats"].values()) == 72
    train = json.loads((run / "train_qids.json").read_text())
    if not train["t"] or not train["f"]:
        pytest.skip("random tiny model produced a single verdict class")

    acts = tmp / "acts"
    assert cn.main(["extract", "--model", str(model), "--run", str(run), "--ids", str(run / "train_qids.json"),
                    "--out", str(acts), "--device", "cpu"]) == 0
    idx = json.loads((acts / "neuron_index.json").read_text())
    assert (idx["n_layers"], idx["n_neurons"], idx["tower"]) == (3, 48, "vision")
    one = np.load(next((acts / "image").glob("act_*.npy")))
    assert one.shape == (3, 48) and np.isfinite(one).all()

    out = tmp / "models"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "classifier.py"), "--acts_root", str(acts),
                        "--ans_dir", "image", "--train_mode", "1-vs-1", "--train_ids", str(run / "train_qids.json"),
                        "--C", "10", "--max_iter", "200", "--out_dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    hn = json.loads((out / "h_neurons.json").read_text())
    assert hn["tower"] == "vision" and hn["arch"] == "clip"
    if not hn["total"]:
        pytest.skip("classifier selected no neurons on random features")

    rep = tmp / "eval.json"
    assert cn.main(["evaluate", "--model", str(model), "--run", str(run), "--ids", str(run / "train_qids.json"),
                    "--h_neurons", str(out / "h_neurons.json"), "--scales", "1", "0", "--out", str(rep),
                    "--device", "cpu"]) == 0
    res = json.loads(rep.read_text())["scales"]
    assert set(res) == {"1.0", "0.0"} and res["1.0"]["n"] > 0

    edited = tmp / "edited"
    assert cn.main(["export", "--model", str(model), "--h_neurons", str(out / "h_neurons.json"),
                    "--scale", "0", "--out", str(edited)]) == 0
    from transformers import CLIPModel
    a, b = CLIPModel.from_pretrained(model), CLIPModel.from_pretrained(edited)
    layer, neurons = next(iter(hn["by_layer"].items()))
    wb = b.vision_model.encoder.layers[int(layer)].mlp.fc2.weight
    assert torch.all(wb[:, neurons] == 0)
    wa = a.vision_model.encoder.layers[int(layer)].mlp.fc2.weight
    keep = [j for j in range(wa.shape[1]) if j not in neurons]
    assert torch.equal(wa[:, keep], wb[:, keep])


def test_pre_hook_matches_weight_edit(setup):
    """Scaling fc2 inputs at runtime must equal scaling fc2 weight columns."""
    _, model = setup
    m, proc = cn.load(str(model))
    img = PIL.fromarray(np.full((40, 40, 3), 128, np.uint8))
    mgr = cn.ClipCETT(m, "vision")
    mgr.remove()
    prof = {0: [1, 5, 7], 2: [0, 3]}
    sc = cn.Scaler(mgr, prof)
    sc.scale = 0.3
    hooked = cn.image_embeddings(m, proc, [img], "cpu")
    sc.remove()
    with torch.no_grad():
        for l, idx in prof.items():
            mgr.modules[l].weight[:, idx] *= 0.3
    edited = cn.image_embeddings(m, proc, [img], "cpu")
    assert torch.allclose(hooked, edited, atol=1e-5)


def test_selective_metrics():
    p = np.array([[0.9, 0.1], [0.4, 0.6], [0.55, 0.45], [0.2, 0.8]])
    m = cn.selective_metrics(p, [0, 0, 1, 1], thr=0.7)
    assert m["accuracy"] == 0.5
    assert m["coverage"] == 0.5 and m["confident_error_rate"] == 0.0 and m["selective_accuracy"] == 1.0


def test_suppress_mmproj(tmp_path):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(1)
    fc1 = rng.normal(size=(48, 32)).astype(np.float16)   # out=48, in=32
    fc2 = rng.normal(size=(32, 48)).astype(np.float16)   # out=32, in=48 (the down projection)
    src = tmp_path / "mmproj.gguf"
    w = gguf.GGUFWriter(str(src), "clip")
    # Legacy naming: fc1 stored as ffn_down, fc2 as ffn_up. Shape detection must cope.
    w.add_tensor("v.blk.0.ffn_down.weight", fc1)
    w.add_tensor("v.blk.0.ffn_up.weight", fc2)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    hn = tmp_path / "h.json"
    hn.write_text(json.dumps({"n_neurons": 48, "tower": "vision", "by_layer": {"0": [2, 9]}}))
    import suppress_mmproj
    out = tmp_path / "out.gguf"
    assert suppress_mmproj.main(["--mmproj", str(src), "--h_neurons", str(hn), "--scale", "0.5",
                                 "--out", str(out)]) == 0
    t = {x.name: np.asarray(x.data) for x in gguf.GGUFReader(str(out)).tensors}
    got = t["v.blk.0.ffn_up.weight"].reshape(32, 48).astype(np.float32)
    assert np.allclose(got[:, [2, 9]], fc2[:, [2, 9]].astype(np.float32) * 0.5, atol=1e-3)
    assert np.array_equal(got[:, 3], fc2[:, 3].astype(np.float32))
    assert np.array_equal(t["v.blk.0.ffn_down.weight"].reshape(48, 32), fc1)


def test_clip_benchmark_harness(setup, tmp_path):
    """Runs CLIP_benchmark's own zero-shot code on our HF model, with and without H-Neuron scaling."""
    pytest.importorskip("clip_benchmark")
    import clip_bench
    tmp, model = setup
    prof = tmp_path / "h.json"
    prof.write_text(json.dumps({"n_neurons": 48, "tower": "vision", "by_layer": {"0": [1, 2, 3], "2": [5]}}))
    out = tmp_path / "bench.json"
    assert clip_bench.main(["eval", "--model", str(model), "--dataset", f"imagefolder:{tmp / 'imgs'}", "dummy",
                            "--dataset-root", str(tmp_path / "ds"), "--h_neurons", str(prof), "--scales", "1", "0",
                            "--batch-size", "16", "--num-workers", "0", "--device", "cpu", "--limit", "40",
                            "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    folder = rep["datasets"][f"imagefolder:{tmp / 'imgs'}"]
    assert folder["n_classes"] == 3 and set(folder["scales"]) == {"1.0", "0.0"}
    for m in folder["scales"].values():
        assert 0 <= m["top1"] <= m["top5"] <= 1 and 0 <= m["confident_error_rate"] <= 1
    # scaling the selected neurons to zero must change the logits
    assert folder["scales"]["1.0"] != folder["scales"]["0.0"]
    assert "dummy" in rep["datasets"]
    exp = tmp_path / "exported"
    assert clip_bench.main(["export", "--dataset", f"imagefolder:{tmp / 'imgs'}", "--per-class", "2",
                            "--out", str(exp)]) == 0
    assert sorted(p.name for p in exp.iterdir()) == ["blue", "green", "red"]
    assert all(len(list(d.iterdir())) == 2 for d in exp.iterdir())
