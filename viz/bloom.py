#!/usr/bin/env python3
"""
NeuronScope frontend backend, and the three.js client.

This serves the API that all three frontends consume:

    GET /api/meta     frames, cells, layers, per-frame z, token labels, flagged
    GET /api/theme    the selected theme, from themes.json
    GET /api/trace    binary: i32 T, N, L | i32 layer[N] | i32 neuron[N]
                              | f32 intensity[T][N] | u8 state[T][N]

State is 0 idle, 1 active, 2 flagged, decided here in Python. The clients only
draw. That is what makes parity between pygfx, Godot and three.js maintainable:
one implementation of the thresholding, the H-Neuron mapping and the scoring,
three renderers. It also keeps the GPU free -- classification is CPU work done
once at load, not per frame on the device running the model.

The page served at / is the three.js client: UnrealBloomPass, additive
blending, depth fog.

viz/timeline.py is the analytical view -- pygfx, exact point sizes, no effects.
This is its sibling for showing people. three.js with UnrealBloomPass, additive
blending, depth fog. Same records, same themes, same flagging logic; different
job.

Keep both. Bloom is actively bad for diagnosis: it blooms neighbours together,
so a bright cell reads as a smear and you lose the spatial precision that made
the view worth having. It is excellent for a screenshot, a talk, or seeing the
shape of a burst at a glance. Use timeline.py to decide anything.

This does not need ROCm. ROCm is compute; a Vega iGPU does Vulkan graphics
fine, which is what the browser uses. With no GPU at all, Chrome falls back to
SwiftShader and it still runs, slowly.

    python viz/bloom.py runs/trace-abc
    python viz/bloom.py --demo            # synthetic trace: try the clients without a model
    python viz/bloom.py runs/trace-abc --theme cool --port 7880 --host 0.0.0.0

Serves a page; open the URL it prints. Positions are sent once and only
intensities stream per frame, so a 40-frame trace is a few hundred kilobytes
rather than a few hundred megabytes.
"""

import argparse
import json
import os
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ns_security as sec  # noqa: E402
from records import Session  # noqa: E402
from timeline import THEMES, classify_frames, load_mask, load_trace  # noqa

PAYLOAD = {}


DEMO_TEXT = ("The Eiffel Tower was completed in 1889 and stands in Paris . It was designed by "
             "the engineer Gustave Eiffel , and in 1923 it was moved to Lyon for the World Fair , "
             "where it remained until 1931 .").split()


def demo_trace(T=160, L=32, N=512, seed=0):
    """A synthetic trace for trying the clients without a model: a quiet field
    with a few dozen designated neurons that burst during an invented claim
    ("moved to Lyon"). Nothing here was measured; the HUD says "demo"."""
    rng = np.random.default_rng(seed)
    frames = rng.gamma(2.0, 0.02, size=(T, L, N)).astype(np.float32)
    frames *= (1 + 0.6 * np.sin(np.linspace(0, 3.1, L)))[None, :, None]   # mid layers busier
    hl = rng.integers(L // 3, L, 24)
    hn = rng.integers(0, N, 24)
    words = [DEMO_TEXT[i % len(DEMO_TEXT)] for i in range(T)]
    burst = np.zeros(T, np.float32)
    for i, w in enumerate(words):
        if w in ("moved", "to", "Lyon", "1923", "1931"):
            burst[i] = 1.0
    burst = np.convolve(burst, [0.3, 1.0, 0.6], "same")
    frames[:, hl, hn] += (0.05 + 1.2 * burst)[:, None] * rng.uniform(0.6, 1.0, 24)[None, :]
    mask = np.zeros((L, N), bool)
    mask[hl, hn] = True
    scores = frames[:, hl, hn].mean(1)
    fields = {"scores": scores.tolist(), "pieces": [" " + w for w in words], "stride": 1,
              "meta": {"model": "demo (synthetic, not measured)"}, "n_frames": T, "n_layers": L}
    return frames, fields, mask


def build_payload(session, h_neurons, active_pct, score_z, max_cells, demo=False):
    if demo:
        frames, fields, mask = demo_trace()
        T, L, N = frames.shape
    else:
        frames, fields = load_trace(session)
        T, L, N = frames.shape
        mask = load_mask(h_neurons, (L, N))
    active, halluc, z = classify_frames(frames, fields, mask,
                                        active_pct, score_z)

    # Only cells that are ever active get geometry. At ~97% inactive, sending
    # the rest would be almost entirely wasted bandwidth and fill rate.
    ever = active.any(axis=0)
    ly, nx = np.nonzero(ever)
    if len(ly) > max_cells:
        # Keep the strongest by peak intensity rather than truncating
        # arbitrarily, so the visible structure survives the cap.
        peak = frames[:, ly, nx].max(axis=0)
        keep = np.argsort(-peak)[:max_cells]
        ly, nx = ly[keep], nx[keep]
    n = len(ly)

    vmax = float(np.percentile(frames, 99.5)) or 1.0
    inten = np.clip(frames[:, ly, nx] / vmax, 0, 1).astype(np.float32)
    # State per cell per frame: 0 idle, 1 active, 2 flagged.
    state = np.zeros((T, n), dtype=np.uint8)
    state[active[:, ly, nx]] = 1
    state[halluc[:, ly, nx]] = 2

    blob = bytearray()
    blob += struct.pack("<iii", T, n, L)
    blob += ly.astype("<i4").tobytes()
    blob += nx.astype("<i4").tobytes()
    blob += inten.tobytes()
    blob += state.tobytes()

    pieces = fields.get("pieces") or []
    stride = int(fields.get("stride", 1))
    labels = ["".join(pieces[i * stride:(i + 1) * stride]) for i in range(T)] \
        if pieces else [str(i) for i in range(T)]
    return bytes(blob), {
        "frames": T, "cells": n, "layers": L, "neurons": N,
        "z": [round(float(v), 3) for v in z], "labels": labels,
        "flagged": [int(i) for i in np.where(halluc.any(axis=(1, 2)))[0]],
        "model": fields.get("meta", {}).get("model"),
        "verdict": fields.get("verdict"),
        "question": (fields.get("question") or "")[:160],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/meta":
            return self._send(200, json.dumps(PAYLOAD["meta"]),
                              "application/json")
        if self.path == "/api/trace":
            return self._send(200, PAYLOAD["blob"], "application/octet-stream")
        if self.path == "/api/theme":
            return self._send(200, json.dumps(PAYLOAD["theme"]),
                              "application/json")
        self._send(404, json.dumps({"error": "not found"}), "application/json")


PAGE = r"""<!DOCTYPE html><meta charset="utf-8"><title>NeuronScope Bloom</title>
<style>
html,body{margin:0;height:100%;background:#05050a;color:#c9c9d2;
font:13px ui-sans-serif,system-ui,sans-serif;overflow:hidden}
#c{position:absolute;inset:0}
#hud{position:absolute;left:14px;top:12px;z-index:2;line-height:1.6;
text-shadow:0 1px 3px #000}
#hud b{font-weight:500;color:#fff}
#bar{position:absolute;left:0;right:0;bottom:0;z-index:2;padding:10px 14px;
background:linear-gradient(transparent,#05050ae0 40%);display:flex;gap:12px;align-items:center}
#track{flex:1;height:4px;background:#2a2a34;border-radius:2px;position:relative;cursor:pointer}
#fill{position:absolute;left:0;top:0;height:4px;border-radius:2px}
#tok{font-family:ui-monospace,monospace;min-width:150px}
button{background:transparent;border:1px solid #3a3a46;color:#c9c9d2;
border-radius:6px;padding:5px 12px;cursor:pointer;font:inherit}
button:hover{background:#16161e}
.warn{position:absolute;inset:0;display:flex;align-items:center;
justify-content:center;text-align:center;padding:2rem;z-index:3}
</style>
<canvas id="c"></canvas>
<div id="hud"></div>
<div id="bar"><button id="play">Pause</button>
<div id="track"><div id="fill"></div></div><div id="tok"></div></div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script>
// The renderer comes from unpkg. If it cannot load (offline, a blocking proxy),
// the module below never runs; say so instead of leaving a black page.
setTimeout(()=>{ if(!window.__nsReady){ const w=document.createElement('div'); w.className='warn';
  w.innerHTML='Could not load three.js from unpkg.com.<br>This viewer needs that CDN, or use the Godot '+
    'client (viz/godot) or viz/timeline.py, which work offline.'; document.body.appendChild(w); } }, 10000);
</script>
<script type="module">
import * as THREE from 'three';
window.__nsReady = true;
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';

const meta = await (await fetch('/api/meta')).json();
const theme = await (await fetch('/api/theme')).json();
const buf = await (await fetch('/api/trace')).arrayBuffer();

const dv = new DataView(buf);
const T = dv.getInt32(0,true), N = dv.getInt32(4,true), L = dv.getInt32(8,true);
let o = 12;
const ly = new Int32Array(buf, o, N); o += N*4;
const nx = new Int32Array(buf, o, N); o += N*4;
const inten = new Float32Array(buf, o, T*N); o += T*N*4;
const state = new Uint8Array(buf, o, T*N);

const hex = s => new THREE.Color(s);
const cIdle = hex('#2a2a3a'), cAct = hex(theme.active), cHal = hex(theme.halluc);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05050a, 0.0016);
const cam = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 6000);
// Fit the view to this trace: neuron columns span W units whatever the trace's
// width (512 bins or 14336 neurons), layers are 8 units apart.
const W = 640, XS = W/Math.max(1, meta.neurons), H = L*8;
const D = 0.65*Math.max(W, H);
cam.position.set(W/2 + 0.35*D, H/2 + 0.3*D, 0.95*D);
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
renderer.setSize(innerWidth, innerHeight);
const controls = new OrbitControls(cam, canvas);
controls.enableDamping = true;
controls.target.set(W/2, H/2, 0);

const pos = new Float32Array(N*3), col = new Float32Array(N*3), siz = new Float32Array(N);
for(let i=0;i<N;i++){ pos[i*3]=nx[i]*XS; pos[i*3+1]=ly[i]*8; pos[i*3+2]=0; }
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
geo.setAttribute('color', new THREE.BufferAttribute(col,3));
geo.setAttribute('size', new THREE.BufferAttribute(siz,1));
const mat = new THREE.ShaderMaterial({
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
  vertexShader:`attribute float size; varying vec3 vC;
    void main(){ vC=color; vec4 mv=modelViewMatrix*vec4(position,1.0);
    gl_PointSize=size*(300.0/-mv.z); gl_Position=projectionMatrix*mv; }`,
  fragmentShader:`varying vec3 vC;
    void main(){ vec2 d=gl_PointCoord-vec2(0.5); float r=length(d);
    if(r>0.5) discard; float a=smoothstep(0.5,0.0,r);
    gl_FragColor=vec4(vC, a); }`,
  vertexColors:true});
scene.add(new THREE.Points(geo, mat));

// Trail: earlier frames persist as a dim wake so a burst reads as motion
// rather than a single flash you can miss between frames.
const trailN = 6, trails = [];
for(let k=0;k<trailN;k++){
  const g = geo.clone();
  const m = mat.clone(); m.opacity = 0.5*(1-k/trailN);
  const p = new THREE.Points(g, m); scene.add(p); trails.push(p);
}

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),
                                  1.15, 0.55, 0.12);
composer.addPass(bloom);

const tmp = new THREE.Color();
// The current token sits at z=0; trail k (an earlier token) sits k+1 steps behind it,
// so the field stays in view however long the trace is.
function paint(geometry, t, dim, dz){
  const c = geometry.getAttribute('color'), s = geometry.getAttribute('size');
  const p = geometry.getAttribute('position');
  const base = t*N;
  for(let i=0;i<N;i++){
    const v = inten[base+i], st = state[base+i];
    tmp.copy(st===2?cHal:st===1?cAct:cIdle);
    const g = (st===0?0.35:0.6+0.8*v)*dim;
    c.array[i*3]=tmp.r*g; c.array[i*3+1]=tmp.g*g; c.array[i*3+2]=tmp.b*g;
    s.array[i] = (st===2?4.2:st===1?2.4:1.1)*(0.6+0.9*v);
    p.array[i*3+2] = dz;
  }
  c.needsUpdate=true; s.needsUpdate=true; p.needsUpdate=true;
}

let t=0, playing=true, acc=0;
const hud=document.getElementById('hud'), tok=document.getElementById('tok');
const fill=document.getElementById('fill'); fill.style.background=theme.halluc;
document.getElementById('play').onclick=e=>{playing=!playing;
  e.target.textContent=playing?'Pause':'Play';};
document.getElementById('track').onclick=e=>{
  const r=e.currentTarget.getBoundingClientRect();
  t=Math.round((e.clientX-r.left)/r.width*(T-1)); update();};

function update(){
  paint(geo, t, 1.0, 0);
  for(let k=0;k<trailN;k++) paint(trails[k].geometry, Math.max(0,t-(k+1)), 0.45, -(k+1)*6);
  const flagged = meta.flagged.includes(t);
  hud.innerHTML = `<b>${meta.model||'trace'}</b><br>`+
    `${meta.frames} tokens · ${meta.cells.toLocaleString()} cells · ${meta.layers} layers<br>`+
    `score z <b>${meta.z[t].toFixed(2)}</b>`+
    (flagged?` · <span style="color:${theme.halluc}">flagged</span>`:'');
  tok.innerHTML = `${t+1}/${T} ` +
    `<span style="color:${flagged?theme.halluc:'#8a8a92'}">${
      (meta.labels[t]||'').replace(/</g,'&lt;')||'·'}</span>`;
  fill.style.width = (100*t/(T-1))+'%';
  bloom.strength = flagged ? 1.9 : 1.15;
}
update();

renderer.setAnimationLoop(dt=>{
  if(playing){ acc++; if(acc%8===0){ t=(t+1)%T; update(); } }
  controls.update(); composer.render();
});
addEventListener('resize', ()=>{
  cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  composer.setSize(innerWidth,innerHeight);
});
</script>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?", help="trace session from scripts/trace_sample.py")
    p.add_argument("--demo", action="store_true",
                   help="a synthetic trace, to try the three.js and Godot clients without a model")
    p.add_argument("--theme", default="ember", choices=sorted(THEMES))
    p.add_argument("--h-neurons")
    p.add_argument("--active-pct", type=float, default=97.0)
    p.add_argument("--score-z", type=float, default=1.0)
    p.add_argument("--max-cells", type=int, default=120000,
                   help="cap on instantiated points; the strongest are kept")
    p.add_argument("--port", type=int, default=7880)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dump", metavar="BIN", help="write the payload and exit")
    p.add_argument("--allow-unauthenticated", action="store_true", help="permit a non-loopback bind (no auth)")
    a = p.parse_args()
    if not a.session and not a.demo:
        p.error("give a trace session, or --demo")
    if not a.dump:
        sec.loopback_only(a.host, "the trace viewer", a.allow_unauthenticated)

    blob, meta = build_payload(a.session, a.h_neurons, a.active_pct,
                               a.score_z, a.max_cells, demo=a.demo)
    PAYLOAD.update({"blob": blob, "meta": meta, "theme": THEMES[a.theme]})
    print(f"{meta['frames']} frames, {meta['cells']} cells, "
          f"{len(blob) / 1e6:.1f} MB payload, theme '{a.theme}'")
    if meta["flagged"]:
        print(f"flagged frames: {meta['flagged'][:12]}"
              f"{' …' if len(meta['flagged']) > 12 else ''}")

    if a.dump:
        with open(a.dump, "wb") as f:
            f.write(blob)
        with open(a.dump + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {a.dump} and {a.dump}.json")
        return

    print(f"\nopen http://{a.host}:{a.port}  (three.js client)")
    print(f"Godot client: set NS_API=http://{a.host}:{a.port} and run "
          "viz/godot/")
    print("This is the presentation view. Bloom blurs neighbours together, so "
          "use\nviz/timeline.py to decide anything.")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
