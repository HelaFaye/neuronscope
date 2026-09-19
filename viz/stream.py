#!/usr/bin/env python3
"""
Near-realtime activation streaming, for a phone or SBC on your network.

The fork this expects does not exist yet: `llama-server --activations` would
reduce in-process and emit alongside the token stream. The eval callback
cett-dump already uses fires during generation as well as prefill, so the C++
side is a known quantity. What is not obvious is the protocol, and that is what
this implements and tests -- against a simulated source now, against the fork
later, without the client changing.

Bandwidth is the whole design. On a 36-layer, 14336-wide model:

    raw                 1.03 MB/token   hopeless over wifi
    512 neuron bins       36 KB/token   fine on a LAN
    score + top-K cells  ~1.6 KB/token  fine over LTE

So the reduction happens server-side and the client chooses a tier. A phone
asks for `sparse`; a desktop on the same switch asks for `binned`.

    python viz/stream.py --simulate --token secret --host 0.0.0.0
    python viz/stream.py --source http://127.0.0.1:8080 --token secret \\
        --tls-cert cert.pem --tls-key key.pem --host 0.0.0.0

Open the printed URL on the phone. Auth is a bearer token or cookie; TLS is
optional and self-signed certs will warn. On a network you do not control,
prefer a WireGuard or Tailscale tunnel and plain HTTP over a self-signed cert
the user is trained to click through.
"""

import argparse
import hashlib
import hmac
import http.cookies
import json
import math
import os
import queue
import ssl
import urllib.request
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"token": None, "tier": "binned", "source": None, "simulate": False}
SUBS = []
SUBS_LOCK = threading.Lock()

TIERS = {
    # name: (description, bytes/token on a 36x14336 model)
    "raw": ("every neuron, every layer", 36 * 14336 * 2),
    "binned": ("512 neuron bins per layer", 36 * 512 * 2),
    "sparse": ("score plus the top-K active cells", 1600),
}


def reduce_frame(cett, tier, top_k=48, bins=512, score=0.0):
    """Server-side reduction. `cett` is [layers, neurons] for one token.

    Binning max-pools rather than means: the cells worth watching are a tiny
    fraction of the field, and averaging over 28 neurons per bin erases them.
    """
    L = len(cett)
    if tier == "raw":
        return {"t": "raw", "s": round(score, 4),
                "v": [[round(float(x), 4) for x in row] for row in cett]}

    if tier == "binned":
        out = []
        for row in cett:
            n = len(row)
            step = max(1, n // bins)
            out.append([round(float(max(row[i:i + step])), 4)
                        for i in range(0, n, step)][:bins])
        return {"t": "binned", "s": round(score, 4), "v": out}

    # sparse: only the strongest cells, as (layer, neuron, value) triples
    flat = []
    for li, row in enumerate(cett):
        for ni, v in enumerate(row):
            flat.append((float(v), li, ni))
    flat.sort(reverse=True)
    return {"t": "sparse", "s": round(score, 4), "l": L,
            "c": [[li, ni, round(v, 4)] for v, li, ni in flat[:top_k]]}


def publish(frame):
    """Fan out to every subscriber, dropping for anyone who cannot keep up.

    A slow phone must not stall the stream for a desktop on the same server,
    and a stalled stream would eventually block generation itself.
    """
    dead = []
    with SUBS_LOCK:
        for q in SUBS:
            try:
                q.put_nowait(frame)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                q.put_nowait({"drop": True})
            except queue.Full:
                pass


def relay(source):
    """Pull /activations from a forked llama-server into our fan-out.

    Frames arrive already reduced by the server, so nothing is re-reduced here;
    this only adds the auth, TLS and backpressure the C++ side does not do.
    Reconnects, because the server restarts more often than the viewer does.
    """
    url = source.rstrip("/") + "/activations"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=None) as r:
                print(f"[relay] connected to {url}")
                for line in r:
                    s = line.decode(errors="replace").strip()
                    if not s.startswith("data:"):
                        continue
                    try:
                        publish(json.loads(s[5:]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[relay] {type(e).__name__}: {e}; retrying in 3s")
            time.sleep(3)


def simulate(layers=36, neurons=14336, rate=4.0):
    """Stand-in for the fork: a plausible activation field at 4 tokens/sec."""
    import random
    rng = random.Random(0)
    hot = [(rng.randrange(layers), rng.randrange(neurons)) for _ in range(6)]
    t = 0
    while True:
        burst = 40 <= (t % 90) <= 46
        cett = []
        for li in range(layers):
            row = [rng.gammavariate(2.0, 0.02) for _ in range(0, neurons, 28)]
            cett.append(row)
        if burst:
            for li, ni in hot:
                cett[li][min(ni // 28, len(cett[li]) - 1)] += 1.4
        score = 2.8 if burst else rng.gauss(0.0, 0.4)
        publish({**reduce_frame(cett, STATE["tier"], score=score),
                 "i": t, "tok": " Golgi" if burst else f" w{t}"})
        t += 1
        time.sleep(1.0 / rate)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _authed(self):
        tok = STATE["token"]
        if not tok:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Bearer ") and hmac.compare_digest(h[7:], tok):
            return True
        raw = self.headers.get("Cookie", "")
        if raw:
            try:
                c = http.cookies.SimpleCookie(raw)
                if "ns_token" in c and hmac.compare_digest(c["ns_token"].value,
                                                           tok):
                    return True
            except Exception:
                pass
        # A token in the query string is accepted only for the stream itself:
        # EventSource cannot set headers, and this is the standard workaround.
        if self.path.startswith("/stream?") and "token=" in self.path:
            q = self.path.split("token=", 1)[1].split("&")[0]
            if hmac.compare_digest(q, tok):
                return True
        return False

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authed():
            if self.path == "/":
                return self._send(401, LOGIN, "text/html; charset=utf-8")
            return self._send(401, json.dumps({"error": "unauthorized"}))
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/tiers":
            return self._send(200, json.dumps(
                {k: {"desc": v[0], "bytes_per_token": v[1]}
                 for k, v in TIERS.items()} | {"current": STATE["tier"]}))
        if self.path.startswith("/stream"):
            return self._stream()
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/login":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if STATE["token"] and hmac.compare_digest(str(req.get("token", "")),
                                                      STATE["token"]):
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie",
                                 f"ns_token={STATE['token']}; Path=/; "
                                 "HttpOnly; SameSite=Strict; Max-Age=604800")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(401, json.dumps({"error": "bad token"}))
            return
        if not self._authed():
            return self._send(401, json.dumps({"error": "unauthorized"}))
        if self.path == "/api/push":
            # scripts/autotrace.py posts already-reduced frames here, so a
            # trace produced anywhere on the network can drive this viewer.
            n = int(self.headers.get("Content-Length", 0))
            try:
                publish(json.loads(self.rfile.read(n) or b"{}"))
            except json.JSONDecodeError:
                return self._send(400, json.dumps({"error": "bad json"}))
            return self._send(200, json.dumps({"ok": True}))
        if self.path == "/api/tier":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if req.get("tier") in TIERS:
                STATE["tier"] = req["tier"]
                return self._send(200, json.dumps({"ok": True,
                                                   "tier": STATE["tier"]}))
            return self._send(400, json.dumps({"error": "unknown tier"}))
        self._send(404, json.dumps({"error": "not found"}))

    def _stream(self):
        q = queue.Queue(maxsize=8)
        with SUBS_LOCK:
            SUBS.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                try:
                    frame = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")   # hold the connection
                    self.wfile.flush()
                    continue
                self.wfile.write(
                    f"data: {json.dumps(frame, separators=(',', ':'))}\n\n"
                    .encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with SUBS_LOCK:
                if q in SUBS:
                    SUBS.remove(q)


LOGIN = """<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuronScope</title><style>body{font:16px system-ui;background:#0b0b10;
color:#c9c9d2;display:flex;align-items:center;justify-content:center;height:100vh;
margin:0}form{width:300px}input{width:100%;padding:.8rem;font-size:16px;
border:1px solid #2a2a34;border-radius:8px;background:#14141c;color:#fff}
button{width:100%;margin-top:.6rem;padding:.8rem;font-size:16px;border:0;
border-radius:8px;background:#2c5f8a;color:#fff}</style>
<form onsubmit="event.preventDefault();fetch('/api/login',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t.value})})
.then(r=>r.ok?location.reload():alert('Incorrect token'))">
<input id="t" type="password" placeholder="access token" autofocus>
<button>Unlock</button></form>"""

PAGE = r"""<!DOCTYPE html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>NeuronScope live</title>
<style>
:root{--hot:#F09595;--act:#EF9F27}
*{box-sizing:border-box}
body{margin:0;background:#05050a;color:#c9c9d2;font:14px system-ui;
overscroll-behavior:none;-webkit-text-size-adjust:100%}
header{padding:10px 14px;border-bottom:1px solid #1c1c24;display:flex;
gap:10px;align-items:center;position:sticky;top:0;background:#05050a;z-index:2}
.dot{width:8px;height:8px;border-radius:50%;background:#444}
.dot.on{background:#1D9E75}
#cv{width:100%;height:44vh;display:block;background:#05050a;touch-action:pan-y}
#sc{width:100%;height:64px;display:block}
.bar{display:flex;gap:8px;padding:10px 14px;align-items:center;flex-wrap:wrap}
select,button{font:14px system-ui;padding:8px 12px;background:#14141c;
color:#c9c9d2;border:1px solid #2a2a34;border-radius:8px;min-height:40px}
.stat{font:12px ui-monospace,monospace;color:#8a8a92;padding:0 14px 10px;
line-height:1.7}
.tok{font:13px ui-monospace,monospace;padding:0 14px 12px;word-break:break-all}
.tok b{color:var(--hot)}
</style>
<header><span class="dot" id="d"></span><b id="st">connecting…</b>
<span style="margin-left:auto;font:12px ui-monospace,monospace" id="rate"></span>
</header>
<canvas id="cv"></canvas>
<canvas id="sc"></canvas>
<div class="bar">
<select id="tier"></select>
<button id="pause">Pause</button>
<button id="clear">Clear</button>
</div>
<div class="stat" id="stat"></div>
<div class="tok" id="toks"></div>
<script>
const $=s=>document.querySelector(s);
const cv=$('#cv'), cx=cv.getContext('2d'), sc=$('#sc'), sx=sc.getContext('2d');
let paused=false, frames=[], toks=[], last=0, bytes=0, t0=Date.now();
function fit(c){const r=devicePixelRatio||1;c.width=c.clientWidth*r;
  c.height=c.clientHeight*r;c.getContext('2d').setTransform(r,0,0,r,0,0);}
addEventListener('resize',()=>{fit(cv);fit(sc);draw();}); fit(cv); fit(sc);

fetch('/api/tiers').then(r=>r.json()).then(t=>{
  $('#tier').innerHTML=Object.entries(t).filter(([k])=>k!=='current')
    .map(([k,v])=>`<option value="${k}"${k===t.current?' selected':''}>${k} — ${
      (v.bytes_per_token/1024).toFixed(1)} KB/tok</option>`).join('');
});
$('#tier').onchange=e=>fetch('/api/tier',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({tier:e.target.value})});
$('#pause').onclick=()=>{paused=!paused;$('#pause').textContent=paused?'Resume':'Pause';};
$('#clear').onclick=()=>{frames=[];toks=[];draw();};

// EventSource cannot set an Authorization header, so the cookie carries auth
// for same-origin; the query fallback exists for clients without cookies.
const es=new EventSource('/stream');
es.onopen=()=>{$('#d').className='dot on';$('#st').textContent='streaming';};
es.onerror=()=>{$('#d').className='dot';$('#st').textContent='disconnected';};
es.onmessage=e=>{
  bytes+=e.data.length;
  if(paused) return;
  const f=JSON.parse(e.data);
  if(f.drop){ $('#st').textContent='streaming (dropped frames)'; return; }
  frames.push(f); if(frames.length>160) frames.shift();
  if(f.tok!==undefined){ toks.push({s:f.s,tok:f.tok});
    if(toks.length>60) toks.shift(); }
  draw();
};
setInterval(()=>{const kb=bytes/1024, s=(Date.now()-t0)/1000;
  $('#rate').textContent=`${(kb/s).toFixed(1)} KB/s`;},1000);

function cells(f){
  if(f.t==='sparse') return f.c.map(([l,n,v])=>({l,n,v,L:f.l}));
  const out=[]; f.v.forEach((row,l)=>row.forEach((v,n)=>{
    if(v>0.25) out.push({l,n,v,L:f.v.length,N:row.length});}));
  return out;
}
function draw(){
  const w=cv.clientWidth, h=cv.clientHeight;
  cx.clearRect(0,0,w,h);
  const n=frames.length; if(!n) return;
  frames.forEach((f,i)=>{
    const age=(n-1-i)/Math.max(n-1,1);
    const x0=w*(1-age);
    const cs=cells(f); const L=cs.length?(cs[0].L||36):36;
    const N=cs.length?(cs[0].N||14336):14336;
    cs.forEach(c=>{
      const y=h-(c.l/L)*h*0.92-6;
      const jitter=((c.n%97)/97-0.5)*10;
        // Without a classifier `s` is peak activation, which says a neuron is
      // busy, not that a token is fabricated. Only flag when scored.
      const hot=(f.scored!==false) && f.s>1.5 && c.v>0.9;
      cx.globalAlpha=hot?1:(0.16+0.5*(1-age));
      cx.fillStyle=hot?'#F09595':'#EF9F27';
      cx.beginPath();
      cx.arc(x0+jitter, y, hot?3.2:1.5*(0.5+c.v), 0, 6.283); cx.fill();
    });
  });
  cx.globalAlpha=1;
  const sw=sc.clientWidth, sh=sc.clientHeight;
  sx.clearRect(0,0,sw,sh);
  sx.strokeStyle='#F09595'; sx.lineWidth=2; sx.beginPath();
  frames.forEach((f,i)=>{const x=sw*i/Math.max(frames.length-1,1);
    const y=sh/2-(f.s||0)*(sh/8); i?sx.lineTo(x,y):sx.moveTo(x,y);});
  sx.stroke();
  sx.strokeStyle='#2a2a34'; sx.lineWidth=1; sx.beginPath();
  sx.moveTo(0,sh/2); sx.lineTo(sw,sh/2); sx.stroke();
  const f=frames[frames.length-1];
  const scored=f.scored!==false;
  $('#stat').textContent=`tier ${f.t} · frame ${f.i} · ${
    scored?'score':'peak (no classifier)'} ${(f.s||0).toFixed(2)}${
    scored&&f.s>1.5?' · FLAGGED':''}`;
  $('#toks').innerHTML=toks.map(t=>(scored&&t.s>1.5)?`<b>${
    t.tok.replace(/</g,'&lt;')}</b>`:t.tok.replace(/</g,'&lt;')).join('');
}
</script>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true",
                   help="synthesise a field; use until the fork exists")
    p.add_argument("--source", help="llama-server with --activations (not yet "
                                    "implemented upstream)")
    p.add_argument("--tier", default="binned", choices=sorted(TIERS))
    p.add_argument("--token", help="require this token")
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    p.add_argument("--port", type=int, default=7890)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()

    STATE.update({"token": a.token or os.environ.get("NS_STUDIO_TOKEN"),
                  "tier": a.tier, "source": a.source,
                  "simulate": a.simulate})

    if not a.simulate and not a.source:
        print("no source; waiting for pushes from scripts/autotrace.py")
    if a.source:
        threading.Thread(target=relay, args=(a.source,), daemon=True).start()

    scheme = "https" if a.tls_cert else "http"
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    if a.tls_cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.tls_cert, a.tls_key or a.tls_cert)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)

    print(f"NeuronScope live on {scheme}://{a.host}:{a.port}")
    print(f"tier {a.tier}: {TIERS[a.tier][0]}, "
          f"~{TIERS[a.tier][1] / 1024:.1f} KB/token")
    if not STATE["token"]:
        print("\n!! no --token: anyone on this network can watch the stream")
    if a.host == "0.0.0.0" and not a.tls_cert:
        print("   plaintext on the LAN. For an untrusted network prefer a "
              "WireGuard or\n   Tailscale tunnel over a self-signed cert users "
              "learn to click through.")

    if a.simulate:
        threading.Thread(target=simulate, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
