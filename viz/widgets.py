"""
Widgets for NeuronScope's views: small, self-contained pieces of UI that take
data and settings from outside and can be put on any page (docs/WIDGETS.md).

The Godot client has the same widgets in viz/godot/widgets/, with the same
settings keys and the same drawing, so a theme reads the same in both.

    jewel          the emotion jewel: colour, size and motion from the emotion
                   vectors at the current token. Styles: core, icosphere,
                   tesseract, off. Click it to hide or show the emotion graph.
    emotion_graph  one lane per emotion over the reply.
    nav_gizmo      (Godot) Blender-style axis gizmo, zoom, pan, view presets.
    chat_box       (Godot) search the trace, or ask the model.

Settings come from three places, later ones winning: DEFAULTS here, the 3D
theme's own "widgets" block in themes.json, and the owner's choices saved by
Studio (Settings -> Appearance -> Widgets). The merged block is served with
the theme (api/theme) as "widgets"; a viewer's own picks in a client (the
jewel's right-click, its toggle) are kept by that client.
"""
from __future__ import annotations

import copy

# Every setting a widget reads, with its kind and range, so Settings can draw
# a form and Studio can refuse bad values.
SPEC = {
    "jewel": {
        "style": {"kind": "choice", "choices": ["core", "icosphere", "tesseract", "off"], "default": "core",
                  "help": "core: a reactor core with a ribbon per emotion; icosphere: a sphere pushed out where "
                          "each emotion is strong; tesseract: a 4D cube turning with the emotions"},
        "speed": {"kind": "float", "min": 0.1, "max": 4.0, "default": 1.0, "help": "animation speed"},
        "glow": {"kind": "float", "min": 0.0, "max": 2.0, "default": 1.0, "help": "halo and core brightness"},
        "scale": {"kind": "float", "min": 0.5, "max": 1.0, "default": 0.9, "help": "size within its box"},
    },
    "emotion_graph": {
        "lanes": {"kind": "int", "min": 1, "max": 12, "default": 6, "help": "emotions drawn, strongest first"},
        "labels": {"kind": "bool", "default": True, "help": "names in their lane colours"},
    },
    "nav_gizmo": {
        "visible": {"kind": "bool", "default": True, "help": "show the navigation gizmo (Godot)"},
        "sway": {"kind": "bool", "default": True,
                 "help": "the slow automatic sway; it pauses when you move the view, and the gizmo's sway button "
                         "or S turns it back on"},
    },
    "chat_box": {
        "max_hits": {"kind": "int", "min": 3, "max": 30, "default": 8, "help": "search results shown"},
    },
}

DEFAULTS = {w: {k: v["default"] for k, v in spec.items()} for w, spec in SPEC.items()}


def validate(block: dict) -> dict:
    """Keep only known widgets and settings with valid values; raise on a bad value."""
    out = {}
    for w, vals in (block or {}).items():
        if w not in SPEC or not isinstance(vals, dict):
            raise ValueError(f"unknown widget: {w}")
        for k, v in vals.items():
            s = SPEC[w].get(k)
            if s is None:
                raise ValueError(f"{w}: unknown setting {k}")
            if s["kind"] == "choice" and v not in s["choices"]:
                raise ValueError(f"{w}.{k}: one of {', '.join(s['choices'])}")
            if s["kind"] == "bool" and not isinstance(v, bool):
                raise ValueError(f"{w}.{k}: true or false")
            if s["kind"] in ("int", "float"):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not s["min"] <= v <= s["max"]:
                    raise ValueError(f"{w}.{k}: a number from {s['min']} to {s['max']}")
                v = int(v) if s["kind"] == "int" else float(v)
            out.setdefault(w, {})[k] = v
    return out


def merged(theme: dict | None, owner: dict | None = None) -> dict:
    out = copy.deepcopy(DEFAULTS)
    for block in ((theme or {}).get("widgets") or {}, owner or {}):
        for w, vals in block.items():
            if w in out and isinstance(vals, dict):
                out[w].update({k: v for k, v in vals.items() if k in SPEC[w]})
    return out


def with_widgets(theme: dict, owner: dict | None = None) -> dict:
    """The theme as served to a 3D view: itself plus the merged widgets block."""
    t = dict(theme)
    t["widgets"] = merged(theme, owner)
    return t


# The web widgets. Each page includes WIDGETS_JS once; the drawing matches
# viz/godot/widgets/jewel.gd so the two clients agree.
WIDGETS_JS = r"""
const NSWidgets = (() => {
  const reg = {};
  const store = { get(k, d){ try{ const v = localStorage.getItem('ns-w-'+k); return v===null ? d : JSON.parse(v); }catch(e){ return d; } },
                  set(k, v){ try{ localStorage.setItem('ns-w-'+k, JSON.stringify(v)); }catch(e){} } };
  function register(name, factory){ reg[name] = factory; }
  function create(name, el, opts){ return reg[name](el, opts || {}); }
  const hex = c => { c = String(c||'#888'); if(c[0]!=='#') c = '#888';
    if(c.length===4) c = '#'+[...c.slice(1)].map(x=>x+x).join(''); const n = parseInt(c.slice(1,7),16);
    return [(n>>16)/255, ((n>>8)&255)/255, (n&255)/255]; };
  const css = (c, a) => `rgba(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)},${a})`;
  const dark = (c, f) => c.map(v => v*(1-f));
  const light = (c, f) => c.map(v => v + (1-v)*f);

  // ---------------------------------------------------------------- jewel
  register('jewel', (canvas, opts) => {
    const ctx = canvas.getContext('2d');
    let S = Object.assign({style:'core', speed:1, glow:1, scale:0.9}, opts.settings||{}, store.get('jewel', {}));
    let items = [], cur = [], mag = 0, mix = [0.4,0.55,1], t = 0, last = performance.now(), dirs = [];
    const ico = (() => { const g = (1+Math.sqrt(5))/2;
      let v = [[-1,g,0],[1,g,0],[-1,-g,0],[1,-g,0],[0,-1,g],[0,1,g],[0,-1,-g],[0,1,-g],[g,0,-1],[g,0,1],[-g,0,-1],[-g,0,1]];
      const f = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]];
      const nrm = p => { const l = Math.hypot(...p); return p.map(x=>x/l); }; v = v.map(nrm);
      const mid = {}, nf = [];
      for(const tri of f){ const m = [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]].map(([a,b]) => { const k = Math.min(a,b)+'_'+Math.max(a,b);
          if(!(k in mid)){ v.push(nrm(v[a].map((x,i)=>(x+v[b][i])/2))); mid[k] = v.length-1; } return mid[k]; });
        nf.push([tri[0],m[0],m[2]],[tri[1],m[1],m[0]],[tri[2],m[2],m[1]],[m[0],m[1],m[2]]); }
      return {v, f: nf}; })();
    const spread = n => [...Array(n)].map((_,i) => { const y = 1-2*(i+0.5)/Math.max(1,n), r = Math.sqrt(Math.max(0,1-y*y)), th = Math.PI*(3-Math.sqrt(5))*i;
      return [Math.cos(th)*r, y, Math.sin(th)*r]; });
    function set(list){ items = list || []; if(cur.length !== items.length){ cur = items.map(()=>0); dirs = spread(items.length); } }
    function step(){
      const now = performance.now(), dt = Math.min(0.1, (now-last)/1000); last = now; t += dt*S.speed;
      const a = 1 - Math.exp(-dt*5); let mx = 0, acc = [0,0,0], ws = 0;
      items.forEach((it,k) => { cur[k] += (it.z - cur[k])*a; mx = Math.max(mx, Math.abs(cur[k])); const w = cur[k]*cur[k];
        let c = hex(it.color); if(cur[k] < 0) c = dark(c, 0.5); acc = acc.map((v,i)=>v + c[i]*w); ws += w; });
      mag += (Math.min(1, mx/3) - mag)*a; if(ws > 1e-4) mix = mix.map((v,i)=> v + (acc[i]/ws - v)*a);
      draw(); requestAnimationFrame(step); }
    function color(k){ return hex(items[k] && items[k].color); }
    function draw(){
      const r0 = canvas.getBoundingClientRect(), dpr = devicePixelRatio||1;
      if(canvas.width !== Math.round(r0.width*dpr)){ canvas.width = r0.width*dpr; canvas.height = r0.height*dpr; }
      ctx.setTransform(dpr,0,0,dpr,0,0); const W = r0.width, H = r0.height, cx = W/2, cy = H/2, r = Math.min(W,H)/2*S.scale;
      ctx.clearRect(0,0,W,H);
      if(S.style==='off' || !items.length){ ctx.strokeStyle='rgba(102,102,115,.5)'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(cx,cy,r*0.35,0,7); ctx.stroke(); return; }
      for(let i=0;i<6;i++){ const f = 1-i/6; ctx.fillStyle = css(mix, 0.05*S.glow*(0.4+mag)*f);
        ctx.beginPath(); ctx.arc(cx,cy, r*(0.45+0.55*(1-f))*(0.7+0.3*mag), 0, 7); ctx.fill(); }
      if(S.style==='icosphere') drawIco(cx,cy,r); else if(S.style==='tesseract') drawTess(cx,cy,r); else drawCore(cx,cy,r);
    }
    function drawCore(cx,cy,r){
      const n = cur.length, back = [], front = [];
      cur.forEach((z,k) => { const s = Math.min(1, Math.abs(z)/3), tilt = 0.35 + 1.1*k/Math.max(1,n), spin = k*2*Math.PI/Math.max(1,n);
        const head = t*(0.8+1.6*s)*(z>=0?1:-1) + spin*1.7, len = 0.6+2.4*s, steps = 24, col = color(k), rr = r*(0.68+0.12*k/Math.max(1,n));
        const orb = u => { const p = [Math.cos(u)*rr, Math.sin(u)*rr*Math.cos(tilt), Math.sin(u)*rr*Math.sin(tilt)];
          return [cx + p[0]*Math.cos(spin) - p[1]*Math.sin(spin), cy + p[0]*Math.sin(spin) + p[1]*Math.cos(spin), p[2]]; };
        for(let j=0;j<steps;j++){ const p0 = orb(head - len*j/steps), p1 = orb(head - len*(j+1)/steps);
          const seg = [p0, p1, col, (1-j/steps)*(0.5+0.5*s), (1.6+3.4*s)*(1-0.6*j/steps)];
          (p0[2] < 0 ? back : front).push(seg); } });
      const line = (sg, f) => { ctx.strokeStyle = css(sg[2], sg[3]*f); ctx.lineWidth = sg[4]; ctx.beginPath(); ctx.moveTo(sg[0][0], sg[0][1]); ctx.lineTo(sg[1][0], sg[1][1]); ctx.stroke(); };
      ctx.lineCap = 'round'; back.forEach(sg => line(sg, 0.35));
      const pulse = 1 + 0.08*Math.sin(t*(2+5*mag))*(0.3+mag), cr = r*(0.26+0.18*mag)*pulse;
      for(let i=0;i<8;i++){ const f = i/8; ctx.fillStyle = css(light(mix, f*f*0.85), (0.35+0.65*f)*Math.min(1, 0.6+0.4*S.glow));
        ctx.beginPath(); ctx.arc(cx,cy, cr*(1-0.11*i), 0, 7); ctx.fill(); }
      front.forEach(sg => line(sg, 1));
    }
    function drawIco(cx,cy,r){
      const ay = t*0.5, ax = 0.4 + 0.2*Math.sin(t*0.3), cyy = Math.cos(ay), sy = Math.sin(ay), cxx = Math.cos(ax), sx = Math.sin(ax);
      const rot = p => { const x = p[0]*cyy + p[2]*sy, z = -p[0]*sy + p[2]*cyy; return [x, p[1]*cxx - z*sx, p[1]*sx + z*cxx]; };
      const pts = [], wts = [];
      ico.v.forEach(v => { let d = 0; const w = cur.map((z,k) => { const f = Math.pow(Math.max(0, v[0]*dirs[k][0]+v[1]*dirs[k][1]+v[2]*dirs[k][2]), 3); d += z/3*f; return f*Math.abs(z); });
        const s = (0.55 + 0.35*Math.max(-0.6, Math.min(1.2, d)))*r; pts.push(rot(v.map(x=>x*s))); wts.push(w); });
      const light3 = [-0.4,-0.6,0.7].map(x=>x/Math.hypot(0.4,0.6,0.7)), faces = [];
      ico.f.forEach(tri => { const [a,b,e] = tri.map(i=>pts[i]); const u = b.map((x,i)=>x-a[i]), w = e.map((x,i)=>x-a[i]);
        let nn = [u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]]; const l = Math.hypot(...nn)||1; nn = nn.map(x=>x/l);
        if(nn[2] < 0) return; faces.push([(a[2]+b[2]+e[2])/3, tri, nn]); });
      faces.sort((p,q)=>p[0]-q[0]);
      for(const [_, tri, nn] of faces){ let col = [0,0,0], ws = 0;
        for(const i of tri) cur.forEach((z,k) => { let ck = color(k); if(z<0) ck = dark(ck,0.5); const w = wts[i][k]; col = col.map((v,j)=>v+ck[j]*w); ws += w; });
        col = ws < 1e-4 ? dark(mix, 0.35) : col.map(v=>v/ws); const sh = 0.35 + 0.65*Math.max(0, nn[0]*light3[0]+nn[1]*light3[1]+nn[2]*light3[2]);
        col = col.map(v=>v*sh); ctx.fillStyle = css(col, 0.95); ctx.strokeStyle = css(light(col,0.3), 0.35); ctx.lineWidth = 1;
        ctx.beginPath(); tri.forEach((i,j)=> j ? ctx.lineTo(cx+pts[i][0], cy+pts[i][1]) : ctx.moveTo(cx+pts[i][0], cy+pts[i][1])); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    }
    function drawTess(cx,cy,r){
      let pos = 0, neg = 0; cur.forEach(z => { pos += Math.max(0,z); neg += Math.max(0,-z); });
      const axw = t*(0.3+0.5*Math.min(2,pos/3)), ayw = t*(0.2+0.5*Math.min(2,neg/3)), axy = t*0.25, proj = [];
      for(let i=0;i<16;i++){ const p = [i&1?1:-1, i&2?1:-1, i&4?1:-1, i&8?1:-1];
        const x = p[0]*Math.cos(axw) - p[3]*Math.sin(axw); let w = p[0]*Math.sin(axw) + p[3]*Math.cos(axw);
        const y = p[1]*Math.cos(ayw) - w*Math.sin(ayw); w = p[1]*Math.sin(ayw) + w*Math.cos(ayw);
        const x2 = x*Math.cos(axy) - p[2]*Math.sin(axy), z2 = x*Math.sin(axy) + p[2]*Math.cos(axy);
        const k4 = 1/(2.6-w), k3 = 1/(3 - z2*k4); proj.push([cx + x2*k4*k3*r*2.4, cy + y*k4*k3*r*2.4, w]); }
      const n = cur.length; let e = 0; ctx.lineCap = 'round';
      for(let i=0;i<16;i++) for(let b=0;b<4;b++){ const j = i ^ (1<<b); if(j < i) continue;
        const inner = proj[i][2] < 0 && proj[j][2] < 0, col = inner||!n ? mix : color(e%n), s = inner||!n ? mag : Math.min(1, Math.abs(cur[e%n])/3);
        ctx.strokeStyle = css(col, 0.35+0.65*s); ctx.lineWidth = 1+2.5*s; ctx.beginPath(); ctx.moveTo(proj[i][0],proj[i][1]); ctx.lineTo(proj[j][0],proj[j][1]); ctx.stroke(); e++; }
      ctx.fillStyle = css(light(mix,0.4), 1); proj.forEach(p => { ctx.beginPath(); ctx.arc(p[0],p[1],1.5+2*mag,0,7); ctx.fill(); });
    }
    const styles = ['core','icosphere','tesseract','off'];
    canvas.title = 'Emotion jewel: click to hide or show the emotion graph, right-click for the next style';
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('contextmenu', e => { e.preventDefault(); S.style = styles[(styles.indexOf(S.style)+1)%styles.length];
      store.set('jewel', Object.assign(store.get('jewel', {}), {style: S.style})); });
    canvas.addEventListener('click', () => opts.onToggle && opts.onToggle());
    requestAnimationFrame(step);
    return { set, settings(s){ S = Object.assign(S, s||{}, store.get('jewel', {})); }, get style(){ return S.style; } };
  });

  // Rows of z values per emotion -> what the jewel shows at token i.
  function emotionsAt(E, i, palette){ if(!E || !E.z || !E.z[i]) return [];
    return E.names.map((n,k) => ({name: n, z: E.z[i][k], color: palette[k % palette.length]})); }
  return { register, create, store, emotionsAt };
})();
"""
