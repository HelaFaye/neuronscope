"""Studio's /system page: what is running and how to restart any part of it, the hardware right now and
over time, and automation rules that react to the host's use and power."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio System</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#5fcf8f;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;align-items:center;gap:.8rem;padding:.6rem 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}main{max-width:1100px;margin:0 auto;padding:1rem 16px 3rem;display:grid;gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
h2{font-size:13px;margin:0 0 .6rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;display:flex;gap:.6rem;align-items:center}
h2 .r{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:400}
.note{font-size:12.5px;color:var(--mut)}a{color:var(--acc)}
button{font:500 13px ui-sans-serif,system-ui;padding:.32rem .75rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button:disabled{opacity:.5;cursor:default}
.svc{display:grid;grid-template-columns:minmax(10rem,14rem) minmax(7rem,9rem) 1fr auto;gap:.3rem .9rem;align-items:center;padding:.5rem 0;border-bottom:1px solid var(--line)}
.svc:last-child{border-bottom:none}.svc .nm{font-weight:600}.svc .det{font-size:12.5px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.svc .acts{display:flex;gap:.35rem;justify-content:flex-end}
.st{font-size:12.5px}.st.on{color:var(--ok)}.st.off{color:var(--mut)}.st.warn{color:var(--warn)}
@media(max-width:700px){.svc{grid-template-columns:1fr auto}.svc .det{grid-column:1/-1}}
#msg{min-height:1.2rem}
</style>
<header><h1>Studio · System</h1><span class="note" id="where"></span></header>
<main>
<section class="card" id="services"><h2>Running parts <span class="r"><button id="restartAll" class="pri" title="Stop every model this Studio started and start Studio again with the same settings">Restart Studio</button></span></h2>
<div class="note" style="margin-bottom:.4rem">Each part Studio runs on this computer. Restarting Studio stops all of them first, so nothing is left holding a port or GPU memory. Jobs keep running in the background only until Studio stops.</div>
<div id="svcList">…</div><div id="msg" class="note"></div></section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const LABEL={restart:'Restart',reload:'Reload',stop:'Stop',reconnect:'Reconnect'};
const TIP={'studio:restart':'Stop every model Studio started, then start Studio again with the same settings',
  'chat:reload':'Stop the chat model server and load the same model again with its settings','chat:stop':'Unload the chat model and free its memory',
  'workers:stop':'Stop every worker model (project tasks wait until one starts again)','embed:restart':'Stop the embedding server; the next document search starts it fresh',
  'mcp:reconnect':'Disconnect and reconnect every MCP server, re-reading mcp.json','labs:stop':'Stop every running lab'};
function draw(list){ $('#svcList').innerHTML=list.map(s=>{ const on=/running|connected/.test(s.state)&&!/^0 of/.test(s.state);
  return `<div class="svc" data-id="${esc(s.id)}"><div class="nm">${esc(s.name)}</div>
    <div class="st ${s.warn?'warn':on?'on':'off'}">${s.warn?'⚠ ':on?'● ':'○ '}${esc(s.state)}</div>
    <div class="det" title="${esc(s.detail)}">${esc(s.detail||'')}</div>
    <div class="acts">${(s.actions||[]).map(a=>`<button data-a="${a}" title="${esc(TIP[s.id+':'+a]||'')}">${LABEL[a]||a}</button>`).join('')}</div></div>`; }).join('');
  document.querySelectorAll('.svc [data-a]').forEach(b=>b.onclick=async()=>{ const id=b.closest('.svc').dataset.id, a=b.dataset.a;
    if(id==='studio') return nsRestart();
    b.disabled=true; const was=b.textContent; b.textContent='…';
    try{ const r=await api('/api/services',{id,action:a}); $('#msg').innerHTML=`<span style="color:var(--ok)">${esc(r.note)}</span>`; draw(r.services); }
    catch(e){ $('#msg').innerHTML=`<span style="color:var(--no)">${esc(e.message)}</span>`; b.disabled=false; b.textContent=was; } }); }
async function load(){ try{ const j=await api('/api/services'); draw(j.services); }catch(e){ $('#svcList').innerHTML=`<div class="note">${esc(e.message)}</div>`; } }
$('#restartAll').onclick=()=>nsRestart();
fetch('/api/status').then(r=>r.json()).then(j=>$('#where').textContent='on '+(j.computer||location.hostname)).catch(()=>{});
load(); setInterval(load,5000);
</script>
"""

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
