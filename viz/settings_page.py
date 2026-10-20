"""Studio's /settings page: workspaces (where models, llama.cpp, pipeline data
and Studio's state live), the model to use per subject, and tool approval
rules for MCP tools in chat. Same as scripts/ns_workspace.py and
scripts/tool_rules.py on the command line."""

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Settings</title><script>try{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{}');for(const k in v)d.style.setProperty('--'+k,v[k])}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--warn:#a8701c;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--warn:#e7b65c;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
a{color:var(--acc)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:15px;margin:0}header a{font-size:13px}
nav.toc{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;font-size:13px}
main{max-width:1050px;margin:0 auto;padding:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;margin-bottom:1rem;scroll-margin-top:1rem}
.card:target{border-color:var(--acc)}
h2{font-size:15px;margin:.1rem 0 .4rem}h3{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:1rem 0 .4rem}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no);font-size:12.5px;white-space:pre-wrap}.okm{color:var(--ok);font-size:12.5px}
label{display:block;font-size:12.5px;margin:.55rem 0 .15rem}label .note{display:block}
input,select,textarea{font:12.5px ui-monospace,monospace;padding:.3rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto}textarea{min-height:3rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.38rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}button.bad{color:var(--no)}button:disabled{opacity:.5}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}.row>*{min-width:0}.grow{flex:1}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 1rem}@media (max-width:760px){.grid{grid-template-columns:1fr}}
.tabs{display:flex;gap:.3rem;flex-wrap:wrap;margin:.4rem 0}.tab{padding:.25rem .7rem;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px;background:var(--bg)}
.tab.on{border-color:var(--acc);color:var(--acc)}.tab .act{font-size:10.5px;color:var(--ok);margin-left:.3rem}
table{width:100%;border-collapse:collapse}td,th{padding:.35rem .3rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-size:11px;color:var(--mut);text-transform:uppercase;font-weight:500}
.miss{color:var(--warn);font-size:11.5px}.dflt{color:var(--mut);font-size:11.5px}
.pill{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 .35rem;white-space:nowrap}
.pill.allow{color:var(--ok)}.pill.deny{color:var(--no)}.pill.ask{color:var(--warn)}
.rule{border:1px solid var(--line);border-radius:6px;padding:.5rem .6rem;margin:.4rem 0;background:var(--bg)}
.rule .row label{margin:0}.cond{display:grid;grid-template-columns:9rem 1fr 6rem 2rem;gap:.3rem;margin-top:.3rem}
code{font:12px ui-monospace,monospace;background:var(--code);padding:.05rem .3rem;border-radius:4px}
.themes{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.6rem;margin-top:.6rem}
.th{border:1px solid var(--line);border-radius:8px;padding:.5rem;cursor:pointer}.th.on{outline:2px solid var(--acc)}
.th .sw{display:flex;height:34px;border-radius:5px;overflow:hidden;margin:.35rem 0}.th .sw span{flex:1}
</style>
<header><h1>Studio · Settings</h1><a href="/">← Studio</a><a href="/setup">Setup</a><a href="/settings">Settings</a><a href="/lab">Lab</a><a href="/review">Review</a><a href="/projects">Projects</a><a href="/workers">Workers</a><a href="/jobs">Jobs</a><a href="/connect">Connect</a></header>
<main>
<nav class="toc"><a href="#appearance">Appearance</a><a href="#workspaces">Workspaces</a><a href="#subjects">Models per subject</a><a href="#tools">Tool approval</a><a href="#sharing">Sharing results</a><a href="#plugins">Plugins</a><span class="note">Behaviour (unloading, routing thresholds, workers) and building llama.cpp are on <a href="/setup">Setup</a>.</span></nav>

<section class="card" id="appearance"><h2>Appearance</h2>
<div class="note">A theme colours Studio's pages, its charts (the model radar) and the 3D views of activations and hallucination risk. Your choice is kept in this browser; <b>Make default</b> sets it for browsers that have not chosen one, and for the Godot client's 3D view.</div>
<div id="themeGrid" class="themes"></div><div id="themeMsg" class="note"></div></section>

<section class="card" id="workspaces"><h2>Workspaces</h2>
<div class="note">Where things are, by name: the models, the llama.cpp build, the folder that holds the pipeline's data (<code>data/</code>, <code>models/</code>, <code>runs/</code>…; jobs run there), Studio's own state, and environment for the GPU stack. Keep one per disk layout or machine, and switch instead of passing paths. Command line: <code>python scripts/ns_workspace.py</code>.</div>
<div class="tabs" id="wstabs"></div>
<div id="wsform"></div>
<div class="row" style="margin-top:.7rem"><button class="pri" id="wssave">Save</button><button id="wsuse">Use this workspace</button>
  <span class="grow"></span><input id="wsnew" placeholder="new workspace name" style="width:13rem"><button id="wsadd">Add</button><button id="wsdup">Duplicate</button><button class="bad" id="wsdel">Delete</button></div>
<div id="wsmsg"></div></section>

<section class="card" id="subjects"><h2>Models per subject</h2>
<div class="note">When a request asks for model <code>auto</code>, Studio sends it to the model measured best for the prompt's subject (from TestQA and benchmark results). Pick a model for a subject to make it the default for that subject instead; leave it on <i>measured best</i> to follow the numbers. Your pick wins whenever that subject is at least 30% of the prompt.</div>
<table id="subjtbl"></table>
<div class="row" style="margin-top:.6rem"><button class="pri" id="subjsave">Save</button><span id="subjmsg"></span></div>
<div class="note" id="subjnote" style="margin-top:.4rem"></div></section>

<section class="card" id="tools"><h2>Tool approval</h2>
<div class="note">When a model calls an MCP tool in chat (Studio → Tools), each call is decided in order: <b>1</b> your rules below, first match wins; <b>2</b> a server's <code>autoApprove</code> in mcp.json; <b>3</b> the quick setting for the tool's kind of operation (from its MCP annotations); <b>4</b> the default. <i>Ask</i> shows Allow/Deny in the chat, where <i>always…</i> adds a rule here. Command line: <code>python scripts/tool_rules.py</code>.</div>
<h3>Quick settings</h3>
<div class="grid" id="cats"></div>
<h3>Rules (first match wins)</h3>
<div id="rules"></div>
<div class="row"><button id="radd">Add a rule</button><button id="rpreset">Add common rules…</button><span class="grow"></span><button class="pri" id="rsave">Save rules</button><span id="rmsg"></span></div>
<h3>Try a call</h3>
<div class="row"><input id="tserver" placeholder="server" style="width:9rem"><input id="ttool" placeholder="tool" style="width:11rem"><input id="targs" class="grow" placeholder='arguments JSON, e.g. {"path": "/home/me/notes/a.md"}'>
<select id="tann" style="width:auto"><option value="{}">no annotations</option><option value='{"readOnlyHint":true}'>read-only</option><option value='{"readOnlyHint":false,"destructiveHint":false}'>writes</option><option value='{"readOnlyHint":false,"destructiveHint":true}'>destructive</option><option value='{"readOnlyHint":true,"openWorldHint":true}'>read-only, open world</option></select><button id="ttest">Test</button></div>
<div id="tout" class="note"></div>
<h3>Connected tools</h3><div id="tlist" class="note"></div></section>
<section class="card" id="plugins"><h2>Plugins</h2>
<div class="note">A plugin is a folder with a <code>plugin.json</code>. It can add jobs, checks under chat replies, a page of its own and MCP tools. <b>Plugins run code on this computer</b>, so none is used until you turn it on. If a plugin's files change after that, it turns itself off until you look at it again. See docs/PLUGINS.md to write one.</div>
<div id="plDirs" class="note"></div><table id="plList"></table><div id="plMsg" class="note"></div></section>

<section class="card" id="sharing"><h2>Sharing results</h2>
<div class="note"><b>Off unless you turn it on.</b> When on, TestQA results can be sent to a scoreboard you name, so models can be compared across machines and setups. NeuronScope runs no scoreboard of its own; anyone can host <code>scripts/scoreboard_server.py</code>.</div>
<details open style="margin:.5rem 0"><summary><b>What is sent, and what it is for</b></summary>
<ul class="note">
<li><b>A card, which may be public</b>:
  <ul>
  <li>the model file's SHA-256, size, name (not its path), quantization, and its Hugging Face repo if the file is verified to come from one;</li>
  <li>what makes this setup different, so results are never mixed: the hash of the H-Neuron set and the suppression scale, the LoRA's hash and scale, fine-tune / merge / concept-edit lineage, the engine and its version, the engine options that can change answers (context, KV cache, speculative decoding, flash attention), and the sampling;</li>
  <li>coarse hardware (OS, CPU model, cores, RAM and GPU memory in whole GB, GPU names);</li>
  <li>the test bank's fingerprint and the right / wrong / declined counts per subject.</li>
  </ul></li>
<li><b>A verification log, private</b>: per test item, its id, the verdict and the model's reply. The scoreboard uses it only to check the card by grading the replies again; it must not publish it.</li>
<li><b>Never sent</b>: your chats, prompts you typed, file paths, user or computer names, tokens, other models.</li>
<li><b>Public only for models anyone can find</b>: the file's SHA-256 matches a file in its Hugging Face repo, or you list an unlisted model below. Everything else is private, or not sent.</li>
<li>Every submission is kept on this computer exactly as sent (<code>~/.neuronscope/submissions/</code>), and listed below.</li>
</ul></details>
<label><input type="checkbox" id="shOn"> Share results</label>
<label>Scoreboard address<input id="shEp" placeholder="https://scores.example.org"></label>
<label><input type="checkbox" id="shAuto"> Submit automatically when a TestQA run finishes, for models found on Hugging Face</label>
<details style="margin:.4rem 0"><summary class="note">Advanced</summary>
<label><input type="checkbox" id="shUnl"> Also submit results for unlisted models automatically <span class="note">(sent privately: only the scoreboard's operator can see them, unless you list the model below)</span></label></details>
<div class="row" style="margin-top:.4rem"><button class="pri" id="shSave">Save</button><span id="shMsg" class="note"></span></div>
<h3>Models</h3><div class="note">Found on Hugging Face = this exact file is in the repo it came from (checking reads the whole file once).</div>
<table id="shModels"></table>
<h3>Submit a result by hand</h3>
<div class="grid"><label>Results (TestQA runs from Jobs)<select id="shRes"></select></label><label>Model<select id="shModel"></select></label></div>
<div class="row" style="margin-top:.4rem"><button id="shPrev">Preview exactly what is sent</button><button id="shSend">Submit</button></div>
<pre id="shOut" class="note" style="max-height:22rem;overflow:auto;white-space:pre-wrap"></pre>
<h3>Sent so far</h3><table id="shHist"></table>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(p,b){const r=await fetch(p,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
const msg=(el,t,bad)=>{$(el).innerHTML=t?`<span class="${bad?'err':'okm'}">${t}</span>`:''};
async function restartBtn(el){ $(el).insertAdjacentHTML('beforeend',' <button id="rs">Restart Studio now</button>');
  $('#rs').onclick=async()=>{ await api('/api/restart',{}).catch(()=>{}); msg(el,'restarting…'); setTimeout(()=>location.reload(),4000); }; }

// ---------- workspaces
let WS=null, cur=null;
const LABEL={root:'Pipeline data root',models_dirs:'Models folders',download_dir:'Downloads folder',llama_dir:'llama.cpp checkout',
  server:'llama-server',cett:'llama-cett-dump',gguf:'Model for the H-Neuron pipeline (.gguf)',state_dir:'Studio state folder',env:'Environment variables',note:'Note'};
function wsTabs(){ $('#wstabs').innerHTML=WS.workspaces.map(w=>`<span class="tab ${w.name===cur?'on':''}" data-w="${esc(w.name)}">${esc(w.name)}${w.active?'<span class="act">active</span>':''}${w.name===WS.running&&!w.active?'<span class="act" style="color:var(--warn)">running</span>':''}</span>`).join('');
  document.querySelectorAll('[data-w]').forEach(t=>t.onclick=()=>{cur=t.dataset.w;wsTabs();wsForm();}); }
function wsForm(){
  const w=WS.workspaces.find(x=>x.name===cur), v=w.values, r=w.resolved, ex=w.exists;
  const field=k=>{ const f=WS.fields[k]; let val=v[k];
    if(k==='models_dirs') val=(val||[]).join('\n'); if(k==='env') val=Object.entries(val||{}).map(([a,b])=>a+'='+b).join('\n');
    const dflt=k==='models_dirs'?r.models_dirs.join(', '):k==='env'?'':r[k]||'';
    const missing=k==='models_dirs'?(ex.models_dirs.every(x=>!x)?'none of these folders exists':''):(k in ex&&r[k]&&!ex[k]?'does not exist':'');
    const big=['models_dirs','env'].includes(k), pick={root:'dir',download_dir:'dir',llama_dir:'dir',state_dir:'dir',gguf:'file',server:'file',cett:'file'}[k];
    return `<label>${esc(LABEL[k]||k)}<span class="note">${esc(f.help)}</span></label>`+
      (big?`<textarea data-f="${k}" hidden>${esc(val||'')}</textarea><div data-wl="${k}" data-dflt="${esc(dflt)}"></div>`:`<input data-f="${k}" value="${esc(val||'')}" placeholder="${esc(dflt?'default: '+dflt:'')}"${pick?` data-pick="${pick}"`:''}${k==='gguf'?' data-ext="gguf"':''}>`)+
      `${missing?`<div class="miss">${esc(missing)}${k==='server'||k==='cett'?' · <a href="/setup#build">build llama.cpp</a>':''}</div>`:''}<div data-err="${k}" class="err"></div>`; };
  $('#wsform').innerHTML=`<div class="grid"><div>${['root','models_dirs','download_dir','gguf','note'].map(field).join('')}</div><div>${['llama_dir','server','cett','state_dir','env'].map(field).join('')}</div></div>`+
    `<div class="note" style="margin-top:.5rem">Saved in <code>${esc(WS.config)}</code>. ${WS.env_override?`Studio was started with workspace ${esc(WS.env_override)} (--workspace or NS_WORKSPACE); “active” applies when it starts without one.`:''} Empty fields use the default shown in grey.</div>`;
  // Per-entry lists, synced into the hidden fields wsValues reads.
  document.querySelectorAll('[data-wl]').forEach(d=>{ const k=d.dataset.wl, ta=document.querySelector(`#wsform textarea[data-f=${k}]`), dirs=k==='models_dirs';
    nsListEditor(d,{values:ta.value.split('\n').map(x=>x.trim()).filter(Boolean),want:'dir',browse:dirs,addLabel:dirs?'+ Add folder':'+ Add variable',
      placeholder:dirs?'':'NAME=value',empty:dirs?(d.dataset.dflt?'None set: the default is '+d.dataset.dflt:'None set.'):'None set.',onchange:v=>ta.value=v.join('\n')}); });
  $('#wsuse').disabled=w.active; $('#wsdel').disabled=w.name==='default'||w.name===WS.running;
}
function wsValues(){ const out={}; document.querySelectorAll('#wsform [data-f]').forEach(e=>{ const k=e.dataset.f; let v=e.value.trim();
    if(k==='models_dirs') v=v?v.split('\n').map(x=>x.trim()).filter(Boolean):null;
    else if(k==='env'){ const o={}; for(const line of v.split('\n')){ const t=line.trim(); if(!t) continue; const i=t.indexOf('='); if(i<1) throw new Error('environment lines are VAR=value'); o[t.slice(0,i).trim()]=t.slice(i+1).trim(); } v=Object.keys(o).length?o:null; }
    else v=v||null; out[k]=v; }); return out; }
async function loadWS(){ WS=await api('/api/workspaces'); if(!cur||!WS.workspaces.some(w=>w.name===cur)) cur=WS.active; wsTabs(); wsForm(); }
$('#wssave').onclick=async()=>{ document.querySelectorAll('[data-err]').forEach(e=>e.textContent=''); try{ const r=await api('/api/workspaces/save',{name:cur,values:wsValues(),replace:true}); WS=r; wsTabs(); wsForm();
  msg('#wsmsg','saved'+(r.restart_needed?' · Studio uses the new paths after a restart':'')); if(r.restart_needed) restartBtn('#wsmsg'); }
  catch(e){ msg('#wsmsg',esc(e.message),true); for(const part of e.message.split('; ')){ const [k,...m]=part.split(': '); const el=document.querySelector(`[data-err="${k}"]`); if(el) el.textContent=m.join(': '); } } };
$('#wsuse').onclick=async()=>{ try{ const r=await api('/api/workspaces/use',{name:cur}); WS=r; wsTabs(); wsForm(); msg('#wsmsg',`${esc(cur)} is now the active workspace`+(r.restart_needed?'; restart Studio to switch to it':'')); if(r.restart_needed) restartBtn('#wsmsg'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsadd').onclick=async()=>{ const n=$('#wsnew').value.trim(); if(!n) return msg('#wsmsg','name the new workspace first',true);
  try{ await api('/api/workspaces/save',{name:n,values:{}}); cur=n; $('#wsnew').value=''; await loadWS(); msg('#wsmsg','added; fill in its folders and Save'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsdup').onclick=async()=>{ const n=$('#wsnew').value.trim(); if(!n) return msg('#wsmsg','type a name for the copy',true);
  try{ const w=WS.workspaces.find(x=>x.name===cur); await api('/api/workspaces/save',{name:n,values:w.values}); cur=n; $('#wsnew').value=''; await loadWS(); msg('#wsmsg','copied'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };
$('#wsdel').onclick=async()=>{ if(!confirm(`Delete workspace ${cur}? Its files stay; only the setting goes.`)) return; try{ await api('/api/workspaces/delete',{name:cur}); cur=null; await loadWS(); msg('#wsmsg','deleted'); }catch(e){ msg('#wsmsg',esc(e.message),true); } };

// ---------- subjects and tools
let ST=null, RULES=null;
function subjTable(){
  const s=ST.subjects, opts=sel=>`<option value="">measured best</option>`+ST.models.map(m=>`<option ${m===sel?'selected':''}>${esc(m)}</option>`).join('');
  $('#subjtbl').innerHTML=`<tr><th>subject</th><th>measured best</th><th>default for this subject</th></tr>`+Object.entries(s).map(([k,v])=>
    `<tr><td><b>${esc(k)}</b></td><td>${v.best?`${esc(v.best)}<div class="dflt">${v.subject_n?`${v.subject_n} graded ${esc(k)} items`:'no '+esc(k)+' items yet: overall numbers'}</div>`:`<span class="dflt">no model has ${ST.min_graded} graded results yet · <a href="/jobs#kind=testqa">run TestQA</a></span>`}</td>
      <td><select data-s="${esc(k)}">${opts(v.chosen)}</select>${v.chosen_missing?'<div class="miss">this model is not in the models folders any more</div>':''}</td></tr>`).join('');
  $('#subjnote').innerHTML=`Measured best needs ${ST.min_graded} graded results per model (Setup → <a href="/setup#paths">routing thresholds</a>). Per-model results: <a href="/">Studio → Models</a>.`;
}
$('#subjsave').onclick=async()=>{ const v={}; document.querySelectorAll('[data-s]').forEach(e=>v[e.dataset.s]=e.value||null);
  try{ const r=await api('/api/settings/subjects',{subject_models:v}); ST.subjects=r.subjects; subjTable(); msg('#subjmsg','saved; applies to the next request'); }catch(e){ msg('#subjmsg',esc(e.message),true); } };

const KLABEL={read_only:'Read-only tools',write:'Tools that write (not destructive)',destructive:'Destructive tools (or not marked otherwise)',open_world:'Tools that reach the outside world',unknown:'Tools with no annotations'};
function cats(){ $('#cats').innerHTML=`<div><label>When nothing else decides</label><select id="dflt">${['ask','allow','deny'].map(a=>`<option ${RULES.default===a?'selected':''}>${a}</option>`).join('')}</select></div>`+
  ST.tool_kinds.map(k=>`<div><label>${esc(KLABEL[k])}</label><select data-c="${k}">${['default','ask','allow','deny'].map(a=>`<option value="${a}" ${RULES.categories[k]===a?'selected':''}>${a==='default'?'same as the default':a}</option>`).join('')}</select></div>`).join(''); }
function ruleHtml(r,i){
  const c=(x,j)=>`<div class="cond" data-c="${j}"><input data-k="path" value="${esc(x.path)}" placeholder="argument path, * = any"><input data-k="regex" value="${esc(x.regex)}" placeholder="regex"><label><input type="checkbox" data-k="negate" ${x.negate?'checked':''}> not</label><button data-cdel="${j}" title="remove">×</button></div>`;
  const exp=r.expires?new Date(r.expires*1000).toISOString().slice(0,16):'';
  return `<div class="rule" data-r="${i}"><div class="row"><b>${i+1}.</b><select data-k="action" style="width:auto">${['allow','ask','deny'].map(a=>`<option ${r.action===a?'selected':''}>${a}</option>`).join('')}</select>
    <input data-k="server" value="${esc(r.server)}" placeholder="server glob" style="width:9rem"><input data-k="tool" value="${esc(r.tool)}" placeholder="tool glob" style="width:11rem">
    <select data-k="kind" style="width:auto">${['any',...ST.tool_kinds].map(k=>`<option value="${k}" ${r.kind===k?'selected':''}>${k==='any'?'any kind':k.replace('_',' ')}</option>`).join('')}</select>
    <label><input type="checkbox" data-k="enabled" ${r.enabled?'checked':''}> on</label>
    <span class="grow"></span><span class="note">${r.hits||0} hits</span><button data-up="${i}" title="earlier">↑</button><button data-down="${i}" title="later">↓</button><button class="bad" data-del="${i}">Delete</button></div>
    ${(r.args||[]).map(c).join('')}
    <div class="row" style="margin-top:.3rem"><button data-cadd="${i}">+ argument condition</button><input data-k="max_per_hour" type="number" min="1" value="${r.max_per_hour||''}" placeholder="max calls / hour" style="width:9rem" title="allow rules only: after N calls in an hour it asks">
      <input data-k="expires" type="datetime-local" value="${esc(exp)}" style="width:13rem" title="stops applying after this"><input data-k="note" value="${esc(r.note||'')}" placeholder="note" class="grow"></div></div>`; }
function readRules(){ const out={default:$('#dflt').value,categories:{},rules:[]};
  document.querySelectorAll('[data-c]').forEach(e=>{ if(e.tagName==='SELECT') out.categories[e.dataset.c]=e.value; });
  document.querySelectorAll('.rule').forEach((el,i)=>{ const g=k=>el.querySelector(`:scope > .row [data-k="${k}"], :scope > .row > [data-k="${k}"]`);
    const old=RULES.rules[+el.dataset.r]||{};
    const r={id:old.id,hits:old.hits,recent:old.recent,action:g('action').value,server:g('server').value||'*',tool:g('tool').value||'*',kind:g('kind').value,enabled:g('enabled').checked,
      note:el.querySelector('[data-k="note"]').value,args:[]};
    const mph=el.querySelector('[data-k="max_per_hour"]').value; if(mph) r.max_per_hour=+mph;
    const ex=el.querySelector('[data-k="expires"]').value; if(ex) r.expires=new Date(ex).getTime()/1000;
    el.querySelectorAll('.cond').forEach(c=>r.args.push({path:c.querySelector('[data-k="path"]').value||'*',regex:c.querySelector('[data-k="regex"]').value,negate:c.querySelector('[data-k="negate"]').checked}));
    out.rules.push(r); }); return out; }
function drawRules(){ $('#rules').innerHTML=RULES.rules.map(ruleHtml).join('')||'<div class="note">No rules yet: every call follows the quick settings and the default.</div>';
  const redo=f=>b=>b.onclick=()=>{ RULES=readRules(); f(+(b.dataset.del??b.dataset.up??b.dataset.down??b.dataset.cadd)); drawRules(); };
  document.querySelectorAll('[data-del]').forEach(redo(i=>RULES.rules.splice(i,1)));
  document.querySelectorAll('[data-up]').forEach(redo(i=>{ if(i>0) [RULES.rules[i-1],RULES.rules[i]]=[RULES.rules[i],RULES.rules[i-1]]; }));
  document.querySelectorAll('[data-down]').forEach(redo(i=>{ if(i<RULES.rules.length-1) [RULES.rules[i+1],RULES.rules[i]]=[RULES.rules[i],RULES.rules[i+1]]; }));
  document.querySelectorAll('[data-cadd]').forEach(redo(i=>RULES.rules[i].args.push({path:'*',regex:'',negate:false})));
  document.querySelectorAll('[data-cdel]').forEach(b=>b.onclick=()=>{ RULES=readRules(); const i=+b.closest('.rule').dataset.r; RULES.rules[i].args.splice(+b.dataset.cdel,1); drawRules(); }); }
const PRESETS=[
  {action:'deny',server:'*',tool:'*',args:[{path:'*',regex:'(?i)\\brm\\s+-rf\\b|mkfs|dd\\s+if=|:\\(\\)\\s*\\{',negate:false}],note:'destructive shell commands, anywhere in the arguments'},
  {action:'deny',server:'*',tool:'*',args:[{path:'*',regex:'(?i)(/\\.ssh/|/\\.gnupg/|id_rsa|\\.env\\b|/etc/shadow)',negate:false}],note:'secrets and keys'},
  {action:'allow',server:'*',tool:'*',kind:'read_only',args:[],max_per_hour:200,note:'read-only tools, up to 200 an hour'},
  {action:'allow',server:'filesystem',tool:'read_*',args:[{path:'path',regex:'^/home/[^/]+/(notes|src)/',negate:false}],note:'example: reading notes and source'},
];
$('#radd').onclick=()=>{ RULES=readRules(); RULES.rules.push({action:'allow',server:'*',tool:'*',kind:'any',args:[],enabled:true,note:''}); drawRules(); };
$('#rpreset').onclick=()=>{ RULES=readRules(); for(const p of PRESETS) RULES.rules.push({kind:'any',enabled:true,...JSON.parse(JSON.stringify(p))}); drawRules(); msg('#rmsg','added; check them, then Save'); };
$('#rsave').onclick=async()=>{ try{ const r=await api('/api/settings/tool-rules',{rules:readRules()}); RULES=r.tool_rules; cats(); drawRules(); msg('#rmsg','saved'); loadTools(); }catch(e){ msg('#rmsg',esc(e.message),true); } };
$('#ttest').onclick=async()=>{ try{ const r=await api('/api/settings/tool-rules/test',{server:$('#tserver').value,tool:$('#ttool').value,args:$('#targs').value||'{}',annotations:JSON.parse($('#tann').value)});
  $('#tout').innerHTML=`<span class="pill ${r.action}">${r.action}</span> ${esc(r.reason)} · kinds: ${esc(r.kinds.join(', '))} <span class="note">(saved rules; Save first to test edits)</span>`; }catch(e){ $('#tout').innerHTML=`<span class="err">${esc(e.message)}</span>`; } };
async function loadTools(){ const s=await api('/api/settings'); ST.tools=s.tools;
  $('#tlist').innerHTML=!s.mcp_configured?'No MCP servers yet: add them in <a href="/#tools">Studio → Tools</a>.':!s.tools.length?'No server connected right now (<a href="/#tools">Studio → Tools</a>).':
    `<table><tr><th>server</th><th>tool</th><th>kinds</th><th>with no arguments</th></tr>`+s.tools.map(t=>`<tr><td>${esc(t.server)}</td><td>${esc(t.tool)}</td><td>${esc(t.decision.kinds.join(', '))}</td><td><span class="pill ${t.decision.action}">${t.decision.action}</span> <span class="note">${esc(t.decision.reason)}</span></td></tr>`).join('')+'</table>'; }
(async()=>{
  try{ await loadWS(); }catch(e){ $('#wsform').innerHTML=`<div class="err">${esc(e.message)}</div>`; }
  try{ ST=await api('/api/settings'); RULES=ST.tool_rules; subjTable(); cats(); drawRules(); loadTools(); }catch(e){ $('#subjtbl').innerHTML=`<tr><td class="err">${esc(e.message)}</td></tr>`; }
  if(location.hash){ const t=document.querySelector(location.hash); if(t) t.scrollIntoView(); }
})();

async function themes(){ const j=await (await fetch('/api/themes')).json(); const cur=nsThemeName();
  $('#themeGrid').innerHTML=Object.entries(j.themes).map(([k,t])=>{ const u=t.ui||{}, c=t.chart||[];
    return `<div class="th${k===cur?' on':''}" data-th="${k}"><b>${esc(t.label||k)}</b>${k===j.default?' <span class="note">(default)</span>':''}
      <div class="sw">${[u.bg,u.panel,u.acc,u.ok,u.no].filter(Boolean).map(x=>`<span style="background:${x}"></span>`).join('')||'<span class="note" style="padding:.4rem">built-in '+esc(t.base)+'</span>'}</div>
      <div class="sw" style="height:12px">${c.map(x=>`<span style="background:${x}"></span>`).join('')}</div>
      <div class="note">3D view: ${esc(t.viz||'dark')}</div>
      <div class="row" style="margin-top:.3rem"><button data-def="${k}">Make default</button></div></div>`; }).join('');
  document.querySelectorAll('[data-th]').forEach(el=>el.onclick=e=>{ if(e.target.dataset.def) return; nsApplyTheme(el.dataset.th, j.themes[el.dataset.th]); themes(); });
  document.querySelectorAll('[data-def]').forEach(b=>b.onclick=async()=>{ const r=await fetch('/api/settings/theme',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theme:b.dataset.def})});
    $('#themeMsg').textContent=r.ok?'Default theme saved.':(await r.json()).error; themes(); }); }
themes();

let SH=null;
async function shLoad(){ const r=await fetch('/api/settings/sharing'); SH=await r.json(); const st=SH.settings;
  $('#shOn').checked=st.enabled; $('#shEp').value=st.endpoint; $('#shAuto').checked=st.auto_listed; $('#shUnl').checked=st.auto_unlisted;
  $('#shModels').innerHTML='<tr><th>model</th><th>Hugging Face</th><th>public</th></tr>'+SH.models.map(m=>{ const c=m.check;
    const hf=c?(c.listed?`<span class="okm">found in ${esc(c.repo)}</span>`:`<span class="note">${esc(c.reason)}</span>`):'<span class="note">not checked</span>';
    const pub=c&&c.listed?'yes (listed)':`<label style="margin:0"><input type="checkbox" data-pub="${esc(m.id)}"${st.public_models.includes(m.id)?' checked':''}> list this unlisted model</label>`;
    return `<tr><td>${esc(m.id)}</td><td>${hf} <button data-chk="${esc(m.path)}">Check</button></td><td>${pub}</td></tr>`; }).join('');
  document.querySelectorAll('[data-chk]').forEach(b=>b.onclick=async()=>{ b.disabled=true; b.textContent='checking…';
    await fetch('/api/settings/sharing/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:b.dataset.chk})}); shLoad(); });
  $('#shRes').innerHTML=SH.results.length?SH.results.flatMap(r=>r.labels.map(l=>`<option value="${esc(r.file)}|${esc(l.label)}">${esc(r.file.split('/').pop())} · ${esc(l.label)}${l.model?' ('+esc(l.model)+')':''}</option>`)).join(''):'<option value="">no finished TestQA runs yet</option>';
  $('#shModel').innerHTML=SH.models.map(m=>`<option value="${esc(m.path)}">${esc(m.id)}</option>`).join('');
  $('#shHist').innerHTML=SH.history.length?'<tr><th>file</th><th>model</th><th>visibility</th><th>answer</th></tr>'+SH.history.map(h=>`<tr><td>${esc(h.file)}</td><td>${esc(h.model)}</td><td>${esc(h.visibility)}</td><td>${esc(h.answer?h.answer.status+(h.answer.shown_publicly?' · shown':''):'')}</td></tr>`).join(''):'<tr><td class="note">Nothing sent.</td></tr>';
}
$('#shSave').onclick=async()=>{ const pub=[...document.querySelectorAll('[data-pub]')].filter(x=>x.checked).map(x=>x.dataset.pub);
  const r=await fetch('/api/settings/sharing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:$('#shOn').checked,endpoint:$('#shEp').value,auto_listed:$('#shAuto').checked,auto_unlisted:$('#shUnl').checked,public_models:pub})});
  $('#shMsg').textContent=r.ok?'Saved.':(await r.json()).error; if(r.ok) shLoad(); };
async function shAct(kind){ const [results,label]=($('#shRes').value||'|').split('|');
  $('#shOut').textContent=kind==='preview'?'building the card (hashing the model file the first time)…':'sending…';
  const r=await fetch('/api/settings/sharing/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({results,label,path:$('#shModel').value})});
  const j=await r.json(); $('#shOut').textContent=r.ok?JSON.stringify(j,null,1):'error: '+j.error; if(kind==='submit') shLoad(); }
$('#shPrev').onclick=()=>shAct('preview'); $('#shSend').onclick=()=>shAct('submit');
shLoad();
async function plLoad(){ let r; try{ r=await api('/api/plugins'); }catch(e){ $('#plList').innerHTML=`<tr><td class="err">${esc(e.message)}</td></tr>`; return; }
  $('#plDirs').innerHTML='Looked in: '+r.dirs.map(d=>'<code>'+esc(d)+'</code>').join(', ');
  const st={on:'<span style="color:var(--ok)">on</span>',off:'off',changed:'<span class="err">changed since you turned it on: off</span>',broken:'<span class="err">broken</span>'};
  $('#plList').innerHTML=r.plugins.length?'<tr><th>plugin</th><th>adds</th><th>status</th><th></th></tr>'+r.plugins.map(p=>{ const a=p.adds||{}, adds=[...(a.jobs||[]).map(x=>'job: '+x),...(a.checks||[]).map(x=>'check: '+x),a.page?'a page':'',a.mcp?'MCP tools':''].filter(Boolean);
    return `<tr><td><b>${esc(p.name)}</b> ${p.version?'<span class="note">'+esc(p.version)+'</span>':''}<div class="note">${esc(p.description||p.error||'')}</div><div class="note"><code>${esc(p.folder)}</code></div></td><td class="note">${adds.map(esc).join('<br>')}</td><td>${st[p.status]||esc(p.status)}</td><td>${p.status==='broken'?'':`<button data-pl="${esc(p.name)}" data-on="${p.status==='on'?0:1}">${p.status==='on'?'Turn off':(p.status==='changed'?'Review and turn on':'Turn on')}</button>`}${p.status==='on'&&a.page?` <a href="/plugins/${encodeURIComponent(p.name)}/">open page</a>`:''}</td></tr>`; }).join(''):'<tr><td class="note">No plugins found. Put plugin folders in the first folder above.</td></tr>';
  document.querySelectorAll('[data-pl]').forEach(b=>b.onclick=async()=>{ const on=b.dataset.on==='1';
    if(on&&!confirm('Turn on '+b.dataset.pl+'? It runs code on this computer with your permissions. Turn it on only if you trust where it came from.')) return;
    try{ await api('/api/plugins/enable',{name:b.dataset.pl,on}); msg('#plMsg',on?'On. Its jobs are under Jobs → Plugins.':'Off.'); }catch(e){ msg('#plMsg',esc(e.message),true); } plLoad(); }); }
plLoad();
</script>
"""


import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ui_links import inject as _inject_links  # noqa: E402
PAGE = _inject_links(PAGE)
