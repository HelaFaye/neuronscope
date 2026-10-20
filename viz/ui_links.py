"""Links between Studio's pages. When a message or a fix says "Setup -> Build
llama.cpp" or "Settings → Workspaces", the page shows it as a link to that
section and option. Every page includes the same linkify(), so a message from
the server reads as a link wherever it is shown.

Targets:
  /setup#paths  #build  #hardware  #health  #requirements  #snapshots
  /settings#workspaces  #subjects  #tools
  /jobs#kind=<kind>&<field>=<value>…   (a job form, filled in)
  /#chats  #models  #hub  #docs  #tools  /#model=<id>&field=<setting>
  /review  /lab  /projects  /connect
"""

# (regex over escaped text, href). Arrows: -> → >. Order matters: specific first.
TARGETS = [
    (r"Setup\s*(?:->|→|&gt;)\s*Build(?: llama\.cpp)?", "/setup#build"),
    (r"Setup\s*(?:->|→|&gt;)\s*Hardware", "/setup#hardware"),
    (r"Setup\s*(?:->|→|&gt;)\s*Requirements", "/setup#requirements"),
    (r"Setup\s*(?:->|→|&gt;)\s*Health check", "/setup#health"),
    (r"Setup\s*(?:->|→|&gt;)\s*(?:Paths|routing thresholds)", "/setup#paths"),
    (r"Settings\s*(?:->|→|&gt;)\s*Workspaces", "/settings#workspaces"),
    (r"Settings\s*(?:->|→|&gt;)\s*Models per subject", "/settings#subjects"),
    (r"Settings\s*(?:->|→|&gt;)\s*Tool approval", "/settings#tools"),
    (r"Jobs\s*(?:->|→|&gt;)\s*Review", "/jobs#kind=review_testqa"),
    (r"Jobs\s*(?:->|→|&gt;)\s*Evaluate", "/jobs#kind=testqa"),
    (r"Jobs\s*(?:->|→|&gt;)\s*H-Neurons", "/jobs#kind=preflight"),
    (r"Studio\s*(?:->|→|&gt;)\s*Tools", "/#tools"),
    (r"Studio\s*(?:->|→|&gt;)\s*Models", "/#models"),
    (r"Studio\s*(?:->|→|&gt;)\s*Docs", "/#docs"),
    (r"Remote\s*(?:->|→|&gt;)\s*Add a host", "/remote#add"),
    (r"Workers page", "/workers"),
]

JS = r"""
// ---- links between pages (viz/ui_links.py)
const NS_LINKS=__TARGETS__;
function linkify(html){ let out=String(html??'');
  for(const [rx,href] of NS_LINKS){ out=out.replace(new RegExp('(?<![">])('+rx+')(?![^<]*</a>)','g'),`<a href="${href}">$1</a>`); }
  return out; }
// ---- themes (viz/themes.json "_studio"): UI colours, chart colours, the 3D viewer's theme
const NS_UI_VARS=['bg','panel','fg','mut','line','acc','accfg','ok','no','warn','soft','code'];
const NS_DEFAULT_CHART=['#6ea8e0','#ff8b7e','#5fcf8f','#e7b65c','#b48ead','#5dcaa5'];
function nsApplyTheme(name,t){ const d=document.documentElement; d.dataset.theme=t.base||'dark';
  NS_UI_VARS.forEach(k=>d.style.removeProperty('--'+k));
  for(const [k,v] of Object.entries(t.ui||{})) d.style.setProperty('--'+k,v);
  try{ localStorage.setItem('ns-theme',name); localStorage.setItem('ns-theme-base',t.base||'dark');
    localStorage.setItem('ns-theme-vars',JSON.stringify(t.ui||{})); localStorage.setItem('ns-chart',JSON.stringify(t.chart||NS_DEFAULT_CHART)); }catch(e){}
  document.cookie='ns_viz='+encodeURIComponent(t.viz||'dark')+'; path=/; SameSite=Strict; max-age=31536000';
  window.dispatchEvent(new CustomEvent('ns-theme',{detail:{name,theme:t}})); }
function nsChart(){ try{ return JSON.parse(localStorage.getItem('ns-chart')||'null')||NS_DEFAULT_CHART; }catch(e){ return NS_DEFAULT_CHART; } }
function nsThemeName(){ try{ return localStorage.getItem('ns-theme')||'midnight'; }catch(e){ return 'midnight'; } }
async function nsCycleTheme(){ const j=await (await fetch('/api/themes')).json(); const names=Object.keys(j.themes);
  const cur=nsThemeName(); const nx=names[(names.indexOf(cur)+1)%names.length]; nsApplyTheme(nx,j.themes[nx]); return nx; }
// ---- files and folders on Studio's computer: a browse dialog, path fields, per-entry lists
function nsEsc(x){ return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function nsRecent(kind){ try{ return JSON.parse(localStorage.getItem('ns-recent-'+kind)||'[]'); }catch(e){ return []; } }
function nsRemember(kind,p){ try{ const r=[p,...nsRecent(kind).filter(x=>x!==p)].slice(0,8); localStorage.setItem('ns-recent-'+kind,JSON.stringify(r)); }catch(e){} }
function nsFmtSize(b){ if(b==null) return ''; const u=['B','KB','MB','GB','TB']; let i=0; while(b>=1024&&i<4){ b/=1024; i++; } return (i?b.toFixed(1):b)+' '+u[i]; }
// opts: {want:'dir'|'file'|'any', ext:['gguf'], start:'/path', title:'Choose…'} -> Promise<path or null>
function nsPickPath(opts){ opts=opts||{}; const want=opts.want||'dir', ext=(opts.ext||[]).join(',');
  return new Promise(resolve=>{
    const back=document.createElement('div'); back.className='ns-pick';
    back.style.cssText='position:fixed;inset:0;z-index:9500;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;padding:12px';
    back.innerHTML=`<div role="dialog" aria-modal="true" aria-label="${nsEsc(opts.title||'Choose')}" style="width:min(860px,100%);height:min(620px,100%);display:flex;flex-direction:column;background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.4);font:14px/1.45 ui-sans-serif,system-ui;overflow:hidden">
     <div style="display:flex;align-items:center;gap:.5rem;padding:.7rem .9rem;border-bottom:1px solid var(--line,#33332f)"><b style="font-size:16px;flex:1">${nsEsc(opts.title||(want==='dir'?'Choose a folder':'Choose a file'))}</b><button data-p="x" aria-label="Close" style="background:none;border:none;color:inherit;font-size:20px;cursor:pointer">×</button></div>
     <div style="display:flex;flex:1;min-height:0">
      <nav data-p="places" style="width:200px;flex:none;overflow:auto;border-right:1px solid var(--line,#33332f);padding:.4rem;font-size:13px"></nav>
      <div style="flex:1;display:flex;flex-direction:column;min-width:0">
       <div style="display:flex;gap:.4rem;padding:.5rem;border-bottom:1px solid var(--line,#33332f)"><button data-p="up" title="Up one folder">↑</button><input data-p="path" aria-label="Path" style="flex:1;min-width:0;font:13px ui-monospace,monospace;padding:.3rem .45rem;border:1px solid var(--line,#33332f);border-radius:6px;background:var(--bg,#151514);color:inherit"><input data-p="filter" placeholder="Filter" aria-label="Filter" style="width:8rem;padding:.3rem .45rem;border:1px solid var(--line,#33332f);border-radius:6px;background:var(--bg,#151514);color:inherit"></div>
       <div data-p="list" role="listbox" style="flex:1;overflow:auto;padding:.25rem"></div>
      </div></div>
     <div style="display:flex;gap:.5rem;align-items:center;padding:.6rem .9rem;border-top:1px solid var(--line,#33332f)"><span data-p="sel" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:12px ui-monospace,monospace;color:var(--mut,#9a9a94)"></span>
      <label style="font-size:12px;color:var(--mut,#9a9a94)"><input type="checkbox" data-p="hidden"> hidden</label><button data-p="cancel">Cancel</button><button data-p="ok" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.35rem .9rem;font-weight:600;cursor:pointer"></button></div></div>`;
    document.body.appendChild(back);
    const q=k=>back.querySelector(`[data-p=${k}]`);
    if(innerWidth<640){ const nav=q('places'); nav.parentNode.style.flexDirection='column'; nav.style.cssText+=';width:auto;max-height:7.5rem;border-right:none;border-bottom:1px solid var(--line,#33332f);display:flex;flex-wrap:wrap;gap:.1rem .3rem;align-content:flex-start';
      q('filter').style.width='5rem'; }
    back.querySelectorAll('button').forEach(b=>{ if(!b.style.cssText) b.style.cssText='border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer'; });
    let cur=null, chosen=null, data=null;
    const done=v=>{ back.remove(); document.removeEventListener('keydown',key,true); if(v){ nsRemember(want,v); } resolve(v); };
    const setSel=p=>{ chosen=p; q('sel').textContent=p||''; q('ok').disabled=!p; q('ok').style.opacity=p?1:.5; };
    q('ok').textContent=want==='dir'?'Choose this folder':'Choose';
    const draw=()=>{ const f=q('filter').value.toLowerCase(); const rows=(data.entries||[]).filter(e=>!f||e.name.toLowerCase().includes(f));
      q('list').innerHTML=(data.error?`<div style="padding:.6rem;color:var(--no,#ff7a6b)">${nsEsc(data.error)}</div>`:'')+(rows.length?rows.map((e,i)=>`<div role="option" tabindex="-1" data-i="${i}" style="display:flex;gap:.6rem;align-items:center;padding:.3rem .5rem;border-radius:6px;cursor:pointer"><span aria-hidden="true">${e.dir?'📁':'📄'}</span><span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nsEsc(e.name)}</span><span style="font-size:12px;color:var(--mut,#9a9a94)">${nsFmtSize(e.size)}</span></div>`).join(''):`<div style="padding:.8rem;color:var(--mut,#9a9a94)">${want==='dir'?'No folders here.':'Nothing here matches.'}</div>`)+(data.truncated?'<div style="padding:.5rem;color:var(--mut,#9a9a94)">Only the first entries are shown; filter to narrow down.</div>':'');
      q('list').querySelectorAll('[data-i]').forEach(el=>{ const e=rows[+el.dataset.i];
        el.onmouseenter=()=>el.style.background='var(--code,#262624)'; el.onmouseleave=()=>{ if(chosen!==e.path) el.style.background=''; };
        el.onclick=()=>{ q('list').querySelectorAll('[data-i]').forEach(x=>x.style.background=''); el.style.background='var(--code,#262624)';
          if(e.dir){ if(want==='file') return go(e.path); setSel(e.path); } else setSel(e.path); };
        el.ondblclick=()=>{ if(e.dir) go(e.path); else done(e.path); }; }); };
    const go=async p=>{ q('list').innerHTML='<div style="padding:.8rem;color:var(--mut,#9a9a94)">…</div>';
      try{ const r=await fetch(`/api/fs/list?want=${want}&ext=${encodeURIComponent(ext)}&hidden=${q('hidden').checked?1:0}&path=${encodeURIComponent(p||'')}`); data=await r.json(); if(!r.ok) throw new Error(data.error||r.status); }
      catch(e){ data={entries:[],error:String(e.message||e),path:p}; }
      cur=data.path; q('path').value=cur||''; q('filter').value=''; setSel(want==='file'?null:cur); draw(); };
    q('filter').oninput=draw; q('hidden').onchange=()=>go(cur);
    q('path').onkeydown=e=>{ if(e.key==='Enter'){ e.preventDefault(); go(q('path').value); } };
    q('up').onclick=()=>data&&data.parent&&go(data.parent);
    q('x').onclick=q('cancel').onclick=()=>done(null); back.onclick=e=>{ if(e.target===back) done(null); };
    q('ok').onclick=()=>chosen&&done(chosen);
    const key=e=>{ if(e.key==='Escape'){ e.stopPropagation(); done(null); } }; document.addEventListener('keydown',key,true);
    fetch('/api/fs/places').then(r=>r.json()).then(j=>{ const rec=nsRecent(want);
      const item=(l,p)=>`<a href="#" data-go="${nsEsc(p)}" title="${nsEsc(p)}" style="display:block;padding:.3rem .45rem;border-radius:6px;color:inherit;text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${nsEsc(l)}</a>`;
      q('places').innerHTML=(rec.length?'<div style="font-size:11px;color:var(--mut,#9a9a94);margin:.3rem .45rem;text-transform:uppercase">Recent</div>'+rec.map(p=>item(p.split(/[\\/]/).filter(Boolean).pop()||p,want==='file'?p.replace(/[\\/][^\\/]*$/,'')||p:p)).join(''):'')
        +'<div style="font-size:11px;color:var(--mut,#9a9a94);margin:.5rem .45rem .3rem;text-transform:uppercase">Places</div>'+(j.places||[]).map(x=>item(x.label,x.path)).join('');
      q('places').querySelectorAll('[data-go]').forEach(a=>a.onclick=e=>{ e.preventDefault(); go(a.dataset.go); });
      go(opts.start||rec[0]&&(want==='file'?rec[0].replace(/[\\/][^\\/]*$/,''):rec[0])||(j.places||[])[0]?.path||''); }).catch(()=>go(opts.start||''));
  }); }
// An input with a Browse… button (and suggestions) next to it. opts as nsPickPath, plus suggest:[paths].
function nsPathField(input,opts){ if(!input||input.dataset.nsPick==='1') return; input.dataset.nsPick='1'; opts=opts||{};
  const wrap=document.createElement('span'); wrap.style.cssText='display:flex;gap:.35rem;align-items:center;width:100%';
  input.parentNode.insertBefore(wrap,input); wrap.appendChild(input); input.style.flex='1'; input.style.minWidth='0';
  const sug=[...new Set([...(opts.suggest||[]),...nsRecent(opts.want||'dir')])];
  if(sug.length){ const id='nsdl'+Math.random().toString(36).slice(2,8); const dl=document.createElement('datalist'); dl.id=id; dl.innerHTML=sug.map(x=>`<option value="${nsEsc(x)}">`).join(''); wrap.appendChild(dl); input.setAttribute('list',id); }
  const b=document.createElement('button'); b.type='button'; b.textContent='Browse…'; b.className='ns-browse';
  b.style.cssText='flex:none;border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border-radius:6px;padding:.28rem .6rem;cursor:pointer;font:500 12px ui-sans-serif,system-ui';
  b.onclick=async()=>{ const v=input.value.trim(); const p=await nsPickPath({...opts,start:v?(opts.want==='file'?v.replace(/[\\/][^\\/]*$/,''):v):opts.start});
    if(p){ input.value=p; input.dispatchEvent(new Event('input',{bubbles:true})); input.dispatchEvent(new Event('change',{bubbles:true})); } };
  wrap.appendChild(b); return wrap; }
// A list edited one entry at a time: each row has its path, Browse…, ↑ ↓ and ×; "+ Add" adds a row.
// opts: {values, want, ext, addLabel, placeholder, onchange}. -> {get(), set(values), el}
function nsListEditor(container,opts){ opts=opts||{}; let vals=[...(opts.values||[])];
  const el=document.createElement('div'); el.className='ns-list'; container.innerHTML=''; container.appendChild(el);
  const changed=()=>opts.onchange&&opts.onchange(vals.filter(Boolean));
  const btn='border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border-radius:6px;padding:.2rem .5rem;cursor:pointer;font:500 12px ui-sans-serif,system-ui';
  const draw=()=>{ el.innerHTML=(vals.length?'':`<div style="font-size:12px;color:var(--mut,#9a9a94);padding:.2rem 0">${nsEsc(opts.empty||'None yet.')}</div>`)+vals.map((v,i)=>`<div data-r="${i}" style="display:flex;gap:.35rem;align-items:center;margin:.25rem 0"><input value="${nsEsc(v)}" placeholder="${nsEsc(opts.placeholder||'')}" aria-label="entry ${i+1}" style="flex:1;min-width:0;font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line,#33332f);border-radius:5px;background:var(--bg,#151514);color:var(--fg,#ececea)">
      ${opts.browse===false?'':`<button type="button" data-a="browse" style="${btn}">Browse…</button>`}<button type="button" data-a="up" title="Move up" aria-label="Move up" style="${btn}" ${i?'':'disabled'}>↑</button><button type="button" data-a="down" title="Move down" aria-label="Move down" style="${btn}" ${i<vals.length-1?'':'disabled'}>↓</button><button type="button" data-a="rm" title="Remove" aria-label="Remove" style="${btn};color:var(--no,#ff7a6b)">×</button></div>`).join('')
      +`<button type="button" data-a="add" style="${btn};margin-top:.2rem">${nsEsc(opts.addLabel||'+ Add')}</button>`;
    el.querySelectorAll('[data-r]').forEach(row=>{ const i=+row.dataset.r, inp=row.querySelector('input');
      inp.oninput=inp.onchange=()=>{ vals[i]=inp.value.trim(); changed(); };
      if(opts.browse!==false) row.querySelector('[data-a=browse]').onclick=async()=>{ const p=await nsPickPath({want:opts.want||'dir',ext:opts.ext,start:vals[i]||undefined}); if(p){ vals[i]=p; draw(); changed(); } };
      row.querySelector('[data-a=up]').onclick=()=>{ [vals[i-1],vals[i]]=[vals[i],vals[i-1]]; draw(); changed(); };
      row.querySelector('[data-a=down]').onclick=()=>{ [vals[i+1],vals[i]]=[vals[i],vals[i+1]]; draw(); changed(); };
      row.querySelector('[data-a=rm]').onclick=()=>{ vals.splice(i,1); draw(); changed(); }; });
    el.querySelector('[data-a=add]').onclick=async()=>{ if(opts.browse===false){ get(); vals.push(''); draw(); const ins=el.querySelectorAll('[data-r] input'); ins[ins.length-1].focus(); return; } const p=await nsPickPath({want:opts.want||'dir',ext:opts.ext}); if(p&&!vals.includes(p)){ vals.push(p); draw(); changed(); } }; };
  const get=()=>{ el.querySelectorAll('[data-r] input').forEach((inp,i)=>vals[i]=inp.value.trim()); return vals.filter(Boolean); };
  draw();
  return {el, get, set:v=>{ vals=[...v]; draw(); }}; }
// Inputs marked data-pick="dir|file|any" (data-ext="gguf,npz") get a Browse… button, also when added later.
// Textareas marked data-lines="dir|file|text" become per-entry lists; the textarea stays (hidden) as the value.
function nsEnhancePickers(root){ (root||document).querySelectorAll('input[data-pick]').forEach(i=>nsPathField(i,{want:i.dataset.pick,ext:(i.dataset.ext||'').split(',').filter(Boolean)}));
  (root||document).querySelectorAll('textarea[data-lines]').forEach(t=>{ if(t.dataset.nsPick==='1') return; t.dataset.nsPick='1'; t.hidden=true;
    const d=document.createElement('div'); t.after(d); const w=t.dataset.lines;
    nsListEditor(d,{values:t.value.split('\n').map(x=>x.trim()).filter(Boolean),want:w,browse:w!=='text',placeholder:t.placeholder,
      addLabel:t.dataset.add||(w==='dir'?'+ Add folder':w==='file'?'+ Add file':'+ Add'),empty:t.dataset.empty||'None yet.',
      onchange:v=>{ t.value=v.join('\n'); t.dispatchEvent(new Event('input',{bubbles:true})); }}); }); }
(function(){ const start=()=>{ nsEnhancePickers(document); try{ new MutationObserver(ms=>{ for(const m of ms) for(const n of m.addedNodes) if(n.nodeType===1){ if(n.matches&&n.matches('input[data-pick]')) nsEnhancePickers(n.parentNode); else nsEnhancePickers(n); } }).observe(document.body,{childList:true,subtree:true}); }catch(e){} };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start); else start(); })();
// ---- the tour: optional, offered once, restartable from the ? button on every page
const NS_TOUR=[
 {path:'/', sel:'header', title:'Welcome to NeuronScope Studio', body:'Studio runs your local models, lets you chat with them, and shows what happens inside them: which neurons fire, and how likely a reply is to be a confident guess. Two minutes, ten stops. Skip any time.'},
 {path:'/', sel:'[data-tab=local]', title:'Your models', body:'Every GGUF model in your folders. Pick one to load it, see its measured accuracy per subject (the radar), and set its engine options.'},
 {path:'/', sel:'#in', title:'Chat', body:'Ask anything. The model picker below chooses which model answers; auto picks the one with the best measured record for the question.'},
 {path:'/', sel:'label:has(#autoCheck),label:has(#confOn)', title:'How sure was it?', body:'With confidence on, Studio marks the words the model was unsure about. check replies scores each reply with the model\'s hallucination classifier and opens a 3D view of its neurons.'},
 {path:'/setup', sel:'#status-card', title:'Setup', body:'What is ready and the one thing to do next. Build llama.cpp here with one button, add model folders, and run the health check when something seems off.'},
 {path:'/jobs', sel:'#kind', title:'Jobs', body:'Longer work runs here: TestQA to measure models, retraining on what they get wrong, benchmarks, the logit lens and the decoding lab. Each job has a form; its log shows live.'},
 {path:'/review', sel:'#heat', title:'Neuron review', body:'Every checked reply leaves its activations here. Colour neurons by how they relate to right and wrong answers, per subject, over time.'},
 {path:'/projects', sel:'main', title:'Projects', body:'Paste a project\'s notes; the director splits it into tasks by skill. You approve the plan, worker models do the tasks, you review the results.'},
 {path:'/workers', sel:'#workers', title:'Workers', body:'Where those tasks run: models on this computer, servers you started elsewhere, and other computers running NeuronScope Remote. Pause, limit, route subjects, and watch GPU temperatures.'},
 {path:'/connect', sel:'main', title:'Connect your other apps', body:'Use these models from Cline, Claude Desktop or any OpenAI-compatible app, and give agents NeuronScope\'s tools over MCP. This page shows who controls whom.'},
 {path:'/settings', sel:'#appearance', title:'Make it yours', body:'Themes, workspaces, models per subject, tool approval, plugins and opt-in sharing. The ? button at the bottom right restarts this tour, and opens the help pages.'}];
function nsTourGet(){ try{ return JSON.parse(localStorage.getItem('ns-tour')||'null'); }catch(e){ return null; } }
function nsTourSet(v){ try{ localStorage.setItem('ns-tour',JSON.stringify(v)); }catch(e){} }
function nsTourEnd(state){ nsTourSet({state}); document.querySelectorAll('.ns-tour').forEach(e=>e.remove()); }
function nsTourStart(i){ nsTourSet({state:'running',i:i||0}); nsTourShow(); }
function nsTourShow(){ const t=nsTourGet(); if(!t||t.state!=='running') return; const st=NS_TOUR[t.i]; if(!st){ nsTourEnd('done'); return; }
  if(location.pathname!==st.path){ location.href=st.path; return; }
  document.querySelectorAll('.ns-tour').forEach(e=>e.remove());
  // The spotlight covers every element the step names (a comma list), or the screen's centre if none is here.
  const els=[...document.querySelectorAll(st.sel)].filter(e=>e.getClientRects().length), el=els[0], r=el?el.getBoundingClientRect():null;
  if(el&&el.scrollIntoView&&(r.top<0||r.bottom>innerHeight)) el.scrollIntoView({block:'center'});
  const rs=els.map(e=>e.getBoundingClientRect()), pad=6;
  const rr=rs.length?{left:Math.min(...rs.map(x=>x.left)),top:Math.min(...rs.map(x=>x.top)),right:Math.max(...rs.map(x=>x.right)),bottom:Math.max(...rs.map(x=>x.bottom))}:null;
  if(rr){ rr.width=rr.right-rr.left; rr.height=rr.bottom-rr.top; }
  const hole=document.createElement('div'); hole.className='ns-tour';
  hole.style.cssText=rr?`position:fixed;z-index:9998;left:${rr.left-pad}px;top:${rr.top-pad}px;width:${Math.min(rr.width+2*pad,innerWidth)}px;height:${Math.min(rr.height+2*pad,innerHeight*0.7)}px;border-radius:8px;box-shadow:0 0 0 9999px rgba(0,0,0,.55);outline:2px solid var(--acc,#6f93ff);pointer-events:none`
    :'position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,.55);pointer-events:none';
  const card=document.createElement('div'); card.className='ns-tour'; card.setAttribute('role','dialog'); card.setAttribute('aria-label',st.title);
  const below=rr&&rr.bottom+220<innerHeight, top=rr?(below?rr.bottom+14:Math.max(12,rr.top-14-200)):innerHeight/2-110;
  card.style.cssText=`position:fixed;z-index:9999;left:${Math.max(12,Math.min((rr?rr.left:innerWidth/2-170),innerWidth-352))}px;top:${Math.max(12,Math.min(top,innerHeight-230))}px;width:min(340px,calc(100vw - 24px));background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:10px;padding:.85rem .95rem;font:14px/1.5 ui-sans-serif,system-ui;box-shadow:0 8px 30px rgba(0,0,0,.35)`;
  const n=NS_TOUR.length;
  card.innerHTML=`<div style="font-size:11px;color:var(--mut,#9a9a94)">${t.i+1} of ${n}</div><b style="font-size:15px">${st.title}</b><div style="margin:.35rem 0 .7rem">${st.body}</div>
   <div style="display:flex;gap:.4rem;align-items:center"><button data-t="skip" style="background:none;border:none;color:var(--mut,#9a9a94);cursor:pointer;font:inherit;padding:0">End tour</button><span style="flex:1"></span>
   ${t.i?'<button data-t="back">Back</button>':''}<button data-t="next" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.3rem .8rem;cursor:pointer;font:600 13px ui-sans-serif,system-ui">${t.i+1<n?'Next':'Finish'}</button></div>`;
  card.querySelectorAll('button[data-t=back]').forEach(b=>b.style.cssText='border:1px solid var(--line,#33332f);background:none;color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer');
  document.body.append(hole,card);
  const go=d=>{ const k=t.i+d; if(k>=n) return nsTourEnd('done'); nsTourSet({state:'running',i:Math.max(0,k)}); nsTourShow(); };
  card.querySelector('[data-t=next]').onclick=()=>go(1); card.querySelector('[data-t=skip]').onclick=()=>nsTourEnd('skipped');
  const back=card.querySelector('[data-t=back]'); if(back) back.onclick=()=>go(-1);
  card.querySelector('[data-t=next]').focus();
  card.onkeydown=e=>{ if(e.key==='Escape') nsTourEnd('skipped'); if(e.key==='ArrowRight') go(1); if(e.key==='ArrowLeft'&&t.i) go(-1); }; }
function nsHelpButton(){ if(document.getElementById('ns-help')||window.self!==window.top) return;
  const b=document.createElement('button'); b.id='ns-help'; b.textContent='?'; b.title='Tour and help'; b.setAttribute('aria-label','Tour and help');
  b.style.cssText='position:fixed;right:14px;bottom:'+(location.pathname==='/'?'150':'14')+'px;z-index:9000;width:34px;height:34px;border-radius:50%;border:1px solid var(--line,#33332f);background:var(--panel,#1d1d1b);color:var(--fg,#ececea);font:700 15px ui-sans-serif,system-ui;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25)';
  const m=document.createElement('div'); m.id='ns-help-menu'; m.hidden=true;
  m.style.cssText='position:fixed;right:14px;bottom:'+(location.pathname==='/'?'190':'54')+'px;z-index:9000;background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--line,#33332f);border-radius:8px;padding:.35rem;font:13px ui-sans-serif,system-ui;box-shadow:0 4px 16px rgba(0,0,0,.3)';
  const here=NS_TOUR.findIndex(s=>s.path===location.pathname);
  m.innerHTML=`<a href="#" data-h="tour" style="display:block;padding:.3rem .6rem;color:inherit">Take the tour</a>${here>=0?'<a href="#" data-h="here" style="display:block;padding:.3rem .6rem;color:inherit">This page in the tour</a>':''}<a href="/help/GETTING_STARTED" style="display:block;padding:.3rem .6rem;color:inherit">Getting started</a><a href="/help/UI_AND_CLI" style="display:block;padding:.3rem .6rem;color:inherit">Where everything is</a><a href="/help/GLOSSARY" style="display:block;padding:.3rem .6rem;color:inherit">Glossary</a><a href="/help/ROLES" style="display:block;padding:.3rem .6rem;color:inherit">Roles and ports</a>`;
  b.onclick=()=>{ m.hidden=!m.hidden; };
  m.querySelector('[data-h=tour]').onclick=e=>{ e.preventDefault(); m.hidden=true; nsTourStart(0); };
  const h=m.querySelector('[data-h=here]'); if(h) h.onclick=e=>{ e.preventDefault(); m.hidden=true; nsTourStart(here); };
  document.body.append(b,m); }
function nsTourOffer(){ if(location.pathname!=='/'||nsTourGet()||window.self!==window.top) return;
  const o=document.createElement('div'); o.className='ns-tour'; o.setAttribute('role','dialog');
  o.style.cssText='position:fixed;right:14px;bottom:'+(location.pathname==='/'?'192':'56')+'px;z-index:9001;width:min(320px,calc(100vw - 28px));background:var(--panel,#1d1d1b);color:var(--fg,#ececea);border:1px solid var(--acc,#6f93ff);border-radius:10px;padding:.8rem .9rem;font:14px/1.5 ui-sans-serif,system-ui;box-shadow:0 8px 30px rgba(0,0,0,.35)';
  o.innerHTML='<b>New here?</b><div style="margin:.3rem 0 .6rem">A two-minute tour shows where everything is. It is optional; the ? button brings it back later.</div><div style="display:flex;gap:.4rem;flex-wrap:wrap"><button data-o="go" style="background:var(--acc,#6f93ff);color:var(--accfg,#0b0b0a);border:1px solid var(--acc,#6f93ff);border-radius:6px;padding:.3rem .8rem;cursor:pointer;font:600 13px ui-sans-serif,system-ui">Show me around</button><button data-o="later" style="border:1px solid var(--line,#33332f);background:none;color:inherit;border-radius:6px;padding:.3rem .7rem;cursor:pointer">Not now</button><button data-o="never" style="border:none;background:none;color:var(--mut,#9a9a94);cursor:pointer">Don\'t ask again</button></div>';
  o.querySelector('[data-o=go]').onclick=()=>{ o.remove(); nsTourStart(0); };
  o.querySelector('[data-o=later]').onclick=()=>o.remove();
  o.querySelector('[data-o=never]').onclick=()=>{ nsTourSet({state:'declined'}); o.remove(); };
  document.body.appendChild(o); }
(function(){ const init=()=>{ nsHelpButton(); const t=nsTourGet(); if(t&&t.state==='running') setTimeout(nsTourShow,350); else nsTourOffer(); };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init(); })();
// Structured links a server reply may carry: [{href, label}].
function linkList(links){ return (links||[]).filter(l=>l&&typeof l.href==='string'&&l.href.startsWith('/'))
  .map(l=>`<a href="${l.href.replace(/"/g,'&quot;')}">${String(l.label||l.href).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</a>`).join(' · '); }
"""


def js() -> str:
    import json
    return JS.replace("__TARGETS__", json.dumps([[rx, href] for rx, href in TARGETS]))


def inject(page: str) -> str:
    """Add linkify() to a page, just before its last </script>."""
    i = page.rindex("</script>")
    return page[:i] + js() + page[i:]
