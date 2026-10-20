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
