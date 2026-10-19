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
