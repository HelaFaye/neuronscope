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
]

JS = r"""
// ---- links between pages (viz/ui_links.py)
const NS_LINKS=__TARGETS__;
function linkify(html){ let out=String(html??'');
  for(const [rx,href] of NS_LINKS){ out=out.replace(new RegExp('(?<![">])('+rx+')(?![^<]*</a>)','g'),`<a href="${href}">$1</a>`); }
  return out; }
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
