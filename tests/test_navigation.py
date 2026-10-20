"""The sidebar is on every page, and every item in it leads somewhere real."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import ui_links  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401


def test_sidebar_items_lead_to_pages(studio_srv):
    js = ui_links.js()
    items = re.findall(r"\{href:'([^']+)',(?:tab:'(\w+)',)?label:'([^']+)'", js)
    labels = [i[2] for i in items]
    assert labels[:3] == ["Chat", "My Models", "Discover"] and "Settings" in labels
    _, home = call(studio_srv, "/")
    for href, tab, label in items:
        path = href.split("#")[0] or "/"
        code, page = call(studio_srv, path)
        assert code == 200 and "nsRail" in page, (label, path)
        if tab:
            assert f'data-tab="{tab}"' in home, label
            assert href.split("#")[1] in js, label                     # the main page's hash names it
    assert 'href="/jobs" class="chip"' not in home                     # the old header row is gone
    assert ">Discover</button>" in home
