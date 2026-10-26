"""System page: what runs, restarting parts, stale-code notice, forgiving page paths."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_studio import call, studio_srv  # noqa: E402,F401


def get_html(base, path):
    req = urllib.request.Request(base + path, headers={"Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_services_and_actions(studio_srv):
    code, j = call(studio_srv, "/api/services")
    ids = [s["id"] for s in j["services"]]
    assert code == 200 and ids[:2] == ["studio", "chat"] and {"workers", "embed", "mcp", "labs", "jobs"} <= set(ids)
    assert j["stale"] is False
    code, r = call(studio_srv, "/api/services", {"id": "workers", "action": "stop"})
    assert code == 200 and "stopped" in r["note"]
    code, r = call(studio_srv, "/api/services", {"id": "chat", "action": "reload"})
    assert code == 400 and "no model" in r["error"]
    code, r = call(studio_srv, "/api/services", {"id": "nope", "action": "explode"})
    assert code == 400


def test_page_paths_and_not_found(studio_srv, monkeypatch):
    import studio
    for p in ("/workers", "/workers/", "/workers?tab=1", "/system", "/jobs/"):
        assert get_html(studio_srv, p)[0] == 200, p
    code, page = get_html(studio_srv, "/no-such-page")
    assert code == 404 and "not in the running Studio" in page and "nsRestart" in page
    assert call(studio_srv, "/api/no-such")[0] == 404                       # APIs still answer JSON
    monkeypatch.setitem(studio.STARTED, "checked", 0)
    monkeypatch.setitem(studio.STARTED, "code", 0)                          # as if the code changed since start
    _, st = call(studio_srv, "/api/status")
    assert st["stale"] is True
    code, page = get_html(studio_srv, "/new-page")
    assert "changed after this Studio started" in page
