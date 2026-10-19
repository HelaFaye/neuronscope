"""Roles are stated wherever one program controls another; docs are readable inside Studio."""
import json
import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import help_page  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401


def test_help_pages_render_safely():
    html = help_page.render("# Title\n\nSee [ROLES.md](ROLES.md#ports), **bold** and `<x>`.\n\n"
                            "| a | b |\n|---|---|\n| 1 | [bad](javascript:alert(1)) |\n\n- one\n- two\n\n```\n<script>\n```")
    assert '<a href="/help/ROLES#ports">' in html and "<b>bold</b>" in html and "<code>&lt;x&gt;</code>" in html
    assert "javascript:" not in html and "<script>" not in html and "&lt;script&gt;" in html
    assert "<table>" in html and "<li>one</li>" in html
    assert help_page.page("../etc") is None and help_page.page("NOPE_NOT_HERE") is None
    for f in (ROOT / "docs").glob("*.md"):
        assert help_page.page(f.stem), f.stem                 # every doc renders


def test_roles_are_stated(studio_srv):
    code, page = call(studio_srv, "/help/ROLES")
    assert code == 200 and "controller" in page and "47871" in page
    code, st = call(studio_srv, "/api/status")
    assert st["computer"] == socket.gethostname() and "controller" in st["roles"]
    code, page = call(studio_srv, "/connect")
    assert "Who controls whom" in page and "CONTROLLER" in page and "/help/ROLES" in page
    req = urllib.request.Request(studio_srv + "/mcp", data=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    out = json.loads(urllib.request.urlopen(req).read())
    ins = out["result"]["instructions"]
    assert "CONTROLLER" in ins and "every tool acts on the computer Studio runs on" in ins
    assert call(studio_srv, "/help/NOPE")[0] == 404
