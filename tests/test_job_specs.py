"""Every job form maps onto real flags of its script, so a job started from the
UI or MCP cannot fail on an unknown argument."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
import jobs  # noqa: E402


@pytest.mark.parametrize("kind", sorted(jobs.SPECS))
def test_job_fields_exist_in_the_script(kind):
    spec = jobs.SPECS[kind]
    argv = [*(spec.get("interp") or [sys.executable]), *[str(x) for x in spec["argv"]], "--help"]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=ROOT)
    if r.returncode != 0 and "ModuleNotFoundError" in r.stderr:
        pytest.skip(r.stderr.strip().splitlines()[-1])        # an optional package missing here
    assert r.returncode == 0, r.stderr[-400:]
    missing = [f["flag"] for f in spec["fields"] if f["flag"] not in r.stdout]
    assert not missing, f"{kind}: {missing}"


def test_build_job_runs_bash_and_values_cannot_inject_flags():
    argv = jobs.build_argv("build_llama", {"backend": "hip", "portable": True})
    assert argv[0] == "bash" and argv[-3:] == ["--backend", "hip", "--portable"]
    with pytest.raises(ValueError):
        jobs.build_argv("build_llama", {"backend": "--evil"})
    with pytest.raises(ValueError):
        jobs.build_argv("split", {"exclude": "--rm"})
