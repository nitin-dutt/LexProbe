"""The page's pure JavaScript (sections, highlight segments, citation text, report text) is tested with node."""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_pure_functions():
    out = subprocess.run(["node", os.path.join("tests", "js", "test_app.js")], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    assert "frontend tests passed" in out.stdout


def test_hidden_elements_stay_hidden():
    """Regression: `.progress { display: grid }` once overrode the hidden attribute, so the loading bar showed before any upload."""
    css = open(os.path.join(ROOT, "api", "static", "style.css"), encoding="utf-8").read()
    assert "[hidden] { display: none !important; }" in css
    html = open(os.path.join(ROOT, "api", "static", "index.html"), encoding="utf-8").read()
    for element_id in ("progress", "error", "results", "not-found"):
        assert f'id="{element_id}"' in html and "hidden" in html.split(f'id="{element_id}"')[1].split(">")[0], element_id
