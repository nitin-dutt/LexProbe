"""The page's pure JavaScript (sections, highlight segments, citation text, report text) is tested with node."""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_pure_functions():
    for script, marker in (("test_app.js", "frontend tests passed"), ("test_report.js", "report tests passed")):
        out = subprocess.run(["node", os.path.join("tests", "js", script)], cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, script + ": " + out.stdout[-800:] + out.stderr[-800:]
        assert marker in out.stdout


def test_hidden_elements_stay_hidden():
    """Regression: `.progress { display: grid }` once overrode the hidden attribute, so the loading bar showed before any upload."""
    css = open(os.path.join(ROOT, "api", "static", "style.css"), encoding="utf-8").read()
    assert "[hidden] { display: none !important; }" in css
    html = open(os.path.join(ROOT, "api", "static", "index.html"), encoding="utf-8").read()
    for element_id in ("screen-busy", "screen-results", "error", "not-found", "new-btn", "drop-overlay", "back-link", "file-chip"):
        assert f'id="{element_id}"' in html and "hidden" in html.split(f'id="{element_id}"')[1].split(">")[0], element_id


def test_print_includes_the_marked_contract():
    """The PDF (browser print) must contain the contract with its marks, not hide the source pane."""
    css = open(os.path.join(ROOT, "api", "static", "style.css"), encoding="utf-8").read()
    printed = css[css.rindex("@media print"):]
    assert ".doc-col { display: block !important;" in printed and "mark.hl {" in printed and "max-height: none" in printed


def test_page_files_are_revalidated_not_cached_blindly():
    """Regression: a browser kept an old style.css against a new index.html and showed a broken layout."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    for path in ("/", "/static/style.css", "/static/app.js"):
        assert client.get(path).headers.get("cache-control") == "no-cache", path
