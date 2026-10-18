"""API behaviour with the model replaced by a fake (alignment_plan E3, E5, E6, E7, E8, B3)."""

import pytest
from fastapi.testclient import TestClient

import api.main as main
import api.model as runtime

TEXT = "This Agreement shall be governed by the laws of the State of Delaware. " * 3


def fake_analyze(text):
    return {
        "contract_text": " ".join(text.split()),
        "chunks_analyzed": 1,
        "clauses": [
            {"clause_type": ct, "detected": ct == "Governing Law", "chunks_flagged": int(ct == "Governing Law"),
             "confidence_margin": 2.5 if ct == "Governing Law" else None,
             "extracted_text": "governed by the laws of the State of Delaware" if ct == "Governing Law" else None,
             "grounded": True if ct == "Governing Law" else None,
             "start": 24 if ct == "Governing Law" else None, "end": 69 if ct == "Governing Law" else None,
             "chunk_index": 0 if ct == "Governing Law" else None}
            for ct in runtime.CLAUSE_TYPES
        ],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(runtime, "load_model", lambda: None)
    monkeypatch.setattr(runtime, "is_ready", lambda: True)
    monkeypatch.setattr(runtime, "analyze", fake_analyze)
    with TestClient(main.app) as c:
        yield c


def test_health_reports_ready_and_adapter(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    assert "lora-v4" in r.json()["adapter"]


def test_scoring_mode_is_reported(client):
    assert client.get("/health").json()["scoring"] == runtime.SCORING_MODE
    body = client.post("/analyze-contract", json={"text": TEXT}).json()
    assert body["scoring"] == runtime.SCORING_MODE
    assert runtime.SCORING_MODE in ("blend_v1", "v4")


def test_default_scoring_is_the_validated_blend():
    """With no SCORING override the API serves the blend, with the threshold fitted on validation (not 0)."""
    assert runtime.SCORING == "blend" and runtime.SCORING_MODE == "blend_v1"
    assert runtime.DECISION_THRESHOLD == runtime.BLEND["threshold"] != 0.0


def test_threshold_is_reported(client):
    assert client.get("/health").json()["decision_threshold"] == runtime.DECISION_THRESHOLD
    assert client.post("/analyze-contract", json={"text": TEXT}).json()["decision_threshold"] == runtime.DECISION_THRESHOLD


def test_health_is_503_while_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(runtime, "is_ready", lambda: False)
    assert client.get("/health").status_code == 503


def test_clause_types_use_cuad_name(client):
    types = client.get("/clause-types").json()["clause_types"]
    assert "Liquidated Damages" in types and "Late Payment / Penalty" not in types


def test_analyze_returns_positions_and_grounding(client):
    r = client.post("/analyze-contract", json={"text": TEXT})
    body = r.json()
    assert r.status_code == 200 and body["flagged_count"] == 1 and body["total_checked"] == 6
    gl = next(c for c in body["clauses"] if c["detected"])
    assert gl["grounded"] is True
    assert body["contract_text"][gl["start"]:gl["end"]] == gl["extracted_text"]


def test_short_text_rejected(client):
    assert client.post("/analyze-contract", json={"text": "too short"}).status_code == 422


def test_internal_errors_are_not_leaked(client, monkeypatch):
    def boom(text):
        raise RuntimeError("CUDA error at C:/secret/path/model.py line 42")
    monkeypatch.setattr(runtime, "analyze", boom)
    r = client.post("/analyze-contract", json={"text": TEXT})
    assert r.status_code == 500
    assert "secret" not in r.text and "CUDA" not in r.text and "request" in r.json()["detail"]


def test_busy_model_returns_503(client, monkeypatch):
    def busy(text):
        raise runtime.ModelBusyError("The model is busy with another request. Please retry shortly.")
    monkeypatch.setattr(runtime, "analyze", busy)
    assert client.post("/analyze-contract", json={"text": TEXT}).status_code == 503


def test_oversized_upload_rejected(client):
    big = b"x" * (2 * 1024 * 1024 + 1)
    assert client.post("/upload", files={"file": ("big.txt", big, "text/plain")}).status_code == 413


def test_legacy_doc_upload_rejected(client):
    r = client.post("/upload", files={"file": ("old.doc", b"\xd0\xcf\x11\xe0" + b"x" * 100, "application/msword")})
    assert r.status_code == 400 and ".docx" in r.json()["detail"]


def test_txt_upload_analysed(client):
    r = client.post("/upload", files={"file": ("c.txt", TEXT.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["flagged_count"] == 1
