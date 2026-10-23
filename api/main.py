"""
LexProbe FastAPI Service
Endpoints: POST /analyze-contract (JSON text), POST /upload (PDF/DOCX/TXT), GET /health, GET /clause-types.
The whole document is analysed chunk by chunk; each flagged clause comes with a quote, whether
that quote was found verbatim in the document, and its character position.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import model as runtime
from api.extractor import extract_text
from api.model import CLAUSE_TYPES

log = logging.getLogger("lexprobe.api")


# ── Lifespan: load model once on startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LexProbe — Contract Risk Flagging API",
    description=(
        "Fine-tuned Llama-3.1-8B (QLoRA) that detects risky clauses in commercial contracts. "
        f"Supported clause types: {', '.join(CLAUSE_TYPES)}"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Browser hardening for the web page. The page has no inline script or style, so a strict policy is possible.
# The interactive API docs (/docs, /redoc) load their assets from a CDN, so they are left out of the policy.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response


# ── Constants & Limits ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB upload ceiling
MAX_TEXT_CHARS      = 15_000           # ~4-5 contract pages; the whole text is analysed
UPLOAD_READ_CHUNK   = 64 * 1024


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        max_length=MAX_TEXT_CHARS,
        description=f"Raw contract text to analyze. Minimum 50 characters, maximum {MAX_TEXT_CHARS} characters (~5 pages).",
        examples=["This Agreement shall automatically renew for successive one-year terms..."],
    )


class ClauseResult(BaseModel):
    clause_type:       str
    detected:          bool
    extracted_text:    str | None = None
    grounded:          bool | None = Field(None, description="True if the quote appears verbatim in the document")
    start:             int | None = Field(None, description="Quote start offset in contract_text")
    end:               int | None = Field(None, description="Quote end offset in contract_text")
    chunk_index:       int | None = None
    chunks_flagged:    int = 0
    confidence_margin: float | None = Field(None, description="logit(Found) - logit(None) of the best chunk")


class AnalyzeResponse(BaseModel):
    flagged_count:   int
    total_checked:   int
    chunks_analyzed: int
    model_version:   str
    scoring:         str = Field(..., description="Scoring mode: 'v6b' (default, one pass), 'blend_v1' (v4 + zero-shot blend, 6 clauses) or 'v4'")
    decision_threshold: float = Field(..., description="Score above which a chunk counts as containing the clause")
    clauses:         list[ClauseResult]
    contract_text:   str  # normalised text that start/end refer to


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _run_analysis(text: str) -> AnalyzeResponse:
    """Run the GPU work off the event loop and turn failures into safe HTTP errors."""
    try:
        result = await run_in_threadpool(runtime.analyze, text)
    except runtime.ModelBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        request_id = uuid.uuid4().hex[:8]
        log.exception("inference failed (request %s)", request_id)
        raise HTTPException(status_code=500, detail=f"Internal error during analysis (request {request_id}).")

    clauses = [ClauseResult(**c) for c in result["clauses"]]
    return AnalyzeResponse(
        flagged_count=sum(c.detected for c in clauses),
        total_checked=len(clauses),
        chunks_analyzed=result["chunks_analyzed"],
        model_version=os.path.basename(os.path.dirname(runtime.ADAPTER_PATH.rstrip("/\\"))) or runtime.ADAPTER_PATH,
        scoring=runtime.SCORING_MODE,
        decision_threshold=runtime.DECISION_THRESHOLD,
        clauses=clauses,
        contract_text=result["contract_text"],
    )


async def _read_limited(file: UploadFile) -> bytes:
    """Read an upload in pieces and stop as soon as it exceeds the size limit."""
    data = bytearray()
    while True:
        piece = await file.read(UPLOAD_READ_CHUNK)
        if not piece:
            return bytes(data)
        data.extend(piece)
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 2 MB.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", tags=["Health"])
def health():
    ready = runtime.is_ready()
    body = {"status": "healthy" if ready else "loading", "model_loaded": ready, "adapter": runtime.ADAPTER_PATH,
            "scoring": runtime.SCORING_MODE, "decision_threshold": runtime.DECISION_THRESHOLD}
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get("/clause-types", tags=["Info"])
def clause_types():
    """Return the list of clause types the model can detect."""
    return {"clause_types": CLAUSE_TYPES}


@app.post("/analyze-contract", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_contract(request: AnalyzeRequest):
    """Analyze contract text (max 15,000 characters). Use /upload for files."""
    return await _run_analysis(request.text)


@app.post("/upload", response_model=AnalyzeResponse, tags=["Analysis"])
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a contract file (.pdf, .docx, .txt) and flag risky clauses.
    Enforces a 2 MB file size limit and a 15,000 character extracted text limit.
    """
    content = await _read_limited(file)

    try:
        text = extract_text(file.filename or "", content)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Could not extract enough text from the file.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Extracted contract text ({len(text):,} characters) exceeds the "
                f"{MAX_TEXT_CHARS:,} character limit (~5 pages). Please upload an excerpt or shorter document."
            ),
        )
    return await _run_analysis(text)
