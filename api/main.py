"""
LexProbe FastAPI Service
Exposes a single endpoint: POST /analyze-contract
Accepts raw contract text, returns flagged clauses with extracted evidence.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os

from api.model import load_model, analyze, CLAUSE_TYPES
from api.extractor import extract_text


# ── Lifespan: load model once on startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LexProbe — Contract Risk Flagging API",
    description=(
        "Fine-tuned Llama-3.1-8B (QLoRA) that detects risky clauses in commercial contracts. "
        f"Supported clause types: {', '.join(CLAUSE_TYPES)}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (JS, CSS if added later)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Constants & Limits ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB upload ceiling
MAX_TEXT_CHARS      = 15_000           # ~4-5 contract pages (~3,500 tokens)


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
    clause_type:    str
    detected:       bool
    extracted_text: str | None = None


class AnalyzeResponse(BaseModel):
    flagged_count:   int
    total_checked:   int
    clauses:         list[ClauseResult]
    contract_text:   str | None = None  # returned for file uploads (used for section detection)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/clause-types", tags=["Info"])
def clause_types():
    """Return the list of clause types the model can detect."""
    return {"clause_types": CLAUSE_TYPES}


@app.post("/analyze-contract", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze_contract(request: AnalyzeRequest):
    """
    Analyze a contract excerpt and flag risky clauses.
    Accepts raw contract text as JSON (max 15,000 characters). Use /upload for file uploads.
    """
    try:
        raw_results = analyze(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    clauses = [ClauseResult(**r) for r in raw_results]
    flagged = [c for c in clauses if c.detected]
    return AnalyzeResponse(flagged_count=len(flagged), total_checked=len(clauses), clauses=clauses)


@app.post("/upload", response_model=AnalyzeResponse, tags=["Analysis"])
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a contract file (.pdf, .docx, .doc, .txt) and flag risky clauses.
    Text is extracted server-side before inference.
    Enforces a 2 MB file size limit and a 15,000 character extracted text limit.
    """
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / (1024 * 1024):.2f} MB). Maximum allowed size is 2 MB.",
        )

    try:
        text = extract_text(file.filename, content)
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

    try:
        raw_results = analyze(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    clauses = [ClauseResult(**r) for r in raw_results]
    flagged = [c for c in clauses if c.detected]
    return AnalyzeResponse(
        flagged_count=len(flagged),
        total_checked=len(clauses),
        clauses=clauses,
        contract_text=text,   # send back so frontend can do section detection
    )
