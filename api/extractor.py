"""
Text extraction from uploaded contract files.
Supports: .txt, .pdf, .docx, .doc (legacy)
"""

import io


def extract_text(filename: str, content: bytes) -> str:
    """
    Extract plain text from a contract file.
    Raises ValueError for unsupported formats or extraction failures.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return content.decode("utf-8", errors="replace")

    elif ext == "pdf":
        return _extract_pdf(content)

    elif ext == "docx":
        return _extract_docx(content)

    elif ext == "doc":
        return _extract_doc(content)

    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            "Please upload a .pdf, .docx, .doc, or .txt file."
        )


def _extract_pdf(content: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    pages = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF. "
            "The file may be a scanned image — please use a text-based PDF."
        )
    return "\n\n".join(pages)


def _extract_docx(content: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    doc = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("No text found in the .docx file.")
    return "\n".join(paragraphs)


def _extract_doc(content: bytes) -> str:
    """
    .doc (legacy binary Word format) support via python-docx2txt.
    Falls back gracefully if not installed.
    """
    try:
        import docx2txt
        import tempfile, os
        # docx2txt works on file paths, not bytes — write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text = docx2txt.process(tmp_path)
        finally:
            os.unlink(tmp_path)
        if not text or not text.strip():
            raise ValueError("No text extracted from .doc file.")
        return text
    except ImportError:
        raise ValueError(
            ".doc (legacy Word) support requires docx2txt. "
            "Run: pip install docx2txt  OR  save the file as .docx and re-upload."
        )
