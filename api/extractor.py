"""
Text extraction from uploaded contract files.
Supports: .txt, .pdf, .docx. Legacy binary .doc is not supported (save as .docx).
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

    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            "Please upload a .pdf, .docx or .txt file (save legacy .doc files as .docx)."
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
