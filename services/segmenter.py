import os
import uuid
from typing import List, Dict

from pypdf import PdfReader
from docx import Document

def _new_seg_id() -> str:
    return f"seg_{uuid.uuid4().hex[:12]}"

def segment_document_file(filepath: str, doc_id: str, segment_type: str = "paragraph") -> List[Dict]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        text_blocks = _extract_pdf_paragraphs(filepath)
    elif ext == ".docx":
        text_blocks = _extract_docx_paragraphs(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    segments: List[Dict] = []
    idx = 1
    for block in text_blocks:
        clean = (block or "").strip()
        if not clean:
            continue
        segments.append({
            "id": _new_seg_id(),
            "doc_id": doc_id,
            "index": idx,
            "segment_type": segment_type,
            "source": clean,
            "llm_outputs": {"model1": "", "model2": "", "model3": ""},
            "human_edit": "",
            "status": "pending",
            "reason": "",
            "last_saved_at": None,
        })
        idx += 1

    return segments

def _extract_pdf_paragraphs(path: str) -> list[str]:
    reader = PdfReader(path)
    full = []
    for page in reader.pages:
        t = page.extract_text() or ""
        full.append(t)
    text = "\n".join(full)

    # Simple paragraph heuristic: split by blank lines first; fallback to line grouping
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 3:
        return paras

    # fallback: group lines into pseudo-paragraphs
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    buf = []
    for ln in lines:
        buf.append(ln)
        if len(" ".join(buf)) > 280:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out

def _extract_docx_paragraphs(path: str) -> list[str]:
    doc = Document(path)
    paras = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            paras.append(t)
    return paras
