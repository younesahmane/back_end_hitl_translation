import os
import uuid
import re
import statistics
from typing import List, Dict, Optional

try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    from pypdf import PdfReader
    USE_PDFPLUMBER = False

from docx import Document

MAX_SEGMENT_LENGTH = 1200  # safety valve for very long paragraphs


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

    limited_blocks: List[str] = []
    for block in text_blocks:
        block = (block or "").strip()
        if not block:
            continue
        limited_blocks.extend(_split_by_char_limit(block, MAX_SEGMENT_LENGTH))

    segments: List[Dict] = []
    idx = 1
    for block in limited_blocks:
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


def _split_by_char_limit(text: str, max_length: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) > max_length:
            if current:
                chunks.append(current.strip())
                current = ""
            words = s.split()
            buf = ""
            for w in words:
                if not buf:
                    buf = w
                elif len(buf) + 1 + len(w) <= max_length:
                    buf += " " + w
                else:
                    chunks.append(buf.strip())
                    buf = w
            if buf:
                chunks.append(buf.strip())
            continue

        trial = (current + " " + s).strip() if current else s
        if len(trial) <= max_length:
            current = trial
        else:
            if current:
                chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())
    return chunks


# -----------------------------
# PDF paragraph extraction
# -----------------------------

def _extract_pdf_paragraphs(path: str) -> List[str]:
    if USE_PDFPLUMBER:
        try:
            paras = _pdfplumber_paragraphs(path)
            if paras:
                return paras
        except Exception:
            pass

    reader = PdfReader(path)
    full_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        full_text.append(t)

    text = "\n\n".join(full_text)
    return _smart_paragraph_split(text)


def _pdfplumber_paragraphs(path: str) -> List[str]:
    with pdfplumber.open(path) as pdf:
        all_paragraphs: List[str] = []

        for page in pdf.pages:
            words = page.extract_words(
                extra_attrs=["size", "fontname"],
                keep_blank_chars=False,
                use_text_flow=False
            )
            if not words:
                continue

            lines = _words_to_lines(words)
            if not lines:
                continue

            # Safer header/footer trimming (won't drop real first paragraph)
            lines = _drop_headers_footers(lines)

            page_paras = _lines_to_paragraphs(lines)
            all_paragraphs.extend(page_paras)

        cleaned: List[str] = []
        for p in all_paragraphs:
            p = _normalize_ws(p)
            if p:
                cleaned.append(p)
        return cleaned


def _words_to_lines(words: List[dict]) -> List[dict]:
    words_sorted = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))
    y_tol = 2.5

    lines: List[List[dict]] = []
    current: List[dict] = []
    current_top: Optional[float] = None

    for w in words_sorted:
        top = float(w.get("top", 0.0))
        if current_top is None:
            current_top = top
            current = [w]
            continue

        if abs(top - current_top) <= y_tol:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            current_top = top

    if current:
        lines.append(current)

    out: List[dict] = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w.get("x0", 0))
        text = " ".join(w.get("text", "") for w in line_words).strip()
        if not text:
            continue

        x0 = min(float(w.get("x0", 0.0)) for w in line_words)
        x1 = max(float(w.get("x1", 0.0)) for w in line_words)
        top = min(float(w.get("top", 0.0)) for w in line_words)
        bottom = max(float(w.get("bottom", w.get("top", 0.0))) for w in line_words)

        sizes = [float(w["size"]) for w in line_words if "size" in w and w["size"] is not None]
        size_med = statistics.median(sizes) if sizes else None

        out.append({
            "text": text,
            "top": top,
            "bottom": bottom,
            "x0": x0,
            "x1": x1,
            "size_med": size_med,
        })

    return out


def _drop_headers_footers(lines: List[dict]) -> List[dict]:
    """
    Much safer than before:
    - Remove pure page numbers at top/bottom.
    - Remove very short header/footer-like strings only if they're clearly not content.
    - Avoid dropping the first real paragraph (common failure in technical PDFs).
    """
    if len(lines) < 6:
        return lines

    tops = [ln["top"] for ln in lines]
    bottoms = [ln["bottom"] for ln in lines]
    page_top = min(tops)
    page_bottom = max(bottoms)
    height = page_bottom - page_top

    header_region = page_top + 0.07 * height
    footer_region = page_bottom - 0.07 * height

    def is_page_number(s: str) -> bool:
        s = s.strip()
        return bool(re.fullmatch(r"\d{1,4}", s))

    def is_obviously_not_content(s: str) -> bool:
        # Very short, mostly punctuation or boilerplate-like
        s = s.strip()
        if len(s) >= 45:
            return False
        if is_page_number(s):
            return True
        # strings like "arXiv:....", "© 2024", etc. can be headers/footers
        if re.search(r"arXiv:\s*\d", s):
            return True
        if re.search(r"^\(?[Cc]opyright\)?", s):
            return True
        # mostly non-letters/numbers (e.g. decorative)
        alnum = sum(ch.isalnum() for ch in s)
        if alnum <= max(3, int(0.25 * len(s))):
            return True
        return False

    filtered: List[dict] = []
    for idx, ln in enumerate(lines):
        t = (ln["text"] or "").strip()
        if not t:
            continue

        near_top = ln["top"] <= header_region
        near_bottom = ln["bottom"] >= footer_region

        # Only remove if it is VERY likely header/footer noise
        if (near_top or near_bottom) and is_obviously_not_content(t):
            continue

        filtered.append(ln)

    return filtered


def _ends_sentence(text: str) -> bool:
    """
    True if line ends with an end-of-sentence marker.
    Supports:
      - '.', '...', '…', '?', '!', Arabic '؟'
    Allows trailing quotes/brackets after punctuation.
    """
    t = (text or "").strip()
    if not t:
        return False

    # strip trailing quotes/brackets
    t = re.sub(r'[\s"\')\]\}»”’]+$', '', t)

    # ellipsis variants
    if t.endswith("...") or t.endswith("…"):
        return True

    return bool(re.search(r"[.!?؟]$", t))


def _lines_to_paragraphs(lines: List[dict]) -> List[str]:
    if not lines:
        return []

    gaps = []
    for a, b in zip(lines, lines[1:]):
        gap = float(b["top"]) - float(a["bottom"])
        if 0 <= gap <= 40:
            gaps.append(gap)

    median_gap = statistics.median(gaps) if gaps else 3.0
    gap_threshold = max(6.0, 2.2 * median_gap)

    indent_threshold = 12.0

    sizes = [ln["size_med"] for ln in lines if ln.get("size_med") is not None]
    median_size = statistics.median(sizes) if sizes else None

    paragraphs: List[str] = []
    current_lines: List[str] = []

    def flush():
        nonlocal current_lines
        if current_lines:
            paragraphs.append(_join_lines(current_lines))
            current_lines = []

    for i, ln in enumerate(lines):
        text = (ln["text"] or "").strip()
        if not text:
            continue

        # Headings still force boundaries (independent of punctuation)
        if _is_heading_line(ln, median_size):
            flush()
            paragraphs.append(_normalize_ws(text))
            continue

        if not current_lines:
            current_lines = [text]
            continue

        prev = lines[i - 1]
        prev_text = (prev.get("text") or "").strip()

        gap = float(ln["top"]) - float(prev["bottom"])
        is_big_gap = gap > gap_threshold

        is_bullet = _looks_like_list_item(text)

        indent = float(ln["x0"]) - float(prev["x0"])
        is_indent_jump = indent > indent_threshold

        prev_ended_sentence = _ends_sentence(prev_text)

        # Lists: start new paragraph always
        if is_bullet:
            flush()
            current_lines = [text]
            continue

        # IMPORTANT CHANGE:
        # Only accept a paragraph break caused by spacing/indent
        # if the previous line ends a sentence.
        if (is_big_gap or is_indent_jump) and prev_ended_sentence:
            flush()
            current_lines = [text]
        else:
            current_lines.append(text)

    flush()
    paragraphs = _post_merge_fragments(paragraphs)
    return paragraphs


def _is_heading_line(line: dict, median_size: Optional[float]) -> bool:
    t = (line.get("text") or "").strip()
    if not t:
        return False
    if len(t) > 80:
        return False
    if re.fullmatch(r"\d+(\.\d+)*", t):
        return False

    size = line.get("size_med")
    if median_size is None or size is None:
        return bool(re.match(r"^\d+(\.\d+)*\s+\S+", t)) and len(t) <= 60

    return size >= (median_size + 1.2) and bool(re.search(r"[A-Za-z]", t))


def _looks_like_list_item(text: str) -> bool:
    return bool(re.match(r"^\s*(?:[-•*]|(\d+[\.\)])|[a-zA-Z][\.\)])\s+\S+", text))


def _join_lines(lines: List[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
            continue
        if out.endswith("-") and ln and ln[0].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return _normalize_ws(out)


def _post_merge_fragments(paragraphs: List[str]) -> List[str]:
    if not paragraphs:
        return paragraphs

    merged: List[str] = []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i].strip()
        if not p:
            i += 1
            continue

        if i < len(paragraphs) - 1:
            nxt = paragraphs[i + 1].strip()
            if len(p) < 40 and nxt and nxt[0].islower() and not _ends_sentence(p):
                merged.append(_normalize_ws(p + " " + nxt))
                i += 2
                continue

        merged.append(p)
        i += 1

    return merged


def _normalize_ws(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


# -----------------------------
# DOCX paragraphs
# -----------------------------

def _extract_docx_paragraphs(path: str) -> List[str]:
    doc = Document(path)
    paras = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            paras.append(t)
    return paras


# -----------------------------
# pypdf fallback paragraph split
# -----------------------------

def _smart_paragraph_split(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    if re.search(r"\n\s*\n", text):
        parts = re.split(r"\n\s*\n+", text)
        return [p.strip() for p in parts if p.strip()]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    blocks: List[str] = []
    buf = ""
    target = 850

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= target:
            buf += " " + s
        else:
            blocks.append(buf.strip())
            buf = s

    if buf:
        blocks.append(buf.strip())

    return blocks
