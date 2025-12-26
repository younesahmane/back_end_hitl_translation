import os
import json
import threading
from typing import Any, Optional, Tuple

_LOCK = threading.Lock()

def ensure_storage_layout(cfg: dict) -> None:
    os.makedirs(cfg["STORAGE_DIR"], exist_ok=True)
    os.makedirs(cfg["UPLOADS_DIR"], exist_ok=True)
    os.makedirs(cfg["EXPORTS_DIR"], exist_ok=True)
    # docs.json and tm.json are created lazily

def _atomic_write_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_json(path: str, default: Any) -> Any:
    if not os.path.isfile(path):
        return default

    try:
        # Handle empty files gracefully
        if os.path.getsize(path) == 0:
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        _atomic_write_json(path, default)
        return default


def load_docs(docs_json_path: str) -> list[dict]:
    with _LOCK:
        return _read_json(docs_json_path, default=[])

def save_docs(docs_json_path: str, docs: list[dict]) -> None:
    with _LOCK:
        _atomic_write_json(docs_json_path, docs)

def segments_path(cfg: dict, doc_id: str) -> str:
    return os.path.join(cfg["STORAGE_DIR"], f"segments_{doc_id}.json")

def load_doc_segments(cfg: dict, doc_id: str) -> Optional[list[dict]]:
    path = segments_path(cfg, doc_id)
    with _LOCK:
        if not os.path.isfile(path):
            return None
        return _read_json(path, default=[])

def save_doc_segments(cfg: dict, doc_id: str, segments: list[dict]) -> None:
    path = segments_path(cfg, doc_id)
    with _LOCK:
        _atomic_write_json(path, segments)

def find_segment_by_id(cfg: dict, segment_id: str) -> Optional[Tuple[str, dict]]:
    docs = load_docs(cfg["DOCS_JSON"])
    for d in docs:
        doc_id = d["doc_id"]
        segs = load_doc_segments(cfg, doc_id) or []
        for s in segs:
            if s.get("id") == segment_id:
                return doc_id, s
    return None

def update_segment(cfg: dict, doc_id: str, updated_segment: dict) -> None:
    segs = load_doc_segments(cfg, doc_id)
    if segs is None:
        return
    sid = updated_segment.get("id")
    for i, s in enumerate(segs):
        if s.get("id") == sid:
            segs[i] = updated_segment
            save_doc_segments(cfg, doc_id, segs)
            return

def try_update_segments_from_batch_results(cfg: dict, frontend_payload: list[dict], orchestrator_output: dict) -> None:
    """
    If frontend payload items include 'segment_id', we update that segment's status/reason.
    This keeps UI in sync after /pipeline/process-batch.
    """
    results = orchestrator_output.get("results") or []
    # Build index->result mapping
    by_index = {}
    for r in results:
        idx = r.get("index")
        if isinstance(idx, int):
            by_index[idx] = r

    for i, item in enumerate(frontend_payload, start=1):
        segment_id = item.get("segment_id")
        if not segment_id:
            continue

        found = find_segment_by_id(cfg, segment_id)
        if not found:
            continue
        doc_id, seg = found

        r = by_index.get(i)
        if not r:
            continue

        corr = r.get("correction") or {}
        decision = corr.get("decision")
        reason = corr.get("reason") or corr.get("message") or ""

        if decision in ("accept", "reject"):
            seg["status"] = decision
            seg["reason"] = reason
            update_segment(cfg, doc_id, seg)
