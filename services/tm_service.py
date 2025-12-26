import os
import json
import uuid
from typing import Any

def _atomic_write(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load(path: str, default: Any):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tm_add(tm_json_path: str, source: str, target: str, language_pair: str) -> str:
    tm = _load(tm_json_path, default=[])
    entry_id = f"tm_{uuid.uuid4().hex[:10]}"
    tm.append({
        "entry_id": entry_id,
        "language_pair": language_pair,
        "source": source,
        "target": target,
    })
    _atomic_write(tm_json_path, tm)
    return entry_id

def _score(query: str, text: str) -> float:
    # Very simple deterministic scoring; replace later with embeddings/FAISS.
    q = query.lower().strip()
    t = text.lower().strip()
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t:
        return min(0.95, 0.6 + (len(q) / max(1, len(t))) * 0.35)
    # overlap
    q_tokens = set(q.split())
    t_tokens = set(t.split())
    inter = len(q_tokens & t_tokens)
    union = len(q_tokens | t_tokens) or 1
    return 0.25 * (inter / union)

def tm_search(tm_json_path: str, query: str, language_pair: str, limit: int = 10) -> list[dict]:
    tm = _load(tm_json_path, default=[])
    scored = []
    for e in tm:
        if e.get("language_pair") != language_pair:
            continue
        s = _score(query, e.get("source", ""))
        if s > 0:
            scored.append({
                "source": e.get("source", ""),
                "target": e.get("target", ""),
                "score": round(float(s), 4),
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]
