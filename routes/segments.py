from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

from services.storage import (
    ensure_storage_layout,
    find_segment_by_id,
    update_segment,
)

bp = Blueprint("segments", __name__, url_prefix="/api")

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

@bp.put("/segments/<segment_id>/draft")
def save_draft(segment_id: str):
    ensure_storage_layout(current_app.config)

    body = request.get_json(silent=True) or {}
    human_edit = body.get("human_edit")
    if human_edit is None:
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "Missing human_edit"}}), 400

    found = find_segment_by_id(current_app.config, segment_id)
    if not found:
        return jsonify({"error": {"code": "NOT_FOUND", "message": "Unknown segment_id"}}), 404

    doc_id, seg = found
    seg["human_edit"] = human_edit
    seg["last_saved_at"] = _now_iso()
    seg.setdefault("status", "pending")

    update_segment(current_app.config, doc_id, seg)

    return jsonify({"segment_id": segment_id, "saved": True, "last_saved_at": seg["last_saved_at"]})
