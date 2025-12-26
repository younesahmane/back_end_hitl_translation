from flask import Blueprint, request, jsonify, current_app
from services.storage import ensure_storage_layout
from services.tm_service import tm_search, tm_add

bp = Blueprint("tm", __name__, url_prefix="/api")

@bp.get("/tm/search")
def search_tm():
    ensure_storage_layout(current_app.config)

    query = (request.args.get("query") or "").strip()
    language_pair = (request.args.get("language_pair") or "en-ar").strip().lower()

    if not query:
        return jsonify({"matches": []})

    matches = tm_search(current_app.config["TM_JSON"], query=query, language_pair=language_pair, limit=10)
    return jsonify({"matches": matches})

@bp.post("/tm/add")
def add_tm():
    ensure_storage_layout(current_app.config)

    body = request.get_json(silent=True) or {}
    source = body.get("source")
    target = body.get("target")
    language_pair = (body.get("language_pair") or "en-ar").strip().lower()

    if not source or not target:
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "source and target are required"}}), 400

    entry_id = tm_add(current_app.config["TM_JSON"], source=source, target=target, language_pair=language_pair)
    return jsonify({"added": True, "entry_id": entry_id})
