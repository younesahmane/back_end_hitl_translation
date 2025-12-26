from flask import Blueprint, request, jsonify, current_app
from services.storage import ensure_storage_layout
from services.orchestrator_adapter import get_orchestrator, run_orchestrator_batch
from services.storage import try_update_segments_from_batch_results

bp = Blueprint("pipeline", __name__, url_prefix="/api")

@bp.post("/pipeline/process-batch")
def process_batch():
    """
    Body (exactly as frontend expects):
    {
      "language_pair": ["en","ar"],
      "check_monitor": true,
      "frontend_payload": [
        { "source": "...", "llm_outputs": {...}, "human_edit": "..." }
      ]
    }
    """
    ensure_storage_layout(current_app.config)

    body = request.get_json(silent=True) or {}
    language_pair = body.get("language_pair", ["en", "ar"])
    check_monitor = bool(body.get("check_monitor", True))
    frontend_payload = body.get("frontend_payload", [])

    if (
        not isinstance(language_pair, list)
        or len(language_pair) != 2
        or any(not isinstance(x, str) for x in language_pair)
    ):
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "language_pair must be [src, tgt]"}}), 400

    if not isinstance(frontend_payload, list):
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "frontend_payload must be a list"}}), 400

    # Basic item validation
    for idx, item in enumerate(frontend_payload, start=1):
        if not isinstance(item, dict):
            return jsonify({"error": {"code": "VALIDATION_ERROR", "message": f"frontend_payload[{idx}] must be object"}}), 400
        for key in ("source", "llm_outputs", "human_edit"):
            if key not in item:
                return jsonify({"error": {"code": "VALIDATION_ERROR", "message": f"Missing {key} in item {idx}"}}), 400

    orchestrator = get_orchestrator(current_app.config)

    output = run_orchestrator_batch(
        orchestrator=orchestrator,
        frontend_payload=frontend_payload,
        language_pair=(language_pair[0], language_pair[1]),
        check_monitor=check_monitor,
    )

    # Optional: update segments status if caller included segment_id per item
    # This is robust: if no IDs, nothing breaks.
    try_update_segments_from_batch_results(current_app.config, frontend_payload, output)

    return jsonify(output)
