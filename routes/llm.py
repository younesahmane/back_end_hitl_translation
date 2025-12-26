from flask import Blueprint, request, jsonify, current_app
from services.storage import ensure_storage_layout
from services.llm_dummy import dummy_generate_llm_outputs

bp = Blueprint("llm", __name__, url_prefix="/api")

@bp.post("/llm/generate")
def generate_llm_outputs():
    """
    Dummy generator endpoint.
    Body:
      {
        "language_pair": "en-ar",
        "sources": ["text1", "text2", ...],
        "models": ["model1","model2","model3"]   # optional
      }
    Returns (deterministic):
      {
        "language_pair": "en-ar",
        "items": [
          {
            "source": "...",
            "llm_outputs": { "model1": "...", "model2": "...", "model3": "..." }
          }
        ]
      }
    """
    ensure_storage_layout(current_app.config)

    body = request.get_json(silent=True) or {}
    language_pair = (body.get("language_pair") or "en-ar").lower()
    sources = body.get("sources") or []
    models = body.get("models") or ["model1", "model2", "model3"]

    if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "sources must be a list of strings"}}), 400

    items = []
    for s in sources:
        items.append({
            "source": s,
            "llm_outputs": dummy_generate_llm_outputs(s, language_pair=language_pair, models=models),
        })

    return jsonify({"language_pair": language_pair, "items": items})
