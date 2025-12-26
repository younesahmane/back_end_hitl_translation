from flask import Blueprint, jsonify

bp = Blueprint("health", __name__, url_prefix="/api")

@bp.get("/health")
def health():
    return jsonify({"status": "ok", "version": "0.1.0"})
