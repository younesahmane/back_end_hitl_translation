from flask import Blueprint, current_app, send_from_directory, jsonify
import os

bp = Blueprint("downloads", __name__, url_prefix="")

@bp.get("/downloads/<path:filename>")
def downloads(filename: str):
    exports_dir = current_app.config["EXPORTS_DIR"]
    full_path = os.path.join(exports_dir, filename)
    if not os.path.isfile(full_path):
        return jsonify({"error": {"code": "NOT_FOUND", "message": "File not found"}}), 404
    return send_from_directory(exports_dir, filename, as_attachment=True)
