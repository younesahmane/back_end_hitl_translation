import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.storage import ensure_storage_layout
from services.agent_bridge import build_orchestrator


def create_app() -> Flask:
    app = Flask(__name__)

    # -------------------------
    # Config
    # -------------------------
    root = os.path.abspath(os.path.dirname(__file__))
    storage_dir = os.path.join(root, "storage")

    app.config.update(
        STORAGE_DIR=storage_dir,
        UPLOADS_DIR=os.path.join(storage_dir, "uploads"),
        DOWNLOADS_DIR=os.path.join(storage_dir, "downloads"),
        DOCS_JSON=os.path.join(storage_dir, "documents.json"),
        SEGMENTS_JSON=os.path.join(storage_dir, "segments.json"),
        TM_JSON=os.path.join(storage_dir, "tm.json"),
        EXPORTS_DIR=os.path.join(storage_dir, "exports"),
        MASTER_JSONL=os.path.join(storage_dir, "master.jsonl"),
        DEDUP_FAISS=os.path.join(storage_dir, "dedup_index.faiss"),
        MONITOR_STATE=os.path.join(storage_dir, "monitor_state.json"),
        ARCHIVE_DIR=os.path.join(storage_dir, "archive"),
        VERSION="0.1.0",
    )

    ensure_storage_layout(app.config)

    # Important: avoids redirect loops when client sends /api/... vs /api/...
    app.url_map.strict_slashes = False

    # CORS for API
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Fix for your logs: OPTIONS /api//... -> 308
    # Normalize repeated slashes BEFORE routing.
    @app.before_request
    def _normalize_path():
        p = request.environ.get("PATH_INFO", "")
        while "//" in p:
            p = p.replace("//", "/")
        request.environ["PATH_INFO"] = p

    # Orchestrator singleton
    app.extensions["orchestrator"] = build_orchestrator(app.config)

    # Blueprints
    from routes.health import bp as health_bp
    from routes.documents import bp as documents_bp
    from routes.segments import bp as segments_bp
    from routes.llm import bp as llm_bp
    from routes.pipeline import bp as pipeline_bp
    from routes.tm import bp as tm_bp
    from routes.downloads import bp as downloads_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(segments_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(tm_bp)
    app.register_blueprint(downloads_bp)

    # Uniform error payload
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": {"code": "NOT_FOUND", "message": "Resource not found"}}), 404

    @app.errorhandler(Exception)
    def unhandled(e):
        # For production, hide details and log instead.
        return jsonify({"error": {"code": "INTERNAL", "message": str(e)}}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=True)
