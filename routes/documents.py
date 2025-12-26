from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

from services.storage import (
    ensure_storage_layout,
    load_docs,
    save_docs,
    save_doc_segments,
    load_doc_segments,
)
from services.segmenter import segment_document_file
from services.llm_dummy import dummy_llm_outputs_for_segments

bp = Blueprint("documents", __name__, url_prefix="/api")

ALLOWED_EXTS = {".pdf", ".docx"}

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

@bp.get("/documents")
def list_documents():
    ensure_storage_layout(current_app.config)
    docs = load_docs(current_app.config["DOCS_JSON"])
    return jsonify({"documents": docs})

@bp.get("/documents/<doc_id>/segments")
def get_segments(doc_id: str):
    ensure_storage_layout(current_app.config)
    segments = load_doc_segments(current_app.config, doc_id)
    if segments is None:
        return jsonify({"error": {"code": "NOT_FOUND", "message": "Unknown doc_id"}}), 404
    return jsonify({"doc_id": doc_id, "segments": segments})

@bp.post("/documents/upload")
def upload_document():
    """
    Form-data:
      - file: PDF or DOCX
      - language_pair: "en-ar" or "fr-ar"
    """
    ensure_storage_layout(current_app.config)

    if "file" not in request.files:
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "Missing file"}}), 400

    file = request.files["file"]
    language_pair = request.form.get("language_pair", "en-ar").strip().lower()

    if not file.filename:
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "Empty filename"}}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": f"Unsupported file type: {ext}"}}), 400

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    upload_dir = os.path.join(current_app.config["UPLOADS_DIR"], doc_id)
    os.makedirs(upload_dir, exist_ok=True)

    stored_name = f"original{ext}"
    stored_path = os.path.join(upload_dir, stored_name)
    file.save(stored_path)

    # Segment
    try:
        segments = segment_document_file(
            filepath=stored_path,
            doc_id=doc_id,
            segment_type="paragraph",
        )
    except Exception as e:
        return jsonify({"error": {"code": "SEGMENTATION_FAILED", "message": str(e)}}), 400

    # Attach dummy llm outputs for initial UI (optional)
    segments = dummy_llm_outputs_for_segments(segments)

    # Persist document metadata
    docs = load_docs(current_app.config["DOCS_JSON"])
    doc_meta = {
        "doc_id": doc_id,
        "filename": filename,
        "language_pair": language_pair,
        "created_at": _now_iso(),
        "stored_path": stored_path,
    }
    docs.append(doc_meta)
    save_docs(current_app.config["DOCS_JSON"], docs)

    # Persist segments
    save_doc_segments(current_app.config, doc_id, segments)

    return jsonify({
        "doc_id": doc_id,
        "filename": filename,
        "language_pair": language_pair,
        "segments": segments,
    })


from services.exporter import export_docx
from services.storage import load_doc_segments

@bp.post("/documents/<doc_id>/export")
def export_document(doc_id: str):
    ensure_storage_layout(current_app.config)

    body = request.get_json(silent=True) or {}
    fmt = (body.get("format") or "docx").lower()
    if fmt != "docx":
        return jsonify({"error": {"code": "VALIDATION_ERROR", "message": "Only docx export supported for now"}}), 400
    
    # Use segments from frontend request body (with latest user edits)
    # If not provided, fall back to loading from disk
    segments = body.get("segments")
    
    if not segments:
        segments = load_doc_segments(current_app.config, doc_id)
    
    if segments is None or not segments:
        return jsonify({"error": {"code": "NOT_FOUND", "message": "Unknown doc_id or no segments"}}), 404
    
    filename = export_docx(doc_id=doc_id, segments=segments, exports_dir=current_app.config["EXPORTS_DIR"])
    return jsonify({"download_url": f"/downloads/{filename}"})
