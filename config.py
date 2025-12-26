import os

class Config:
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"

    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8080,http://localhost:5173").split(",")

    # Storage root
    STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(os.path.dirname(__file__), "storage"))

    # JSON stores
    DOCS_JSON = os.path.join(STORAGE_DIR, "docs.json")
    TM_JSON = os.path.join(STORAGE_DIR, "tm.json")

    # Subfolders
    UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
    EXPORTS_DIR = os.path.join(STORAGE_DIR, "exports")

    # File limits
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(25 * 1024 * 1024)))  # 25MB

    # Orchestrator integration
    # Point this to where OrchestratorAgent is importable, e.g.:
    # ORCHESTRATOR_IMPORT="my_pipeline.orchestrator:OrchestratorAgent"
    ORCHESTRATOR_IMPORT = os.getenv("ORCHESTRATOR_IMPORT", "")

    # Orchestrator constructor kwargs (paths, thresholds)
    ORCHESTRATOR_MASTER_JSONL = os.getenv("ORCHESTRATOR_MASTER_JSONL", "master.jsonl")
    ORCHESTRATOR_FAISS_INDEX = os.getenv("ORCHESTRATOR_FAISS_INDEX", "dedup_index.faiss")
    ORCHESTRATOR_ARCHIVE_DIR = os.getenv("ORCHESTRATOR_ARCHIVE_DIR", "archive")
    ORCHESTRATOR_MONITOR_STATE = os.getenv("ORCHESTRATOR_MONITOR_STATE", "monitor_state.json")
    ORCHESTRATOR_LINE_THRESHOLD = int(os.getenv("ORCHESTRATOR_LINE_THRESHOLD", "10000"))
    ORCHESTRATOR_DAYS_THRESHOLD = int(os.getenv("ORCHESTRATOR_DAYS_THRESHOLD", "7"))
    ORCHESTRATOR_FEW_SHOT_N = int(os.getenv("ORCHESTRATOR_FEW_SHOT_N", "50"))
