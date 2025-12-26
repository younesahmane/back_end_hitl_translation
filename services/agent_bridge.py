import os
from typing import Dict


def build_orchestrator(cfg: Dict[str, str]):
    """
    Build a single orchestrator instance.

    Use:
      export AGENTS_MODE=stub
    to avoid heavy ML deps during dev.
    """
    mode = os.environ.get("AGENTS_MODE", "real").lower().strip()
    if mode == "stub":
        from agentic.stub_orchestrator import StubOrchestrator

        return StubOrchestrator(master_jsonl=cfg["MASTER_JSONL"])

    # "real" mode: use your notebook agentic classes
    from agentic.orchestrator import OrchestratorAgent

    return OrchestratorAgent(
        path_file_jsonl=cfg["MASTER_JSONL"],
        faiss_index_path=cfg["DEDUP_FAISS"],
        archive_dir=cfg["ARCHIVE_DIR"],
        monitor_state_path=cfg["MONITOR_STATE"],
        line_threshold=10_000,
        days_threshold=7,
        few_shot_n=50,
    )
