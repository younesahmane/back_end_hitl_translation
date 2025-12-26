import importlib
from typing import Any

class DummyOrchestrator:
    """
    Safe fallback if ORCHESTRATOR_IMPORT is not configured or import fails.
    Mimics the shape of OrchestratorAgent.process_batch output.
    Deterministic rules:
      - empty human_edit => reject
      - len(human_edit)<8 => reject
      - else accept
      - dataset_path when accepted_count>=4
    """
    def process_batch(self, frontend_payload: list[dict], language_pair=("en", "ar"), check_monitor=True) -> dict:
        results = []
        accepted = 0
        for i, item in enumerate(frontend_payload, 1):
            human_edit = (item.get("human_edit") or "").strip()
            if not human_edit:
                decision = "reject"
                reason = "empty human_edit"
            elif len(human_edit) < 8:
                decision = "reject"
                reason = "too short"
            else:
                decision = "accept"
                reason = ""
                accepted += 1

            results.append({
                "index": i,
                "correction": {
                    "decision": decision,
                    "reason": reason,
                    "dedup": {"is_duplicate": False}
                }
            })

        dataset_path = None
        if check_monitor:
            if accepted >= 4:
                dataset_path = "archive/final_dataset_dummy.jsonl"
            elif accepted == 0:
                dataset_path = "no correction was accepted"

        return {"results": results, "dataset_path": dataset_path}

def _import_from_path(path: str):
    """
    path format: "package.module:ClassName"
    """
    mod_path, cls_name = path.split(":")
    module = importlib.import_module(mod_path)
    return getattr(module, cls_name)

_cached_orchestrator: Any = None

def get_orchestrator(cfg: dict):
    global _cached_orchestrator
    if _cached_orchestrator is not None:
        return _cached_orchestrator

    import_path = (cfg.get("ORCHESTRATOR_IMPORT") or "").strip()
    if not import_path:
        _cached_orchestrator = DummyOrchestrator()
        return _cached_orchestrator

    try:
        OrchestratorCls = _import_from_path(import_path)
        _cached_orchestrator = OrchestratorCls(
            path_file_jsonl=cfg.get("ORCHESTRATOR_MASTER_JSONL", "master.jsonl"),
            faiss_index_path=cfg.get("ORCHESTRATOR_FAISS_INDEX", "dedup_index.faiss"),
            archive_dir=cfg.get("ORCHESTRATOR_ARCHIVE_DIR", "archive"),
            monitor_state_path=cfg.get("ORCHESTRATOR_MONITOR_STATE", "monitor_state.json"),
            line_threshold=int(cfg.get("ORCHESTRATOR_LINE_THRESHOLD", 10000)),
            days_threshold=int(cfg.get("ORCHESTRATOR_DAYS_THRESHOLD", 7)),
            few_shot_n=int(cfg.get("ORCHESTRATOR_FEW_SHOT_N", 50)),
        )
        return _cached_orchestrator
    except Exception:
        # Never break the app due to orchestrator import/config issues.
        _cached_orchestrator = DummyOrchestrator()
        return _cached_orchestrator

def run_orchestrator_batch(orchestrator, frontend_payload, language_pair, check_monitor):
    # Keep adapter thin and stable.
    return orchestrator.process_batch(
        frontend_payload=frontend_payload,
        language_pair=language_pair,
        check_monitor=check_monitor,
    )
