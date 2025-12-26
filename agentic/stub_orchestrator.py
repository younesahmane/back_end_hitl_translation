from typing import Dict, List, Tuple
from services.storage import append_jsonl
from services.utils import now_iso


class StubOrchestrator:
    """
    Deterministic minimal orchestrator:
    - Always 'accept' if human_edit is non-empty, else 'reject'
    - Appends accepted rows to master.jsonl
    """

    def __init__(self, master_jsonl: str):
        self.master_jsonl = master_jsonl

    def process_batch(
        self,
        frontend_payload: List[Dict],
        language_pair: Tuple[str, str] = ("en", "ar"),
        check_monitor: bool = True,
    ) -> Dict:
        results = []
        for i, item in enumerate(frontend_payload, 1):
            source = item.get("source", "")
            llm_outputs = item.get("llm_outputs", {})
            human_edit = item.get("human_edit", "")

            decision = "accept" if (human_edit or "").strip() else "reject"
            correction = {
                "timestamp": now_iso(),
                "source": source,
                "llm_outputs": llm_outputs,
                "human_edit": human_edit,
                "decision": decision,
                "reason": "stub_accept_nonempty" if decision == "accept" else "stub_reject_empty",
            }
            if decision == "accept":
                append_jsonl(self.master_jsonl, correction)

            results.append({"index": i, "correction": correction})

        return {"results": results, "dataset_path": None}
