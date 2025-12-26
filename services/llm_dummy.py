from typing import Dict, List

# Deterministic “always the same” outputs for easy swapping later.
_FIXED = {
    "model1": "صباح الخير جميعا.",
    "model2": "أهلا صباح الخير للجميع.",
    "model3": "صباح الخير يا الجميع."
}

def dummy_generate_llm_outputs(source: str, language_pair: str, models: List[str]) -> Dict[str, str]:
    # Always returns same outputs regardless of source.
    out = {}
    for m in models:
        out[m] = _FIXED.get(m, "نص تجريبي ثابت.")
    return out

def dummy_llm_outputs_for_segments(segments: list[dict]) -> list[dict]:
    for s in segments:
        s["llm_outputs"] = {
            "model1": _FIXED["model1"],
            "model2": _FIXED["model2"],
            "model3": _FIXED["model3"],
        }
    return segments
