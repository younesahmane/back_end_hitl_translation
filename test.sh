#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
LANG_PAIR="${LANG_PAIR:-en-ar}"

# Provide a file path via env or default to ./sample.docx (recommended)
TEST_FILE="${TEST_FILE:-./NLP.pdf}"

echo "== Using BASE_URL: $BASE_URL"
echo "== Using LANG_PAIR: $LANG_PAIR"
echo "== Using TEST_FILE: $TEST_FILE"
echo

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }
}
need_cmd curl
need_cmd python

if [[ ! -f "$TEST_FILE" ]]; then
  echo "TEST_FILE not found: $TEST_FILE"
  echo "Tip: set TEST_FILE=/path/to/your.docx (or .pdf)"
  exit 1
fi

sep() { echo; echo "------------------------------------------------------------"; echo; }

echo "1) GET /api/health"
curl -sS "$BASE_URL/api/health" | python -m json.tool
sep

echo "2) POST /api/documents/upload (multipart form-data)"
UPLOAD_JSON="$(curl -sS -X POST "$BASE_URL/api/documents/upload" \
  -F "file=@${TEST_FILE}" \
  -F "language_pair=${LANG_PAIR}")"

echo "$UPLOAD_JSON" | python -m json.tool

DOC_ID="$(python - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["doc_id"])
PY
<<< "$UPLOAD_JSON")"

echo
echo "== Extracted DOC_ID: $DOC_ID"
sep

echo "3) GET /api/documents"
curl -sS "$BASE_URL/api/documents" | python -m json.tool
sep

echo "4) GET /api/documents/:doc_id/segments"
SEGS_JSON="$(curl -sS "$BASE_URL/api/documents/${DOC_ID}/segments")"
echo "$SEGS_JSON" | python -m json.tool

SEG_ID="$(python - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
segs = data.get("segments", [])
if not segs:
    raise SystemExit("No segments returned")
print(segs[0]["id"])
PY
<<< "$SEGS_JSON")"

SEG_SOURCE="$(python - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["segments"][0]["source"])
PY
<<< "$SEGS_JSON")"

echo
echo "== Extracted SEG_ID (first segment): $SEG_ID"
echo "== Extracted SEG_SOURCE (first segment): ${SEG_SOURCE:0:90}..."
sep

echo "5) PUT /api/segments/:segment_id/draft"
DRAFT_TEXT="هذه ترجمة تجريبية محفوظة كمسودة."
curl -sS -X PUT "$BASE_URL/api/segments/${SEG_ID}/draft" \
  -H "Content-Type: application/json" \
  -d "{\"human_edit\": \"${DRAFT_TEXT}\"}" | python -m json.tool
sep

echo "6) POST /api/llm/generate (dummy deterministic)"
curl -sS -X POST "$BASE_URL/api/llm/generate" \
  -H "Content-Type: application/json" \
  -d "$(python - <<PY
import json
print(json.dumps({
  "language_pair": "${LANG_PAIR}",
  "sources": [
    "Good morning, everyone.",
    "The meeting was postponed to next Monday."
  ],
  "models": ["model1","model2","model3"]
}))
PY
)" | python -m json.tool
sep

echo "7) POST /api/pipeline/process-batch"
# Grab a couple segments to submit. We include segment_id to let backend update JSON status/reason.
PIPELINE_BODY="$(python - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
segs = data["segments"][:3]  # take first 3
payload = []
for s in segs:
    payload.append({
        "segment_id": s["id"],
        "source": s["source"],
        "llm_outputs": s.get("llm_outputs", {"model1":"","model2":"","model3":""}),
        # Make sure human_edit non-empty so dummy orchestrator accepts
        "human_edit": s.get("human_edit") or "تمت مراجعة هذه الجملة وتجهيزها للاعتماد."
    })
print(json.dumps({
    "language_pair": ["en","ar"],  # adjust if needed
    "check_monitor": True,
    "frontend_payload": payload
}))
PY
<<< "$SEGS_JSON")"

curl -sS -X POST "$BASE_URL/api/pipeline/process-batch" \
  -H "Content-Type: application/json" \
  -d "$PIPELINE_BODY" | python -m json.tool
sep

echo "8) POST /api/tm/add"
TM_ADD_JSON="$(curl -sS -X POST "$BASE_URL/api/tm/add" \
  -H "Content-Type: application/json" \
  -d "$(python - <<PY
import json
print(json.dumps({
  "source": "Good morning",
  "target": "صباح الخير",
  "language_pair": "${LANG_PAIR}"
}))
PY
)")"
echo "$TM_ADD_JSON" | python -m json.tool
sep

echo "9) GET /api/tm/search"
curl -sS "$BASE_URL/api/tm/search?query=Good%20morning&language_pair=${LANG_PAIR}" | python -m json.tool
sep

echo "10) POST /api/documents/:doc_id/export (docx)"
EXPORT_JSON="$(curl -sS -X POST "$BASE_URL/api/documents/${DOC_ID}/export" \
  -H "Content-Type: application/json" \
  -d '{"format":"docx"}')"
echo "$EXPORT_JSON" | python -m json.tool

DL_URL="$(python - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["download_url"])
PY
<<< "$EXPORT_JSON")"

echo
echo "== Export download URL: $DL_URL"
sep

echo "11) GET /downloads/<filename> (download export)"
OUT_FILE="export_${DOC_ID}.docx"
curl -sS -L "${BASE_URL}${DL_URL}" -o "$OUT_FILE"
echo "Downloaded to: $OUT_FILE"
echo
echo "ALL TESTS PASSED"
