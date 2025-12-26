#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
LANG_PAIR="${LANG_PAIR:-en-ar}"
TEST_FILE="${TEST_FILE:-./NLP.pdf}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }
need_cmd curl
need_cmd python

BASE_URL="${BASE_URL%/}"

echo "== Using BASE_URL: $BASE_URL"
echo "== Using LANG_PAIR: $LANG_PAIR"
echo "== Using TEST_FILE: $TEST_FILE"
echo

if [[ ! -f "$TEST_FILE" ]]; then
  echo "TEST_FILE not found: $TEST_FILE"
  exit 1
fi

sep() { echo; echo "------------------------------------------------------------"; echo; }

json_pretty() { python -m json.tool; }

echo "1) GET /api/health"
curl -fsS "$BASE_URL/api/health" | json_pretty
sep

echo "2) POST /api/documents/upload (multipart form-data)"
# Capture ONLY body; also save to a file to avoid shell/encoding issues
UPLOAD_BODY_FILE="$(mktemp)"
curl -fsS -o "$UPLOAD_BODY_FILE" -X POST "$BASE_URL/api/documents/upload" \
  -H "Accept: application/json" \
  -F "file=@${TEST_FILE}" \
  -F "language_pair=${LANG_PAIR}"

cat "$UPLOAD_BODY_FILE" | json_pretty

DOC_ID="$(python - <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["doc_id"])
PY
"$UPLOAD_BODY_FILE")"

echo
echo "== Extracted DOC_ID: $DOC_ID"
sep

echo "3) GET /api/documents"
curl -fsS "$BASE_URL/api/documents" | json_pretty
sep

echo "4) GET /api/documents/:doc_id/segments"
SEGS_BODY_FILE="$(mktemp)"
curl -fsS -o "$SEGS_BODY_FILE" "$BASE_URL/api/documents/${DOC_ID}/segments"
cat "$SEGS_BODY_FILE" | json_pretty

SEG_ID="$(python - <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
segs = data.get("segments", [])
if not segs:
    raise SystemExit("No segments returned")
print(segs[0]["id"])
PY
"$SEGS_BODY_FILE")"

echo
echo "== Extracted SEG_ID (first segment): $SEG_ID"
sep

echo "5) PUT /api/segments/:segment_id/draft"
DRAFT_TEXT="هذه ترجمة تجريبية محفوظة كمسودة."
curl -fsS -X PUT "$BASE_URL/api/segments/${SEG_ID}/draft" \
  -H "Content-Type: application/json" \
  -d "$(python - <<PY
import json
print(json.dumps({"human_edit": "${DRAFT_TEXT}"}))
PY
)" | json_pretty
sep

echo "6) POST /api/llm/generate (dummy deterministic)"
curl -fsS -X POST "$BASE_URL/api/llm/generate" \
  -H "Content-Type: application/json" \
  -d "$(python - <<PY
import json
print(json.dumps({
  "language_pair": "${LANG_PAIR}",
  "sources": ["Good morning, everyone.", "The meeting was postponed to next Monday."],
  "models": ["model1","model2","model3"]
}))
PY
)" | json_pretty
sep

echo "7) POST /api/pipeline/process-batch"
PIPELINE_BODY="$(python - <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
segs = data["segments"][:5]  # try 5
payload = []
for s in segs:
    payload.append({
        "segment_id": s["id"],
        "source": s["source"],
        "llm_outputs": s.get("llm_outputs", {"model1":"","model2":"","model3":""}),
        "human_edit": "تمت مراجعة هذه الجملة وتجهيزها للاعتماد."
    })
print(json.dumps({
    "language_pair": ["en","ar"],
    "check_monitor": True,
    "frontend_payload": payload
}))
PY
"$SEGS_BODY_FILE")"

curl -fsS -X POST "$BASE_URL/api/pipeline/process-batch" \
  -H "Content-Type: application/json" \
  -d "$PIPELINE_BODY" | json_pretty
sep

echo "8) POST /api/tm/add"
TM_ADD_BODY="$(curl -fsS -X POST "$BASE_URL/api/tm/add" \
  -H "Content-Type: application/json" \
  -d "$(python - <<PY
import json
print(json.dumps({"source":"Good morning","target":"صباح الخير","language_pair":"${LANG_PAIR}"}))
PY
)")"
echo "$TM_ADD_BODY" | json_pretty
sep

echo "9) GET /api/tm/search"
curl -fsS "$BASE_URL/api/tm/search?query=Good%20morning&language_pair=${LANG_PAIR}" | json_pretty
sep

echo "10) POST /api/documents/:doc_id/export (docx)"
EXPORT_BODY_FILE="$(mktemp)"
curl -fsS -o "$EXPORT_BODY_FILE" -X POST "$BASE_URL/api/documents/${DOC_ID}/export" \
  -H "Content-Type: application/json" \
  -d '{"format":"docx"}'
cat "$EXPORT_BODY_FILE" | json_pretty

DL_URL="$(python - <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["download_url"])
PY
"$EXPORT_BODY_FILE")"

echo
echo "== Export download URL: $DL_URL"
sep

echo "11) GET /downloads/<filename>"
OUT_FILE="export_${DOC_ID}.docx"
curl -fsS -L "${BASE_URL}${DL_URL}" -o "$OUT_FILE"
echo "Downloaded to: $OUT_FILE"
echo
echo "ALL TESTS PASSED"
