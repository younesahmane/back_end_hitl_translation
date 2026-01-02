# Orchestrator Integration Documentation

## Overview

The HITL (Human-in-the-Loop) Backend is designed to integrate with an **Orchestrator Agent** system that processes translation corrections through a multi-stage pipeline. This document describes how the backend and orchestrator should be linked together.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Client                      │
│                    (React/Web Application)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │      HITL Backend (Flask Server)        │
        │                                          │
        │  ┌──────────────────────────────────┐   │
        │  │  /api/pipeline/process-batch     │   │
        │  └────────┬─────────────────────────┘   │
        │           │                              │
        │           ▼                              │
        │  ┌──────────────────────────────────┐   │
        │  │   Orchestrator Adapter           │   │
        │  │ (orchestrator_adapter.py)        │   │
        │  └────────┬─────────────────────────┘   │
        │           │                              │
        │           ▼                              │
        │  ┌──────────────────────────────────┐   │
        │  │      Orchestrator Agent          │   │
        │  │   (agentic/orchestrator.py)      │   │
        │  │                                   │   │
        │  │  ├─ PreprocessorAgent            │   │
        │  │  ├─ ValidatorAgent               │   │
        │  │  ├─ DeduplicatorAgent            │   │
        │  │  ├─ ExecutorAgent                │   │
        │  │  └─ MonitorAgent                 │   │
        │  └────────┬─────────────────────────┘   │
        │           │                              │
        │           ▼                              │
        │  ┌──────────────────────────────────┐   │
        │  │      Storage Layer               │   │
        │  │  - master.jsonl                  │   │
        │  │  - dedup_index.faiss             │   │
        │  │  - monitor_state.json            │   │
        │  │  - archive/                      │   │
        │  └──────────────────────────────────┘   │
        └─────────────────────────────────────────┘
```

---

## Integration Points

### 1. Flask Application Setup

**File:** [app.py](app.py)

The Flask application initializes the orchestrator as a singleton on startup:

```python
# Orchestrator singleton
app.extensions["orchestrator"] = build_orchestrator(app.config)
```

**Configuration Keys Required:**
- `MASTER_JSONL`: Path to master dataset file
- `DEDUP_FAISS`: Path to FAISS index for deduplication
- `ARCHIVE_DIR`: Directory for archived datasets
- `MONITOR_STATE`: Path to monitor state file

### 2. Orchestrator Adapter

**File:** [services/orchestrator_adapter.py](services/orchestrator_adapter.py)

The adapter acts as a bridge between the Flask backend and the orchestrator. It provides:

**Key Functions:**

- **`get_orchestrator(cfg: dict)`**
  - Dynamically loads the orchestrator class based on `ORCHESTRATOR_IMPORT` config
  - Falls back to `DummyOrchestrator` if import fails (ensures graceful degradation)
  - Caches the orchestrator instance globally
  - **Configuration Variables:**
    - `ORCHESTRATOR_IMPORT`: Format `"package.module:ClassName"` (e.g., `"agentic.orchestrator:OrchestratorAgent"`)
    - `ORCHESTRATOR_MASTER_JSONL`: Path to master dataset
    - `ORCHESTRATOR_FAISS_INDEX`: Path to FAISS dedup index
    - `ORCHESTRATOR_ARCHIVE_DIR`: Archive directory
    - `ORCHESTRATOR_MONITOR_STATE`: Monitor state path
    - `ORCHESTRATOR_LINE_THRESHOLD`: Lines before triggering monitor (default: 10000)
    - `ORCHESTRATOR_DAYS_THRESHOLD`: Days before triggering monitor (default: 7)
    - `ORCHESTRATOR_FEW_SHOT_N`: Few-shot examples count (default: 50)

- **`run_orchestrator_batch(orchestrator, frontend_payload, language_pair, check_monitor)`**
  - Calls `orchestrator.process_batch()` with standardized parameters
  - Keeps the adapter thin and stable

### 3. Pipeline Route

**File:** [routes/pipeline.py](routes/pipeline.py)

The main entry point for orchestrator requests.

**Endpoint:** `POST /api/pipeline/process-batch`

**Request Format:**
```json
{
  "language_pair": ["en", "ar"],
  "check_monitor": true,
  "frontend_payload": [
    {
      "source": "English text",
      "llm_outputs": {
        "model1": "Arabic translation 1",
        "model2": "Arabic translation 2"
      },
      "human_edit": "Corrected Arabic translation",
      "segment_id": "optional_segment_uuid"
    }
  ]
}
```

**Response Format:**
```json
{
  "results": [
    {
      "index": 1,
      "correction": {
        "decision": "accept|reject",
        "reason": "explanation",
        "dedup": {
          "is_duplicate": false
        }
      }
    }
  ],
  "dataset_path": "path/to/finetune_dataset.jsonl"
}
```

---

## Orchestrator Agent Pipeline

### Flow

The `OrchestratorAgent.process_batch()` method executes the following pipeline for each item:

```
Input (frontend_payload item)
        │
        ▼
1. PreprocessorAgent
   └─ Normalize text, handle language-specific rules
        │
        ▼
2. ValidatorAgent
   └─ Validate correction quality using:
       - LLM-based semantic validation
       - Language quality metrics (CHRF, similarity)
       - Grammar/spelling checks
        │
        ▼
3. DeduplicatorAgent
   └─ Check against FAISS index
   └─ Detect duplicates (threshold: 0.85 similarity)
        │
        ▼
4. ExecutorAgent
   └─ Persist to master.jsonl
   └─ Track decision (accept/reject)
        │
        ▼
5. MonitorAgent (if check_monitor=true)
   └─ Check if cycle triggers (by line_threshold or days_threshold)
   └─ Generate finetune dataset if triggered
        │
        ▼
Output: {"results": [...], "dataset_path": "..."}
```

### Processing Stages Detail

#### 1. **PreprocessorAgent**
- **Normalizes** Arabic text (diacritics, digits, punctuation)
- **Normalizes** Latin text (quotes, apostrophes, hyphens)
- **Structures** input for downstream agents
- **Handles** language-specific rules

#### 2. **ValidatorAgent**
- **Validates** the human edit against:
  - Original source text
  - LLM outputs
  - Language pair constraints
- **Returns** decision with quality score
- **Uses** sentence transformers for semantic validation

#### 3. **DeduplicatorAgent**
- **Checks** against FAISS index of previous corrections
- **Similarity threshold**: 0.85
- **Prevents** duplicate corrections
- **Updates** FAISS index if correction accepted

#### 4. **ExecutorAgent**
- **Persists** accepted corrections to `master.jsonl`
- **Tracks** all decisions in dataset
- **Ensures** data durability

#### 5. **MonitorAgent**
- **Monitors** accumulated corrections
- **Triggers** finetune dataset generation when:
  - `line_threshold` accumulated corrections reached, OR
  - `days_threshold` time elapsed
- **Archives** previous dataset
- **Generates** new finetune dataset

---

## Configuration

### Environment Variables

Set these before running the Flask app:

```bash
# Orchestrator mode: "real" or "stub" (stub is lightweight for testing)
export AGENTS_MODE=real

# Dynamic import path (format: "package.module:ClassName")
export ORCHESTRATOR_IMPORT="agentic.orchestrator:OrchestratorAgent"

# Paths
export ORCHESTRATOR_MASTER_JSONL="storage/master.jsonl"
export ORCHESTRATOR_FAISS_INDEX="storage/dedup_index.faiss"
export ORCHESTRATOR_ARCHIVE_DIR="storage/archive"
export ORCHESTRATOR_MONITOR_STATE="storage/monitor_state.json"

# Thresholds
export ORCHESTRATOR_LINE_THRESHOLD=10000
export ORCHESTRATOR_DAYS_THRESHOLD=7
export ORCHESTRATOR_FEW_SHOT_N=50
```

### Configuration via Flask Config

Alternatively, configure in your config.py or pass to Flask app:

```python
app.config.update(
    ORCHESTRATOR_IMPORT="agentic.orchestrator:OrchestratorAgent",
    ORCHESTRATOR_MASTER_JSONL="storage/master.jsonl",
    ORCHESTRATOR_FAISS_INDEX="storage/dedup_index.faiss",
    ORCHESTRATOR_ARCHIVE_DIR="storage/archive",
    ORCHESTRATOR_MONITOR_STATE="storage/monitor_state.json",
    ORCHESTRATOR_LINE_THRESHOLD=10000,
    ORCHESTRATOR_DAYS_THRESHOLD=7,
    ORCHESTRATOR_FEW_SHOT_N=50,
)
```

---

## Data Flow

### Input: Frontend Payload

Each item in `frontend_payload` must contain:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | ✓ | Source language text (e.g., English) |
| `llm_outputs` | dict | ✓ | Dictionary of model outputs (key: model name, value: translation) |
| `human_edit` | string | ✓ | Human's corrected translation |
| `segment_id` | string | ✗ | Optional UUID linking to segment in database |
| `expected` | string | ✗ | For testing: expected decision ("accept"/"reject") |

### Output: Correction Result

Each result item contains:

```python
{
    "index": int,  # Position in original payload
    "correction": {
        "decision": "accept" | "reject",  # Final decision
        "reason": str,  # Explanation for decision
        "source": str,  # Original source
        "llm_outputs": dict,  # Original LLM outputs
        "human_translation": str,  # Corrected translation
        "validation_score": float,  # Quality score (0-1)
        "is_duplicate": bool,  # Deduplication result
        "stored_at": str,  # ISO timestamp if accepted
        "dedup": {
            "is_duplicate": bool,
            "similarity": float  # If duplicate detected
        }
    },
    # Optional test fields
    "expected": str,  # If provided in input
    "test_passed": bool  # If expected was provided
}
```

---

## Error Handling

### Graceful Degradation

If the orchestrator fails to load:
- The system falls back to `DummyOrchestrator`
- Simple deterministic rules apply:
  - Empty human_edit → reject
  - human_edit < 8 chars → reject
  - Otherwise → accept

### Recovery Strategies

1. **Import Failure**: Falls back to dummy orchestrator
2. **Processing Error**: Logs exception, marks item as "error"
3. **Storage Failure**: Exceptions bubble up to Flask handler (500 error)

---

## Integration Checklist

To successfully link your orchestrator with the backend:

- [ ] **Configuration**
  - [ ] Set `ORCHESTRATOR_IMPORT` environment variable or config
  - [ ] Ensure paths for MASTER_JSONL, FAISS_INDEX, etc. exist
  - [ ] Configure thresholds (LINE_THRESHOLD, DAYS_THRESHOLD)

- [ ] **Orchestrator Implementation**
  - [ ] `OrchestratorAgent` class has `process_batch()` method
  - [ ] Method signature: `process_batch(frontend_payload, language_pair, check_monitor) -> dict`
  - [ ] Returns dict with "results" and "dataset_path" keys

- [ ] **Storage**
  - [ ] `storage/` directory exists with required subdirectories
  - [ ] `master.jsonl` file is initialized
  - [ ] `dedup_index.faiss` FAISS index is created
  - [ ] `monitor_state.json` is initialized

- [ ] **Testing**
  - [ ] Send test request to `/api/pipeline/process-batch`
  - [ ] Verify response structure matches expected format
  - [ ] Check logs for orchestrator execution

---

## Example Integration Request

```bash
curl -X POST http://localhost:8001/api/pipeline/process-batch \
  -H "Content-Type: application/json" \
  -d '{
    "language_pair": ["en", "ar"],
    "check_monitor": true,
    "frontend_payload": [
      {
        "source": "Good morning",
        "llm_outputs": {
          "model1": "صباح الخير",
          "model2": "أهلا صباح الخير"
        },
        "human_edit": "صباح الخير جميعاً"
      }
    ]
  }'
```

---

## Extension Points

### Custom Orchestrator Implementation

To create a custom orchestrator:

1. **Create a class** inheriting from orchestrator pattern
2. **Implement** `process_batch(frontend_payload, language_pair, check_monitor) -> dict`
3. **Set** `ORCHESTRATOR_IMPORT="your.module:YourOrchestratorClass"`
4. **Ensure** returned dict has "results" and "dataset_path" keys

### Custom Preprocessing

Override `PreprocessorAgent` methods:
- `normalize_arabic(text)`
- `normalize_latin(text)`
- `process(source, llm_outputs, human_edit)`

### Custom Validation Rules

Override `ValidatorAgent` to implement custom validation logic for specific language pairs or domains.

---

## Monitoring & Debugging

### Logs

The orchestrator logs:
- Item processing start/finish
- Validation scores
- Deduplication results
- Monitor state changes

### Monitor State File

The `monitor_state.json` tracks:
- Last finetune cycle timestamp
- Current accumulated corrections count
- Threshold status

Example:
```json
{
  "last_finetuning_timestamp": "2024-01-02T10:30:00Z",
  "current_line_count": 5234,
  "days_since_last_finetuning": 3,
  "triggered": false
}
```

---

## Performance Considerations

- **Orchestrator Instantiation**: Cached as singleton (one instance per app)
- **FAISS Index**: Loaded in memory during initialization (~100MB for typical use)
- **Batch Processing**: Sequential per-item processing
- **Recommendation**: Keep batch size ≤ 100 items for API responsiveness

---

## Related Files

- [app.py](app.py) - Flask application setup
- [routes/pipeline.py](routes/pipeline.py) - Pipeline endpoint
- [services/orchestrator_adapter.py](services/orchestrator_adapter.py) - Adapter/bridge
- [services/agent_bridge.py](services/agent_bridge.py) - Alternative orchestrator builder
- [agentic/orchestrator.py](agentic/orchestrator.py) - Main orchestrator implementation
- [agentic/stub_orchestrator.py](agentic/stub_orchestrator.py) - Lightweight test version

