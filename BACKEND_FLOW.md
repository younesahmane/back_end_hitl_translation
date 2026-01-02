# Complete Backend Flow Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Data Models](#data-models)
4. [Complete API Endpoints](#complete-api-endpoints)
5. [Workflow Scenarios](#workflow-scenarios)
6. [Service Layer](#service-layer)
7. [Storage & Persistence](#storage--persistence)
8. [Error Handling](#error-handling)
9. [Configuration](#configuration)
10. [Performance & Scalability](#performance--scalability)

---

## System Overview

The HITL Backend is a Human-in-the-Loop translation correction system that:
- **Ingests** documents (PDF/DOCX) in multiple languages
- **Segments** documents into translatable units
- **Generates** translations using multiple LLM models
- **Collects** human corrections/edits
- **Validates** corrections through an orchestrator pipeline
- **Manages** translation memories (TM)
- **Exports** corrected documents
- **Trains** models on validated corrections (via orchestrator)

```
┌──────────────────────────────────────────────────────────────────┐
│                      Frontend Application                        │
│              (React/Vue - Translation Editor UI)                 │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   │ HTTP/REST
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     HITL Flask Backend                           │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              API Routes & Request Validation              │  │
│  │  ┌─────────────────┬─────────────────┬──────────────────┐ │  │
│  │  │  Documents      │  Segments       │  Pipeline        │ │  │
│  │  │  - upload       │  - save_draft   │  - process-batch │ │  │
│  │  │  - list         │  - status       │  - validation    │ │  │
│  │  │  - segments     │  - reasoning    │  - dedup         │ │  │
│  │  │  - export       │  - human_edit   │  - persist       │ │  │
│  │  └─────────────────┴─────────────────┴──────────────────┘ │  │
│  │  ┌──────────────────┬──────────────────┬─────────────────┐ │  │
│  │  │  LLM Services    │  TM Services     │  Downloads      │ │  │
│  │  │  - generate      │  - search        │  - files        │ │  │
│  │  │  - multi-model   │  - add entries   │  - exports      │ │  │
│  │  └──────────────────┴──────────────────┴─────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         Service Layer (Business Logic)                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │Segmenter │  │LLM Dummy │  │Storage   │  │Exporter  │  │  │
│  │  │(PDF/DOCX)│  │(Mocking) │  │(JSON/FS) │  │(Export)  │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  │  ┌────────────────────────────┐  ┌────────────────────┐  │  │
│  │  │Orchestrator Adapter        │  │TM Service          │  │  │
│  │  │(Pipeline coordination)     │  │(Translation Memory)│  │  │
│  │  └────────────────────────────┘  └────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         Orchestrator Agent (Complex Pipeline)             │  │
│  │  ┌────────────┐ ┌──────────┐ ┌─────────────┐ ┌─────────┐ │  │
│  │  │Preprocessor│ │Validator │ │Deduplicator│ │Executor │ │  │
│  │  │(Normalize) │ │(Quality) │ │(FAISS)     │ │(Persist)│ │  │
│  │  └────────────┘ └──────────┘ └─────────────┘ └─────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │         Monitor Agent (Cycle Management)            │ │  │
│  │  │  - Tracks corrections count                          │ │  │
│  │  │  - Triggers finetuning when threshold reached       │ │  │
│  │  │  - Archives datasets                                │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Storage Layer (Persistence)                  │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ Filesystem (File-based)                              │ │  │
│  │  │ ├─ storage/uploads/{doc_id}/           [Documents]  │ │  │
│  │  │ ├─ storage/documents.json               [Doc Index]  │ │  │
│  │  │ ├─ storage/segments_{doc_id}.json       [Segments]   │ │  │
│  │  │ ├─ storage/tm.json                      [TM Index]   │ │  │
│  │  │ ├─ storage/master.jsonl                 [Training]   │ │  │
│  │  │ ├─ storage/dedup_index.faiss            [FAISS]      │ │  │
│  │  │ ├─ storage/monitor_state.json           [Monitor]    │ │  │
│  │  │ ├─ storage/archive/                     [Datasets]   │ │  │
│  │  │ └─ storage/exports/                     [Downloads]  │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Architecture

### Layered Design

```
┌─────────────────────────────────────────┐
│      Request/Response Layer             │
│   (Flask Routes, HTTP Contracts)        │
├─────────────────────────────────────────┤
│   Orchestration/Coordination Layer      │
│  (Pipeline Routes: documents, segments, │
│    pipeline, llm, tm, downloads)        │
├─────────────────────────────────────────┤
│       Service/Business Logic Layer      │
│  (Segmenter, Storage, Exporter, TM,    │
│    Orchestrator Adapter, LLM Service)  │
├─────────────────────────────────────────┤
│     Agent Pipeline Layer (Optional)     │
│  (Complex multi-stage processing via    │
│    Orchestrator Agent if configured)    │
├─────────────────────────────────────────┤
│     Storage/Persistence Layer           │
│  (JSON files, FAISS index, filesystem)  │
└─────────────────────────────────────────┘
```

### Responsibility Breakdown

| Layer | Responsibility |
|-------|-----------------|
| **Routes** | HTTP request validation, response formatting |
| **Services** | Business logic, file processing, data transformation |
| **Orchestrator Agent** | Complex multi-stage ML pipeline (validation, dedup, monitoring) |
| **Storage** | Thread-safe persistence, atomic operations |

---

## Data Models

### Document Model
```python
{
  "doc_id": "doc_abc1234567",       # Unique identifier
  "filename": "resume.pdf",          # Original filename
  "language_pair": "en-ar",         # Source-target language
  "created_at": "2024-01-02T10:30:00Z",
  "stored_path": "storage/uploads/doc_abc1234567/original.pdf"
}
```

### Segment Model
```python
{
  "id": "seg_abc1234567890",        # Unique segment ID
  "doc_id": "doc_abc1234567",       # Parent document
  "index": 1,                        # Position in document
  "segment_type": "paragraph",       # Type (paragraph, sentence, etc)
  "source": "Original English text",
  "llm_outputs": {
    "model1": "ترجمة من النموذج 1",
    "model2": "ترجمة من النموذج 2",
    "model3": "ترجمة من النموذج 3"
  },
  "human_edit": "التصحيح البشري",
  "status": "pending|accept|reject",
  "reason": "Optional explanation",
  "last_saved_at": "2024-01-02T10:35:00Z"
}
```

### Correction/Result Model
```python
{
  "index": 1,
  "source": "Original text",
  "llm_outputs": { "model1": "...", ... },
  "human_translation": "Corrected translation",
  "decision": "accept|reject",
  "reason": "Validation reason",
  "validation_score": 0.85,
  "is_duplicate": false,
  "stored_at": "2024-01-02T10:35:00Z"
}
```

### Translation Memory Entry
```python
{
  "entry_id": "tm_abc1234567",
  "language_pair": "en-ar",
  "source": "Hello world",
  "target": "مرحبا بالعالم"
}
```

### Monitor State Model
```python
{
  "last_finetuning_timestamp": "2024-01-01T10:00:00Z",
  "current_line_count": 5234,
  "days_since_last_finetuning": 2,
  "triggered": false
}
```

---

## Complete API Endpoints

### 1. Health Check
**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-02T10:30:00Z"
}
```

**Purpose:** Service availability check

---

### 2. Document Management

#### 2.1 List All Documents
**Endpoint:** `GET /api/documents`

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "doc_abc1234567",
      "filename": "resume.pdf",
      "language_pair": "en-ar",
      "created_at": "2024-01-02T10:30:00Z",
      "stored_path": "storage/uploads/doc_abc1234567/original.pdf"
    }
  ]
}
```

**Purpose:** Fetch all uploaded documents with metadata

---

#### 2.2 Upload Document
**Endpoint:** `POST /api/documents/upload`

**Content-Type:** `multipart/form-data`

**Request:**
```
file: <PDF or DOCX file>
language_pair: "en-ar" (optional, default: "en-ar")
```

**Response:**
```json
{
  "doc_id": "doc_abc1234567",
  "filename": "resume.pdf",
  "language_pair": "en-ar",
  "segments_count": 42,
  "segment_ids": ["seg_001", "seg_002", ...],
  "segments": [
    {
      "id": "seg_001",
      "index": 1,
      "source": "First paragraph text...",
      "llm_outputs": {
        "model1": "...",
        "model2": "...",
        "model3": "..."
      },
      "human_edit": "",
      "status": "pending"
    }
  ]
}
```

**Process Flow:**
1. File validation (PDF/DOCX only)
2. Secure filename processing
3. Storage in `uploads/{doc_id}/`
4. Document segmentation (by paragraph/sentence)
5. Dummy LLM outputs generation (mock models)
6. Segment length limit enforcement (1200 chars max)
7. Metadata persistence

---

#### 2.3 Get Document Segments
**Endpoint:** `GET /api/documents/<doc_id>/segments`

**Response:**
```json
{
  "doc_id": "doc_abc1234567",
  "segments": [
    {
      "id": "seg_001",
      "source": "...",
      "llm_outputs": {...},
      "human_edit": "",
      "status": "pending",
      "last_saved_at": null
    }
  ]
}
```

**Purpose:** Retrieve all segments for a document

---

#### 2.4 Export Document
**Endpoint:** `POST /api/documents/<doc_id>/export`

**Request:**
```json
{
  "export_format": "docx"
}
```

**Response:**
```json
{
  "export_filename": "doc_abc1234567_translated.docx",
  "download_url": "/downloads/doc_abc1234567_translated.docx",
  "segments_count": 42
}
```

**Process Flow:**
1. Retrieve all segments for document
2. Use `human_edit` (or fallback to `llm_outputs.model1`)
3. Generate DOCX file with translations
4. Store in `exports/` directory
5. Return download URL

---

### 3. Segment Management

#### 3.1 Save Segment Draft
**Endpoint:** `PUT /api/segments/<segment_id>/draft`

**Request:**
```json
{
  "human_edit": "التصحيح المقترح من المستخدم"
}
```

**Response:**
```json
{
  "segment_id": "seg_001",
  "saved": true,
  "last_saved_at": "2024-01-02T10:35:00Z"
}
```

**Process Flow:**
1. Find segment by ID across all documents
2. Update `human_edit` field
3. Set `status` to "pending"
4. Timestamp `last_saved_at`
5. Persist to storage

---

### 4. LLM Services

#### 4.1 Generate LLM Outputs
**Endpoint:** `POST /api/llm/generate`

**Request:**
```json
{
  "language_pair": "en-ar",
  "sources": ["Hello world", "Good morning"],
  "models": ["model1", "model2", "model3"]
}
```

**Response:**
```json
{
  "language_pair": "en-ar",
  "items": [
    {
      "source": "Hello world",
      "llm_outputs": {
        "model1": "مرحبا بالعالم",
        "model2": "مرحبا بالعالم",
        "model3": "أهلا بالعالم"
      }
    },
    {
      "source": "Good morning",
      "llm_outputs": {
        "model1": "صباح الخير",
        "model2": "أهلا صباح الخير",
        "model3": "صباح النور"
      }
    }
  ]
}
```

**Purpose:** Generate translations from multiple LLM models (currently mock/dummy)

---

### 5. Translation Memory

#### 5.1 Search Translation Memory
**Endpoint:** `GET /api/tm/search?query=<query>&language_pair=<pair>`

**Response:**
```json
{
  "matches": [
    {
      "source": "Hello world",
      "target": "مرحبا بالعالم",
      "score": 0.95
    },
    {
      "source": "Hello there",
      "target": "مرحبا هناك",
      "score": 0.75
    }
  ]
}
```

**Scoring Algorithm:**
- Exact match: 1.0
- Substring match: 0.6 - 0.95
- Token overlap: 0.0 - 0.25

**Purpose:** Find similar translations from TM (supports fuzzy matching)

---

#### 5.2 Add to Translation Memory
**Endpoint:** `POST /api/tm/add`

**Request:**
```json
{
  "source": "Hello world",
  "target": "مرحبا بالعالم",
  "language_pair": "en-ar"
}
```

**Response:**
```json
{
  "added": true,
  "entry_id": "tm_abc1234567"
}
```

**Purpose:** Add new translation to memory (typically from validated corrections)

---

### 6. Pipeline Processing (Core Orchestrator)

#### 6.1 Process Batch (Main Entry Point)
**Endpoint:** `POST /api/pipeline/process-batch`

**Request:**
```json
{
  "language_pair": ["en", "ar"],
  "check_monitor": true,
  "frontend_payload": [
    {
      "source": "Good morning, everyone.",
      "llm_outputs": {
        "model1": "صباح الخير جميعا",
        "model2": "أهلا صباح الخير",
        "model3": "صباح الخير يا الجميع"
      },
      "human_edit": "صباح الخير جميعاً",
      "segment_id": "seg_001",
      "expected": "accept"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 1,
      "correction": {
        "decision": "accept",
        "reason": "Validation passed with good score",
        "source": "Good morning, everyone.",
        "llm_outputs": {...},
        "human_translation": "صباح الخير جميعاً",
        "validation_score": 0.92,
        "is_duplicate": false,
        "stored_at": "2024-01-02T10:35:00Z",
        "dedup": {
          "is_duplicate": false,
          "similarity": 0.45
        }
      },
      "test_passed": true
    }
  ],
  "dataset_path": "archive/finetune_dataset_2024-01-02.jsonl"
}
```

**Complete Process Flow:**
1. **Input Validation**
   - Validate language_pair format
   - Validate frontend_payload structure
   - Check required fields (source, llm_outputs, human_edit)

2. **Orchestrator Processing** (per item)
   - **PreprocessorAgent**: Normalize text (Arabic diacritics, Latin quotes, etc.)
   - **ValidatorAgent**: Validate quality using:
     - Semantic similarity to source
     - Consistency with LLM outputs
     - Language-specific rules
   - **DeduplicatorAgent**: Check FAISS index for duplicates
   - **ExecutorAgent**: Persist to master.jsonl
   - **MonitorAgent** (if enabled): Check cycle triggers

3. **Segment Status Update** (if segment_id provided)
   - Find segment in storage
   - Update status (accept/reject)
   - Save reason

4. **Monitor Check** (if check_monitor=true)
   - Check accumulated corrections
   - Trigger finetuning dataset if thresholds met
   - Archive previous datasets

5. **Response Generation**
   - Format results with decisions
   - Include dataset_path if monitoring triggered

**Orchestrator Pipeline Stages:**

```
Input Item
    ├─ source text
    ├─ LLM outputs (3 models)
    └─ human_edit (correction)
         │
         ▼
   1. PREPROCESSOR
      ├─ Normalize Arabic (diacritics, digits, quotes)
      ├─ Normalize Latin (apostrophes, quotes, hyphens)
      └─ Structure data
         │
         ▼
   2. VALIDATOR
      ├─ Semantic similarity check
      ├─ Quality scoring (CHRF, semantic)
      ├─ Grammar/language rules
      └─ Compute decision + reason
         │
         ▼
   3. DEDUPLICATOR
      ├─ Query FAISS index (threshold: 0.85)
      ├─ Check against master.jsonl
      └─ Detect & flag duplicates
         │
         ▼
   4. EXECUTOR
      ├─ Persist correction to master.jsonl
      ├─ Track decision
      └─ Ensure durability
         │
         ▼
   5. MONITOR (optional)
      ├─ Accumulate corrections
      ├─ Check line_threshold (default 10000)
      ├─ Check days_threshold (default 7)
      └─ Trigger finetuning + archive if needed
         │
         ▼
     Output Result
      ├─ decision (accept/reject)
      ├─ reason
      ├─ validation_score
      └─ dedup info
```

---

### 7. Download Endpoint

#### 7.1 Download Exported File
**Endpoint:** `GET /downloads/<path:filename>`

**Response:** Binary file (DOCX) with `Content-Disposition: attachment`

**Purpose:** Download exported translated document

---

## Workflow Scenarios

### Scenario 1: Complete Translation Workflow

```
1. USER UPLOADS DOCUMENT
   POST /api/documents/upload
   ├─ File saved to storage/uploads/
   ├─ Document segmented by paragraphs
   ├─ LLM outputs generated (3 models)
   └─ Segments stored as JSON

2. USER VIEWS DOCUMENT
   GET /api/documents/{doc_id}/segments
   ├─ Retrieve all segments
   ├─ Display source + LLM outputs
   └─ Show human_edit field (empty initially)

3. USER EDITS SEGMENTS
   PUT /api/segments/{segment_id}/draft
   ├─ Save human correction
   ├─ Update timestamp
   └─ Store in segment JSON

4. USER SUBMITS FOR VALIDATION
   POST /api/pipeline/process-batch
   ├─ Orchestrator validates each correction
   ├─ Check deduplication
   ├─ Persist accepted corrections
   └─ Update segment status

5. MONITOR TRIGGERS FINETUNING
   (After 10000 lines or 7 days)
   ├─ Generate finetune dataset
   ├─ Archive old dataset
   └─ Reset counters

6. USER EXPORTS DOCUMENT
   POST /api/documents/{doc_id}/export
   ├─ Generate DOCX with accepted corrections
   ├─ Store in exports/ directory
   └─ Return download URL

7. USER DOWNLOADS DOCUMENT
   GET /downloads/{filename}
   └─ Receive translated DOCX
```

### Scenario 2: Translation Memory Workflow

```
1. SEARCH EXISTING TRANSLATIONS
   GET /api/tm/search?query=...
   ├─ Fuzzy match against TM
   └─ Return scored results

2. USER ACCEPTS SUGGESTION
   PUT /api/segments/{segment_id}/draft
   ├─ Use TM suggestion as human_edit
   └─ Save

3. VALIDATION & ACCEPTANCE
   POST /api/pipeline/process-batch
   ├─ Validate correction
   └─ Accept or reject

4. ADD VALIDATED CORRECTION TO TM
   POST /api/tm/add
   ├─ Store source + target pair
   ├─ Link to language_pair
   └─ Make available for future searches
```

### Scenario 3: Batch Processing Workflow

```
1. PREPARE BATCH
   ├─ Collect multiple segments
   └─ Include source, llm_outputs, human_edit

2. SUBMIT BATCH
   POST /api/pipeline/process-batch
   ├─ Process all items
   ├─ Generate results array
   └─ Check monitor thresholds

3. PROCESS RESULTS
   ├─ Update segment statuses
   ├─ Log decisions & reasons
   ├─ Accumulate metrics
   └─ Trigger finetuning if needed

4. EXPORT & DOWNLOAD
   POST /api/documents/{doc_id}/export
   GET /downloads/{filename}
   └─ Get final translated document
```

---

## Service Layer

### 1. Storage Service (`services/storage.py`)

**Responsibilities:**
- Atomic JSON read/write operations
- Thread-safe persistence using locks
- Graceful handling of corrupted files
- Segment lookup by ID across all documents

**Key Functions:**
```python
ensure_storage_layout(cfg)              # Create directory structure
load_docs(docs_json_path)               # Load document index
save_docs(docs_json_path, docs)         # Persist documents
load_doc_segments(cfg, doc_id)          # Load segments for doc
save_doc_segments(cfg, doc_id, segments) # Persist segments
find_segment_by_id(cfg, segment_id)     # Find segment globally
update_segment(cfg, doc_id, segment)    # Update single segment
try_update_segments_from_batch_results() # Update from orchestrator
```

**Data Files:**
- `documents.json`: Document metadata index
- `segments_{doc_id}.json`: Segments for each document
- `tm.json`: Translation memory entries
- `master.jsonl`: Training dataset (from orchestrator)

---

### 2. Segmenter Service (`services/segmenter.py`)

**Responsibilities:**
- Extract text from PDF/DOCX files
- Split into logical segments (paragraphs, sentences)
- Enforce length limits
- Generate unique segment IDs

**Key Functions:**
```python
segment_document_file(filepath, doc_id, segment_type)
  ├─ Extract paragraphs from PDF/DOCX
  ├─ Split by length limit (1200 chars)
  └─ Return segments with metadata
```

**Features:**
- PDF support (pdfplumber or pypdf)
- DOCX support (python-docx)
- Intelligent paragraph splitting
- Word-level splitting for long segments
- Sentence-boundary awareness

---

### 3. LLM Service (`services/llm_dummy.py`)

**Responsibilities:**
- Generate mock/dummy LLM outputs
- Simulate multiple model responses
- Deterministic results for testing

**Key Functions:**
```python
dummy_generate_llm_outputs(source, language_pair, models)
  └─ Generate mock translations from models

dummy_llm_outputs_for_segments(segments)
  └─ Add LLM outputs to all segments
```

**Note:** Currently returns deterministic mock outputs. In production, replace with actual LLM API calls.

---

### 4. Exporter Service (`services/exporter.py`)

**Responsibilities:**
- Export segments to documents
- Generate DOCX files
- Handle fallback (human_edit → llm_outputs)

**Key Functions:**
```python
export_docx(doc_id, segments, exports_dir)
  ├─ Create DOCX document
  ├─ Add translated content (human_edit preferred)
  ├─ Save to exports directory
  └─ Return filename
```

---

### 5. Translation Memory Service (`services/tm_service.py`)

**Responsibilities:**
- Search translation memory
- Add new entries
- Score fuzzy matches

**Key Functions:**
```python
tm_search(tm_json_path, query, language_pair, limit)
  ├─ Load TM entries
  ├─ Score against query
  └─ Return top matches

tm_add(tm_json_path, source, target, language_pair)
  ├─ Create entry with ID
  └─ Persist to TM
```

**Scoring Algorithm:**
- Exact match: 1.0
- Substring: 0.6 - 0.95 (based on ratio)
- Token overlap: 0.0 - 0.25 (Jaccard similarity)

---

### 6. Orchestrator Adapter (`services/orchestrator_adapter.py`)

**Responsibilities:**
- Dynamic orchestrator loading
- Graceful fallback to dummy
- Caching for performance

**Key Functions:**
```python
get_orchestrator(cfg)
  ├─ Load from ORCHESTRATOR_IMPORT config
  ├─ Initialize with config params
  └─ Fall back to DummyOrchestrator if needed

run_orchestrator_batch(orchestrator, payload, language_pair, check_monitor)
  └─ Execute process_batch with standardized params
```

**Configuration Variables:**
- `ORCHESTRATOR_IMPORT`: Format `"module:ClassName"`
- `ORCHESTRATOR_MASTER_JSONL`: Path to training data
- `ORCHESTRATOR_FAISS_INDEX`: Path to dedup index
- `ORCHESTRATOR_ARCHIVE_DIR`: Path to archives
- `ORCHESTRATOR_MONITOR_STATE`: Path to state file
- `ORCHESTRATOR_LINE_THRESHOLD`: Lines before trigger
- `ORCHESTRATOR_DAYS_THRESHOLD`: Days before trigger
- `ORCHESTRATOR_FEW_SHOT_N`: Few-shot examples count

---

## Storage & Persistence

### Directory Structure

```
storage/
├── documents.json                    # Document index
├── tm.json                          # Translation memory
├── master.jsonl                     # Training dataset (from orchestrator)
├── monitor_state.json               # Monitor state (from orchestrator)
├── dedup_index.faiss                # FAISS index (from orchestrator)
│
├── uploads/                         # Original documents
│   ├── doc_abc1234567/
│   │   └── original.pdf
│   └── doc_xyz9876543/
│       └── original.docx
│
├── segments/                        # Segment JSON files (per document)
│   ├── segments_doc_abc1234567.json
│   └── segments_doc_xyz9876543.json
│
├── exports/                         # Exported translated documents
│   ├── doc_abc1234567_translated.docx
│   └── doc_xyz9876543_translated.docx
│
└── archive/                         # Archived datasets (from orchestrator)
    ├── finetune_dataset_2024-01-01.jsonl
    └── finetune_dataset_2024-01-02.jsonl
```

### Persistence Strategy

**Thread Safety:**
- Global lock (`_LOCK`) protects all JSON operations
- Atomic writes using temporary files + `os.replace()`
- Prevents corruption on concurrent access

**Error Recovery:**
- Invalid JSON → reset to defaults
- Empty files → treat as empty collection
- Missing files → create on demand

**Data Durability:**
- All writes are synchronous
- No in-memory caching (except orchestrator singleton)
- Safe for multi-process deployments

---

## Error Handling

### HTTP Error Responses

All errors follow consistent format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

### Common Error Codes

| Code | HTTP | Scenario |
|------|------|----------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource doesn't exist |
| `SEGMENTATION_FAILED` | 400 | PDF/DOCX parsing failed |
| `INTERNAL` | 500 | Unexpected server error |

### Orchestrator Fallback

If orchestrator fails to load:
1. Log exception
2. Use `DummyOrchestrator` instead
3. Apply simple rules:
   - Empty human_edit → reject
   - human_edit < 8 chars → reject
   - Otherwise → accept

**Benefits:**
- System remains operational
- No data loss
- Allows graceful degradation

---

## Configuration

### Environment Variables

```bash
# Flask
FLASK_ENV=production
PORT=8001

# Orchestrator
AGENTS_MODE=real                    # "real" or "stub"
ORCHESTRATOR_IMPORT="agentic.orchestrator:OrchestratorAgent"

# Orchestrator Paths
ORCHESTRATOR_MASTER_JSONL=storage/master.jsonl
ORCHESTRATOR_FAISS_INDEX=storage/dedup_index.faiss
ORCHESTRATOR_ARCHIVE_DIR=storage/archive
ORCHESTRATOR_MONITOR_STATE=storage/monitor_state.json

# Orchestrator Thresholds
ORCHESTRATOR_LINE_THRESHOLD=10000
ORCHESTRATOR_DAYS_THRESHOLD=7
ORCHESTRATOR_FEW_SHOT_N=50
```

### Flask Configuration (app.py)

```python
app.config.update(
    STORAGE_DIR="storage",
    UPLOADS_DIR="storage/uploads",
    DOWNLOADS_DIR="storage/downloads",
    EXPORTS_DIR="storage/exports",
    DOCS_JSON="storage/documents.json",
    SEGMENTS_JSON="storage/segments.json",
    TM_JSON="storage/tm.json",
    MASTER_JSONL="storage/master.jsonl",
    DEDUP_FAISS="storage/dedup_index.faiss",
    MONITOR_STATE="storage/monitor_state.json",
    ARCHIVE_DIR="storage/archive",
    VERSION="0.1.0"
)
```

---

## Performance & Scalability

### Current Limitations

1. **JSON-based Storage**
   - Linear search for segments by ID
   - O(n) segment lookup across documents
   - Scales to ~10,000 documents

2. **Single-threaded Processing**
   - Sequential per-item orchestrator processing
   - No batch parallelization
   - Suitable for ~100 items/batch

3. **In-Memory FAISS Index**
   - Index loaded entirely in memory
   - ~100MB for typical use
   - Requires restart for large updates

4. **LLM Mocking**
   - No actual LLM calls
   - Useful for development
   - Production needs real LLM integration

### Optimization Strategies

**To improve throughput:**
1. Add database layer (PostgreSQL)
2. Implement segment indexing
3. Use async request processing
4. Add caching layer (Redis)
5. Parallelize batch processing

**To improve latency:**
1. Cache document metadata
2. Lazy-load segments on demand
3. Index segments by doc_id
4. Use FAISS GPU acceleration

**To improve reliability:**
1. Add request logging
2. Implement retry logic
3. Add metrics collection
4. Use database transactions

---

## Integration Checklist

Before deploying to production:

- [ ] Configure orchestrator properly (ORCHESTRATOR_IMPORT)
- [ ] Test document upload (PDF + DOCX)
- [ ] Verify segment segmentation
- [ ] Test pipeline processing
- [ ] Confirm monitor thresholds
- [ ] Set up export downloads
- [ ] Configure TM searches
- [ ] Test batch operations
- [ ] Verify error handling
- [ ] Load test with realistic data

---

## Related Documentation

- [ORCHESTRATOR_INTEGRATION.md](ORCHESTRATOR_INTEGRATION.md) - Detailed orchestrator integration guide
- [README.md](README.md) - Project overview
- [requirements.txt](requirements.txt) - Python dependencies
- [config.py](config.py) - Configuration file

