# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

### Backend
```bash
cd backend

# Install dependencies (prefer uv)
uv venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Run dev server (with auto-reload)
python main.py
# → http://localhost:8000   API docs: http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
npm run build      # Production build to dist/
npm run preview    # Preview production build
```

### Quick start (Windows only)
```
start.bat          # Launches both servers in separate terminals
```
Note: `start.bat` uses plain `pip`, not `uv`. For clean isolation, activate the uv venv manually before running the backend.

---

## Environment Variables

Copy `.env.example` → `.env`. Required keys:

```env
# OpenAI (Whisper transcription)
OPENAI_API_KEY=

# YouTube Data API v3 (comments; optional — app works without it)
YOUTUBE_API_KEY=

# Qdrant Cloud (vector database)
QDRANT_URL=
QDRANT_API_KEY=

# Azure OpenAI — chat + embeddings
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4.1-ria-dev-01
AZURE_OPENAI_LLM_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01

# Azure OpenAI — brand verification (web_search tool, separate deployment)
AZURE_OPENAI_BRAND_DEPLOYMENT=gpt-5.4-mini
AZURE_OPENAI_BRAND_API_VERSION=2025-04-01-preview
```

The app uses **two separate Azure OpenAI clients**: one for chat/embeddings (`gpt-4.1-ria-dev-01`) and one exclusively for brand verification that calls the `web_search` tool (`gpt-5.4-mini`). These are different deployments, different API versions.

---

## Architecture Overview

### Stack
- **Backend**: Python FastAPI + Uvicorn, served on port 8000
- **Frontend**: React 18 + Vite, served on port 5173 (proxies `/api` to 8000 via `vite.config.js`)
- **Vector DB**: Qdrant Cloud, collection `youtube_rag`, cosine similarity, 1536-dim embeddings
- **LLM**: Azure OpenAI (`gpt-4.1-ria-dev-01`) for all RAG + analysis; OpenAI Whisper (`whisper-1`) for transcription
- **Embeddings**: Azure OpenAI `text-embedding-3-small`

### Backend modules and responsibilities

| File | Role |
|---|---|
| `main.py` | FastAPI routes, request/response models, orchestration |
| `youtube_extractor.py` | YouTube transcript (captions + Whisper fallback), comments via Data API, local audio transcription |
| `teambhp_scraper.py` | Real Chrome browser (nodriver) to bypass Cloudflare; BeautifulSoup parses vBulletin HTML |
| `chunker.py` | Semantic chunking (embedding-based sentence similarity), comment thread chunking, TeamBHP post chunking |
| `translator.py` | langdetect → batch-translate non-English chunks to English before indexing |
| `vector_store.py` | Qdrant CRUD: embed + upsert, cosine search, delete by video_id filter, scroll all chunks |
| `retriever.py` | Hybrid search: dense (Qdrant) + BM25 (rank-bm25) + FlashRank cross-encoder reranking |
| `rag_engine.py` | Build context from retrieved chunks, call Azure LLM, stream tokens |
| `guardrail.py` | Two-layer input safety: regex injection patterns + OpenAI moderation API |
| `analysis_pipeline.py` | LangGraph 7-node DAG: extract brands/models/features → verify → dedup → aggregate → cache |
| `brand_dedup_pipeline.py` | Post-processing pass over cached analysis: company-level + model-level LLM dedup |

---

## Data Flow: URL → Chat Answer

```
User submits URL or audio file
        │
        ▼
main.py routes to _process_youtube / _process_teambhp / upload_audio
        │
        ├─ YouTube: parallel fetch (metadata, transcript/Whisper, comments)
        ├─ TeamBHP: nodriver Chrome → BS4 parse vBulletin, paginate
        └─ Audio: ffmpeg denoise → Whisper (chunked parallel if >25MB)
        │
        ▼
chunker.py — semantic_chunk_transcript / chunk_comment_threads / chunk_teambhp_posts
  • Sentence embed → cosine similarity breaks → min 500 / max 2000 tokens per chunk
        │
        ▼
translator.py — langdetect each chunk → batch-translate non-English to English
        │
        ▼
vector_store.py — embed all chunks (text-embedding-3-small) → upsert to Qdrant

=== CHAT ===

User asks question → POST /api/chat/stream
        │
        ▼
retriever.hybrid_search()
  1. Dense: Qdrant cosine (top 1000) filtered by video_id
  2. BM25: in-memory index over all chunks for that video
  3. Fusion: 0.70 × dense_norm + 0.30 × bm25_norm
  4. Rerank: FlashRank cross-encoder (ms-marco-MiniLM-L-12-v2)
        │
        ▼
rag_engine — token budget (40k), build context sections, call Azure LLM
  • Streaming response: first line "__SOURCES__{json}\n", then token stream
        │
        ▼
Frontend ChatInterface.jsx parses __SOURCES__ line, streams remaining tokens into message
```

---

## Analysis Pipeline (LangGraph)

`POST /api/analyze/{video_id}` triggers a 7-node LangGraph DAG defined in `analysis_pipeline.py`. Results cache to `backend/analysis_cache/{video_id}.json`.

```
load_chunks → extract_info → dedup_verify_brands → analyze_sentiments
           → update_metadata → aggregate_results → save_results
```

- **extract_info**: Groups chunks into ~6000-char batches → one JSON-mode LLM call per batch. Returns brands → models → features + verbatim + sentiment per chunk.
- **dedup_verify_brands**: Single LLM call with `web_search` tool to confirm brand names are real manufacturers (drops model names, codes, non-manufacturers).
- **analyze_sentiments**: Parallel per-brand model dedup + feature dedup (LLM), then sentiment aggregation (majority-vote per feature, weighted average per model — no LLM call). Feature dedup uses a strict prompt that maps raw feature names to broad, standard automotive categories (e.g. "mileage"/"kmpl"/"fuel consumption" → "Fuel Efficiency"; "ABS"/"airbags"/"ADAS" → "Safety Features"). The goal is the fewest meaningful categories, not granular labels.
- **aggregate_results**: Recalculates bottom-up: feature → model → company → overall. Produces sentiment distribution percentages.

`POST /api/brand_dedup/{video_id}` runs `brand_dedup_pipeline.py` as a second pass — company-level merging (e.g., "Tata" + "Tata Motors" → canonical) then model-level merging (e.g., "XUV 700", "700", "XUV700" → "XUV 700"). Reads and overwrites the same JSON cache file.

Sentiment scoring: `positive=0.8, neutral=0.5, negative=0.2`. Thresholds: `≥0.6 → positive`, `≤0.4 → negative`.

---

## Key Non-Obvious Design Decisions

**TeamBHP scraping uses a real visible Chrome window** — nodriver explicitly avoids `--headless` to defeat Cloudflare's canvas/GPU/font fingerprinting. This is intentional; do not add headless flags. The scraper waits up to 15 seconds for any Cloudflare challenge to resolve before parsing.

**Streaming format has a metadata line prefix** — the `/api/chat/stream` endpoint yields `__SOURCES__{json}\n` as its very first line, then raw text tokens. The frontend in `ChatInterface.jsx` detects this prefix to split sources from response text. Do not change this protocol without updating both sides.

**Two Azure OpenAI clients in the codebase** — `analysis_pipeline.py` and `brand_dedup_pipeline.py` use a separate `_brand_openai` client configured with `AZURE_OPENAI_BRAND_*` env vars and `responses.create()` (Responses API with built-in `web_search` tool). All other files use the standard `chat.completions.create()` client.

**Translation happens before embedding** — `translate_chunks()` in `vector_store.add_chunks()` translates non-English chunks to English *before* generating embeddings, so the entire vector space is English-only. Translated chunks retain `original_text` and `language` in metadata.

**BM25 index is rebuilt per query** — `retriever._bm25_search()` calls `fetch_chunks_for_bm25()` (Qdrant scroll) and rebuilds the in-memory rank-bm25 index on every request. There is no caching. For large video collections this is the main retrieval latency bottleneck.

**Whisper large-file handling** — files >25 MB are split into 5-minute audio segments (pydub), transcribed in parallel (up to 8 workers), then stitched in original order. Each segment is preprocessed with ffmpeg: `afftdn=nf=-25` (FFT denoiser) + `dynaudnorm=p=0.95` + mono 16 kHz. If ffmpeg is unavailable, preprocessing is skipped silently.

**Qdrant collection is created once with a keyword index on `video_id`** — `vector_store._ensure_collection()` is idempotent and called on first use. The keyword index enables efficient `delete_video_chunks()` filtering without full-scan.

**Audio upload IDs are MD5-stable** — `audio_` + first 12 chars of MD5(file content), so re-uploading the same file is idempotent (old chunks are deleted and re-indexed).

**Report generation is entirely frontend-side** — `DownloadReport.jsx` calls `/api/videos` then `/api/analysis/{video_id}` for each video, merges results via `combineBrands()` (weighted by mention count), and writes a self-contained HTML document into a new browser tab via `window.open()`. There is no backend report endpoint. The report contains five sections:
1. **Executive Summary** — stat cards (videos indexed/analyzed, unique brands, data chunks) + combined SVG donut chart + sentiment percentage bars.
2. **Company Leaderboard** — SVG horizontal bar chart (top 14 brands by mention count); bars are stacked by pos/neu/neg ratio; hover tooltip shows sentiment breakdown.
3. **Company & Brand Comparison** (interactive, `no-print`) — dropdowns to pick Company A vs Company B (or Brand/Model A vs B); renders a side-by-side card with praised/criticized columns and a "Get Recommendations" button that streams from `/api/recommend`.
4. **Company & Brand Deep Dive** (interactive, `no-print`) — single-select dropdown; "Analyse" button streams from `/api/brand-insight` which auto-identifies competitors from the dataset.
5. **Per-brand insight cards** — praised/criticized columns aggregated across all models; expandable `<details>` sub-sections per model. The "Save as PDF" button triggers `window.print()`; `.no-print` CSS hides the interactive sections when printing.

The generated HTML embeds all JavaScript inline so the comparison and deep-dive features work after the browser tab is opened. These features call back to `http://localhost:8000` directly (not via the Vite proxy) because the report runs in a plain browser tab with no dev-server context.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/process` | Index YouTube URL or TeamBHP thread |
| POST | `/api/upload-audio` | Upload + transcribe local audio |
| POST | `/api/chat` | Non-streaming chat |
| POST | `/api/chat/stream` | SSE streaming chat |
| GET | `/api/videos` | List all indexed videos + chunk count |
| DELETE | `/api/videos/{video_id}` | Remove all chunks for a video |
| POST | `/api/analyze/{video_id}` | Run LangGraph analysis pipeline |
| GET | `/api/analysis/{video_id}` | Fetch cached analysis (404 if not run) |
| POST | `/api/brand_dedup/{video_id}` | Run post-processing brand dedup |
| POST | `/api/recommend` | Stream LLM comparison recommendations for two brands/companies |
| POST | `/api/brand-insight` | Stream LLM deep-dive analysis for a single brand (auto-identifies competitors) |

`/api/process` auto-detects YouTube vs TeamBHP via `teambhp_scraper.is_teambhp_url()`.

---

## Frontend Structure

```
frontend/src/
  App.jsx              # Layout: header (title + DownloadReport button) + sidebar + main (Chat | Analysis tabs)
  components/
    YouTubeInput.jsx   # URL mode (YouTube + TeamBHP) + audio upload mode (drag-and-drop)
    StoredVideos.jsx   # Video list; click to filter chat by video_id
    ChatInterface.jsx  # Streaming chat; parses __SOURCES__ prefix; markdown rendering
    AnalysisPanel.jsx  # Collapsible brand → model → feature tree with sentiment viz
    DownloadReport.jsx # Generates a standalone interactive HTML report opened in a new tab (print → PDF)
```

Vite proxies all `/api/*` requests to `http://localhost:8000` (configured in `vite.config.js`), so no CORS issues in development. In production, the backend's CORS middleware allows `localhost:5173` and `localhost:3000`.

---

## Guardrail Layers

Every chat request passes through `guardrail.check_input()` before retrieval:

1. **Regex** (instant) — ~15 patterns for prompt injection, role-switching, system-prompt extraction, DAN, `<system>`, `[INST]`, `<<SYS>>` tokens.
2. **Moderation API** (~50ms) — Azure OpenAI moderation prompt; checks hate/harassment/self-harm/sexual/violence. **Fails open** — if the API call errors, the user is NOT blocked.

Layer 1 runs first; Layer 2 only runs if Layer 1 passes.

---

## Recommend & Brand-Insight Endpoints

### `POST /api/recommend` — `RecommendRequest`
Compares two brands or companies side-by-side. Streams plain text (markdown) from the LLM.

```json
{
  "name_a": "Mahindra", "name_b": "Tata Motors",
  "mode": "company",              // "company" | "model"
  "sentiment_a": "positive", "sentiment_b": "neutral",
  "positives_a": ["ride quality"], "negatives_a": ["service"],
  "positives_b": ["value for money"], "negatives_b": ["build quality"],
  "company_a": null, "company_b": null   // only for model mode
}
```

Response: streamed markdown with four sections — "What to improve" for each, "Competitive edge", "Key battleground". Max 350 words.

### `POST /api/brand-insight` — `BrandInsightRequest`
Deep-dive on a single brand. In two steps: (1) non-streaming LLM call to identify real competitors from `all_brands`; (2) streaming analysis. Max 450 words.

```json
{
  "name": "Mahindra", "mode": "company",
  "sentiment": "positive",
  "positives": ["ride quality"], "negatives": ["service cost"],
  "company": null,               // parent company (model mode only)
  "all_brands": [{ "name": "Tata Motors", "sentiment": "neutral", "mention_count": 12,
                   "positives": [...], "negatives": [...] }]
}
```

Response: streamed markdown — "Overall Strengths", "Key Weaknesses", "Improvement Areas", "What Competitors Do Better".

---

## Complete System Flowchart

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        USER ENTRY POINTS                                ║
╚══════════════════════════════════════════════════════════════════════════╝
        │                       │                        │
        ▼                       ▼                        ▼
  YouTube URL            TeamBHP URL             Local Audio File
  (YouTubeInput)         (YouTubeInput)          (YouTubeInput drag-drop)
        │                       │                        │
        ▼                       ▼                        ▼
  POST /api/process       POST /api/process       POST /api/upload-audio
        │                       │                        │
        ▼                       │                        ▼
  extract_video_id()      is_teambhp_url()=True   ffmpeg preprocess
        │                       │                  (denoise + mono 16kHz)
        │                       ▼                        │
        │              nodriver Chrome (visible)         ▼
        │              bypass Cloudflare           OpenAI Whisper
        │              BeautifulSoup parse         (chunked parallel
        │              vBulletin HTML              if file > 25MB)
        │              paginate thread                    │
        │                       │                        │
        ▼                       │                        │
  ThreadPoolExecutor(3)         │                        │
  ┌─────────────────────────────┐                        │
  │  get_video_metadata()       │                        │
  │  get_transcript() or        │                        │
  │    get_transcript_whisper() │                        │
  │  get_comments_with_replies()│                        │
  └─────────────────────────────┘                        │
        │                       │                        │
        ▼                       ▼                        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                         chunker.py                                │
  │  YouTube transcript → semantic_chunk_transcript()                 │
  │    • Split into sentences                                         │
  │    • Batch-embed all sentences (text-embedding-3-small)           │
  │    • Walk sentence-by-sentence:                                   │
  │        sim < 0.5 AND tokens ≥ min → semantic break               │
  │        tokens + next > max → force break                          │
  │    • Min 500 / Max 2000 tokens per chunk                          │
  │  YouTube comments → chunk_comment_threads()                       │
  │    • 1 comment thread (top + all replies) = 1 chunk               │
  │  TeamBHP posts → chunk_teambhp_posts()                            │
  │    • 1 post = 1 chunk (with [Post by author | date]: header)      │
  │  Audio → semantic_chunk_transcript() (same as YouTube)            │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                       translator.py                               │
  │  langdetect each chunk → if non-English → batch-translate to EN   │
  │  Translated chunks keep original_text + language in metadata      │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                       vector_store.py                             │
  │  embed all chunks (text-embedding-3-small, 1536-dim)              │
  │  upsert to Qdrant Cloud (collection: youtube_rag, cosine)         │
  │  keyword index on video_id for fast filtering + deletion          │
  └───────────────────────────────────────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════╗
║                           CHAT FLOW                                     ║
╚══════════════════════════════════════════════════════════════════════════╝

  User types question → ChatInterface.jsx → POST /api/chat/stream
        │
        ▼
  guardrail.check_input()
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 1: regex (~15 patterns) — injection, role-switch │
  │  Layer 2: Azure moderation API (fails open on error)    │
  └─────────────────────────────────────────────────────────┘
        │ (passes)
        ▼
  retriever.hybrid_search(question, video_id)
  ┌───────────────────────────────────────────────────────────────────┐
  │  1. Dense:  Qdrant cosine search (top 1000, filtered by video_id) │
  │  2. BM25:   scroll ALL chunks for video → rebuild rank-bm25       │
  │             index in-memory (rebuilt every query — no cache)      │
  │  3. Fusion: 0.70 × dense_norm + 0.30 × bm25_norm                 │
  │  4. Rerank: FlashRank cross-encoder ms-marco-MiniLM-L-12-v2       │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  rag_engine.answer_question_stream()
  • Token budget 40k — build context sections from reranked chunks
  • Call Azure LLM (gpt-4.1-ria-dev-01) with streaming
  • First yield: "__SOURCES__{json}\n"
  • Subsequent yields: raw text tokens
        │
        ▼
  StreamingResponse → ChatInterface.jsx
  • Detects "__SOURCES__" prefix on first chunk → extracts sources JSON
  • Streams remaining tokens into assistant message bubble
  • Renders markdown via ReactMarkdown
  • Shows source chips (Transcript / Comments / TeamBHP)


╔══════════════════════════════════════════════════════════════════════════╗
║                     ANALYSIS PIPELINE FLOW                              ║
╚══════════════════════════════════════════════════════════════════════════╝

  AnalysisPanel "Run Analysis" → POST /api/analyze/{video_id}
        │
        ▼
  analysis_pipeline.run_analysis()  [LangGraph 7-node DAG]
        │
        ▼
  ┌─ Node 1: load_chunks ─────────────────────────────────────────────┐
  │  Scroll all chunks for video_id from Qdrant (paginated, 100/page) │
  │  Sort by (source, chunk_index)                                     │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 2: extract_info ────────────────────────────────────────────┐
  │  Group chunks by source → split into ~6000-char batches            │
  │  ThreadPoolExecutor(20): one JSON-mode LLM call per batch          │
  │  Each call returns: brand → models → features + verbatim +        │
  │    sentiment per feature; overall_sentiment per model              │
  │  Result replicated back to every chunk in the batch               │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 3: dedup_verify_brands ─────────────────────────────────────┐
  │  Collect all unique raw brand names across all chunks              │
  │  Single web-search LLM call (_brand_openai / Responses API):      │
  │    • Category A (manufacturer) → web-search confirm → keep        │
  │    • Category B (model name) → drop                               │
  │    • Category C (brand+model combined) → drop                     │
  │    • Category D (number/code) → drop                              │
  │  Multiple raws → same canonical → merged into one entry            │
  │  Rewrites all extractions: drop unverified, remap raw → canonical │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 4: analyze_sentiments ──────────────────────────────────────┐
  │  Step 1 — Model-name dedup (parallel, one LLM call per brand):    │
  │    "700" → "XUV 700", "3XO" → "XUV 3XO" etc.                     │
  │    Merges index entries that resolve to same canonical model       │
  │  Step 2 — Feature dedup (parallel, one LLM call per brand):       │
  │    "mileage"/"kmpl" → "Fuel Efficiency"                           │
  │    "ABS"/"airbags"  → "Safety Features" etc.                      │
  │    Maps to broad standard automotive categories (2-4 words)       │
  │  Step 3 — Sentiment aggregation (no LLM call):                    │
  │    Per feature: majority-vote across all verbatim occurrences      │
  │    Per model: mention-count-weighted average of feature scores     │
  │    Score map: positive=0.8, neutral=0.5, negative=0.2             │
  │    Thresholds: ≥0.6 → positive, ≤0.4 → negative                  │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 5: update_metadata ─────────────────────────────────────────┐
  │  ThreadPoolExecutor(20): write brands/models/features back into    │
  │  Qdrant payload for each chunk (enables future filtered search)    │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 6: aggregate_results ───────────────────────────────────────┐
  │  Bottom-up recalculation:                                         │
  │    feature mention_count-weighted score → model sentiment          │
  │    model mention_count-weighted score   → company sentiment        │
  │    company mention_count-weighted score → overall sentiment        │
  │  Sentiment distribution %: per-model mention-count weighted        │
  │  Sort brands by total mention_count descending                    │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Node 7: save_results ────────────────────────────────────────────┐
  │  Write JSON to backend/analysis_cache/{video_id}.json             │
  │  Schema: { video_id, video_title, aggregated: {                   │
  │    overall_sentiment, total_chunks_analyzed, total_brands,        │
  │    sentiment_distribution, brand_analysis: {                      │
  │      company: { overall_sentiment, mention_count, models: {       │
  │        model: { overall_sentiment, mention_count, features: [     │
  │          { name, sentiment, mention_count, verbatim[] } ] } } } } │
  │  } }                                                              │
  └───────────────────────────────────────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════╗
║                     BRAND DEDUP PIPELINE FLOW                           ║
╚══════════════════════════════════════════════════════════════════════════╝

  "Run Brand Dedup" button → POST /api/brand_dedup/{video_id}
        │
        ▼
  brand_dedup_pipeline.run_brand_dedup()
  Reads analysis_cache/{video_id}.json
        │
        ▼
  ┌─ Phase 1: Company-level dedup ────────────────────────────────────┐
  │  Single web-search LLM call for ALL company names                 │
  │  "Tata" + "Tata Motors" → "Tata Motors"                          │
  │  "MG" + "MG Motor" → "MG Motor"                                  │
  │  Merges company entries that map to same canonical                │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Phase 2: Model-level dedup (parallel per company) ───────────────┐
  │  ThreadPoolExecutor(10): one web-search LLM call per company       │
  │  "XUV 700" + "700" + "XUV700" → "XUV 700"                        │
  │  Merges model entries, recalculates company sentiment              │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  Recalculate aggregated stats (brand counts, sentiment distribution)
  Overwrite analysis_cache/{video_id}.json


╔══════════════════════════════════════════════════════════════════════════╗
║                     REPORT GENERATION FLOW                              ║
╚══════════════════════════════════════════════════════════════════════════╝

  "↓ Download Report" button (DownloadReport.jsx in App header)
        │
        ▼
  GET /api/videos → all video_ids
        │
        ▼
  Promise.all: GET /api/analysis/{video_id} for each video
  (skips videos with no cached analysis)
        │
        ▼
  combineBrands(analyzed)
  • Merge brand_analysis across all videos
  • Weighted by mention_count (more-mentioned = higher weight)
  • Produces unified company → model → feature tree with avg scores
        │
        ▼
  generateReport(analyses, allVideos)
  • svgDonut() — CSS donut chart for combined sentiment
  • svgHBars() — stacked horizontal bars for brand leaderboard
  • generateComparisonSectionHtml() — embeds JavaScript + data JSON
  • brandInsightsHtml() — per-brand praised/criticized + model cards
        │
        ▼
  window.open("", "_blank") → write HTML string → window.document.close()

  [Inside the generated HTML tab — interactive features]
        │
        ├─ Compare button → fetch POST http://localhost:8000/api/recommend
        │    • Streams markdown recommendations for two brands/companies
        │
        └─ Analyse button → fetch POST http://localhost:8000/api/brand-insight
             • Step 1: LLM identifies real competitors from all_brands list
             • Step 2: Streams markdown deep-dive analysis


╔══════════════════════════════════════════════════════════════════════════╗
║                     LLM CLIENT ROUTING                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────┐
  │  Standard client (_openai / chat.completions.create)    │
  │  Model: gpt-4.1-ria-dev-01                              │
  │  API version: 2024-12-01-preview                        │
  │  Used by:                                               │
  │    • rag_engine.py         — chat Q&A                   │
  │    • analysis_pipeline.py  — extract_info node          │
  │    • analysis_pipeline.py  — model dedup per brand      │
  │    • analysis_pipeline.py  — feature dedup per brand    │
  │    • main.py /api/recommend — comparison streaming      │
  │    • main.py /api/brand-insight — deep-dive streaming   │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Brand client (_brand_openai / responses.create)        │
  │  Model: gpt-5.4-mini                                    │
  │  API version: 2025-04-01-preview (Responses API)        │
  │  Tool: web_search (built-in)                            │
  │  Used by:                                               │
  │    • analysis_pipeline.py  — dedup_verify_brands node   │
  │    • brand_dedup_pipeline.py — Phase 1 company dedup    │
  │    • brand_dedup_pipeline.py — Phase 2 model dedup      │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Embedding client (AzureOpenAI in chunker.py)           │
  │  Model: text-embedding-3-small (1536-dim)               │
  │  API version: 2024-02-01                                │
  │  Used by:                                               │
  │    • chunker.py — sentence embeddings for semantic split│
  │    • vector_store.py — chunk embeddings before upsert   │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  OpenAI (non-Azure) — Whisper transcription only        │
  │  Model: whisper-1                                       │
  │  Used by:                                               │
  │    • youtube_extractor.py — captions fallback + audio   │
  └─────────────────────────────────────────────────────────┘
```
