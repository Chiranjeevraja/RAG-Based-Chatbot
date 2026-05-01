import asyncio
import hashlib
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# On Windows the default SelectorEventLoop cannot run subprocesses (needed by
# nodriver/Chrome).  Set ProactorEventLoopPolicy once at process start so both
# uvicorn and asyncio.run() inside sync route handlers use the right loop type.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

from youtube_extractor import extract_video_id, get_transcript, get_transcript_whisper, get_comments_with_replies, get_video_metadata, transcribe_local_audio
from chunker import semantic_chunk_transcript, chunk_comment_threads, chunk_teambhp_posts
from vector_store import add_chunks, delete_video_chunks, list_stored_videos, get_collection_stats, get_video_title
from rag_engine import answer_question, answer_question_stream, client as _llm_client, CHAT_MODEL as _CHAT_MODEL
from analysis_pipeline import run_analysis, get_cached_analysis
from brand_dedup_pipeline import run_brand_dedup
from guardrail import check_input
from teambhp_scraper import is_teambhp_url, scrape_teambhp_thread

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac", ".wma"}

app = FastAPI(title="YouTube RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    url: str = Field(..., min_length=1)
    include_comments: bool = True
    use_whisper: bool = False
    chunk_size: int = Field(default=500, gt=0, le=4096)

    @field_validator("url")
    @classmethod
    def url_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("url must not be blank")
        return stripped


class ProcessResponse(BaseModel):
    video_id: str
    title: str
    channel: str
    source_type: Literal["youtube", "teambhp", "audio"] = "youtube"
    transcript_chunks: int = Field(default=0, ge=0)
    comment_chunks: int = Field(default=0, ge=0)
    post_chunks: Optional[int] = Field(default=None, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    transcript_available: bool = False
    transcript_method: Literal["captions", "whisper", ""] = ""
    comments_available: bool = False
    warnings: List[str] = []


class RecommendRequest(BaseModel):
    name_a: str
    name_b: str
    mode: Literal["company", "model"]
    sentiment_a: str
    sentiment_b: str
    positives_a: List[str] = []
    negatives_a: List[str] = []
    positives_b: List[str] = []
    negatives_b: List[str] = []
    company_a: Optional[str] = None
    company_b: Optional[str] = None


class BrandInsightRequest(BaseModel):
    name: str
    mode: Literal["company", "model"]
    sentiment: str
    positives: List[str] = []
    negatives: List[str] = []
    company: Optional[str] = None
    all_brands: List[dict] = []  # [{name, sentiment, mention_count, positives, negatives}]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    video_id: Optional[str] = None
    history: List[ChatMessage] = []
    stream: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "YouTube RAG API is running"}


@app.post("/api/process", response_model=ProcessResponse)
def process_video(req: ProcessRequest):
    """
    Process a YouTube OR TeamBHP URL:
    - YouTube: extracts transcript + comments, chunks, stores in vector DB.
    - TeamBHP: scrapes thread posts via Selenium, chunks (1 post = 1 chunk), stores.
    """
    if is_teambhp_url(req.url):
        return _process_teambhp(req)
    return _process_youtube(req)


def _process_teambhp(req: ProcessRequest) -> ProcessResponse:
    """Scrape a TeamBHP thread and index each post as one chunk."""
    print(f"[teambhp] Detected TeamBHP URL: {req.url}")

    result = scrape_teambhp_thread(req.url)

    if not result["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"TeamBHP scraping failed: {result.get('error', 'Unknown error')}",
        )

    thread_id = result["thread_id"]
    thread_title = result["thread_title"]
    posts = result["posts"]

    if not posts:
        raise HTTPException(status_code=422, detail="No posts found in the thread.")

    # Clear any previous chunks for this thread
    delete_video_chunks(thread_id)

    chunks = chunk_teambhp_posts(posts, thread_id=thread_id, thread_title=thread_title)
    stored = add_chunks(chunks)
    print(f"[teambhp] Stored {stored} post chunks for '{thread_title}'")

    return ProcessResponse(
        video_id=thread_id,
        title=thread_title,
        channel="TeamBHP",
        source_type="teambhp",
        post_chunks=stored,
        total_chunks=stored,
    )


def _process_youtube(req: ProcessRequest) -> ProcessResponse:
    """Extract transcript + comments from a YouTube URL and store in vector DB."""
    warnings = []

    # 1. Extract video ID
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Fetch metadata, transcript, and comments in parallel
    def _fetch_metadata():
        return get_video_metadata(video_id)

    def _fetch_transcript():
        if req.use_whisper:
            print("[transcript] Whisper forced by user")
            return get_transcript_whisper(video_id), "whisper"
        result = get_transcript(video_id)
        if not result["success"]:
            print(f"[transcript] Captions failed ({result.get('error')}), falling back to Whisper")
            return get_transcript_whisper(video_id), "whisper"
        return result, "captions"

    def _fetch_comments():
        if not req.include_comments:
            return None
        return get_comments_with_replies(video_id)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_meta       = pool.submit(_fetch_metadata)
        f_transcript = pool.submit(_fetch_transcript)
        f_comments   = pool.submit(_fetch_comments)

        metadata                          = f_meta.result()
        transcript_result, transcript_method = f_transcript.result()
        comments_result                   = f_comments.result()

    title   = metadata["title"]
    channel = metadata["channel"]

    # 3. Clear old chunks for this video
    delete_video_chunks(video_id)

    # ── Transcript ─────────────────────────────────────────────────────────────
    transcript_chunks_count = 0
    transcript_available = False

    if transcript_result["success"] and transcript_result["text"]:
        transcript_available = True
        print(f"[transcript] method={transcript_method}, {len(transcript_result['text'])} chars")
        t_chunks = semantic_chunk_transcript(
            transcript_result["text"],
            min_tokens=req.chunk_size,
            max_tokens=max(req.chunk_size, 2000),
            video_id=video_id,
            video_title=title,
        )
        transcript_chunks_count = add_chunks(t_chunks)
    else:
        warnings.append(f"Transcript: {transcript_result.get('error', 'Unavailable')}")

    # ── Comments ───────────────────────────────────────────────────────────────
    comment_chunks_count = 0
    comments_available = False

    if req.include_comments and comments_result is not None:
        threads = comments_result.get("comment_threads", [])
        print(f"[comments] success={comments_result['success']} "
              f"threads={len(threads)} "
              f"error={comments_result.get('error', '-')}")
        if comments_result["success"] and threads:
            comments_available = True
            c_chunks = chunk_comment_threads(
                threads,
                video_id=video_id,
                video_title=title,
            )
            print(f"[comments] produced {len(c_chunks)} chunks (1 per thread), storing...")
            comment_chunks_count = add_chunks(c_chunks)
            print(f"[comments] stored {comment_chunks_count} chunks")
        else:
            warnings.append(f"Comments: {comments_result.get('error', 'Unavailable')}")

    total = transcript_chunks_count + comment_chunks_count
    if total == 0:
        raise HTTPException(
            status_code=422,
            detail="No content could be extracted from this video. " + "; ".join(warnings),
        )

    return ProcessResponse(
        video_id=video_id,
        title=title,
        channel=channel,
        source_type="youtube",
        transcript_chunks=transcript_chunks_count,
        comment_chunks=comment_chunks_count,
        total_chunks=total,
        transcript_available=transcript_available,
        transcript_method=transcript_method if transcript_available else "",
        comments_available=comments_available,
        warnings=warnings,
    )


@app.post("/api/upload-audio", response_model=ProcessResponse)
async def upload_audio(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    chunk_size: int = Form(default=500),
):
    """
    Upload a local audio file (mp3, wav, m4a, ogg, flac, webm, etc.),
    transcribe it with Whisper, chunk and store in the vector DB.
    Returns the same shape as /api/process so the frontend needs no changes.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Stable ID derived from file content so re-uploading the same file is idempotent
    audio_id    = "audio_" + hashlib.md5(content).hexdigest()[:12]
    audio_title = title.strip() or os.path.splitext(file.filename or "audio")[0]

    # Write to a named temp file so ffmpeg / pydub can open it by path
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe_local_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not result["success"] or not result["text"]:
        raise HTTPException(
            status_code=422,
            detail=f"Transcription failed: {result.get('error', 'Unknown error')}",
        )

    delete_video_chunks(audio_id)

    chunks = semantic_chunk_transcript(
        result["text"],
        min_tokens=chunk_size,
        max_tokens=max(chunk_size, 2000),
        video_id=audio_id,
        video_title=audio_title,
    )
    stored = add_chunks(chunks)

    print(f"[upload-audio] '{audio_title}' → {stored} chunks stored (id={audio_id})")

    return ProcessResponse(
        video_id=audio_id,
        title=audio_title,
        channel="Local Audio",
        source_type="audio",
        transcript_chunks=stored,
        total_chunks=stored,
        transcript_available=True,
        transcript_method="whisper",
    )


def _enforce_guardrail(question: str) -> None:
    result = check_input(question)
    if result.blocked:
        raise HTTPException(status_code=400, detail=result.reason)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming chat endpoint."""
    _enforce_guardrail(req.question)
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = answer_question(req.question, video_id=req.video_id, chat_history=history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events streaming chat endpoint."""
    _enforce_guardrail(req.question)
    history = [{"role": m.role, "content": m.content} for m in req.history]

    def generate():
        try:
            for token in answer_question_stream(req.question, video_id=req.video_id, chat_history=history):
                yield token
        except Exception as exc:
            import traceback
            print(f"[stream] Error during generation: {exc}")
            traceback.print_exc()
            yield f"\n\n_Error during response generation: {exc}_"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/videos")
def list_videos():
    """List all videos stored in the vector DB."""
    videos = list_stored_videos()
    stats = get_collection_stats()
    return {"videos": videos, "total_chunks": stats["total_chunks"]}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    """Remove all chunks for a specific video."""
    deleted = delete_video_chunks(video_id)
    return {"deleted_chunks": deleted, "video_id": video_id}


@app.post("/api/analyze/{video_id}")
def analyze_video(video_id: str):
    """Run the LangGraph sentiment analysis + feature extraction pipeline for a video."""
    video_title = get_video_title(video_id)

    aggregated = run_analysis(video_id, video_title=video_title)
    return {"video_id": video_id, "video_title": video_title, "aggregated": aggregated}


@app.get("/api/analysis/{video_id}")
def get_analysis(video_id: str):
    """Return cached analysis for a video (404 if not yet run)."""
    cached = get_cached_analysis(video_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Analysis not found. Run /api/analyze first.")
    return cached


@app.post("/api/brand_dedup/{video_id}")
def brand_dedup(video_id: str):
    """Run post-processing brand/model name deduplication on a cached analysis."""
    try:
        aggregated = run_brand_dedup(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No cached analysis found. Run /api/analyze first.")
    return {"video_id": video_id, "aggregated": aggregated}


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """Stream LLM recommendations for a brand/company comparison."""
    mode_label = "Brand / Model" if req.mode == "model" else "Company"

    def fmt(items):
        return "\n".join(f"  • {i}" for i in items[:8]) if items else "  • None recorded"

    ctx_a = f"(under {req.company_a})" if req.company_a else ""
    ctx_b = f"(under {req.company_b})" if req.company_b else ""

    prompt = f"""You are an expert automotive market analyst. The data below comes from user sentiment analysis of YouTube videos and automotive forum discussions.

Comparison type: {mode_label}

--- {req.name_a.upper()} {ctx_a} ---
Overall sentiment: {req.sentiment_a}
Praised for:
{fmt(req.positives_a)}
Criticized for:
{fmt(req.negatives_a)}

--- {req.name_b.upper()} {ctx_b} ---
Overall sentiment: {req.sentiment_b}
Praised for:
{fmt(req.positives_b)}
Criticized for:
{fmt(req.negatives_b)}

Based solely on the user feedback data above, give concise, actionable recommendations. Structure your response as:

**{req.name_a} — What to improve:**
(focus on the criticized areas; be specific)

**{req.name_b} — What to improve:**
(focus on the criticized areas; be specific)

**Competitive edge:**
(which has the stronger position and why, based on the data)

**Key battleground:**
(one or two areas where both brands compete directly and what each must do to win)

Keep the entire response under 350 words. Be direct and evidence-based."""

    def generate():
        stream = _llm_client.chat.completions.create(
            model=_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=600,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/brand-insight")
def brand_insight(req: BrandInsightRequest):
    """Stream LLM single-brand analysis with LLM-identified competitors from the dataset."""
    import json as _json

    ctx = f"(under {req.company})" if req.company else ""
    mode_label = "Brand / Model" if req.mode == "model" else "Company"

    def fmt(items):
        return "\n".join(f"  • {i}" for i in items[:8]) if items else "  • None recorded"

    # ── Step 1: Ask LLM to identify real market competitors from the dataset ──
    competitor_data = []
    if req.all_brands:
        names_list = ", ".join(b.get("name", "") for b in req.all_brands if b.get("name"))
        identify_prompt = (
            f"You are an automotive market expert. The brand under analysis is: {req.name}.\n"
            f"From the following list, identify up to 4 brands that are GENUINE market competitors "
            f"(same vehicle segment, similar price range, overlapping target customers):\n\n"
            f"{names_list}\n\n"
            f"Return ONLY a JSON array of matching brand names exactly as they appear in the list. "
            f"Example: [\"Brand A\", \"Brand B\"]. If none are genuine competitors, return []."
        )
        try:
            id_resp = _llm_client.chat.completions.create(
                model=_CHAT_MODEL,
                messages=[{"role": "user", "content": identify_prompt}],
                stream=False,
                max_tokens=120,
            )
            raw = id_resp.choices[0].message.content.strip()
            # extract JSON array from response
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                competitor_names = set(_json.loads(raw[start:end + 1]))
                competitor_data = [
                    b for b in req.all_brands
                    if b.get("name") in competitor_names
                ]
        except Exception:
            pass  # fall through with no competitor data

    # ── Step 2: Build competitor block from identified brands ─────────────────
    def fmt_competitor_block(comps):
        if not comps:
            return "  • No competitor data identified from the dataset."
        lines = []
        for c in comps:
            name = c.get("name", "Unknown")
            sent = c.get("sentiment", "neutral")
            pos  = c.get("positives", [])
            neg  = c.get("negatives", [])
            pos_str = ", ".join(pos[:5]) if pos else "none recorded"
            neg_str = ", ".join(neg[:4]) if neg else "none recorded"
            lines.append(
                f"  • {name} (overall: {sent})\n"
                f"    Praised for: {pos_str}\n"
                f"    Criticized for: {neg_str}"
            )
        return "\n".join(lines)

    competitor_section = f"""
Competitors identified from the dataset:
{fmt_competitor_block(competitor_data)}
""" if competitor_data else ""

    # ── Step 3: Stream full analysis ──────────────────────────────────────────
    prompt = f"""You are an expert automotive market analyst. The data below comes from user sentiment analysis of YouTube videos and automotive forum discussions.

{mode_label}: {req.name.upper()} {ctx}
Overall sentiment: {req.sentiment}

Praised for:
{fmt(req.positives)}

Criticized for:
{fmt(req.negatives)}
{competitor_section}
Based solely on the user feedback data above, provide a concise analysis. Structure your response exactly as:

**Overall Strengths:**
(key areas where users are satisfied — be specific)

**Key Weaknesses:**
(main pain points from user criticism — be specific)

**Improvement Areas:**
(concrete, actionable recommendations based on the weaknesses)

**What Competitors Do Better:**
(for each identified competitor, name them and state the specific areas where their users praise them more than {req.name}'s users do — be direct and evidence-based)

Keep the entire response under 450 words. Be direct."""

    def generate():
        stream = _llm_client.chat.completions.create(
            model=_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=750,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
