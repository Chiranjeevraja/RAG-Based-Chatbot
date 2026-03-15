import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from youtube_extractor import extract_video_id, get_transcript, get_transcript_whisper, get_comments_with_replies, get_video_metadata
from chunker import semantic_chunk_transcript, chunk_comment_threads
from vector_store import add_chunks, delete_video_chunks, list_stored_videos, get_collection_stats
from rag_engine import answer_question, answer_question_stream
from analysis_pipeline import run_analysis, get_cached_analysis

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
    url: str
    include_comments: bool = True
    use_whisper: bool = False   # force Whisper even if captions exist
    chunk_size: int = 500
    chunk_overlap: int = 100


class ProcessResponse(BaseModel):
    video_id: str
    title: str
    channel: str
    transcript_chunks: int
    comment_chunks: int
    total_chunks: int
    transcript_available: bool
    transcript_method: str   # "captions" | "whisper"
    comments_available: bool
    warnings: List[str] = []


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    video_id: Optional[str] = None
    history: List[ChatMessage] = []
    stream: bool = False


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
    Extract transcript + comments from a YouTube URL,
    chunk them, and store in the vector database.
    """
    warnings = []

    # 1. Extract video ID
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2 & 4 & 5 — fetch metadata, transcript, and comments in parallel
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
        f_meta      = pool.submit(_fetch_metadata)
        f_transcript = pool.submit(_fetch_transcript)
        f_comments  = pool.submit(_fetch_comments)

        metadata         = f_meta.result()
        transcript_result, transcript_method = f_transcript.result()
        comments_result  = f_comments.result()

    title   = metadata["title"]
    channel = metadata["channel"]

    # 3. Clear old chunks now that we have the video id confirmed
    delete_video_chunks(video_id)

    # ── Handle transcript ──────────────────────────────────────────────────────
    transcript_chunks_count = 0
    transcript_available = False

    if transcript_result["success"] and transcript_result["text"]:
        transcript_available = True
        print(f"[transcript] method={transcript_method}, {len(transcript_result['text'])} chars")
        t_chunks = semantic_chunk_transcript(
            transcript_result["text"],
            video_id=video_id,
            video_title=title,
        )
        transcript_chunks_count = add_chunks(t_chunks)
    else:
        warnings.append(f"Transcript: {transcript_result.get('error', 'Unavailable')}")

    # ── Handle comments ────────────────────────────────────────────────────────
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
        transcript_chunks=transcript_chunks_count,
        comment_chunks=comment_chunks_count,
        total_chunks=total,
        transcript_available=transcript_available,
        transcript_method=transcript_method,
        comments_available=comments_available,
        warnings=warnings,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming chat endpoint."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = answer_question(req.question, video_id=req.video_id, chat_history=history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events streaming chat endpoint."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in req.history]

    def generate():
        for token in answer_question_stream(req.question, video_id=req.video_id, chat_history=history):
            yield token

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
    # Resolve the video title from stored videos
    videos = list_stored_videos()
    video_title = ""
    for v in videos:
        if v["video_id"] == video_id:
            video_title = v["title"]
            break

    aggregated = run_analysis(video_id, video_title=video_title)
    return {"video_id": video_id, "video_title": video_title, "aggregated": aggregated}


@app.get("/api/analysis/{video_id}")
def get_analysis(video_id: str):
    """Return cached analysis for a video (404 if not yet run)."""
    cached = get_cached_analysis(video_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Analysis not found. Run /api/analyze first.")
    return cached


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
