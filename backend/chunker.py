import re
import os
import math
from typing import List

import tiktoken
from openai import AzureOpenAI

_enc = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _batch_embed(texts: List[str]) -> List[List[float]]:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01"),
    )
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries; keep non-empty sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _make_chunk(sentences: List[str], index: int, source: str,
                video_id: str, video_title: str) -> dict:
    return {
        "text": " ".join(sentences),
        "metadata": {
            "source": source,
            "video_id": video_id,
            "video_title": video_title,
            "chunk_index": index,
        },
    }


# ── Semantic transcript chunking ───────────────────────────────────────────────

def semantic_chunk_transcript(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 2000,
    similarity_threshold: float = 0.5,
    video_id: str = "",
    video_title: str = "",
) -> List[dict]:
    """
    Chunk a transcript using cosine similarity between consecutive sentences.

    Algorithm:
    1. Split into sentences.
    2. Batch-embed all sentences in one OpenAI API call.
    3. Walk sentence-by-sentence:
       - If adding the next sentence would exceed max_tokens → force a new chunk.
       - Else if similarity with the next sentence < threshold AND current chunk
         already has >= min_tokens → start a new chunk (semantic boundary).
       - Otherwise keep accumulating.
    4. If the final chunk is below min_tokens, merge it into the previous one.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    print(f"[chunker] {len(sentences)} sentences — embedding for semantic split...")
    embeddings = _batch_embed(sentences)
    token_counts = [_count_tokens(s) for s in sentences]
    print(f"[chunker] building semantic chunks "
          f"(min={min_tokens} tok, max={max_tokens} tok, threshold={similarity_threshold})")

    chunks: List[dict] = []
    current_sents: List[str] = [sentences[0]]
    current_tokens: int = token_counts[0]
    chunk_idx: int = 0

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        next_tok = token_counts[i]

        force_break = (current_tokens + next_tok) > max_tokens
        semantic_break = (sim < similarity_threshold) and (current_tokens >= min_tokens)

        if force_break or semantic_break:
            chunks.append(_make_chunk(current_sents, chunk_idx, "transcript", video_id, video_title))
            chunk_idx += 1
            current_sents = [sentences[i]]
            current_tokens = next_tok
        else:
            current_sents.append(sentences[i])
            current_tokens += next_tok

    # Handle the last batch
    if current_sents:
        if current_tokens < min_tokens and chunks:
            # Try merging into the previous chunk
            candidate = chunks[-1]["text"] + " " + " ".join(current_sents)
            if _count_tokens(candidate) <= max_tokens:
                chunks[-1]["text"] = candidate
            else:
                chunks.append(_make_chunk(current_sents, chunk_idx, "transcript", video_id, video_title))
        else:
            chunks.append(_make_chunk(current_sents, chunk_idx, "transcript", video_id, video_title))

    print(f"[chunker] → {len(chunks)} semantic chunks")
    return chunks


# ── TeamBHP post chunking ──────────────────────────────────────────────────────

def chunk_teambhp_posts(
    posts: List[dict],
    thread_id: str = "",
    thread_title: str = "",
) -> List[dict]:
    """
    Each post scraped from a TeamBHP thread becomes ONE chunk.
    Metadata mirrors the YouTube chunk schema so the rest of the pipeline
    (vector store, RAG engine, analysis) works without modification.
    """
    chunks: List[dict] = []

    for idx, post in enumerate(posts):
        text = post.get("text", "").strip()
        if not text:
            continue

        author = post.get("author", "Unknown")
        date = post.get("date", "")
        post_id = post.get("post_id", "")

        header = f"[Post by {author}"
        if date:
            header += f" | {date}"
        header += f"]:\n{text}"

        chunks.append(
            {
                "text": header,
                "metadata": {
                    "source": "teambhp",
                    "video_id": thread_id,        # reuses video_id key for pipeline compat
                    "video_title": thread_title,
                    "chunk_index": idx,
                    "post_id": post_id,
                    "author": author,
                },
            }
        )

    return chunks


# ── Comment-thread chunking ────────────────────────────────────────────────────

def chunk_comment_threads(
    threads: List[dict],
    video_id: str = "",
    video_title: str = "",
) -> List[dict]:
    """
    Each comment thread (top-level comment + all its replies) becomes ONE chunk.
    Preserves the conversational order: comment first, then replies in order.
    """
    chunks: List[dict] = []

    for idx, thread in enumerate(threads):
        comment_text = thread.get("text", "").strip()
        if not comment_text:
            continue

        lines = [
            f"[Comment by {thread.get('author', 'User')} "
            f"| 👍 {thread.get('likes', 0)}]:\n{comment_text}"
        ]

        for reply in thread.get("replies", []):
            reply_text = reply.get("text", "").strip()
            if reply_text:
                lines.append(
                    f"  ↳ [Reply by {reply.get('author', 'User')} "
                    f"| 👍 {reply.get('likes', 0)}]: {reply_text}"
                )

        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "source": "comments",
                "video_id": video_id,
                "video_title": video_title,
                "chunk_index": idx,
            },
        })

    return chunks
