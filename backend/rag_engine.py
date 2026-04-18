import os
from typing import Iterator, List
import tiktoken
from openai import AzureOpenAI
from retriever import hybrid_search

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _select_chunks_by_budget(chunks: List[dict], max_tokens: int) -> List[dict]:
    """
    Greedily select chunks in relevance order until the token budget is full.
    Chunks are already sorted by reranker score (highest first).
    """
    selected = []
    total = 0
    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk["text"])
        if total + chunk_tokens > max_tokens:
            break
        selected.append(chunk)
        total += chunk_tokens
    print(f"[budget] {len(selected)}/{len(chunks)} chunks selected — {total} / {max_tokens} tokens used")
    return selected


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION", "2024-12-01-preview"),
)

CHAT_MODEL = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-ria-dev-01")
MAX_CONTEXT_TOKENS = 40000
RETRIEVE_N = 1000  # large candidate pool; token budget trims this down

SYSTEM_PROMPT = """You are an expert assistant that answers questions based on indexed content, which may include YouTube video transcripts, viewer comments, or TeamBHP forum posts.

Guidelines:
- Answer questions using ONLY the provided context.
- If the context doesn't contain enough information, say so clearly.
- When referencing comments or forum posts, note it's from a user/community perspective.
- Be concise, accurate, and helpful.
- If asked about opinions, summarize what users said.
- Do NOT include any citations, source references, footnotes, or bracketed numbers like [1] or [2] in your response."""


def build_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context string."""
    transcript_chunks = [c for c in chunks if c["metadata"].get("source") == "transcript"]
    comment_chunks    = [c for c in chunks if c["metadata"].get("source") == "comments"]
    teambhp_chunks    = [c for c in chunks if c["metadata"].get("source") == "teambhp"]

    parts = []

    if transcript_chunks:
        parts.append("=== VIDEO TRANSCRIPT (relevant sections) ===")
        for chunk in transcript_chunks:
            parts.append(chunk["text"])

    if comment_chunks:
        parts.append("=== VIEWER COMMENTS (relevant sections) ===")
        for chunk in comment_chunks:
            parts.append(chunk["text"])

    if teambhp_chunks:
        parts.append("=== TEAM-BHP FORUM POSTS (relevant sections) ===")
        for chunk in teambhp_chunks:
            parts.append(chunk["text"])

    return "\n\n".join(parts)


def answer_question(
    question: str,
    video_id: str = None,
    chat_history: List[dict] = None,
) -> dict:
    """Non-streaming: retrieve relevant chunks, generate answer."""
    chunks = hybrid_search(question, video_id=video_id, n_results=RETRIEVE_N)

    if not chunks:
        return {
            "answer": "No content has been processed yet. Please add a YouTube video or TeamBHP thread first.",
            "sources": [],
        }

    chunks = _select_chunks_by_budget(chunks, MAX_CONTEXT_TOKENS)
    context = build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    print("\n" + "="*60)
    print("[PROMPT] System:")
    print(SYSTEM_PROMPT)
    print("\n[PROMPT] User message (truncated):")
    print(user_message[:500], "...")
    print("="*60 + "\n")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=4000,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "source": c["metadata"].get("source", "unknown"),
                "score": c["score"],
            }
            for c in chunks
        ],
    }


def answer_question_stream(
    question: str,
    video_id: str = None,
    chat_history: List[dict] = None,
) -> Iterator[str]:
    """
    Streaming version: yields tokens as they arrive.
    First yields JSON metadata line with sources, then tokens.
    """
    import json

    chunks = hybrid_search(question, video_id=video_id, n_results=RETRIEVE_N)

    if not chunks:
        yield "No content has been processed yet. Please add a YouTube video or TeamBHP thread first."
        return

    chunks = _select_chunks_by_budget(chunks, MAX_CONTEXT_TOKENS)
    context = build_context(chunks)

    sources = [
        {
            "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            "source": c["metadata"].get("source", "unknown"),
            "score": c["score"],
        }
        for c in chunks
    ]

    yield f"__SOURCES__{json.dumps(sources)}\n"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    print("\n" + "="*60)
    print("[PROMPT] System:")
    print(SYSTEM_PROMPT)
    print("\n[PROMPT] User message (truncated):")
    print(user_message[:500], "...")
    print("="*60 + "\n")

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=4000,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
