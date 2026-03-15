import os
from typing import Iterator, List
from openai import OpenAI
from vector_store import query_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 3000

SYSTEM_PROMPT = """You are an expert assistant that answers questions based on YouTube video content.
You have access to both the video transcript and viewer comments.

Guidelines:
- Answer questions using ONLY the provided context (transcript + comments).
- If the context doesn't contain enough information, say so clearly.
- When referencing comments, note it's from viewer perspective.
- Be concise, accurate, and helpful.
- If asked about opinions, summarize what commenters said.
- Do NOT include any citations, source references, footnotes, or bracketed numbers like [1] or [2] in your response."""


def build_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context string."""
    transcript_chunks = [c for c in chunks if c["metadata"].get("source") == "transcript"]
    comment_chunks = [c for c in chunks if c["metadata"].get("source") == "comments"]

    parts = []

    if transcript_chunks:
        parts.append("=== VIDEO TRANSCRIPT (relevant sections) ===")
        for chunk in transcript_chunks:
            parts.append(chunk["text"])

    if comment_chunks:
        parts.append("\n=== VIEWER COMMENTS (relevant sections) ===")
        for chunk in comment_chunks:
            parts.append(chunk["text"])

    return "\n\n".join(parts)


def answer_question(
    question: str,
    video_id: str = None,
    chat_history: List[dict] = None,
) -> dict:
    """
    Non-streaming: retrieve relevant chunks, generate answer.
    Returns dict with answer and source chunks.
    """
    chunks = query_chunks(question, video_id=video_id, n_results=6)

    if not chunks:
        return {
            "answer": "No content has been processed yet. Please add a YouTube video first.",
            "sources": [],
        }

    context = build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:  # last 3 turns
            messages.append(msg)

    user_message = f"Context from the video:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    print("\n" + "="*60)
    print("[PROMPT] System:")
    print(SYSTEM_PROMPT)
    print("\n[PROMPT] User message:")
    print(user_message)
    print("="*60 + "\n")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
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
            for c in chunks[:4]
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

    chunks = query_chunks(question, video_id=video_id, n_results=6)

    if not chunks:
        yield "data: No content has been processed yet. Please add a YouTube video first."
        return

    context = build_context(chunks)

    sources = [
        {
            "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            "source": c["metadata"].get("source", "unknown"),
            "score": c["score"],
        }
        for c in chunks[:4]
    ]

    # First, yield sources metadata
    yield f"__SOURCES__{json.dumps(sources)}\n"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    user_message = f"Context from the video:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    print("\n" + "="*60)
    print("[PROMPT] System:")
    print(SYSTEM_PROMPT)
    print("\n[PROMPT] User message:")
    print(user_message)
    print("="*60 + "\n")

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
