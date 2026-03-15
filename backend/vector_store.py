import os
import uuid
from typing import List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PayloadSchemaType,
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimensions for text-embedding-3-small
COLLECTION_NAME = "youtube_rag"

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_qdrant: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        _qdrant = QdrantClient(url=url, api_key=api_key)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    # Create keyword index on video_id so filtering works.
    # This is idempotent — safe to call even if the index already exists.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="video_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _embed_texts(texts: List[str]) -> List[List[float]]:
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def add_chunks(chunks: List[dict]) -> int:
    """Embed and upsert chunks into Qdrant. Returns number of chunks added."""
    if not chunks:
        return 0

    client = _get_client()
    texts = [c["text"] for c in chunks]
    embeddings = _embed_texts(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text, **chunk["metadata"]},
        )
        for text, embedding, chunk in zip(texts, embeddings, chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def query_chunks(query: str, video_id: str = None, n_results: int = 6) -> List[dict]:
    """Find the most relevant chunks for a query."""
    client = _get_client()

    query_embedding = _embed_texts([query])[0]

    search_filter = (
        Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )
        if video_id
        else None
    )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=n_results,
        query_filter=search_filter,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": round(hit.score, 4),
        }
        for hit in results
    ]


def delete_video_chunks(video_id: str) -> int:
    """Delete all chunks for a given video. Returns approximate count deleted."""
    client = _get_client()

    # Count before deletion
    count_result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
    )
    count = count_result.count

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            )
        ),
    )
    return count


def list_stored_videos() -> List[dict]:
    """Return a deduplicated list of stored video IDs and titles."""
    client = _get_client()

    seen = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["video_id", "video_title"],
        )
        for record in records:
            vid = record.payload.get("video_id", "")
            if vid and vid not in seen:
                seen[vid] = record.payload.get("video_title", vid)

        if next_offset is None:
            break
        offset = next_offset

    return [{"video_id": k, "title": v} for k, v in seen.items()]


def get_collection_stats() -> dict:
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    return {"total_chunks": info.points_count}
