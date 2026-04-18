"""
Hybrid retrieval: dense vector search (70 %) + BM25 keyword search (30 %)
followed by cross-encoder reranking via FlashRank.

Pipeline per query
──────────────────
1.  Dense   – top-N results from Qdrant (cosine similarity)
2.  BM25    – top-N results scored over all stored chunks for the video
3.  Fusion  – linear combination: 0.70 * norm(dense) + 0.30 * norm(bm25)
4.  Rerank  – cross-encoder (ms-marco-MiniLM-L-12-v2 via FlashRank)
"""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from vector_store import query_chunks, fetch_chunks_for_bm25

DENSE_WEIGHT: float = 0.70
BM25_WEIGHT: float = 0.30

# Lazy-loaded FlashRank cross-encoder
_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        # ms-marco-MiniLM-L-12-v2 (~25 MB, downloaded on first use)
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _minmax_normalize(scores: List[float]) -> List[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ── BM25 ───────────────────────────────────────────────────────────────────────

def _bm25_search(query: str, corpus: List[dict], n_results: int) -> dict[str, float]:
    """
    Build a BM25 index over *corpus* and return a mapping of
    chunk-text → normalised BM25 score for the top-n_results hits.
    """
    tokenized = [_tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    raw = list(bm25.get_scores(_tokenize(query)))

    norm = _minmax_normalize(raw)

    # Collect top-n_results by raw score (normalised values keep the same order)
    indexed = sorted(enumerate(raw), key=lambda x: x[1], reverse=True)[:n_results]
    return {corpus[i]["text"]: norm[i] for i, _ in indexed}


# ── Fusion ─────────────────────────────────────────────────────────────────────

def _fuse(
    dense_results: List[dict],
    dense_norm: List[float],
    bm25_scores: dict[str, float],
    corpus_by_text: dict[str, dict],
    candidate_pool: int,
) -> List[dict]:
    """
    Merge dense and BM25 candidates into a single ranked list using
    weighted linear combination.
    """
    fused: dict[str, dict] = {}

    # Dense candidates
    for chunk, d_score in zip(dense_results, dense_norm):
        key = chunk["text"]
        b_score = bm25_scores.get(key, 0.0)
        fused[key] = {
            **chunk,
            "dense_score": round(d_score, 4),
            "bm25_score": round(b_score, 4),
            "hybrid_score": DENSE_WEIGHT * d_score + BM25_WEIGHT * b_score,
        }

    # Pure BM25 hits not already in dense results
    for text, b_score in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True):
        if text not in fused and text in corpus_by_text:
            chunk = corpus_by_text[text]
            fused[text] = {
                **chunk,
                "dense_score": 0.0,
                "bm25_score": round(b_score, 4),
                "hybrid_score": BM25_WEIGHT * b_score,
            }

    candidates = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:candidate_pool]


# ── Reranker ───────────────────────────────────────────────────────────────────

def _rerank(query: str, chunks: List[dict], top_n: int) -> List[dict]:
    """
    Rerank *chunks* with a cross-encoder and return the top_n results.
    Falls back to hybrid-score ordering if FlashRank is unavailable.
    """
    try:
        from flashrank import RerankRequest

        ranker = _get_ranker()
        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
        hits = ranker.rerank(RerankRequest(query=query, passages=passages))

        reranked = []
        for hit in hits[:top_n]:
            # Passage is a dataclass — use attribute access, not dict access
            hit_id = hit.id if hasattr(hit, "id") else hit["id"]
            hit_score = hit.score if hasattr(hit, "score") else hit["score"]
            chunk = chunks[hit_id]
            reranked.append({**chunk, "score": round(float(hit_score), 4)})
        return reranked

    except Exception as exc:
        print(f"[rerank] FlashRank failed ({exc}), using hybrid scores")
        for c in chunks:
            c["score"] = round(c.get("hybrid_score", c.get("score", 0.0)), 4)
        return chunks[:top_n]


# ── Public API ─────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    video_id: Optional[str] = None,
    n_results: int = 6,
) -> List[dict]:
    """
    Hybrid retrieval pipeline.

    1. Dense search  – Qdrant cosine similarity (70 % weight)
    2. BM25 search   – keyword scoring over all stored chunks (30 % weight)
    3. Score fusion  – linear combination of normalised scores
    4. Reranking     – cross-encoder (FlashRank) for final ordering

    Returns a list of chunk dicts (same shape as query_chunks) with an
    extra 'dense_score', 'bm25_score', and 'hybrid_score' field for
    debugging / observability.
    """
    candidate_pool = n_results * 4  # over-retrieve before reranking

    # ── 1. Dense retrieval ─────────────────────────────────────────────────────
    dense_results = query_chunks(query, video_id=video_id, n_results=candidate_pool)
    if not dense_results:
        return []

    # ── 2. BM25 over all stored chunks ─────────────────────────────────────────
    corpus = fetch_chunks_for_bm25(video_id=video_id)
    corpus_by_text: dict[str, dict] = {c["text"]: c for c in corpus}

    bm25_scores: dict[str, float] = {}
    if corpus:
        bm25_scores = _bm25_search(query, corpus, n_results=candidate_pool)

    # ── 3. Normalise dense scores and fuse ─────────────────────────────────────
    dense_norm = _minmax_normalize([c["score"] for c in dense_results])
    candidates = _fuse(dense_results, dense_norm, bm25_scores, corpus_by_text, candidate_pool)

    print(
        f"[hybrid] dense={len(dense_results)}  bm25_hits={len(bm25_scores)}  "
        f"candidates={len(candidates)}  reranking to top {n_results}"
    )

    # ── 4. Rerank ──────────────────────────────────────────────────────────────
    return _rerank(query, candidates, top_n=n_results)
