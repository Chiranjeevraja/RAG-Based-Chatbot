import os
import json
import pathlib
from typing import List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.graph import StateGraph, END

from vector_store import _get_client, COLLECTION_NAME

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path(__file__).parent / "analysis_cache"
CACHE_DIR.mkdir(exist_ok=True)

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_WORKERS = 10

# ── Step 1 prompt: EXTRACT brand → model → features (no sentiment) ─────────────

EXTRACT_PROMPT = """\
Extract car companies/manufacturers, their specific models, and associated features from this YouTube video text.

Return ONLY valid JSON in this format:
{{
  "brands": [
    {{
      "brand": "CompanyName",
      "models": [
        {{
          "name": "ModelName",
          "features": ["feature one", "feature two"]
        }}
      ]
    }}
  ]
}}

Rules:
- "brand" is the manufacturer/company name (e.g., Toyota, Honda, BMW).
- Each entry in "models" is a specific car model with its own feature list.
- If features apply to the brand generally (not a specific model), use model name "General".
- Features should be specific (e.g., "fuel efficiency", "safety rating", "infotainment system").
- If no car brands are mentioned, return {{"brands": []}}.
- Return ONLY the JSON object.

Text:
{text}"""

# ── Step 2 prompt: DEDUPLICATE brand names via LLM ────────────────────────────

DEDUP_PROMPT = """\
You are given a list of car company/brand names extracted from YouTube video text.
Some entries may refer to the same company under different spellings, abbreviations, capitalizations, or naming conventions.

Return a JSON object mapping EVERY input name to its single canonical (most widely recognized) form:
{{
  "input_name_1": "Canonical Name",
  "input_name_2": "Canonical Name"
}}

Input names (one per line):
{names}

Rules:
- Every input name must appear as a key.
- Use the most official/recognized brand name (e.g., "BMW" not "B.M.W.", "Mercedes-Benz" not "Mercedes Benz").
- Merge obvious variants: different capitalizations, abbreviations, partial names referring to the same company.
- If a name is already canonical, map it to itself.
- Return ONLY the JSON object."""

# ── Step 3 prompt: SENTIMENT per company+model ──────────────────────────────────

SENTIMENT_PROMPT = """\
You are analyzing viewer/creator sentiment in YouTube video content about the "{model}" by "{brand}".

Brand (Company): {brand}
Model: {model}
Features to evaluate: {features}

Text excerpts from the video:
---
{text}
---

For EACH feature listed, determine the sentiment expressed in the text.
Also determine the overall sentiment towards the {brand} {model}.

Return ONLY valid JSON:
{{
  "overall_sentiment": "positive",
  "overall_score": 0.8,
  "positives": ["brief positive point about {brand} {model}"],
  "negatives": ["brief negative point about {brand} {model}"],
  "features": [
    {{"name": "feature name", "sentiment": "positive", "score": 0.85}}
  ]
}}

Rules:
- overall_score / score: 0.0=very negative, 0.5=neutral, 1.0=very positive
- positives / negatives: max 3 short phrases
- features: include all features from the list above, set score=0.5/neutral if not enough info
- Return ONLY the JSON object."""


# ── State ──────────────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    video_id: str
    video_title: str
    chunks: List[dict]                # raw chunks from Qdrant
    extractions: List[dict]           # per-chunk: brand → model → features
    brand_mapping: dict               # raw_name → canonical_name (from LLM dedup)
    company_model_sentiments: dict    # { company: { model: sentiment_result } }
    aggregated: dict                  # final output


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Normalize brand/model name for deduplication (title case)."""
    return " ".join(w.capitalize() for w in name.strip().split()) if name.strip() else ""


# ── Node 1: Load chunks ────────────────────────────────────────────────────────

def load_chunks_node(state: AnalysisState) -> AnalysisState:
    qdrant = _get_client()
    scroll_filter = Filter(
        must=[FieldCondition(key="video_id", match=MatchValue(value=state["video_id"]))]
    )
    all_records, offset = [], None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        all_records.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    chunks = sorted([
        {
            "id": str(r.id),
            "text": r.payload.get("text", ""),
            "source": r.payload.get("source", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
        }
        for r in all_records
    ], key=lambda c: (c["source"], c["chunk_index"]))

    print(f"[load_chunks] {len(chunks)} chunks loaded")
    return {**state, "chunks": chunks}


# ── Node 2: Extract brand → model → features (no sentiment) ───────────────────

def _extract_one(chunk: dict) -> dict:
    text = chunk["text"].strip()
    if not text:
        return {"chunk_id": chunk["id"], "source": chunk["source"], "text": text, "brands": []}
    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:2000])}],
            temperature=0,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"[extract] Error on chunk {chunk['id']}: {exc}")
        parsed = {"brands": []}

    brands = []
    for b in parsed.get("brands", []) or []:
        if not isinstance(b, dict) or not b.get("brand"):
            continue
        company = _normalize(b["brand"])
        if not company:
            continue
        models = []
        for m in b.get("models", []) or []:
            if not isinstance(m, dict) or not m.get("name"):
                continue
            model_name = _normalize(m["name"]) or "General"
            features = [f.strip().lower() for f in m.get("features", []) or [] if f]
            models.append({"name": model_name, "features": features})
        if models:
            brands.append({"brand": company, "models": models})

    return {"chunk_id": chunk["id"], "source": chunk["source"], "text": text, "brands": brands}


def extract_info_node(state: AnalysisState) -> AnalysisState:
    """Step 1 — extract brand → model → features from every chunk in parallel."""
    chunks = state["chunks"]
    total = len(chunks)
    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_extract_one, c): c for c in chunks}
        for future in as_completed(future_map):
            ext = future.result()
            results[ext["chunk_id"]] = ext
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"[extract] {completed}/{total} chunks done")

    extractions = [results[c["id"]] for c in chunks if c["id"] in results]
    return {**state, "extractions": extractions}


# ── Node 3: Deduplicate brand names via LLM ───────────────────────────────────

def deduplicate_brands_node(state: AnalysisState) -> AnalysisState:
    """
    Collect all unique brand names across all extractions → single LLM call
    to produce a canonical mapping → remap every extraction in place.
    """
    all_brands: set[str] = set()
    for ext in state["extractions"]:
        for b in ext["brands"]:
            all_brands.add(b["brand"])

    if not all_brands:
        return {**state, "brand_mapping": {}}

    names_str = "\n".join(sorted(all_brands))
    print(f"[dedup] {len(all_brands)} raw brand names: {sorted(all_brands)}")

    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": DEDUP_PROMPT.format(names=names_str)}],
            temperature=0,
        )
        mapping: dict = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"[dedup] LLM error: {exc} — skipping deduplication")
        mapping = {b: b for b in all_brands}

    # Ensure every raw name has a mapping (fallback to itself)
    for b in all_brands:
        if b not in mapping:
            mapping[b] = b

    canonical_set = set(mapping.values())
    print(f"[dedup] {len(canonical_set)} canonical names: {sorted(canonical_set)}")
    print(f"[dedup] mapping: {mapping}")

    # Apply mapping to all extractions
    new_extractions = []
    for ext in state["extractions"]:
        new_brands = [
            {**b, "brand": mapping.get(b["brand"], b["brand"])}
            for b in ext["brands"]
        ]
        new_extractions.append({**ext, "brands": new_brands})

    return {**state, "extractions": new_extractions, "brand_mapping": mapping}


# ── Node 5: Sentiment per company→model (using all collected context) ──────────

def _build_company_model_index(extractions: List[dict]) -> dict:
    """
    Group extractions by company → model:
    { company: { model: { texts: [], features: set(), chunk_ids: [] } } }
    Deduplication is handled by _normalize() in _extract_one.
    """
    index: dict = {}
    for ext in extractions:
        for b in ext["brands"]:
            company = b["brand"]
            if company not in index:
                index[company] = {}
            for m in b["models"]:
                model_name = m["name"]
                if model_name not in index[company]:
                    index[company][model_name] = {"texts": [], "features": set(), "chunk_ids": []}
                entry = index[company][model_name]
                if ext["text"]:
                    entry["texts"].append(ext["text"])
                entry["features"].update(m["features"])
                entry["chunk_ids"].append(ext["chunk_id"])
    return index


def _sentiment_one(company: str, model: str, entry: dict) -> tuple[str, str, dict]:
    """Run sentiment analysis for one company+model using all its collected text."""
    combined_text = "\n---\n".join(entry["texts"])[:4000]
    features_str = ", ".join(sorted(entry["features"])) or "N/A"

    try:
        resp = _openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": SENTIMENT_PROMPT.format(
                brand=company, model=model,
                features=features_str, text=combined_text,
            )}],
            temperature=0,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"[sentiment] Error for '{company}/{model}': {exc}")
        parsed = {}

    osent = parsed.get("overall_sentiment", "neutral")
    if osent not in ("positive", "negative", "neutral"):
        osent = "neutral"
    try:
        oscore = max(0.0, min(1.0, float(parsed.get("overall_score", 0.5))))
    except (TypeError, ValueError):
        oscore = 0.5

    features = []
    for f in parsed.get("features", []) or []:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        fsent = f.get("sentiment", "neutral")
        if fsent not in ("positive", "negative", "neutral"):
            fsent = "neutral"
        try:
            fscore = max(0.0, min(1.0, float(f.get("score", 0.5))))
        except (TypeError, ValueError):
            fscore = 0.5
        features.append({"name": f["name"].strip().lower(), "sentiment": fsent, "score": fscore})

    def _uniq(items, n=3):
        seen, out = set(), []
        for item in (items or []):
            item = item.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower()); out.append(item)
            if len(out) >= n: break
        return out

    return company, model, {
        "overall_sentiment": osent,
        "overall_score":     oscore,
        "mention_count":     len(entry["texts"]),
        "features":          features,
        "top_positives":     _uniq(parsed.get("positives", []), 3),
        "top_negatives":     _uniq(parsed.get("negatives", []), 3),
    }


def analyze_sentiments_node(state: AnalysisState) -> AnalysisState:
    """Step 2 — run sentiment per company+model pair in parallel."""
    company_model_index = _build_company_model_index(state["extractions"])
    pairs = [
        (company, model, entry)
        for company, models in company_model_index.items()
        for model, entry in models.items()
    ]
    print(f"[sentiment] {len(pairs)} company+model pairs: "
          f"{[(c, m) for c, m, _ in pairs]}")

    company_model_sentiments: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_sentiment_one, company, model, entry): (company, model)
            for company, model, entry in pairs
        }
        for future in as_completed(futures):
            try:
                company, model, result = future.result()
                if company not in company_model_sentiments:
                    company_model_sentiments[company] = {}
                company_model_sentiments[company][model] = result
                print(f"[sentiment] ✓ {company} / {model} → "
                      f"{result['overall_sentiment']} ({result['overall_score']:.2f})")
            except Exception as exc:
                print(f"[sentiment] Failed: {exc}")

    return {**state, "company_model_sentiments": company_model_sentiments}


# ── Node 6: Update Qdrant metadata ────────────────────────────────────────────

def update_metadata_node(state: AnalysisState) -> AnalysisState:
    qdrant = _get_client()

    def _update(ext: dict):
        companies = [b["brand"] for b in ext["brands"]]
        models    = [m["name"] for b in ext["brands"] for m in b["models"]]
        features  = [f for b in ext["brands"] for m in b["models"] for f in m["features"]]
        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "brands_mentioned":   companies,
                "models_mentioned":   models,
                "features_mentioned": features,
            },
            points=[ext["chunk_id"]],
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_update, ext): ext["chunk_id"] for ext in state["extractions"]}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[update_metadata] Failed: {exc}")

    print(f"[update_metadata] Updated {len(state['extractions'])} chunks")
    return state


# ── Node 7: Aggregate ──────────────────────────────────────────────────────────

def aggregate_results_node(state: AnalysisState) -> AnalysisState:
    cms = state["company_model_sentiments"]

    # Build brand_analysis: company → { overall, models: { model → result } }
    brand_analysis = {}
    all_company_scores = []

    for company, model_results in cms.items():
        model_scores = [r["overall_score"] for r in model_results.values()]
        company_score = round(sum(model_scores) / len(model_scores), 4) if model_scores else 0.5
        company_sentiment = (
            "positive" if company_score >= 0.6 else
            "negative" if company_score <= 0.4 else "neutral"
        )
        total_mentions = sum(r["mention_count"] for r in model_results.values())
        all_company_scores.append(company_score)

        brand_analysis[company] = {
            "overall_sentiment": company_sentiment,
            "overall_score":     company_score,
            "mention_count":     total_mentions,
            "models": dict(
                sorted(model_results.items(), key=lambda x: -x[1]["mention_count"])
            ),
        }

    # Sort companies by mention count
    brand_analysis = dict(
        sorted(brand_analysis.items(), key=lambda x: -x[1]["mention_count"])
    )

    # Overall sentiment = average of company scores
    overall_score = round(sum(all_company_scores) / len(all_company_scores), 4) if all_company_scores else 0.5
    overall_sentiment = (
        "positive" if overall_score >= 0.6 else
        "negative" if overall_score <= 0.4 else "neutral"
    )

    # Sentiment distribution across companies
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for v in brand_analysis.values():
        sent_counts[v["overall_sentiment"]] = sent_counts.get(v["overall_sentiment"], 0) + 1
    total_companies = max(len(brand_analysis), 1)
    sentiment_distribution = {k: round(v / total_companies * 100, 1) for k, v in sent_counts.items()}

    aggregated = {
        "overall_sentiment":      overall_sentiment,
        "overall_score":          overall_score,
        "total_chunks_analyzed":  len(state["extractions"]),
        "total_brands":           len(brand_analysis),
        "sentiment_distribution": sentiment_distribution,
        "brand_analysis":         brand_analysis,
    }
    return {**state, "aggregated": aggregated}


# ── Node 8: Save ──────────────────────────────────────────────────────────────

def save_results_node(state: AnalysisState) -> AnalysisState:
    cache_path = CACHE_DIR / f"{state['video_id']}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id":    state["video_id"],
            "video_title": state["video_title"],
            "aggregated":  state["aggregated"],
        }, f, ensure_ascii=False, indent=2)
    print(f"[save_results] Saved to {cache_path}")
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(AnalysisState)
    for name, fn in [
        ("load_chunks",         load_chunks_node),
        ("extract_info",        extract_info_node),
        ("deduplicate_brands",  deduplicate_brands_node),
        ("analyze_sentiments",  analyze_sentiments_node),
        ("update_metadata",     update_metadata_node),
        ("aggregate_results",   aggregate_results_node),
        ("save_results",        save_results_node),
    ]:
        builder.add_node(name, fn)

    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",        "extract_info")
    builder.add_edge("extract_info",       "deduplicate_brands")
    builder.add_edge("deduplicate_brands", "analyze_sentiments")
    builder.add_edge("analyze_sentiments", "update_metadata")
    builder.add_edge("update_metadata",    "aggregate_results")
    builder.add_edge("aggregate_results",  "save_results")
    builder.add_edge("save_results",        END)
    return builder.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_analysis(video_id: str, video_title: str = "") -> dict:
    final = _graph.invoke({
        "video_id": video_id, "video_title": video_title,
        "chunks": [], "extractions": [], "brand_mapping": {},
        "company_model_sentiments": {}, "aggregated": {},
    })
    return final["aggregated"]


def get_cached_analysis(video_id: str) -> dict | None:
    cache_path = CACHE_DIR / f"{video_id}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
