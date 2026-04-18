import os
import json
import pathlib
from typing import List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import AzureOpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.graph import StateGraph, END

from vector_store import _get_client, COLLECTION_NAME

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path(__file__).parent / "analysis_cache"
CACHE_DIR.mkdir(exist_ok=True)

_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION", "2024-12-01-preview"),
)
_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-ria-dev-01")

# Separate client for brand extraction and verification — uses the web search model
_brand_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_BRAND_API_VERSION", "2025-04-01-preview"),
)
_BRAND_DEPLOYMENT = os.getenv("AZURE_OPENAI_BRAND_DEPLOYMENT", "gpt-5.4-mini")

MAX_WORKERS = 20


# ── Step 1 prompt: EXTRACT brand → model → features ──────────────────────────

EXTRACT_PROMPT = """\
You are a car content analyst. Read the text below and extract car manufacturers, their models, and features.

── DEFINITIONS ──────────────────────────────────────────────────────────────────
• "brand"  = the car MANUFACTURER / COMPANY name only
             e.g. "Mahindra", "Tata", "Hyundai", "Toyota", "BMW"
• "model"  = the specific CAR MODEL name
             e.g. "XUV 700", "Nexon", "Creta", "Fortuner", "3 Series"

── CRITICAL BRAND vs MODEL RULE ─────────────────────────────────────────────────
Think through each name you see:

  • "Mahindra XUV 700" → brand = "Mahindra",  model = "XUV 700"   ✓
  • "XUV 700"          → brand = "Mahindra",  model = "XUV 700"   ✓  (infer brand)
  • "700"              → brand = "Mahindra",  model = "XUV 700"   ✓  (it's a model code)
  • "Mahindra"         → brand = "Mahindra",  model = "General"   ✓

  NEVER put a model name ("XUV 700", "Nexon", "700") in the "brand" field.
  If only a model name appears without a brand, infer the correct manufacturer from your knowledge.

── STEPS ─────────────────────────────────────────────────────────────────────────
1. Identify every car manufacturer (company) mentioned.
2. For each manufacturer, list all models discussed. Use "General" if no specific model.
3. For each model, list generic features discussed. For each feature provide:
   • "name"     — concise generic label (2–4 words, e.g. "ride quality", "fuel efficiency")
   • "verbatim" — the exact phrase or sentence from the text that mentions this feature

Return ONLY valid JSON:
{{
  "brands": [
    {{
      "brand": "CompanyName",
      "models": [
        {{
          "name": "ModelName",
          "overall_sentiment": "positive",
          "features": [
            {{"name": "ride quality", "verbatim": "The ride quality is very smooth even on bad roads", "sentiment": "positive"}},
            {{"name": "fuel efficiency", "verbatim": "It delivers around 18 kmpl on the highway", "sentiment": "neutral"}}
          ]
        }}
      ]
    }}
  ]
}}

Rules:
- "brand" field = manufacturer/company name ONLY — never a model name or combined string.
- "name" must be concise and generic (2–4 words max).
- "verbatim" must be a short phrase or single sentence copied directly from the text — not invented.
- "sentiment" per feature: "positive", "negative", or "neutral" — infer from the verbatim quote.
- "overall_sentiment" per model: aggregate sentiment across its features.
- If no car content found, return {{"brands": []}}.
- Return ONLY the JSON object.

Text:
{text}"""

# ── Step 2 prompt: DEDUP + VERIFY all raw brand names together in one batch ────

DEDUP_VERIFY_PROMPT = """\
You are a car industry expert with web search access. Below is a list of names extracted from video text as potential car manufacturer names.

Your task: for EACH name, reason step-by-step (chain of thought) to decide if it is a real, standalone car manufacturer. Then return a single JSON result.

Names to verify:
{names_list}

━━━ CHAIN OF THOUGHT — apply to EVERY name ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Classify the name before searching:

  (A) Pure manufacturer/company name → e.g. "Mahindra", "Tata Motors", "Hyundai"
  (B) Car model name (with or without numbers) → e.g. "XUV 700", "Nexon", "Creta", "Scorpio", "Thar", "i20"
  (C) Brand + model combined → e.g. "Mahindra XUV 700", "Toyota Fortuner"
  (D) Standalone number or alphanumeric code → e.g. "700", "800", "3008", "Q5", "EV6"
  (E) Ambiguous — proceed to web search

STEP 2 — Apply immediate rules (no web search needed):

  • Category (B) — model name: verified = false
    Reasoning: "XUV 700" is a car model made by Mahindra. Models are never manufacturers.
  • Category (C) — brand+model: verified = false
    Reasoning: "Mahindra XUV 700" contains both the company and a model — it is not a standalone manufacturer name.
  • Category (D) — number/code: verified = false
    Reasoning: "700" is a model designation (part of "XUV 700"). Numbers alone are never manufacturers.

STEP 3 — For Category (A) and (E), use web search to confirm:

  • Does this company independently manufacture and sell cars under its own brand?
  • Is the web result unambiguous?
  • If any doubt remains → verified = false

STEP 4 — For confirmed manufacturers, pick the single most widely recognised official canonical name.

━━━ KEY EXAMPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "Mahindra XUV 700" → Step 1: Category C (brand+model) → verified: false
  "XUV 700"          → Step 1: Category B (model name)   → verified: false
  "700"              → Step 1: Category D (number code)   → verified: false
  "Mahindra"         → Step 1: Category A → web search confirms manufacturer → verified: true, canonical: "Mahindra"
  "Tata"             → Step 1: Category A → web search confirms manufacturer → verified: true, canonical: "Tata Motors"
  "Nexon"            → Step 1: Category B (Tata model)   → verified: false

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a JSON object with every input name as a key exactly as written:
{{
  "Mahindra": {{"verified": true,  "canonical_name": "Mahindra"}},
  "XUV 700":  {{"verified": false, "canonical_name": null}},
  "700":      {{"verified": false, "canonical_name": null}},
  "Tata":     {{"verified": true,  "canonical_name": "Tata Motors"}}
}}

Rules:
- Every input name must appear exactly once as a key.
- Set verified: true ONLY when 100% certain the name is a standalone car manufacturer.
- Return ONLY the JSON object — no explanation, no markdown."""



# ── State ──────────────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    video_id: str
    video_title: str
    chunks: List[dict]                # raw chunks from Qdrant
    extractions: List[dict]           # per-chunk: brand → model → features
    brand_mapping: dict               # raw_name → canonical_name | null (from dedup)
    verified_brands: dict             # canonical_name → official_name (100% confirmed via web)
    company_model_sentiments: dict    # { company: { model: sentiment_result } }
    aggregated: dict                  # final output


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Normalize brand/model name for deduplication (title case)."""
    return " ".join(w.capitalize() for w in name.strip().split()) if name.strip() else ""



def _extract_json_from_content(content: str) -> dict:
    """Extract JSON object from model output that may contain extra text or citations."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


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


# ── Node 2: Extract brand → model → features ──────────────────────────────────
# Chunks are grouped by source and combined into batches before extraction.
# This gives the model full context rather than a narrow 500-char window.
# Batch size: ~6000 chars per call. Results are mapped back to every chunk
# in the batch so verbatim tracking still works per-chunk.

BATCH_CHAR_LIMIT = 6000  # max combined text per extraction call


def _make_batches(chunks: list[dict]) -> list[list[dict]]:
    """
    Group chunks by source (transcript / comments), then split into
    batches where the combined text stays under BATCH_CHAR_LIMIT.
    """
    from collections import defaultdict
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c)

    batches: list[list[dict]] = []
    for source_chunks in by_source.values():
        batch: list[dict] = []
        length = 0
        for c in source_chunks:
            clen = len(c["text"])
            if batch and length + clen > BATCH_CHAR_LIMIT:
                batches.append(batch)
                batch, length = [], 0
            batch.append(c)
            length += clen
        if batch:
            batches.append(batch)
    return batches


def _extract_batch(batch: list[dict]) -> list[dict]:
    """
    Combine all chunks in the batch into one text block, run a single
    extraction call, then replicate the result back to every chunk in
    the batch (so each chunk carries the brands/features found in its
    surrounding context and verbatim lookup still works).
    """
    combined_text = "\n\n".join(c["text"].strip() for c in batch if c["text"].strip())
    if not combined_text:
        return [
            {"chunk_id": c["id"], "source": c["source"], "text": c["text"], "brands": []}
            for c in batch
        ]

    try:
        resp = _openai.chat.completions.create(
            model=_LLM_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=combined_text[:8000])}],
            temperature=0,
        )
        parsed = _extract_json_from_content(resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"[extract] Batch error ({len(batch)} chunks): {exc}")
        parsed = {}

    brands = []
    for b in (parsed.get("brands") or []):
        if not isinstance(b, dict) or not b.get("brand"):
            continue
        company = _normalize(b["brand"])
        if not company:
            continue
        models = []
        for m in (b.get("models") or []):
            if not isinstance(m, dict) or not m.get("name"):
                continue
            model_name = _normalize(m["name"]) or "General"
            features = []
            for f in (m.get("features") or []):
                if isinstance(f, dict) and f.get("name"):
                    fsent = f.get("sentiment", "neutral")
                    if fsent not in ("positive", "negative", "neutral"):
                        fsent = "neutral"
                    features.append({
                        "name":      f["name"].strip(),
                        "verbatim":  (f.get("verbatim") or "").strip(),
                        "sentiment": fsent,
                    })
                elif isinstance(f, str) and f.strip():
                    features.append({"name": f.strip(), "verbatim": "", "sentiment": "neutral"})
            overall_sent = m.get("overall_sentiment", "neutral")
            if overall_sent not in ("positive", "negative", "neutral"):
                overall_sent = "neutral"
            models.append({"name": model_name, "overall_sentiment": overall_sent, "features": features})
        if models:
            brands.append({"brand": company, "models": models})

    # Replicate extracted brands back to every chunk in the batch
    return [
        {"chunk_id": c["id"], "source": c["source"], "text": c["text"], "brands": brands}
        for c in batch
    ]


def extract_info_node(state: AnalysisState) -> AnalysisState:
    """
    Combine chunks by source into batches, extract brands+features per batch
    in parallel, then flatten results back to per-chunk extractions.
    """
    chunks = state["chunks"]
    batches = _make_batches(chunks)
    print(f"[extract] {len(chunks)} chunks → {len(batches)} batches")

    # chunk_id → extraction result
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_extract_batch, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(future_map):
            batch_results = future.result()
            for ext in batch_results:
                results[ext["chunk_id"]] = ext
            completed += 1
            print(f"[extract] {completed}/{len(batches)} batches done")

    extractions = [results[c["id"]] for c in chunks if c["id"] in results]
    return {**state, "extractions": extractions}


# ── Nodes 3+4 merged: Dedup + Verify ALL brands in one batch web-search call ──

def _dedup_verify_batch(all_raw: list[str]) -> dict[str, str]:
    """
    Single web-search call with ALL raw brand names together.
    Returns raw_name → canonical_name for verified manufacturers only.
    Unverified / invalid entries are absent from the result.
    """
    names_list = "\n".join(f"- {b}" for b in sorted(all_raw))
    prompt = DEDUP_VERIFY_PROMPT.replace("{names_list}", names_list)
    try:
        resp = _brand_openai.responses.create(
            model=_BRAND_DEPLOYMENT,
            input=prompt,
            tools=[{"type": "web_search"}],
        )
        parsed = _extract_json_from_content(resp.output_text or "")
    except Exception as exc:
        print(f"[dedup_verify] Batch call error: {exc}")
        return {}

    raw_to_canonical: dict[str, str] = {}
    for raw, result in parsed.items():
        if not isinstance(result, dict):
            continue
        if result.get("verified") and result.get("canonical_name"):
            canonical = str(result["canonical_name"]).strip()
            if canonical:
                raw_to_canonical[raw] = canonical
    return raw_to_canonical


def dedup_verify_brands_node(state: AnalysisState) -> AnalysisState:
    """
    LangGraph node — single batch web-search call for ALL raw brand names.

    Sends every unique raw name extracted across chunks in one request.
    The model searches the web and returns a verified canonical mapping.
    Multiple raw names resolving to the same canonical are merged.
    Unverified names (model codes, numbers, non-manufacturers) are dropped.
    """
    all_raw: list[str] = sorted({
        b["brand"]
        for ext in state["extractions"]
        for b in ext["brands"]
    })

    if not all_raw:
        print("[dedup_verify] No brands found")
        return {**state, "brand_mapping": {}, "verified_brands": {}}

    print(f"[dedup_verify] Batch verifying {len(all_raw)} raw names: {all_raw}")

    raw_to_canonical = _dedup_verify_batch(all_raw)

    for raw, canonical in raw_to_canonical.items():
        print(f"[dedup_verify] ✓ '{raw}' → '{canonical}'")
    dropped = sorted(set(all_raw) - set(raw_to_canonical.keys()))
    for d in dropped:
        print(f"[dedup_verify] ✗ DROPPED '{d}' — not a confirmed manufacturer")

    # Deduplicate: multiple raws → same canonical collapse into one
    canonical_to_raws: dict[str, set] = {}
    for raw, canonical in raw_to_canonical.items():
        canonical_to_raws.setdefault(canonical, set()).add(raw)

    verified_brands = {canonical: canonical for canonical in canonical_to_raws}
    print(f"[dedup_verify] {len(verified_brands)} verified: {sorted(verified_brands)}, "
          f"{len(dropped)} dropped: {dropped}")

    # Rewrite extractions — drop unverified, remap raw → canonical
    new_extractions = []
    for ext in state["extractions"]:
        clean_brands = []
        for b in ext["brands"]:
            canonical = raw_to_canonical.get(b["brand"])
            if canonical:
                clean_brands.append({**b, "brand": canonical})
        new_extractions.append({**ext, "brands": clean_brands})

    return {
        **state,
        "extractions":     new_extractions,
        "brand_mapping":   raw_to_canonical,
        "verified_brands": verified_brands,
    }


# ── Brand-level model dedup prompt ────────────────────────────────────────────

MODEL_DEDUP_PROMPT = """\
You are a car industry expert. Below is a list of model names extracted from video text for the brand "{brand}".
There may be duplicates, abbreviations, partial names, and number-only codes that all refer to the same model.

Raw model names:
{models}

Produce a mapping from every raw model name to a single canonical official model name.

Examples:
  "700"     → "XUV 700"    (number code → full model name)
  "XUV 700" → "XUV 700"   (already canonical)
  "3XO"     → "XUV 3XO"   (abbreviation → full model name)
  "XUV 3XO" → "XUV 3XO"  (already canonical)
  "General" → "General"   (always keep as-is)
  "Scorpio" → "Scorpio"   (no ambiguity — keep as-is)

Rules:
- Use the most complete, officially recognised model name as the canonical.
- Multiple raw names may map to the same canonical — that is expected and desired.
- "General" must always map to "General".
- If a name clearly cannot be a model for this brand, still map it to its closest match or itself.
- Every input name must appear as a key exactly once.
- Return ONLY a JSON object, no markdown, no explanation:

{{"raw name": "canonical name", ...}}"""


def _dedup_models_for_brand(brand: str, company_index: dict) -> dict:
    """
    One LLM call per brand: map all raw model names to canonical names,
    then merge the index entries that resolve to the same canonical.

    The merge combines texts, chunk_ids, features, and feature_texts so
    downstream feature-dedup and sentiment steps see a single unified entry.
    """
    raw_models = list(company_index.keys())
    if len(raw_models) <= 1:
        return company_index

    try:
        resp = _openai.chat.completions.create(
            model=_LLM_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": MODEL_DEDUP_PROMPT.format(
                brand=brand,
                models="\n".join(f"- {m}" for m in sorted(raw_models)),
            )}],
            temperature=0,
        )
        mapping: dict = _extract_json_from_content(resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"[model_dedup] Error for '{brand}': {exc} — skipping")
        return company_index

    # Fallback: any raw name missing from the LLM response maps to itself
    for m in raw_models:
        if m not in mapping:
            mapping[m] = m

    # Validate mapping values — must be non-empty strings
    for raw in raw_models:
        val = mapping.get(raw)
        if not isinstance(val, str) or not val.strip():
            mapping[raw] = raw
        else:
            mapping[raw] = val.strip()

    # Merge index entries that share the same canonical name
    merged: dict = {}
    for raw_model in raw_models:
        canonical = mapping[raw_model]
        if canonical not in merged:
            merged[canonical] = {
                "texts":              [],
                "features":           set(),
                "chunk_ids":          [],
                "feature_sentiments": {},
            }
        target = merged[canonical]
        src    = company_index[raw_model]

        for t in src["texts"]:
            if t not in target["texts"]:
                target["texts"].append(t)

        for cid in src["chunk_ids"]:
            if cid not in target["chunk_ids"]:
                target["chunk_ids"].append(cid)

        target["features"] |= src["features"]

        for fname, records in src["feature_sentiments"].items():
            if fname not in target["feature_sentiments"]:
                target["feature_sentiments"][fname] = []
            for r in records:
                if r not in target["feature_sentiments"][fname]:
                    target["feature_sentiments"][fname].append(r)

    # Log merges
    for raw, canon in mapping.items():
        if raw != canon:
            print(f"[model_dedup] '{brand}': '{raw}' → '{canon}'")
    print(f"[model_dedup] '{brand}': {len(raw_models)} raw → {len(merged)} canonical models")
    return merged


# ── Brand-level feature dedup prompt ──────────────────────────────────────────

BRAND_FEATURE_DEDUP_PROMPT = """\
You are given a list of raw car feature names collected from multiple text chunks about cars made by "{brand}".
There are duplicates, synonyms, and near-duplicates.

Raw feature list:
{features}

Produce a mapping from every raw feature to a single canonical concise name (2–4 words).
Merge synonyms and near-duplicates: e.g. "fuel economy", "mileage", "kmpl" → "fuel efficiency".

Return ONLY a JSON object:
{{"raw feature": "canonical name", ...}}

Rules:
- Every input feature must appear as a key exactly once.
- Canonical names must be concise and generic (2–4 words).
- Multiple raw features may map to the same canonical name — that is expected.
- Return ONLY the JSON object."""


def _dedup_features_for_brand(brand: str, company_index: dict) -> dict:
    """
    One LLM call per brand: collect all raw features across all models,
    produce a canonical mapping, return updated model entries.
    """
    all_raw: set[str] = set()
    for entry in company_index.values():
        all_raw.update(entry["features"])

    if not all_raw:
        return company_index

    try:
        resp = _openai.chat.completions.create(
            model=_LLM_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": BRAND_FEATURE_DEDUP_PROMPT.format(
                brand=brand,
                features="\n".join(sorted(all_raw)),
            )}],
            temperature=0,
        )
        mapping: dict = _extract_json_from_content(resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"[feature_dedup] Error for '{brand}': {exc} — skipping")
        return company_index

    # Ensure every raw feature has a mapping
    for f in all_raw:
        if f not in mapping:
            mapping[f] = f

    # Apply mapping to every model entry under this brand
    updated: dict = {}
    for model_name, entry in company_index.items():
        canonical_features: set[str] = {mapping.get(f, f) for f in entry["features"]}
        new_feature_sentiments: dict[str, list] = {}
        for raw_f, records in entry["feature_sentiments"].items():
            canon = mapping.get(raw_f, raw_f)
            if canon not in new_feature_sentiments:
                new_feature_sentiments[canon] = []
            for r in records:
                if r not in new_feature_sentiments[canon]:
                    new_feature_sentiments[canon].append(r)
        updated[model_name] = {**entry, "features": canonical_features, "feature_sentiments": new_feature_sentiments}

    print(f"[feature_dedup] '{brand}': {len(all_raw)} raw → {len(set(mapping.values()))} canonical features")
    return updated


# ── Node 5: Sentiment per company→model ───────────────────────────────────────

def _build_company_model_index(extractions: List[dict]) -> dict:
    """Group extractions by company → model, tracking per-feature verbatim + sentiment."""
    index: dict = {}
    for ext in extractions:
        for b in ext["brands"]:
            company = b["brand"]
            if company not in index:
                index[company] = {}
            for m in b["models"]:
                model_name = m["name"]
                if model_name not in index[company]:
                    index[company][model_name] = {
                        "texts":              [],
                        "features":           set(),
                        "chunk_ids":          [],
                        "feature_sentiments": {},  # feature_name → [{"text": str, "sentiment": str}]
                    }
                entry = index[company][model_name]
                if ext["text"]:
                    entry["texts"].append(ext["text"])
                entry["chunk_ids"].append(ext["chunk_id"])
                for feature in m["features"]:
                    fname     = feature["name"]
                    fverbatim = feature.get("verbatim", "")
                    fsent     = feature.get("sentiment", "neutral")
                    entry["features"].add(fname)
                    if fname not in entry["feature_sentiments"]:
                        entry["feature_sentiments"][fname] = []
                    if fverbatim:
                        record = {"text": fverbatim, "sentiment": fsent}
                        if record not in entry["feature_sentiments"][fname]:
                            entry["feature_sentiments"][fname].append(record)
    return index


def _aggregate_sentiment_from_extraction(company: str, model: str, entry: dict) -> tuple[str, str, dict]:
    """
    Aggregate sentiment and verbatim from extraction-time data — no extra LLM call.
    Sentiment was already assessed per feature verbatim during the extraction step.
    """
    _score_map = {"positive": 0.8, "neutral": 0.5, "negative": 0.2}
    canonical_features  = sorted(entry["features"])
    feature_sentiments  = entry.get("feature_sentiments", {})

    features = []
    for canon in canonical_features:
        records = feature_sentiments.get(canon, [])
        # Drop features with no verbatim evidence
        if not records:
            continue
        texts = [r["text"] for r in records if r.get("text")]
        if not texts:
            continue
        # Majority-vote sentiment across all verbatim occurrences
        sents = [r.get("sentiment", "neutral") for r in records]
        pos, neg = sents.count("positive"), sents.count("negative")
        fsent = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        features.append({
            "name":          canon,
            "sentiment":     fsent,
            "mention_count": len(records),
            "verbatim":      texts[:5],
        })

    # Overall sentiment: mention-count-weighted average of feature scores
    if features:
        total_w = sum(f["mention_count"] for f in features)
        oscore  = sum(_score_map.get(f["sentiment"], 0.5) * f["mention_count"] for f in features) / total_w
    else:
        oscore = 0.5
    osent = "positive" if oscore >= 0.6 else "negative" if oscore <= 0.4 else "neutral"

    # mention_count = unique verbatim texts across all features
    all_verbatims: set[str] = set()
    for records in feature_sentiments.values():
        all_verbatims.update(r["text"] for r in records if r.get("text"))
    mention_count = len(all_verbatims) if all_verbatims else 1

    return company, model, {
        "overall_sentiment": osent,
        "overall_score":     round(oscore, 4),
        "mention_count":     mention_count,
        "features":          features,
    }


def analyze_sentiments_node(state: AnalysisState) -> AnalysisState:
    """
    Step 1 — model-name dedup per brand (one LLM call per brand, in parallel).
    Step 2 — feature dedup per brand (one LLM call per brand, in parallel).
    Step 3 — aggregate sentiment from extraction-time data (no LLM call).
    """
    company_model_index = _build_company_model_index(state["extractions"])

    # Step 1: deduplicate model names per brand in parallel
    print(f"[model_dedup] Deduplicating models for {len(company_model_index)} brands")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        model_dedup_futures = {
            pool.submit(_dedup_models_for_brand, brand, models): brand
            for brand, models in company_model_index.items()
        }
        for future in as_completed(model_dedup_futures):
            brand = model_dedup_futures[future]
            try:
                company_model_index[brand] = future.result()
            except Exception as exc:
                print(f"[model_dedup] Failed for '{brand}': {exc}")

    # Step 2: deduplicate features brand-wise in parallel
    print(f"[feature_dedup] Deduplicating features for {len(company_model_index)} brands")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        dedup_futures = {
            pool.submit(_dedup_features_for_brand, brand, models): brand
            for brand, models in company_model_index.items()
        }
        for future in as_completed(dedup_futures):
            brand = dedup_futures[future]
            company_model_index[brand] = future.result()

    # Step 3: aggregate sentiment from extraction-time verbatim — no LLM call
    pairs = [
        (company, model, entry)
        for company, models in company_model_index.items()
        for model, entry in models.items()
    ]
    print(f"[sentiment] Aggregating {len(pairs)} company+model pairs from extraction data")

    company_model_sentiments: dict = {}
    for company, model, entry in pairs:
        try:
            company_out, model_out, result = _aggregate_sentiment_from_extraction(company, model, entry)
            if company_out not in company_model_sentiments:
                company_model_sentiments[company_out] = {}
            company_model_sentiments[company_out][model_out] = result
            print(f"[sentiment] ✓ {company_out} / {model_out} → {result['overall_sentiment']}")
        except Exception as exc:
            print(f"[sentiment] Failed for '{company}/{model}': {exc}")

    return {**state, "company_model_sentiments": company_model_sentiments}


# ── Node 6: Update Qdrant metadata ────────────────────────────────────────────

def update_metadata_node(state: AnalysisState) -> AnalysisState:
    qdrant = _get_client()

    def _update(ext: dict):
        companies = [b["brand"] for b in ext["brands"]]
        models    = [m["name"] for b in ext["brands"] for m in b["models"]]
        features  = [
            f["name"] if isinstance(f, dict) else f
            for b in ext["brands"] for m in b["models"] for f in m["features"]
        ]
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
    _score_map = {"positive": 0.8, "neutral": 0.5, "negative": 0.2}

    def _sentiment(score: float) -> str:
        return "positive" if score >= 0.6 else "negative" if score <= 0.4 else "neutral"

    brand_analysis = {}
    company_weighted_scores: list[tuple[float, int]] = []

    for company, model_results in cms.items():
        clean_model_results = {}
        model_weighted_scores: list[tuple[float, int]] = []

        for m, r in model_results.items():
            features = r.get("features", [])
            mention_count = r.get("mention_count", 1)

            # Recompute model sentiment bottom-up from feature sentiments
            # weighted by each feature's mention_count — more reliable than LLM's overall
            if features:
                total_weight = sum(f["mention_count"] for f in features)
                if total_weight > 0:
                    model_score = sum(
                        _score_map.get(f["sentiment"], 0.5) * f["mention_count"]
                        for f in features
                    ) / total_weight
                else:
                    model_score = _score_map.get(r.get("overall_sentiment", "neutral"), 0.5)
            else:
                # No features → fall back to LLM's overall_sentiment
                model_score = _score_map.get(r.get("overall_sentiment", "neutral"), 0.5)

            model_score = round(model_score, 4)
            model_sentiment = _sentiment(model_score)

            clean_model_results[m] = {
                k: v for k, v in r.items() if k not in ("overall_score",)
            }
            clean_model_results[m]["overall_sentiment"] = model_sentiment

            model_weighted_scores.append((model_score, mention_count))

        # Company score = mention-weighted average of model scores
        total_model_mentions = sum(w for _, w in model_weighted_scores)
        company_score = (
            sum(s * w for s, w in model_weighted_scores) / total_model_mentions
            if total_model_mentions > 0 else 0.5
        )
        company_score = round(company_score, 4)
        company_sentiment = _sentiment(company_score)
        total_company_mentions = sum(r["mention_count"] for r in model_results.values())
        company_weighted_scores.append((company_score, total_company_mentions))

        brand_analysis[company] = {
            "overall_sentiment": company_sentiment,
            "mention_count":     total_company_mentions,
            "models": dict(
                sorted(clean_model_results.items(), key=lambda x: -x[1]["mention_count"])
            ),
        }

    brand_analysis = dict(
        sorted(brand_analysis.items(), key=lambda x: -x[1]["mention_count"])
    )

    # Overall = mention-weighted average across all companies
    total_all_mentions = sum(w for _, w in company_weighted_scores)
    overall_score = (
        sum(s * w for s, w in company_weighted_scores) / total_all_mentions
        if total_all_mentions > 0 else 0.5
    )
    overall_score = round(overall_score, 4)
    overall_sentiment = _sentiment(overall_score)

    # Sentiment distribution: mention-weighted across all models (not per-company count)
    sent_weighted: dict[str, float] = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    total_model_mentions_all = 0
    for v in brand_analysis.values():
        for mdata in v["models"].values():
            mc = mdata["mention_count"]
            sent_weighted[mdata["overall_sentiment"]] += mc
            total_model_mentions_all += mc
    sentiment_distribution = {
        k: round(v / total_model_mentions_all * 100, 1) if total_model_mentions_all > 0 else 0.0
        for k, v in sent_weighted.items()
    }

    aggregated = {
        "overall_sentiment":      overall_sentiment,
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
        ("load_chunks",        load_chunks_node),
        ("extract_info",       extract_info_node),
        ("dedup_verify_brands", dedup_verify_brands_node),  # parallel dedup + web verification
        ("analyze_sentiments", analyze_sentiments_node),
        ("update_metadata",    update_metadata_node),
        ("aggregate_results",  aggregate_results_node),
        ("save_results",       save_results_node),
    ]:
        builder.add_node(name, fn)

    builder.set_entry_point("load_chunks")
    builder.add_edge("load_chunks",         "extract_info")
    builder.add_edge("extract_info",        "dedup_verify_brands")
    builder.add_edge("dedup_verify_brands", "analyze_sentiments")
    builder.add_edge("analyze_sentiments",  "update_metadata")
    builder.add_edge("update_metadata",     "aggregate_results")
    builder.add_edge("aggregate_results",   "save_results")
    builder.add_edge("save_results",        END)
    return builder.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_analysis(video_id: str, video_title: str = "") -> dict:
    final = _graph.invoke({
        "video_id": video_id, "video_title": video_title,
        "chunks": [], "extractions": [], "brand_mapping": {},
        "verified_brands": {},
        "company_model_sentiments": {}, "aggregated": {},
    })
    return final["aggregated"]


def get_cached_analysis(video_id: str) -> dict | None:
    cache_path = CACHE_DIR / f"{video_id}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
