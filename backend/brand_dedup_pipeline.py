"""
Post-processing pipeline: two-phase deduplication on saved analysis JSON.

Run this AFTER the main analysis pipeline has saved results to cache.

Phase 1 — Company-level dedup (single LLM web-search call)
  e.g.  "Tata" + "Tata Motors"  →  "Tata Motors"
        "MG"   + "MG Motor"     →  "MG Motor"

Phase 2 — Model-level dedup per company (one LLM web-search call per company, in parallel)
  e.g.  Under Mahindra: "XUV 700", "700", "XUV700"  →  "XUV 700"
        Under Tata:     "Nexon EV", "Nexon ev"       →  "Nexon EV"

Both phases merge data (features, mention counts, sentiments) and
recalculate top-level aggregated fields before overwriting the cache file.

Public API:
  run_brand_dedup(video_id: str) -> dict   # returns updated aggregated dict
"""

import os
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import AzureOpenAI

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path(__file__).parent / "analysis_cache"
MAX_WORKERS = 10

_brand_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_BRAND_API_VERSION", "2025-04-01-preview"),
)
_BRAND_DEPLOYMENT = os.getenv("AZURE_OPENAI_BRAND_DEPLOYMENT", "gpt-5.4-mini")


# ── Prompts ────────────────────────────────────────────────────────────────────

# Phase 1: company / manufacturer dedup
COMPANY_DEDUP_PROMPT = """\
You are a car industry expert with web search access.

The manufacturer names below were extracted from a video analysis.
Some may be duplicate variants of the same company:
  e.g. "Tata" and "Tata Motors", "MG" and "MG Motor", "VW" and "Volkswagen"

Manufacturer names found:
{names_list}

Use web search to confirm which names belong to the same company.
For EACH name return the single most widely recognised official canonical name.

Return ONLY a JSON object where every input name is a key:
{{
  "Tata": "Tata Motors",
  "Tata Motors": "Tata Motors",
  "MG": "MG Motor",
  "MG Motor": "MG Motor",
  "Hyundai": "Hyundai"
}}

Rules:
- Every input name must appear as a key exactly once.
- Map a name to itself if it is already the correct canonical form.
- Return ONLY the JSON object — no explanation, no markdown."""


# Phase 2: model dedup within a single company
MODEL_DEDUP_PROMPT = """\
You are a car industry expert with web search access.

The model names below were all extracted from a video about cars made by "{company}".
Some may be duplicates or variant spellings of the same model:
  e.g. "XUV 700", "700", "XUV700", "Xuv 700"  →  all refer to the same car

Model names found under {company}:
{names_list}

Use web search to confirm which names refer to the same car model.
For EACH name return the single correct official canonical model name.

Return ONLY a JSON object where every input name is a key:
{{
  "XUV 700":  "XUV700",
  "700":      "XUV700",
  "XUV700":   "XUV700",
  "Scorpio":  "Scorpio N",
  "Scorpio N": "Scorpio N"
}}

Rules:
- Every input name must appear as a key exactly once.
- Map a name to itself if it is already the correct canonical form.
- Use the official model name as found on {company}'s website.
- Return ONLY the JSON object — no explanation, no markdown."""


# ── JSON helper ────────────────────────────────────────────────────────────────

def _extract_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_company_dedup_llm(company_names: list[str]) -> dict[str, str]:
    """
    Single web-search call for all company names.
    Returns raw → canonical for every input; falls back to identity on error.
    """
    names_list = "\n".join(f"- {n}" for n in sorted(company_names))
    prompt = COMPANY_DEDUP_PROMPT.format(names_list=names_list)
    try:
        resp = _brand_openai.responses.create(
            model=_BRAND_DEPLOYMENT,
            input=prompt,
            tools=[{"type": "web_search"}],
        )
        mapping = _extract_json(resp.output_text or "")
    except Exception as exc:
        print(f"[company_dedup] LLM error: {exc}")
        return {n: n for n in company_names}

    for n in company_names:
        if n not in mapping or not isinstance(mapping[n], str) or not mapping[n].strip():
            mapping[n] = n
    return mapping


def _call_model_dedup_llm(company: str, model_names: list[str]) -> dict[str, str]:
    """
    Single web-search call for all model names under one company.
    Returns raw_model → canonical_model; falls back to identity on error.
    """
    names_list = "\n".join(f"- {m}" for m in sorted(model_names))
    prompt = MODEL_DEDUP_PROMPT.format(company=company, names_list=names_list)
    try:
        resp = _brand_openai.responses.create(
            model=_BRAND_DEPLOYMENT,
            input=prompt,
            tools=[{"type": "web_search"}],
        )
        mapping = _extract_json(resp.output_text or "")
    except Exception as exc:
        print(f"[model_dedup] LLM error for '{company}': {exc}")
        return {m: m for m in model_names}

    for m in model_names:
        if m not in mapping or not isinstance(mapping[m], str) or not mapping[m].strip():
            mapping[m] = m
    return mapping


# ── Merge helpers ──────────────────────────────────────────────────────────────

def _merge_features(base: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge two feature lists, combining entries that share the same name."""
    by_name: dict[str, dict] = {f["name"]: dict(f) for f in base}
    for feat in incoming:
        name = feat["name"]
        if name not in by_name:
            by_name[name] = dict(feat)
        else:
            ef = by_name[name]
            ef["mention_count"] = ef.get("mention_count", 0) + feat.get("mention_count", 0)
            existing_v: list = ef.get("verbatim", [])
            extra_v = [v for v in feat.get("verbatim", []) if v not in existing_v]
            ef["verbatim"] = (existing_v + extra_v)[:5]
    return list(by_name.values())


def _sentiment_from_features(features: list[dict]) -> str:
    sents = [f.get("sentiment", "neutral") for f in features]
    pos, neg = sents.count("positive"), sents.count("negative")
    return "positive" if pos > neg else "negative" if neg > pos else "neutral"


def _merge_two_models(base: dict, incoming: dict) -> dict:
    """Merge two model-level dicts into one."""
    merged_features = _merge_features(
        base.get("features", []), incoming.get("features", [])
    )
    merged_mentions = base.get("mention_count", 0) + incoming.get("mention_count", 0)
    return {
        **base,
        "mention_count":     merged_mentions,
        "overall_sentiment": _sentiment_from_features(merged_features),
        "features":          merged_features,
    }


def _merge_company_entries(base: dict, incoming: dict) -> dict:
    """Merge two company-level brand_analysis entries."""
    merged_models: dict = {k: dict(v) for k, v in base.get("models", {}).items()}
    for model_name, model_data in incoming.get("models", {}).items():
        if model_name not in merged_models:
            merged_models[model_name] = dict(model_data)
        else:
            merged_models[model_name] = _merge_two_models(merged_models[model_name], model_data)

    total_mentions = base.get("mention_count", 0) + incoming.get("mention_count", 0)
    sents = [m.get("overall_sentiment", "neutral") for m in merged_models.values()]
    pos, neg = sents.count("positive"), sents.count("negative")
    overall = "positive" if pos > neg else "negative" if neg > pos else "neutral"

    return {
        "overall_sentiment": overall,
        "mention_count":     total_mentions,
        "models": dict(
            sorted(merged_models.items(), key=lambda x: -x[1].get("mention_count", 0))
        ),
    }


# ── Aggregated stats recalc ────────────────────────────────────────────────────

def _recalc_aggregated(cached: dict, new_brand_analysis: dict) -> dict:
    score_map = {"positive": 0.8, "neutral": 0.5, "negative": 0.2}
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
    all_scores = []
    for v in new_brand_analysis.values():
        s = v.get("overall_sentiment", "neutral")
        sent_counts[s] = sent_counts.get(s, 0) + 1
        all_scores.append(score_map.get(s, 0.5))

    total_brands = len(new_brand_analysis)
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.5
    overall_sentiment = (
        "positive" if overall_score >= 0.6 else
        "negative" if overall_score <= 0.4 else "neutral"
    )
    total_companies = max(total_brands, 1)
    sentiment_distribution = {
        k: round(v / total_companies * 100, 1) for k, v in sent_counts.items()
    }
    updated_aggregated = {
        **cached.get("aggregated", {}),
        "total_brands":           total_brands,
        "overall_sentiment":      overall_sentiment,
        "sentiment_distribution": sentiment_distribution,
        "brand_analysis":         new_brand_analysis,
    }
    return {**cached, "aggregated": updated_aggregated}


# ── Phase 1: company-level dedup ──────────────────────────────────────────────

def _run_company_dedup(cached: dict) -> dict:
    """
    Ask LLM to canonicalise all company names. Merges entries that
    resolve to the same canonical name. Returns updated cached dict.
    """
    brand_analysis: dict = cached.get("aggregated", {}).get("brand_analysis", {})
    company_names = list(brand_analysis.keys())

    if len(company_names) <= 1:
        print(f"[company_dedup] {len(company_names)} company — skipping")
        return cached

    print(f"[company_dedup] Checking {len(company_names)} company names: {company_names}")
    mapping = _call_company_dedup_llm(company_names)

    new_analysis: dict = {}
    for raw, data in brand_analysis.items():
        canonical = mapping.get(raw, raw)
        if canonical not in new_analysis:
            new_analysis[canonical] = dict(data)
        else:
            print(f"[company_dedup] Merging '{raw}' → '{canonical}'")
            new_analysis[canonical] = _merge_company_entries(new_analysis[canonical], data)

    for raw, canonical in sorted(mapping.items()):
        tag = "→" if raw != canonical else "✓"
        print(f"[company_dedup] {tag}  '{raw}' → '{canonical}'")

    new_analysis = dict(
        sorted(new_analysis.items(), key=lambda x: -x[1].get("mention_count", 0))
    )
    return _recalc_aggregated(cached, new_analysis)


# ── Phase 2: model-level dedup per company ────────────────────────────────────

def _dedup_models_for_company(company: str, company_data: dict) -> tuple[str, dict]:
    """
    One LLM web-search call: dedup model names for a single company.
    Returns (company, updated_company_data).
    """
    models: dict = company_data.get("models", {})
    model_names = list(models.keys())

    if len(model_names) <= 1:
        return company, company_data

    print(f"[model_dedup] '{company}': checking {len(model_names)} models: {model_names}")
    mapping = _call_model_dedup_llm(company, model_names)

    # Log
    for raw, canonical in sorted(mapping.items()):
        tag = "→" if raw != canonical else "✓"
        print(f"[model_dedup]   {company} / {tag}  '{raw}' → '{canonical}'")

    # Merge models that share the same canonical name
    new_models: dict = {}
    for raw_model, model_data in models.items():
        canonical = mapping.get(raw_model, raw_model)
        if canonical not in new_models:
            new_models[canonical] = dict(model_data)
        else:
            new_models[canonical] = _merge_two_models(new_models[canonical], model_data)

    new_models = dict(
        sorted(new_models.items(), key=lambda x: -x[1].get("mention_count", 0))
    )

    # Re-derive company-level sentiment from merged models
    sents = [m.get("overall_sentiment", "neutral") for m in new_models.values()]
    pos, neg = sents.count("positive"), sents.count("negative")
    overall = "positive" if pos > neg else "negative" if neg > pos else "neutral"

    updated = {
        **company_data,
        "overall_sentiment": overall,
        "models":            new_models,
    }
    return company, updated


def _run_model_dedup(cached: dict) -> dict:
    """
    Run model dedup in parallel — one LLM web-search call per company.
    Returns updated cached dict.
    """
    brand_analysis: dict = cached.get("aggregated", {}).get("brand_analysis", {})

    if not brand_analysis:
        return cached

    new_analysis: dict = dict(brand_analysis)  # start with current state

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_dedup_models_for_company, company, data): company
            for company, data in brand_analysis.items()
        }
        for future in as_completed(futures):
            try:
                company, updated_data = future.result()
                new_analysis[company] = updated_data
            except Exception as exc:
                company = futures[future]
                print(f"[model_dedup] Failed for '{company}': {exc}")

    new_analysis = dict(
        sorted(new_analysis.items(), key=lambda x: -x[1].get("mention_count", 0))
    )
    return _recalc_aggregated(cached, new_analysis)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_brand_dedup(video_id: str) -> dict:
    """
    Two-phase post-processing deduplication pipeline.

    Phase 1 — Company dedup  (single LLM web-search call for all company names)
    Phase 2 — Model dedup    (one LLM web-search call per company, run in parallel)

    Reads  analysis_cache/{video_id}.json
    Writes corrected JSON back to the same file.
    Returns the updated aggregated dict.
    """
    cache_path = CACHE_DIR / f"{video_id}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached analysis for video_id={video_id!r}")

    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    brand_analysis: dict = cached.get("aggregated", {}).get("brand_analysis", {})
    if not brand_analysis:
        print("[brand_dedup] Empty brand_analysis — nothing to do")
        return cached.get("aggregated", {})

    # ── Phase 1: company-level ────────────────────────────────────────────────
    print("\n── Phase 1: company-level dedup ─────────────────────────────────────")
    cached = _run_company_dedup(cached)

    # ── Phase 2: model-level per company ─────────────────────────────────────
    print("\n── Phase 2: model-level dedup (per company, parallel) ───────────────")
    cached = _run_model_dedup(cached)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    final_brands = cached.get("aggregated", {}).get("brand_analysis", {})
    print(f"\n[brand_dedup] Done. {len(brand_analysis)} companies → {len(final_brands)} after dedup. Saved to {cache_path}")
    return cached.get("aggregated", {})
