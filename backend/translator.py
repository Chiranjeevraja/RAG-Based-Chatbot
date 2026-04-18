"""
Detect and translate non-English chunks to English before RAG indexing.

Detection:   langdetect (local, free, ~1 ms per chunk)
Translation: Azure OpenAI (batched — one API call per ~6 000-char group)

Translated chunks carry extra metadata fields:
  original_text  — untranslated source text
  language       — ISO 639-1 code detected (e.g. "hi", "fr", "ta")
  translated     — True (only set on translated chunks)
"""

import json
import os
from typing import List

from langdetect import detect, LangDetectException, DetectorFactory

from openai import AzureOpenAI

# Deterministic language detection (langdetect is probabilistic by default)
DetectorFactory.seed = 0

_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION", "2024-12-01-preview"),
)

_TRANSLATION_MODEL = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-ria-dev-01")
_BATCH_CHAR_LIMIT = 6_000   # max combined chars sent in a single translation call
_MIN_TEXT_LENGTH = 20       # texts shorter than this are left as-is


# ── Language detection ─────────────────────────────────────────────────────────

def _detect(text: str) -> str:
    """Return an ISO 639-1 code, or 'en' when detection is not possible."""
    if len(text.strip()) < _MIN_TEXT_LENGTH:
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


# ── Translation ────────────────────────────────────────────────────────────────

_TRANSLATE_PROMPT = """\
Translate each text value to English. Preserve the original tone and meaning.
If a text is already in English, return it unchanged.
Input JSON:
{payload}

Return ONLY a JSON object with the same integer keys and English-translated string values."""


def _translate_batch(items: list[tuple[int, str]]) -> dict[int, str]:
    """
    Translate a batch of (original_index, text) pairs in a single API call.
    Returns index → translated_text.  Falls back to originals on any error.
    """
    if not items:
        return {}

    payload_str = json.dumps({str(i): text for i, text in items}, ensure_ascii=False)
    try:
        resp = _openai.chat.completions.create(
            model=_TRANSLATION_MODEL,
            messages=[{"role": "user", "content": _TRANSLATE_PROMPT.format(payload=payload_str)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000,
        )
        result = json.loads(resp.choices[0].message.content)
        return {int(k): v for k, v in result.items() if isinstance(v, str)}
    except Exception as exc:
        print(f"[translator] Translation API error: {exc} — keeping originals for this batch")
        return {i: text for i, text in items}


# ── Public API ─────────────────────────────────────────────────────────────────

def translate_chunks(chunks: List[dict]) -> List[dict]:
    """
    Detect the language of every chunk.  Translate non-English chunks to English
    in batches and return the full list with enriched metadata.

    English chunks pass through unchanged (no API call).
    """
    if not chunks:
        return chunks

    # 1. Detect language for all chunks
    languages: list[str] = [_detect(c["text"]) for c in chunks]
    non_en_idx: list[int] = [i for i, lang in enumerate(languages) if lang != "en"]

    if not non_en_idx:
        return chunks   # everything already in English

    detected_langs = {languages[i] for i in non_en_idx}
    print(
        f"[translator] {len(non_en_idx)}/{len(chunks)} chunks need translation "
        f"(detected: {detected_langs})"
    )

    # 2. Group non-English chunks into character-size batches
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    for i in non_en_idx:
        text = chunks[i]["text"]
        if current and current_chars + len(text) > _BATCH_CHAR_LIMIT:
            batches.append(current)
            current, current_chars = [], 0
        current.append((i, text))
        current_chars += len(text)
    if current:
        batches.append(current)

    # 3. Translate each batch
    translations: dict[int, str] = {}
    for batch_num, batch in enumerate(batches, 1):
        print(f"[translator] batch {batch_num}/{len(batches)} — {len(batch)} chunks")
        translations.update(_translate_batch(batch))

    # 4. Apply translations; leave English chunks untouched
    result: list[dict] = []
    for i, chunk in enumerate(chunks):
        if i in translations:
            result.append({
                **chunk,
                "text": translations[i],
                "metadata": {
                    **chunk["metadata"],
                    "original_text": chunk["text"],
                    "language": languages[i],
                    "translated": True,
                },
            })
        else:
            result.append(chunk)

    print(f"[translator] {len(translations)} chunks translated to English")
    return result
