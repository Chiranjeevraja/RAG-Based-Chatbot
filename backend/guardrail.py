"""
Two-layer input guardrail applied to every chat question.

Layer 1 — Regex pattern check (instant, no API call):
    Flags prompt injection, jailbreaks, system-prompt extraction attempts.

Layer 2 — OpenAI Moderation API (free, ~50 ms):
    Flags hate speech, harassment, self-harm, sexual content, violence.

Fails open on API errors — moderation unavailability never blocks the user.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import AzureOpenAI

_moderation_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION", "2024-12-01-preview"),
)
_MODERATION_MODEL = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4.1-ria-dev-01")


# ── Layer 1: injection / jailbreak patterns ────────────────────────────────────

_INJECTION_PATTERNS = [
    # Override / ignore instructions
    r"ignore\s+(previous|above|prior|all|your)\s+(instructions?|prompt|context|rules?|directives?)",
    r"(forget|disregard|override)\s+(your|all|the|previous)\s+(instructions?|training|rules?|guidelines?|context)",
    # Role-switch attempts
    r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|roleplay\s+as)\s+.{0,40}(no\s+restrictions?|without\s+(limits?|rules?|filters?|restrictions?))",
    # System-prompt extraction
    r"(repeat|print|show|reveal|output|tell\s+me|display)\s+(your\s+)?(system\s+prompt|instructions?|base\s+prompt|initial\s+prompt|rules?)",
    r"what\s+(are|is)\s+(your\s+)?(system\s+prompt|instructions?|initial\s+prompt|training\s+data)",
    # Known jailbreak keywords
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"(developer|training|admin|god|sudo|unrestricted)\s+mode",
    # Injection markers from other model families
    r"<\s*(system|instructions?)\s*>",
    r"\[INST\]|\[\/INST\]|<<SYS>>|<\|im_start\|>",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    blocked: bool
    reason: Optional[str] = None
    layer: Optional[str] = None   # "injection" | "moderation"


# ── Layer 1 ────────────────────────────────────────────────────────────────────

def _check_injection(text: str) -> GuardrailResult:
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                blocked=True,
                reason="Your message contains content that cannot be processed (prompt injection or jailbreak attempt).",
                layer="injection",
            )
    return GuardrailResult(blocked=False)


# ── Layer 2 ────────────────────────────────────────────────────────────────────

_MODERATION_PROMPT = """\
You are a content safety classifier. Check whether the user message contains any of these violations:
- hate: hate speech or discrimination based on race, religion, gender, nationality, etc.
- harassment: harassment, threats, or intimidation
- self_harm: self-harm or suicide content
- sexual: explicit sexual content
- violence: graphic violence or instructions to harm others

Respond with ONLY a JSON object.
If safe: {"flagged": false}
If unsafe: {"flagged": true, "category": "<violation type>"}"""

_CATEGORY_LABELS = {
    "hate":       "hate speech",
    "harassment": "harassment",
    "self_harm":  "self-harm",
    "sexual":     "sexual content",
    "violence":   "violence",
}


def _check_moderation(text: str) -> GuardrailResult:
    try:
        response = _moderation_client.chat.completions.create(
            model=_MODERATION_MODEL,
            messages=[
                {"role": "system", "content": _MODERATION_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=50,
        )
        result = json.loads(response.choices[0].message.content or "{}")

        if not result.get("flagged"):
            return GuardrailResult(blocked=False)

        category = result.get("category", "policy-violating")
        label = _CATEGORY_LABELS.get(category, category)
        return GuardrailResult(
            blocked=True,
            reason=f"Your message was flagged for {label} content and cannot be processed.",
            layer="moderation",
        )

    except Exception as exc:
        # Fail open — don't block users if the Moderation API is unavailable
        print(f"[guardrail] Moderation API error (failing open): {exc}")
        return GuardrailResult(blocked=False)


# ── Public API ─────────────────────────────────────────────────────────────────

def check_input(text: str) -> GuardrailResult:
    """
    Run both guardrail layers against *text*.
    Layer 1 (injection) always runs first; Layer 2 (moderation) runs only if Layer 1 passes.
    Returns GuardrailResult(blocked=True, reason=...) when the input should be rejected.
    """
    result = _check_injection(text)
    if result.blocked:
        print(f"[guardrail] BLOCKED by layer=injection")
        return result

    result = _check_moderation(text)
    if result.blocked:
        print(f"[guardrail] BLOCKED by layer=moderation, reason={result.reason!r}")
    return result
