"""Provider-agnostic LLM calling, shared by the detector
(app/detection/llm_detector.py) and the eval synthetic-data generator
(eval/generate_synthetic_data.py). Kept separate from both so the generator
doesn't import the detector's internals, and so generation and detection can
use different providers without duplicating API boilerplate.

Calls go through an ordered CHAIN rather than a primary/fallback pair. The
earlier two-provider version raised once both were exhausted, and callers
turned that into "no PII found" - during a 67-document eval both free tiers
hit their daily limits and recall silently collapsed to 0.43 on part of the
set while looking like a detection problem. A chain keeps going, and
`call_llm` reports which providers it burned through so exhaustion is
visible in the logs rather than inferred afterwards from bad numbers.
"""
import re
from typing import Callable, Dict, List

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODEL,
    PROVIDER_CHAIN,
)


def strip_fences(raw: str) -> str:
    return re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip())


def call_gemini(prompt: str, system_prompt: str, model: str = GEMINI_MODEL) -> str:
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def call_groq(prompt: str, system_prompt: str, model: str = GROQ_MODEL) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_openai_compatible(prompt: str, system_prompt: str, model: str = OPENAI_COMPAT_MODEL) -> str:
    """Generic adapter for any OpenAI-compatible chat-completions endpoint -
    OpenRouter, Together, Mistral, Cerebras, a local vLLM, etc. Configured by
    OPENAI_COMPAT_BASE_URL / _API_KEY / _MODEL, so adding a provider is
    configuration rather than code."""
    import requests

    if not (OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL):
        raise RuntimeError("OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY are not set")

    response = requests.post(
        f"{OPENAI_COMPAT_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_COMPAT_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


PROVIDERS: Dict[str, Callable[[str, str], str]] = {
    "gemini": call_gemini,
    "groq": call_groq,
    "openai_compat": call_openai_compatible,
}


def _ordered_chain(preferred: str | None) -> List[str]:
    """`preferred` first, then the configured chain, skipping unknown names
    and duplicates."""
    order = ([preferred] if preferred else []) + list(PROVIDER_CHAIN)
    seen, chain = set(), []
    for name in order:
        if name in PROVIDERS and name not in seen:
            seen.add(name)
            chain.append(name)
    return chain


def call_llm(prompt: str, system_prompt: str, provider: str | None = None) -> str:
    """Tries each provider in turn; raises only if every one fails.

    Raising matters: a caller that swallows the exception and returns an
    empty result makes provider exhaustion indistinguishable from a document
    genuinely containing no PII.
    """
    chain = _ordered_chain(provider)
    if not chain:
        raise RuntimeError(f"No usable providers configured (chain={PROVIDER_CHAIN})")

    errors = []
    for name in chain:
        try:
            return PROVIDERS[name](prompt, system_prompt)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            if len(chain) > 1:
                print(f"[llm_client] provider {name!r} failed, trying next: {str(e)[:100]}")

    raise RuntimeError("all LLM providers failed -> " + " | ".join(errors))
