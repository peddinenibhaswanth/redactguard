"""Thin, provider-agnostic LLM calling helpers shared by the detector
(app/detection/llm_detector.py) and the eval synthetic-data generator
(eval/generate_synthetic_data.py). Kept separate from both so the generator
doesn't need to import the detector's private functions, and so generation
and detection can deliberately use different providers/system prompts
without duplicating the API-calling boilerplate.
"""
import re

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL


def strip_fences(raw: str) -> str:
    return re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip())


def call_gemini(prompt: str, system_prompt: str, model: str = GEMINI_MODEL) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def call_groq(prompt: str, system_prompt: str, model: str = GROQ_MODEL) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_llm(prompt: str, system_prompt: str, provider: str) -> str:
    """Tries `provider` first, falls back to the other on any failure."""
    primary, fallback = (call_gemini, call_groq) if provider == "gemini" else (call_groq, call_gemini)
    try:
        return primary(prompt, system_prompt)
    except Exception:
        return fallback(prompt, system_prompt)
