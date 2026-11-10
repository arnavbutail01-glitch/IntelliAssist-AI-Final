"""Thin REST clients for Gemini / OpenAI (no SDK version headaches)."""
from __future__ import annotations

from typing import Optional

import time

import requests

from config.settings import GEMINI_MODEL, OPENAI_MODEL, get_secret


class LLMError(Exception):
    """Raised for any LLM failure. Never contains the API key."""


class GeminiLLM:
    provider = "Gemini"

    def __init__(self, api_key: str, model: str = GEMINI_MODEL, timeout: int = 60) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    # Tried in order if the configured model has been retired (HTTP 404).
    FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash")

    def _call(self, model: str, body: dict):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            return requests.post(url, json=body, timeout=self.timeout,
                                 headers={"x-goog-api-key": self.api_key})
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Gemini ({type(exc).__name__}).") from exc

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        candidates = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        r = None
        for model in candidates:
            r = self._call(model, body)
            if r.status_code in (429, 500, 503):      # transient overload / rate limit: retry once
                time.sleep(1.5)
                r = self._call(model, body)
            if r.status_code != 404:      # 404 = model retired/unknown -> try the next one
                self.model = model        # remember the working model
                break
        if r.status_code != 200:
            raise LLMError(f"Gemini API error {r.status_code}: {r.text[:160]}")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Gemini returned no answer (the response may have been blocked).") from exc
        if not text:
            raise LLMError("Gemini returned an empty answer.")
        return text


class OpenAILLM:
    provider = "OpenAI"

    def __init__(self, api_key: str, model: str = OPENAI_MODEL, timeout: int = 60) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions", timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": 0.2},
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach OpenAI ({type(exc).__name__}).") from exc
        if r.status_code != 200:
            raise LLMError(f"OpenAI API error {r.status_code}: {r.text[:160]}")
        try:
            text = r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError("OpenAI returned an unexpected response.") from exc
        if not text:
            raise LLMError("OpenAI returned an empty answer.")
        return text


def make_llm(provider: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    return OpenAILLM(api_key) if provider.lower().startswith("openai") else GeminiLLM(api_key)


def get_llm():
    """Pick an LLM from environment variables / Streamlit secrets, or None."""
    key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
    if key:
        return GeminiLLM(key)
    key = get_secret("OPENAI_API_KEY")
    if key:
        return OpenAILLM(key)
    return None
