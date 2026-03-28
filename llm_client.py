# llm_client.py
import os
import json
import time
from typing import Optional
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
class LLMResponse:
    """Structured response from an LLM call, including usage metadata."""

    def __init__(self, content: str, raw: dict):
        self.content = content
        self.raw = raw
        usage = raw.get("usage") or {}
        self.prompt_tokens: Optional[int] = usage.get("prompt_tokens")
        self.completion_tokens: Optional[int] = usage.get("completion_tokens")
        self.total_tokens: Optional[int] = usage.get("total_tokens")
        self.model: str = raw.get("model", "")

    def __str__(self) -> str:
        return self.content


class OpenRouterClient:
    """
    OpenAI-compatible client targeting OpenRouter.
    Injects into skills via SkillContext.config['llm'].

    The API key is read from the OPENROUTER_API_KEY environment variable.
    Never commit a key directly in source code.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        site_url: str = "",
        site_name: str = "INFRA-SKILL",
    ):
        self.model = model
        # Prefer explicit arg, then env var
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set the OPENROUTER_API_KEY environment variable."
            )
        self.site_url = site_url
        self.site_name = site_name

    # ------------------------------------------------------------------
    # Primary method — returns structured LLMResponse
    # ------------------------------------------------------------------

    def call(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Make an LLM request and return a structured LLMResponse."""
        payload = json.dumps({
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()

        req = urllib.request.Request(
            self.BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                raw = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"OpenRouter HTTP {e.code}: {body[:500]}"
            ) from e

        content = raw["choices"][0]["message"]["content"]
        return LLMResponse(content=content, raw=raw)

    # ------------------------------------------------------------------
    # Legacy convenience method — returns plain string
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Backward-compatible wrapper: returns only the text content."""
        return self.call(system=system, user=user, temperature=temperature).content
