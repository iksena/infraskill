# llm_client.py
import os
import json
import time
from typing import Optional
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()


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

    Timeout & retry behaviour
    -------------------------
    - Each HTTP request has a hard wall-clock timeout (default_timeout).
    - Failed requests are retried up to max_retries times with exponential
      backoff (1s, 2s, 4s, ...).  HTTP 429 (rate-limit) and 5xx errors
      are retried; 401/403/400 are not retried (configuration errors).
    - If all retries are exhausted a TimeoutError or RuntimeError is raised
      so the caller (orchestrator skill timeout) can catch and escalate.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Non-retryable HTTP status codes
    _NO_RETRY_CODES = {400, 401, 403, 404, 422}

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        site_url: str = "",
        site_name: str = "INFRA-SKILL",
        default_timeout: int = 90,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set the OPENROUTER_API_KEY environment variable."
            )
        self.site_url = site_url
        self.site_name = site_name
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def call(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        """
        Make an LLM request and return a structured LLMResponse.

        Parameters
        ----------
        timeout : int, optional
            Per-request wall-clock timeout in seconds.  Defaults to
            self.default_timeout (90 s).  Pass a smaller value for
            time-sensitive calls (e.g. skill selector = 15 s).
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout

        payload = json.dumps({
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }).encode()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                self.BASE_URL, data=payload, headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                    raw = json.loads(resp.read())
                content = raw["choices"][0]["message"]["content"]
                return LLMResponse(content=content, raw=raw)

            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code == 401:
                    raise RuntimeError(
                        f"OpenRouter 401 Unauthorized: check OPENROUTER_API_KEY. {body[:300]}"
                    ) from e
                if e.code == 403:
                    raise RuntimeError(
                        f"OpenRouter 403 Forbidden: {body[:300]}"
                    ) from e
                if e.code in self._NO_RETRY_CODES:
                    raise RuntimeError(
                        f"OpenRouter HTTP {e.code} (not retryable): {body[:300]}"
                    ) from e
                # 429, 500, 502, 503 — retry
                last_exc = RuntimeError(
                    f"OpenRouter HTTP {e.code} (attempt {attempt}/{self.max_retries}): "
                    f"{body[:200]}"
                )

            except TimeoutError as e:
                last_exc = TimeoutError(
                    f"OpenRouter request timed out after {effective_timeout}s "
                    f"(attempt {attempt}/{self.max_retries})"
                )

            except OSError as e:
                # socket.timeout subclasses OSError on some Python versions
                if "timed out" in str(e).lower():
                    last_exc = TimeoutError(
                        f"OpenRouter socket timed out after {effective_timeout}s "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                else:
                    last_exc = RuntimeError(
                        f"OpenRouter network error (attempt {attempt}): {e}"
                    )

            if attempt < self.max_retries:
                wait = self.retry_backoff_base * (2 ** (attempt - 1))  # 1, 2, 4 ...
                time.sleep(wait)

        raise last_exc

    # ------------------------------------------------------------------
    # Legacy convenience wrapper
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
    ) -> str:
        """Backward-compatible wrapper: returns only the text content."""
        return self.call(
            system=system, user=user,
            temperature=temperature, timeout=timeout,
        ).content
