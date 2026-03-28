# llm_client.py
import os
import json
from typing import Optional
import urllib.request

class OpenRouterClient:
    """
    OpenAI-compatible client targeting OpenRouter.
    Injects into skills via SkillContext.config['llm'].
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
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.site_url = site_url
        self.site_name = site_name

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
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
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]