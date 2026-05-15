from __future__ import annotations

import os
import time
from typing import Any


class VLMConnector:
    """OpenAI-compatible vision-language model connector."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("VLM_MODEL", "gpt-4o")
        self.max_workers = max_workers
        self._client: Any = None  # Lazy init

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def call(self, image_b64: str, prompt: str) -> str:
        """Send an image + text prompt to the VLM and return the raw response text."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Cannot call VLM.")

        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise drone-detection image analyst. "
                                "Analyze images carefully and respond with only the requested JSON format."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "auto",
                                    },
                                },
                            ],
                        },
                    ],
                    max_completion_tokens=512,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                return content if content else ""
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                continue

        raise RuntimeError(f"VLM call failed after {max_retries} attempts: {last_error}")
