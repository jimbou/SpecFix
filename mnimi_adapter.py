import json
import time
from typing import Any, Dict, Optional

from cached_llm import _BaseBufferedModel


class OpenAIChatModelAdapter(_BaseBufferedModel):
    """Wrap the existing OpenAI python client with Mnimi's sampling protocol."""

    def __init__(self, client, model_name: str, temperature: float, alias: Optional[str] = None, max_batch: int = 20):
        super().__init__(model_name=model_name, temperature=temperature, alias=alias, max_batch=max_batch)
        self._client = client
        self._total_token_count = (0, 0)
        self._total_query_time = 0.0

    def _query(self, payload: str, n: int):
        request = json.loads(payload)
        instruction = request.get("system", "")
        prompt = request.get("user", "")
        settings: Dict[str, Any] = request.get("settings") or {}

        params: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            "model": self.model_name,
            "n": n,
        }

        params.update({k: v for k, v in settings.items() if v is not None})

        start = time.perf_counter()
        completion = self._client.chat.completions.create(**params)
        self._total_query_time += time.perf_counter() - start

        usage = getattr(completion, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        else:
            prompt_tokens = 0
            completion_tokens = 0
        previous_prompt_tokens, previous_completion_tokens = self._total_token_count
        self._total_token_count = (
            previous_prompt_tokens + prompt_tokens,
            previous_completion_tokens + completion_tokens,
        )

        # For batched sampling we return exactly n choices; for single calls we still respect n.
        responses = []
        for choice in completion.choices[:n]:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", "") if message is not None else ""
            responses.append(content if content is not None else "")
        return responses

    def total_query_time(self) -> float:
        return self._total_query_time

    def total_token_count(self):
        return self._total_token_count