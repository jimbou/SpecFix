import os
from openai import OpenAI
import time
import json
from typing import Optional

from openai import OpenAI

from cached_llm import Persistent, Independent, Repeatable
from mnimi_adapter import OpenAIChatModelAdapter

class Model:
    def __init__(self, model: str, temperature: Optional[float], cache_dir: Optional[str] = None, replication: bool = False):
        self.model_name = model
        self.client = self.model_setup()
        if temperature is not None:
            self.temperature = temperature

        self._live_prompt_tokens = 0
        self._live_completion_tokens = 0
        self._live_query_time = 0.0

        self._mnimi_enabled = cache_dir is not None
        self._mnimi_persistent = None
        self._mnimi_repeatable = None
        self._mnimi_independent = None
        if self._mnimi_enabled:
            base_temperature = temperature if temperature is not None else 1.0
            base = OpenAIChatModelAdapter(
                self.client,
                self.model_name,
                base_temperature,
                alias=self.model_name,
            )
            self._mnimi_persistent = Persistent(base, cache_dir, replication=replication)
            self._mnimi_repeatable = Repeatable(self._mnimi_persistent)
            self._mnimi_independent = Independent(self._mnimi_persistent)

    def model_setup(self):
        api_key = os.environ['LLM_API_KEY']
        if "qwen" in self.model_name:
            client = OpenAI(
                api_key=api_key,
                base_url="",
            )
        elif "deepseek" in self.model_name:
            client = OpenAI(
                api_key=api_key,
                base_url="",
            )
        elif "gpt" in self.model_name or "o1" in self.model_name or "o3" in self.model_name:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.302.ai/v1",
            )
        elif "llama" in self.model_name:
            client = OpenAI(
                api_key=api_key,
                base_url=""
            )
        else:
            raise ValueError("Invalid model")

        return client
    
    def _build_payload(self, instruction: str, prompt: str, use_model_settings, call_type: str) -> str:
        settings = None
        if use_model_settings is not None:
            if call_type == "sample":
                settings = {"temperature": self.temperature}
            else:
                settings = {
                    "temperature": 0,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                }
        return json.dumps({
            "system": instruction or "",
            "user": prompt,
            "settings": settings,
            "call_type": call_type,
        })

    def _select_cache_model(self, cache_mode: Optional[str]):
        if not self._mnimi_enabled:
            return None
        if cache_mode == "independent":
            return self._mnimi_independent
        if cache_mode == "repeatable":
            return self._mnimi_repeatable
        if cache_mode == "repeatable_attempt":
            return Repeatable(self._mnimi_independent)
        if cache_mode == "persistent" or cache_mode is None:
            return self._mnimi_persistent
        raise ValueError(f"Unknown cache mode: {cache_mode}")

    def _cached_sample(self, instruction, prompt, n, use_model_settings, cache_mode):
        payload = self._build_payload(instruction, prompt, use_model_settings, "sample")
        model = self._select_cache_model(cache_mode)
        iterator = model.sample(payload, batch=n)
        return [next(iterator) for _ in range(n)]

    def _cached_single(self, instruction, prompt, use_model_settings, cache_mode):
        payload = self._build_payload(instruction, prompt, use_model_settings, "single")
        model = self._select_cache_model(cache_mode)
        iterator = model.sample(payload, batch=1)
        return next(iterator)
    
    def _record_live_usage(self, completion, elapsed: float):
        usage = getattr(completion, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
        self._live_prompt_tokens += prompt_tokens
        self._live_completion_tokens += completion_tokens
        self._live_query_time += elapsed

    def get_response_sample(self, instruction, prompt, n=20, use_model_settings=None,
                            cache_mode: str = "persistent"):
        if self._mnimi_enabled:
            return self._cached_sample(instruction, prompt, n, use_model_settings, cache_mode)
        try:
            start = time.perf_counter()
            if use_model_settings is None:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    n=n
                )
                # responses = [chat_completion.choices[i].message.content for i in range(n)]
                # return responses
            else:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    n=n
                )
            self._record_live_usage(chat_completion, time.perf_counter() - start)
            responses = [chat_completion.choices[i].message.content for i in range(n)]
            return responses
        except Exception as e:
            print('[ERROR]', e)
            time.sleep(5)

    def get_response(self, instruction, prompt, use_model_settings=None, cache_mode: str = "persistent"):
        if self._mnimi_enabled:
            return self._cached_single(instruction, prompt, use_model_settings, cache_mode)
        try:
            start = time.perf_counter()
            if use_model_settings is None:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                )
            else:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0,
                    top_p=0.95,
                    frequency_penalty=0,
                )
            self._record_live_usage(chat_completion, time.perf_counter() - start)

            response = chat_completion.choices[0].message.content
            if response:
                return response
            else:
                return ""
        except Exception as e:
            print('[ERROR]', e)
            time.sleep(5)

    def get_usage_stats(self):
        """Return aggregated provider usage for the current process.

        The totals report prompt tokens, completion tokens, and API time. When the
        Mnimi cache is enabled these counters exclude cache hits because the
        underlying adapter only records metrics for real model requests."""
        if self._mnimi_enabled and self._mnimi_persistent is not None:
            prompt_tokens, completion_tokens = self._mnimi_persistent.total_token_count()
            total_time = self._mnimi_persistent.total_query_time()
        else:
            prompt_tokens = self._live_prompt_tokens
            completion_tokens = self._live_completion_tokens
            total_time = self._live_query_time
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "api_time_seconds": total_time,
        }