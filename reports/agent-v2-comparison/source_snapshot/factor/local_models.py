"""Explicit local-only inference transport. No model management or cloud fallback."""
from __future__ import annotations

import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, ProxyHandler, build_opener

MAX_RESPONSE_BYTES = 2_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise LocalModelError("HTTP redirects are disabled for local model requests")


def urlopen(request, *, timeout):
    """Disable environment proxies and every redirect, including loopback hops."""
    return build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


class LocalModelError(RuntimeError):
    pass


class LocalChatClient:
    """Small client for a running OpenAI-compatible loopback model server.

    complete() returns a dictionary with text, raw_response, elapsed_seconds,
    model, usage, finish_reason and request_settings. Raw model output is retained;
    this transport does not interpret a scientific answer or execute tool calls.
    """

    def __init__(self, base_url="http://127.0.0.1:1234/v1",
                 model="factor-qwen35-9b", timeout=180, transport="openai"):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("LocalChatClient only accepts an HTTP loopback endpoint")
        if not model or not isinstance(model, str):
            raise ValueError("An explicit model identifier is required")
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if transport not in {"openai", "lmstudio-native"}:
            raise ValueError("transport must be openai or lmstudio-native")
        self.transport = transport

    def complete(self, messages, max_tokens=256, temperature=0.0, *,
                 response_format=None, seed=20260905, disable_thinking=True, timeout=None):
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not messages or any(m.get("role") not in {"system", "user", "assistant", "tool"}
                               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("messages must contain supported roles and text content")
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed, "stream": False}
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format is not None:
            payload["response_format"] = response_format
        endpoint = self.base_url + "/chat/completions"
        if self.transport == "lmstudio-native":
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            transcript = [m for m in messages if m["role"] != "system"]
            text = (transcript[0]["content"] if len(transcript) == 1 else
                    "\n\n".join(m["role"].upper()+":\n"+m["content"] for m in transcript))
            if response_format is not None:
                system += "\nReturn only an object satisfying this JSON schema: " + json.dumps(response_format.get("json_schema", {}).get("schema", {}))
            payload = {"model": self.model, "input": text, "system_prompt": system,
                       "reasoning": "off" if disable_thinking else "on", "temperature": temperature,
                       "max_output_tokens": max_tokens, "stream": False, "store": False}
            endpoint = "http://" + urlsplit(self.base_url).netloc + "/api/v1/chat"
        request = Request(endpoint,
                          json.dumps(payload, allow_nan=False).encode(),
                          {"Content-Type": "application/json"}, method="POST")
        start = time.monotonic()
        request_timeout = self.timeout if timeout is None else timeout
        if not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        try:
            with urlopen(request, timeout=request_timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES+1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise LocalModelError("Local model response exceeds the size limit")
                raw = json.loads(body)
        except HTTPError as error:
            raise LocalModelError(f"Local server HTTP {error.code}: {error.read(800).decode(errors='replace')}") from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise LocalModelError(f"Local model request failed: {error}") from error
        if not isinstance(raw, dict):
            raise LocalModelError("Local model response must be a JSON object")
        if self.transport == "lmstudio-native":
            output = raw.get("output", [])
            content = "\n".join(item["content"] for item in output
                                if item.get("type") == "message" and isinstance(item.get("content"), str))
            for item in output:
                if item.get("type") == "reasoning":
                    item["reasoning_present_redacted"] = bool(item.pop("content", None))
            stats = raw.get("stats", {})
            tokens = stats.get("total_output_tokens")
            # This endpoint does not expose a finish reason. Reject exhausted or
            # unknown budgets; explicitly distinguish this inference in metadata.
            finish = "stop" if isinstance(tokens, (int, float)) and tokens < max_tokens and content else "length_or_unknown"
            return {"text": content, "raw_response": raw, "elapsed_seconds": time.monotonic()-start,
                    "model": raw.get("model_instance_id", self.model), "usage": stats,
                    "finish_reason": finish, "finish_reason_source": "inferred_from_native_token_count",
                    "request_settings": {k:v for k,v in payload.items() if k not in {"input", "system_prompt"}},
                    "schema_enforcement": "prompt_only_requires_local_validation",
                    "transport": self.transport}
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("text content missing")
        except (KeyError, IndexError, TypeError) as error:
            raise LocalModelError("Local server did not return a text completion") from error
        # Keep public outputs and usage, not internal reasoning text.
        for item in raw.get("choices", []):
            message = item.get("message", {})
            reasoning = message.pop("reasoning_content", None)
            if reasoning is not None:
                message["reasoning_present_redacted"] = bool(reasoning)
        return {"text": content, "raw_response": raw,
                "elapsed_seconds": time.monotonic() - start,
                "model": raw.get("model", self.model), "usage": raw.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
                "request_settings": {k: v for k, v in payload.items() if k != "messages"}}
