"""Backends for model-graded evaluation.

Three real providers, no stubs:

* **Anthropic** over the public Messages API, used when ``ANTHROPIC_API_KEY`` is
  present in the environment. The key is never read from config or stored in the
  database -- a judge configuration that carried a credential would end up in
  ``config_hash`` and, from there, into query results.
* **Ollama** over its local HTTP API, which needs no credential and is what the
  dogfooding runs on this machine use.
* **Groq** over its OpenAI-compatible chat completions API, used when
  ``GROQ_API_KEY`` is present.

A judge can also be wrapped in a :class:`ReplayJudge`, which records each real
response to a cassette file and replays it afterwards. That is not a stub: the
recorded text is what the real model said, the prompt is part of the key, and a
prompt that was never recorded fails as "judge unavailable" rather than being
answered by something else. It exists so a demo or a CI run can reuse real
judgements without holding a model credential.

A judge call is a network call to a non-deterministic system, so everything here
is built around that being true: bounded retries, an explicit timeout, and
failures surfaced as failures rather than as a zero score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
#: Pinned. The Messages API is versioned by date and an unpinned client can
#: change behaviour under you without a code change.
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    """What a judge backend returned, plus what it cost."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    model: str = ""


class JudgeUnavailable(RuntimeError):
    """The backend could not be reached or is not configured.

    Distinct from a bad response: this means we never got a judgement, so the
    result must be recorded as an evaluator error, not as a low score.
    """


class JudgeBackend(Protocol):
    """Minimal contract a judge backend must satisfy."""

    model: str

    def complete(self, prompt: str, *, system: str, max_tokens: int, temperature: float) -> JudgeResponse:
        ...


class AnthropicJudge:
    """Real Anthropic Messages API client.

    Only constructed when a key is actually present; :func:`build_judge` decides.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeUnavailable("ANTHROPIC_API_KEY is not set")
        self._key = key
        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(
        self, prompt: str, *, system: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> JudgeResponse:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    ANTHROPIC_API_URL, json=payload, headers=headers, timeout=self._timeout
                )
                if response.status_code in (429, 500, 502, 503, 529):
                    # Retryable. Anything else is a permanent error and retrying
                    # it just burns quota.
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                text = "".join(
                    block.get("text", "") for block in body.get("content", [])
                    if block.get("type") == "text"
                )
                usage = body.get("usage", {})
                return JudgeResponse(
                    text=text,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=body.get("model", self.model),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self._max_retries:
                    # Exponential backoff. Judges are called in bulk, so a
                    # tight retry loop against a rate limit makes it worse.
                    time.sleep(2**attempt)
        raise JudgeUnavailable(f"anthropic judge failed: {last_error}") from last_error


class OllamaJudge:
    """Real local Ollama client.

    Uses ``/api/chat`` with streaming disabled. Token counts come from Ollama's
    own ``prompt_eval_count`` / ``eval_count``, so they are measured rather than
    estimated.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        *,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._base_url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
        self._timeout = timeout

    def complete(
        self, prompt: str, *, system: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> JudgeResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailable(f"ollama judge failed: {exc}") from exc

        return JudgeResponse(
            text=body.get("message", {}).get("content", ""),
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=body.get("model", self.model),
        )


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _default_reasoning(model: str) -> str | None:
    """Lowest reasoning setting each model family accepts on Groq."""
    if model.startswith("openai/gpt-oss"):
        return "low"
    if model.startswith("qwen/qwen3"):
        return "none"
    return None


class GroqJudge:
    """Judge backed by Groq's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 6,
        base_url: str = GROQ_API_URL,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_url = base_url
        # Reasoning models spend output tokens thinking before they answer. With
        # a judge-sized max_tokens that can leave the visible answer empty, and
        # it multiplies cost. The default keeps reasoning minimal per family.
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else _default_reasoning(model)
        )

    def complete(
        self, prompt: str, *, system: str, max_tokens: int, temperature: float
    ) -> JudgeResponse:
        if not self._api_key:
            raise JudgeUnavailable("GROQ_API_KEY is not set")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    self._base_url, json=payload, headers=headers, timeout=self._timeout
                )
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(2.0**attempt, 20.0))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                # Honour the back-off the server asks for when it gives one. A
                # guessed delay shorter than Retry-After only burns the retry budget.
                last_error = RuntimeError(f"HTTP {response.status_code}")
                try:
                    delay = float(response.headers.get("retry-after", ""))
                except ValueError:
                    delay = 2.0**attempt
                time.sleep(min(max(delay, 1.0), 30.0))
                continue
            if response.status_code != 200:
                raise JudgeUnavailable(
                    f"groq returned HTTP {response.status_code}: {response.text[:200]}"
                )

            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise JudgeUnavailable("groq returned no choices")
            usage = body.get("usage") or {}
            return JudgeResponse(
                text=str(choices[0].get("message", {}).get("content") or ""),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_ms=(time.perf_counter() - started) * 1000,
                model=str(body.get("model") or self.model),
            )

        raise JudgeUnavailable(f"groq unreachable after retries: {last_error}")


class ReplayJudge:
    """Record a real judge once, then replay its responses.

    The cassette key is a hash of everything that determines the response:
    model, system prompt, user prompt, token limit and temperature. A changed
    prompt is therefore a cache miss, never a stale hit -- the same property
    ``config_hash`` protects elsewhere.

    On a miss the wrapped backend is called if one is available, and its answer
    is appended to the cassette. With no backend, a miss raises
    :class:`JudgeUnavailable`, which the evaluator records as an error rather
    than a score.
    """

    def __init__(self, model: str, cassette_path: str, live: JudgeBackend | None = None) -> None:
        self.model = model
        self._path = cassette_path
        self._live = live
        self._entries: dict[str, dict[str, Any]] = {}
        if os.path.exists(cassette_path):
            with open(cassette_path, encoding="utf-8") as handle:
                self._entries = json.load(handle).get("entries", {})

    def _key(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        material = json.dumps(
            [self.model, system, prompt, max_tokens, temperature], ensure_ascii=False
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def complete(
        self, prompt: str, *, system: str, max_tokens: int, temperature: float
    ) -> JudgeResponse:
        key = self._key(prompt, system, max_tokens, temperature)
        hit = self._entries.get(key)
        if hit is not None:
            return JudgeResponse(
                text=hit["text"],
                input_tokens=hit.get("input_tokens"),
                output_tokens=hit.get("output_tokens"),
                latency_ms=float(hit.get("latency_ms", 0.0)),
                model=hit.get("model", self.model),
            )
        if self._live is None:
            raise JudgeUnavailable(
                f"no recorded response for this prompt in {self._path} and no live judge configured"
            )
        response = self._live.complete(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )
        self._entries[key] = {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": round(response.latency_ms, 2),
            "model": response.model,
        }
        try:
            self._save()
        except OSError as exc:
            # The live response is valid whatever happens to the file. Losing
            # a grade because the cassette could not be written would turn a
            # recording problem into an evaluation error. The entry stays in
            # memory and is written with the next successful save.
            logger.warning("could not write judge cassette %s: %s", self._path, exc)
        return response

    def _save(self) -> None:
        # Re-read before writing: two judge models share one cassette file, and
        # each ReplayJudge instance only knows the entries it loaded or added.
        merged: dict[str, dict[str, Any]] = {}
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as handle:
                merged = json.load(handle).get("entries", {})
        merged.update(self._entries)
        self._entries = merged
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "note": "Real judge responses, recorded once and replayed. Keyed by a "
                    "hash of model, system prompt, prompt, max_tokens and temperature.",
                    "entries": self._entries,
                },
                handle,
                ensure_ascii=False,
                indent=1,
                sort_keys=True,
            )
        # On Windows a reader (an editor, an indexer, antivirus) can hold the
        # target open for a moment and os.replace then fails with a sharing
        # violation. Retry briefly before giving up.
        for attempt in range(5):
            try:
                os.replace(tmp, self._path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))


def ollama_available(base_url: str | None = None) -> bool:
    """Whether a local Ollama is reachable right now."""
    url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        return httpx.get(f"{url}/api/tags", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def build_judge(provider: str, model: str, **kwargs: Any) -> JudgeBackend:
    """Construct a judge backend by provider name.

    Raises :class:`JudgeUnavailable` rather than falling back to a different
    provider. A silent fallback would change what the scores mean without
    changing ``config_hash``, which is precisely the failure the hash exists to
    prevent.
    """
    if provider == "anthropic":
        backend: JudgeBackend = AnthropicJudge(model, **kwargs)
    elif provider == "ollama":
        backend = OllamaJudge(model, **kwargs)
    elif provider == "groq":
        backend = GroqJudge(model, **kwargs)
    else:
        raise JudgeUnavailable(f"unknown judge provider {provider!r}")

    # Optional record/replay. The provider and model stay exactly what the
    # evaluator config says they are, so ``config_hash`` still names the real
    # judge; the cassette only decides whether the network is used.
    cassette = os.environ.get("AGENTLENS_JUDGE_CASSETTE")
    if cassette:
        record = os.environ.get("AGENTLENS_JUDGE_RECORD", "").strip().lower() in {"1", "true", "yes"}
        return ReplayJudge(model, cassette, live=backend if record else None)
    return backend


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose and fences however they like, and a judge whose
    output cannot be parsed is a judge that produces no signal. Tries the whole
    string first, then the outermost brace-delimited span.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
