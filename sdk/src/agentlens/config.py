"""SDK configuration.

Every knob is settable in code and via environment variable, because the two
audiences differ: application authors configure in code, operators configure a
deployed container without a rebuild. Code wins over environment when both are
supplied, so a deployment cannot silently override an explicit decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from opentelemetry.sdk.trace.export import SpanExporter

from .pricing import DEFAULT_PRICING_VERSION, PricingTable, default_table
from .redaction import DEFAULT_REDACTOR, Redactor

ExporterKind = Literal["otlp", "console", "memory", "none"]

#: Header carrying the scoped API key to the Phase 2 OTLP receiver.
API_KEY_HEADER = "x-agentlens-api-key"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass
class AgentLensConfig:
    """Complete SDK configuration."""

    # -- identity -------------------------------------------------------
    service_name: str = "unknown-service"
    #: Tenant this process reports as. Carried as a resource attribute and used
    #: by the Phase 2 receiver for isolation.
    tenant: str | None = None
    project: str | None = None
    environment: str | None = None

    # -- transport ------------------------------------------------------
    exporter: ExporterKind = "otlp"
    #: Explicit exporter instance. Takes precedence over :attr:`exporter` and is
    #: how the benchmark harness and any custom sink attach without going
    #: through OTLP. Still wrapped in redaction and the bounded queue.
    custom_exporter: SpanExporter | None = None
    endpoint: str = "http://localhost:4317"
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    #: gRPC without TLS. Correct for a local collector; must be False in
    #: production, which is why it defaults to True only for localhost.
    insecure: bool | None = None

    # -- queueing / batching --------------------------------------------
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    schedule_delay_millis: float = 5000.0
    export_timeout_millis: float = 30_000.0

    # -- capture --------------------------------------------------------
    #: When False, prompts, completions, tool arguments/results and retrieved
    #: document text are omitted entirely. Token counts, latency, cost and
    #: structure are still captured, so a privacy-restricted deployment keeps
    #: every metric that does not require content.
    capture_content: bool = True
    #: Hard cap on any single captured text attribute, applied after redaction.
    max_attribute_length: int = 8192
    redactor: Redactor = DEFAULT_REDACTOR
    #: Set False only if the caller has an out-of-band guarantee that no PII can
    #: reach a span. Disabling is recorded on the resource so a trace never
    #: looks redacted when it was not.
    redaction_enabled: bool = True

    # -- sampling -------------------------------------------------------
    #: Head sampling ratio in [0, 1]. Applied parent-based, so a sampled trace
    #: is never partially captured.
    sample_ratio: float = 1.0

    # -- cost -----------------------------------------------------------
    pricing_version: str = DEFAULT_PRICING_VERSION
    pricing_table: PricingTable | None = None

    # -- global behaviour ------------------------------------------------
    #: Kill switch. When True the SDK installs nothing and every API call is a
    #: no-op, so instrumentation can be disabled without editing the app.
    disabled: bool = False
    #: Install as the global OpenTelemetry tracer provider. False when the host
    #: application already owns the global provider.
    set_global_tracer_provider: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_ratio <= 1.0:
            raise ValueError(
                f"sample_ratio must be in [0, 1], got {self.sample_ratio}"
            )
        if self.max_attribute_length <= 0:
            raise ValueError("max_attribute_length must be positive")
        if self.insecure is None:
            self.insecure = _is_local_endpoint(self.endpoint)

    def resolved_headers(self) -> dict[str, str]:
        """Export headers, including the API key when configured."""
        headers = dict(self.headers)
        if self.api_key:
            headers[API_KEY_HEADER] = self.api_key
        return headers

    def resolved_pricing(self) -> PricingTable:
        if self.pricing_table is not None:
            return self.pricing_table
        if self.pricing_version == DEFAULT_PRICING_VERSION:
            return default_table()
        return PricingTable.load(self.pricing_version)

    def content_redactor(self) -> Redactor | None:
        """Redactor to apply at capture time, or ``None`` when disabled."""
        if not self.redaction_enabled:
            return None
        return self.redactor.with_max_length(self.max_attribute_length)

    @classmethod
    def from_env(cls, **overrides: object) -> AgentLensConfig:
        """Build config from ``AGENTLENS_*`` env vars, then apply ``overrides``.

        Explicit keyword arguments always win, so code-level decisions are not
        silently overridden by the environment.
        """
        endpoint = (
            os.environ.get("AGENTLENS_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "http://localhost:4317"
        )
        exporter = os.environ.get("AGENTLENS_EXPORTER", "otlp").strip().lower()
        if exporter not in ("otlp", "console", "memory", "none"):
            raise ValueError(
                f"AGENTLENS_EXPORTER must be otlp|console|memory|none, got {exporter!r}"
            )

        insecure_raw = os.environ.get("AGENTLENS_INSECURE")
        base = cls(
            service_name=os.environ.get(
                "AGENTLENS_SERVICE_NAME",
                os.environ.get("OTEL_SERVICE_NAME", "unknown-service"),
            ),
            tenant=os.environ.get("AGENTLENS_TENANT"),
            project=os.environ.get("AGENTLENS_PROJECT"),
            environment=os.environ.get("AGENTLENS_ENVIRONMENT"),
            exporter=exporter,  # type: ignore[arg-type]
            endpoint=endpoint,
            api_key=os.environ.get("AGENTLENS_API_KEY"),
            insecure=(
                None
                if insecure_raw is None
                else insecure_raw.strip().lower() in _TRUTHY
            ),
            max_queue_size=_env_int("AGENTLENS_MAX_QUEUE_SIZE", 2048),
            max_export_batch_size=_env_int("AGENTLENS_MAX_EXPORT_BATCH_SIZE", 512),
            schedule_delay_millis=_env_float("AGENTLENS_SCHEDULE_DELAY_MS", 5000.0),
            export_timeout_millis=_env_float("AGENTLENS_EXPORT_TIMEOUT_MS", 30_000.0),
            capture_content=_env_bool("AGENTLENS_CAPTURE_CONTENT", True),
            max_attribute_length=_env_int("AGENTLENS_MAX_ATTRIBUTE_LENGTH", 8192),
            redaction_enabled=_env_bool("AGENTLENS_REDACTION_ENABLED", True),
            sample_ratio=_env_float("AGENTLENS_SAMPLE_RATIO", 1.0),
            pricing_version=os.environ.get(
                "AGENTLENS_PRICING_VERSION", DEFAULT_PRICING_VERSION
            ),
            disabled=_env_bool("AGENTLENS_DISABLED", False),
        )

        for key, value in overrides.items():
            if not hasattr(base, key):
                raise TypeError(f"unknown config option {key!r}")
            setattr(base, key, value)
        base.__post_init__()
        return base


def _is_local_endpoint(endpoint: str) -> bool:
    lowered = endpoint.lower()
    return any(
        host in lowered for host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    )
