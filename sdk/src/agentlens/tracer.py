"""SDK lifecycle: build the provider, install it, tear it down.

Failure policy: constructing telemetry must never break the application. If the
exporter cannot be built (missing extra, bad endpoint, TLS misconfiguration) the
SDK logs once and degrades to a no-op provider rather than raising into the
caller's import or startup path.
"""

from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    ParentBased,
    TraceIdRatioBased,
)

from .config import AgentLensConfig
from .export import (
    BoundedQueueSpanProcessor,
    CountingSpanExporter,
    ExportStats,
    RedactingSpanExporter,
)
from .semconv import SDK_LANGUAGE, SDK_VERSION
from .version import __version__

logger = logging.getLogger(__name__)

INSTRUMENTATION_SCOPE = "agentlens"

#: Resource attribute keys. ``service.name`` and ``deployment.environment`` are
#: standard resource conventions; tenant/project are platform concepts that the
#: resource conventions do not model.
SERVICE_NAME_KEY = "service.name"
SERVICE_VERSION_KEY = "service.version"
DEPLOYMENT_ENVIRONMENT_KEY = "deployment.environment"
TENANT_KEY = "agentlens.tenant"
PROJECT_KEY = "agentlens.project"
REDACTION_ENABLED_KEY = "agentlens.redaction.enabled"
PRICING_VERSION_KEY = "agentlens.cost.pricing_version"


@dataclass
class AgentLens:
    """Handle to an initialised SDK instance."""

    config: AgentLensConfig
    provider: TracerProvider
    stats: ExportStats
    processor: BoundedQueueSpanProcessor | None
    #: Present only when ``exporter="memory"``; the in-process sink used by
    #: tests and benchmarks.
    memory_exporter: InMemorySpanExporter | None = None

    def tracer(self, name: str = INSTRUMENTATION_SCOPE):
        return self.provider.get_tracer(name, __version__)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Block until queued spans are exported or the timeout elapses.

        Intended for short-lived processes and tests. Never call this on a
        request path -- it is the one blocking operation in the SDK.
        """
        try:
            return self.provider.force_flush(timeout_millis)
        except Exception:
            logger.warning("agentlens: force_flush raised", exc_info=True)
            return False

    def shutdown(self, drain_timeout_millis: float = 5000) -> None:
        """Tear down, bounding how long the final drain may block.

        ``TracerProvider.shutdown()`` drains the processor backlog with no time
        limit of its own, so the bound is installed on the exporter first.
        """
        if self.processor is not None:
            try:
                self.processor.begin_shutdown(drain_timeout_millis)
            except Exception:
                logger.warning("agentlens: failed to set drain deadline", exc_info=True)
        try:
            self.provider.shutdown()
        except Exception:
            logger.warning("agentlens: shutdown raised", exc_info=True)


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.instance: AgentLens | None = None


_state = _State()


def _build_resource(config: AgentLensConfig) -> Resource:
    attributes: dict[str, object] = {
        SERVICE_NAME_KEY: config.service_name,
        SERVICE_VERSION_KEY: __version__,
        SDK_VERSION: __version__,
        SDK_LANGUAGE: "python",
        REDACTION_ENABLED_KEY: config.redaction_enabled,
        PRICING_VERSION_KEY: config.pricing_version,
    }
    if config.tenant:
        attributes[TENANT_KEY] = config.tenant
    if config.project:
        attributes[PROJECT_KEY] = config.project
    if config.environment:
        attributes[DEPLOYMENT_ENVIRONMENT_KEY] = config.environment
    return Resource.create(attributes)  # type: ignore[arg-type]


def _build_exporter(
    config: AgentLensConfig,
) -> tuple[SpanExporter | None, InMemorySpanExporter | None]:
    """Return ``(exporter, memory_exporter)``; ``(None, None)`` disables export."""
    if config.custom_exporter is not None:
        return config.custom_exporter, None
    if config.exporter == "none":
        return None, None
    if config.exporter == "console":
        return ConsoleSpanExporter(), None
    if config.exporter == "memory":
        memory = InMemorySpanExporter()
        return memory, memory

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        logger.error(
            "agentlens: OTLP exporter requested but "
            "opentelemetry-exporter-otlp-proto-grpc is not installed. "
            "Install agentlens-sdk[otlp]. Tracing is disabled; the application "
            "is unaffected."
        )
        return None, None

    try:
        exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=tuple(config.resolved_headers().items()) or None,
            insecure=bool(config.insecure),
            timeout=max(1, int(config.export_timeout_millis / 1000)),
        )
    except Exception:
        logger.error(
            "agentlens: failed to construct OTLP exporter for endpoint %r; "
            "tracing is disabled. The application is unaffected.",
            config.endpoint,
            exc_info=True,
        )
        return None, None
    return exporter, None


def init(config: AgentLensConfig | None = None, **overrides: object) -> AgentLens:
    """Initialise the SDK and return a handle.

    Idempotent: a second call returns the existing instance rather than
    installing a second exporter pipeline. Call :func:`shutdown` first to
    reconfigure.
    """
    if config is None:
        config = AgentLensConfig.from_env(**overrides)
    elif overrides:
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"unknown config option {key!r}")
            setattr(config, key, value)
        config.__post_init__()

    with _state.lock:
        if _state.instance is not None:
            logger.debug("agentlens: already initialised; returning existing instance")
            return _state.instance

        stats = ExportStats()

        if config.disabled:
            # A provider with no processors: spans are created and immediately
            # discarded, so instrumented code paths still run and still cost
            # essentially nothing.
            provider = TracerProvider(
                resource=_build_resource(config), sampler=ParentBased(ALWAYS_ON)
            )
            instance = AgentLens(
                config=config, provider=provider, stats=stats, processor=None
            )
            _state.instance = instance
            return instance

        sampler = (
            ParentBased(ALWAYS_ON)
            if config.sample_ratio >= 1.0
            else ParentBased(TraceIdRatioBased(config.sample_ratio))
        )
        provider = TracerProvider(resource=_build_resource(config), sampler=sampler)

        exporter, memory = _build_exporter(config)
        processor: BoundedQueueSpanProcessor | None = None

        if exporter is not None:
            if config.redaction_enabled:
                exporter = RedactingSpanExporter(exporter, config.redactor)

            if config.exporter == "memory":
                # Deterministic, no background thread: tests assert on exact
                # span sets and must not race the batch timer. Still wrapped in
                # CountingSpanExporter so ExportStats reports real numbers here
                # too -- a stats block that silently reads zero in one mode is
                # worse than no stats block at all.
                provider.add_span_processor(
                    SimpleSpanProcessor(CountingSpanExporter(exporter, stats))
                )
            else:
                processor = BoundedQueueSpanProcessor(
                    exporter,
                    stats=stats,
                    max_queue_size=config.max_queue_size,
                    max_export_batch_size=config.max_export_batch_size,
                    schedule_delay_millis=config.schedule_delay_millis,
                    export_timeout_millis=config.export_timeout_millis,
                )
                provider.add_span_processor(processor)

        if config.set_global_tracer_provider:
            _set_global_provider(provider)

        instance = AgentLens(
            config=config,
            provider=provider,
            stats=stats,
            processor=processor,
            memory_exporter=memory,
        )
        _state.instance = instance

    atexit.register(shutdown)
    return instance


def _set_global_provider(provider: TracerProvider) -> None:
    """Install as global provider, tolerating a provider already being set.

    OpenTelemetry logs and ignores a second ``set_tracer_provider`` call. We
    detect that case and say something actionable instead of leaving the user
    wondering why their spans are missing.
    """
    existing = otel_trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        logger.warning(
            "agentlens: a global TracerProvider is already installed; agentlens "
            "spans will be exported through the existing provider only if it is "
            "configured to reach the collector. Pass "
            "set_global_tracer_provider=False and use the returned handle's "
            "tracer() to avoid ambiguity."
        )
        return
    otel_trace.set_tracer_provider(provider)


def get_instance() -> AgentLens | None:
    """Return the initialised handle, or ``None`` if :func:`init` has not run."""
    with _state.lock:
        return _state.instance


def get_tracer(name: str = INSTRUMENTATION_SCOPE):
    """Return a tracer, falling back to the global provider when uninitialised.

    Instrumented code therefore works whether or not the host application chose
    to call :func:`init` -- it simply produces no exported spans.
    """
    instance = get_instance()
    if instance is not None:
        return instance.tracer(name)
    return otel_trace.get_tracer(name, __version__)


def shutdown(timeout_millis: int = 5000) -> None:
    """Flush and tear down. Safe to call more than once.

    Bounded by ``timeout_millis`` so an unreachable collector cannot stall
    process exit.
    """
    with _state.lock:
        instance = _state.instance
        _state.instance = None
    if instance is None:
        return
    # Arm the deadline BEFORE flushing. force_flush also walks the whole
    # backlog, so installing the bound only for the drain phase leaves the
    # flush unbounded -- which is where teardown actually stalls when the
    # collector is unreachable.
    if instance.processor is not None:
        try:
            instance.processor.begin_shutdown(timeout_millis)
        except Exception:
            logger.warning("agentlens: failed to arm drain deadline", exc_info=True)
    instance.force_flush(timeout_millis)
    instance.shutdown(drain_timeout_millis=timeout_millis)
