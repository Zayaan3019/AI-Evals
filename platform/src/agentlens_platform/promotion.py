"""Trace-to-eval promotion.

Turning a real production trace into a permanent eval case. This is the feature
that makes the platform a loop rather than two disconnected tools: incidents
become regression tests, and the eval set is drawn from the traffic the system
actually sees instead of from what someone imagined it would see.

**The snapshot rule.** A promoted case copies everything it needs. It does not
hold a pointer into ClickHouse. Spans live under a 30-day TTL; a case is
permanent. If a case re-read its inputs at run time it would work for a month
and then silently start failing to find its own data -- and because the eval
would error rather than score, the failure would look like a broken evaluator
rather than expired storage. ``source_trace_id`` is kept for provenance only,
and is explicitly documented as possibly-dangling.

This is the direct, deliberate consequence of splitting storage across two
engines with no shared transaction (ARCHITECTURE.md section 1). The boundary
buys columnar analytics on spans; the price is paid here, once, explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import Dataset, EvalCase

logger = logging.getLogger(__name__)

#: Span operation names that carry the user-facing input/output of a trace.
#: A workflow span is preferred because it spans the whole interaction.
_ROOT_OPERATIONS = ("invoke_workflow", "invoke_agent", "chat")


@dataclass(frozen=True, slots=True)
class ExtractedCase:
    """What promotion pulled out of a trace, before it is persisted.

    Kept as a separate value from the ORM row so extraction can be tested and
    reasoned about without a database, and so a caller can inspect or amend a
    candidate before committing it.
    """

    external_id: str
    inputs: dict[str, Any]
    expected_output: str | None
    tags: dict[str, Any] = field(default_factory=dict)
    context: tuple[str, ...] = ()
    tools_called: tuple[str, ...] = ()
    source_trace_id: str | None = None
    source_span_id: str | None = None
    source_captured_at: datetime | None = None

    def to_case_inputs(self) -> dict[str, Any]:
        """The JSONB payload stored on the case.

        Context and tool calls live inside ``inputs`` rather than in separate
        columns because they are part of the frozen snapshot: a groundedness
        evaluator needs the retrieved documents as they were *at capture time*,
        not as a re-run of retrieval would produce them today.
        """
        payload = dict(self.inputs)
        if self.context:
            payload["context"] = list(self.context)
        if self.tools_called:
            payload["tools_called"] = list(self.tools_called)
        return payload


def _attribute(span: dict[str, Any], key: str) -> Any:
    """Read a span attribute whether it was promoted to a column or not."""
    attributes = span.get("attributes") or {}
    return attributes.get(key)


def _json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _message_text(raw: Any) -> str | None:
    """Flatten a GenAI messages attribute into plain text.

    The convention stores messages as a JSON array of ``{role, content}``. We
    keep only the content, joined, because an evaluator scores what was said,
    not the envelope.
    """
    parsed = _json_maybe(raw)
    if parsed is None:
        return None
    if isinstance(parsed, str):
        return parsed or None
    if isinstance(parsed, dict):
        content = parsed.get("content")
        return str(content) if content is not None else None
    if isinstance(parsed, list):
        parts: list[str] = []
        for entry in parsed:
            if isinstance(entry, dict):
                content = entry.get("content")
                if content is not None:
                    parts.append(str(content))
            elif entry is not None:
                parts.append(str(entry))
        return "\n".join(parts) or None
    return str(parsed)


def _stable_external_id(trace_id: str, span_id: str) -> str:
    """Deterministic id for a promoted case.

    Derived from the source identifiers so promoting the same span twice
    produces the same ``external_id`` and is caught by the dataset's uniqueness
    constraint. Idempotent promotion matters because the obvious way to build a
    dataset is to re-run a query and promote its results, repeatedly.
    """
    digest = hashlib.sha256(f"{trace_id}:{span_id}".encode("utf-8")).hexdigest()
    return f"trace-{digest[:16]}"


def extract_case_from_trace(
    spans: Sequence[dict[str, Any]],
    *,
    external_id: str | None = None,
    include_context: bool = True,
) -> ExtractedCase:
    """Build a case candidate from one trace's spans.

    Chooses the most representative span as the source of input and output:
    the workflow root if present, otherwise the outermost agent, otherwise the
    first model call. Retrieved documents and tool names are collected across
    the whole trace, since they are what distinguishes evaluating an *agent*
    from evaluating a single completion.
    """
    if not spans:
        raise ValueError("cannot promote an empty trace")

    def rank(span: dict[str, Any]) -> tuple[Any, ...]:
        operation = span.get("gen_ai_operation_name")
        return (
            _ROOT_OPERATIONS.index(operation)
            if operation in _ROOT_OPERATIONS
            else len(_ROOT_OPERATIONS),
            # Prefer an actual root, then the earliest span.
            0 if not span.get("parent_span_id") else 1,
            span.get("start_time") or datetime.max.replace(tzinfo=timezone.utc),
        )

    ranked = sorted(spans, key=rank)
    # The outermost span defines provenance -- which trace this came from and
    # when -- even if it carries no content of its own.
    primary = ranked[0]

    # Content is taken from the outermost span that actually *has* content,
    # which is frequently not the outermost span. A workflow wrapper often
    # records only structure while the input and output live on the model call
    # inside it. Ranking purely by nesting produced empty cases from real
    # traces: every field was present in ClickHouse and none of it reached the
    # eval case.
    question = next(
        (
            text
            for span in ranked
            if (text := _message_text(_attribute(span, "gen_ai.input.messages"))) is not None
        ),
        None,
    )
    answer = next(
        (
            text
            for span in ranked
            if (text := _message_text(_attribute(span, "gen_ai.output.messages"))) is not None
        ),
        None,
    )

    context: list[str] = []
    if include_context:
        for span in spans:
            documents = _json_maybe(_attribute(span, "gen_ai.retrieval.documents"))
            if isinstance(documents, list):
                for document in documents:
                    if isinstance(document, dict):
                        content = document.get("content") or document.get("text")
                        if content:
                            context.append(str(content))
                    elif document:
                        context.append(str(document))

    tools = tuple(
        sorted(
            {
                span["gen_ai_tool_name"]
                for span in spans
                if span.get("gen_ai_tool_name")
            }
        )
    )

    trace_id = str(primary.get("trace_id") or "")
    span_id = str(primary.get("span_id") or "")

    models = sorted({s["gen_ai_request_model"] for s in spans if s.get("gen_ai_request_model")})
    total_cost = sum(s.get("cost_micro_usd") or 0 for s in spans)
    errored = any(s.get("status_code") == "error" for s in spans)

    return ExtractedCase(
        external_id=external_id or _stable_external_id(trace_id, span_id),
        inputs={"question": question} if question is not None else {},
        # The production answer becomes the *reference* only if a human later
        # confirms it. Stored as expected_output so it is visible for review,
        # and tagged so nobody mistakes it for verified ground truth.
        expected_output=answer,
        tags={
            "source": "production_trace",
            "reference_reviewed": False,
            "service": primary.get("service_name"),
            "models": models,
            "trace_cost_micro_usd": total_cost,
            "trace_had_error": errored,
            "span_count": len(spans),
        },
        context=tuple(context),
        tools_called=tools,
        source_trace_id=trace_id or None,
        source_span_id=span_id or None,
        source_captured_at=primary.get("start_time"),
    )


async def get_or_create_dataset(
    session: AsyncSession, *, tenant_id: uuid.UUID, slug: str, name: str | None = None
) -> Dataset:
    """Fetch a dataset by slug, creating it if absent."""
    existing = (
        await session.execute(
            select(Dataset).where(Dataset.tenant_id == tenant_id, Dataset.slug == slug)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    dataset = Dataset(tenant_id=tenant_id, slug=slug, name=name or slug)
    session.add(dataset)
    await session.flush()
    return dataset


@dataclass(frozen=True, slots=True)
class PromotionResult:
    created: tuple[str, ...]
    skipped_existing: tuple[str, ...]

    @property
    def created_count(self) -> int:
        return len(self.created)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_existing)


async def promote_cases(
    session: AsyncSession,
    candidates: Iterable[ExtractedCase],
    *,
    tenant_id: uuid.UUID,
    dataset: Dataset,
) -> PromotionResult:
    """Persist case candidates, skipping ones already in the dataset.

    Idempotent by ``(dataset_id, external_id)``. Existing rows are skipped
    rather than updated: a case that has already been used to produce results
    must not change underneath them, or every historical comparison against it
    becomes meaningless.
    """
    candidates = list(candidates)
    if not candidates:
        return PromotionResult((), ())

    wanted = [candidate.external_id for candidate in candidates]
    present = set(
        (
            await session.execute(
                select(EvalCase.external_id).where(
                    EvalCase.dataset_id == dataset.id, EvalCase.external_id.in_(wanted)
                )
            )
        )
        .scalars()
        .all()
    )

    created: list[str] = []
    skipped: list[str] = []
    for candidate in candidates:
        if candidate.external_id in present:
            skipped.append(candidate.external_id)
            continue
        session.add(
            EvalCase(
                dataset_id=dataset.id,
                tenant_id=tenant_id,
                external_id=candidate.external_id,
                inputs=candidate.to_case_inputs(),
                expected_output=candidate.expected_output,
                tags=candidate.tags,
                source_trace_id=candidate.source_trace_id,
                source_span_id=candidate.source_span_id,
                source_captured_at=candidate.source_captured_at,
            )
        )
        created.append(candidate.external_id)
        # Guard against a batch containing the same trace twice.
        present.add(candidate.external_id)

    await session.flush()
    logger.info(
        "promoted %d cases into dataset %s (%d already present)",
        len(created),
        dataset.slug,
        len(skipped),
    )
    return PromotionResult(tuple(created), tuple(skipped))
