"""Postgres relational state.

What lives here and why: this side of the boundary holds everything that needs
a transaction, a foreign key, an update in place, or a uniqueness guarantee --
none of which ClickHouse provides. Spans do not appear here at all; they are
referenced by identifier only (``trace_id`` / ``span_id`` as plain strings),
because there is no cross-store transaction and pretending otherwise would
produce dangling joins.

The consequence, stated in ARCHITECTURE.md section 1 and enforced by
:class:`EvalCase`, is that an eval case **snapshots** the inputs it was promoted
from. It cannot re-read them from ClickHouse later: spans expire on a TTL and
the case must outlive them.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


class Tenant(Base, TimestampMixin):
    """An isolation boundary. Every other row in the schema hangs off one.

    ``slug`` is the identifier that travels on spans as ``tenant_id`` into
    ClickHouse, so it is immutable in practice: changing it would orphan the
    tenant's existing span data, which lives in a store with no foreign keys.
    """

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("slug ~ '^[a-z0-9][a-z0-9_-]*$'", name="ck_tenant_slug_format"),
    )


class ApiKey(Base, TimestampMixin):
    """A credential for one tenant.

    Only the hash is stored. ``prefix`` is the first few characters of the
    plaintext, kept so an operator can identify a key in a list without the
    system ever being able to reproduce it.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    #: SHA-256 of the plaintext key. Unique so lookup is a single indexed read.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


# ---------------------------------------------------------------------------
# Datasets and eval cases
# ---------------------------------------------------------------------------


class Dataset(Base, TimestampMixin):
    """A named, versioned collection of eval cases."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    cases: Mapped[list["EvalCase"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_dataset_tenant_slug"),
    )


class EvalCase(Base, TimestampMixin):
    """One frozen input, optionally with a reference output.

    ``inputs`` is a full snapshot, not a pointer. A case promoted from a
    production trace records ``source_trace_id`` for provenance, but the trace
    itself is on a 30-day TTL in ClickHouse while the case is permanent -- so
    the case must be self-sufficient. This is the direct, deliberate consequence
    of splitting storage across two engines with no shared transaction.
    """

    __tablename__ = "eval_cases"

    id: Mapped[uuid.UUID] = _uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Stable identifier within the dataset. Used to pair results across runs,
    #: which is what makes the paired bootstrap in Phase 5 valid.
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expected_output: Mapped[str | None] = mapped_column(Text)
    #: Free-form labels for slicing results (difficulty, language, customer...).
    tags: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # -- provenance, when promoted from a real trace ----------------------
    source_trace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    source_span_id: Mapped[str | None] = mapped_column(String(16))
    #: When the source trace ran. Retained after the span itself expires.
    source_captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dataset: Mapped[Dataset] = relationship(back_populates="cases")

    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_case_dataset_external"),
    )


# ---------------------------------------------------------------------------
# Evaluators and runs
# ---------------------------------------------------------------------------


class EvaluatorRecord(Base, TimestampMixin):
    """A registered, versioned scoring function, as persisted.

    Named ``EvaluatorRecord`` rather than ``Evaluator`` because
    ``evaluators.base.Evaluator`` is the executable class and the two are
    imported together from Phase 4 onward. One name for two different things is
    how the wrong one gets used.

    ``config_hash`` covers the settings that change what a score *means* (judge
    model, prompt, threshold). Two runs may only be compared when this matches;
    the regression gate in Phase 5 enforces that, because comparing scores from
    different judges is the most common way to produce a confident wrong answer.
    """

    __tablename__ = "evaluators"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    #: One of the four tiers. Validated in the registry (Phase 3), and
    #: constrained here so a bad write cannot enter the table out of band.
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        # Uniqueness includes config_hash on purpose. A registration *is* the
        # (slug, version, config) triple: two different configurations are two
        # different evaluators even at the same version, because their scores
        # are not on the same scale. Keying on (slug, version) alone meant a
        # changed config reused the existing row, so the stored hash was the
        # first one ever seen rather than the one that produced the scores --
        # which made the gate's drift check impossible to trigger.
        UniqueConstraint(
            "tenant_id", "slug", "version", "config_hash",
            name="uq_evaluator_tenant_slug_version_config",
        ),
        CheckConstraint(
            "tier IN ('deterministic', 'statistical', 'model_graded', 'human')",
            name="ck_evaluator_tier",
        ),
    )


class EvalRun(Base, TimestampMixin):
    """One execution of a dataset against a system under test."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    #: Identifies the thing being tested: a git sha, a prompt version, a model.
    #: Free-form because what varies differs per team.
    system_label: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled')",
            name="ck_eval_run_status",
        ),
        Index("ix_eval_runs_tenant_started", "tenant_id", "started_at"),
    )


class EvalResult(Base, TimestampMixin):
    """One (case, evaluator) score within a run.

    The uniqueness constraint is what makes results *paired*: for a given run,
    case and evaluator there is exactly one score, so two runs over the same
    dataset can be joined case-by-case. An unpaired comparison would throw away
    the per-case correlation that makes the bootstrap test powerful.
    """

    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    evaluator_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluators.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    score: Mapped[float | None] = mapped_column(Float)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    #: Why the score is what it is. For a judge, its reasoning; for a
    #: deterministic check, the diff. Never null for a failure.
    explanation: Mapped[str | None] = mapped_column(Text)
    #: Populated when the evaluator itself failed, as distinct from the system
    #: under test scoring badly. Conflating the two corrupts the pass rate.
    error: Mapped[str | None] = mapped_column(Text)

    actual_output: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    cost_micro_usd: Mapped[int | None] = mapped_column(BigInteger)
    #: Links back to the trace produced while running this case, when the system
    #: under test was instrumented. Identifier only -- no foreign key exists
    #: across the store boundary.
    trace_id: Mapped[str | None] = mapped_column(String(32), index=True)

    run: Mapped[EvalRun] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("run_id", "case_id", "evaluator_id", name="uq_result_run_case_evaluator"),
    )


class HumanLabel(Base, TimestampMixin):
    """Ground truth from a person, used to calibrate model-graded evaluators.

    Kept separate from :class:`EvalResult` on purpose: a human label is an
    opinion about a case, not the output of a run. Storing them together would
    make it easy to accidentally calibrate a judge against itself.
    """

    __tablename__ = "human_labels"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The output that was judged. Snapshotted for the same reason as
    #: EvalCase.inputs: the run that produced it may be long gone.
    graded_output: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[float] = mapped_column(Float, nullable=False)
    labeller: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("case_id", "labeller", "graded_output", name="uq_label_case_labeller_output"),
    )
