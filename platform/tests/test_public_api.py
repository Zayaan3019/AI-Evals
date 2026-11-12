"""Public read access and the evaluation read APIs, against real Postgres.

The auth tests pin the property that makes a public deployment safe: anonymous
access exists only when configured, only for one tenant, and only for reads.
The evals tests check that the HTTP layer returns what the gate and the
calibration code compute, including refusals.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from fastapi import FastAPI

from agentlens_platform.api import deps
from agentlens_platform.api.app import app as full_app
from agentlens_platform.api.evals import _GATE_CACHE, router as evals_router
from agentlens_platform.config import Settings
from agentlens_platform.db.models import Dataset, HumanLabel, Tenant
from agentlens_platform.db.postgres import dispose_engine, session_scope
from agentlens_platform.evaluators import build_evaluator
from agentlens_platform.evaluators.judges import ReplayJudge
from agentlens_platform.promotion import ExtractedCase, get_or_create_dataset, promote_cases
from agentlens_platform.runner import run_eval
from agentlens_platform.tenancy import tenant_session


def _client(public_tenant: str | None, monkeypatch) -> httpx.AsyncClient:
    monkeypatch.setattr(deps, "get_settings", lambda: Settings(public_read_tenant=public_tenant))
    app = FastAPI()
    app.include_router(evals_router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _tenant(slug: str | None = None) -> tuple[uuid.UUID, str]:
    slug = slug or f"p{uuid.uuid4().hex[:14]}"
    async with session_scope() as session:
        row = Tenant(slug=slug, name="public api test")
        session.add(row)
        await session.flush()
        return row.id, slug


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)
    await dispose_engine()


async def _dataset(tenant_id, n: int) -> uuid.UUID:
    cases = [
        ExtractedCase(external_id=f"c{i:03d}", inputs={"q": f"q{i}"}, expected_output=f"answer {i}")
        for i in range(n)
    ]
    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
        await promote_cases(session, cases, tenant_id=tenant_id, dataset=dataset)
        return dataset.id


async def _run(tenant_id, dataset_id, system, label, evaluators):
    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        run = await run_eval(
            session, tenant_id=tenant_id, dataset=dataset, system=system,
            evaluators=evaluators, system_label=label,
        )
        return run.id


# -- auth -------------------------------------------------------------------


async def test_no_key_is_rejected_when_no_public_tenant_is_configured(pg_engine, monkeypatch):
    async with _client(None, monkeypatch) as client:
        assert (await client.get("/v1/evals/runs")).status_code == 401


async def test_no_key_reads_the_configured_public_tenant_only(pg_engine, monkeypatch):
    public_id, public_slug = await _tenant()
    other_id, _ = await _tenant()
    try:
        await _dataset(public_id, 3)
        public_ds = await _dataset(public_id, 3)
        other_ds = await _dataset(other_id, 3)
        await _run(public_id, public_ds, lambda i: "x", "public-run", [build_evaluator("exact_match")])
        await _run(other_id, other_ds, lambda i: "x", "other-run", [build_evaluator("exact_match")])

        async with _client(public_slug, monkeypatch) as client:
            body = (await client.get("/v1/evals/runs")).json()
        labels = {run["system_label"] for run in body["runs"]}
        assert labels == {"public-run"}, "anonymous reads must be scoped to the public tenant by RLS"
    finally:
        await _cleanup(public_id)
        await _cleanup(other_id)


async def test_a_wrong_key_is_rejected_even_when_a_public_tenant_exists(pg_engine, monkeypatch, tenant):
    public_slug, _ = tenant
    async with _client(public_slug, monkeypatch) as client:
        response = await client.get("/v1/evals/runs", headers={"x-agentlens-api-key": "al_wrong"})
    assert response.status_code == 401


async def test_an_inactive_public_tenant_is_not_served(pg_engine, monkeypatch):
    tenant_id, slug = await _tenant()
    try:
        async with session_scope() as session:
            (await session.get(Tenant, tenant_id)).is_active = False
        async with _client(slug, monkeypatch) as client:
            assert (await client.get("/v1/evals/runs")).status_code == 401
    finally:
        await _cleanup(tenant_id)


def test_every_state_changing_route_still_requires_a_key():
    """Anonymous access must never reach a handler that writes."""
    for route in full_app.routes:
        methods = getattr(route, "methods", set()) or set()
        if methods & {"POST", "PUT", "PATCH", "DELETE"}:
            names = {d.call.__name__ for d in route.dependant.dependencies}
            assert "current_reader" not in names, f"{route.path} accepts anonymous callers"
            assert "current_tenant" in names, f"{route.path} is not authenticated"


# -- evals API ----------------------------------------------------------------


async def test_gate_endpoint_reports_a_regression_with_its_interval(pg_engine, monkeypatch):
    tenant_id, slug = await _tenant()
    try:
        ds = await _dataset(tenant_id, 40)
        good = lambda i: f"answer {i['q'][1:]}"  # noqa: E731
        bad = lambda i: "wrong" if int(i["q"][1:]) % 3 else f"answer {i['q'][1:]}"  # noqa: E731
        base = await _run(tenant_id, ds, good, "base", [build_evaluator("exact_match")])
        cand = await _run(tenant_id, ds, bad, "cand", [build_evaluator("exact_match")])
        _GATE_CACHE.clear()
        async with _client(slug, monkeypatch) as client:
            body = (await client.get(f"/v1/evals/gate?baseline={base}&candidate={cand}")).json()
        assert body["verdict"] == "fail" and body["blocks_release"] is True
        cmp = body["evaluators"][0]["comparison"]
        assert cmp["n_pairs"] == 40
        assert cmp["ci_high"] < 0, "a clear regression's interval must exclude zero"
        assert body["policy"]["resamples"] == 10_000
    finally:
        await _cleanup(tenant_id)


async def test_gate_endpoint_returns_a_refusal_rather_than_a_verdict(pg_engine, monkeypatch):
    tenant_id, slug = await _tenant()
    try:
        ds = await _dataset(tenant_id, 5)
        base = await _run(tenant_id, ds, lambda i: "x", "base", [build_evaluator("exact_match")])
        cand = await _run(tenant_id, ds, lambda i: "y", "cand", [build_evaluator("exact_match")])
        _GATE_CACHE.clear()
        async with _client(slug, monkeypatch) as client:
            body = (await client.get(f"/v1/evals/gate?baseline={base}&candidate={cand}")).json()
        assert body["verdict"] == "insufficient_data"
        assert body["blocks_release"] is True
    finally:
        await _cleanup(tenant_id)


async def test_gate_endpoint_rejects_comparing_a_run_with_itself(pg_engine, monkeypatch):
    tenant_id, slug = await _tenant()
    try:
        run = str(uuid.uuid4())
        async with _client(slug, monkeypatch) as client:
            response = await client.get(f"/v1/evals/gate?baseline={run}&candidate={run}")
        assert response.status_code == 400
    finally:
        await _cleanup(tenant_id)


async def test_calibration_endpoint_measures_a_judge_against_human_labels(
    pg_engine, monkeypatch, tmp_path
):
    """A recorded judge that agrees with people on 9 of 10 outputs."""
    tenant_id, slug = await _tenant()
    try:
        ds = await _dataset(tenant_id, 10)
        evaluator = build_evaluator(
            "correctness_vs_reference", provider="groq", model="recorded-judge", threshold=0.5
        )
        # Judge verdicts written to a cassette: correct for even cases, plus
        # one disagreement on case 9.
        cassette = tmp_path / "cassette.json"
        judge = ReplayJudge("recorded-judge", str(cassette))
        from agentlens_platform.evaluators.base import EvalInput
        from agentlens_platform.evaluators.model_graded import JUDGE_SYSTEM

        entries = {}
        for i in range(10):
            output = f"answer {i}" if i % 2 == 0 else "wrong"
            prompt = evaluator.render_prompt(
                EvalInput(output=output, inputs={"q": f"q{i}"}, expected=f"answer {i}")
            )
            score = 1.0 if (i % 2 == 0) != (i == 9) else 0.0
            key = judge._key(prompt, JUDGE_SYSTEM, evaluator.config["max_tokens"], 0.0)
            entries[key] = {"text": json.dumps({"reason": "r", "score": score}), "model": "recorded-judge"}
        cassette.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        monkeypatch.setenv("AGENTLENS_JUDGE_CASSETTE", str(cassette))
        monkeypatch.delenv("AGENTLENS_JUDGE_RECORD", raising=False)

        system = lambda i: f"answer {i['q'][1:]}" if int(i["q"][1:]) % 2 == 0 else "wrong"  # noqa: E731
        await _run(tenant_id, ds, system, "graded", [evaluator])

        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, ds)
            from sqlalchemy import select

            from agentlens_platform.db.models import EvalCase

            for case in (await session.execute(select(EvalCase).where(EvalCase.dataset_id == dataset.id))).scalars():
                i = int(case.external_id[1:])
                session.add(HumanLabel(
                    tenant_id=tenant_id, case_id=case.id,
                    graded_output=f"answer {i}" if i % 2 == 0 else "wrong",
                    label=1.0 if i % 2 == 0 else 0.0, labeller="test annotator",
                ))

        async with _client(slug, monkeypatch) as client:
            body = (await client.get("/v1/evals/calibration")).json()
        [judge_entry] = body["judges"]
        c = judge_entry["calibration"]
        assert c["n"] == 10
        assert c["binary_agreement"] == pytest.approx(0.9)
        assert 0.7 < c["cohens_kappa"] < 0.9
        assert judge_entry["model"] == "recorded-judge"
        assert body["labellers"] == ["test annotator"]
    finally:
        await _cleanup(tenant_id)
