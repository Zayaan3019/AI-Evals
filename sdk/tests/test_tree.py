"""Trace-tree reconstruction: the executable form of the Phase 1 claim."""

from __future__ import annotations

import agentlens
from agentlens import Document
from agentlens.tree import build_forest, render_forest, tree_stats


def _run_workload() -> None:
    with agentlens.workflow("support_request"):
        with agentlens.agent_step("triage"):
            with agentlens.retrieval("kb", query="refund policy", top_k=3) as r:
                r.set_documents([Document(id="d1", score=0.9, content="policy")])
            with agentlens.llm_call("claude-opus-5", provider="anthropic") as llm:
                llm.set_usage(1_000_000, 0)
        with agentlens.agent_step("respond"):
            with agentlens.tool_call("send_email"):
                pass


def test_forest_reconstructs_nesting(lens) -> None:
    _run_workload()
    roots = build_forest(lens.memory_exporter.get_finished_spans())

    assert len(roots) == 1
    root = roots[0]
    assert root.name == "invoke_workflow support_request"
    assert [c.name for c in root.children] == [
        "invoke_agent triage",
        "invoke_agent respond",
    ]
    triage = root.children[0]
    assert [c.name for c in triage.children] == [
        "retrieval kb",
        "chat claude-opus-5",
    ]
    # Every span is reachable from the root -- nothing orphaned.
    assert len(list(root.walk())) == 6


def test_children_are_ordered_by_start_time(lens) -> None:
    _run_workload()
    root = build_forest(lens.memory_exporter.get_finished_spans())[0]
    for node in root.walk():
        starts = [c.span.start_time for c in node.children]
        assert starts == sorted(starts)


def test_orphaned_spans_still_render_as_roots(lens) -> None:
    """A partially captured trace must render, not vanish."""
    _run_workload()
    spans = lens.memory_exporter.get_finished_spans()
    # Drop the workflow root, orphaning both agent steps.
    without_root = [s for s in spans if not s.name.startswith("invoke_workflow")]
    roots = build_forest(without_root)
    assert {r.name for r in roots} == {"invoke_agent triage", "invoke_agent respond"}


def test_render_includes_durations_and_key_attributes(lens) -> None:
    _run_workload()
    output = render_forest(lens.memory_exporter.get_finished_spans(), ascii_only=False)
    assert "invoke_workflow support_request" in output
    assert "ms]" in output
    assert "tokens=1000000/0" in output
    assert "cost=$5.000000" in output
    assert "docs=1" in output
    # Tree drawing characters prove nesting is rendered, not a flat list.
    assert "└─" in output or "├─" in output


def test_ascii_rendering_is_encodable_on_a_cp1252_console(lens) -> None:
    """Regression: box-drawing glyphs raised UnicodeEncodeError on Windows."""
    _run_workload()
    output = render_forest(lens.memory_exporter.get_finished_spans(), ascii_only=True)
    output.encode("cp1252")  # must not raise
    assert "|--" in output or "`--" in output
    assert "└─" not in output


def test_stats_aggregate_tokens_and_cost(lens) -> None:
    _run_workload()
    stats = tree_stats(lens.memory_exporter.get_finished_spans())
    assert stats["span_count"] == 6
    assert stats["root_count"] == 1
    assert stats["input_tokens"] == 1_000_000
    assert stats["cost_micro_usd"] == 5_000_000
    assert stats["by_operation"]["invoke_agent"] == 2
    assert stats["by_operation"]["retrieval"] == 1


def test_unpriced_run_reports_none_cost_not_zero(lens) -> None:
    """An unpriced model must not be summarised as free."""
    with agentlens.llm_call("some-local-model", provider="ollama") as span:
        span.set_usage(100, 50)
    stats = tree_stats(lens.memory_exporter.get_finished_spans())
    assert stats["cost_micro_usd"] is None
    assert stats["input_tokens"] == 100


def test_error_is_visible_in_the_rendered_tree(lens) -> None:
    try:
        with agentlens.tool_call("boom"):
            raise ValueError("nope")
    except ValueError:
        pass
    output = render_forest(lens.memory_exporter.get_finished_spans())
    assert "ERROR:ValueError" in output


def test_empty_span_list_renders_empty(lens) -> None:
    assert render_forest([]) == ""
    assert tree_stats([])["span_count"] == 0
