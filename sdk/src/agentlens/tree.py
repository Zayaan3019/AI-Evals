"""Reconstruct and render a trace tree from finished spans.

The Phase 1 claim is that the full decision tree is reconstructable from the
trace. This module is the executable form of that claim: given the spans a run
produced, it rebuilds parent/child structure and renders it, so the tree can be
inspected without a UI or a backend.

Also used by the CLI reporter, which has to show a trace in a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import semconv as sc


@dataclass
class TraceNode:
    """One span plus its children, ordered by start time."""

    span: object
    children: list["TraceNode"] = field(default_factory=list)

    @property
    def name(self) -> str:
        return getattr(self.span, "name", "<unnamed>")

    @property
    def duration_ms(self) -> float:
        start = getattr(self.span, "start_time", None)
        end = getattr(self.span, "end_time", None)
        if start is None or end is None:
            return 0.0
        return (end - start) / 1e6

    @property
    def attributes(self) -> dict:
        return dict(getattr(self.span, "attributes", {}) or {})

    def walk(self) -> Iterable["TraceNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


def build_forest(spans: Sequence[object]) -> list[TraceNode]:
    """Rebuild parent/child structure from a flat span list.

    Returns one node per root. Spans whose parent is absent from ``spans`` are
    treated as roots, so a partially-captured trace still renders rather than
    silently disappearing.
    """
    nodes: dict[int, TraceNode] = {}
    for span in spans:
        context = getattr(span, "context", None)
        if context is None:
            continue
        nodes[context.span_id] = TraceNode(span=span)

    roots: list[TraceNode] = []
    for span_id, node in nodes.items():
        parent = getattr(node.span, "parent", None)
        parent_id = getattr(parent, "span_id", None) if parent else None
        if parent_id is not None and parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            roots.append(node)

    def _sort(node: TraceNode) -> None:
        node.children.sort(key=lambda n: getattr(n.span, "start_time", 0) or 0)
        for child in node.children:
            _sort(child)

    for root in roots:
        _sort(root)
    roots.sort(key=lambda n: getattr(n.span, "start_time", 0) or 0)
    return roots


def _summarise(node: TraceNode) -> str:
    """One-line summary of the attributes that matter for this span kind."""
    attrs = node.attributes
    parts: list[str] = []

    operation = attrs.get(sc.GEN_AI_OPERATION_NAME)

    if attrs.get(sc.GEN_AI_USAGE_INPUT_TOKENS) is not None:
        parts.append(
            f"tokens={attrs.get(sc.GEN_AI_USAGE_INPUT_TOKENS)}"
            f"/{attrs.get(sc.GEN_AI_USAGE_OUTPUT_TOKENS)}"
        )
    if attrs.get(sc.GEN_AI_REQUEST_TEMPERATURE) is not None:
        parts.append(f"temp={attrs[sc.GEN_AI_REQUEST_TEMPERATURE]}")

    cost = attrs.get(sc.COST_MICRO_USD)
    if cost is not None:
        parts.append(f"cost=${cost / 1e6:.6f}")
    elif attrs.get(sc.COST_STATUS) not in (None, "ok"):
        parts.append(f"cost={attrs[sc.COST_STATUS]}")

    if operation == sc.OperationName.RETRIEVAL.value:
        count = attrs.get(sc.RETRIEVAL_DOCUMENT_COUNT)
        if count is not None:
            parts.append(f"docs={count}")
        top_k = attrs.get(sc.RETRIEVAL_TOP_K)
        if top_k is not None:
            parts.append(f"top_k={top_k}")

    rules = attrs.get(sc.REDACTION_RULES)
    if rules:
        parts.append(f"redacted={','.join(rules)}")

    status = getattr(getattr(node.span, "status", None), "status_code", None)
    if status is not None and getattr(status, "name", "") == "ERROR":
        error_type = attrs.get(sc.ERROR_TYPE, "error")
        parts.append(f"ERROR:{error_type}")

    return "  ".join(parts)


#: Box-drawing glyphs, and an ASCII fallback. Windows consoles commonly default
#: to cp1252, which cannot encode the box-drawing set -- rendering a tree there
#: raised UnicodeEncodeError, so the glyph set is selected rather than assumed.
_GLYPHS = {
    "unicode": {"last": "└─ ", "mid": "├─ ", "pipe": "│  ", "bullet": "·"},
    "ascii": {"last": "`-- ", "mid": "|-- ", "pipe": "|   ", "bullet": "*"},
}


def _supports_unicode() -> bool:
    import sys

    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "└─│·".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def render_forest(
    spans: Sequence[object],
    *,
    show_attributes: bool = False,
    ascii_only: bool | None = None,
) -> str:
    """Render spans as an indented tree with durations and key attributes.

    ``ascii_only=None`` picks the glyph set from the current stdout encoding, so
    the default works on a cp1252 Windows console and a UTF-8 terminal alike.
    """
    if ascii_only is None:
        ascii_only = not _supports_unicode()
    glyphs = _GLYPHS["ascii" if ascii_only else "unicode"]

    roots = build_forest(spans)
    lines: list[str] = []

    def _render(node: TraceNode, prefix: str, is_last: bool, is_root: bool) -> None:
        connector = "" if is_root else (glyphs["last"] if is_last else glyphs["mid"])
        summary = _summarise(node)
        lines.append(
            f"{prefix}{connector}{node.name}  [{node.duration_ms:.1f}ms]"
            + (f"  {summary}" if summary else "")
        )
        pad = " " * len(glyphs["last"])
        child_prefix = prefix + ("" if is_root else (pad if is_last else glyphs["pipe"]))
        if show_attributes:
            for key, value in sorted(node.attributes.items()):
                text = str(value)
                if len(text) > 100:
                    text = text[:100] + "..."
                lines.append(f"{child_prefix}    {glyphs['bullet']} {key} = {text}")
        for index, child in enumerate(node.children):
            _render(child, child_prefix, index == len(node.children) - 1, False)

    for root in roots:
        _render(root, "", True, True)
    return "\n".join(lines)


def tree_stats(spans: Sequence[object]) -> dict[str, object]:
    """Aggregate a trace: span counts by operation, tokens, and cost."""
    total_input = 0
    total_output = 0
    total_cost_micro = 0
    cost_known = False
    by_operation: dict[str, int] = {}

    for span in spans:
        attrs = dict(getattr(span, "attributes", {}) or {})
        operation = str(attrs.get(sc.GEN_AI_OPERATION_NAME, "other"))
        by_operation[operation] = by_operation.get(operation, 0) + 1
        total_input += int(attrs.get(sc.GEN_AI_USAGE_INPUT_TOKENS) or 0)
        total_output += int(attrs.get(sc.GEN_AI_USAGE_OUTPUT_TOKENS) or 0)
        micro = attrs.get(sc.COST_MICRO_USD)
        if micro is not None:
            total_cost_micro += int(micro)
            cost_known = True

    roots = build_forest(spans)
    return {
        "span_count": len(spans),
        "root_count": len(roots),
        "by_operation": dict(sorted(by_operation.items())),
        "input_tokens": total_input,
        "output_tokens": total_output,
        # None rather than 0: an unpriced model must not look free.
        "cost_micro_usd": total_cost_micro if cost_known else None,
        "wall_ms": max((n.duration_ms for n in roots), default=0.0),
    }


__all__ = ["TraceNode", "build_forest", "render_forest", "tree_stats"]
