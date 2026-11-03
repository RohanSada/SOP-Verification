"""Prerequisite graphs: validation, transitive reduction, derivation, inference and SVG rendering.

Edge direction everywhere: `p -> s` means p must be complete before s completes.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence

import networkx as nx

from sopverify.procedure.schema import Step, Vocabulary

START = "__start__"


# ---------------------------------------------------------------------- building
def requires_graph(steps: Sequence[Step]) -> nx.DiGraph:
    g = nx.DiGraph()
    for s in steps:
        g.add_node(s.id)
    for s in steps:
        for p in s.requires:
            g.add_edge(p, s.id)
    return g


def graph_from_requires(requires: Mapping[str, Iterable[str]], nodes: Iterable[str] | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes or [])
    for s, ps in requires.items():
        g.add_node(s)
        for p in ps:
            g.add_edge(p, s)
    return g


def to_requires(g: nx.DiGraph, order: Sequence[str] | None = None) -> dict[str, list[str]]:
    nodes = list(order) if order is not None else list(g.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    return {n: sorted(g.predecessors(n), key=lambda p: pos.get(p, 1 << 30)) for n in nodes if n in g}


def transitive_reduction(g: nx.DiGraph) -> nx.DiGraph:
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"graph has a cycle: {find_cycle(g)}")
    red = nx.transitive_reduction(g)
    red.add_nodes_from(g.nodes(data=True))
    return red


def find_cycle(g: nx.DiGraph) -> list[str] | None:
    try:
        cyc = nx.find_cycle(g)
    except nx.NetworkXNoCycle:
        return None
    return [u for u, _ in cyc] + [cyc[-1][1]]


# ---------------------------------------------------------------------- validation
def validate_steps(steps: Sequence[Step], vocab: Vocabulary | None = None, step_set: str = "") -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errors: list[str] = []
    ids = [s.id for s in steps]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate step ids: {dup}")
    idset = set(ids)
    for s in steps:
        for p in s.requires:
            if p not in idset:
                errors.append(f"{s.id}: requires unknown step {p!r}")
            if p == s.id:
                errors.append(f"{s.id}: requires itself")
    g = requires_graph([s for s in steps])
    g.remove_nodes_from([n for n in list(g.nodes) if n not in idset])
    cyc = find_cycle(g)
    if cyc:
        errors.append(f"prerequisite cycle: {' -> '.join(cyc)}")
    else:
        # Every step reachable from a virtual START that precedes all root steps.
        h = g.copy()
        h.add_node(START)
        for n in g.nodes:
            if g.in_degree(n) == 0:
                h.add_edge(START, n)
        unreachable = sorted(set(g.nodes) - nx.descendants(h, START))
        if unreachable:
            errors.append(f"unreachable steps: {unreachable}")
    for s in steps:
        kinds = s.gt_mapping.kinds()
        if not kinds:
            errors.append(f"{s.id}: gt_mapping is empty")
        if vocab is not None:
            for e in s.gt_mapping.psr_events or []:
                if e not in vocab.psr_ids():
                    errors.append(f"{s.id}: psr event {e!r} not in vocabulary")
            for lab in s.gt_mapping.tas_s_labels or []:
                if vocab.tas_s_labels and lab not in vocab.tas_s_labels:
                    errors.append(f"{s.id}: TAS-S label {lab!r} not in vocabulary")
    return errors


# ---------------------------------------------------------------------- derivation
def derive_group_graph(
    steps: Sequence[Step], event_graph: nx.DiGraph, reduce: bool = True
) -> dict[str, list[str]]:
    """Grouped-step graph from an event-level graph (spec 8.2, graph_source: released).

    A requires B if any member event of A has a path from any member event of B.
    """
    members = {s.id: set(s.gt_mapping.psr_events or []) for s in steps}
    anc = {n: nx.ancestors(event_graph, n) for n in event_graph.nodes}
    g = nx.DiGraph()
    g.add_nodes_from(members)
    for a, ea in members.items():
        for b, eb in members.items():
            if a == b:
                continue
            if any(e in event_graph and (eb & anc.get(e, set())) for e in ea):
                g.add_edge(b, a)
    cyc = find_cycle(g)
    if cyc:
        raise ValueError(f"derived group graph has a cycle (groups split a dependency chain): {cyc}")
    if reduce:
        g = transitive_reduction(g)
    return to_requires(g, [s.id for s in steps])


def infer_graph(
    completions: Sequence[Mapping[str, float | None]],
    step_ids: Sequence[str],
    min_support: int = 5,
    reduce: bool = True,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], int]]:
    """Infer p -> s when p completed strictly before s in 100% of executions where both completed.

    Returns (requires, support) where support[(p, s)] counts executions backing the edge.
    Ties count against the edge. Any cycle (only possible with disjoint supports) is broken
    by dropping its weakest edge.
    """
    support: dict[tuple[str, str], int] = {}
    for p in step_ids:
        for s in step_ids:
            if p == s:
                continue
            n_both = n_before = 0
            for c in completions:
                tp, ts = c.get(p), c.get(s)
                if tp is None or ts is None:
                    continue
                n_both += 1
                if tp < ts:
                    n_before += 1
            if n_both >= min_support and n_before == n_both:
                support[(p, s)] = n_both
    g = nx.DiGraph()
    g.add_nodes_from(step_ids)
    g.add_edges_from(support)
    while (cyc := find_cycle(g)) is not None:
        edges = list(zip(cyc[:-1], cyc[1:]))
        weakest = min(edges, key=lambda e: support.get(e, 0))
        g.remove_edge(*weakest)
        support.pop(weakest, None)
    if reduce:
        g = transitive_reduction(g)
    return to_requires(g, list(step_ids)), support


def ancestors_map(requires: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    g = graph_from_requires(requires)
    return {n: nx.ancestors(g, n) for n in g.nodes}


# ---------------------------------------------------------------------- rendering
VERDICT_COLORS = {
    "DONE": "#2e7d32",
    "NOT_DONE": "#c62828",
    "OUT_OF_ORDER": "#ef6c00",
    "UNCERTAIN": "#f9a825",
    "NOT_VERIFIABLE": "#78909c",
    None: "#455a64",
}


def render_svg(
    requires: Mapping[str, Iterable[str]],
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    node_w: int = 190,
    node_h: int = 46,
    title: str | None = None,
) -> str:
    """Layered left-to-right SVG of a DAG; no Graphviz needed. Colors are fill colors per node."""
    g = graph_from_requires(requires, order)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"cannot render a cyclic graph: {find_cycle(g)}")
    layers = [sorted(layer, key=lambda n: (order.index(n) if order and n in order else 0, n))
              for layer in nx.topological_generations(g)]
    gap_x, gap_y, pad = 70, 18, 20
    top = 34 if title else 0
    height = max((len(lay) for lay in layers), default=1) * (node_h + gap_y) - gap_y + 2 * pad + top
    width = len(layers) * (node_w + gap_x) - gap_x + 2 * pad
    pos: dict[str, tuple[float, float]] = {}
    for li, layer in enumerate(layers):
        col_h = len(layer) * (node_h + gap_y) - gap_y
        y0 = top + (height - top - col_h) / 2
        for ni, n in enumerate(layer):
            pos[n] = (pad + li * (node_w + gap_x), y0 + ni * (node_h + gap_y))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>',
    ]
    if title:
        parts.append(f'<text x="{pad}" y="22" font-size="14" font-weight="600" fill="currentColor">{html.escape(title)}</text>')
    for u, v in g.edges:
        x1, y1 = pos[u][0] + node_w, pos[u][1] + node_h / 2
        x2, y2 = pos[v][0], pos[v][1] + node_h / 2
        mx = (x1 + x2) / 2
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {mx:.1f} {y1:.1f}, {mx:.1f} {y2:.1f}, {x2 - 2:.1f} {y2:.1f}" '
            'fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#arr)"/>'
        )
    for n, (x, y) in pos.items():
        fill = (colors or {}).get(n, VERDICT_COLORS[None])
        text = (labels or {}).get(n, n)
        lines = _wrap(f"{n}: {text}" if labels and text != n else n, 28)[:2]
        parts.append(f'<g><title>{html.escape(f"{n}: {text}")}</title>'
                     f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" fill="{fill}"/>')
        for i, line in enumerate(lines):
            ty = y + node_h / 2 + (i - (len(lines) - 1) / 2) * 14 + 4
            parts.append(f'<text x="{x + node_w / 2}" y="{ty:.1f}" text-anchor="middle" fill="#fff">{html.escape(line)}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        lines = [lines[0], (" ".join(lines[1:]))[: width - 1] + "…"]
    return lines
