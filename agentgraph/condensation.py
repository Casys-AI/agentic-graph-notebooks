"""SCC condensation of the agent action graph.

The condensation theorem states that contracting every strongly connected
component (SCC) of a directed graph into a single macro-node always yields a
DAG.  The theorem holds for *any* directed graph — it demands nothing about what
happens inside the component (no convergence, no termination guarantee).

## What to read in the output

Because the theorem holds for any directed graph, "agents loop, therefore their
condensation is a DAG" says nothing specific about agents — it holds for batch
workflows and cron jobs too.  The quantity to read is the **size distribution of
the SCCs**.

- If every SCC is a singleton (size = 1) *and* carries no self-loop, the graph is
  already a DAG — no feedback at all at this level of abstraction. A singleton
  with a self-loop is still a cycle, and `scc_summary()` counts it in
  `n_cycle_sccs`; read that field, not the size distribution alone.
- If one SCC absorbs almost every node (a "giant" component), the condensation
  collapses to a single macro-node and loses all scheduling power.
- In between, each SCC represents a feedback zone: a set of actions that
  mutually imply one another.  Contracting them tells you which zones must be
  resolved before others can start.

## Usage

    from agentgraph.extract import build_hyperedges
    from agentgraph.condensation import transition_graph, scc_summary

    edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
    G = transition_graph(edges)
    report = scc_summary(G)
    print(report)

Standard library only: the Tarjan SCC computation is in this module.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

from .extract import Hyperedge

# ---------------------------------------------------------------------------
# Transition graph construction
# ---------------------------------------------------------------------------

def transition_graph(edges: list[Hyperedge]) -> dict[str, set[str]]:
    """Build a directed transition graph from a list of hyperedges.

    Within each hyperedge, every pair (A, B) where A precedes B in the atom
    list is added as a directed edge A → B.  The atom list is sorted, so the
    order is deterministic but reflects co-occurrence structure, not temporal
    ordering within a single command.

    For temporal ordering (A was observed before B across turns), use the
    ``session_transition_graph`` function instead.
    """
    G: dict[str, set[str]] = collections.defaultdict(set)
    for edge in edges:
        atoms = edge.atoms
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                G[a].add(b)
                G[b].add(a)   # co-occurrence → symmetric; remove for directed
    return dict(G)


def session_transition_graph(edges: list[Hyperedge]) -> dict[str, set[str]]:
    """Build a directed transition graph from temporal ordering within sessions.

    Atoms that appear in turn T are considered predecessors of atoms in turn
    T+1 within the same session.  This is the directed graph used by the
    article's condensation measurement.

    The graph can be large.  If you only care about SCC sizes, pass the result
    directly to ``scc_summary`` — no need to materialise the full adjacency.
    """
    G: dict[str, set[str]] = collections.defaultdict(set)

    # Group by session, sort by timestamp.
    by_session: dict[str, list[Hyperedge]] = collections.defaultdict(list)
    for edge in edges:
        by_session[edge.session].append(edge)

    for session_edges in by_session.values():
        ordered = sorted(
            [e for e in session_edges if e.timestamp],
            key=lambda e: e.timestamp,
        )
        for i in range(len(ordered) - 1):
            for a in ordered[i].atoms:
                for b in ordered[i + 1].atoms:
                    if a != b:
                        G[a].add(b)

    return dict(G)


# ---------------------------------------------------------------------------
# SCC computation (Tarjan's algorithm)
# ---------------------------------------------------------------------------

def _tarjan_sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components.

    Iterative, with an explicit call stack: a recursive version hits Python's
    recursion limit on long chains of actions.
    """
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect_iter(root: str) -> None:
        call_stack = [(root, iter(graph.get(root, set())), False)]
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack.add(root)

        while call_stack:
            v, children, _ = call_stack[-1]
            try:
                w = next(children)
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, iter(graph.get(w, set())), False))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            except StopIteration:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)

    for node in list(graph):
        if node not in index:
            strongconnect_iter(node)

    return sccs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class SCCSummary:
    """Summary of the SCC structure of the action graph."""
    n_nodes: int
    n_edges: int
    n_sccs: int
    n_cycle_sccs: int           # SCCs with > 1 node, or a self-loop
    singleton_sccs: int         # SCCs with exactly 1 node and no self-loop
    largest_scc_size: int
    largest_scc_fraction: float  # largest / n_nodes
    size_distribution: dict[int, int]  # size → count

    def condensation_is_useful(self) -> bool:
        """True when the condensation retains scheduling information.

        A condensation is *not* useful when the largest SCC absorbs the
        majority of nodes — the condensed DAG then has one macro-node and
        provides no ordering information.
        """
        return self.largest_scc_fraction < 0.5

    def __str__(self) -> str:
        lines = [
            f"Nodes: {self.n_nodes}  Edges: {self.n_edges}  SCCs: {self.n_sccs}",
            f"Cycle SCCs: {self.n_cycle_sccs}  Singletons: {self.singleton_sccs}",
            f"Largest SCC: {self.largest_scc_size} nodes "
            f"({100 * self.largest_scc_fraction:.1f}% of graph)",
        ]
        if not self.condensation_is_useful():
            lines.append(
                "⚠ Condensation collapses to ~1 macro-node: the graph is "
                "dominated by one feedback zone and yields no DAG ordering."
            )
        else:
            lines.append(
                "✓ Condensation retains useful structure: "
                "SCCs are local, DAG ordering is meaningful."
            )
        return "\n".join(lines)


def scc_summary(graph: dict[str, set[str]]) -> SCCSummary:
    """Compute SCC sizes and return a summary.

    Pass the output of ``session_transition_graph`` or ``transition_graph``.

    Any directed graph has SCCs, so read the size distribution:

    - Many small SCCs → feedback is local, condensation is informative.
    - One giant SCC → feedback spans the whole graph, condensation is trivial.
    """
    n_edges = sum(len(v) for v in graph.values())
    sccs = _tarjan_sccs(graph)

    n_cycle = 0
    n_singleton = 0
    size_dist: dict[int, int] = collections.Counter()
    largest = 0

    for scc in sccs:
        sz = len(scc)
        size_dist[sz] += 1
        if sz > largest:
            largest = sz
        if sz > 1:
            n_cycle += 1
        elif sz == 1:
            # Check for self-loop
            node = scc[0]
            if node in graph.get(node, set()):
                n_cycle += 1
            else:
                n_singleton += 1

    n_nodes = len(graph)
    return SCCSummary(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_sccs=len(sccs),
        n_cycle_sccs=n_cycle,
        singleton_sccs=n_singleton,
        largest_scc_size=largest,
        largest_scc_fraction=largest / n_nodes if n_nodes else 0.0,
        size_distribution=dict(sorted(size_dist.items())),
    )
