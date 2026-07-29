"""Graph metrics and motif extraction.

The decisive criterion: a graph is only useful if knowing a node's neighbours
tells you something *specific to that node*.

There are therefore two opposite ways to fail:

- **too sparse** — most nodes have no neighbours; there is nothing to look at;
- **too dense** — every node touches nearly every other; node A's neighbourhood
  is the same as node B's, so it discriminates nothing.

The informative band observed on real traces: density 1–6%, mean degree 10–15.
These bounds are empirical, not theoretical — measure your own.
"""

from __future__ import annotations

import collections
import itertools
import math
import random
from dataclasses import dataclass

from .extract import Hyperedge, mask_secrets


@dataclass
class GraphMetrics:
    nodes: int
    edges: int
    density: float
    mean_degree: float
    degree_entropy: float
    top5_mass: float
    singleton_rate: float

    def verdict(self) -> str:
        """Locate the graph relative to the two degenerate regimes."""
        if self.density > 0.15:
            return "TOO DENSE — close to a clique, neighbourhoods no longer discriminate"
        if self.nodes < 20:
            return "TOO FEW NODES — vocabulary too coarse, or corpus too small"
        if self.density < 0.002:
            return "TOO SPARSE — dust of singletons, almost no edges"
        return "INFORMATIVE REGIME — neighbourhoods carry information"

    def __str__(self) -> str:
        return (f"N={self.nodes} · edges={self.edges} · density={self.density:.4f} · "
                f"mean degree={self.mean_degree:.1f} · entropy={self.degree_entropy:.3f}\n"
                f"→ {self.verdict()}")


def cooccurrence(edges: list[Hyperedge], min_occurrences: int = 2
                 ) -> collections.Counter:
    """Project hyperedges into weighted binary edges.

    `min_occurrences` drops atoms seen only once. On real traces the singleton
    rate hovers around 70% regardless of normalisation: it is a property of
    traces (many unique commands), not a tuning defect. The filter absorbs it.
    """
    counts = collections.Counter(a for edge in edges for a in edge.atoms)
    keep = {a for a, k in counts.items() if k >= min_occurrences}
    pairs: collections.Counter = collections.Counter()
    for edge in edges:
        present = sorted({a for a in edge.atoms if a in keep})
        for pair in itertools.combinations(present, 2):
            pairs[pair] += 1
    return pairs


def metrics(edges: list[Hyperedge], min_occurrences: int = 2) -> GraphMetrics:
    """Compute the structural metrics of the co-occurrence graph."""
    counts = collections.Counter(a for edge in edges for a in edge.atoms)
    singleton_rate = (sum(1 for k in counts.values() if k == 1) / len(counts)
                      if counts else 0.0)

    pairs = cooccurrence(edges, min_occurrences)
    nodes = {n for pair in pairs for n in pair}
    if len(nodes) < 3:
        return GraphMetrics(len(nodes), len(pairs), 0.0, 0.0, 0.0, 0.0, singleton_rate)

    degree: collections.Counter = collections.Counter()
    for left, right in pairs:
        degree[left] += 1
        degree[right] += 1

    n = len(nodes)
    total_degree = sum(degree.values())
    possible = n * (n - 1) / 2

    # Shannon entropy of the degree distribution. Two graphs with the same
    # density can be a star (one hub captures everything, low entropy) or a
    # varied structure (high entropy). Only the latter is usable for clustering
    # or PageRank.
    entropy = 0.0
    for value in degree.values():
        p = value / total_degree
        if p > 0:
            entropy -= p * math.log2(p)

    top5 = sum(k for _, k in degree.most_common(5)) / total_degree

    return GraphMetrics(
        nodes=n,
        edges=len(pairs),
        density=len(pairs) / possible,
        mean_degree=total_degree / n,
        degree_entropy=entropy,
        top5_mass=top5,
        singleton_rate=singleton_rate,
    )


def granularity_sweep(sessions_glob: str, variants: dict[str, dict]) -> list[dict]:
    """Sweep several granularities and compare the resulting regimes.

    This is the most instructive experiment in the set: it shows where your
    corpus tips from clique to dust, and whether an informative band exists.
    """
    from .extract import build_hyperedges

    rows = []
    for name, params in variants.items():
        edges = build_hyperedges(sessions_glob, **params)
        result = metrics(edges)
        rows.append({
            "variant": name,
            "hyperedges": len(edges),
            "N": result.nodes,
            "density": round(result.density, 4),
            "mean_degree": round(result.mean_degree, 1),
            "entropy": round(result.degree_entropy, 3),
            "verdict": result.verdict().split(" —")[0],
        })
    return rows


# --------------------------------------------------------------------------
# Motifs
# --------------------------------------------------------------------------

def communities(pairs: collections.Counter, iterations: int = 20,
                seed: int = 0) -> dict[str, int]:
    """Community detection by label propagation.

    Intentionally minimal and dependency-free: each node iteratively adopts
    the majority label of its neighbours, weighted by edge weight. Sufficient
    to surface motifs; replace with Louvain (python-louvain, networkx) if you
    want modularity scores.
    """
    adjacency: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for (left, right), weight in pairs.items():
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))

    labels = {node: i for i, node in enumerate(adjacency)}
    rng = random.Random(seed)
    nodes = list(adjacency)

    for _ in range(iterations):
        rng.shuffle(nodes)
        changed = False
        for node in nodes:
            tally: collections.Counter = collections.Counter()
            for neighbour, weight in adjacency[node]:
                tally[labels[neighbour]] += weight
            if tally:
                best = tally.most_common(1)[0][0]
                if labels[node] != best:
                    labels[node] = best
                    changed = True
        if not changed:
            break
    return labels


def motifs(edges: list[Hyperedge], min_occurrences: int = 2,
           min_size: int = 3) -> list[dict]:
    """Extract recurring motifs and trace them back to real commands.

    A statistical motif without a human name proves nothing. The output always
    includes examples of commands — masked — so you can judge for yourself
    whether the grouping makes sense or whether the algorithm has assembled
    unrelated things.

    Expect three categories: usable motifs, real but trivial ones (`cd` then
    `ls` is a genuine motif and a worthless one), and pure artefacts.
    """
    pairs = cooccurrence(edges, min_occurrences)
    labels = communities(pairs)

    grouped: dict[int, list[str]] = collections.defaultdict(list)
    for node, label in labels.items():
        grouped[label].append(node)

    results = []
    for label, members in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_size:
            continue
        member_set = set(members)
        examples = [
            mask_secrets(edge.command)[:200]
            for edge in edges
            if len(member_set & set(edge.atoms)) >= 2
        ]
        results.append({
            "size": len(members),
            "occurrences": len(examples),
            "atoms": sorted(members)[:12],
            "examples": examples[:3],
        })
    return results
