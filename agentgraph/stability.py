"""Temporal stability of motifs across time windows.

Splits the hyperedges into N time windows and compares the first window with the
last on three axes: shared vocabulary (Jaccard), shared graph structure, and rank
correlation of edge weights.

Two windows from the same agent tend to share vocabulary — same machine, same
tools, same operator — so a raw persistence score cannot be read on its own.
`temporal_stability()` redistributes the same hyperedges randomly across the same
windows over several trials and reports the observed score next to that baseline.
A score at or below the shuffled distribution carries no signal.
"""

from __future__ import annotations

import collections
import random
import statistics
from dataclasses import dataclass

from .extract import Hyperedge
from .graph import cooccurrence


@dataclass
class StabilityResult:
    label: str
    observed: float
    baseline_mean: float
    baseline_std: float

    @property
    def beats_chance(self) -> bool:
        return self.observed > self.baseline_mean + 2 * self.baseline_std

    def __str__(self) -> str:
        if self.beats_chance:
            verdict = "ABOVE chance"
        elif self.observed < self.baseline_mean - 2 * self.baseline_std:
            verdict = "BELOW chance"
        else:
            verdict = "indistinguishable from chance"
        return (f"{self.label:<28} real={self.observed:.3f}  "
                f"chance={self.baseline_mean:.3f}±{self.baseline_std:.3f}  → {verdict}")


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, with no external dependency."""
    if len(xs) < 3:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def neighbourhoods(edges: list[Hyperedge], min_occurrences: int = 2
                   ) -> dict[str, set[str]]:
    """Neighbourhood of each atom within a time window."""
    pairs = cooccurrence(edges, min_occurrences)
    result: dict[str, set[str]] = collections.defaultdict(set)
    for left, right in pairs:
        result[left].add(right)
        result[right].add(left)
    return result


def _compare(window_a: list[Hyperedge], window_b: list[Hyperedge]) -> dict[str, float]:
    """The three persistence measures between two windows."""
    vocab_a = {a for e in window_a for a in e.atoms}
    vocab_b = {a for e in window_b for a in e.atoms}

    nbr_a = neighbourhoods(window_a)
    nbr_b = neighbourhoods(window_b)
    shared = set(nbr_a) & set(nbr_b)

    # An atom can survive from one window to the next while completely changing
    # its context. Neighbourhood persistence is therefore far more demanding
    # than vocabulary persistence — and it is what distinguishes a routine from
    # a frequent word.
    structure = [jaccard(nbr_a[n], nbr_b[n]) for n in shared]

    pairs_a = cooccurrence(window_a)
    pairs_b = cooccurrence(window_b)
    common = set(pairs_a) & set(pairs_b)
    weights = spearman([pairs_a[p] for p in common],
                       [pairs_b[p] for p in common]) if len(common) >= 3 else 0.0

    return {
        "vocabulary": jaccard(vocab_a, vocab_b),
        "structure": statistics.median(structure) if structure else 0.0,
        "weights": weights,
    }


def temporal_stability(edges: list[Hyperedge], n_windows: int = 2,
                       trials: int = 20, seed: int = 0) -> list[StabilityResult]:
    """Compare real persistence between windows to a random baseline.

    Windows are cut by equal count rather than calendar duration: agent activity
    is highly irregular, and calendar windows would give incomparable sizes.

    Points to keep in mind when reading the result:

    - **Session autocorrelation** — commands within one session necessarily
      resemble each other. If a single session dominates a window, the
      "stability" you measure is just that session's internal coherence. Check
      `dominant_session_pct` in the `describe_windows()` output.
    - **Agent mixing** — if your windows correspond to different agents, you are
      measuring a difference in identity, not drift over time. Filter to one
      agent before calling this function.
    """
    dated = sorted([e for e in edges if e.timestamp], key=lambda e: e.timestamp)
    if len(dated) < n_windows * 30:
        raise ValueError(
            f"Corpus too small: {len(dated)} dated hyperedges for {n_windows} "
            f"windows. At least 30 dated hyperedges per window are required."
        )

    size = len(dated) // n_windows
    windows = [dated[i * size:(i + 1) * size] for i in range(n_windows)]

    observed = _compare(windows[0], windows[-1])

    # Baseline: same hyperedges, same window sizes, temporal order destroyed.
    # Whatever remains is structural chance.
    rng = random.Random(seed)
    shuffled_runs: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(trials):
        pool = list(dated)
        rng.shuffle(pool)
        fake = [pool[i * size:(i + 1) * size] for i in range(n_windows)]
        for key, value in _compare(fake[0], fake[-1]).items():
            shuffled_runs[key].append(value)

    return [
        StabilityResult(
            label=key,
            observed=observed[key],
            baseline_mean=statistics.mean(shuffled_runs[key]),
            baseline_std=statistics.pstdev(shuffled_runs[key]) or 1e-9,
        )
        for key in observed
    ]


def describe_windows(edges: list[Hyperedge], n_windows: int = 2) -> list[dict]:
    """Per-window composition: date range, session count, agent mix, dominance."""
    dated = sorted([e for e in edges if e.timestamp], key=lambda e: e.timestamp)
    size = len(dated) // n_windows
    rows = []
    for i in range(n_windows):
        window = dated[i * size:(i + 1) * size]
        sessions = collections.Counter(e.session for e in window)
        agents = collections.Counter(e.agent for e in window)
        dominant = sessions.most_common(1)[0][1] / len(window) if window else 0
        rows.append({
            "window": i + 1,
            "hyperedges": len(window),
            "from": window[0].timestamp[:10] if window else "",
            "to": window[-1].timestamp[:10] if window else "",
            "sessions": len(sessions),
            "dominant_session_pct": round(100 * dominant, 1),
            "agents": dict(agents),
            "warning": "one session dominates the window" if dominant > 0.4 else "",
        })
    return rows
