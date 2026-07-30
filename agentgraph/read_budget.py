"""Reading-budget analysis: what does the agent ask for in its own head/tail?

When an agent truncates output, it writes the bound itself — `head -n 3`,
`tail -n 80`.  That number is committed *before* the output is visible, which
makes it a bet on how much information is about to matter.

## The key insight

The agent is not just deciding *how much* to read.  It is deciding *how much*
to read **based on what it just did**.  After a git delivery the median is 3
lines — enough to see whether the commit landed.  After a Python script, 80 —
the order of magnitude of a full output or a traceback.  A factor of ×27 on
the same read gesture.

This makes `n_lines` a **conditioning variable**: its distribution depends on
the preceding action type (the "edge" in the transition graph).  A
Kruskal-Wallis test measures whether the edge explains a significant fraction
of the variance in `n_lines`.  ε² (eta-squared for the KW test) quantifies
the effect size.

A **permutation test** (1,000 reshuffles of the observations across edges) is
not optional.  Without it, a high ε² is uninterpretable: the same number of
distinct edges would produce a high statistic even if `n_lines` were drawn
from one distribution.

The article reports:
- Median 3 lines (IQR 2) after git delivery; median 80 (IQR 160) after a
  Python script; factor ×27.
- ε² = 0.600 over 16 transitions and 295 unique observations.
- 0 / 1,000 permutations reach ε² = 0.600.

## What this module does NOT do

It does not infer the agent's *intent*.  "3 lines after a git commit" is a
measurement, not evidence that the agent "wants to verify the commit".  Avoid
that language in notebooks.

## Usage

    from agentgraph.extract import iter_shell_commands
    from agentgraph.read_budget import extract_transitions, kw_epsilon2, permutation_test

    shell_iter = iter_shell_commands("~/.openclaw/agents/*/sessions/*.jsonl")
    transitions = extract_transitions(shell_iter)
    result = kw_epsilon2(transitions, param="n_lines", min_obs=5)
    perm = permutation_test(transitions, param="n_lines", n_perm=1000)
    print(result, perm)
"""

from __future__ import annotations

import collections
import re
import shlex
import random
import statistics
from dataclasses import dataclass
from typing import Iterator

# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def _extract_n_lines(tokens: list[str]) -> int | None:
    """Extract N from `head -n N`, `tail -n N`, `head -N`, `tail -N`."""
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t == "-n" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif re.match(r"^-\d+$", t):
            return int(t[1:])
        elif re.match(r"^-n\d+$", t):
            try:
                return int(t[2:])
            except ValueError:
                pass
            i += 1
        else:
            i += 1
    return None


def _segment_binary(raw: str) -> str | None:
    """Return the binary name of a shell segment, or None."""
    raw = raw.strip().strip("()").rstrip("\\")
    if not raw or raw.startswith("#"):
        return None
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        tokens = raw.split()
    while tokens and (
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0])
        or tokens[0] in ("sudo", "time", "nohup", "exec", "env")
    ):
        tokens.pop(0)
    return tokens[0].split("/")[-1] if tokens else None


def _normalize_binary(binary: str) -> str:
    """Coarse action type for edge labeling (preceding action)."""
    GIT  = {"git", "gh"}
    INSP = {"cat", "head", "tail", "ls", "find", "grep", "wc", "stat",
             "file", "du", "df", "jq", "sed", "awk", "diff"}
    NET  = {"curl", "wget", "ssh", "scp", "rsync", "nc"}
    RUN  = {"python3", "python", "node", "deno", "bun", "npm", "pnpm",
             "tsc", "cargo", "go", "make"}
    OPS  = {"docker", "systemctl", "kill", "pm2"}

    if binary in GIT:  return "git_vcs"
    if binary in INSP: return "inspect_fs_text"
    if binary in NET:  return "network_http"
    if binary in RUN:  return "runtime_build"
    if binary in OPS:  return "ops_process"
    return "other_shell"


# ---------------------------------------------------------------------------
# Transition extraction
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A (preceding_action → head/tail call) pair with the n_lines value."""
    session: str
    preceding_action: str    # normalized binary of the segment before head/tail
    n_lines: int


def extract_transitions(
    shell_iter,  # iterable of (ToolCall, str)
) -> list[Transition]:
    """Extract all (preceding_action, n_lines) pairs from shell commands.

    For each command, scans segments left to right.  When a `head` or `tail`
    segment is found with a parseable `-n N` value, and the immediately
    preceding segment is not itself a head/tail, records a Transition.

    The sleep segment is treated as a *weight* on the preceding edge, not as
    an action in its own right — consistent with `opt_conditioning.py`.

    A preceding_action of ``"<start>"`` means the head/tail is the first
    segment (no preceding action in this command).
    """
    from agentgraph.extract import split_segments

    transitions: list[Transition] = []

    for call, command in shell_iter:
        segs = split_segments(command)
        prev_binary: str | None = "<start>"

        for raw in segs:
            b = _segment_binary(raw)
            if b is None:
                continue
            try:
                tokens = shlex.split(raw, posix=True)
            except ValueError:
                tokens = raw.split()

            if b in ("head", "tail"):
                n = _extract_n_lines(tokens)
                if n is not None and n > 0 and prev_binary is not None:
                    transitions.append(Transition(
                        session=call.session,
                        preceding_action=_normalize_binary(prev_binary)
                        if prev_binary != "<start>" else "<start>",
                        n_lines=n,
                    ))

            # Do not update prev_binary for sleep — it is a weight, not an action.
            if b != "sleep":
                prev_binary = b

    return transitions


# ---------------------------------------------------------------------------
# Statistics: Kruskal-Wallis ε² and permutation test
# ---------------------------------------------------------------------------

def _kruskal_wallis(groups: list[list[float]]) -> tuple[float, int]:
    """Manual Kruskal-Wallis H and degrees of freedom (no scipy required)."""
    n = sum(len(g) for g in groups)
    if n == 0 or len(groups) < 2:
        return float("nan"), 0

    all_vals = [(v, i) for i, g in enumerate(groups) for v in g]
    all_vals.sort(key=lambda x: x[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and all_vals[j][0] == all_vals[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j

    rank_sums = collections.defaultdict(float)
    group_ns: dict[int, int] = collections.defaultdict(int)
    for rank, (_, gi) in zip(ranks, all_vals):
        rank_sums[gi] += rank
        group_ns[gi] += 1

    H = (12.0 / (n * (n + 1))) * sum(
        rank_sums[gi] ** 2 / group_ns[gi] for gi in rank_sums
    ) - 3 * (n + 1)
    return H, len(groups) - 1


def kw_epsilon2(
    transitions: list[Transition],
    param: str = "n_lines",
    min_obs: int = 5,
) -> dict:
    """Kruskal-Wallis H and ε² (eta-squared) for the reading budget.

    ε² = (H - k + 1) / (n - k)

    where k = number of groups (action types) and n = total observations.

    ε² ∈ [0, 1]: 0 means the action type explains nothing; 1 means it
    explains everything.  The article reports ε² = 0.600.

    Parameters
    ----------
    transitions:
        Output of ``extract_transitions``.
    param:
        Which parameter to analyse.  Only ``"n_lines"`` is currently extracted.
    min_obs:
        Drop action types with fewer than this many observations.

    Returns
    -------
    dict with ``H``, ``k``, ``n``, ``eps2``, ``group_medians``, and a textual
    ``interpretation``.
    """
    groups_raw: dict[str, list[float]] = collections.defaultdict(list)
    for t in transitions:
        groups_raw[t.preceding_action].append(float(t.n_lines))

    groups = {e: v for e, v in groups_raw.items() if len(v) >= min_obs}
    if len(groups) < 2:
        return {
            "error": f"Fewer than 2 action types with ≥{min_obs} observations.",
            "groups_found": len(groups),
        }

    arrays = list(groups.values())
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    H, dof = _kruskal_wallis(arrays)
    eps2 = max(0.0, (H - k + 1) / (n - k)) if n > k else float("nan")

    medians = {e: statistics.median(v) for e, v in groups.items()}
    sorted_groups = sorted(medians.items(), key=lambda kv: -kv[1])

    return {
        "H": round(H, 3),
        "k": k,
        "n": n,
        "eps2": round(eps2, 4),
        "group_medians": {e: round(m, 1) for e, m in sorted_groups},
        "interpretation": (
            f"ε² = {eps2:.3f}: the preceding action explains "
            f"~{100*eps2:.0f}% of the variance in n_lines across "
            f"{k} action types and {n} observations.  "
            "Run permutation_test() to confirm this is above chance."
        ),
    }


def permutation_test(
    transitions: list[Transition],
    param: str = "n_lines",
    min_obs: int = 5,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation baseline for the KW ε² reading budget.

    Redistributes the observed n_lines values randomly across action types
    (preserving group sizes) and measures how often the permuted ε² reaches
    the observed value.

    p_perm = fraction of permutations with ε²_perm ≥ ε²_observed.

    The article reports p_perm = 0/1,000 = 0.000 for n_lines over 16 action
    types and 295 observations.

    **This test is not optional.**  The same ε² on different group structures
    has different meaning.  Without the permutation control, a high ε² is
    uninterpretable.
    """
    groups_raw: dict[str, list[float]] = collections.defaultdict(list)
    for t in transitions:
        groups_raw[t.preceding_action].append(float(t.n_lines))

    groups = {e: v for e, v in groups_raw.items() if len(v) >= min_obs}
    if len(groups) < 2:
        return {"error": "Fewer than 2 groups after filtering."}

    arrays = list(groups.values())
    ns = [len(a) for a in arrays]
    n = sum(ns)
    k = len(arrays)

    H_obs, _ = _kruskal_wallis(arrays)
    eps2_obs = max(0.0, (H_obs - k + 1) / (n - k)) if n > k else float("nan")

    all_vals = [v for a in arrays for v in a]
    rng = random.Random(seed)
    count_ge = 0
    for _ in range(n_perm):
        shuffled = list(all_vals)
        rng.shuffle(shuffled)
        perm_arrays = []
        idx = 0
        for ni in ns:
            perm_arrays.append(shuffled[idx : idx + ni])
            idx += ni
        H_p, _ = _kruskal_wallis(perm_arrays)
        eps2_p = max(0.0, (H_p - k + 1) / (n - k)) if n > k else 0.0
        if eps2_p >= eps2_obs:
            count_ge += 1

    p_perm = count_ge / n_perm
    return {
        "eps2_observed": round(eps2_obs, 4),
        "p_perm": round(p_perm, 4),
        "n_perm": n_perm,
        "count_ge": count_ge,
        "interpretation": (
            f"p_perm = {p_perm:.3f} ({count_ge}/{n_perm} permutations reach "
            f"ε² ≥ {eps2_obs:.3f}).  "
            + (
                "The result is NOT above chance: permuted data achieves the "
                "same effect size."
                if p_perm >= 0.05 else
                "The result is above chance: the preceding action genuinely "
                "predicts the reading budget."
            )
        ),
    }
