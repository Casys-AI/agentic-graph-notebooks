"""RE-PAIR grammar induction on agent action sequences.

RE-PAIR is a grammar-compression algorithm.  Given a sequence of symbols, it
repeatedly finds the most frequent adjacent pair, replaces every occurrence with
a fresh non-terminal symbol, and records the substitution as a grammar rule.
The process stops when no pair appears at least ``min_uses`` times.

## What this measures

The compressed grammar is a **summary of repetition structure** in the agent's
action sequences.  Each grammar rule represents a recurring subsequence.  Rules
can compose: rule R1 can expand into (A, R2), and R2 can expand into (B, C) —
giving a two-level hierarchy.

**Composition depth** is the key quantity.  Depth 0 means a rule expands
directly into two terminal atoms.  Depth 1 means at least one child is itself a
rule.  Depth 2 introduces a third level, and so on.

**Important:** depth 0 does NOT mean the rule has no children.  A RE-PAIR
extension can reuse an existing rule *without* increasing the depth counter if
the rule appears as a direct child without further nesting.  Depth increases
only when a rule at level N *introduces* sub-rules at level N-1, producing a
strictly taller hierarchy.

## Expected results

On the corpus that motivated the article (8,395 actions after deduplication and
sleep-delay unification, min_uses=3):

- 216 rules
- Maximum composition depth: 3
- Distribution: 194 at depth 0 / 16 at depth 1 / 5 at depth 2 / 1 at depth 3
- 2,320 blocks retained (compressed sequence length)

**These specific figures are not reproducible from the available result JSON
files** (the intermediate run that produced them was not serialized).  The
closest available reference is g2_grammar.json (VPS, graph-refine/): 262 rules,
9,499 actions (before sleep-delay unification), depth_comp 228/28/5/1.  If you
run RE-PAIR on your own corpus, your numbers will differ; the structure to look
for is whether the depth distribution is concentrated at depth 0 (shallow
hierarchy) or spread (genuine multi-level composition).

## Usage

    from agentgraph.extract import build_hyperedges
    from agentgraph.repair import build_sequence, repair, measure_grammar

    edges = build_hyperedges(
        "~/.openclaw/agents/*/sessions/*.jsonl",
        normalize_sleep=True,   # unify sleep.3 == sleep.15 == sleep
    )
    seq = build_sequence(edges)
    compressed, rules = repair(seq, min_uses=3)
    report = measure_grammar(seq, compressed, rules)
    print(f"{report['rules_count']} rules, max depth {report['max_depth']}")
"""

from __future__ import annotations

import collections
import re
import time

# Turn boundary — ensures the grammar does not create rules that span turns.
BOUNDARY = "§TURN§"

# Numeric-only suffix pattern for sleep normalization.
_NUM_SUFFIX = re.compile(r"\.\d+$")


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequence(
    edges,                   # list[Hyperedge]
    normalize_sleep: bool = True,
) -> list[str]:
    """Build a flat action sequence from hyperedges, ordered by timestamp.

    Turn boundaries (``BOUNDARY``) are inserted between consecutive edges so
    that RE-PAIR cannot create rules that span turns.  This is the same
    convention used in the article's analysis.

    Parameters
    ----------
    edges:
        Output of ``agentgraph.extract.build_hyperedges``.
    normalize_sleep:
        If True, unifies ``sleep.3``, ``sleep.15``, etc. into ``sleep`` so
        that polling loops with different delay values are treated as one
        pattern.  This is what produces the article's 216-rule result.
    """
    dated = sorted(
        [e for e in edges if e.timestamp],
        key=lambda e: e.timestamp,
    )

    def maybe_normalize(atom: str) -> str:
        if normalize_sleep and ".exec.other_shell.sleep." in atom:
            # Remove the numeric delay suffix.
            return _NUM_SUFFIX.sub("", atom)
        return atom

    seq: list[str] = []
    for edge in dated:
        if seq:
            seq.append(BOUNDARY)
        seq.extend(maybe_normalize(a) for a in edge.atoms)

    return seq


# ---------------------------------------------------------------------------
# RE-PAIR core
# ---------------------------------------------------------------------------

def _count_pairs(
    seq: list[str],
    boundary: str = BOUNDARY,
) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a != boundary and b != boundary:
            counter[(a, b)] += 1
    return counter


def repair(
    seq: list[str],
    min_uses: int = 3,
    boundary: str = BOUNDARY,
    verbose: bool = False,
) -> tuple[list[str], dict[str, dict]]:
    """Run RE-PAIR on a sequence of symbols.

    Parameters
    ----------
    seq:
        Input sequence (output of ``build_sequence``).
    min_uses:
        Minimum number of times a pair must appear to be replaced.  3 is the
        value used in the article's analysis.
    boundary:
        Separator that cannot be part of any rule.
    verbose:
        Print progress every 20 iterations.

    Returns
    -------
    compressed:
        The sequence after all replacements.
    rules:
        ``{non_terminal: {"expansion": [symbol, symbol], "uses": int}}``
    """
    seq = list(seq)
    rules: dict[str, dict] = {}
    nt_counter = [0]

    def new_nt() -> str:
        s = f"§R{nt_counter[0]}§"
        nt_counter[0] += 1
        return s

    t0 = time.time()
    iteration = 0

    while True:
        counts = _count_pairs(seq, boundary)
        if not counts:
            break
        (best_a, best_b), best_count = counts.most_common(1)[0]
        if best_count < min_uses:
            break

        nt = new_nt()
        new_seq: list[str] = []
        i = 0
        replaced = 0
        while i < len(seq):
            if (
                i + 1 < len(seq)
                and seq[i] == best_a
                and seq[i + 1] == best_b
            ):
                new_seq.append(nt)
                i += 2
                replaced += 1
            else:
                new_seq.append(seq[i])
                i += 1

        if replaced < min_uses:
            break

        rules[nt] = {"expansion": [best_a, best_b], "uses": replaced}
        seq = new_seq
        iteration += 1

        if verbose and iteration % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  iter={iteration}  rules={len(rules)}  "
                f"seq_len={len(seq)}  max_freq={best_count}  t={elapsed:.1f}s",
                flush=True,
            )

    if verbose:
        print(
            f"  Done: {iteration} iterations, {len(rules)} rules, "
            f"seq_len={len(seq)}, {time.time() - t0:.1f}s"
        )

    return seq, rules


# ---------------------------------------------------------------------------
# Grammar analysis
# ---------------------------------------------------------------------------

def _rule_composition_depth(
    symbol: str,
    rules: dict[str, dict],
    cache: dict[str, int] | None = None,
) -> int:
    """Composition depth of a symbol.

    Depth 0: the symbol is a terminal, or expands only into terminals.
    Depth N: the symbol expands into at least one symbol of depth N-1.

    This differs from a simple recursion count — it measures the *height* of
    the derivation tree for this rule, counting only the rule-application
    steps, not the terminal symbols.
    """
    if cache is None:
        cache = {}
    if symbol in cache:
        return cache[symbol]
    if symbol not in rules:
        cache[symbol] = 0
        return 0

    child_depths = [
        _rule_composition_depth(s, rules, cache)
        for s in rules[symbol]["expansion"]
        if s in rules
    ]
    # Depth = 1 + max child depth if any child is itself a rule; else 0.
    depth = (1 + max(child_depths)) if child_depths else 0
    cache[symbol] = depth
    return depth


def measure_grammar(
    original_seq: list[str],
    compressed_seq: list[str],
    rules: dict[str, dict],
    boundary: str = BOUNDARY,
) -> dict:
    """Compute summary statistics for a RE-PAIR grammar.

    The most informative output is ``depth_distribution``.  A distribution
    concentrated at depth 0 means the grammar captures repetition but not
    composition — pairs of atoms recur, but they do not combine into deeper
    structures.  Depth ≥ 2 indicates genuine multi-level factorization.

    Parameters
    ----------
    original_seq, compressed_seq, rules:
        Outputs of ``build_sequence`` and ``repair``.

    Returns
    -------
    dict with keys:
        ``rules_count``, ``orig_len``, ``compressed_len``, ``max_depth``,
        ``depth_distribution``, ``compression_ratio``.
    """
    depth_cache: dict[str, int] = {}
    depths = {nt: _rule_composition_depth(nt, rules, depth_cache) for nt in rules}

    depth_dist: dict[int, int] = collections.Counter(depths.values())
    max_depth = max(depths.values()) if depths else 0

    orig_len = len(original_seq)
    comp_len = len(compressed_seq)
    grammar_size = comp_len + 2 * len(rules)
    ratio = (orig_len - grammar_size) / orig_len if orig_len else 0.0

    n_actions = sum(1 for s in original_seq if s != boundary)

    return {
        "n_actions": n_actions,
        "rules_count": len(rules),
        "orig_len": orig_len,
        "compressed_len": comp_len,
        "grammar_size": grammar_size,
        "compression_ratio": round(ratio, 4),
        "max_depth": max_depth,
        "depth_distribution": dict(sorted(depth_dist.items())),
        "note": (
            "depth_distribution is the key output. "
            "Depth 0 = pair of terminals; depth N = rule uses at least one depth-(N-1) child. "
            "Depth 0 does NOT mean the rule has no children — a rule can reuse another rule "
            "as a direct expansion child without increasing the depth counter. "
            "Depth increases only when a rule composes sub-rules one level deeper."
        ),
    }
