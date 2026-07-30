"""Syntactic inventory of explicit loops in shell commands.

The segmentation that builds the action graph drops loop keywords: `for`, `while`
and `until` are in `LANG_KEYWORDS`, so is the `do` that opens the body.  Two
consequences — the iteration itself leaves no trace, and the first command of the
body, which `split_segments` returns glued to `do`, leaves no atom either.  Later
commands in the body do produce atoms, with nothing marking them as repeated.  On
the corpus behind the companion article, 91 of the 103 commands carrying an
explicit loop construct fall in that blind spot.

This module counts them from the raw command text instead, along three axes:

1. **How many commands contain an explicit loop?**
   Counts `for`, `while`, `until`, `xargs` and `find -exec` constructs
   separately.

2. **How many of those loops are invisible to the segment graph?**
   An "invisible" loop is a `for`/`while`/`until` keyword construct: the keyword
   is dropped, so nothing in the graph says the body ran more than once.  A
   `xargs` or `find -exec` loop is visible — the binary itself becomes an atom.

3. **Are loops local?**
   Decomposes each loop-bearing command into prefix / loop-body / suffix, and
   reports how often the prefix or suffix carries *another* loop.  A low rate
   means loops sit at one level rather than nesting.

## Methodological note

This is a syntactic inventory, not a graph traversal.  It sees no feedback that
is not written as a loop keyword: a `curl → jq → curl` cycle in the transition
graph is real feedback this module does not count.  Read it next to
`condensation.py`, which measures the graph side.

## Usage

    from agentgraph.extract import iter_shell_commands
    from agentgraph.loops import classify_loops, envelope_analysis, invisibility

    records = list(iter_shell_commands("~/.openclaw/agents/*/sessions/*.jsonl"))
    counts, loop_cmds = classify_loops(records)
    ratio = invisibility(counts)
    envelope = envelope_analysis(loop_cmds)
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Iterator

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Heredoc masking — avoids false positives inside heredoc bodies.
_HEREDOC = re.compile(
    r"<<-?\s*['\"]?(\w+)['\"]?.*?^\s*\1\s*$",
    re.S | re.M,
)

# Keyword-based constructs (invisible to the segment graph).
_FOR_IN    = re.compile(r"\bfor\b[^#\n]*?\bin\b[^;]+?;?\s*do\b",    re.DOTALL)
_FOR_ARITH = re.compile(r"\bfor\s*\(\(",                              re.DOTALL)
_WHILE     = re.compile(r"\bwhile\b[^;#\n]+?;?\s*do\b",              re.DOTALL)
_UNTIL     = re.compile(r"\buntil\b[^;#\n]+?;?\s*do\b",              re.DOTALL)

# Parsable loop constructs (xargs / find -exec do appear in the segment graph).
_XARGS     = re.compile(r"\bxargs\b")
_FIND_EXEC = re.compile(r"\bfind\b.+?-exec\b",                        re.DOTALL)

# Loop body extractor: outermost do…done.
_LOOP_HDR  = re.compile(
    r"\b(for|while|until)\b([^#\n]*?)(?:;?\s*)do\b",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LoopRecord:
    """A command found to contain at least one explicit loop construct."""
    command: str
    forms: set[str]          # e.g. {"for_in", "xargs"}

    @property
    def is_invisible(self) -> bool:
        """True when the loop construct leaves no atom in the segment graph."""
        return bool(self.forms & {"for_in", "for_arith", "while", "until"})


@dataclass
class EnvelopeRecord:
    """Decomposition of a loop-bearing command into prefix / body / suffix."""
    prefix_atoms: list[str]
    loop_type: str
    suffix_atoms: list[str]
    prefix_has_loop: bool = False
    suffix_has_loop: bool = False


# ---------------------------------------------------------------------------
# Loop classification
# ---------------------------------------------------------------------------

def _mask(command: str) -> str:
    return _HEREDOC.sub(" __HEREDOC__ ", command)


def classify_loops(
    shell_iter,   # iterable of (ToolCall, str) pairs, e.g. from iter_shell_commands
) -> tuple[dict[str, int], list[LoopRecord]]:
    """Classify all loop forms found in the shell commands.

    Parameters
    ----------
    shell_iter:
        Iterable of ``(ToolCall, command_str)`` pairs.  The simplest source is
        ``agentgraph.extract.iter_shell_commands(sessions_glob)``.

    Returns
    -------
    counts:
        ``{form_name: total_count}`` across all commands.
    loop_records:
        One ``LoopRecord`` per command that contains at least one loop.

    Form names
    ----------
    ``for_in``     — ``for VAR in LIST; do … done``
    ``for_arith``  — ``for ((i=0; i<N; i++))``
    ``while``      — ``while COND; do … done``
    ``until``      — ``until COND; do … done``
    ``xargs``      — ``xargs [opts] CMD``
    ``find_exec``  — ``find … -exec … \\;`` or ``+``

    Invisible vs. parsable
    ----------------------
    Keyword forms (``for_in``, ``for_arith``, ``while``, ``until``) do not
    produce atoms in the segment graph — the keyword is in ``LANG_KEYWORDS``
    and is skipped during normalization.  ``xargs`` and ``find -exec`` are
    segment-graph-visible because they appear as a binary name.
    """
    counts: dict[str, int] = collections.Counter()
    loop_records: list[LoopRecord] = []

    for _call, command in shell_iter:
        stripped = _mask(command)
        found: set[str] = set()

        if _FOR_ARITH.search(stripped):
            found.add("for_arith")
        elif _FOR_IN.search(stripped):
            found.add("for_in")
        if _WHILE.search(stripped):
            found.add("while")
        if _UNTIL.search(stripped):
            found.add("until")
        if _XARGS.search(stripped):
            found.add("xargs")
        if _FIND_EXEC.search(stripped):
            found.add("find_exec")

        for f in found:
            counts[f] += 1

        if found:
            loop_records.append(LoopRecord(command=command, forms=found))

    return dict(counts), loop_records


def invisibility(counts: dict[str, int]) -> dict:
    """Compute the invisible vs. parsable split.

    "Invisible" means the loop construct uses a shell keyword (``for``,
    ``while``, ``until``) and therefore leaves **no atom** in the segment
    graph.  "Parsable" means the construct IS visible in the graph (``xargs``,
    ``find -exec``).

    The article reports 91 / 103 (88.3%) invisible on a 2,656-command corpus.
    On a small corpus the ratio will vary; what matters is whether it is
    substantial, and whether you account for it when interpreting graph metrics.
    """
    invisible = sum(counts.get(k, 0) for k in ("for_in", "for_arith", "while", "until"))
    parsable  = sum(counts.get(k, 0) for k in ("xargs", "find_exec"))
    total = invisible + parsable
    return {
        "invisible": invisible,
        "parsable":  parsable,
        "total":     total,
        "invisible_pct": round(100 * invisible / total, 1) if total else 0.0,
        "warning": (
            "The segment graph therefore undercounts loops. "
            "Graph metrics (density, motifs, SCC sizes) reflect only the "
            "parsable fraction."
        ) if total > 0 else "No loops found.",
    }


# ---------------------------------------------------------------------------
# Envelope analysis
# ---------------------------------------------------------------------------

def _extract_outermost_loop(command: str):
    """Extract prefix / loop-type / body / suffix from the outermost loop.

    Returns (prefix_text, loop_type, spec, body_text, suffix_text) or None.
    Uses a do/done counter to handle nested loops.
    """
    text = _mask(command)
    match = _LOOP_HDR.search(text)
    if not match:
        return None

    loop_type = match.group(1)
    start = match.end()
    depth = 1
    i = start

    while i < len(text) and depth > 0:
        do_m   = re.search(r"\bdo\b",   text[i:])
        done_m = re.search(r"\bdone\b", text[i:])
        if not done_m:
            break
        if do_m and do_m.start() < done_m.start():
            depth += 1
            i += do_m.end()
        else:
            depth -= 1
            if depth == 0:
                body   = text[start : i + done_m.start()].strip()
                suffix = text[i + done_m.end() :].strip()
                prefix = text[:match.start()].strip()
                return prefix, loop_type, body, suffix
            i += done_m.end()

    return None


def _has_loop_kw(text: str) -> bool:
    stripped = _mask(text)
    return bool(
        _FOR_IN.search(stripped)
        or _FOR_ARITH.search(stripped)
        or _WHILE.search(stripped)
        or _UNTIL.search(stripped)
        or _XARGS.search(stripped)
        or _FIND_EXEC.search(stripped)
    )


def envelope_analysis(loop_records: list[LoopRecord]) -> dict:
    """Decompose loop-bearing commands into prefix / loop / suffix.

    The key question: are loops *local*?  A loop is local when neither its
    prefix nor its suffix contains another loop.  If they do, the command
    nests loops — a qualitatively different and rarer pattern.

    The article reports:
    - 90 of 103 loop commands decompose into prefix / loop / suffix.
    - 0 of 68 prefixes contain another loop.
    - 3 of 45 suffixes contain another loop (6.7%).
    - Loops are therefore structurally local in this corpus.

    Returns
    -------
    A dict with counts, form distribution, and the locality verdict.
    """
    forms: dict[str, int] = collections.Counter()
    total_decomposed = 0
    total_failed = 0
    prefixes_with_loop = 0
    suffixes_with_loop = 0
    n_prefixes = 0
    n_suffixes = 0

    for rec in loop_records:
        result = _extract_outermost_loop(rec.command)
        if result is None:
            total_failed += 1
            continue

        prefix, loop_type, body, suffix = result
        has_prefix = bool(prefix.strip())
        has_suffix = bool(suffix.strip())
        total_decomposed += 1

        if has_prefix and has_suffix:
            forms["prefix_loop_suffix"] += 1
        elif has_prefix:
            forms["prefix_loop"] += 1
        elif has_suffix:
            forms["loop_suffix"] += 1
        else:
            forms["loop_only"] += 1

        if has_prefix:
            n_prefixes += 1
            if _has_loop_kw(prefix):
                prefixes_with_loop += 1
        if has_suffix:
            n_suffixes += 1
            if _has_loop_kw(suffix):
                suffixes_with_loop += 1

    suffix_loop_pct = (
        round(100 * suffixes_with_loop / n_suffixes, 1) if n_suffixes else 0.0
    )
    loops_are_local = (
        prefixes_with_loop == 0
        and suffix_loop_pct < 15.0
        and total_decomposed > 0
    )

    return {
        "total_loop_commands": len(loop_records),
        "decomposed": total_decomposed,
        "not_decomposed": total_failed,
        "form_distribution": dict(forms),
        "n_prefixes": n_prefixes,
        "n_suffixes": n_suffixes,
        "prefixes_with_nested_loop": prefixes_with_loop,
        "suffixes_with_nested_loop": suffixes_with_loop,
        "suffix_nested_loop_pct": suffix_loop_pct,
        "loops_are_local": loops_are_local,
        "interpretation": (
            "Loops are local: neither prefixes nor suffixes routinely contain "
            "another loop.  The recursive pressure is contained."
            if loops_are_local else
            "Loops are NOT consistently local: nested-loop patterns appear "
            "frequently.  Inspect individual commands before concluding."
        ),
    }
