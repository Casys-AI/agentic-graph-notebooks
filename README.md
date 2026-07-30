# agentic-graph-notebooks

**Turn your own agent execution traces into a graph — then measure whether that graph means anything.**

This repository ships no data. Everything runs on *your* traces, on *your* machine.

[English](README.md) · [Français](README.fr.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md)

---

## Why

In July 2026 the "loops vs graphs" debate produced a great deal of assertion and very little measurement. These notebooks do the opposite: a reproducible protocol answering three questions about your own agents.

**1. Does the graph have the right shape?**

A graph is only useful if knowing a node's neighbours tells you something *specific to that node*. There are two opposite ways to fail. Too sparse: most nodes have no neighbours, so there is nothing to look at. Too dense: every node touches nearly every other, so A's neighbourhood is the same as B's and discriminates nothing. Information lives only in between.

**2. Does the graph mean anything?**

A statistical motif with no human name proves nothing. These notebooks always trace motifs back to the real commands so you can judge for yourself — separating usable motifs from real-but-trivial ones (`cd` then `ls` is a genuine motif and a worthless one) and from pure artefacts.

**3. Does the graph repeat?**

This is the question that decides whether "procedural memory" means anything. It is also where nearly everyone goes wrong, because answering it honestly requires a random baseline.

---

## The random baseline, and why it changes everything

Two time windows from the same agent necessarily share vocabulary: same machine, same tools, same person. A raw persistence score is therefore **uninterpretable** — and it will be high, therefore convincing.

The only question that matters is: *beyond chance?*

These notebooks reshuffle the same actions across the same windows, twenty times, and compare. On the corpus that motivated this work, one neighbourhood-persistence measure reached a median of **1.000** — a spectacular result, until the random baseline also returned **1.000**. The action repertoire was simply too narrow to vary.

Without that control, the number would have been published as a discovery.

---

## Install

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

No heavy dependencies. Community detection and rank correlation are implemented in pure Python; `networkx` and `matplotlib` are used for display only.

---

## Use

The notebooks expect JSONL sessions — one event per line, with `toolCall` blocks. OpenClaw's format works as-is:

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
# (actual numbers depend on your corpus)
N=<nodes> · edges=<edges> · density=<density> · mean degree=<mean_deg> · entropy=<entropy>
→ INFORMATIVE REGIME — neighbourhoods carry information
```

For another agent format, adapt `iter_tool_calls()` in `agentgraph/extract.py`. It is the only entry point you need to change.

| Notebook | Question |
|---|---|
| `01-extract` | What is in your traces? Tool distribution, share of shell |
| `02-granularity` | At what granularity does your graph stop being degenerate? |
| `03-motifs` | Do your motifs have names? |
| `04-stability` | Do your routines repeat beyond chance? |
| `05-loops` | How many commands contain explicit loops — and how many are invisible to the graph? |
| `06-condensation` | SCC condensation: how local is the feedback in your graph? |
| `07-repair` | RE-PAIR grammar: recurring subsequences and how deep they compose |
| `08-read-budget` | Reading budget: does the preceding action predict `head -n N`? |

---

## One parsing detail that matters more than it looks

Agents that write shell do not produce *actions*, they produce *programs*. Roughly four out of five compound commands contain `&&`, `;`, `|` or a newline. Those operators are the boundaries the model itself wrote — so we split on them.

Splitting naively with a regex is a trap: it also cuts inside quoted strings. `grep -E 'a|b|c'` becomes three "commands", and each fragment produces a phantom atom (`awk.print1`, `awk.print2`, …) with no real command behind it. On real corpora this single bug manufactures a substantial fraction of the vocabulary from thin air.

`split_segments()` uses `shlex` with `punctuation_chars`, which respects quotes. If you adapt this code, keep that property.

Inline Python gets a real AST rather than regular expressions — a decisive advantage over shell, which only yields to heuristics.

---

## Two warnings

**Your traces contain secrets.** Not accidentally — systematically: bearer tokens, API keys, environment variables in clear text. `mask_secrets()` filters common shapes and is applied to all motif output, but it reduces the risk without eliminating it. **Review any output before sharing it.**

**Two methodological traps** the notebooks flag but cannot fix for you:

- *Session autocorrelation* — commands within one session necessarily resemble each other. If a single session dominates a window, the "stability" you measure is just that session's internal coherence. `describe_windows()` warns past 40%.
- *Agent mixing* — if your time windows correspond to different agents, you are measuring a difference in identity, not drift over time. Filter to one agent before running stability analysis.

---

## What this protocol does not do

It measures the **shape** of the graph and the **repetition** of its motifs. It validates no downstream task: a well-formed graph is not evidence that it improves a prediction, a routing decision or a recommendation. Necessary, not sufficient.

It also describes *what* an agent did, never *when* to trigger it or which precondition to check. That is the boundary between a graph and a replayable procedure.

---

## License

MIT. Companion article: [casys.ai/blog/graph-engineering-multi-agentic-systems](https://casys.ai/blog/graph-engineering-multi-agentic-systems)
