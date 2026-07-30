# agentic-graph-notebooks

Build an action graph from your own agent execution traces, then test whether that graph carries information.

No data is bundled. Everything runs locally, on your traces.

[English](README.md) · [Français](README.fr.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md)

---

## Scope

**1. Graph shape.** A neighbourhood is informative only if it differs from one node to the next. Two failure modes: too sparse, where the average node touches almost none of the others and there is little co-occurrence to compare, and too dense, where the average node touches a large share of them and neighbourhoods stop discriminating. `metrics()` reports density, mean degree and degree entropy, and flags both regimes. Density is the mean degree as a fraction of the other nodes, so it is a global average — read degree entropy alongside it, since a globally sparse graph can still hold one node wired to everything.

**2. Motif interpretability.** Every motif is traced back to the commands that produced it. Trivial motifs (`cd` then `ls`) and parsing artefacts are separated from usable ones by inspection, not by score.

**3. Repetition above chance.** Two time windows from the same agent tend to share vocabulary: same machine, same tools, same operator. A raw persistence score is not interpretable on its own.

---

## Random baseline

`temporal_stability()` reshuffles the same actions across the same windows — 20 trials, fixed seed — and reports the observed score next to the shuffled distribution.

On the corpus this was built for, one neighbourhood-persistence measure reached a median of 1.000. The shuffled baseline also returned 1.000: the action repertoire was too narrow to vary, and the measure carried no signal.

---

## Install

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

`agentgraph/` depends on the standard library only. Community detection, rank correlation, Tarjan SCC and Kruskal-Wallis are implemented in-tree. `requirements.txt` installs JupyterLab and nothing else.

---

## Input format

The notebooks read JSONL sessions, one event per line, containing `toolCall` blocks. OpenClaw traces work unchanged:

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
N=<nodes> · edges=<edges> · density=<density> · mean degree=<mean_deg> · entropy=<entropy>
→ INFORMATIVE REGIME — density between 0.002 and 0.15
```

For any other agent, adapt `iter_tool_calls()` in `agentgraph/extract.py`. It is the only entry point that depends on the trace format: everything downstream consumes `ToolCall` objects or an iterator of `(call, command)` pairs. A Claude Code adapter maps `tool_use` blocks and their `input` field in about 20 lines.

| Notebook | Measures |
|---|---|
| `01-extract` | Tool distribution, share of shell commands |
| `02-granularity` | Density and entropy across granularity variants |
| `03-motifs` | Frequent motifs, traced back to their commands |
| `04-stability` | Motif persistence against a shuffled baseline |
| `05-loops` | Explicit shell loops, and the fraction invisible to a segment-level graph |
| `06-condensation` | SCC condensation and feedback locality |
| `07-repair` | RE-PAIR grammar: recurring subsequences and composition depth |
| `08-read-budget` | Whether the preceding action predicts `head -n N` |

---

## Segmentation

Compound commands dominate: roughly four in five contain `&&`, `;`, `|` or a newline. Those operators are boundaries the model wrote itself, so segmentation splits on them.

A regex split also cuts inside quoted strings. `grep -E 'a|b|c'` yields three fragments, and each one produces an atom (`awk.print1`, `awk.print2`, …) with no command behind it. On real corpora this inflates the vocabulary substantially. `split_segments()` uses `shlex` with `punctuation_chars`, which respects quoting — keep that property if you adapt the code.

Inline Python is parsed with `ast`. Shell has no equivalent and is handled heuristically.

---

## Warnings

**Traces contain secrets in clear text**: bearer tokens, API keys, environment variables. `mask_secrets()` filters common shapes and is applied to motif output. It reduces the risk without eliminating it. Review output before sharing it.

Two effects the notebooks detect but do not correct:

- **Session autocorrelation.** Commands within one session resemble each other. If a single session dominates a window, the measured stability reflects that session's internal coherence. `describe_windows()` warns above 40%.
- **Agent mixing.** If windows map to different agents, the measurement captures a difference in identity rather than drift over time. Filter to one agent before running stability analysis.

---

## Limits

The protocol measures graph shape and motif repetition. It evaluates no downstream task: a well-formed graph is not evidence that prediction, routing or recommendation improve.

It records which actions an agent took, without their trigger conditions or preconditions. A replayable procedure needs both.

---

## License

MIT. Companion article: [casys.ai/blog/graph-engineering-multi-agentic-systems](https://casys.ai/blog/graph-engineering-multi-agentic-systems)
