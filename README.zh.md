# agentic-graph-notebooks

从你自己的智能体执行轨迹构建行动图，然后检验这个图是否承载信息。

本仓库不携带任何数据。一切都在本地、在你的轨迹上运行。

[English](README.md) · [Français](README.fr.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md)

---

## 范围

**1. 图的形态。** 只有当邻域在不同节点之间存在差异时，它才承载信息。有两种失败模式：过于稀疏，平均节点几乎不与其余节点共现，可比较的共现太少；过于稠密，平均节点与其余节点中很大一部分相连，邻域不再具有区分力。`metrics()` 报告密度、平均度和度熵，并对两种情形分别给出提示。密度是平均度相对于其余节点数的占比，因此是一个全局平均值——请同时读度熵，因为全局稀疏的图里仍可能有一个节点连接所有其他节点。

**2. 模式的可解释性。** 每个模式都会回溯到产生它的命令。区分平凡模式（`cd` 后接 `ls`）、解析产生的伪影和可用模式，靠的是人工检查，而不是分数。

**3. 超出随机水平的重复。** 同一智能体的两个时间窗口通常会共享词汇：同一台机器、同一批工具、同一个操作者。因此单独的持久性原始分数无法解释。

---

## 随机基线

`temporal_stability()` 把同样的行动在同样的窗口中重新打散——20 次抽样，固定随机种子——并将观测分数与打散后的分布并列输出。

在推动这项工作的语料上，某个邻域持久性指标的中位数达到了 1.000。打散基线同样返回 1.000：行动词表太窄，无法产生变化，因此该指标不承载任何信号。

---

## 安装

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

`agentgraph/` 只依赖标准库。社区发现、Spearman 相关、Tarjan 强连通分量和 Kruskal-Wallis 都在仓库内实现。`requirements.txt` 只安装 JupyterLab。

---

## 输入格式

Notebook 读取 JSONL 会话文件，每行一个事件，其中包含 `toolCall` 块。OpenClaw 的格式可直接使用：

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
N=<节点> · edges=<边> · density=<密度> · mean degree=<平均度> · entropy=<熵>
→ INFORMATIVE REGIME — density between 0.002 and 0.15
```

对于其他智能体，请改写 `agentgraph/extract.py` 中的 `iter_tool_calls()`。它是唯一依赖轨迹格式的入口：下游全部消费 `ToolCall` 对象或 `(call, command)` 对的迭代器。一个 Claude Code 适配器只需约 20 行，映射 `tool_use` 块及其 `input` 字段。

| Notebook | 测量内容 |
|---|---|
| `01-extract` | 工具分布，shell 命令占比 |
| `02-granularity` | 不同粒度变体下的密度与熵 |
| `03-motifs` | 高频模式，回溯到对应命令 |
| `04-stability` | 模式持久性与打散基线的对比 |
| `05-loops` | 显式 shell 循环，以及对分段图不可见的比例 |
| `06-condensation` | SCC 缩合与反馈的局部性 |
| `07-repair` | RE-PAIR 语法：重复子序列及其组合深度 |
| `08-read-budget` | 前一个行动是否预测 `head -n N` |

---

## 分段

复合命令占主导：大约五条中有四条包含 `&&`、`;`、`|` 或换行。这些操作符是模型自己写下的边界，因此分段就在这些位置切分。

用正则分段会连引号内的字符串一起切开。`grep -E 'a|b|c'` 会得到三个片段，每个片段都产生一个背后没有真实命令的原子（`awk.print1`、`awk.print2`……）。在真实语料上，这会显著膨胀词表。`split_segments()` 使用带 `punctuation_chars` 的 `shlex`，它尊重引号——如果你改写这段代码，请保留这个性质。

内联 Python 用 `ast` 解析。shell 没有等价物，只能用启发式处理。

---

## 注意事项

**轨迹中含有明文密钥**：Bearer token、API key、环境变量。`mask_secrets()` 过滤常见形式，并应用于模式输出。它降低风险，但不能消除风险。分享输出前请先复核。

以下两个效应 notebook 会检测，但不会替你修正：

- **会话自相关。** 同一会话内的命令彼此相似。如果单个会话主导了一个窗口，测到的稳定性反映的是该会话的内部一致性。`describe_windows()` 在超过 40% 时给出警告。
- **智能体混合。** 如果窗口对应不同的智能体，测到的是身份差异，而非随时间的漂移。做稳定性分析前请先筛选到单个智能体。

---

## 局限

本协议测量图的形态和模式的重复。它不评估任何下游任务：图形态良好，并不能证明预测、路由或推荐会因此改善。

它记录智能体采取了哪些行动，但不记录触发条件和前置条件。可重放的过程需要这两者。

---

## 许可证

MIT。配套文章：[casys.ai/zh/blog/graph-engineering-multi-agentic-systems](https://casys.ai/zh/blog/graph-engineering-multi-agentic-systems)
