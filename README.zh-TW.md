# agentic-graph-notebooks

從你自己的智能體執行軌跡建立動作圖，然後檢驗這個圖是否承載資訊。

本儲存庫不附帶任何資料。一切都在本機、在你的軌跡上執行。

[English](README.md) · [Français](README.fr.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md)

---

## 範圍

**1. 圖的形態。** 只有當鄰域在不同節點之間有所差異時，它才承載資訊。有兩種失敗模式：過於稀疏，平均節點幾乎不與其餘節點共現，可比較的共現太少；過於稠密，平均節點與其餘節點中很大一部分相連，鄰域不再具有區分力。`metrics()` 會回報密度、平均度與度熵，並針對這兩種情形分別提示。密度是平均度相對於其餘節點數的占比，因此是一個全域平均值——請一併讀度熵，因為全域稀疏的圖裡仍可能有一個節點連向所有其他節點。

**2. 模式的可解讀性。** 每個模式都會回溯到產生它的指令。要區分平凡模式（`cd` 接著 `ls`）、解析產生的假影與可用模式，靠的是人工檢視，而不是分數。

**3. 超出隨機水準的重複。** 同一智能體的兩個時間視窗通常會共享詞彙：同一台機器、同一批工具、同一位操作者。因此單獨的持久性原始分數無法解讀。

---

## 隨機基線

`temporal_stability()` 會把同樣的動作在同樣的視窗中重新打散——20 次抽樣，固定隨機種子——並將觀測分數與打散後的分布並列輸出。

在推動這項工作的語料庫上，某個鄰域持久性指標的中位數達到 1.000。打散基線同樣回傳 1.000：動作詞彙表太窄，無法產生變化，因此該指標不承載任何訊號。

---

## 安裝

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

`agentgraph/` 只依賴標準函式庫。社群偵測、Spearman 秩相關、Tarjan 強連通分量與 Kruskal-Wallis 都在儲存庫內實作。`requirements.txt` 只安裝 JupyterLab。

---

## 輸入格式

notebook 讀取 JSONL 會話檔案，每行一個事件，其中包含 `toolCall` 區塊。OpenClaw 的格式可以直接使用：

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
N=<節點數> · edges=<邊數> · density=<密度> · mean degree=<平均度> · entropy=<熵>
→ INFORMATIVE REGIME — density between 0.002 and 0.15
```

若要用於其他智能體，請改寫 `agentgraph/extract.py` 中的 `iter_tool_calls()`。這是唯一依賴軌跡格式的入口：下游全部消費 `ToolCall` 物件或 `(call, command)` 配對的迭代器。一個 Claude Code 轉接器只需約 20 行，把 `tool_use` 區塊與其 `input` 欄位對應過去。

| Notebook | 測量內容 |
|---|---|
| `01-extract` | 工具分布、shell 指令佔比 |
| `02-granularity` | 各粒度變體下的密度與熵 |
| `03-motifs` | 高頻模式，回溯到對應指令 |
| `04-stability` | 模式持久性與打散基線的對照 |
| `05-loops` | 顯式 shell 迴圈，以及對分段圖不可見的比例 |
| `06-condensation` | SCC 凝聚與回饋的局部性 |
| `07-repair` | RE-PAIR 語法：重複子序列及其組合深度 |
| `08-read-budget` | 前一個動作是否能預測 `head -n N` |

---

## 分段

複合指令佔多數：大約五條中有四條含有 `&&`、`;`、`|` 或換行。這些運算子是模型自己寫下的邊界，因此分段就在這些位置切開。

用正規表示式分段會連引號內的字串一起切開。`grep -E 'a|b|c'` 會得到三個片段，每個片段都產生一個背後沒有真實指令的原子（`awk.print1`、`awk.print2`……）。在真實語料庫上，這會顯著膨脹詞彙表。`split_segments()` 使用帶 `punctuation_chars` 的 `shlex`，它會尊重引號——如果你改寫這段程式碼，請保留這個性質。

內嵌 Python 以 `ast` 解析。shell 沒有對等機制，只能以啟發式處理。

---

## 注意事項

**軌跡中含有明文機密**：Bearer token、API key、環境變數。`mask_secrets()` 會過濾常見形式，並套用於模式輸出。它降低風險，但無法消除風險。分享輸出前請先複查。

以下兩個效應 notebook 會偵測，但不會替你修正：

- **會話自相關。** 同一會話內的指令彼此相似。若單一會話主導了一個視窗，測到的穩定性反映的是該會話的內部一致性。`describe_windows()` 在超過 40% 時提出警告。
- **智能體混雜。** 若視窗對應到不同的智能體，測到的是身分差異，而非隨時間的漂移。進行穩定性分析前請先篩選到單一智能體。

---

## 限制

本協定測量圖的形態與模式的重複。它不評估任何下游任務：圖形態良好，並不能證明預測、路由或推薦會因此改善。

它記錄智能體採取了哪些動作，但不記錄觸發條件與前置條件。可重放的流程需要這兩者。

---

## 授權條款

MIT。配套文章：[casys.ai/zh-TW/blog/graph-engineering-multi-agentic-systems](https://casys.ai/zh-TW/blog/graph-engineering-multi-agentic-systems)
