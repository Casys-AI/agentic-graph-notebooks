# agentic-graph-notebooks

Construis un graphe d'actions à partir de tes propres traces d'exécution d'agents, puis teste si ce graphe porte de l'information.

Aucune donnée n'est fournie. Tout s'exécute en local, sur tes traces.

[English](README.md) · [Français](README.fr.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md)

---

## Périmètre

**1. Forme du graphe.** Un voisinage n'est informatif que s'il diffère d'un nœud à l'autre. Deux modes d'échec : trop épars, où le nœud moyen ne touche presque aucun des autres et où il reste peu de cooccurrences à comparer, et trop dense, où le nœud moyen en touche une large part et les voisinages ne discriminent plus. `metrics()` rapporte densité, degré moyen et entropie de degré, et signale les deux régimes. La densité est le degré moyen exprimé en fraction des autres nœuds, donc une moyenne globale — lis l'entropie de degré à côté, car un graphe globalement épars peut abriter un nœud relié à tout.

**2. Interprétabilité des motifs.** Chaque motif est remonté jusqu'aux commandes qui l'ont produit. Les motifs triviaux (`cd` puis `ls`) et les artefacts de parsing se distinguent des motifs exploitables par inspection, pas par score.

**3. Répétition au-delà du hasard.** Deux fenêtres temporelles du même agent partagent le plus souvent du vocabulaire : même machine, mêmes outils, même opérateur. Un score de persistance brut n'est pas interprétable seul.

---

## Ligne de base aléatoire

`temporal_stability()` redistribue les mêmes actions dans les mêmes fenêtres — 20 tirages, graine fixée — et rapporte le score observé à côté de la distribution obtenue par mélange.

Sur le corpus qui a motivé ce travail, une mesure de persistance de voisinage a atteint une médiane de 1,000. La ligne de base par mélange a retourné 1,000 elle aussi : le répertoire d'actions était trop étroit pour varier, et la mesure ne portait aucun signal.

---

## Installation

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

`agentgraph/` ne dépend que de la bibliothèque standard. Détection de communautés, corrélation de Spearman, SCC de Tarjan et Kruskal-Wallis sont implémentés dans l'arbre. `requirements.txt` n'installe que JupyterLab.

---

## Format d'entrée

Les notebooks lisent des sessions JSONL, un événement par ligne, contenant des blocs `toolCall`. Les traces OpenClaw fonctionnent tel quel :

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
N=<nœuds> · edges=<arêtes> · density=<densité> · mean degree=<degré moyen> · entropy=<entropie>
→ INFORMATIVE REGIME — density between 0.002 and 0.15
```

Pour tout autre agent, adapte `iter_tool_calls()` dans `agentgraph/extract.py`. C'est le seul point d'entrée qui dépend du format de trace : tout l'aval consomme des objets `ToolCall` ou un itérateur de paires `(call, command)`. Un adaptateur Claude Code mappe les blocs `tool_use` et leur champ `input` en une vingtaine de lignes.

| Notebook | Mesure |
|---|---|
| `01-extract` | Distribution des outils, part des commandes shell |
| `02-granularity` | Densité et entropie selon les variantes de granularité |
| `03-motifs` | Motifs fréquents, remontés jusqu'à leurs commandes |
| `04-stability` | Persistance des motifs face à une ligne de base par mélange |
| `05-loops` | Boucles shell explicites, et la fraction invisible à un graphe de segments |
| `06-condensation` | Condensation SCC et localité de la rétroaction |
| `07-repair` | Grammaire RE-PAIR : sous-séquences récurrentes et profondeur de composition |
| `08-read-budget` | Si l'action précédente prédit `head -n N` |

---

## Segmentation

Les commandes composées dominent : environ quatre sur cinq contiennent `&&`, `;`, `|` ou un retour à la ligne. Ces opérateurs sont des frontières que le modèle a écrites lui-même, la segmentation coupe donc à ces endroits.

Une segmentation par regex coupe aussi à l'intérieur des chaînes entre guillemets. `grep -E 'a|b|c'` donne trois fragments, et chacun produit un atome (`awk.print1`, `awk.print2`, …) sans commande derrière. Sur des corpus réels, cela gonfle le vocabulaire de façon substantielle. `split_segments()` utilise `shlex` avec `punctuation_chars`, ce qui respecte les guillemets — conserve cette propriété si tu adaptes le code.

Le Python inline est parsé avec `ast`. Le shell n'a pas d'équivalent et reste traité par heuristiques.

---

## Avertissements

**Les traces contiennent des secrets en clair** : Bearer tokens, clés API, variables d'environnement. `mask_secrets()` filtre les formes courantes et s'applique aux sorties de motifs. Cela réduit le risque sans l'éliminer. Relis les sorties avant de les partager.

Deux effets que les notebooks détectent sans les corriger :

- **Autocorrélation de session.** Les commandes d'une même session se ressemblent. Si une seule session domine une fenêtre, la stabilité mesurée reflète la cohérence interne de cette session. `describe_windows()` avertit au-delà de 40 %.
- **Mélange d'agents.** Si les fenêtres correspondent à des agents différents, la mesure capte une différence d'identité plutôt qu'une dérive dans le temps. Filtre sur un seul agent avant l'analyse de stabilité.

---

## Limites

Le protocole mesure la forme du graphe et la répétition de ses motifs. Il n'évalue aucune tâche aval : un graphe bien formé ne prouve pas qu'une prédiction, un routage ou une recommandation s'améliorent.

Il enregistre les actions qu'un agent a prises, sans leurs conditions de déclenchement ni leurs préconditions. Une procédure rejouable a besoin des deux.

---

## Licence

MIT. Article compagnon : [casys.ai/fr/blog/graph-engineering-multi-agentic-systems](https://casys.ai/fr/blog/graph-engineering-multi-agentic-systems)
