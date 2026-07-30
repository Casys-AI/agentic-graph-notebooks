# agentic-graph-notebooks

**Transforme tes propres traces d'exécution d'agents en graphe — puis vérifie si ce graphe signifie quelque chose.**

Ce dépôt ne contient aucune donnée. Tout s'exécute sur *tes* traces, sur *ta* machine.

[English](README.md) · [简体中文](README.zh.md) · [繁體中文](README.zh-TW.md) · [Français](README.fr.md)

---

## Pourquoi

En juillet 2026, le débat « boucles vs graphes » a produit beaucoup d'affirmations et très peu de mesures. Ces notebooks font le contraire : un protocole reproductible qui répond à trois questions sur tes propres agents.

**1. Le graphe a-t-il la bonne forme ?**

Un graphe n'est utile que si connaître les voisins d'un nœud te dit quelque chose de *spécifique à ce nœud*. Il y a deux façons opposées d'échouer. Trop épars : la plupart des nœuds n'ont pas de voisins, il n'y a rien à observer. Trop dense : chaque nœud touche presque tous les autres, donc le voisinage de A est identique à celui de B et ne discrimine rien. L'information ne vit qu'entre ces deux extrêmes.

**2. Le graphe signifie-t-il quelque chose ?**

Un motif statistique sans nom humain ne prouve rien. Ces notebooks remontent toujours des motifs vers les vraies commandes, pour que tu puisses juger toi-même — en distinguant les motifs exploitables, les motifs réels mais triviaux (`cd` puis `ls` est un motif réel et inutile) et les purs artefacts.

**3. Le graphe se répète-t-il ?**

C'est la question qui décide si « mémoire procédurale » signifie quelque chose. C'est aussi là où presque tout le monde se trompe, car y répondre honnêtement exige une ligne de base aléatoire.

---

## La ligne de base aléatoire, et pourquoi elle change tout

Deux fenêtres temporelles du même agent partagent nécessairement du vocabulaire : même machine, mêmes outils, même personne. Un score de persistance brut est donc **ininterprétable** — et il sera élevé, donc convaincant.

La seule question qui compte : *au-delà du hasard ?*

Ces notebooks redistribuent aléatoirement les mêmes actions dans les mêmes fenêtres, vingt fois, et comparent. Sur le corpus qui a motivé ce travail, une mesure de persistance de voisinage a atteint une médiane de **1,000** — résultat spectaculaire, jusqu'à ce que la ligne de base aléatoire retourne aussi **1,000**. Le répertoire d'actions était simplement trop étroit pour varier.

Sans ce contrôle, ce chiffre aurait été publié comme une découverte.

---

## Installation

```bash
git clone https://github.com/Casys-AI/agentic-graph-notebooks
cd agentic-graph-notebooks
pip install -r requirements.txt
jupyter lab
```

Pas de dépendances lourdes. La détection de communautés et la corrélation de Spearman sont implémentées en Python pur ; `networkx` et `matplotlib` servent uniquement à l'affichage.

---

## Utilisation

Les notebooks attendent des sessions JSONL — un événement par ligne, avec des blocs `toolCall`. Le format OpenClaw fonctionne tel quel :

```python
from agentgraph import build_hyperedges, metrics

edges = build_hyperedges("~/.openclaw/agents/*/sessions/*.jsonl")
print(metrics(edges))
```

```
# (les valeurs réelles dépendent de ton corpus)
N=<nœuds> · edges=<arêtes> · density=<densité> · mean degree=<degré moyen> · entropy=<entropie>
→ INFORMATIVE REGIME — neighbourhoods carry information
```

Pour adapter un autre format d'agent, modifie `iter_tool_calls()` dans `agentgraph/extract.py`. C'est le seul point d'entrée à changer.

| Notebook | Question |
|---|---|
| `01-extract` | Qu'y a-t-il dans tes traces ? Distribution des outils, part du shell |
| `02-granularity` | À quelle granularité ton graphe cesse-t-il d'être dégénéré ? |
| `03-motifs` | Tes motifs ont-ils des noms ? |
| `04-stability` | Tes routines se répètent-elles au-delà du hasard ? |
| `05-loops` | Combien de commandes contiennent des boucles explicites — et combien sont invisibles au graphe ? |
| `06-condensation` | Condensation SCC : à quel point les boucles de rétroaction sont-elles locales ? |
| `07-repair` | Grammaire RE-PAIR : sous-séquences récurrentes et profondeur de composition |
| `08-read-budget` | Budget de lecture : l'action précédente prédit-elle `head -n N` ? |

---

## Un détail de parsing plus important qu'il n'y paraît

Les agents qui écrivent du shell ne produisent pas des *actions*, ils produisent des *programmes*. En gros, quatre commandes composées sur cinq contiennent `&&`, `;`, `|` ou un retour à la ligne. Ces opérateurs sont les frontières que le modèle lui-même a écrites — on segmente à ces endroits.

Segmenter naïvement avec une regex est un piège : elle coupe aussi à l'intérieur des chaînes entre guillemets. `grep -E 'a|b|c'` devient trois « commandes », et chaque fragment produit un atome fantôme (`awk.print1`, `awk.print2`, …) sans aucune commande réelle derrière. Sur des corpus réels, ce seul bug fabrique une fraction substantielle du vocabulaire à partir de rien.

`split_segments()` utilise `shlex` avec `punctuation_chars`, ce qui respecte les guillemets. Si tu adaptes ce code, conserve cette propriété.

L'inline Python bénéficie d'un vrai AST plutôt que d'expressions régulières — un avantage décisif par rapport au shell, qui ne se prête qu'à des heuristiques.

---

## Deux avertissements

**Tes traces contiennent des secrets.** Pas accidentellement — systématiquement : Bearer tokens, clés API, variables d'environnement en clair. `mask_secrets()` filtre les formes courantes et s'applique à toutes les sorties de motifs, mais réduit le risque sans l'éliminer. **Relis toujours les sorties avant de les partager.**

**Deux pièges méthodologiques** que les notebooks signalent mais ne peuvent pas corriger pour toi :

- *Autocorrélation de session* — les commandes d'une même session se ressemblent nécessairement. Si une seule session domine une fenêtre, la « stabilité » mesurée n'est que la cohérence interne de cette session. `describe_windows()` avertit au-delà de 40 %.
- *Mélange d'agents* — si tes fenêtres temporelles correspondent à des agents différents, tu mesures une différence d'identité, pas une dérive dans le temps. Filtre sur un seul agent avant de lancer l'analyse de stabilité.

---

## Ce que ce protocole ne fait pas

Il mesure la **forme** du graphe et la **répétition** de ses motifs. Il ne valide aucune tâche aval : un graphe bien formé n'est pas la preuve qu'il améliore une prédiction, une décision de routage ou une recommandation. Condition nécessaire, pas suffisante.

Il décrit *ce que* l'agent a fait, jamais *quand* le déclencher ni quelle précondition vérifier. C'est la frontière entre un graphe et une procédure rejouable.

---

## Licence

MIT. Article compagnon : [casys.ai/fr/blog/graph-engineering-multi-agentic-systems](https://casys.ai/fr/blog/graph-engineering-multi-agentic-systems)
