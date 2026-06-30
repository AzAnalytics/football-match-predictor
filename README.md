# ⚽ Prédiction de résultats de matchs internationaux

Deux modèles complémentaires estiment l'issue **et le score final** d'un match
international de football — victoire à domicile / match nul / victoire à l'extérieur,
plus le **score exact le plus probable** (y compris les nuls : 1-1, 0-0…) — à partir de
près de **150 ans d'historique** (≈ 49 000 matchs, 1872 → 2026). Le projet est livré avec
une **application Streamlit** permettant d'interroger n'importe quelle affiche en quelques clics.

> Projet réalisé par **Alexis Zueras — AZ Analytics**.

<img width="1469" height="845" alt="Capture d’écran 2026-06-16 à 15 25 19" src="https://github.com/user-attachments/assets/f14c274b-51e1-4ab1-bd0c-d5e8c0dd4491" />

---

## Contexte

Prédire le résultat d'un match de football est un cas d'école de **classification sur
données temporelles** : le niveau des équipes évolue dans le temps, les rencontres ne se
valent pas (un amical n'est pas une finale de Coupe du monde), et le résultat reste
partiellement imprévisible par nature.

L'objectif n'était pas de « battre les bookmakers », mais de construire une chaîne complète,
honnête et reproductible : de la donnée brute jusqu'à un outil interactif, en évitant les
pièges classiques qui gonflent artificiellement les performances.

## Démarche

### 1. Données
Jeu de données public [`martj42/international_results`](https://github.com/martj42/international_results) :
date, équipes, score, tournoi, lieu, indicateur de terrain neutre. Nettoyage minimal
(retrait des matchs non joués) et création de la cible à trois classes.

### 2. Feature engineering
Le cœur du projet. Les variables prédictives sont reconstruites match par match, dans
l'ordre chronologique :

- **Classement Elo maison** : chaque équipe possède une note qui évolue après chaque match,
  pondérée par **l'importance de la rencontre** (amical → Coupe du monde) et **l'ampleur de
  la victoire**. C'est la variable la plus prédictive.
- **Forme récente** : moyenne de points sur les cinq derniers matchs.
- **Avantage du terrain**, neutralisé lorsque le match se joue sur terrain neutre.

### 3. Validation temporelle (point méthodologique clé)
Les modèles sont **entraînés sur le passé et évalués sur le futur** (coupure au 01/01/2022).
Un découpage aléatoire classique serait ici une **fuite de données** : il reviendrait à
entraîner le modèle sur des matchs postérieurs à ceux qu'on cherche à prédire, et gonflerait
le score sans valeur réelle. Toutes les variables sont par ailleurs calculées de façon
**causale** (uniquement à partir d'informations disponibles avant le coup d'envoi).

### 4. Modélisation
Comparaison d'une **régression logistique** (simple, interprétable) et d'un **gradient
boosting**. Les deux sont mesurés contre une baseline naïve (« toujours l'équipe à domicile »).

### 5. Deux modes selon l'objectif
Un modèle qui maximise la précision globale apprend vite à **toujours pencher vers l'équipe
à domicile** et n'annonce quasiment jamais de match nul : sa précision brille (~60 %) mais il
est aveugle à une classe sur trois. Pour y remédier, le projet expose deux **modes** :

- **`precision`** — le modèle d'origine. Maximise la précision globale (~60,5 %), mais le
  recall sur les nuls est de ~0,6 %.
- **`equilibre`** (défaut) — ajoute des features ciblant les matchs serrés (probabilité Elo
  intégrant le terrain, écart de niveau absolu, taux de nul récent, repos, maturité de l'Elo)
  et une **pondération de classe `balanced`**. La précision globale descend (~55,8 %) mais le
  modèle prédit **réellement** les nuls (recall ~37 %) et son **macro-F1** — la métrique qui
  récompense l'équilibre entre les trois classes — passe de **0,446 à 0,530**.

## Résultats

| Modèle | Précision | Recall nul | Macro-F1 |
| --- | --- | --- | --- |
| Baseline (toujours domicile) | 47,8 % | — | — |
| Régression logistique | 60,1 % | ~0 % | — |
| Gradient boosting — mode `precision` | **60,5 %** | 0,6 % | 0,446 |
| Gradient boosting — mode `equilibre` | 55,8 % | **36,6 %** | **0,530** |

Le modèle dépasse la baseline de **~13 points** en précision. L'essentiel du gain provient des
variables (Elo, forme), non du choix de l'algorithme — un enseignement en soi. Le mode
`equilibre` illustre un second enseignement : **la précision globale seule est trompeuse** sur
données déséquilibrées ; il faut regarder le détail par classe (macro-F1, recall).

## Prédiction du score exact (double Poisson)

En complément du classifieur, un second modèle prédit le **score final**. Plutôt que de
classer directement l'issue, on entraîne **deux régressions de Poisson** qui estiment
l'espérance de buts de chaque équipe (`prediction_score.py`). De ces deux moyennes on
construit la **matrice de tous les scores possibles** (avec une correction Dixon-Coles
sur les petits scores), d'où l'on tire :

- le **score exact le plus probable** (1-0, 2-1, et surtout les nuls 1-1 / 0-0 qui
  émergent naturellement quand les équipes sont proches) ;
- les probabilités d'issue 1X2 en sommant la matrice ;
- le score le plus probable **conditionnel à chaque issue**.

C'est une réponse plus élégante à l'angle mort des nuls : ici le nul n'est plus une classe
qu'il faut forcer, c'est une conséquence de la distribution des buts.

| Lecture du modèle de score | Précision 1X2 | Recall nul |
| --- | --- | --- |
| Issue via somme des probabilités | ~60 % | ~0 % |
| Issue via score exact le plus probable | ~54 % | **~49 %** |

Le **score exact** est correct dans **~14 %** des cas et la **différence de buts** dans
**~25 %** — des niveaux solides pour du football, stables en walk-forward (2020-2025).

## Évaluation probabiliste et calibration

La précision (accuracy) seule ne suffit pas à juger un modèle qui produit des
probabilités : elle ignore la confiance. Le module `footpredictor.evaluation` ajoute les
métriques de référence du pronostic sportif :

- **RPS** (Ranked Probability Score) — la métrique standard du 1X2, qui tient
  compte de l'ordre des issues (se tromper « de peu » coûte moins cher) ;
- **log-loss** et **Brier** — qui pénalisent une confiance mal placée ;
- une **table de calibration** : quand le modèle annonce 30 %, l'issue arrive-t-elle
  vraiment ~30 % du temps ?

Classement par RPS (test ≥ 2022, plus bas = meilleur) :

| Modèle | RPS | Précision |
| --- | --- | --- |
| Modèle de score (Poisson) | **0,1716** | 60,1 % |
| Classifieur `precision` | 0,1724 | 60,0 % |
| Classifieur `equilibre` **calibré** | 0,1725 | 60,1 % |
| Classifieur `equilibre` (brut) | 0,1790 | 55,4 % |
| Baseline (fréquences de base) | 0,2284 | 47,8 % |

Deux enseignements : le **modèle de score est le meilleur prédicteur probabiliste**
(pas seulement pour les nuls), et la **calibration sigmoïde (Platt) « répare » le mode
`equilibre`** — ses probabilités brutes étaient trop sûres d'elles (il annonçait 36 % de
nul là où la réalité était 27 %). Une fois calibré, son RPS rejoint celui du mode
précision tout en gardant sa lecture équilibrée. L'application affiche désormais des
**probabilités calibrées**.

## Modèle Dixon-Coles et ensemble

Pour aller plus loin, un second modèle de buts a été ajouté (`footpredictor.dixon_coles`) :
le modèle **Dixon-Coles**, où chaque équipe a une **force d'attaque et de défense**
estimées par maximum de vraisemblance (régression de Poisson pondérée par récence,
demi-vie ≈ 4 ans), plus la correction `rho` ajustée. Ses classements sont crédibles
(meilleures attaques : Espagne, Brésil, Allemagne ; meilleures défenses : Argentine,
Angleterre).

Pris seul, Dixon-Coles fait **jeu égal** avec le modèle Elo-Poisson — un modèle plus
sophistiqué ne bat pas forcément un bon Elo. Mais comme les deux apportent des points de
vue **différents**, leur **ensemble** (moyenne des matrices de scores) bat chacun pris
isolément, de façon stable :

| Modèle (test ≥ 2022) | RPS | log-loss | Brier |
| --- | --- | --- | --- |
| Modèle Elo-Poisson | 0,1716 | 0,8757 | 0,5148 |
| Modèle Dixon-Coles | 0,1750 | 0,8868 | 0,5208 |
| **Ensemble des deux** | **0,1700** | **0,8690** | **0,5104** |

C'est le **meilleur prédicteur du projet**, et c'est lui que l'application utilise désormais
pour le score. Enseignement : l'**ensemble de modèles diversifiés** est un levier plus
fiable que la course au modèle unique le plus complexe.

## Limites et pistes d'amélioration

Avec ces seules données historiques, la prédiction en trois classes **plafonne autour de
60 %** en précision globale, ce qui est cohérent avec la littérature. Le match **nul** reste
la classe la plus difficile : le mode `equilibre` le rend prédictible (recall ~37 %) mais sa
précision propre plafonne (~29 %). Les leviers pour aller plus loin sont externes au jeu de
données :

- intégration des **cotes des bookmakers** (qui agrègent l'information publique) ;
- données de **compositions / blessures** ;
- signaux d'**enjeu** du match (qualification en jeu, rivalités).

## Ce que le projet illustre

- Feature engineering à partir de connaissances métier (système Elo sur mesure).
- Évaluation rigoureuse et sans fuite de données sur série temporelle.
- Comparaison et arbitrage entre modèles.
- **Mise en production** d'un modèle sous forme d'application interactive.

## Stack technique

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `Plotly`

## Données auto-rafraîchies

Le projet est packagé comme une **librairie Python installable** (`footpredictor`). Les
données ne sont plus saisies à la main : elles sont **téléchargées et mises à jour
automatiquement** depuis le dépôt public [`martj42/international_results`](https://github.com/martj42/international_results)
(licence CC0), mises en cache localement, et re-téléchargées seulement quand le cache est
périmé. Hors-ligne, la librairie bascule sur le cache puis sur un instantané embarqué — elle
fonctionne donc toujours.

## Installation et utilisation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows : .\venv\Scripts\Activate.ps1
pip install -e ".[app]"           # installe la librairie + dépendances de l'app
```

### Comme librairie

```python
from footpredictor import Predictor

p = Predictor()                                   # télécharge les données + entraîne
res = p.predict("France", "Senegal", neutral=True)
print(res["score_exact"])                          # ex. (1, 0)
print(res["proba_issue"])                          # probas 1X2 calibrées
p.update()                                          # forcer le rafraîchissement des données
p.chercher_equipe("ivory")                          # retrouver le nom exact d'une équipe
```

### En ligne de commande

```bash
footpredictor France Senegal --neutral     # prédiction issue + score
footpredictor --top-elo 10                 # classement Elo actuel
footpredictor --search ivory               # chercher une équipe
footpredictor France Brazil --update       # forcer le rafraîchissement des données
```

### Application

```bash
streamlit run app/main.py                  # issue calibrée + score ensemble + heatmap
```

### Benchmarks

```bash
python benchmarks/backtest_walkforward.py  # validation glissante 2014-2025
python benchmarks/compare_models.py        # RPS / log-loss / Brier de tous les modèles
```

## Structure du projet

```
.
├── src/footpredictor/            # la librairie
│   ├── data.py                   # téléchargement auto + cache + fallback hors-ligne
│   ├── features.py               # feature engineering causal (Elo, forme, nuls)
│   ├── classifier.py             # classifieur 1X2 (modes + calibration)
│   ├── score.py                  # modèle de score Elo-Poisson + ensemble
│   ├── dixon_coles.py            # modèle force attaque/défense
│   ├── evaluation.py             # métriques RPS / log-loss / Brier
│   ├── predictor.py              # API haut niveau : classe Predictor
│   ├── cli.py                    # interface ligne de commande
│   └── data/results.csv          # instantané embarqué (secours hors-ligne)
├── app/main.py                   # application Streamlit
├── benchmarks/                   # backtests et comparaison de modèles
├── tests/                        # tests de fumée (pytest)
├── pyproject.toml                # packaging installable
├── BACKTEST.md
└── README.md
```

---

*Les prédictions reposent sur l'historique des résultats (force et forme des équipes) et ne
constituent en aucun cas un conseil de pari.*
