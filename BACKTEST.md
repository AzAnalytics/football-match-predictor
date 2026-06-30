# 🔁 Backtest en validation glissante (walk-forward)

> Réalisé le 2026-06-22. Script : `backtest_walkforward.py`. Résultats : `backtest_resultats.csv`.

## Pourquoi
Le pipeline initial valide le modèle sur **un seul découpage** (avant/après 2022). Bon, mais ça ne dit pas si la performance est **stable dans le temps**. Le walk-forward teste le modèle sur plusieurs années successives pour le vérifier.

## Méthode
Pour chaque saison Y de 2014 à 2025 :
1. entraîner sur tous les matchs joués **avant** le 1er janvier de l'année Y (fenêtre expanding) ;
2. prédire les matchs de l'année Y (jamais vus) ;
3. comparer à la baseline « toujours l'équipe à domicile ».

Les features (Elo, forme) étant calculées de façon **causale** (uniquement à partir du passé de chaque match), le découpage par date ne crée aucune fuite de données.

## Résultats (12 saisons)
- **Précision moyenne du modèle : 59,2 %** vs **baseline 47,4 %** → **+11,8 points**.
- Le modèle **bat la baseline chaque année** (de +9 à +16,4 points).
- **Écart-type : 3,3 points** → performance stable, pas un artefact d'un seul découpage.
- **Recall sur les nuls : ~0,7 %** → le modèle ne prédit quasiment jamais les matchs nuls. C'est son angle mort principal, confirmé sur toutes les saisons.

## Conclusion
Les ~60 % annoncés sont **réels et robustes dans le temps**. La marge de progression connue est la classe « nul ».

## 🆕 Suite (2026-06-30) : attaquer l'angle mort des nuls

Deux ajouts pour que le modèle prédise **réellement** les nuls et le **score**.

### 1. Mode `equilibre` du classifieur
Ajout de features ciblant les matchs serrés (proba Elo avec terrain, écart de niveau absolu, taux de nul récent, repos, maturité de l'Elo) + pondération de classe `balanced`. Walk-forward 2014-2025 :

| | Mode `precision` (origine) | Mode `equilibre` |
| --- | --- | --- |
| Précision moyenne | **59,2 %** | 54,7 % |
| Recall nul | 0,7 % | **37,1 %** |
| Macro-F1 | 0,437 | **0,520** |

Le mode `equilibre` bat la baseline chaque année et débloque les nuls, au prix de ~4,5 pts de précision « titre ». Leçon : **sur données déséquilibrées, la précision globale seule est trompeuse** — regarder le macro-F1 et le recall par classe.

### 2. Modèle de score (double Poisson)
Deux régressions de Poisson estiment les buts attendus de chaque équipe → matrice de scores (correction Dixon-Coles) → score exact + probas 1X2. Walk-forward 2020-2025 (issue déduite du score exact) :
- précision 1X2 ~54 %, **recall nul ~49 %** (encore mieux que le mode equilibre) ;
- **score exact correct ~14 %**, bonne différence de buts ~25 % ;
- les nuls (1-1, 0-0) émergent naturellement, sans pondération artificielle.

Script de référence : `prediction_score.py`.

## 🔗 Liens
- Projet : [[01 - Projets/Décrocher une mission ou un CDI pour la rentrée]]
- Domaine : [[02 - Domaines/Apprentissage/🎓 Apprentissage]]
