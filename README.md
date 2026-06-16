# ⚽ Prédiction de résultats de matchs internationaux

Modèle de **classification** qui estime l'issue d'un match international de football
— victoire à domicile / match nul / victoire à l'extérieur — à partir de près de
**150 ans d'historique** (≈ 49 000 matchs, 1872 → 2026). Le projet est livré avec une
**application Streamlit** permettant d'interroger n'importe quelle affiche en quelques clics.

> Projet réalisé par **Alexis Zueras — AZ Analytics**.

<!-- Ajoute ici une capture d'écran ou un GIF de l'app : ![Aperçu](docs/demo.png) -->

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

## Résultats

| Modèle | Précision (jeu de test) |
| --- | --- |
| Baseline (toujours domicile) | 47,8 % |
| Régression logistique | 60,1 % |
| Gradient boosting | **60,5 %** |

Le modèle dépasse la baseline de **~13 points**. L'essentiel du gain provient des variables
(Elo, forme), non du choix de l'algorithme — un enseignement en soi.

## Limites et pistes d'amélioration

Avec ces seules données historiques, la prédiction en trois classes **plafonne autour de
60 %**, ce qui est cohérent avec la littérature. Les leviers pour aller plus loin sont
externes au jeu de données :

- intégration des **cotes des bookmakers** (qui agrègent l'information publique) ;
- données de **compositions / blessures** ;
- passage à une **validation glissante** pour mesurer la stabilité dans le temps.

Le match **nul** reste la classe la plus difficile à prédire — un comportement attendu.

## Ce que le projet illustre

- Feature engineering à partir de connaissances métier (système Elo sur mesure).
- Évaluation rigoureuse et sans fuite de données sur série temporelle.
- Comparaison et arbitrage entre modèles.
- **Mise en production** d'un modèle sous forme d'application interactive.

## Stack technique

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit`

## Installation et utilisation

```bash
# 1. Environnement
python3 -m venv venv
source venv/bin/activate        # Windows : .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Données : placer results.csv dans le dossier attendu (cf. CHEMIN_CSV)

# 3a. Lancer l'analyse en ligne de commande
python prediction_foot.py

# 3b. Ou lancer l'application
streamlit run app.py
```

## Structure du projet

```
.
├── prediction_foot.py   # pipeline : chargement, features, entraînement, prédiction
├── app.py               # interface Streamlit (réutilise prediction_foot.py)
├── requirements.txt
├── archive/
│   └── results.csv      # jeu de données
└── README.md
```

---

*Les prédictions reposent sur l'historique des résultats (force et forme des équipes) et ne
constituent en aucun cas un conseil de pari.*
