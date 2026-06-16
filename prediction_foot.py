"""
Prédiction de résultats de matchs internationaux de football.
Données : martj42/international_results (results.csv).

Pipeline :
  1. Chargement + nettoyage + création de la cible
  2. Calcul des features en UNE passe chronologique (Elo pondéré + forme récente)
  3. Découpage temporel train/test (on prédit le futur, pas de mélange aléatoire)
  4. Entraînement de deux modèles + évaluation
  5. Prédiction des probabilités pour n'importe quel match
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# Paramètres (tout ce qui se règle est regroupé ici)
# ---------------------------------------------------------------------------
CHEMIN_CSV = "archive/results.csv"   # ou l'URL brute GitHub si tu préfères
DATE_CUTOFF = "2022-01-01"           # avant = entraînement, à partir de = test
BASE_ELO = 1500                      # note de départ d'une équipe
HOME_ADV = 65                        # bonus Elo à domicile (annulé si terrain neutre)
FENETRE_FORME = 5                    # nb de matchs récents pris en compte pour la forme
FEATURES = ["elo_diff", "form_diff", "home_elo", "away_elo", "neutral"]


# ---------------------------------------------------------------------------
# 1. Chargement et nettoyage
# ---------------------------------------------------------------------------
def charger_donnees(chemin=CHEMIN_CSV):
    """Charge le CSV, retire les matchs non joués, crée la cible et trie par date."""
    df = pd.read_csv(chemin, parse_dates=["date"])

    # On écarte les matchs sans score (matchs futurs déjà programmés)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Cible (version vectorisée : bien plus rapide que df.apply ligne à ligne)
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["resultat"] = np.select(conditions, ["domicile", "exterieur"], default="nul")

    # Tri chronologique indispensable pour le calcul causal des features
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Pondérations de l'Elo
# ---------------------------------------------------------------------------
def poids_importance(tournoi):
    """Toutes les rencontres ne se valent pas : une Coupe du monde pèse plus qu'un amical."""
    t = tournoi.lower()
    if t == "friendly":
        return 20
    if "fifa world cup" in t and "qualif" not in t:        # phase finale Coupe du monde
        return 60
    grandes = ["uefa euro", "copa am", "african cup", "asian cup", "gold cup", "confederations"]
    if any(k in t for k in grandes) and "qualif" not in t:  # grandes compétitions continentales
        return 50
    if "qualif" in t:                                       # matchs de qualification
        return 40
    return 30                                               # le reste


def mult_buts(ecart):
    """Une large victoire doit faire bouger les notes davantage (formule World Football Elo)."""
    ecart = abs(ecart)
    if ecart <= 1:
        return 1.0
    if ecart == 2:
        return 1.5
    return (11 + ecart) / 8


# ---------------------------------------------------------------------------
# 3. Calcul des features en UNE seule passe chronologique
# ---------------------------------------------------------------------------
def calculer_features(df):
    """
    Parcourt les matchs dans l'ordre du temps. Pour chaque match, on LIT l'état
    des deux équipes (Elo + forme) AVANT le coup d'envoi -> ce sont les features,
    donc aucune information du futur ne fuite. On MET À JOUR ensuite.

    Renvoie le df enrichi, plus l'état final `elos` et `hist` qui serviront
    à prédire de nouveaux matchs.
    """
    elos = {}                                              # note courante de chaque équipe
    hist = defaultdict(lambda: deque(maxlen=FENETRE_FORME))  # points des N derniers matchs

    home_elos, away_elos = [], []
    home_form, away_form = [], []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team

        # --- (a) LECTURE de l'état d'avant-match = les features ---
        r_home = elos.get(home, BASE_ELO)
        r_away = elos.get(away, BASE_ELO)
        home_elos.append(r_home)
        away_elos.append(r_away)

        f_home = np.mean(hist[home]) if len(hist[home]) else 1.0  # 1.0 = forme neutre par défaut
        f_away = np.mean(hist[away]) if len(hist[away]) else 1.0
        home_form.append(f_home)
        away_form.append(f_away)

        # --- (b) Résultat réel du match ---
        if row.home_score > row.away_score:
            s_home, pts_home, pts_away = 1.0, 3, 0
        elif row.home_score == row.away_score:
            s_home, pts_home, pts_away = 0.5, 1, 1
        else:
            s_home, pts_home, pts_away = 0.0, 0, 3

        # --- (c) MISE À JOUR de l'Elo (pondérée importance × écart de buts) ---
        adv = 0 if row.neutral else HOME_ADV
        exp_home = 1 / (1 + 10 ** ((r_away - (r_home + adv)) / 400))
        k = poids_importance(row.tournament) * mult_buts(row.home_score - row.away_score)
        elos[home] = r_home + k * (s_home - exp_home)
        elos[away] = r_away + k * ((1 - s_home) - (1 - exp_home))

        # --- (d) MISE À JOUR de la forme ---
        hist[home].append(pts_home)
        hist[away].append(pts_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_form"] = home_form
    df["away_form"] = away_form
    df["form_diff"] = df["home_form"] - df["away_form"]

    return df, elos, hist


# ---------------------------------------------------------------------------
# 4. Entraînement et évaluation
# ---------------------------------------------------------------------------
def entrainer(df, rapport=True):
    """Entraîne deux modèles sur le passé, évalue sur le futur, renvoie le meilleur."""
    train = df[df["date"] < DATE_CUTOFF]
    test = df[df["date"] >= DATE_CUTOFF]

    X_train, y_train = train[FEATURES], train["resultat"]
    X_test, y_test = test[FEATURES], test["resultat"]

    baseline = (y_test == "domicile").mean()
    print(f"Baseline (toujours domicile) : {baseline:.4f}")

    # Modèle simple et interprétable
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(X_train, y_train)
    print(f"Régression logistique        : {accuracy_score(y_test, logit.predict(X_test)):.4f}")

    # Modèle non-linéaire (souvent un poil meilleur sur tableau de données)
    gb = HistGradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    print(f"Gradient boosting            : {accuracy_score(y_test, pred):.4f}")

    if rapport:
        print("\nDétail par classe (gradient boosting) :")
        print(classification_report(y_test, pred, digits=3))

    return gb


# ---------------------------------------------------------------------------
# 5. Prédiction d'un match au choix
# ---------------------------------------------------------------------------
def chercher_equipe(motif, elos):
    """Retrouve l'orthographe exacte d'une équipe : chercher_equipe('cote', elos)."""
    return [t for t in sorted(elos) if motif.lower() in t.lower()]


def predire(modele, elos, hist, equipe_dom, equipe_ext, terrain_neutre=False):
    """Affiche les probabilités domicile / nul / extérieur pour le match choisi."""
    for t in (equipe_dom, equipe_ext):
        if t not in elos:
            print(f"⚠️  '{t}' introuvable. Essaie chercher_equipe('...', elos).")
            return

    r_dom, r_ext = elos[equipe_dom], elos[equipe_ext]
    f_dom = np.mean(hist[equipe_dom]) if len(hist[equipe_dom]) else 1.0
    f_ext = np.mean(hist[equipe_ext]) if len(hist[equipe_ext]) else 1.0

    # On reconstruit EXACTEMENT la même ligne de features qu'à l'entraînement
    X = pd.DataFrame([{
        "elo_diff": r_dom - r_ext,
        "form_diff": f_dom - f_ext,
        "home_elo": r_dom,
        "away_elo": r_ext,
        "neutral": terrain_neutre,
    }])[FEATURES]

    proba = modele.predict_proba(X)[0]
    print(f"\n{equipe_dom} vs {equipe_ext}  (terrain neutre = {terrain_neutre})")
    for classe, p in sorted(zip(modele.classes_, proba), key=lambda x: -x[1]):
        print(f"  {classe:<10}: {p:6.1%}")


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = charger_donnees()
    print(f"{df.shape[0]} matchs chargés "
          f"({df['date'].min().date()} -> {df['date'].max().date()})\n")

    df, elos, hist = calculer_features(df)

    print("Top 5 Elo actuel :")
    for team, rating in sorted(elos.items(), key=lambda x: -x[1])[:5]:
        print(f"  {team:<15} {rating:.0f}")
    print()

    modele = entrainer(df)

    # Change librement les équipes ici (noms anglais, cf. chercher_equipe)
    predire(modele, elos, hist, "France", "Senegal", terrain_neutre=True)
