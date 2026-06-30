"""
Prédiction de résultats de matchs internationaux de football.
Données : martj42/international_results (results.csv).

Pipeline :
  1. Chargement + nettoyage + création de la cible
  2. Calcul des features en UNE passe chronologique (Elo pondéré + forme + signaux de nul)
  3. Découpage temporel train/test (on prédit le futur, pas de mélange aléatoire)
  4. Entraînement selon le MODE choisi + évaluation
  5. Prédiction des probabilités pour n'importe quel match

Deux MODES (cf. config_mode) :
  - "precision"  : maximise la précision globale (~60 %), mais ne prédit
                   quasiment jamais les nuls. C'est le modèle d'origine.
  - "equilibre"  : ajoute des features ciblant les matchs serrés + une
                   pondération de classe "balanced". La précision globale
                   baisse (~55 %) mais le modèle prédit RÉELLEMENT les nuls
                   (recall ~37 % vs 0,7 %) et son macro-F1 est nettement
                   meilleur (~0,52 vs ~0,44). À privilégier si on veut un
                   modèle utile sur les 3 classes, pas seulement "domicile".
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             recall_score, f1_score)


# ---------------------------------------------------------------------------
# Paramètres (tout ce qui se règle est regroupé ici)
# ---------------------------------------------------------------------------
CHEMIN_CSV = "archive/results.csv"   # ou l'URL brute GitHub si tu préfères
DATE_CUTOFF = "2022-01-01"           # avant = entraînement, à partir de = test
BASE_ELO = 1500                      # note de départ d'une équipe
HOME_ADV = 65                        # bonus Elo à domicile (annulé si terrain neutre)
FENETRE_FORME = 5                    # nb de matchs récents pris en compte pour la forme
FENETRE_NUL = 10                     # nb de matchs récents pour le taux de nul d'une équipe

MODE = "equilibre"                   # "precision" ou "equilibre" (défaut conseillé)

# Features communes (modèle d'origine) et features supplémentaires (mode equilibre)
FEATURES_PRECISION = ["elo_diff", "form_diff", "home_elo", "away_elo", "neutral"]
FEATURES_EQUILIBRE = FEATURES_PRECISION + [
    "exp_home",       # probabilité Elo de gagner à domicile (intègre l'avantage terrain)
    "abs_elo_diff",   # écart de niveau absolu : petit = match serré = nul plus probable
    "draw_rate_sum",  # propension récente des deux équipes à faire des nuls
    "rest_diff",      # différence de jours de repos avant le match
    "min_played",     # nb de matchs joués par la moins expérimentée (maturité de l'Elo)
]
# Alias rétro-compatible : pointe sur les features du MODE courant.
FEATURES = FEATURES_EQUILIBRE if MODE == "equilibre" else FEATURES_PRECISION


def config_mode(mode=MODE):
    """Renvoie (liste de features, modèle non entraîné) correspondant au mode."""
    if mode == "precision":
        return FEATURES_PRECISION, HistGradientBoostingClassifier(random_state=0)
    if mode == "equilibre":
        return (FEATURES_EQUILIBRE,
                HistGradientBoostingClassifier(random_state=0, class_weight="balanced"))
    raise ValueError(f"Mode inconnu : {mode!r} (attendu 'precision' ou 'equilibre')")


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
# 3. Construction des features AVANT un match (source unique de vérité)
# ---------------------------------------------------------------------------
def _features_avant_match(home, away, neutral, date, etat):
    """
    Calcule toutes les features d'un match à partir de l'état des équipes AVANT
    le coup d'envoi. Utilisé à la fois pour l'entraînement (passe chronologique)
    et pour la prédiction d'un nouveau match -> aucune divergence de logique.

    `date=None` (prédiction d'un match hypothétique sans date) neutralise le repos.
    """
    elos, hist = etat["elos"], etat["hist"]
    draw, last, npl = etat["draw"], etat["last_date"], etat["n_played"]

    r_home = elos.get(home, BASE_ELO)
    r_away = elos.get(away, BASE_ELO)
    f_home = np.mean(hist[home]) if len(hist[home]) else 1.0
    f_away = np.mean(hist[away]) if len(hist[away]) else 1.0

    adv = 0 if neutral else HOME_ADV
    exp_home = 1 / (1 + 10 ** ((r_away - (r_home + adv)) / 400))

    dr_home = np.mean(draw[home]) if len(draw[home]) else 0.25
    dr_away = np.mean(draw[away]) if len(draw[away]) else 0.25

    if date is not None:
        rest_h = (date - last[home]).days if home in last else 180
        rest_a = (date - last[away]).days if away in last else 180
        rest_diff = np.clip(rest_h, 0, 365) - np.clip(rest_a, 0, 365)
    else:
        rest_diff = 0

    return {
        "elo_diff": r_home - r_away,
        "form_diff": f_home - f_away,
        "home_elo": r_home,
        "away_elo": r_away,
        "neutral": bool(neutral),
        "exp_home": exp_home,
        "abs_elo_diff": abs(r_home - r_away),
        "draw_rate_sum": dr_home + dr_away,
        "rest_diff": rest_diff,
        "min_played": min(npl[home], npl[away]),
    }


def calculer_features(df):
    """
    Parcourt les matchs dans l'ordre du temps. Pour chaque match, on LIT l'état
    des deux équipes AVANT le coup d'envoi -> ce sont les features, donc aucune
    information du futur ne fuite. On MET À JOUR ensuite.

    Renvoie : (df enrichi, elos, hist, etat). `elos` et `hist` restent exposés
    pour la rétro-compatibilité ; `etat` regroupe tout ce dont la prédiction a
    besoin (y compris taux de nul, dernière date jouée, nb de matchs joués).
    """
    etat = {
        "elos": {},
        "hist": defaultdict(lambda: deque(maxlen=FENETRE_FORME)),   # points récents
        "draw": defaultdict(lambda: deque(maxlen=FENETRE_NUL)),     # 1 si nul, 0 sinon
        "last_date": {},                                            # dernière date jouée
        "n_played": defaultdict(int),                               # nb de matchs joués
    }
    elos = etat["elos"]
    hist = etat["hist"]
    draw = etat["draw"]

    lignes = []
    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team

        # --- (a) LECTURE de l'état d'avant-match = les features ---
        feats = _features_avant_match(home, away, row.neutral, row.date, etat)
        lignes.append(feats)
        r_home, r_away = feats["home_elo"], feats["away_elo"]
        exp_home = feats["exp_home"]

        # --- (b) Résultat réel du match ---
        if row.home_score > row.away_score:
            s_home, pts_home, pts_away, est_nul = 1.0, 3, 0, 0
        elif row.home_score == row.away_score:
            s_home, pts_home, pts_away, est_nul = 0.5, 1, 1, 1
        else:
            s_home, pts_home, pts_away, est_nul = 0.0, 0, 3, 0

        # --- (c) MISE À JOUR de l'Elo (pondérée importance × écart de buts) ---
        k = poids_importance(row.tournament) * mult_buts(row.home_score - row.away_score)
        elos[home] = r_home + k * (s_home - exp_home)
        elos[away] = r_away + k * ((1 - s_home) - (1 - exp_home))

        # --- (d) MISE À JOUR de la forme, des nuls, du repos, du compteur ---
        hist[home].append(pts_home); hist[away].append(pts_away)
        draw[home].append(est_nul); draw[away].append(est_nul)
        etat["last_date"][home] = row.date; etat["last_date"][away] = row.date
        etat["n_played"][home] += 1; etat["n_played"][away] += 1

    feat_df = pd.DataFrame(lignes, index=df.index)
    # "neutral" existe déjà dans df : on évite de le dupliquer au concat.
    feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c in df.columns])
    df = pd.concat([df, feat_df], axis=1)
    return df, elos, hist, etat


# ---------------------------------------------------------------------------
# 4. Entraînement et évaluation
# ---------------------------------------------------------------------------
def entrainer(df, mode=MODE, rapport=True):
    """Entraîne le modèle du mode choisi sur le passé, évalue sur le futur."""
    feats, modele = config_mode(mode)
    train = df[df["date"] < DATE_CUTOFF]
    test = df[df["date"] >= DATE_CUTOFF]

    X_train, y_train = train[feats], train["resultat"]
    X_test, y_test = test[feats], test["resultat"]

    baseline = (y_test == "domicile").mean()
    pred = modele.fit(X_train, y_train).predict(X_test)

    acc = accuracy_score(y_test, pred)
    rec_nul = recall_score(y_test, pred, labels=["nul"], average="macro", zero_division=0)
    f1m = f1_score(y_test, pred, average="macro", zero_division=0)

    print(f"Mode                          : {mode}")
    print(f"Baseline (toujours domicile)  : {baseline:.4f}")
    print(f"Précision globale             : {acc:.4f}")
    print(f"Recall sur les nuls           : {rec_nul:.4f}")
    print(f"Macro-F1 (équilibre 3 classes): {f1m:.4f}")

    if rapport:
        print("\nDétail par classe :")
        print(classification_report(y_test, pred, digits=3))

    return modele


# ---------------------------------------------------------------------------
# 5. Prédiction d'un match au choix
# ---------------------------------------------------------------------------
def chercher_equipe(motif, elos):
    """Retrouve l'orthographe exacte d'une équipe : chercher_equipe('cote', elos)."""
    return [t for t in sorted(elos) if motif.lower() in t.lower()]


def proba_match(modele, etat, equipe_dom, equipe_ext, terrain_neutre=False, mode=MODE):
    """Renvoie un dict {classe: proba} pour le match choisi (utilisé par l'app)."""
    feats = config_mode(mode)[0]
    ligne = _features_avant_match(equipe_dom, equipe_ext, terrain_neutre, None, etat)
    X = pd.DataFrame([ligne])[feats]
    proba = modele.predict_proba(X)[0]
    return dict(zip(modele.classes_, proba))


def predire(modele, etat, equipe_dom, equipe_ext, terrain_neutre=False, mode=MODE):
    """Affiche les probabilités domicile / nul / extérieur pour le match choisi."""
    for t in (equipe_dom, equipe_ext):
        if t not in etat["elos"]:
            print(f"⚠️  '{t}' introuvable. Essaie chercher_equipe('...', elos).")
            return

    p = proba_match(modele, etat, equipe_dom, equipe_ext, terrain_neutre, mode)
    print(f"\n{equipe_dom} vs {equipe_ext}  (terrain neutre = {terrain_neutre})")
    for classe, proba in sorted(p.items(), key=lambda x: -x[1]):
        print(f"  {classe:<10}: {proba:6.1%}")


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = charger_donnees()
    print(f"{df.shape[0]} matchs chargés "
          f"({df['date'].min().date()} -> {df['date'].max().date()})\n")

    df, elos, hist, etat = calculer_features(df)

    print("Top 5 Elo actuel :")
    for team, rating in sorted(elos.items(), key=lambda x: -x[1])[:5]:
        print(f"  {team:<15} {rating:.0f}")
    print()

    modele = entrainer(df, mode=MODE)

    # Change librement les équipes ici (noms anglais, cf. chercher_equipe)
    predire(modele, etat, "France", "Senegal", terrain_neutre=True, mode=MODE)
