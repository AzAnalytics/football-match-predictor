"""
Feature engineering causal : Elo maison pondéré + forme + signaux de match serré.

Tout est calculé en UNE passe chronologique. Pour chaque match on LIT l'état des
deux équipes AVANT le coup d'envoi (ce sont les features) puis on MET À JOUR
l'état avec le résultat — aucune information du futur ne fuit.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

BASE_ELO = 1500       # note de départ d'une équipe
HOME_ADV = 65         # bonus Elo à domicile (annulé si terrain neutre)
FENETRE_FORME = 5     # nb de matchs récents pour la forme
FENETRE_NUL = 10      # nb de matchs récents pour le taux de nul

# Jeux de features
FEATURES_PRECISION = ["elo_diff", "form_diff", "home_elo", "away_elo", "neutral"]
FEATURES_EQUILIBRE = FEATURES_PRECISION + [
    "exp_home", "abs_elo_diff", "draw_rate_sum", "rest_diff", "min_played",
]
FEATURES_SCORE = ["elo_diff", "home_elo", "away_elo", "neutral", "form_diff",
                  "exp_home", "abs_elo_diff", "rest_diff", "min_played"]


def poids_importance(tournoi: str) -> int:
    """Une Coupe du monde pèse plus qu'un amical dans la mise à jour de l'Elo."""
    t = tournoi.lower()
    if t == "friendly":
        return 20
    if "fifa world cup" in t and "qualif" not in t:
        return 60
    grandes = ["uefa euro", "copa am", "african cup", "asian cup", "gold cup", "confederations"]
    if any(k in t for k in grandes) and "qualif" not in t:
        return 50
    if "qualif" in t:
        return 40
    return 30


def mult_buts(ecart: int) -> float:
    """Une large victoire fait davantage bouger les notes (World Football Elo)."""
    ecart = abs(ecart)
    if ecart <= 1:
        return 1.0
    if ecart == 2:
        return 1.5
    return (11 + ecart) / 8


def features_avant_match(home, away, neutral, date, etat):
    """
    Features d'un match à partir de l'état des équipes AVANT le coup d'envoi.
    Source unique de vérité, utilisée à l'entraînement ET à la prédiction.
    `date=None` (match hypothétique sans date) neutralise le repos.
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


def calculer_features(df: pd.DataFrame):
    """
    Passe chronologique unique. Renvoie (df enrichi, etat) où `etat` contient tout
    ce dont la prédiction a besoin : Elo, forme, taux de nul, dernière date, compteur.
    """
    etat = {
        "elos": {},
        "hist": defaultdict(lambda: deque(maxlen=FENETRE_FORME)),
        "draw": defaultdict(lambda: deque(maxlen=FENETRE_NUL)),
        "last_date": {},
        "n_played": defaultdict(int),
    }
    elos, hist, draw = etat["elos"], etat["hist"], etat["draw"]

    lignes = []
    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        feats = features_avant_match(home, away, row.neutral, row.date, etat)
        lignes.append(feats)
        r_home, r_away, exp_home = feats["home_elo"], feats["away_elo"], feats["exp_home"]

        if row.home_score > row.away_score:
            s_home, pts_home, pts_away, est_nul = 1.0, 3, 0, 0
        elif row.home_score == row.away_score:
            s_home, pts_home, pts_away, est_nul = 0.5, 1, 1, 1
        else:
            s_home, pts_home, pts_away, est_nul = 0.0, 0, 3, 0

        k = poids_importance(row.tournament) * mult_buts(row.home_score - row.away_score)
        elos[home] = r_home + k * (s_home - exp_home)
        elos[away] = r_away + k * ((1 - s_home) - (1 - exp_home))
        hist[home].append(pts_home); hist[away].append(pts_away)
        draw[home].append(est_nul); draw[away].append(est_nul)
        etat["last_date"][home] = row.date; etat["last_date"][away] = row.date
        etat["n_played"][home] += 1; etat["n_played"][away] += 1

    feat_df = pd.DataFrame(lignes, index=df.index)
    feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c in df.columns])
    df = pd.concat([df, feat_df], axis=1)
    return df, etat
