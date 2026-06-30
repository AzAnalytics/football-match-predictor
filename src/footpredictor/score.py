"""
Prédiction du score final par double régression de Poisson (+ ensemble Dixon-Coles).

Deux régressions estiment les buts attendus de chaque équipe -> matrice de tous
les scores (correction Dixon-Coles), d'où le score exact le plus probable, les
probas 1X2 et le score conditionnel à chaque issue. Les nuls émergent naturellement.
"""
from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES_SCORE, features_avant_match

MAXG = 8       # scores modélisés de 0 à 8 buts par équipe
RHO = -0.05    # correction Dixon-Coles (rehausse 0-0 / 1-1)


def _pois(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def _dc_tau(i, j, lh, la, rho=RHO):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def matrice_scores(lam_dom, lam_ext, dc=True, maxg=MAXG):
    """Matrice (maxg+1)² des probabilités de chaque score, normalisée."""
    M = np.outer([_pois(i, lam_dom) for i in range(maxg + 1)],
                 [_pois(j, lam_ext) for j in range(maxg + 1)])
    if dc:
        for i in range(2):
            for j in range(2):
                M[i, j] *= _dc_tau(i, j, lam_dom, lam_ext)
    return M / M.sum()


def analyser_matrice(M, top=5):
    """Score exact, probas 1X2, top scores, score conditionnel à chaque issue."""
    i, j = np.unravel_index(M.argmax(), M.shape)
    p_dom = float(np.tril(M, -1).sum())
    p_nul = float(np.trace(M))
    p_ext = float(np.triu(M, 1).sum())
    plats = [((a, b), float(M[a, b])) for a in range(M.shape[0]) for b in range(M.shape[1])]
    plats.sort(key=lambda x: -x[1])

    def meilleur(filtre):
        cand = [(s, p) for s, p in plats if filtre(*s)]
        return cand[0] if cand else (None, 0.0)

    return {
        "score_exact": (int(i), int(j)),
        "proba_1x2": {"domicile": p_dom, "nul": p_nul, "exterieur": p_ext},
        "top_scores": plats[:top],
        "score_si_domicile": meilleur(lambda a, b: a > b),
        "score_si_nul": meilleur(lambda a, b: a == b),
        "score_si_exterieur": meilleur(lambda a, b: a < b),
    }


def entrainer_score(df, cutoff="2022-01-01", sur_tout=True):
    """Entraîne les deux régressions de Poisson. Renvoie (mh, ma)."""
    train = df if sur_tout else df[df["date"] < cutoff]
    mh = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    ma = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    mh.fit(train[FEATURES_SCORE], train["home_score"].clip(upper=MAXG))
    ma.fit(train[FEATURES_SCORE], train["away_score"].clip(upper=MAXG))
    return mh, ma


def _lambdas_elo(mh, ma, etat, dom, ext, neutre):
    ligne = features_avant_match(dom, ext, neutre, None, etat)
    X = pd.DataFrame([ligne])[FEATURES_SCORE]
    return (float(np.clip(mh.predict(X)[0], 0.05, MAXG)),
            float(np.clip(ma.predict(X)[0], 0.05, MAXG)))


def predire_score(mh, ma, etat, dom, ext, terrain_neutre=False, top=5, dc=None):
    """
    Analyse complète d'un match. Si `dc` (modèle DixonColes ajusté) est fourni,
    utilise l'ENSEMBLE (moyenne des deux matrices) — le meilleur prédicteur.
    """
    for t in (dom, ext):
        if t not in etat["elos"]:
            raise ValueError(f"'{t}' introuvable parmi les équipes connues.")
    lh, la = _lambdas_elo(mh, ma, etat, dom, ext, terrain_neutre)
    M_elo = matrice_scores(lh, la)
    if dc is not None:
        M_dc = dc.matrice(dom, ext, terrain_neutre)
        M = (M_elo + M_dc) / 2
        lh_dc, la_dc = dc.lambdas(dom, ext, terrain_neutre)
        lam_dom, lam_ext = (lh + lh_dc) / 2, (la + la_dc) / 2
    else:
        M, lam_dom, lam_ext = M_elo, lh, la
    info = analyser_matrice(M, top=top)
    info.update({"matrice": M, "lambda_dom": lam_dom, "lambda_ext": lam_ext})
    return info


def afficher(info, dom, ext):
    """Affichage console lisible d'une prédiction de score."""
    i, j = info["score_exact"]
    p = info["proba_1x2"]
    print(f"\n{dom} {i} - {j} {ext}   (score le plus probable)")
    print(f"  buts attendus : {dom} {info['lambda_dom']:.2f}  |  {ext} {info['lambda_ext']:.2f}")
    print(f"  issue 1X2     : domicile {p['domicile']:.1%}  "
          f"nul {p['nul']:.1%}  exterieur {p['exterieur']:.1%}")
    print("  scores probables : "
          + ", ".join(f"{x}-{y} {q:.1%}" for (x, y), q in info["top_scores"]))
