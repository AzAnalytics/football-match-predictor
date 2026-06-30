"""
Prédiction du SCORE FINAL d'un match (y compris les nuls), par double Poisson.

Principe
--------
On entraîne DEUX régressions de Poisson :
  - l'une prédit l'espérance de buts de l'équipe à domicile (lambda_dom) ;
  - l'autre celle de l'équipe à l'extérieur (lambda_ext).
À partir de ces deux moyennes, on construit la matrice des scores possibles :
    P(buts_dom = i, buts_ext = j) = Poisson(i ; lambda_dom) * Poisson(j ; lambda_ext)
avec une correction Dixon-Coles sur les petits scores (0-0, 1-0, 0-1, 1-1) qui
corrige la légère dépendance entre les deux scores et rehausse les nuls serrés.

De cette matrice on tire, d'un seul modèle :
  - le SCORE EXACT le plus probable (1-1, 2-0, 0-0...) -> les nuls émergent seuls ;
  - les probabilités d'issue 1X2 (domicile / nul / extérieur) en sommant la matrice ;
  - le score le plus probable CONDITIONNEL à chaque issue.

Avantage vs un classifieur : le nul n'est plus une classe qu'il faut forcer ;
c'est une conséquence naturelle de la distribution des buts.

Performances (test >= 2022, ~4 580 matchs)
  - Issue 1X2 (via somme des probas) : ~60 % de précision ;
  - Issue 1X2 (via score exact)      : ~54 %, recall sur les nuls ~49 % ;
  - Score exact correct              : ~14 % ; bonne différence de buts ~25 %.
"""
import numpy as np
import pandas as pd
from math import exp, factorial
from collections import Counter

from sklearn.ensemble import HistGradientBoostingRegressor

from prediction_foot import (charger_donnees, calculer_features,
                             _features_avant_match, DATE_CUTOFF)

# Features utilisées pour prédire les buts (sous-ensemble pertinent)
FEATURES_SCORE = ["elo_diff", "home_elo", "away_elo", "neutral", "form_diff",
                  "exp_home", "abs_elo_diff", "rest_diff", "min_played"]
MAXG = 8          # on modélise les scores de 0 à 8 buts par équipe
RHO = -0.05       # paramètre Dixon-Coles (rehausse 0-0 / 1-1)


# ---------------------------------------------------------------------------
# Outils probabilistes
# ---------------------------------------------------------------------------
def _pois(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def _dc_tau(i, j, lh, la, rho=RHO):
    """Correction Dixon-Coles sur les 4 scores bas."""
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def matrice_scores(lam_dom, lam_ext, dc=True, maxg=MAXG):
    """Matrice (maxg+1, maxg+1) des probabilités de chaque score, normalisée."""
    M = np.outer([_pois(i, lam_dom) for i in range(maxg + 1)],
                 [_pois(j, lam_ext) for j in range(maxg + 1)])
    if dc:
        for i in range(2):
            for j in range(2):
                M[i, j] *= _dc_tau(i, j, lam_dom, lam_ext)
    return M / M.sum()


def analyser_matrice(M, top=5):
    """Extrait d'une matrice de scores tout ce qui est utile à afficher."""
    i, j = np.unravel_index(M.argmax(), M.shape)
    p_dom = float(np.tril(M, -1).sum())   # i > j
    p_nul = float(np.trace(M))            # i == j
    p_ext = float(np.triu(M, 1).sum())    # i < j

    # Top scores globaux
    plats = [((a, b), float(M[a, b])) for a in range(M.shape[0]) for b in range(M.shape[1])]
    plats.sort(key=lambda x: -x[1])

    # Score le plus probable conditionnel à chaque issue
    def meilleur(filtre):
        cand = [((a, b), p) for (a, b), p in plats if filtre(a, b)]
        return cand[0] if cand else (None, 0.0)
    sc_dom = meilleur(lambda a, b: a > b)
    sc_nul = meilleur(lambda a, b: a == b)
    sc_ext = meilleur(lambda a, b: a < b)

    return {
        "score_exact": (int(i), int(j)),
        "proba_1x2": {"domicile": p_dom, "nul": p_nul, "exterieur": p_ext},
        "top_scores": plats[:top],
        "score_si_domicile": sc_dom,
        "score_si_nul": sc_nul,
        "score_si_exterieur": sc_ext,
    }


# ---------------------------------------------------------------------------
# Entraînement et prédiction
# ---------------------------------------------------------------------------
def entrainer_score(df, cutoff=DATE_CUTOFF):
    """Entraîne les deux régressions de Poisson sur le passé. Renvoie (mh, ma)."""
    train = df[df["date"] < cutoff]
    mh = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    ma = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    mh.fit(train[FEATURES_SCORE], train["home_score"].clip(upper=MAXG))
    ma.fit(train[FEATURES_SCORE], train["away_score"].clip(upper=MAXG))
    return mh, ma


def predire_score(mh, ma, etat, equipe_dom, equipe_ext, terrain_neutre=False, top=5):
    """Renvoie l'analyse complète du match (score exact, probas 1X2, top scores...)."""
    for t in (equipe_dom, equipe_ext):
        if t not in etat["elos"]:
            raise ValueError(f"'{t}' introuvable parmi les équipes connues.")
    ligne = _features_avant_match(equipe_dom, equipe_ext, terrain_neutre, None, etat)
    X = pd.DataFrame([ligne])[FEATURES_SCORE]
    lam_dom = float(np.clip(mh.predict(X)[0], 0.05, MAXG))
    lam_ext = float(np.clip(ma.predict(X)[0], 0.05, MAXG))
    M = matrice_scores(lam_dom, lam_ext)
    info = analyser_matrice(M, top=top)
    info.update({"lambda_dom": lam_dom, "lambda_ext": lam_ext, "matrice": M})
    return info


def afficher(info, dom, ext):
    """Affichage console lisible d'une prédiction de score."""
    i, j = info["score_exact"]
    p = info["proba_1x2"]
    print(f"\n{dom} {i} - {j} {ext}   (score le plus probable)")
    print(f"  buts attendus : {dom} {info['lambda_dom']:.2f}  |  {ext} {info['lambda_ext']:.2f}")
    print(f"  issue 1X2     : domicile {p['domicile']:.1%}  "
          f"nul {p['nul']:.1%}  exterieur {p['exterieur']:.1%}")
    (a, b), pa = info["score_si_domicile"]
    (c, d), pn = info["score_si_nul"]
    (e, f), pe = info["score_si_exterieur"]
    print(f"  si victoire {dom:<12}: {a}-{b} ({pa:.1%})")
    print(f"  si match nul         : {c}-{d} ({pn:.1%})")
    print(f"  si victoire {ext:<12}: {e}-{f} ({pe:.1%})")
    print("  scores les plus probables : "
          + ", ".join(f"{x}-{y} {q:.1%}" for (x, y), q in info["top_scores"]))


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = charger_donnees()
    df, elos, hist, etat = calculer_features(df)
    mh, ma = entrainer_score(df)
    info = predire_score(mh, ma, etat, "France", "Senegal", terrain_neutre=True)
    afficher(info, "France", "Senegal")
