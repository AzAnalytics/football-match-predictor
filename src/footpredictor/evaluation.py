"""
Évaluation probabiliste : RPS, log-loss, Brier (+ calibration).

La précision seule ignore la confiance ; ces métriques sont les références du
pronostic sportif. Toutes attendent une matrice de probas P (n×3) ordonnée selon
ORDER = [domicile, nul, exterieur].
"""
from __future__ import annotations

import numpy as np

ORDER = ["domicile", "nul", "exterieur"]


def _onehot(y):
    pos = {c: k for k, c in enumerate(ORDER)}
    O = np.zeros((len(y), 3))
    for r, v in enumerate(y):
        O[r, pos[v]] = 1
    return O


def rps(y_true, P):
    """Ranked Probability Score (ordinal). Plus bas = mieux."""
    O = _onehot(y_true)
    cp = np.cumsum(P, axis=1)[:, :2]
    co = np.cumsum(O, axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2))


def log_loss_mc(y_true, P, eps=1e-12):
    O = _onehot(y_true)
    return float(-np.mean(np.sum(O * np.log(np.clip(P, eps, 1)), axis=1)))


def brier(y_true, P):
    O = _onehot(y_true)
    return float(np.mean(np.sum((P - O) ** 2, axis=1)))


def accuracy(y_true, P):
    pred = [ORDER[i] for i in P.argmax(1)]
    return float(np.mean(np.array(pred) == np.array(y_true)))


def toutes_metriques(y_true, P):
    return {"acc": accuracy(y_true, P), "RPS": rps(y_true, P),
            "logloss": log_loss_mc(y_true, P), "Brier": brier(y_true, P)}


def proba_ordonnee(proba, classes):
    """Réordonne les colonnes d'une matrice de probas selon ORDER."""
    pos = [list(classes).index(c) for c in ORDER]
    return proba[:, pos]


def table_fiabilite(y_true, P, classe="nul", nbins=10, min_n=20):
    """Par tranche de proba prédite : (tranche, n, proba moyenne, fréquence réelle)."""
    k = ORDER.index(classe)
    p = P[:, k]
    o = (np.array(y_true) == classe).astype(int)
    bornes = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(p, bornes) - 1, 0, nbins - 1)
    rows = []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= min_n:
            rows.append((f"{bornes[b]:.1f}-{bornes[b+1]:.1f}", int(m.sum()),
                         float(p[m].mean()), float(o[m].mean())))
    return rows
