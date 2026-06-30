"""
Modèle Dixon-Coles : force d'attaque / de défense par équipe.

    log(lambda_dom) = mu + attaque[dom] - defense[ext] + avantage_terrain
    log(lambda_ext) = mu + attaque[ext] - defense[dom]

Estimé comme une régression de Poisson sur données "longues" (2 lignes/match),
avec pondération temporelle (les matchs récents pèsent plus) et correction `rho`
ajustée. Seul, il fait jeu égal avec le modèle Elo-Poisson ; son intérêt est de
fournir un point de vue différent pour l'ensemble (cf. score.predire_score).
"""
from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

from .score import _pois, MAXG

DEMI_VIE_J = 1460     # demi-vie de la pondération temporelle (jours) ~ 4 ans


class DixonColes:
    """Modèle force attaque/défense, estimé par régression de Poisson pondérée."""

    def __init__(self, maxg=MAXG, demi_vie_j=DEMI_VIE_J, alpha=1e-4):
        self.maxg = maxg
        self.demi_vie_j = demi_vie_j
        self.alpha = alpha

    def _encoder(self, att, deff, home):
        idx, n = self._idx, self._n
        m = len(att)
        a = np.array([idx.get(t, -1) for t in att])
        d = np.array([idx.get(t, -1) for t in deff])
        va = np.where(a >= 0, 1.0, 0.0)
        vd = np.where(d >= 0, -1.0, 0.0)
        a = np.where(a >= 0, a, 0)
        d = np.where(d >= 0, d, 0)
        rows = np.repeat(np.arange(m), 2)
        cols = np.empty(2 * m, dtype=int)
        vals = np.empty(2 * m)
        cols[0::2] = a; cols[1::2] = n + d
        vals[0::2] = va; vals[1::2] = vd
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(m, 2 * n))
        X = sparse.hstack([X, sparse.csr_matrix(np.asarray(home, float).reshape(-1, 1))]).tocsr()
        return X

    def fit(self, df, asof=None):
        asof = pd.Timestamp(asof) if asof is not None else df["date"].max()
        self._equipes = sorted(set(df.home_team) | set(df.away_team))
        self._idx = {t: k for k, t in enumerate(self._equipes)}
        self._n = len(self._equipes)

        att = np.concatenate([df.home_team.values, df.away_team.values])
        deff = np.concatenate([df.away_team.values, df.home_team.values])
        home = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
        buts = np.concatenate([df.home_score.clip(upper=self.maxg).values,
                               df.away_score.clip(upper=self.maxg).values])
        dates = np.concatenate([df.date.values, df.date.values])
        age = (asof - pd.to_datetime(dates)).days.values
        w = np.exp(-log(2) * np.clip(age, 0, None) / self.demi_vie_j)

        self._reg = PoissonRegressor(alpha=self.alpha, max_iter=500)
        self._reg.fit(self._encoder(att, deff, home), buts, sample_weight=w)

        lh = self.lambdas_vec(df.home_team.values, df.away_team.values, np.ones(len(df)))
        la = self.lambdas_vec(df.away_team.values, df.home_team.values, np.zeros(len(df)))
        wm = np.exp(-log(2) * np.clip((asof - df.date).dt.days.values, 0, None) / self.demi_vie_j)
        self.rho = self._ajuster_rho(df.home_score.values, df.away_score.values, lh, la, wm)
        return self

    def lambdas_vec(self, att, deff, home):
        return np.clip(self._reg.predict(self._encoder(att, deff, home)), 0.05, self.maxg)

    def _ajuster_rho(self, hs, as_, lh, la, w):
        def negll(rho):
            t = np.ones_like(lh)
            m = (hs == 0) & (as_ == 0); t[m] = 1 - lh[m] * la[m] * rho
            m = (hs == 0) & (as_ == 1); t[m] = 1 + lh[m] * rho
            m = (hs == 1) & (as_ == 0); t[m] = 1 + la[m] * rho
            m = (hs == 1) & (as_ == 1); t[m] = 1 - rho
            return -np.sum(w * np.log(np.clip(t, 1e-6, None)))
        return float(minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded").x)

    def lambdas(self, dom, ext, terrain_neutre=False):
        h = 0.0 if terrain_neutre else 1.0
        lh = self.lambdas_vec([dom], [ext], [h])[0]
        la = self.lambdas_vec([ext], [dom], [0.0])[0]
        return float(lh), float(la)

    def matrice(self, dom, ext, terrain_neutre=False):
        lh, la = self.lambdas(dom, ext, terrain_neutre)
        M = np.outer([_pois(i, lh) for i in range(self.maxg + 1)],
                     [_pois(j, la) for j in range(self.maxg + 1)])
        M[0, 0] *= 1 - lh * la * self.rho
        M[0, 1] *= 1 + lh * self.rho
        M[1, 0] *= 1 + la * self.rho
        M[1, 1] *= 1 - self.rho
        return M / M.sum()

    def forces(self, top=5):
        """Renvoie (meilleures attaques, meilleures défenses)."""
        coefs = self._reg.coef_
        eq = np.array(self._equipes)
        att, deff = coefs[:self._n], coefs[self._n:2 * self._n]
        return list(eq[np.argsort(-att)[:top]]), list(eq[np.argsort(-deff)[:top]])


def entrainer_dc(df, demi_vie_j=DEMI_VIE_J):
    return DixonColes(demi_vie_j=demi_vie_j).fit(df)
