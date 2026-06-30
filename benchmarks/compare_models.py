"""
Comparaison probabiliste de tous les modèles (RPS / log-loss / Brier), basée
sur la librairie. Découpage temporel : train < 2020, calibration [2020,2022),
test >= 2022.

    python benchmarks/compare_models.py
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from footpredictor import load_results, calculer_features, config_mode
from footpredictor.classifier import DATE_CUTOFF
from footpredictor.score import entrainer_score, matrice_scores, MAXG
from footpredictor.dixon_coles import DixonColes
from footpredictor.features import FEATURES_SCORE, FEATURES_PRECISION, FEATURES_EQUILIBRE
from footpredictor.evaluation import toutes_metriques, proba_ordonnee

CUT_TRAIN, CUT_TEST = "2020-01-01", "2022-01-01"


def _p1x2(mats):
    return np.array([[np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()] for M in mats])


def main():
    df = load_results()
    df, _ = calculer_features(df)
    tr = df[df["date"] < CUT_TRAIN]
    cal = df[(df["date"] >= CUT_TRAIN) & (df["date"] < CUT_TEST)]
    te = df[df["date"] >= CUT_TEST]
    y = te["resultat"].values
    res = {}

    def ligne(nom, P):
        m = toutes_metriques(y, P)
        print(f"{nom:<42} acc={m['acc']:6.1%}  RPS={m['RPS']:.4f}  "
              f"logloss={m['logloss']:.4f}  Brier={m['Brier']:.4f}")
        res[nom] = m["RPS"]

    f = tr["resultat"].value_counts(normalize=True)
    ligne("Baseline (fréquences)", np.tile([f["domicile"], f["nul"], f["exterieur"]], (len(te), 1)))

    for mode, feats in [("precision", FEATURES_PRECISION), ("equilibre", FEATURES_EQUILIBRE)]:
        _, base = config_mode(mode)
        base.fit(tr[feats], tr["resultat"])
        ligne(f"Classifieur {mode} (brut)", proba_ordonnee(base.predict_proba(te[feats]), base.classes_))
        cc = CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(cal[feats], cal["resultat"])
        ligne(f"Classifieur {mode} (calibré)", proba_ordonnee(cc.predict_proba(te[feats]), cc.classes_))

    mh, ma = entrainer_score(tr, sur_tout=True)
    lh = np.clip(mh.predict(te[FEATURES_SCORE]), 0.05, MAXG)
    la = np.clip(ma.predict(te[FEATURES_SCORE]), 0.05, MAXG)
    mats_elo = [matrice_scores(h, a) for h, a in zip(lh, la)]
    ligne("Modèle Elo-Poisson", _p1x2(mats_elo))

    dc = DixonColes().fit(tr, asof=CUT_TEST)
    mats_dc = [dc.matrice(d, x, bool(n)) for d, x, n in zip(te.home_team, te.away_team, te.neutral)]
    ligne("Modèle Dixon-Coles", _p1x2(mats_dc))
    ligne("Ensemble (Elo-Poisson + Dixon-Coles)", _p1x2([(a + b) / 2 for a, b in zip(mats_elo, mats_dc)]))

    print("\n>>> Classement par RPS (plus bas = meilleur) :")
    for nom, r in sorted(res.items(), key=lambda x: x[1]):
        print(f"   {nom:<42} RPS={r:.4f}")


if __name__ == "__main__":
    main()
