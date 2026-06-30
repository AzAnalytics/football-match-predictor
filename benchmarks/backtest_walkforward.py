"""
Backtest en validation glissante (walk-forward) 2014-2025, basé sur la librairie.

Pour chaque saison Y : entraîner sur tout le passé, prédire l'année Y (jamais vue),
comparer à la baseline. Vérifie que la performance est STABLE dans le temps.

    python benchmarks/backtest_walkforward.py
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score

from footpredictor import load_results, calculer_features, config_mode

ANNEES_TEST = range(2014, 2026)


def backtest(mode="equilibre"):
    df = load_results()
    df, _ = calculer_features(df)
    feats, modele_ref = config_mode(mode)
    print(f"Mode : {mode}\n")
    print(f"{'Année':>6} {'Matchs':>7} {'Baseline':>9} {'Modèle':>8} {'Gain':>7} {'Recall nul':>11}")

    lignes = []
    for annee in ANNEES_TEST:
        debut = pd.Timestamp(f"{annee}-01-01")
        fin = pd.Timestamp(f"{annee + 1}-01-01")
        train = df[df["date"] < debut]
        test = df[(df["date"] >= debut) & (df["date"] < fin)]
        if len(test) < 50:
            continue
        modele = clone(modele_ref).fit(train[feats], train["resultat"])
        pred = modele.predict(test[feats])
        acc = accuracy_score(test["resultat"], pred)
        base = (test["resultat"] == "domicile").mean()
        rec = recall_score(test["resultat"], pred, labels=["nul"], average="macro", zero_division=0)
        lignes.append({"annee": annee, "n": len(test), "base": base, "acc": acc, "rec": rec})
        print(f"{annee:>6} {len(test):>7} {base:>8.1%} {acc:>7.1%} {(acc-base)*100:>+6.1f} {rec:>10.1%}")

    res = pd.DataFrame(lignes)
    print("-" * 54)
    print(f"{'MOY':>6} {int(res.n.mean()):>7} {res.base.mean():>8.1%} {res.acc.mean():>7.1%} "
          f"{(res.acc-res.base).mean()*100:>+6.1f} {res.rec.mean():>10.1%}")
    print(f"\nÉcart-type précision : {res.acc.std()*100:.1f} pts (plus bas = plus stable)")
    return res


if __name__ == "__main__":
    backtest()
