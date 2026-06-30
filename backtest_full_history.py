"""
Backtest walk-forward sur TOUT l'historique (depuis le premier match, 1872).

Même principe que backtest_walkforward.py, mais on teste chaque saison depuis
le début, et on agrège les résultats PAR DÉCENNIE pour que ce soit lisible.

Intérêt pédagogique : voir l'évolution de la performance dans le temps.
Aux débuts (peu de matchs, Elo qui démarre à froid à 1500), le modèle est
souvent MOINS bon que la baseline. Plus l'historique grandit, plus il prend
l'avantage. C'est une illustration concrète de "un modèle a besoin de données".
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from prediction_foot import charger_donnees, calculer_features, config_mode, MODE

# Garde-fous : on ne teste une saison que si elle a assez de matchs ET
# que le modèle a un minimum d'historique pour apprendre.
MIN_MATCHS_TEST = 50
MIN_MATCHS_TRAIN = 200


def backtest_complet(mode=MODE):
    df = charger_donnees()
    df, _, _, _ = calculer_features(df)
    feats, modele_ref = config_mode(mode)
    an_min = int(df["date"].dt.year.min())
    print(f"Mode : {mode}\n")

    lignes = []
    for annee in range(an_min + 1, 2026):
        debut = pd.Timestamp(f"{annee}-01-01")
        fin = pd.Timestamp(f"{annee + 1}-01-01")
        train = df[df["date"] < debut]
        test = df[(df["date"] >= debut) & (df["date"] < fin)]
        if len(test) < MIN_MATCHS_TEST or len(train) < MIN_MATCHS_TRAIN:
            continue

        modele = clone(modele_ref)
        modele.fit(train[feats], train["resultat"])
        acc = accuracy_score(test["resultat"], modele.predict(test[feats]))
        base = (test["resultat"] == "domicile").mean()
        lignes.append({"annee": annee, "n": len(test), "base": base, "acc": acc})

    res = pd.DataFrame(lignes)
    res["decennie"] = (res["annee"] // 10) * 10

    print(f"Période testée : {res.annee.min()} -> {res.annee.max()} "
          f"({len(res)} saisons)\n")
    print(f"{'Décennie':>9} {'Saisons':>8} {'Matchs/an':>10} "
          f"{'Baseline':>9} {'Modèle':>8} {'Gain':>7}")
    for dec, g in res.groupby("decennie"):
        print(f"{int(dec):>7}s {len(g):>8} {int(g.n.mean()):>10} "
              f"{g.base.mean():>8.1%} {g.acc.mean():>7.1%} "
              f"{(g.acc - g.base).mean() * 100:>+6.1f}")
    print("-" * 54)
    print(f"{'GLOBAL':>9} {len(res):>8} {int(res.n.mean()):>10} "
          f"{res.base.mean():>8.1%} {res.acc.mean():>7.1%} "
          f"{(res.acc - res.base).mean() * 100:>+6.1f}")
    recent = res[res.annee >= 2000]
    print(f"{'2000+':>9} {len(recent):>8} {int(recent.n.mean()):>10} "
          f"{recent.base.mean():>8.1%} {recent.acc.mean():>7.1%} "
          f"{(recent.acc - recent.base).mean() * 100:>+6.1f}")

    # Version lisible pour le CSV : pourcentages arrondis à 2 décimales
    res_csv = res.copy()
    res_csv["base"] = (res_csv["base"] * 100).round(2)
    res_csv["acc"] = (res_csv["acc"] * 100).round(2)
    res_csv["gain_pts"] = (res_csv["acc"] - res_csv["base"]).round(2)
    res_csv = res_csv.rename(columns={"n": "n_matchs", "base": "baseline_%", "acc": "modele_%"})
    res_csv.to_csv("backtest_full_history.csv", index=False)
    print("\nDétail année par année : backtest_full_history.csv")
    return res


if __name__ == "__main__":
    backtest_complet()
