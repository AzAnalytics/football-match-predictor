"""
Backtest en validation glissante (walk-forward) du modèle de prédiction foot.

Idée : au lieu d'UN seul découpage train/test (avant/après 2022), on teste le
modèle sur PLUSIEURS années successives. Pour chaque année cible Y :
  - on entraîne sur TOUS les matchs joués AVANT le 1er janvier de l'année Y
    (fenêtre "expanding" : l'historique grandit à chaque pas) ;
  - on prédit les matchs de l'année Y (que le modèle n'a jamais vus) ;
  - on compare à la baseline "toujours l'équipe à domicile".

Les features (Elo, forme) sont déjà calculées de façon causale dans
prediction_foot.py (uniquement à partir du passé de chaque match), donc
découper ensuite par date ne crée aucune fuite de données.

But : vérifier que le ~60% est STABLE dans le temps, pas un coup de chance.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score

# On réutilise TON code, sans le dupliquer
from prediction_foot import charger_donnees, calculer_features, config_mode, MODE

ANNEES_TEST = range(2014, 2026)   # on teste chaque saison de 2014 à 2025

def backtest(mode=MODE):
    df = charger_donnees()
    df, _, _, _ = calculer_features(df)   # features causales calculées une seule fois
    feats, modele_ref = config_mode(mode)
    print(f"Mode : {mode}\n")

    lignes = []
    for annee in ANNEES_TEST:
        debut = pd.Timestamp(f"{annee}-01-01")
        fin = pd.Timestamp(f"{annee + 1}-01-01")

        train = df[df["date"] < debut]
        test = df[(df["date"] >= debut) & (df["date"] < fin)]
        if len(test) < 50:           # on saute les années sans assez de matchs
            continue

        modele = clone(modele_ref)
        modele.fit(train[feats], train["resultat"])
        pred = modele.predict(test[feats])

        acc = accuracy_score(test["resultat"], pred)
        baseline = (test["resultat"] == "domicile").mean()
        # capacité à détecter les nuls (recall sur la classe "nul")
        rec_nul = recall_score(test["resultat"], pred, labels=["nul"],
                               average="macro", zero_division=0)

        lignes.append({
            "annee": annee, "n_matchs": len(test),
            "baseline": baseline, "modele": acc,
            "gain_pts": (acc - baseline) * 100, "recall_nul": rec_nul,
        })

    res = pd.DataFrame(lignes)
    # Affichage propre
    print(f"{'Année':>6} {'Matchs':>7} {'Baseline':>9} {'Modèle':>8} {'Gain':>7} {'Recall nul':>11}")
    for _, r in res.iterrows():
        print(f"{int(r.annee):>6} {int(r.n_matchs):>7} "
              f"{r.baseline:>8.1%} {r.modele:>7.1%} {r.gain_pts:>+6.1f} {r.recall_nul:>10.1%}")

    print("-" * 52)
    print(f"{'MOYENNE':>6} {int(res.n_matchs.mean()):>7} "
          f"{res.baseline.mean():>8.1%} {res.modele.mean():>7.1%} "
          f"{res.gain_pts.mean():>+6.1f} {res.recall_nul.mean():>10.1%}")
    print(f"\nÉcart-type de la précision du modèle : {res.modele.std()*100:.1f} pts "
          f"(plus c'est bas, plus c'est stable)")
    # Version lisible pour le CSV : pourcentages arrondis à 2 décimales
    res_csv = res.copy()
    for c in ["baseline", "modele", "recall_nul"]:
        res_csv[c] = (res_csv[c] * 100).round(2)
    res_csv["gain_pts"] = res_csv["gain_pts"].round(2)
    res_csv = res_csv.rename(columns={"baseline": "baseline_%", "modele": "modele_%",
                                      "recall_nul": "recall_nul_%"})
    res_csv.to_csv("backtest_resultats.csv", index=False)
    print("Résultats détaillés sauvegardés dans backtest_resultats.csv")
    return res

if __name__ == "__main__":
    backtest()
