"""
Classifieur d'issue 1X2 (domicile / nul / extérieur) avec deux modes et calibration.

  - mode "precision"  : maximise la précision globale (~60 %), ignore les nuls ;
  - mode "equilibre"  : pondération de classe + features de matchs serrés -> prédit
                        réellement les nuls (recall ~37 %, macro-F1 ~0,52).
La calibration sigmoïde rend les probabilités affichées honnêtes.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             recall_score, f1_score)

from .features import (FEATURES_PRECISION, FEATURES_EQUILIBRE, features_avant_match)

DATE_CUTOFF = "2022-01-01"   # avant = entraînement, à partir de = test/calibration


def calibrer(base, X, y, method="sigmoid"):
    """
    Calibre un modèle DÉJÀ entraîné, en restant compatible avec toutes les versions
    de scikit-learn : `FrozenEstimator` (sklearn >= 1.6) ou `cv='prefit'` (plus anciennes,
    où FrozenEstimator n'existe pas). sklearn >= 1.8 a supprimé `cv='prefit'`.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X, y)
    except ImportError:
        return CalibratedClassifierCV(base, method=method, cv="prefit").fit(X, y)


def config_mode(mode="equilibre"):
    """Renvoie (features, modèle non entraîné) pour le mode demandé."""
    if mode == "precision":
        return FEATURES_PRECISION, HistGradientBoostingClassifier(random_state=0)
    if mode == "equilibre":
        return (FEATURES_EQUILIBRE,
                HistGradientBoostingClassifier(random_state=0, class_weight="balanced"))
    raise ValueError(f"Mode inconnu : {mode!r} (attendu 'precision' ou 'equilibre')")


def entrainer(df, mode="equilibre", cutoff=DATE_CUTOFF, rapport=False):
    """Entraîne sur le passé, évalue sur le futur (métriques imprimées)."""
    feats, modele = config_mode(mode)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    pred = modele.fit(train[feats], train["resultat"]).predict(test[feats])
    if rapport:
        y = test["resultat"]
        print(f"Mode {mode} | acc={accuracy_score(y, pred):.3f} "
              f"recall_nul={recall_score(y, pred, labels=['nul'], average='macro', zero_division=0):.3f} "
              f"macroF1={f1_score(y, pred, average='macro', zero_division=0):.3f}")
        print(classification_report(y, pred, digits=3))
    return modele


def entrainer_calibre(df, mode="equilibre", cutoff=DATE_CUTOFF, method="sigmoid"):
    """
    Modèle de PRODUCTION aux probabilités calibrées. Base entraînée sur < cutoff,
    calibration (Platt) sur la période récente >= cutoff. Renvoie un estimateur
    compatible predict_proba / classes_.
    """
    feats, base = config_mode(mode)
    base.fit(df[df["date"] < cutoff][feats], df[df["date"] < cutoff]["resultat"])
    recent = df[df["date"] >= cutoff]
    return calibrer(base, recent[feats], recent["resultat"], method=method)


def proba_match(modele, etat, equipe_dom, equipe_ext, terrain_neutre=False, mode="equilibre"):
    """Renvoie {classe: proba} pour le match choisi."""
    feats = config_mode(mode)[0]
    ligne = features_avant_match(equipe_dom, equipe_ext, terrain_neutre, None, etat)
    X = pd.DataFrame([ligne])[feats]
    proba = modele.predict_proba(X)[0]
    return dict(zip(modele.classes_, proba))
