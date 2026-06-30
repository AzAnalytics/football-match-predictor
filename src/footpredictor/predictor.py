"""
API haut niveau : la classe Predictor.

C'est le point d'entrée de la librairie. Elle gère tout : téléchargement
automatique des données, calcul des features, entraînement des modèles, et
prédiction d'un match en une ligne.

    from footpredictor import Predictor
    p = Predictor()                                  # télécharge + entraîne (en cache)
    res = p.predict("France", "Senegal", neutral=True)
    print(res["score_exact"], res["proba_1x2"])
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import load_results, update_data
from .features import calculer_features
from .classifier import entrainer_calibre, proba_match
from .score import entrainer_score, predire_score
from .dixon_coles import entrainer_dc


@dataclass
class Predictor:
    """Façade : données auto-rafraîchies + modèles entraînés + prédiction."""

    mode: str = "equilibre"          # "equilibre" ou "precision"
    refresh: bool = False            # forcer le téléchargement à l'init
    max_age_days: float = 1.0        # tolérance de fraîcheur du cache
    verbose: bool = False
    avec_score: bool = True          # entraîner aussi les modèles de score
    _pret: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.charger()

    # -- cycle de vie --
    def charger(self):
        """(Re)charge les données et (ré)entraîne tous les modèles."""
        df = load_results(refresh=self.refresh, max_age_days=self.max_age_days,
                          verbose=self.verbose)
        self.df, self.etat = calculer_features(df)
        self.derniere_date = self.df["date"].max().date()
        self.modele = entrainer_calibre(self.df, mode=self.mode)
        if self.avec_score:
            self.mh, self.ma = entrainer_score(self.df)
            self.dc = entrainer_dc(self.df)
        self._pret = True
        return self

    def update(self):
        """Force le téléchargement des dernières données puis ré-entraîne."""
        update_data(verbose=self.verbose)
        self.refresh = False
        return self.charger()

    # -- utilitaires équipes --
    @property
    def equipes(self):
        return sorted(self.etat["elos"])

    def chercher_equipe(self, motif):
        return [t for t in self.equipes if motif.lower() in t.lower()]

    def top_elo(self, n=10):
        return sorted(self.etat["elos"].items(), key=lambda x: -x[1])[:n]

    def elo(self, equipe):
        return self.etat["elos"].get(equipe)

    # -- prédiction --
    def predict(self, domicile, exterieur, neutral=False):
        """
        Prédit un match. Renvoie un dict avec :
          - 'proba_issue'  : probas 1X2 calibrées du classifieur ;
          - 'score_exact'  : score le plus probable (ensemble Elo-Poisson + Dixon-Coles) ;
          - 'proba_1x2'    : probas 1X2 issues du modèle de score ;
          - 'lambda_dom' / 'lambda_ext', 'top_scores', 'matrice', etc.
        """
        for t in (domicile, exterieur):
            if t not in self.etat["elos"]:
                suggest = self.chercher_equipe(t[:3]) if len(t) >= 3 else []
                raise ValueError(
                    f"Équipe inconnue : {t!r}. " +
                    (f"Vouliez-vous : {suggest[:5]} ?" if suggest else
                     "Utilisez .chercher_equipe('...')."))
        res = {"proba_issue": proba_match(self.modele, self.etat, domicile, exterieur,
                                          neutral, mode=self.mode)}
        if self.avec_score:
            res.update(predire_score(self.mh, self.ma, self.etat, domicile, exterieur,
                                     neutral, dc=self.dc))
        return res
