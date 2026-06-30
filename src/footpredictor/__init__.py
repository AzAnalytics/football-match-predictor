"""
footpredictor — prédiction de l'issue et du score des matchs internationaux.

Données auto-rafraîchies depuis martj42/international_results (CC0).

Usage rapide :
    from footpredictor import Predictor
    p = Predictor()
    p.predict("France", "Senegal", neutral=True)

Projet : Alexis Zueras — AZ Analytics.
"""
from .predictor import Predictor
from .data import load_results, update_data
from .features import calculer_features
from .classifier import entrainer, entrainer_calibre, proba_match, config_mode
from .score import entrainer_score, predire_score, afficher, matrice_scores
from .dixon_coles import DixonColes, entrainer_dc
from . import evaluation

__version__ = "1.0.0"

__all__ = [
    "Predictor",
    "load_results", "update_data", "calculer_features",
    "entrainer", "entrainer_calibre", "proba_match", "config_mode",
    "entrainer_score", "predire_score", "afficher", "matrice_scores",
    "DixonColes", "entrainer_dc", "evaluation",
    "__version__",
]
