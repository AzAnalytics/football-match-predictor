"""
Accès aux données — téléchargées et rafraîchies AUTOMATIQUEMENT.

Plus besoin d'ajouter les résultats à la main : les données proviennent du dépôt
public `martj42/international_results` (licence CC0, mis à jour après les matchs).
`load_results()` télécharge le CSV, le met en cache localement, et ne le
re-télécharge que lorsqu'il est périmé. Hors-ligne, il bascule sur la copie cache
puis sur un instantané embarqué dans le package — la librairie marche donc toujours.

Exemples :
    from footpredictor.data import load_results
    df = load_results()                  # cache si frais (< 1 j), sinon télécharge
    df = load_results(refresh=True)       # force le téléchargement
    df = load_results(max_age_days=7)     # tolère un cache d'une semaine
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Source officielle (CSV brut, public, CC0)
RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED = _PKG_DIR / "data" / "results.csv"     # instantané de secours hors-ligne


def _cache_path() -> Path:
    """Emplacement du cache (surchargé par la variable FOOTPREDICTOR_CACHE)."""
    base = Path(os.environ.get("FOOTPREDICTOR_CACHE",
                               Path.home() / ".cache" / "footpredictor"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "results.csv"


def _telecharger(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "footpredictor"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def chemin_donnees(refresh: bool = False, max_age_days: float = 1.0,
                   url: str = RAW_URL, verbose: bool = False) -> Path:
    """
    Renvoie le chemin d'un CSV de résultats à jour, en le téléchargeant si besoin.

    Stratégie : cache frais -> on le garde ; cache périmé ou `refresh` -> on tente
    le téléchargement ; échec réseau -> on retombe sur le cache puis sur
    l'instantané embarqué.
    """
    cache = _cache_path()
    frais = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_days * 86400

    if not refresh and frais:
        return cache

    try:
        donnees = _telecharger(url)
        if len(donnees) < 100_000:                       # garde-fou : réponse trop courte
            raise ValueError("réponse de téléchargement anormalement courte")
        cache.write_bytes(donnees)
        if verbose:
            print(f"[footpredictor] données mises à jour depuis {url}")
        return cache
    except Exception as e:                                # pas de réseau, URL down, etc.
        if cache.exists():
            if verbose:
                print(f"[footpredictor] téléchargement impossible ({e}); cache local utilisé")
            return cache
        if verbose:
            print(f"[footpredictor] téléchargement impossible ({e}); instantané embarqué utilisé")
        return _BUNDLED


def _nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les matchs non joués, crée la cible 'resultat', trie par date."""
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    conditions = [df["home_score"] > df["away_score"],
                  df["home_score"] < df["away_score"]]
    df["resultat"] = np.select(conditions, ["domicile", "exterieur"], default="nul")
    return df.sort_values("date").reset_index(drop=True)


def load_results(refresh: bool = False, max_age_days: float = 1.0,
                 url: str = RAW_URL, verbose: bool = False) -> pd.DataFrame:
    """Charge les résultats (auto-rafraîchis) nettoyés et prêts pour le pipeline."""
    chemin = chemin_donnees(refresh=refresh, max_age_days=max_age_days,
                            url=url, verbose=verbose)
    df = pd.read_csv(chemin, parse_dates=["date"])
    return _nettoyer(df)


def update_data(url: str = RAW_URL, verbose: bool = True) -> Path:
    """Force le téléchargement de la dernière version et renvoie le chemin du cache."""
    return chemin_donnees(refresh=True, url=url, verbose=verbose)
