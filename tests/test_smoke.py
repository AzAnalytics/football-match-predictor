"""Tests de fumée : la librairie charge, entraîne et prédit de façon cohérente."""
import os
import shutil
from pathlib import Path

import pytest

from footpredictor import Predictor
# Utilise l'instantané embarqué comme cache (pas de réseau pendant les tests)
import footpredictor.data as data


@pytest.fixture(scope="session", autouse=True)
def _cache_offline(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cache")
    os.environ["FOOTPREDICTOR_CACHE"] = str(cache_dir)
    shutil.copy(data._BUNDLED, cache_dir / "results.csv")
    yield


def test_load_results():
    df = data.load_results(max_age_days=10**9)   # cache frais -> pas de téléchargement
    assert len(df) > 40000
    assert set(df["resultat"].unique()) <= {"domicile", "nul", "exterieur"}
    assert df["date"].is_monotonic_increasing


@pytest.fixture(scope="session")
def predictor():
    
    return Predictor(max_age_days=10**9)


def test_predict_structure(predictor):
    res = predictor.predict("France", "Senegal", neutral=True)
    assert abs(sum(res["proba_issue"].values()) - 1) < 1e-6
    p = res["proba_1x2"]
    assert abs(p["domicile"] + p["nul"] + p["exterieur"] - 1) < 1e-6
    i, j = res["score_exact"]
    assert 0 <= i <= 8 and 0 <= j <= 8


def test_draw_emerges_for_close_teams(predictor):
    # deux grosses équipes proches -> un nul doit avoir une proba notable
    res = predictor.predict("Brazil", "Argentina", neutral=False)
    assert res["proba_1x2"]["nul"] > 0.15


def test_unknown_team_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict("Wakanda", "France")
