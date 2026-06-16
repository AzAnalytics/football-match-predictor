"""
Interface Streamlit pour le modèle de prédiction de matchs.

Lancement :
    pip install streamlit
    streamlit run app.py

Le fichier prediction_foot.py doit être dans le MÊME dossier (l'app y puise
ses fonctions). Le CSV doit être accessible via le CHEMIN_CSV défini dans
prediction_foot.py (par défaut "archive/results.csv").
"""

import numpy as np
import pandas as pd
import streamlit as st

# On réutilise tout le travail déjà fait dans prediction_foot.py.
# Grâce au garde "if __name__ == '__main__'", cet import ne relance PAS
# l'entraînement : il ne charge que les fonctions et les constantes.
from prediction_foot import (
    charger_donnees,
    calculer_features,
    entrainer,
    FEATURES,
)

st.set_page_config(page_title="Prédiction Foot", page_icon="⚽", layout="centered")


# ---------------------------------------------------------------------------
# Préparation (mise en cache : ne s'exécute qu'une seule fois par session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des données et entraînement du modèle…")
def preparer():
    df = charger_donnees()
    df, elos, hist = calculer_features(df)
    modele = entrainer(df, rapport=False)
    derniere_date = df["date"].max().date()
    return modele, elos, hist, derniere_date


def proba_match(modele, elos, hist, dom, ext, terrain_neutre):
    """Renvoie un dict {classe: probabilité} pour le match demandé."""
    r_dom, r_ext = elos[dom], elos[ext]
    f_dom = np.mean(hist[dom]) if len(hist[dom]) else 1.0
    f_ext = np.mean(hist[ext]) if len(hist[ext]) else 1.0

    X = pd.DataFrame([{
        "elo_diff": r_dom - r_ext,
        "form_diff": f_dom - f_ext,
        "home_elo": r_dom,
        "away_elo": r_ext,
        "neutral": terrain_neutre,
    }])[FEATURES]

    proba = modele.predict_proba(X)[0]
    return dict(zip(modele.classes_, proba))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
modele, elos, hist, derniere_date = preparer()
equipes = sorted(elos)

st.title("⚽ Prédiction de match international")
st.caption(f"Modèle Elo + forme récente · données à jour au {derniere_date}")

# Valeurs par défaut si présentes dans le dataset
def index_defaut(nom, secours=0):
    return equipes.index(nom) if nom in equipes else secours

col1, col2 = st.columns(2)
with col1:
    domicile = st.selectbox("Équipe à domicile", equipes, index=index_defaut("France"))
with col2:
    exterieur = st.selectbox("Équipe à l'extérieur", equipes, index=index_defaut("Senegal", 1))

terrain_neutre = st.checkbox(
    "Terrain neutre (ex : Coupe du monde)",
    value=True,
    help="Coché, l'avantage du terrain est neutralisé pour l'équipe à domicile.",
)

if domicile == exterieur:
    st.warning("Choisis deux équipes différentes.")
    st.stop()

# --- Prédiction ---
p = proba_match(modele, elos, hist, domicile, exterieur, terrain_neutre)
labels = {"domicile": f"Victoire {domicile}", "nul": "Match nul", "exterieur": f"Victoire {exterieur}"}

st.subheader("Probabilités")
c1, c2, c3 = st.columns(3)
c1.metric(labels["domicile"], f"{p['domicile']:.1%}")
c2.metric(labels["nul"], f"{p['nul']:.1%}")
c3.metric(labels["exterieur"], f"{p['exterieur']:.1%}")

# Graphique en barres (ordre logique domicile / nul / extérieur)
ordre = ["domicile", "nul", "exterieur"]
chart_df = pd.DataFrame(
    {"probabilité": [p[k] for k in ordre]},
    index=[labels[k] for k in ordre],
)
st.bar_chart(chart_df)

# Issue la plus probable
favori = max(p, key=p.get)
st.success(f"Issue la plus probable : **{labels[favori]}**  ({p[favori]:.1%})")

# Repères Elo
st.divider()
e1, e2 = st.columns(2)
e1.metric(f"Elo {domicile}", f"{elos[domicile]:.0f}")
e2.metric(f"Elo {exterieur}", f"{elos[exterieur]:.0f}")

st.caption(
    "Rappel : ces probabilités reposent sur l'historique des résultats "
    "(force et forme des équipes), sans tenir compte des compositions, "
    "blessures ou enjeu du match."
)