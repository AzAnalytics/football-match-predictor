"""
Interface Streamlit pour le modèle de prédiction de matchs internationaux.

Lancement :
    pip install -r requirements.txt
    streamlit run main.py

prediction_foot.py doit être dans le MÊME dossier (l'app y puise ses fonctions).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from prediction_foot import charger_donnees, calculer_features, entrainer, FEATURES

# ==========================================================================
# CONFIG + STYLE
# ==========================================================================
st.set_page_config(page_title="Oracle des matchs", page_icon="⚽", layout="wide")

VERT = "#21BF73"   # victoire domicile
GRIS = "#8A93A6"   # nul
ROUGE = "#E4572E"  # victoire extérieur

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1000px;}
h1, h2, h3 {letter-spacing: .2px;}
.vs-card {background:#16263B; border-radius:18px; padding:22px 18px; text-align:center;}
.vs-team {font-size:26px; font-weight:800; color:#FFFFFF;}
.vs-elo {font-size:14px; color:#9FB3C8; margin-top:2px;}
.vs-mid {font-size:20px; font-weight:800; color:#21BF73;}
.fav-box {background:#13351f; border:1px solid #21BF73; border-radius:14px;
          padding:16px 20px; font-size:18px; color:#EAFBF0;}
.note {color:#8A93A6; font-size:13px;}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# DONNÉES + MODÈLE (cache : une seule fois par session)
# ==========================================================================
@st.cache_resource(show_spinner="Chargement des données et entraînement du modèle…")
def preparer():
    df = charger_donnees()
    df, elos, hist = calculer_features(df)
    modele = entrainer(df, rapport=False)
    return modele, elos, hist, df["date"].max().date()


def proba_match(modele, elos, hist, dom, ext, neutre):
    f_dom = np.mean(hist[dom]) if len(hist[dom]) else 1.0
    f_ext = np.mean(hist[ext]) if len(hist[ext]) else 1.0
    X = pd.DataFrame([{
        "elo_diff": elos[dom] - elos[ext], "form_diff": f_dom - f_ext,
        "home_elo": elos[dom], "away_elo": elos[ext], "neutral": neutre,
    }])[FEATURES]
    proba = modele.predict_proba(X)[0]
    return dict(zip(modele.classes_, proba)), f_dom, f_ext


modele, elos, hist, derniere_date = preparer()
equipes = sorted(elos)


def idx(nom, secours=0):
    return equipes.index(nom) if nom in equipes else secours


# ==========================================================================
# EN-TÊTE + SAISIE
# ==========================================================================
st.title("⚽ Oracle des matchs")
st.caption(f"Probabilités estimées par un modèle Elo + forme récente · "
           f"≈ 49 000 matchs depuis 1872 · données à jour au {derniere_date}")

c1, c2, c3 = st.columns([5, 5, 4])
with c1:
    domicile = st.selectbox("Équipe A (domicile)", equipes, index=idx("France"))
with c2:
    exterieur = st.selectbox("Équipe B (extérieur)", equipes, index=idx("Senegal", 1))
with c3:
    neutre = st.toggle("Terrain neutre", value=True,
                       help="Coché (ex. Coupe du monde), l'avantage du terrain est neutralisé.")

if domicile == exterieur:
    st.warning("Choisis deux équipes différentes.")
    st.stop()

p, f_dom, f_ext = proba_match(modele, elos, hist, domicile, exterieur, neutre)
labels = {"domicile": f"Victoire {domicile}", "nul": "Match nul",
          "exterieur": f"Victoire {exterieur}"}

# ==========================================================================
# BANDEAU "AFFICHE"
# ==========================================================================
st.markdown(f"""
<div class="vs-card">
  <div style="display:flex; align-items:center; justify-content:space-around;">
    <div><div class="vs-team">{domicile}</div><div class="vs-elo">Elo {elos[domicile]:.0f}</div></div>
    <div class="vs-mid">VS</div>
    <div><div class="vs-team">{exterieur}</div><div class="vs-elo">Elo {elos[exterieur]:.0f}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================================================
# PROBABILITÉS : barre 100 % empilée (Plotly)
# ==========================================================================
st.subheader("Probabilités")
segments = [("domicile", VERT), ("nul", GRIS), ("exterieur", ROUGE)]
fig = go.Figure()
for cle, couleur in segments:
    fig.add_bar(x=[p[cle]*100], y=["match"], orientation="h", name=labels[cle],
                marker_color=couleur,
                text=[f"{labels[cle]}<br>{p[cle]*100:.0f}%"], textposition="inside",
                insidetextanchor="middle", hovertemplate="%{x:.1f}%<extra></extra>")
fig.update_layout(barmode="stack", height=120, showlegend=False,
                  margin=dict(l=10, r=10, t=6, b=6),
                  xaxis=dict(visible=False, range=[0, 100]),
                  yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")

# Trois cartes chiffrées
m1, m2, m3 = st.columns(3)
m1.metric(labels["domicile"], f"{p['domicile']:.0%}")
m2.metric(labels["nul"], f"{p['nul']:.0%}")
m3.metric(labels["exterieur"], f"{p['exterieur']:.0%}")

# ==========================================================================
# FAVORI + CONFIANCE
# ==========================================================================
favori = max(p, key=p.get)
proba_fav = p[favori]
if proba_fav >= 0.65:
    confiance = "élevée 💪"
elif proba_fav >= 0.45:
    confiance = "moyenne 🤔"
else:
    confiance = "faible, match indécis 🎲"

st.markdown(f"""
<div class="fav-box">
  🔮 Issue la plus probable : <b>{labels[favori]}</b> à <b>{proba_fav:.0%}</b>
  &nbsp;·&nbsp; confiance {confiance}
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# COMPARAISON DES DEUX ÉQUIPES
# ==========================================================================
st.divider()
st.subheader("Comparaison des équipes")
compa = pd.DataFrame({
    "Indicateur": ["Classement Elo", "Forme récente (pts/match, 5 derniers)"],
    domicile: [f"{elos[domicile]:.0f}", f"{f_dom:.2f}"],
    exterieur: [f"{elos[exterieur]:.0f}", f"{f_ext:.2f}"],
})
st.dataframe(compa, hide_index=True, width="stretch")

st.markdown('<p class="note">Rappel : ces probabilités reposent sur la force et la '
            'forme des équipes (historique des résultats). Elles ne tiennent pas compte '
            'des compositions, blessures ou enjeu du match. Ce n\'est pas un conseil de pari.</p>',
            unsafe_allow_html=True)
