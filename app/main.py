"""
Application Streamlit — construite sur la librairie `footpredictor`.

Lancement :
    pip install -e ".[app]"
    streamlit run app/main.py

Les données sont récupérées et rafraîchies automatiquement par la librairie.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from footpredictor import Predictor

VERT, GRIS, ROUGE = "#21BF73", "#8A93A6", "#E4572E"

st.set_page_config(page_title="Oracle des matchs", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1000px;}
.vs-card {background:#16263B; border-radius:18px; padding:22px 18px; text-align:center;}
.vs-team {font-size:26px; font-weight:800; color:#FFFFFF;}
.vs-elo {font-size:14px; color:#9FB3C8; margin-top:2px;}
.vs-mid {font-size:20px; font-weight:800; color:#21BF73;}
.fav-box {background:#13351f; border:1px solid #21BF73; border-radius:14px;
          padding:16px 20px; font-size:18px; color:#EAFBF0;}
.note {color:#8A93A6; font-size:13px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Récupération des données et entraînement des modèles…")
def charger(mode):
    return Predictor(mode=mode)


with st.sidebar:
    st.header("Réglage du modèle")
    mode = st.radio("Mode", ["equilibre", "precision"],
                    format_func={"equilibre": "Équilibré (prédit les nuls)",
                                 "precision": "Précision max"}.get)
    st.caption("✓ Données auto-rafraîchies depuis le dépôt public martj42 (CC0).")
    st.caption("✓ Probabilités calibrées (sigmoïde).")
    if st.button("🔄 Mettre à jour les données"):
        charger.clear()
        charger(mode).update()
        st.rerun()

pred = charger(mode)
equipes = pred.equipes


def idx(nom, secours=0):
    return equipes.index(nom) if nom in equipes else secours


st.title("⚽ Oracle des matchs")
st.caption(f"Issue + score estimés sur ≈ 49 000 matchs depuis 1872 · "
           f"données à jour au {pred.derniere_date}")

c1, c2, c3 = st.columns([5, 5, 4])
domicile = c1.selectbox("Équipe A (domicile)", equipes, index=idx("France"))
exterieur = c2.selectbox("Équipe B (extérieur)", equipes, index=idx("Senegal", 1))
neutre = c3.toggle("Terrain neutre", value=True,
                   help="Coché, l'avantage du terrain est neutralisé.")

if domicile == exterieur:
    st.warning("Choisis deux équipes différentes.")
    st.stop()

res = pred.predict(domicile, exterieur, neutral=neutre)
p = res["proba_issue"]
labels = {"domicile": f"Victoire {domicile}", "nul": "Match nul",
          "exterieur": f"Victoire {exterieur}"}

# Bandeau affiche
st.markdown(f"""
<div class="vs-card">
  <div style="display:flex; align-items:center; justify-content:space-around;">
    <div><div class="vs-team">{domicile}</div><div class="vs-elo">Elo {pred.elo(domicile):.0f}</div></div>
    <div class="vs-mid">VS</div>
    <div><div class="vs-team">{exterieur}</div><div class="vs-elo">Elo {pred.elo(exterieur):.0f}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# Probabilités d'issue (barre empilée)
st.subheader("Probabilités d'issue")
fig = go.Figure()
for cle, couleur in [("domicile", VERT), ("nul", GRIS), ("exterieur", ROUGE)]:
    fig.add_bar(x=[p[cle] * 100], y=["match"], orientation="h", marker_color=couleur,
                text=[f"{labels[cle]}<br>{p[cle]*100:.0f}%"], textposition="inside",
                insidetextanchor="middle", hovertemplate="%{x:.1f}%<extra></extra>")
fig.update_layout(barmode="stack", height=120, showlegend=False,
                  margin=dict(l=10, r=10, t=6, b=6),
                  xaxis=dict(visible=False, range=[0, 100]),
                  yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric(labels["domicile"], f"{p['domicile']:.0%}")
m2.metric(labels["nul"], f"{p['nul']:.0%}")
m3.metric(labels["exterieur"], f"{p['exterieur']:.0%}")

favori = max(p, key=p.get)
conf = ("élevée 💪" if p[favori] >= 0.65 else
        "moyenne 🤔" if p[favori] >= 0.45 else "faible, match indécis 🎲")
st.markdown(f"""<div class="fav-box">🔮 Issue la plus probable :
  <b>{labels[favori]}</b> à <b>{p[favori]:.0%}</b> &nbsp;·&nbsp; confiance {conf}</div>""",
            unsafe_allow_html=True)

# Score final (ensemble Elo-Poisson + Dixon-Coles)
st.divider()
st.subheader("Score final probable")
i, j = res["score_exact"]
sa, sb = st.columns([5, 7])
with sa:
    st.markdown(f"""
    <div class="vs-card">
      <div class="vs-elo">Score le plus probable</div>
      <div style="font-size:46px; font-weight:800; color:#FFFFFF; margin:4px 0;">
        {i} <span style="color:#21BF73;">–</span> {j}</div>
      <div class="vs-elo">{domicile} · {exterieur}</div>
      <div class="vs-elo" style="margin-top:10px;">
        Buts attendus : {res['lambda_dom']:.2f} vs {res['lambda_ext']:.2f}</div>
    </div>""", unsafe_allow_html=True)
    (a, b), pa = res["score_si_domicile"]
    (c, d), pn = res["score_si_nul"]
    (e, f), pe = res["score_si_exterieur"]
    st.markdown(
        f"<p class='note'>Score probable selon l'issue :<br>"
        f"• {labels['domicile']} : <b>{a}-{b}</b> ({pa:.0%})<br>"
        f"• Match nul : <b>{c}-{d}</b> ({pn:.0%})<br>"
        f"• {labels['exterieur']} : <b>{e}-{f}</b> ({pe:.0%})</p>", unsafe_allow_html=True)
with sb:
    NB = 6
    M = res["matrice"][:NB, :NB]
    z = (M * 100).round(1)
    fig_m = go.Figure(data=go.Heatmap(
        z=z, x=[str(k) for k in range(NB)], y=[str(k) for k in range(NB)],
        colorscale="Greens", showscale=False, text=z, texttemplate="%{text:.1f}%",
        textfont={"size": 11},
        hovertemplate=(domicile + " %{y} - %{x} " + exterieur + " : %{z:.1f}%<extra></extra>")))
    fig_m.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title=f"Buts {exterieur}", yaxis_title=f"Buts {domicile}",
                        yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, width="stretch")

st.markdown('<p class="note">Score estimé par un <b>ensemble</b> de deux modèles de buts '
            '(Elo-Poisson + Dixon-Coles). Les nuls (1-1, 0-0…) ressortent naturellement quand '
            'les équipes sont proches. Ceci n\'est pas un conseil de pari.</p>',
            unsafe_allow_html=True)
