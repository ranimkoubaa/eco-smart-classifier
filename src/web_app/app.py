"""
Eco-Smart Classifier — Streamlit Web App  (Personne 3)
3 tabs: Dashboard Data | Prédiction Manuelle | Assistant NLP
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Eco-Smart Classifier ♻️", page_icon="♻️",
                   layout="wide", initial_sidebar_state="collapsed")

API_URL = os.getenv("API_URL", "http://localhost:8000")

COLORS = {"Plastique": "#2196F3", "Verre": "#4CAF50",
          "Papier": "#FF9800", "Métal": "#9C27B0"}
ICONS  = {"Plastique": "🧴", "Verre": "🫙", "Papier": "📄", "Métal": "🔩"}
SOURCES = ["Centre_Tri", "Collecte_Citoyenne", "Inconnu", "Usine_A", "Usine_B"]

st.markdown("""
<style>
  .result-box {
    border-radius:12px; padding:1.5rem; text-align:center;
    font-size:1.6rem; font-weight:700; margin-top:1rem;
    background:linear-gradient(135deg,#E8F5E9,#C8E6C9);
  }
  .info-box {
    background:#F3F4F6; border-left:4px solid #2E7D32;
    border-radius:6px; padding:0.8rem 1rem; margin:0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def load_clean():
    for p in ["data/processed/dataset_clean.csv",
              "../data/processed/dataset_clean.csv",
              "/app/data/processed/dataset_clean.csv"]:
        if Path(p).exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def load_clusters():
    for p in ["reports/cluster_results.csv",
              "../reports/cluster_results.csv",
              "/app/reports/cluster_results.csv"]:
        if Path(p).exists():
            return pd.read_csv(p)
    return None


def call_api(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API unreachable — start the FastAPI server first.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#2E7D32'>♻️ Eco-Smart Classifier</h1>"
    "<p style='text-align:center;color:#777'>Classification de déchets · Estimation de valeur</p>",
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard Data",
                              "🎛️ Prédiction Manuelle",
                              "💬 Assistant NLP"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    df = load_clean()
    if df is None:
        st.error("dataset_clean.csv introuvable. Lancez d'abord 02_preprocessing.ipynb.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total échantillons",  f"{len(df):,}")
    c2.metric("Étiquetés",           f"{df['Categorie'].notna().sum():,}")
    c3.metric("Non étiquetés",       f"{df['Categorie'].isna().sum()}")
    c4.metric("Valeurs manquantes",  f"{df[['Poids','Volume','Conductivite','Opacite','Rigidite']].isnull().sum().sum():,}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution des catégories")
        fig = px.pie(df["Categorie"].value_counts(dropna=False).reset_index(),
                     names="Categorie", values="count",
                     color="Categorie", color_discrete_map=COLORS, hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribution par source")
        # Source was OHE-encoded by Personne 1 — reconstruct from dummy columns
        source_cols = [c for c in df.columns if c.startswith("Source_") and c != "Source_encoded"]
        if source_cols:
            src_counts = {col.replace("Source_", ""): int(df[col].sum()) for col in source_cols}
            src = pd.DataFrame(src_counts.items(), columns=["Source", "count"])
            fig2 = px.bar(src, x="Source", y="count", color="Source", text="count")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Colonnes Source non trouvées.")

    st.divider()
    st.subheader("Distribution des features numériques")
    num_col = st.selectbox("Feature", ["Poids", "Volume", "Conductivite",
                                        "Opacite", "Rigidite", "Prix_Revente"])
    fig3 = px.histogram(df.dropna(subset=[num_col]), x=num_col,
                         color="Categorie", color_discrete_map=COLORS,
                         barmode="overlay", opacity=0.7, nbins=60)
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Clusters PCA 2D")
    clusters = load_clusters()
    if clusters is not None:
        fig4 = px.scatter(clusters, x="PCA1", y="PCA2", color="Cluster",
                           title="Clusters projetés en 2D (KMeans + PCA)")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Lancez 05_clustering.ipynb et exportez reports/cluster_results.csv.")

    st.divider()
    st.subheader("Aperçu dataset")
    st.dataframe(df.head(200), use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Prédiction Manuelle
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Ajustez les curseurs — prédiction en temps réel")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            poids        = st.slider("Poids (kg)",       0.0, 500.0,  40.0, 0.5)
            volume       = st.slider("Volume (L)",        0.0, 560.0, 100.0, 1.0)
            conductivite = st.slider("Conductivité",      0.0,   1.0,   0.2, 0.01)
        with c2:
            opacite      = st.slider("Opacité",           0.0,  55.0,   1.0, 0.1)
            rigidite     = st.slider("Rigidité (1-10)",   1.0,  10.0,   5.0, 0.5)
            source       = st.selectbox("Source", SOURCES)

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

    if submitted:
        result = call_api("/predict", {
            "poids": poids, "volume": volume, "conductivite": conductivite,
            "opacite": opacite, "rigidite": rigidite, "source": source,
        })
        if result:
            cat  = result.get("categorie", "?")
            col  = COLORS.get(cat, "#4CAF50")
            icon = ICONS.get(cat, "♻️")
            st.markdown(
                f"<div class='result-box' style='border:3px solid {col}'>"
                f"{icon} Catégorie : <span style='color:{col}'>{cat}</span></div>",
                unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            if result.get("confidence"):
                m1.metric("Confiance", f"{result['confidence']*100:.1f}%")
            if result.get("prix_estime"):
                m2.metric("Prix estimé", f"{result['prix_estime']:.2f} €")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Assistant NLP
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Décrivez le déchet — le pipeline NLP l'identifie")

    st.markdown("""
    <div class='info-box'>
    <b>Exemples :</b><br>
    • <i>Lot de bouteilles en plastique transparent, léger, souple, non conducteur.</i><br>
    • <i>Bris de verre récupéré en conteneur, matériau dur et très conducteur.</i><br>
    • <i>Cartons et feuilles collectés au centre de tri, matériau opaque et souple.</i>
    </div>
    """, unsafe_allow_html=True)

    rapport = st.text_area("Rapport de collecte",
                            placeholder="Ex : Lot de plastique souple récupéré à l'Usine A…",
                            height=130)

    if st.button("🤖 Analyser", use_container_width=True):
        if not rapport.strip():
            st.warning("Entrez une description.")
        else:
            with st.spinner("Analyse NLP…"):
                result = call_api("/predict-nlp", {"rapport": rapport})
            if result:
                cat  = result.get("categorie", "?")
                col  = COLORS.get(cat, "#4CAF50")
                icon = ICONS.get(cat, "♻️")
                st.markdown(
                    f"<div class='result-box' style='border:3px solid {col}'>"
                    f"{icon} Catégorie : <span style='color:{col}'>{cat}</span></div>",
                    unsafe_allow_html=True)
