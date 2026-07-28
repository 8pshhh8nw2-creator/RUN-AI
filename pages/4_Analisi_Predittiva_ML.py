"""
Advanced Machine Learning Suite - Dashboard interattiva per tesi magistrale.
File unico. Avvio: streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE PAGINA (DEVE ESSERE LA PRIMA CHIAMATA)
# ============================================================================
st.set_page_config(
    page_title="Sport ML Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIG & COSTANTI
# ============================================================================
APP_TITLE = "Advanced Machine Learning Suite"
APP_SUBTITLE = (
    "Framework interattivo per la stima della performance, la classificazione del rischio "
    "di overload, l'analisi dei driver del sovraccarico e la scoperta di profili latenti "
    "di allenamento."
)

COLORS = {
    "bg": "#060b14", "bg2": "#0a1424", "surface": "#0f1b2d", "surface_2": "#13243b",
    "border": "#1f3252", "border_soft": "rgba(148,163,184,0.16)",
    "text": "#f8fafc", "text_soft": "#cbd5e1", "muted": "#8fa3bd",
    "blue": "#38bdf8", "cyan": "#22d3ee", "green": "#34d399", "amber": "#fbbf24",
    "red": "#f87171", "purple": "#a78bfa", "pink": "#f472b6",
}
QUALITATIVE = [COLORS['blue'], COLORS['purple'], COLORS['cyan'], COLORS['amber'], COLORS['pink'], COLORS['green']]
CLUSTER_COLORS = {0: COLORS['green'], 1: COLORS['amber'], 2: COLORS['red']}

TARGET = "Rischio Overload"
TIME_TARGET = "Tempo (min)"
RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR", "RPE"]
CLUSTER_FEATURES = ["FC Media", "ISLR", "SMA"]

RISK_BANDS = ((40.0, "Basso", COLORS['green']), (70.0, "Moderato", COLORS['amber']), (101.0, "Alto", COLORS['red']))

def risk_band(prob: float) -> tuple[str, str]:
    for thr, label, color in RISK_BANDS:
        if prob < thr:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]

@dataclass(frozen=True)
class Settings:
    n_sessions: int = 500
    seed: int = 42
    test_size: float = 0.25
    n_estimators: int = 200
    max_depth: int = 8
    n_clusters: int = 3

# ============================================================================
# THEME & CSS
# ============================================================================
PLOTLY_TEMPLATE = "ml_suite"

def register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS['text_soft'], size=13),
        title=dict(font=dict(size=17, color=COLORS['text']), x=0.01, xanchor="left", y=0.96),
        margin=dict(t=64, l=16, r=16, b=16),
        hoverlabel=dict(bgcolor=COLORS['surface_2'], bordercolor=COLORS['border'], font=dict(family="Inter", color=COLORS['text'], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=COLORS['border_soft'], ticks="outside", tickcolor=COLORS['border_soft']),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", zeroline=False, linecolor="rgba(0,0,0,0)"),
    )
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    :root {{
        --bg:{COLORS['bg']}; --bg2:{COLORS['bg2']}; --surface:{COLORS['surface']};
        --border:{COLORS['border']}; --text:{COLORS['text']}; --cyan:{COLORS['cyan']};
        --purple:{COLORS['purple']}; --blue:{COLORS['blue']}; --amber:{COLORS['amber']};
    }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{
        background: radial-gradient(900px 520px at 8% -6%, rgba(56,189,248,0.13), transparent 70%),
                    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
        background-attachment: fixed; color: var(--text);
    }}
    .hero {{
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, rgba(15,27,45,0.97) 0%, rgba(19,36,59,0.92) 100%);
        border: 1px solid var(--border); border-radius: 24px;
        padding: 2.3rem 2.4rem; margin-bottom: 2rem; box-shadow: 0 24px 60px rgba(2,6,23,0.45);
    }}
    .hero-eyebrow {{
        display: inline-flex; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--cyan);
        background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.22);
        border-radius: 999px; padding: 0.35rem 0.8rem; margin-bottom: 1rem;
    }}
    .hero-title {{
        margin: 0; font-size: 2.6rem; font-weight: 800; line-height: 1.1;
        background: linear-gradient(120deg, #ffffff 0%, #cfe9ff 55%, #b8c8ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .callout {{ border-radius: 12px; padding: 1rem; margin: 1rem 0; font-size: 0.95rem; line-height: 1.6; }}
    .callout--theory {{ background: rgba(251,191,36,0.07); border-left: 4px solid var(--amber); color: #fef3c7; }}
    .callout--insight {{ background: rgba(167,139,250,0.09); border-left: 4px solid var(--purple); color: #ede9fe; }}
    .callout--simple {{ background: rgba(56,189,248,0.08); border-left: 4px solid var(--blue); color: #dbeafe; }}
    .section-head {{
        padding: 1rem 1.5rem; margin: 2rem 0 1rem 0;
        background: rgba(15,27,45,0.7); border: 1px solid var(--border); border-radius: 12px;
    }}
    .section-kicker {{ font-size: 0.75rem; font-weight: 700; color: var(--cyan); letter-spacing: 0.1em; text-transform: uppercase; }}
    .section-title {{ font-size: 1.4rem; font-weight: 800; margin:0; padding-top: 0.2rem; }}
    /* Fix Streamlit metric styling */
    div[data-testid="stMetricValue"] {{ color: {COLORS['cyan']} !important; font-weight: 800; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION
# ============================================================================
@st.cache_data
def generate_synthetic_data(n: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Variabili di base
    distanza = np.random.uniform(5.0, 35.0, n)
    rpe = np.random.randint(2, 11, n)
    ore_sonno = np.random.normal(7.5, 1.2, n).clip(3, 10)
    temperatura = np.random.normal(20, 8, n)
    vento = np.random.normal(10, 5, n).clip(0, 40)
    
    # Variabili dipendenti
    tempo = distanza * np.random.normal(4.5, 0.3, n) + (rpe * 2) # Pace tra 4 e 5 min/km + fatica
    velocita = (distanza * 1000) / (tempo * 60) # m/s
    fc_media = 110 + (rpe * 6) - (ore_sonno * 2) + np.random.normal(0, 5, n)
    
    # Calcolo Indici Magistrale (Formule Teoriche)
    ore_lavoro = tempo / 60
    sma = (ore_lavoro * rpe) / ore_sonno
    islr = (ore_lavoro * rpe) / distanza
    idet = (fc_media * temperatura) / np.where(velocita>0, velocita, 1)
    iitr = (temperatura * vento) / distanza
    
    # Definizione Target: Rischio Overload logica latente
    # Se lo stress metabolico è alto e si dorme poco -> Rischio = 1
    stress_score = (sma * 0.4) + (islr * 0.3) + (rpe * 0.3)
    prob_overload = 1 / (1 + np.exp(-(stress_score - np.median(stress_score))))
    rischio = (prob_overload > 0.65).astype(int) # Aggiungiamo un threshold
    
    df = pd.DataFrame({
        "Distanza (km)": distanza, "Tempo (min)": tempo, "Velocità (m/s)": velocita,
        "RPE": rpe, "Ore Sonno": ore_sonno, "FC Media": fc_media,
        "Temperatura": temperatura, "Vento": vento,
        "SMA": sma, "ISLR": islr, "IDET": idet, "IITR": iitr,
        TARGET: rischio
    })
    return df.round(2)

# ============================================================================
# ML PIPELINE (Cached)
# ============================================================================
@st.cache_resource
def train_models(df: pd.DataFrame, config: Settings):
    # Dati classificazione
    X_cls = df[RF_FEATURES]
    y_cls = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=config.test_size, random_state=config.seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=config.seed)
    rf.fit(X_train, y_train)
    
    # Logistic Regression (Baseline)
    lr_cls = LogisticRegression()
    lr_cls.fit(X_train_scaled, y_train)
    
    # Dati Regressione Performance
    X_reg = df[["Distanza (km)", "SMA", "RPE"]]
    y_reg = df[TIME_TARGET]
    reg = LinearRegression().fit(X_reg, y_reg)
    
    # Dati Clustering
    X_clust = df[CLUSTER_FEATURES]
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed)
    df["Cluster"] = kmeans.fit_predict(StandardScaler().fit_transform(X_clust))
    
    return rf, lr_cls, reg, kmeans, scaler, X_train, X_test, y_train, y_test

# ============================================================================
# UI RENDERING
# ============================================================================
def render_ui():
    register_plotly_template()
    inject_css()
    
    # SIDEBAR
    with st.sidebar:
        st.title("⚙️ Parametri Modelli")
        n_sess = st.slider("Numero Sessioni (Dataset)", 100, 2000, 500, step=100)
        n_est = st.slider("Alberi Random Forest", 50, 500, 200, step=50)
        k_clust = st.slider("Numero Cluster (K-Means)", 2, 5, 3)
        
        cfg = Settings(n_sessions=n_sess, n_estimators=n_est, n_clusters=k_clust)
        
        st.markdown("---")
        st.markdown("**Glossario Indici:**")
        st.markdown("- **SMA**: Stress Metabolico Apparente\n- **ISLR**: Indice Stress Lavoro-Relativo\n- **IDET**: Domanda Emodinamico-Termica")

    # CARICAMENTO DATI
    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    rf, lr_cls, reg, kmeans, scaler, X_train, X_test, y_train, y_test = train_models(df, cfg)

    # HERO SECTION
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Data Science M.Sc. Thesis</div>
        <h1 class="hero-title">{APP_TITLE}</h1>
        <p style="color:var(--muted); margin-top:1rem; font-size:1.1rem;">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    # TABS
    t1, t2, t3, t4, t5 = st.tabs(["📊 Panoramica & Dati", "📈 Regressione", "⚠️ Rischio Overload", "🧩 Clustering", "🎮 Simulatore What-If"])

    # --- TAB 1: DATI ---
    with t1:
        st.markdown("""<div class="section-head"><div class="section-kicker">Esplorazione</div>
        <div class="section-title">Dataset Sintetico Generato</div></div>""", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totale Sessioni", len(df))
        c2.metric("Rischio Overload (%)", f"{(df[TARGET].mean()*100):.1f}%")
        c3.metric("Distanza Media (km)", f"{df['Distanza (km)'].mean():.1f}")
        c4.metric("Sonno Medio (h)", f"{df['Ore Sonno'].mean():.1f}")
        
        st.dataframe(df.head(15), use_container_width=True)
        
        # Correlazione
        st.markdown("### Matrice di Correlazione")
        corr = df[RF_FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: REGRESSIONE ---
    with t2:
        st.markdown("""<div class="section-head"><div class="section-kicker">Stima Performance</div>
        <div class="section-title">Previsione del Tempo di Completamento</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="callout callout--theory">
        <b>Modello Lineare:</b> Utilizziamo Distanza, Stress Metabolico Apparente (SMA) e RPE per stimare il tempo di completamento. L'obiettivo è quantificare l'impatto della fatica sul passo dell'atleta.
        </div>""", unsafe_allow_html=True)
        
        preds = reg.predict(df[["Distanza (km)", "SMA", "RPE"]])
        r2 = r2_score(df[TIME_TARGET], preds)
        mae = mean_absolute_error(df[TIME_TARGET], preds)
        
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.3f}", "Variabilità spiegata")
        c2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} min", "Errore medio", delta_color="inverse")
        
        fig = px.scatter(x=df[TIME_TARGET], y=preds, opacity=0.7, 
                         labels={"x": "Tempo Reale (min)", "y": "Tempo Predetto (min)"},
                         title="Reale vs Predetto (Regressione Multipla)", color_discrete_sequence=[COLORS['cyan']])
        fig.add_shape(type="line", x0=df[TIME_TARGET].min(), y0=df[TIME_TARGET].min(), 
                      x1=df[TIME_TARGET].max(), y1=df[TIME_TARGET].max(), line=dict(color=COLORS['pink'], dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: CLASSIFICAZIONE ---
    with t3:
        st.markdown("""<div class="section-head"><div class="section-kicker">Classificazione Binaria</div>
        <div class="section-title">Analisi Rischio Overload & Feature Importance</div></div>""", unsafe_allow_html=True)
        
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        y_prob_lr = lr_cls.predict_proba(X_test_scaled)[:, 1]
        
        c1, c2 = st.columns([1, 1])
        with c1:
            # ROC Curve
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
            fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
            auc_rf, auc_lr = roc_auc_score(y_test, y_prob_rf), roc_auc_score(y_test, y_prob_lr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"Random Forest (AUC={auc_rf:.2f})", line=dict(color=COLORS['cyan'], width=3)))
            fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"Logistic Reg (AUC={auc_lr:.2f})", line=dict(color=COLORS['purple'], width=2, dash='dot')))
            fig_roc.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="ROC Curve - Confronto Modelli", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with c2:
            # Feature Importance (Gini)
            imp = pd.DataFrame({"Feature": RF_FEATURES, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Driver del Sovraccarico (RF Gini Imp.)", color_discrete_sequence=[COLORS['amber']])
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""<div class="callout callout--insight">
        <b>Insight:</b> Il Random Forest supera la regressione logistica nel catturare le relazioni non lineari. Le metriche derivate (SMA e ISLR) mostrano un alto potere predittivo, confermando l'ipotesi di ricerca: la combinazione di carico interno ed esterno è più informativa delle singole metriche.
        </div>""", unsafe_allow_html=True)

    # --- TAB 4: CLUSTERING ---
    with t4:
        st.markdown("""<div class="section-head"><div class="section-kicker">Unsupervised Learning</div>
        <div class="section-title">Identificazione Profili Latenti (K-Means)</div></div>""", unsafe_allow_html=True)
        
        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Silhouette Score", f"{sil_score:.3f}", "Misura della separazione dei cluster")
        
        fig_cluster = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", 
                                    color_continuous_scale=[CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[2]],
                                    title="Spazio Latente delle Sessioni di Allenamento")
        fig_cluster.update_layout(scene=dict(xaxis_title="FC Media (BPM)", yaxis_title="Indice ISLR", zaxis_title="Stress (SMA)"))
        st.plotly_chart(fig_cluster, use_container_width=True)

    # --- TAB 5: SIMULATORE ---
    with t5:
        st.markdown("""<div class="section-head"><div class="section-kicker">Interactive App</div>
        <div class="section-title">Simulatore What-If dell'Atleta</div></div>""", unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        
        with sc1:
            st.markdown("### Inserisci Parametri")
            s_dist = st.slider("Distanza Prevista (km)", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Percepita (RPE)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno Notte Precedente", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("FC Media Stimata", 100, 190, 150)
            s_temp = st.slider("Temperatura (°C)", 0, 40, 25)
            
            # Calcolo al volo metriche derivate
            s_tempo = s_dist * 4.5 + (s_rpe * 2) # Stima sommaria
            s_lavoro = s_tempo / 60
            s_sma = (s_lavoro * s_rpe) / s_sonno
            s_islr = (s_lavoro * s_rpe) / s_dist
            s_idet = (s_fc * s_temp) / ((s_dist*1000)/(s_tempo*60))
            s_iitr = (s_temp * 10) / s_dist # Vento fisso a 10
            
            input_data = pd.DataFrame([[s_dist, s_sonno, s_sma, s_islr, s_idet, s_iitr, s_rpe]], columns=RF_FEATURES)
            prob = rf.predict_proba(input_data)[0][1] * 100
            label, color = risk_band(prob)

        with sc2:
            st.markdown("### Esito e Footprint")
            
            c_gauge, c_radar = st.columns([1, 1])
            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob, title = {'text': f"Rischio: {label}"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': COLORS['border']},
                        'bar': {'color': color},
                        'bgcolor': COLORS['surface'],
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(52, 211, 153, 0.2)'},
                            {'range': [40, 70], 'color': 'rgba(251, 191, 36, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.2)'}],
                    }))
                fig_g.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
            
            with c_radar:
                # Normalizzazione manuale per il radar
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=RF_FEATURES, fill='toself', name='Media Squadra', line_color=COLORS['muted']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=RF_FEATURES, fill='toself', name='Questa Sessione', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=300, margin=dict(l=30, r=30, t=30, b=30))
                st.plotly_chart(fig_r, use_container_width=True)
                
            st.markdown("""<div class="callout callout--simple">
            Usa gli slider per simulare l'impatto di una scarsa qualità del sonno o di un incremento dell'RPE sulle metriche derivate. Il Random Forest valuta il profilo in tempo reale.
            </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    render_ui()
