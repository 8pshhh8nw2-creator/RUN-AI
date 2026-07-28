"""
Advanced Machine Learning Suite - Dashboard interattiva per tesi magistrale.
File unico. Avvio: streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE PAGINA
# ============================================================================
st.set_page_config(
    page_title="Sport ML Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# COSTANTI E COLORI
# ============================================================================
APP_TITLE = "Advanced Machine Learning Suite"
APP_SUBTITLE = "Dashboard Analitica Interattiva per la valutazione del rischio e della performance."

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
# CSS & THEME PLOTLY
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
    .callout {{ border-radius: 12px; padding: 1rem; margin: 1rem 0; font-size: 0.95rem; line-height: 1.6; border: 1px solid var(--border); }}
    .callout--theory {{ background: rgba(251,191,36,0.07); border-left: 4px solid var(--amber); color: #fef3c7; }}
    .callout--insight {{ background: rgba(167,139,250,0.09); border-left: 4px solid var(--purple); color: #ede9fe; }}
    .callout--simple {{ background: rgba(56,189,248,0.08); border-left: 4px solid var(--blue); color: #dbeafe; }}
    .section-head {{
        padding: 1rem 1.5rem; margin: 2rem 0 1rem 0;
        background: rgba(15,27,45,0.7); border: 1px solid var(--border); border-radius: 12px;
    }}
    .section-kicker {{ font-size: 0.75rem; font-weight: 700; color: var(--cyan); letter-spacing: 0.1em; text-transform: uppercase; }}
    .section-title {{ font-size: 1.4rem; font-weight: 800; margin:0; padding-top: 0.2rem; }}
    div[data-testid="stMetricValue"] {{ color: {COLORS['cyan']} !important; font-weight: 800; }}
    
    /* Styling per il selettore orizzontale */
    div.row-widget.stRadio > div {{ flex-direction: row; flex-wrap: wrap; gap: 10px; background: rgba(15,27,45,0.7); padding: 15px; border-radius: 12px; border: 1px solid var(--border); }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION
# ============================================================================
@st.cache_data
def generate_synthetic_data(n: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    distanza = np.random.uniform(5.0, 35.0, n)
    rpe = np.random.randint(2, 11, n)
    ore_sonno = np.random.normal(7.5, 1.2, n).clip(3, 10)
    temperatura = np.random.normal(20, 8, n)
    vento = np.random.normal(10, 5, n).clip(0, 40)
    
    tempo = distanza * np.random.normal(4.5, 0.3, n) + (rpe * 2) 
    velocita = (distanza * 1000) / (tempo * 60) 
    fc_media = 110 + (rpe * 6) - (ore_sonno * 2) + np.random.normal(0, 5, n)
    
    ore_lavoro = tempo / 60
    sma = (ore_lavoro * rpe) / ore_sonno
    islr = (ore_lavoro * rpe) / distanza
    idet = (fc_media * temperatura) / np.where(velocita>0, velocita, 1)
    iitr = (temperatura * vento) / distanza
    
    stress_score = (sma * 0.4) + (islr * 0.3) + (rpe * 0.3)
    prob_overload = 1 / (1 + np.exp(-(stress_score - np.median(stress_score))))
    rischio = (prob_overload > 0.65).astype(int) 
    
    df = pd.DataFrame({
        "Distanza (km)": distanza, "Tempo (min)": tempo, "Velocità (m/s)": velocita,
        "RPE": rpe, "Ore Sonno": ore_sonno, "FC Media": fc_media,
        "Temperatura": temperatura, "Vento": vento,
        "SMA": sma, "ISLR": islr, "IDET": idet, "IITR": iitr,
        TARGET: rischio
    })
    return df.round(2)

# ============================================================================
# ML PIPELINE
# ============================================================================
@st.cache_resource
def train_models(df: pd.DataFrame, config: Settings):
    X_cls = df[RF_FEATURES]
    y_cls = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=config.test_size, random_state=config.seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=config.seed)
    rf.fit(X_train, y_train)
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Linear Regression
    X_reg = df[["Distanza (km)", "SMA", "RPE"]]
    y_reg = df[TIME_TARGET]
    reg = LinearRegression().fit(X_reg, y_reg)
    
    # KMeans
    X_clust = df[CLUSTER_FEATURES]
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed)
    df["Cluster"] = kmeans.fit_predict(StandardScaler().fit_transform(X_clust))
    
    return rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

# ============================================================================
# UI RENDERING
# ============================================================================
def render_ui():
    register_plotly_template()
    inject_css()
    
    # SIDEBAR
    with st.sidebar:
        st.title("⚙️ Parametri Generali")
        n_sess = st.slider("Numero Sessioni (Dataset)", 100, 2000, 500, step=100)
        n_est = st.slider("Alberi Random Forest", 50, 500, 200, step=50)
        k_clust = st.slider("Numero Cluster (K-Means)", 2, 5, 3)
        cfg = Settings(n_sessions=n_sess, n_estimators=n_est, n_clusters=k_clust)
        
        st.markdown("---")
        st.markdown("**Glossario Metriche:**")
        st.markdown("- **SMA**: Stress Metabolico Apparente\n- **ISLR**: Indice Stress Lavoro-Relativo\n- **IDET**: Domanda Emodinamico-Termica")

    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = train_models(df, cfg)

    # HERO SECTION
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Sport Data Science M.Sc.</div>
        <h1 class="hero-title">{APP_TITLE}</h1>
        <p style="color:var(--muted); margin-top:1rem; font-size:1.1rem;">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    # MENU ORIZZONTALE
    sezioni = [
        "📊 Dataset & Esplorazione", 
        "📈 Regressione Lineare", 
        "⚖️ Regressione Logistica", 
        "🌲 Random Forest", 
        "🧩 K-Means Clustering", 
        "🎮 Simulatore What-If"
    ]
    
    scelta = st.radio("Seleziona il Modulo Analitico:", sezioni, horizontal=True, label_visibility="collapsed")

    # ==============================================================
    # 1. DATASET
    if scelta == sezioni[0]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Data Overview</div>
        <div class="section-title">Esplorazione del Dataset e Variabili</div></div>""", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totale Sessioni", len(df))
        c2.metric("Rischio Overload (%)", f"{(df[TARGET].mean()*100):.1f}%")
        c3.metric("Distanza Media (km)", f"{df['Distanza (km)'].mean():.1f}")
        c4.metric("Sonno Medio (h)", f"{df['Ore Sonno'].mean():.1f}")
        
        st.dataframe(df.head(12), use_container_width=True)
        
        st.markdown("### Matrice di Correlazione")
        corr = df[RF_FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    # ==============================================================
    # 2. REGRESSIONE LINEARE
    elif scelta == sezioni[1]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Stima della Performance</div>
        <div class="section-title">Regressione Lineare Multipla</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="callout callout--theory">
        <b>Spiegazione Teorica:</b> Questo modello stima il <b>Tempo di completamento</b> in funzione della distanza, dello stress metabolico (SMA) e della fatica percepita (RPE). L'obiettivo è analizzare in che misura i marker di stress prolungato degradano la performance attesa a parità di distanza.
        </div>""", unsafe_allow_html=True)
        
        preds = reg.predict(df[["Distanza (km)", "SMA", "RPE"]])
        
        c1, c2 = st.columns(2)
        c1.metric("R² Score (Varianza spiegata)", f"{r2_score(df[TIME_TARGET], preds):.3f}")
        c2.metric("MAE (Errore medio assoluto)", f"{mean_absolute_error(df[TIME_TARGET], preds):.2f} min", delta_color="inverse")
        
        fig = px.scatter(x=df[TIME_TARGET], y=preds, opacity=0.7, 
                         labels={"x": "Tempo Reale (min)", "y": "Tempo Predetto dal Modello (min)"},
                         title="Valori Reali vs Predizioni", color_discrete_sequence=[COLORS['cyan']])
        fig.add_shape(type="line", x0=df[TIME_TARGET].min(), y0=df[TIME_TARGET].min(), 
                      x1=df[TIME_TARGET].max(), y1=df[TIME_TARGET].max(), line=dict(color=COLORS['pink'], dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    # ==============================================================
    # 3. REGRESSIONE LOGISTICA
    elif scelta == sezioni[2]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Baseline Classificatore</div>
        <div class="section-title">Regressione Logistica (Rischio Overload)</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="callout callout--theory">
        <b>Spiegazione Teorica:</b> Modello interpretabile che stima la probabilità log-odds che un allenamento causi <i>sovraccarico</i>. Funge da <b>baseline lineare</b> per dimostrare se le relazioni tra metriche fisiche e rischio siano proporzionali o complesse (non lineari).
        </div>""", unsafe_allow_html=True)
        
        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (Test)", f"{accuracy_score(y_test, y_pred_lr):.2f}")
        c2.metric("F1-Score", f"{f1_score(y_test, y_pred_lr):.2f}")
        c3.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob_lr):.2f}")
        
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            cm = confusion_matrix(y_test, y_pred_lr)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Classe Predetta", y="Classe Reale"), x=['No Rischio', 'Overload'], y=['No Rischio', 'Overload'], title="Matrice di Confusione (Test Set)")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_fig2:
            coefs = pd.DataFrame({"Feature": RF_FEATURES, "Peso (Coefficiente)": lr.coef_[0]}).sort_values("Peso (Coefficiente)")
            fig_coef = px.bar(coefs, x="Peso (Coefficiente)", y="Feature", orientation="h", title="Impatto Logistico delle Variabili", color_discrete_sequence=[COLORS['purple']])
            st.plotly_chart(fig_coef, use_container_width=True)

    # ==============================================================
    # 4. RANDOM FOREST
    elif scelta == sezioni[3]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Modello Avanzato</div>
        <div class="section-title">Random Forest Classifier (Rischio Overload)</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="callout callout--insight">
        <b>Spiegazione Teorica & Insight:</b> Modello d'insieme ad alberi decisionali, capace di intercettare dinamiche <b>non lineari</b> (es. l'effetto combinato di tanto allenamento e poco sonno). Se le sue performance superano la Logistica, conferma che il sovraccarico è un fenomeno multifattoriale e interattivo.
        </div>""", unsafe_allow_html=True)
        
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (Test)", f"{accuracy_score(y_test, y_pred_rf):.2f}")
        c2.metric("F1-Score", f"{f1_score(y_test, y_pred_rf):.2f}")
        c3.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob_rf):.2f}")
        
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="Random Forest", line=dict(color=COLORS['cyan'], width=3)))
            fig_roc.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="Curva ROC", xaxis_title="Falsi Positivi", yaxis_title="Veri Positivi")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with col_fig2:
            imp = pd.DataFrame({"Feature": RF_FEATURES, "Importanza (Gini)": rf.feature_importances_}).sort_values("Importanza (Gini)")
            fig_imp = px.bar(imp, x="Importanza (Gini)", y="Feature", orientation="h", title="Driver di Overload (Feature Importance)", color_discrete_sequence=[COLORS['amber']])
            st.plotly_chart(fig_imp, use_container_width=True)

    # ==============================================================
    # 5. CLUSTERING
    elif scelta == sezioni[4]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Unsupervised Learning</div>
        <div class="section-title">K-Means Clustering: Profili Latenti</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="callout callout--theory">
        <b>Spiegazione Teorica:</b> Algoritmo non supervisionato che ricerca "pattern nascosti" senza conoscere a priori le etichette di rischio. Segmenta le sessioni raggruppandole per similarità basata su FC Media, Indice di Lavoro (ISLR) e Stress Metabolico (SMA).
        </div>""", unsafe_allow_html=True)
        
        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Silhouette Score", f"{sil_score:.3f}", "Rappresenta quanto sono ben definiti e separati i cluster")
        
        fig_cluster = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", 
                                    color_continuous_scale=[CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[2]],
                                    title="Visualizzazione 3D dello Spazio Latente degli Allenamenti")
        fig_cluster.update_layout(scene=dict(xaxis_title="FC Media", yaxis_title="Indice ISLR", zaxis_title="Stress SMA"), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_cluster, use_container_width=True)

    # ==============================================================
    # 6. SIMULATORE
    elif scelta == sezioni[5]:
        st.markdown("""<div class="section-head"><div class="section-kicker">Interactive App</div>
        <div class="section-title">Simulatore What-If dell'Atleta</div></div>""", unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.markdown("### Modifica i parametri")
            s_dist = st.slider("Distanza Prevista (km)", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Percepita (RPE)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno Notte Precedente", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("FC Media Stimata", 100, 190, 150)
            s_temp = st.slider("Temperatura (°C)", 0, 40, 25)
            
            s_tempo = s_dist * 4.5 + (s_rpe * 2) 
            s_lavoro = s_tempo / 60
            s_sma = (s_lavoro * s_rpe) / s_sonno
            s_islr = (s_lavoro * s_rpe) / s_dist
            s_idet = (s_fc * s_temp) / ((s_dist*1000)/(s_tempo*60))
            s_iitr = (s_temp * 10) / s_dist
            
            input_data = pd.DataFrame([[s_dist, s_sonno, s_sma, s_islr, s_idet, s_iitr, s_rpe]], columns=RF_FEATURES)
            prob = rf.predict_proba(input_data)[0][1] * 100
            label, color = risk_band(prob)

        with sc2:
            st.markdown("### Valutazione Predittiva (Random Forest)")
            
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
                fig_g.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
            
            with c_radar:
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=RF_FEATURES, fill='toself', name='Media Squadra', line_color=COLORS['muted']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=RF_FEATURES, fill='toself', name='Sessione Simulata', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=320, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_r, use_container_width=True)

if __name__ == "__main__":
    render_ui()
