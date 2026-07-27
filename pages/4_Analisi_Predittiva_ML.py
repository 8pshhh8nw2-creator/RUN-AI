import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, silhouette_score

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Core Tesi Magistrale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DESIGN TOKENS
# ============================================================================
COLORS = {
    "bg": "#06101d",
    "bg2": "#091728",
    "surface": "#0f1b2d",
    "surface_2": "#13243b",
    "border": "#233651",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "blue": "#38bdf8",
    "cyan": "#22d3ee",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "purple": "#a855f7",
    "pink": "#ec4899"
}

RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR"]

# ============================================================================
# GLOBAL STYLE
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 12% 18%, rgba(56,189,248,0.10), transparent 18%),
            radial-gradient(circle at 88% 10%, rgba(168,85,247,0.10), transparent 20%),
            radial-gradient(circle at 50% 100%, rgba(34,211,238,0.06), transparent 25%),
            linear-gradient(180deg, {COLORS["bg"]} 0%, {COLORS["bg2"]} 100%);
        color: {COLORS["text"]};
    }}

    .block-container {{
        max-width: 1550px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(15,27,45,0.96) 0%, rgba(19,36,59,0.92) 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 26px;
        padding: 2rem 2rem 1.7rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.32);
    }}

    .hero::after {{
        content: "";
        position: absolute;
        top: -70px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, rgba(56,189,248,0.00) 70%);
        pointer-events: none;
    }}

    .hero-title {{
        margin: 0;
        font-size: 2.55rem;
        line-height: 1.05;
        letter-spacing: -0.04em;
        font-weight: 800;
        color: {COLORS["text"]};
    }}

    .hero-subtitle {{
        margin-top: 0.8rem;
        color: {COLORS["muted"]};
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 1050px;
    }}

    .ml-simple-box {{
        margin-top: 1rem;
        background: rgba(34,211,238,0.07);
        border: 1px solid rgba(34,211,238,0.18);
        border-left: 4px solid {COLORS["cyan"]};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #d9faff;
    }}

    .section-shell {{
        background: rgba(15, 27, 45, 0.84);
        border: 1px solid {COLORS["border"]};
        border-radius: 22px;
        padding: 1.2rem 1.2rem 0.9rem 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.16);
    }}

    .section-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.25rem;
    }}

    .section-subtitle {{
        color: {COLORS["muted"]};
        margin-bottom: 0.9rem;
        font-size: 0.98rem;
    }}

    .mini-title {{
        color: {COLORS["text"]};
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }}

    .theory-box {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.20);
        border-left: 4px solid {COLORS["amber"]};
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #fde68a;
    }}

    .simple-box {{
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.20);
        border-left: 4px solid {COLORS["blue"]};
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #dbeafe;
    }}

    .insight-box {{
        background: rgba(168, 85, 247, 0.10);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-left: 4px solid {COLORS["purple"]};
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #f3e8ff;
    }}

    .warroom-box {{
        background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(31,41,55,0.90));
        border: 1px solid rgba(56,189,248,0.20);
        border-radius: 24px;
        padding: 1.3rem;
        margin-top: 0.6rem;
        box-shadow: 0 16px 35px rgba(0,0,0,0.28);
    }}

    .small-note {{
        color: {COLORS["muted"]};
        font-size: 0.92rem;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(15,27,45,0.96), rgba(19,36,59,0.95));
        border: 1px solid {COLORS["border"]};
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]};
    }}

    div[data-testid="stMetricValue"] {{
        color: {COLORS["text"]};
        font-weight: 800;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION
# ============================================================================
@st.cache_data
def generate_data(n=320, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "Distanza (km)": rng.uniform(5, 30, n),
        "FC Media": rng.uniform(120, 180, n),
        "Velocità (km/h)": rng.uniform(9, 16, n),
        "Ore Sonno": rng.uniform(4, 9, n),
        "Stress Lavoro": rng.uniform(1, 10, n),
        "Ore Lavoro": rng.uniform(0, 10, n),
        "RPE": rng.uniform(1, 10, n),
        "Temp (°C)": rng.uniform(10, 35, n),
        "Vento (km/h)": rng.uniform(0, 25, n)
    })

    df["Tempo (min)"] = (df["Distanza (km)"] / df["Velocità (km/h)"]) * 60 + rng.normal(0, 4.5, n)

    df["SMA"] = (df["Stress Lavoro"] * df["RPE"]) / df["Ore Sonno"]
    df["ISLR"] = (df["Ore Lavoro"] * df["Stress Lavoro"]) / df["Distanza (km)"]
    df["IITR"] = (df["Temp (°C)"] * df["Vento (km/h)"]) / df["Distanza (km)"]
    df["IDET"] = (df["FC Media"] * df["Temp (°C)"]) / df["Velocità (km/h)"]

    risk_score = (df["ISLR"] * 0.5) + (df["IDET"] * 0.02) - (df["Ore Sonno"] * 0.6)
    df["Rischio Overload"] = (risk_score > np.quantile(risk_score, 0.70)).astype(int)

    return df

# ============================================================================
# TRAINING
# ============================================================================
@st.cache_resource
def train_models(df):
    out = {}

    # Linear Regression
    X_lr = df[["Distanza (km)"]]
    y_lr = df["Tempo (min)"]
    lr = LinearRegression().fit(X_lr, y_lr)
    df["Tempo_Predetto"] = lr.predict(X_lr)
    df["Residuo"] = df["Tempo (min)"] - df["Tempo_Predetto"]
    out["lr"] = lr
    out["lr_r2"] = r2_score(y_lr, df["Tempo_Predetto"])

    # Logistic Regression
    X_log = df[["ISLR"]]
    y_log = df["Rischio Overload"]
    log = LogisticRegression().fit(X_log, y_log)
    df["Prob_Overload"] = log.predict_proba(X_log)[:, 1]
    out["log"] = log
    out["log_acc"] = accuracy_score(y_log, log.predict(X_log))
    out["log_auc"] = roc_auc_score(y_log, df["Prob_Overload"])
    out["x_range"] = np.linspace(df["ISLR"].min(), df["ISLR"].max(), 300).reshape(-1, 1)
    out["y_prob_curve"] = log.predict_proba(out["x_range"])[:, 1]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    ).fit(df[RF_FEATURES], df["Rischio Overload"])
    rf_pred = rf.predict(df[RF_FEATURES])
    rf_proba = rf.predict_proba(df[RF_FEATURES])[:, 1]
    out["rf"] = rf
    out["rf_acc"] = accuracy_score(df["Rischio Overload"], rf_pred)
    out["rf_auc"] = roc_auc_score(df["Rischio Overload"], rf_proba)
    out["imp_df"] = pd.DataFrame({
        "Feature": RF_FEATURES,
        "Importanza": rf.feature_importances_
    }).sort_values("Importanza", ascending=True).reset_index(drop=True)

    # KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[["FC Media", "ISLR"]])
    df["Cluster_ID"] = km.labels_
    out["km"] = km
    out["sil"] = silhouette_score(df[["FC Media", "ISLR"]], km.labels_)

    centroids = pd.DataFrame(km.cluster_centers_, columns=["FC Media", "ISLR"])
    order = centroids["ISLR"].sort_values().index.tolist()
    labels = ["Rigenerativo", "Qualità / Misto", "Elevato Stress"]
    cluster_map = {cluster_id: labels[i] for i, cluster_id in enumerate(order)}
    df["Profilo_Corsa"] = df["Cluster_ID"].map(cluster_map)
    out["cluster_map"] = cluster_map

    out["df"] = df
    return out

# ============================================================================
# HELPERS
# ============================================================================
def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"]),
        margin=dict(t=55, l=20, r=20, b=20),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right")
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig

def open_section(title, subtitle):
    st.markdown(f"""
    <div class="section-shell">
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)

def close_section():
    st.markdown("</div>", unsafe_allow_html=True)

def traffic_color(prob):
    if prob < 40:
        return COLORS["green"], "Basso"
    elif prob < 70:
        return COLORS["amber"], "Moderato"
    return COLORS["red"], "Alto"

# ============================================================================
# LOAD
# ============================================================================
df = generate_data()
res = train_models(df)
data = res["df"]

# ============================================================================
# HERO
# ============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">Advanced Machine Learning Suite</div>
    <div class="hero-subtitle">
        Framework interattivo per la stima della performance, la classificazione del rischio di overload,
        l’analisi dei driver del sovraccarico e la scoperta di profili latenti di allenamento.
        Questa dashboard non mostra solo risultati: accompagna il lettore dentro la logica della tesi.
    </div>
    <div class="ml-simple-box">
        <b>Cos’è il Machine Learning, in modo semplice?</b><br>
        È un insieme di algoritmi che imparano dai dati. Invece di seguire solo regole fisse,
        osservano esempi reali, trovano schemi ricorrenti e li usano per fare previsioni,
        classificare situazioni e supportare decisioni future.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# KPI HEADER
# ============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Regressione Lineare", f"R² {res['lr_r2']:.2f}", "Baseline performance")
k2.metric("Regressione Logistica", f"AUC {res['log_auc']:.2f}", f"ACC {res['log_acc']*100:.0f}%")
k3.metric("Random Forest", f"AUC {res['rf_auc']:.2f}", f"ACC {res['rf_acc']*100:.0f}%")
k4.metric("K-Means", f"Silhouette {res['sil']:.2f}", "3 profili")

# ============================================================================
# MODEL SELECTOR
# ============================================================================
model_view = st.segmented_control(
    "Esplora il cuore analitico della tesi",
    options=[
        "📈 Regressione Lineare",
        "🎯 Regressione Logistica",
        "🌳 Random Forest",
        "🔍 K-Means",
        "🚀 Simulatore Finale"
    ],
    default="📈 Regressione Lineare"
)

# ============================================================================
# LINEAR REGRESSION
# ============================================================================
if model_view == "📈 Regressione Lineare":
    open_section(
        "📈 Regressione Lineare | Dal chilometraggio al tempo atteso",
        "Qui il modello impara una relazione semplice: all’aumentare della distanza, quanto cresce il tempo necessario per completare la sessione."
    )

    a, b, c = st.columns(3)
    a.metric("R²", f"{res['lr_r2']:.2f}")
    b.metric("Tempo medio", f"{data['Tempo (min)'].mean():.1f} min")
    c.metric("Errore assoluto medio", f"{data['Residuo'].abs().mean():.1f} min")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Questo algoritmo serve a stimare “quanto tempo potrei impiegare”.
        È il punto di partenza ideale, perché traduce una relazione intuitiva in un modello misurabile.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
        <b>Spiegazione tecnica.</b> La regressione lineare cerca la retta che minimizza gli errori complessivi
        tra valori osservati e valori stimati. In questo modo costruisce una baseline quantitativa,
        utile per confrontare successivamente modelli più complessi.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["Distanza (km)"], y=data["Tempo (min)"],
            mode="markers", name="Osservato",
            marker=dict(color=COLORS["blue"], size=8, opacity=0.65)
        ))
        fig.add_trace(go.Scatter(
            x=data["Distanza (km)"], y=data["Tempo_Predetto"],
            mode="lines", name="Trend lineare",
            line=dict(color=COLORS["pink"], width=3)
        ))
        fig.update_layout(title="Relazione tra distanza e tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        fig = px.histogram(
            data, x="Residuo", nbins=24,
            title="Distribuzione dei residui",
            color_discrete_sequence=[COLORS["purple"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            data, x="Tempo_Predetto", y="Tempo (min)",
            trendline="ols",
            title="Predetto vs osservato",
            color_discrete_sequence=[COLORS["cyan"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        fig = px.scatter(
            data, x="Distanza (km)", y="Residuo",
            title="Residui rispetto alla distanza",
            color_discrete_sequence=[COLORS["amber"]]
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["red"])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Interpretazione da tesi.</b> Questa sezione è importante perché introduce il lettore con un modello leggibile.
        Mostra che il machine learning non parte subito da “scatole nere”, ma può cominciare da relazioni semplici e trasparenti.
    </div>
    """, unsafe_allow_html=True)

    close_section()

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================
elif model_view == "🎯 Regressione Logistica":
    open_section(
        "🎯 Regressione Logistica | Quando una sessione entra in area critica",
        "Qui non prevediamo un tempo, ma la probabilità che una sessione sia classificata come potenzialmente a rischio overload."
    )

    a, b, c = st.columns(3)
    a.metric("AUC", f"{res['log_auc']:.2f}")
    b.metric("Accuracy", f"{res['log_acc']*100:.0f}%")
    c.metric("Sessioni a rischio", f"{data['Rischio Overload'].mean()*100:.0f}%")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Questo modello risponde a una domanda chiave:
        “quanto è probabile che questa seduta mi porti verso il sovraccarico?”.
        Non dà solo un sì o un no, ma una probabilità.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
        <b>Spiegazione tecnica.</b> La regressione logistica usa una funzione sigmoide per trasformare l’input
        in una probabilità tra 0 e 1. Questo è molto utile quando l’obiettivo è classificare situazioni
        in due stati: normale vs rischio.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["ISLR"], y=data["Rischio Overload"],
            mode="markers", name="Osservazioni",
            marker=dict(color="#94a3b8", size=8, opacity=0.45)
        ))
        fig.add_trace(go.Scatter(
            x=res["x_range"].flatten(), y=res["y_prob_curve"],
            mode="lines", name="Curva sigmoide",
            line=dict(color=COLORS["amber"], width=3)
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["red"])
        fig.update_layout(title="Transizione verso il rischio", xaxis_title="ISLR", yaxis_title="Probabilità")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        fig = px.box(
            data, x="Rischio Overload", y="Prob_Overload",
            color="Rischio Overload", title="Separabilità delle classi",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        prob_bins = pd.cut(data["Prob_Overload"], bins=5, duplicates="drop")
        calib = data.groupby(prob_bins, observed=False)["Rischio Overload"].mean().reset_index()
        calib["Fascia"] = calib["Prob_Overload"].astype(str)
        fig = px.bar(
            calib, x="Fascia", y="Rischio Overload",
            title="Rischio osservato per fascia di probabilità",
            color_discrete_sequence=[COLORS["purple"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        fig = px.histogram(
            data, x="Prob_Overload", color="Rischio Overload",
            nbins=24, title="Distribuzione delle probabilità stimate",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Interpretazione da tesi.</b> Qui il machine learning diventa decisionale:
        non si limita a descrivere il passato, ma aiuta a identificare quando una sessione
        merita cautela prima ancora che il problema si manifesti.
    </div>
    """, unsafe_allow_html=True)

    close_section()

# ============================================================================
# RANDOM FOREST
# ============================================================================
elif model_view == "🌳 Random Forest":
    open_section(
        "🌳 Random Forest | Perché nasce il rischio",
        "Qui il modello non usa un solo indicatore, ma combina più variabili per capire quali fattori pesano davvero nella comparsa del sovraccarico."
    )

    a, b, c = st.columns(3)
    a.metric("AUC", f"{res['rf_auc']:.2f}")
    b.metric("Accuracy", f"{res['rf_acc']*100:.0f}%")
    c.metric("Feature usate", f"{len(RF_FEATURES)}")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Se la logistica risponde “quanto rischio c’è?”,
        il Random Forest aiuta a capire “da dove nasce quel rischio”.
        Per questo è uno dei modelli più forti dell’intera suite.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
        <b>Spiegazione tecnica.</b> Il modello costruisce molti alberi decisionali e poi combina le loro decisioni.
        Questo approccio ensemble migliora robustezza, coglie relazioni non lineari
        e permette di calcolare l’importanza relativa delle feature.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            res["imp_df"], x="Importanza", y="Feature",
            orientation="h", title="Feature importance globale",
            color="Importanza",
            color_continuous_scale=["#164e63", "#0891b2", "#67e8f9"]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        top2 = res["imp_df"].tail(2)["Feature"].tolist()
        fig = px.scatter(
            data, x=top2[0], y=top2[1],
            color="Rischio Overload",
            title=f"Interazione tra {top2[0]} e {top2[1]}",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        rf_scores = res["rf"].predict_proba(data[RF_FEATURES])[:, 1]
        fig = px.histogram(
            x=rf_scores, nbins=24,
            title="Distribuzione degli score Random Forest",
            color_discrete_sequence=[COLORS["green"]]
        )
        fig.update_xaxes(title="Probabilità stimata")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        feature_means = data.groupby("Rischio Overload")[RF_FEATURES].mean().T.reset_index()
        feature_means.columns = ["Feature", "Sicuro", "Rischio"]
        melt_df = feature_means.melt(id_vars="Feature", var_name="Classe", value_name="Valore")
        fig = px.bar(
            melt_df, x="Feature", y="Valore", color="Classe", barmode="group",
            title="Differenze medie tra sessioni sicure e a rischio",
            color_discrete_map={"Sicuro": COLORS["blue"], "Rischio": COLORS["red"]}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Interpretazione da tesi.</b> Questa è la sezione più esplicativa del progetto:
        mostra che il sovraccarico non è il risultato di un singolo fattore, ma di una combinazione di carico,
        recupero, stress e condizioni contestuali.
    </div>
    """, unsafe_allow_html=True)

    close_section()

# ============================================================================
# K-MEANS
# ============================================================================
elif model_view == "🔍 K-Means":
    open_section(
        "🔍 K-Means | I profili nascosti delle sessioni",
        "Qui il modello non riceve etichette già note: osserva i dati e raggruppa automaticamente sessioni simili tra loro."
    )

    a, b, c = st.columns(3)
    a.metric("Silhouette", f"{res['sil']:.2f}")
    b.metric("Numero cluster", "3")
    c.metric("Sessioni", f"{len(data)}")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Questo algoritmo cerca famiglie di allenamento ricorrenti.
        In pratica prova a capire se nei dati esistono tipi di sessione che si somigliano naturalmente.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
        <b>Spiegazione tecnica.</b> K
