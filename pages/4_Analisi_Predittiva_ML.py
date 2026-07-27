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
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Core Tesi Magistrale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DESIGN SYSTEM
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
        background: linear-gradient(135deg, rgba(15,27,45,0.96) 0%, rgba(19,36,59,0.92) 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 26px;
        padding: 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.32);
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
# DATA
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
# MODELS
# ============================================================================
@st.cache_resource
def train_models(df):
    out = {}
    work_df = df.copy()

    # Linear Regression
    X_lr = work_df[["Distanza (km)"]]
    y_lr = work_df["Tempo (min)"]
    lr = LinearRegression().fit(X_lr, y_lr)
    work_df["Tempo_Predetto"] = lr.predict(X_lr)
    work_df["Residuo"] = work_df["Tempo (min)"] - work_df["Tempo_Predetto"]
    out["lr"] = lr
    out["lr_r2"] = r2_score(y_lr, work_df["Tempo_Predetto"])

    # Logistic Regression
    X_log = work_df[["ISLR"]]
    y_log = work_df["Rischio Overload"]
    log = LogisticRegression().fit(X_log, y_log)
    work_df["Prob_Overload"] = log.predict_proba(X_log)[:, 1]
    out["log"] = log
    out["log_acc"] = accuracy_score(y_log, log.predict(X_log))
    out["log_auc"] = roc_auc_score(y_log, work_df["Prob_Overload"])
    out["x_range"] = np.linspace(work_df["ISLR"].min(), work_df["ISLR"].max(), 300).reshape(-1, 1)
    out["y_prob_curve"] = log.predict_proba(out["x_range"])[:, 1]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    ).fit(work_df[RF_FEATURES], work_df["Rischio Overload"])
    rf_pred = rf.predict(work_df[RF_FEATURES])
    rf_proba = rf.predict_proba(work_df[RF_FEATURES])[:, 1]
    out["rf"] = rf
    out["rf_acc"] = accuracy_score(work_df["Rischio Overload"], rf_pred)
    out["rf_auc"] = roc_auc_score(work_df["Rischio Overload"], rf_proba)
    out["imp_df"] = (
        pd.DataFrame({
            "Feature": RF_FEATURES,
            "Importanza": rf.feature_importances_
        })
        .sort_values("Importanza", ascending=True)
        .reset_index(drop=True)
    )

    # KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(work_df[["FC Media", "ISLR"]])
    work_df["Cluster_ID"] = km.labels_
    out["km"] = km
    out["sil"] = silhouette_score(work_df[["FC Media", "ISLR"]], km.labels_)

    centroids = pd.DataFrame(km.cluster_centers_, columns=["FC Media", "ISLR"])
    order = centroids["ISLR"].sort_values().index.tolist()
    labels = ["Rigenerativo", "Qualità / Misto", "Elevato Stress"]
    cluster_map = {cluster_id: labels[i] for i, cluster_id in enumerate(order)}
    work_df["Profilo_Corsa"] = work_df["Cluster_ID"].map(cluster_map)
    out["cluster_map"] = cluster_map
    out["df"] = work_df

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
        Questa dashboard accompagna il lettore dentro la logica della tesi, trasformando i dati in decisioni.
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
# KPI SUMMARY
# ============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Regressione Lineare", f"R² {res['lr_r2']:.2f}", "Baseline")
k2.metric("Regressione Logistica", f"AUC {res['log_auc']:.2f}", f"ACC {res['log_acc']*100:.0f}%")
k3.metric("Random Forest", f"AUC {res['rf_auc']:.2f}", f"ACC {res['rf_acc']*100:.0f}%")
k4.metric("K-Means", f"Silhouette {res['sil']:.2f}", "3 profili")

# ============================================================================
# MODEL NAVIGATION
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
        "Il modello impara una relazione semplice: all’aumentare della distanza, cresce il tempo necessario per completare la sessione."
    )

    a, b, c = st.columns(3)
    a.metric("R²", f"{res['lr_r2']:.2f}")
    b.metric("Tempo medio", f"{data['Tempo (min)'].mean():.1f} min")
    c.metric("Errore assoluto medio", f"{data['Residuo'].abs().mean():.1f} min")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Questo algoritmo serve a stimare quanto tempo potrei impiegare.
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
            trendline="ols", title="Predetto vs osservato",
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
        <b>Interpretazione da tesi.</b> Questa sezione introduce il lettore con un modello leggibile.
        Mostra che il machine learning non parte subito da scatole nere, ma può cominciare da relazioni semplici e trasparenti.
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
        quanto è probabile che questa seduta mi porti verso il sovraccarico?
        Non dà solo un sì o un no, ma una probabilità.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
        <b>Spiegazione tecnica.</b> La regressione logistica usa una funzione sigmoide per trasformare l’input
        in una probabilità tra 0 e 1. Questo è utile quando l’obiettivo è classificare situazioni
        in due stati: normale contro rischio.
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
            color="Rischio Overload",
            title="Separabilità delle classi",
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
            nbins=24,
            title="Distribuzione delle probabilità stimate",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Interpretazione da tesi.</b> Qui il machine learning diventa decisionale:
        non si limita a descrivere il passato, ma aiuta a identificare quando una sessione merita cautela
        prima ancora che il problema si manifesti.
    </div>
    """, unsafe_allow_html=True)

    close_section()

# ============================================================================
# RANDOM FOREST
# ============================================================================
elif model_view == "🌳 Random Forest":
    open_section(
        "🌳 Random Forest | Perché nasce il rischio",
        "Qui il modello combina più variabili per capire quali fattori pesano davvero nella comparsa del sovraccarico."
    )

    a, b, c = st.columns(3)
    a.metric("AUC", f"{res['rf_auc']:.2f}")
    b.metric("Accuracy", f"{res['rf_acc']*100:.0f}%")
    c.metric("Feature usate", f"{len(RF_FEATURES)}")

    st.markdown("""
    <div class="simple-box">
        <b>Spiegazione semplice.</b> Se la logistica risponde a quanto rischio c’è,
        il Random Forest aiuta a capire da dove nasce quel rischio.
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
        <b>Spiegazione tecnica.</b> K-Means minimizza la distanza dei punti dai centroidi del proprio gruppo.
        Qui viene usato per scoprire profili latenti di sessione in base a FC media e ISLR.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            data, x="ISLR", y="FC Media",
            color="Profilo_Corsa",
            title="Segmentazione dei profili di allenamento",
            color_discrete_sequence=[COLORS["green"], COLORS["amber"], COLORS["red"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        cluster_means = data.groupby("Profilo_Corsa")[["Ore Sonno", "RPE", "Tempo (min)"]].mean().reset_index()
        fig = px.bar(
            cluster_means,
            x="Profilo_Corsa",
            y=["Ore Sonno", "RPE", "Tempo (min)"],
            barmode="group",
            title="Profilo medio dei cluster"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        cluster_counts = data["Profilo_Corsa"].value_counts().reset_index()
        cluster_counts.columns = ["Profilo", "Conteggio"]

        fig = px.pie(
            cluster_counts,
            names="Profilo",
            values="Conteggio",
            title="Peso percentuale dei cluster",
            color="Profilo",
            color_discrete_sequence=[COLORS["green"], COLORS["amber"], COLORS["red"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        fig = px.box(
            data,
            x="Profilo_Corsa",
            y="ISLR",
            color="Profilo_Corsa",
            title="Distribuzione ISLR per cluster",
            color_discrete_sequence=[COLORS["green"], COLORS["amber"], COLORS["red"]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Interpretazione da tesi.</b> K-Means aggiunge una lettura esplorativa:
        non ti dice solo cosa accadrà, ma mostra come i dati si organizzano spontaneamente in archetipi di allenamento.
    </div>
    """, unsafe_allow_html=True)

    close_section()

# ============================================================================
# FINAL SIMULATOR
# ============================================================================
else:
    st.markdown("""
    <div class="warroom-box">
        <div class="section-title">🚀 Simulatore Finale | Decision Room della Tesi</div>
        <div class="section-subtitle">
            Qui convergono tutti i modelli. Imposti una sessione ipotetica e il sistema ti restituisce
            tempo previsto, probabilità di overload, lettura rapida logistica e profilo K-Means più vicino.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div class="simple-box">
            <b>Come funziona.</b> Inserisci i parametri della sessione. Quando premi il pulsante,
            il simulatore attiva insieme i diversi modelli della tesi e li trasforma in una valutazione unica.
        </div>
        """, unsafe_allow_html=True)

        with st.form("simulator_form"):
            sim_dist = st.slider("Distanza pianificata (km)", 5.0, 30.0, 10.0, 0.5)
            sim_sleep = st.slider("Ore di sonno", 3.0, 10.0, 7.5, 0.5)
            sim_stress = st.slider("Stress lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
            sim_work = st.slider("Ore lavoro", 0.0, 12.0, 8.0, 0.5)
            sim_rpe = st.slider("RPE previsto", 1.0, 10.0, 6.0, 1.0)

            submitted = st.form_submit_button("Simula la sessione")

    with right:
        if submitted:
            sim_fc = 145.0
            sim_temp = 25.0
            sim_vel = 12.0
            sim_vento = 5.0

            sim_sma = (sim_stress * sim_rpe) / sim_sleep
            sim_islr = (sim_work * sim_stress) / sim_dist
            sim_idet = (sim_fc * sim_temp) / sim_vel
            sim_iitr = (sim_temp * sim_vento) / sim_dist

            input_df = pd.DataFrame([{
                "Distanza (km)": sim_dist,
                "Ore Sonno": sim_sleep,
                "SMA": sim_sma,
                "ISLR": sim_islr,
                "IDET": sim_idet,
                "IITR": sim_iitr
            }])

            tempo_pred = float(
                res["lr"].predict(pd.DataFrame({"Distanza (km)": [sim_dist]}))[0]
            )
            prob_rf = float(
                res["rf"].predict_proba(input_df)[0, 1] * 100
            )
            prob_log = float(
                res["log"].predict_proba(pd.DataFrame({"ISLR": [sim_islr]}))[0, 1] * 100
            )

            sim_point = np.array([[sim_fc, sim_islr]])
            dists = np.linalg.norm(res["km"].cluster_centers_ - sim_point, axis=1)
            nearest_cluster = int(np.argmin(dists))
            nearest_profile = res["cluster_map"][nearest_cluster]

            color, level = traffic_color(prob_rf)

            st.markdown(f"""
            <div style="
                background: linear-gradient(180deg, rgba(15,27,45,0.98), rgba(19,36,59,0.96));
                border: 1px solid {COLORS["border"]};
                border-radius: 22px;
                padding: 1.35rem;
                text-align: center;
                box-shadow: 0 16px 35px rgba(0,0,0,0.28);
            ">
                <div style="color: {COLORS['muted']}; font-size: 0.82rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700;">
                    Esito finale simulazione
                </div>
                <div style="font-size: 4rem; font-weight: 900; line-height: 1; margin-top: 0.35rem; color: {color};">
                    {prob_rf:.1f}%
                </div>
                <div style="color: {COLORS['text']}; font-size: 1.05rem; margin-top: 0.35rem; font-weight: 700;">
                    Rischio overload: {level}
                </div>
                <div style="color: {COLORS['muted']}; margin-top: 0.45rem;">
                    Tempo stimato: {tempo_pred:.1f} min · Prob. logistica: {prob_log:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("SMA", f"{sim_sma:.2f}")
            m2.metric("ISLR", f"{sim_islr:.2f}")
            m3.metric("Profilo vicino", nearest_profile)

            st.markdown(f"""
            <div class="insight-box">
                <b>Lettura finale.</b> La sessione simulata assomiglia di più al profilo <b>{nearest_profile}</b>.
                Il Random Forest stima un rischio del <b>{prob_rf:.1f}%</b>, mentre la regressione logistica
                fornisce una lettura rapida di <b>{prob_log:.1f}%</b>. Il tempo previsto dalla regressione lineare
                è <b>{tempo_pred:.1f} minuti</b>.
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=prob_rf,
                title={"text": "Rischio overload"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(34,197,94,0.25)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.25)"}
                    ]
                }
            ))
            fig.update_layout(
                height=320,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"])
            )
            st.plotly_chart(fig, use_container_width=True)

            compare_df = pd.DataFrame({
                "Indicatore": ["Tempo previsto (min)", "Probabilità Logistica", "Probabilità Random Forest"],
                "Valore": [tempo_pred, prob_log, prob_rf]
            })

            fig = px.bar(
                compare_df,
                x="Indicatore",
                y="Valore",
                color="Indicatore",
                title="Lettura sintetica della simulazione",
                color_discrete_sequence=[COLORS["cyan"], COLORS["amber"], COLORS["pink"]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        else:
            st.markdown("""
            <div class="warroom-box">
                <div class="simple-box">
                    <b>Simulatore finale pronto.</b> Premi “Simula la sessione” per ottenere
                    il verdetto completo del cuore predittivo della tesi.
                </div>
            </div>
            """, unsafe_allow_html=True)
