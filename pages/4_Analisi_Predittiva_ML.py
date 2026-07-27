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
    page_title="Advanced ML Suite | Interactive Thesis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

COLORS = {
    "bg": "#07111f",
    "surface": "#0f1b2d",
    "surface_2": "#13233a",
    "border": "#22344f",
    "text": "#f8fafc",
    "muted": "#93a4b8",
    "blue": "#38bdf8",
    "cyan": "#22d3ee",
    "amber": "#f59e0b",
    "green": "#22c55e",
    "red": "#ef4444",
    "purple": "#a855f7",
    "pink": "#ec4899"
}

RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR"]

# ============================================================================
# STYLE
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 20%),
            radial-gradient(circle at left bottom, rgba(168,85,247,0.08), transparent 18%),
            linear-gradient(180deg, {COLORS["bg"]} 0%, #08101a 100%);
        color: {COLORS["text"]};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .hero {{
        background: linear-gradient(135deg, {COLORS["surface"]} 0%, {COLORS["surface_2"]} 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 24px;
        padding: 2rem 2rem 1.6rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.30);
    }}

    .hero-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin: 0;
        letter-spacing: -0.04em;
    }}

    .hero-subtitle {{
        margin-top: 0.7rem;
        color: {COLORS["muted"]};
        font-size: 1.02rem;
        max-width: 1000px;
        line-height: 1.6;
    }}

    .glass {{
        background: rgba(15, 27, 45, 0.82);
        border: 1px solid {COLORS["border"]};
        border-radius: 20px;
        padding: 1.15rem 1.15rem 0.8rem 1.15rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }}

    .model-title {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.2rem;
    }}

    .model-subtitle {{
        color: {COLORS["muted"]};
        margin-bottom: 1rem;
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

    .explain-box {{
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.20);
        border-left: 4px solid {COLORS["cyan"]};
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #cffafe;
    }}

    .insight-box {{
        background: rgba(168, 85, 247, 0.10);
        border: 1px solid rgba(168, 85, 247, 0.20);
        border-left: 4px solid {COLORS["purple"]};
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #f3e8ff;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, {COLORS["surface"]}, {COLORS["surface_2"]});
        border: 1px solid {COLORS["border"]};
        padding: 1rem;
        border-radius: 18px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]};
    }}

    div[data-testid="stMetricValue"] {{
        color: {COLORS["text"]};
        font-weight: 800;
    }}

    .small-note {{
        color: {COLORS["muted"]};
        font-size: 0.92rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA
# ============================================================================
@st.cache_data
def generate_data(n=300, seed=42):
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
        "Vento (km/h)": rng.uniform(0, 25, n),
    })

    df["Tempo (min)"] = (df["Distanza (km)"] / df["Velocità (km/h)"]) * 60 + rng.normal(0, 5, n)
    df["SMA"] = (df["Stress Lavoro"] * df["RPE"]) / df["Ore Sonno"]
    df["ISLR"] = (df["Ore Lavoro"] * df["Stress Lavoro"]) / df["Distanza (km)"]
    df["IITR"] = (df["Temp (°C)"] * df["Vento (km/h)"]) / df["Distanza (km)"]
    df["IDET"] = (df["FC Media"] * df["Temp (°C)"]) / df["Velocità (km/h)"]

    risk_score = (df["ISLR"] * 0.5) + (df["IDET"] * 0.02) - (df["Ore Sonno"] * 0.6)
    df["Rischio Overload"] = (risk_score > np.quantile(risk_score, 0.70)).astype(int)
    return df

@st.cache_resource
def train_all(df):
    out = {}

    X_lr = df[["Distanza (km)"]]
    y_lr = df["Tempo (min)"]
    lr = LinearRegression().fit(X_lr, y_lr)
    df["Tempo_Predetto"] = lr.predict(X_lr)
    df["Residuo"] = df["Tempo (min)"] - df["Tempo_Predetto"]
    out["lr_r2"] = r2_score(y_lr, df["Tempo_Predetto"])

    X_log = df[["ISLR"]]
    y_log = df["Rischio Overload"]
    log = LogisticRegression().fit(X_log, y_log)
    df["Prob_Overload"] = log.predict_proba(X_log)[:, 1]
    out["log_acc"] = accuracy_score(y_log, log.predict(X_log))
    out["log_auc"] = roc_auc_score(y_log, df["Prob_Overload"])
    out["x_range"] = np.linspace(df["ISLR"].min(), df["ISLR"].max(), 300).reshape(-1, 1)
    out["y_prob_curve"] = log.predict_proba(out["x_range"])[:, 1]

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    ).fit(df[RF_FEATURES], df["Rischio Overload"])
    rf_pred = rf.predict(df[RF_FEATURES])
    rf_proba = rf.predict_proba(df[RF_FEATURES])[:, 1]
    out["rf_acc"] = accuracy_score(df["Rischio Overload"], rf_pred)
    out["rf_auc"] = roc_auc_score(df["Rischio Overload"], rf_proba)
    out["rf"] = rf
    out["imp_df"] = pd.DataFrame({
        "Feature": RF_FEATURES,
        "Importanza": rf.feature_importances_
    }).sort_values("Importanza", ascending=True)

    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[["FC Media", "ISLR"]])
    df["Cluster_ID"] = km.labels_
    out["sil"] = silhouette_score(df[["FC Media", "ISLR"]], km.labels_)
    centroids = pd.DataFrame(km.cluster_centers_, columns=["FC Media", "ISLR"])
    order = centroids["ISLR"].sort_values().index.tolist()
    labels = ["Rigenerativo", "Qualità / Misto", "Elevato Stress"]
    cluster_map = {cluster_id: labels[i] for i, cluster_id in enumerate(order)}
    df["Profilo_Corsa"] = df["Cluster_ID"].map(cluster_map)

    out["km"] = km
    out["cluster_map"] = cluster_map
    out["df"] = df
    return out

def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"]),
        margin=dict(t=50, l=20, r=20, b=20),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig

def model_header(title, subtitle):
    st.markdown(f"""
    <div class="glass">
        <div class="model-title">{title}</div>
        <div class="model-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)

def model_footer():
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# APP START
# ============================================================================
df = generate_data()
res = train_all(df)
data = res["df"]

st.markdown("""
<div class="hero">
    <div class="hero-title">Advanced Machine Learning Suite</div>
    <div class="hero-subtitle">
        Una dashboard interattiva in cui ogni algoritmo vive in una vista dedicata:
        scegli orizzontalmente il modello che vuoi analizzare, osserva più grafici,
        leggi la teoria, interpreta i risultati e trasforma i numeri in insight da tesi.
    </div>
</div>
""", unsafe_allow_html=True)

a, b, c, d = st.columns(4)
a.metric("Linear Regression", f"R² {res['lr_r2']:.2f}")
b.metric("Logistic Regression", f"AUC {res['log_auc']:.2f}")
c.metric("Random Forest", f"AUC {res['rf_auc']:.2f}")
d.metric("K-Means", f"Silhouette {res['sil']:.2f}")

model_view = st.segmented_control(
    "Scegli il modello da esplorare",
    options=[
        "📈 Regressione Lineare",
        "🎯 Regressione Logistica",
        "🌳 Random Forest",
        "🔍 K-Means"
    ],
    default="📈 Regressione Lineare"
)

# ============================================================================
# VIEW 1
# ============================================================================
if model_view == "📈 Regressione Lineare":
    model_header(
        "📈 Regressione Lineare",
        "Stimare il tempo atteso di performance a partire dalla distanza."
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("R²", f"{res['lr_r2']:.2f}")
    o2.metric("Tempo medio", f"{data['Tempo (min)'].mean():.1f} min")
    o3.metric("Errore medio assoluto", f"{data['Residuo'].abs().mean():.1f} min")

    st.markdown(f"""
    <div class="theory-box">
        <b>Fondamento teorico.</b> La regressione lineare stima una relazione continua tra input e output.
        Qui traduce i chilometri percorsi in una previsione del tempo finale, offrendo un modello semplice,
        leggibile e molto utile come baseline interpretativa.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Overview", "Grafici avanzati"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["Distanza (km)"],
                y=data["Tempo (min)"],
                mode="markers",
                name="Dati reali",
                marker=dict(color=COLORS["blue"], size=8, opacity=0.65)
            ))
            fig.add_trace(go.Scatter(
                x=data["Distanza (km)"],
                y=data["Tempo_Predetto"],
                mode="lines",
                name="Trend OLS",
                line=dict(color=COLORS["pink"], width=3)
            ))
            fig.update_layout(title="Relazione distanza-tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            fig = px.histogram(
                data,
                x="Residuo",
                nbins=24,
                title="Distribuzione dei residui",
                color_discrete_sequence=[COLORS["purple"]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="explain-box">
            <b>Come leggere questi grafici.</b> Il primo mostra la nuvola reale delle osservazioni e la linea ottimale calcolata dal modello.
            Il secondo rivela quanto la previsione sbaglia e se l’errore è distribuito in modo equilibrato.
        </div>
        """, unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(
                data,
                x="Tempo_Predetto",
                y="Tempo (min)",
                trendline="ols",
                title="Predetto vs osservato",
                color_discrete_sequence=[COLORS["cyan"]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            fig = px.scatter(
                data,
                x="Distanza (km)",
                y="Residuo",
                title="Residui vs distanza",
                color_discrete_sequence=[COLORS["amber"]]
            )
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS["red"])
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <b>Insight da tesi.</b> Questo modello è il più adatto quando vuoi mostrare una relazione intuitiva e facilmente discutibile.
            Non coglie tutta la complessità fisiologica, ma è perfetto per aprire il percorso analitico con una baseline chiara.
        </div>
        """, unsafe_allow_html=True)

    model_footer()

# ============================================================================
# VIEW 2
# ============================================================================
elif model_view == "🎯 Regressione Logistica":
    model_header(
        "🎯 Regressione Logistica",
        "Stimare la probabilità che una sessione entri in area di overload."
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("AUC", f"{res['log_auc']:.2f}")
    o2.metric("Accuracy", f"{res['log_acc']*100:.0f}%")
    o3.metric("Overload rate", f"{data['Rischio Overload'].mean()*100:.0f}%")

    st.markdown("""
    <div class="theory-box">
        <b>Fondamento teorico.</b> La regressione logistica non predice un valore continuo, ma una probabilità compresa tra 0 e 1.
        È ideale quando la domanda non è “quanto?”, ma “quanto è probabile che accada?”.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Overview", "Grafici avanzati"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["ISLR"],
                y=data["Rischio Overload"],
                mode="markers",
                name="Osservazioni",
                marker=dict(color="#94a3b8", size=8, opacity=0.45)
            ))
            fig.add_trace(go.Scatter(
                x=res["x_range"].flatten(),
                y=res["y_prob_curve"],
                mode="lines",
                name="Curva sigmoide",
                line=dict(color=COLORS["amber"], width=3)
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["red"])
            fig.update_layout(title="Curva di transizione verso l'overload", xaxis_title="ISLR", yaxis_title="Probabilità")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            fig = px.box(
                data,
                x="Rischio Overload",
                y="Prob_Overload",
                color="Rischio Overload",
                title="Separabilità delle classi",
                color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="explain-box">
            <b>Come leggere i risultati.</b> La curva sigmoide mostra il punto in cui l’aumento dell’indice ISLR fa impennare il rischio.
            Il boxplot verifica se il modello distingue bene le sessioni sicure da quelle critiche.
        </div>
        """, unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)

        with c1:
            prob_bins = pd.cut(data["Prob_Overload"], bins=5)
            calib = data.groupby(prob_bins)["Rischio Overload"].mean().reset_index()
            calib["bin"] = calib["Prob_Overload"].astype(str)

            fig = px.bar(
                calib,
                x="bin",
                y="Rischio Overload",
                title="Rischio osservato per fasce di probabilità",
                color_discrete_sequence=[COLORS["purple"]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            fig = px.histogram(
                data,
                x="Prob_Overload",
                color="Rischio Overload",
                nbins=25,
                title="Distribuzione delle probabilità stimate",
                color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <b>Insight da tesi.</b> Questo modello è ottimo quando vuoi trasformare un indice sintetico in una decisione binaria interpretabile.
            È il ponte perfetto tra descrizione e decision making.
        </div>
        """, unsafe_allow_html=True)

    model_footer()

# ============================================================================
# VIEW 3
# ============================================================================
elif model_view == "🌳 Random Forest":
    model_header(
        "🌳 Random Forest",
        "Spiegare il rischio di overload con una lettura multifattoriale."
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("AUC", f"{res['rf_auc']:.2f}")
    o2.metric("Accuracy", f"{res['rf_acc']*100:.0f}%")
    o3.metric("N. feature", f"{len(RF_FEATURES)}")

    st.markdown("""
    <div class="theory-box">
        <b>Fondamento teorico.</b> Il Random Forest unisce molti alberi decisionali e sintetizza i loro voti.
        Questo gli permette di cogliere interazioni non lineari tra variabili fisiologiche, ambientali e di recupero.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Overview", "Grafici avanzati"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                res["imp_df"],
                x="Importanza",
                y="Feature",
                orientation="h",
                title="Feature importance",
                color="Importanza",
                color_continuous_scale=["#164e63", "#0891b2", "#67e8f9"]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            top2 = res["imp_df"].tail(2)["Feature"].tolist()
            fig = px.scatter(
                data,
                x=top2[0],
                y=top2[1],
                color="Rischio Overload",
                title=f"Interazione: {top2[0]} vs {top2[1]}",
                color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="explain-box">
            <b>Come leggere i risultati.</b> La feature importance ordina i fattori che il modello giudica più utili per classificare il rischio.
            Il grafico di interazione ti fa vedere dove le sessioni critiche si concentrano nello spazio delle due variabili più forti.
        </div>
        """, unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                x=res["rf"].predict_proba(data[RF_FEATURES])[:, 1],
                nbins=24,
                title="Distribuzione score Random Forest",
                color_discrete_sequence=[COLORS["green"]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with c2:
            feature_means = data.groupby("Rischio Overload")[RF_FEATURES].mean().T.reset_index()
            feature_means.columns = ["Feature", "Sicuro", "Rischio"]
            melt_df = feature_means.melt(id_vars="Feature", var_name="Classe", value_name="Valore")
            fig = px.bar(
                melt_df,
                x="Feature",
                y="Valore",
                color="Classe",
                barmode="group",
                title="Profilo medio feature per classe",
                color_discrete_map={"Sicuro": COLORS["blue"], "Rischio": COLORS["red"]}
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <b>Insight da tesi.</b> Qui il racconto si fa davvero forte: non stai solo dicendo se c’è rischio,
            ma stai mostrando quali combinazioni di fattori lo fanno emergere.
        </div>
        """, unsafe_allow_html=True)

    model_footer()

# ============================================================================
# VIEW 4
# ============================================================================
elif model_view == "🔍 K-Means":
    model_header(
        "🔍 K-Means Clustering",
        "Scoprire profili di allenamento ricorrenti senza etichette predefinite."
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("Silhouette", f"{res['sil']:.2f}")
    o2.metric("Cluster", "3")
    o3.metric("Sessioni", f"{len(data)}")

    st.markdown("""
    <div class="theory-box">
        <b>Fondamento teorico.</b> K-Means cerca gruppi naturali nei dati in base alla distanza dai centroidi.
        È uno strumento eccellente per passare dalla logica predittiva alla logica esplorativa.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Overview", "Grafici avanzati"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(
                data,
                x="ISLR",
                y="FC Media",
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

        st.markdown("""
        <div class="explain-box">
            <b>Come leggere i risultati.</b> Il primo grafico ti fa vedere le famiglie di sessioni scoperte automaticamente.
            Il secondo traduce quei gruppi in comportamento medio, così i cluster diventano leggibili anche a livello narrativo.
        </div>
        """, unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)

        with c1:
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

        with c2:
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
            <b>Insight da tesi.</b> Questa sezione è potentissima perché mostra che nei dati esistono archetipi ricorrenti di allenamento.
            È la parte più “scoperta” e meno guidata da etichette, quindi visivamente colpisce molto.
        </div>
        """, unsafe_allow_html=True)

    model_footer()
