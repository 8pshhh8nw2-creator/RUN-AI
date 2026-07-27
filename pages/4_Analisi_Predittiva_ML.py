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
    page_title="Advanced ML Suite | Tesi Magistrale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

COLORS = {
    "bg": "#0b1020",
    "surface": "#121a2b",
    "surface_2": "#182235",
    "border": "#263247",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "blue": "#38bdf8",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "green": "#22c55e",
    "purple": "#a855f7"
}

RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR"]

# ============================================================================
# THEME
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 20%),
            linear-gradient(180deg, {COLORS["bg"]} 0%, #0a0f1a 100%);
        color: {COLORS["text"]};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }}

    .hero {{
        background: linear-gradient(135deg, {COLORS["surface"]} 0%, {COLORS["surface_2"]} 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
    }}

    .hero h1 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        color: {COLORS["text"]};
        letter-spacing: -0.03em;
    }}

    .hero p {{
        margin: 0.65rem 0 0 0;
        color: {COLORS["muted"]};
        font-size: 1rem;
        max-width: 900px;
    }}

    .section-card {{
        background: rgba(18, 26, 43, 0.88);
        border: 1px solid {COLORS["border"]};
        border-radius: 18px;
        padding: 1.25rem 1.25rem 0.8rem 1.25rem;
        margin-bottom: 1rem;
    }}

    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-bottom: 0.3rem;
    }}

    .section-subtitle {{
        color: {COLORS["muted"]};
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }}

    .info-box {{
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid {COLORS["blue"]};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #dbeafe;
    }}

    .theory-box {{
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.18);
        border-left: 4px solid {COLORS["amber"]};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0 1rem 0;
        color: #fef3c7;
    }}

    .transition-box {{
        background: rgba(168, 85, 247, 0.08);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-left: 4px solid {COLORS["purple"]};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 1.2rem 0;
        color: #e9d5ff;
        font-style: italic;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, {COLORS["surface"]}, {COLORS["surface_2"]});
        border: 1px solid {COLORS["border"]};
        padding: 1rem;
        border-radius: 16px;
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
        font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA + MODELS
# ============================================================================
@st.cache_data
def generate_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
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
def train_models(df: pd.DataFrame) -> dict:
    results = {}

    # Linear Regression
    X_lr = df[["Distanza (km)"]]
    y_lr = df["Tempo (min)"]
    lr_model = LinearRegression().fit(X_lr, y_lr)
    tempo_pred = lr_model.predict(X_lr)
    residui = y_lr - tempo_pred
    lr_r2 = r2_score(y_lr, tempo_pred)

    # Logistic Regression
    X_log = df[["ISLR"]]
    y_log = df["Rischio Overload"]
    log_model = LogisticRegression().fit(X_log, y_log)
    prob_overload = log_model.predict_proba(X_log)[:, 1]
    log_pred = log_model.predict(X_log)
    log_acc = accuracy_score(y_log, log_pred)
    log_auc = roc_auc_score(y_log, prob_overload)

    x_range = np.linspace(df["ISLR"].min(), df["ISLR"].max(), 300).reshape(-1, 1)
    y_prob_curve = log_model.predict_proba(x_range)[:, 1]

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    ).fit(df[RF_FEATURES], df["Rischio Overload"])

    rf_proba = rf_model.predict_proba(df[RF_FEATURES])[:, 1]
    rf_pred = rf_model.predict(df[RF_FEATURES])
    rf_acc = accuracy_score(df["Rischio Overload"], rf_pred)
    rf_auc = roc_auc_score(df["Rischio Overload"], rf_proba)

    imp_df = (
        pd.DataFrame({
            "Feature": RF_FEATURES,
            "Importanza": rf_model.feature_importances_
        })
        .sort_values("Importanza", ascending=True)
        .reset_index(drop=True)
    )

    # KMeans
    km_model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[["FC Media", "ISLR"]])
    clusters = km_model.labels_
    sil = silhouette_score(df[["FC Media", "ISLR"]], clusters)

    centroids = pd.DataFrame(km_model.cluster_centers_, columns=["FC Media", "ISLR"])
    order = centroids["ISLR"].sort_values().index.tolist()
    labels = ["Rigenerativo", "Qualità / Misto", "Elevato Stress"]
    cluster_map = {cluster_id: labels[i] for i, cluster_id in enumerate(order)}

    enriched = df.copy()
    enriched["Tempo_Predetto"] = tempo_pred
    enriched["Errore (Residuo)"] = residui
    enriched["Probabilità_Overload"] = prob_overload
    enriched["Cluster_ID"] = clusters
    enriched["Profilo_Corsa"] = enriched["Cluster_ID"].map(cluster_map)

    results.update({
        "df": enriched,
        "lr_model": lr_model,
        "lr_r2": lr_r2,
        "log_model": log_model,
        "log_acc": log_acc,
        "log_auc": log_auc,
        "x_range": x_range,
        "y_prob_curve": y_prob_curve,
        "rf_model": rf_model,
        "rf_acc": rf_acc,
        "rf_auc": rf_auc,
        "imp_df": imp_df,
        "km_model": km_model,
        "sil_score": sil,
        "cluster_map": cluster_map
    })
    return results


def style_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"]),
        margin=dict(t=50, l=20, r=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def section_open(title: str, subtitle: str = ""):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# APP
# ============================================================================
df = generate_data()
res = train_models(df)
data = res["df"]

st.markdown("""
<div class='hero'>
    <h1>Advanced Machine Learning Suite</h1>
    <p>
        Dashboard analitica per la stima della performance, la classificazione del rischio di overload
        e la segmentazione dei profili di allenamento in ambito sportivo.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Quadro di sintesi")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Regressione Lineare", f"R² {res['lr_r2']:.2f}", "Tempo stimato")
k2.metric("Regressione Logistica", f"AUC {res['log_auc']:.2f}", f"ACC {res['log_acc']*100:.0f}%")
k3.metric("Random Forest", f"AUC {res['rf_auc']:.2f}", f"ACC {res['rf_acc']*100:.0f}%")
k4.metric("K-Means", f"Silhouette {res['sil_score']:.2f}", "3 profili")

mapping_df = pd.DataFrame({
    "Modello": ["Regressione Lineare", "Regressione Logistica", "Random Forest", "K-Means"],
    "Domanda": [
        "Quanto tempo impiegherò?",
        "Quanto è probabile il rischio?",
        "Quali fattori incidono di più?",
        "Quali profili ricorrono nei dati?"
    ],
    "Output": [
        "Predizione continua",
        "Probabilità / classe",
        "Feature importance",
        "Segmentazione"
    ]
})
st.dataframe(mapping_df, use_container_width=True, hide_index=True)

tab_models, tab_sim = st.tabs(["🧠 Analisi modelli", "🎮 Simulatore what-if"])

with tab_models:
    section_open(
        "Regressione lineare",
        "Stima del tempo di esecuzione in funzione della distanza."
    )
    st.markdown(
        f"<div class='theory-box'><b>Insight:</b> il modello lineare raggiunge R² = {res['lr_r2']:.2f} e fornisce una baseline leggibile e interpretabile.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["Distanza (km)"],
            y=data["Tempo (min)"],
            mode="markers",
            name="Osservazioni",
            marker=dict(color=COLORS["blue"], size=8, opacity=0.65)
        ))
        fig.add_trace(go.Scatter(
            x=data["Distanza (km)"],
            y=data["Tempo_Predetto"],
            mode="lines",
            name="Trend OLS",
            line=dict(color=COLORS["red"], width=3)
        ))
        fig.update_layout(title="Distanza vs Tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
        st.plotly_chart(style_figure(fig), use_container_width=True)

    with c2:
        fig = px.histogram(
            data,
            x="Errore (Residuo)",
            nbins=24,
            color_discrete_sequence=[COLORS["purple"]],
            title="Distribuzione dei residui"
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)
    section_close()

    st.markdown("<div class='transition-box'>Dal tempo stimato passiamo alla probabilità di rischio della sessione.</div>", unsafe_allow_html=True)

    section_open(
        "Regressione logistica",
        "Classificazione probabilistica del rischio overload a partire dall'ISLR."
    )
    st.markdown(
        f"<div class='theory-box'><b>Insight:</b> la regressione logistica ottiene AUC = {res['log_auc']:.2f} con accuratezza = {res['log_acc']*100:.0f}%.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["ISLR"],
            y=data["Rischio Overload"],
            mode="markers",
            name="Osservazioni",
            marker=dict(color="#94a3b8", opacity=0.45, size=8)
        ))
        fig.add_trace(go.Scatter(
            x=res["x_range"].flatten(),
            y=res["y_prob_curve"],
            mode="lines",
            name="Curva sigmoide",
            line=dict(color=COLORS["amber"], width=3)
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["red"])
        fig.update_layout(title="Probabilità di overload", xaxis_title="ISLR", yaxis_title="Probabilità")
        st.plotly_chart(style_figure(fig), use_container_width=True)

    with c2:
        fig = px.box(
            data,
            x="Rischio Overload",
            y="Probabilità_Overload",
            color="Rischio Overload",
            title="Separazione delle classi",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)
    section_close()

    st.markdown("<div class='transition-box'>Dal rischio stimato passiamo ai driver principali del rischio con un modello multivariato.</div>", unsafe_allow_html=True)

    section_open(
        "Random Forest",
        "Modello ensemble per spiegare il peso relativo delle feature."
    )
    st.markdown(
        f"<div class='theory-box'><b>Insight:</b> il Random Forest migliora la lettura multifattoriale del rischio con AUC = {res['rf_auc']:.2f} e accuratezza = {res['rf_acc']*100:.0f}%.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            res["imp_df"],
            x="Importanza",
            y="Feature",
            orientation="h",
            color="Importanza",
            color_continuous_scale=["#164e63", "#0891b2", "#67e8f9"],
            title="Feature importance"
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)

    with c2:
        top2 = res["imp_df"].tail(2)["Feature"].tolist()
        fig = px.scatter(
            data,
            x=top2[0],
            y=top2[1],
            color="Rischio Overload",
            title=f"Interazione tra {top2[0]} e {top2[1]}",
            color_discrete_map={0: COLORS["blue"], 1: COLORS["red"]}
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)
    section_close()

    st.markdown("<div class='transition-box'>Infine, esploriamo i profili latenti delle sessioni senza etichette note.</div>", unsafe_allow_html=True)

    section_open(
        "K-Means clustering",
        "Segmentazione non supervisionata dei profili di allenamento."
    )
    st.markdown(
        f"<div class='theory-box'><b>Insight:</b> il clustering produce 3 gruppi con silhouette = {res['sil_score']:.2f}.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            data,
            x="ISLR",
            y="FC Media",
            color="Profilo_Corsa",
            title="Profili di allenamento",
            color_discrete_sequence=[COLORS["green"], COLORS["amber"], COLORS["red"]]
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)

    with c2:
        cluster_means = (
            data.groupby("Profilo_Corsa")[["Ore Sonno", "RPE", "Tempo (min)"]]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            cluster_means,
            x="Profilo_Corsa",
            y=["Ore Sonno", "RPE"],
            barmode="group",
            title="Medie per cluster"
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)
    section_close()

with tab_sim:
    st.subheader("Simulatore scenario")
    left, right = st.columns([1, 1], gap="large")

    with left:
        sim_dist = st.slider("Distanza pianificata (km)", 5.0, 30.0, 10.0, 0.5)
        sim_sleep = st.slider("Ore di sonno", 3.0, 10.0, 7.5, 0.5)
        sim_stress = st.slider("Stress lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
        sim_work = st.slider("Ore lavoro", 0.0, 12.0, 8.0, 0.5)
        sim_rpe = st.slider("RPE previsto", 1.0, 10.0, 6.0, 1.0)

    with right:
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

        prob = res["rf_model"].predict_proba(input_df)[0, 1] * 100

        sim_point = np.array([[sim_fc, sim_islr]])
        dists = np.linalg.norm(res["km_model"].cluster_centers_ - sim_point, axis=1)
        nearest_cluster = int(np.argmin(dists))
        nearest_profile = res["cluster_map"][nearest_cluster]

        status_color = COLORS["blue"] if prob < 40 else COLORS["amber"] if prob < 70 else COLORS["red"]
        status_label = "Verde" if prob < 40 else "Attenzione" if prob < 70 else "Critico"

        st.markdown(f"""
        <div style="
            background: linear-gradient(180deg, {COLORS["surface"]}, {COLORS["surface_2"]});
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 1.4rem;
            text-align: center;">
            <div style="color:{COLORS["muted"]}; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">
                Probabilità di overload
            </div>
            <div style="font-size:4rem; font-weight:800; color:{status_color}; line-height:1; margin:0.5rem 0;">
                {prob:.1f}%
            </div>
            <div style="color:{COLORS["muted"]}; font-size:0.95rem;">
                Stato: {status_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("SMA", f"{sim_sma:.2f}")
        m2.metric("ISLR", f"{sim_islr:.2f}")
        m3.metric("Profilo vicino", nearest_profile)

        if prob < 40:
            st.success("Carico sotto controllo. Sessione coerente con buon equilibrio recupero-stress.")
        elif prob < 70:
            st.warning("Area intermedia. Valuta riduzione del carico o aumento del recupero.")
        else:
            st.error("Rischio elevato. Meglio alleggerire il carico o rinviare la sessione.")

        st.caption("Il simulatore usa il Random Forest per il rischio e K-Means per il profilo più vicino.")
