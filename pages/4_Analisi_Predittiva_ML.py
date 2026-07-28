"""
RUN AI - Sport ML Suite v2
Dashboard interattiva per tesi magistrale.
File unico. Avvio: streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

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
    precision_score, r2_score, roc_auc_score, roc_curve, precision_recall_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE PAGINA
# ============================================================================
st.set_page_config(
    page_title="RUN AI - Sport ML Suite",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏃"
)

# ============================================================================
# COSTANTI E COLORI
# ============================================================================
COLORS = {
    "bg": "#0b0f19",
    "surface": "#111827",
    "surface_2": "#1e293b",
    "border": "#1f2937",
    "border_light": "rgba(255,255,255,0.06)",
    "text": "#ffffff",
    "text_soft": "#9ca3af",
    "cyan": "#00e5ff",
    "cyan_dim": "#131c2c",
    "green": "#a3e635",
    "amber": "#fbbf24",
    "red": "#f87171",
    "purple": "#a78bfa",
}

QUALITATIVE = [COLORS['cyan'], COLORS['purple'], COLORS['amber'], COLORS['green'], COLORS['red']]
CLUSTER_COLORS = {0: COLORS['cyan'], 1: COLORS['purple'], 2: COLORS['amber'], 3: COLORS['green']}

TARGET = "Rischio Overload"
TIME_TARGET = "Tempo (min)"
RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR", "RPE"]
CLUSTER_FEATURES = ["FC Media", "ISLR", "SMA"]

RISK_BANDS = ((40.0, "Sicuro", COLORS['green']), (70.0, "Attenzione", COLORS['amber']), (101.0, "Pericolo", COLORS['red']))


def risk_band(prob: float) -> tuple[str, str]:
    for thr, label, color in RISK_BANDS:
        if prob < thr:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    test_size: float = 0.25
    n_estimators: int = 220
    max_depth: int = 8
    n_clusters: int = 3
    n_sessions: int = 180  # sessioni "giornaliere" per costruire un calendario realistico


# ============================================================================
# CSS
# ============================================================================
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {COLORS['bg']} !important;
        color: {COLORS['text']} !important;
    }}
    .stApp {{ background-color: {COLORS['bg']} !important; }}
    [data-testid="collapsedControl"], #MainMenu, footer, header {{ display: none !important; }}
    .block-container {{ max-width: 1350px; padding-top: 2rem !important; }}

    .runai-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 10px;
        padding: 2.5rem;
        position: relative;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .runai-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['cyan']} 0%, {COLORS['green']} 100%);
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }}
    .runai-kicker {{
        color: {COLORS['cyan']};
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}
    .runai-title {{
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 1rem;
    }}
    .runai-subtitle {{
        color: {COLORS['text_soft']};
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.55;
    }}
    .runai-info-box {{
        background-color: {COLORS['cyan_dim']};
        border-left: 3px solid {COLORS['cyan']};
        border-radius: 4px;
        padding: 1.2rem;
        color: {COLORS['text_soft']};
        font-size: 0.9rem;
    }}
    .runai-info-box strong {{ color: #ffffff; }}
    .runai-info-box.warn {{ border-left-color: {COLORS['amber']}; background-color: rgba(251,191,36,0.06); }}

    /* TABS come pulsanti tech */
    button[data-baseweb="tab"] {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 6px !important;
        margin-right: 6px;
        padding: 10px 16px !important;
    }}
    button[data-baseweb="tab"] p {{
        color: {COLORS['text_soft']} !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        border-color: {COLORS['cyan']} !important;
        background-color: rgba(0,229,255,0.06) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] p {{ color: {COLORS['cyan']} !important; }}
    div[data-baseweb="tab-highlight"] {{ background-color: transparent !important; }}
    div[data-baseweb="tab-border"] {{ background-color: {COLORS['border']} !important; }}

    .coach-insight {{
        background: {COLORS['surface']};
        padding: 1rem 1.1rem;
        border-radius: 6px;
        border: 1px solid {COLORS['border']};
        font-size: 0.85rem;
        color: {COLORS['text_soft']};
        margin-top: 6px;
        margin-bottom: 22px;
        line-height: 1.5;
    }}
    .coach-insight span {{
        color: {COLORS['cyan']};
        font-weight: 800;
        text-transform: uppercase;
        font-size: 0.68rem;
        display: block;
        margin-bottom: 5px;
        letter-spacing: 0.06em;
    }}

    .kpi-pill {{
        display: inline-block;
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 0.72rem;
        font-weight: 700;
        color: {COLORS['text_soft']};
        margin-right: 8px;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }}

    div[data-testid="stMetric"] {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 14px 16px 6px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {COLORS['text_soft']} !important; }}

    hr {{ border-color: {COLORS['border']} !important; }}
    </style>
    """, unsafe_allow_html=True)


def register_plotly_template():
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS['text_soft'], size=12),
        title=dict(font=dict(size=14, color=COLORS['text'], family="Inter"), x=0.01, xanchor="left", y=0.95),
        margin=dict(t=40, l=10, r=10, b=20),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border_light'], zeroline=False, linecolor=COLORS['border']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border_light'], zeroline=False, linecolor=COLORS['border']),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["runai"] = tpl
    pio.templates.default = "runai"


def insight(label: str, text: str):
    st.markdown(f"<div class='coach-insight'><span>{label}</span>{text}</div>", unsafe_allow_html=True)


# ============================================================================
# DATI SINTETICI (calendario giornaliero, così nascono trend e ACWR)
# ============================================================================
@st.cache_data
def generate_synthetic_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=int(i * rng.uniform(0.9, 1.6))) for i in range(n)]

    # una leggera periodizzazione: blocchi di carico crescente + scarico ogni 4 settimane
    week = np.array([(d - start).days // 7 for d in dates])
    ciclo = (week % 4)
    fattore_blocco = np.where(ciclo == 3, 0.6, 1.0 + 0.08 * ciclo)  # settimana di scarico ogni 4

    distanza = (rng.uniform(5.0, 30.0, n) * fattore_blocco).clip(3, 42)
    rpe = np.clip(rng.integers(2, 11, n) * (fattore_blocco > 0.7), 1, 10)
    ore_sonno = rng.normal(7.5, 1.2, n).clip(3, 10)
    temperatura = rng.normal(20, 8, n)
    vento = rng.normal(10, 5, n).clip(0, 40)

    tempo = distanza * rng.normal(4.5, 0.3, n) + (rpe * 2)
    velocita = (distanza * 1000) / (tempo * 60)
    fc_media = 110 + (rpe * 6) - (ore_sonno * 2) + rng.normal(0, 5, n)

    ore_lavoro = tempo / 60
    sma = (ore_lavoro * rpe) / ore_sonno
    islr = (ore_lavoro * rpe) / distanza
    idet = (fc_media * temperatura) / np.where(velocita > 0, velocita, 1)
    iitr = (temperatura * vento) / distanza

    stress_score = (sma * 0.4) + (islr * 0.3) + (rpe * 0.3)
    prob_overload = 1 / (1 + np.exp(-(stress_score - np.median(stress_score))))
    rischio = (prob_overload > 0.65).astype(int)

    df = pd.DataFrame({
        "Data": dates,
        "Distanza (km)": distanza, "Tempo (min)": tempo, "Velocità (m/s)": velocita,
        "RPE": rpe, "Ore Sonno": ore_sonno, "FC Media": fc_media,
        "Temperatura": temperatura, "Vento": vento,
        "SMA": sma, "ISLR": islr, "IDET": idet, "IITR": iitr,
        TARGET: rischio,
    }).round(2)
    df = df.sort_values("Data").reset_index(drop=True)

    # Carico allenante giornaliero (TRIMP semplificato) per calcolare ACWR
    df["Carico"] = (df["Tempo (min)"] * df["RPE"]).round(1)
    df["Carico Acuto (7g)"] = df["Carico"].rolling(7, min_periods=1).mean()
    df["Carico Cronico (28g)"] = df["Carico"].rolling(28, min_periods=1).mean()
    df["ACWR"] = (df["Carico Acuto (7g)"] / df["Carico Cronico (28g)"]).round(2)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame, config: Settings):
    X_cls = df[RF_FEATURES]
    y_cls = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=config.test_size, random_state=config.seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=config.seed)
    rf.fit(X_train, y_train)

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)

    X_reg = df[["Distanza (km)", "SMA", "RPE"]]
    y_reg = df[TIME_TARGET]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=config.test_size, random_state=config.seed)
    reg = LinearRegression().fit(X_train_r, y_train_r)

    X_clust = df[CLUSTER_FEATURES]
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed, n_init=10)
    df["Cluster"] = kmeans.fit_predict(StandardScaler().fit_transform(X_clust))

    return rf, lr, reg, kmeans, scaler, X_test, y_test, X_test_scaled, X_test_r, y_test_r


# ============================================================================
# INTERFACCIA
# ============================================================================
def render_ui():
    inject_css()
    register_plotly_template()
    cfg = Settings()
    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    rf, lr, reg, kmeans, scaler, X_test, y_test, X_test_scaled, X_test_r, y_test_r = train_models(df, cfg)

    # ------------------------------------------------------------------ HERO
    st.markdown(f"""
    <div class="runai-card">
        <div class="runai-kicker">● MODULO 06 - SPORT DATA SCIENCE</div>
        <div class="runai-title">AI PERFORMANCE ANALYSIS & INJURY PREDICTION</div>
        <div class="runai-subtitle">
            Questa dashboard analizza i dati storici del team e li mette in relazione con carico, sonno e recupero.
            L'Intelligenza Artificiale impara come il corpo degli atleti reagisce agli stimoli, prevede cali di rendimento
            e segnala il rischio di sovraccarico prima che diventi un infortunio.
        </div>
        <div>
            <span class="kpi-pill">SESSIONI: {len(df)}</span>
            <span class="kpi-pill">PERIODO: {df['Data'].min():%d/%m/%y} → {df['Data'].max():%d/%m/%y}</span>
            <span class="kpi-pill">FEATURE: {len(RF_FEATURES)}</span>
            <span class="kpi-pill">MODELLI: 4</span>
        </div>
        <br>
        <div class="runai-info-box">
            <strong>Analisi Fisiologica Avanzata:</strong> estrazione e calcolo dello Stress Metabolico Accumulato (SMA)
            e dell'Indice di Lavoro (ISLR), KPI proprietari sviluppati per la tesi. I modelli predittivi valutano come
            la mancanza di sonno, la percezione della fatica (RPE) e l'ambiente si combinano per alterare il ritmo gara
            e logorare l'atleta nel tempo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📊 Esplorazione Dati",
        "📅 Carico & ACWR",
        "⏱️ Stima del Ritmo",
        "⚠️ Rischio Overload",
        "🧬 Profilazione Atleti",
        "🎛️ Simulatore Coach",
    ])

    # ================================================================ TAB 1: EDA
    with tabs[0]:
        st.markdown("<h3 style='color:white;'>Analisi Esplorativa del Dataset</h3>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessioni totali", len(df))
        c2.metric("Distanza media", f"{df['Distanza (km)'].mean():.1f} km")
        c3.metric("Sonno medio", f"{df['Ore Sonno'].mean():.1f} h")
        c4.metric("% sessioni a rischio", f"{df[TARGET].mean()*100:.1f}%")

        g1, g2 = st.columns([1.3, 1])
        with g1:
            corr = df[RF_FEATURES + ["Tempo (min)", "FC Media"]].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Matrice di Correlazione tra le Variabili")
            fig_corr.update_layout(height=440)
            st.plotly_chart(fig_corr, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Rosso = le due variabili crescono insieme, blu = una sale mentre l'altra scende. È il primo "
                    "controllo scientifico: se SMA e RPE non fossero correlati, il KPI andrebbe rivisto prima di darlo in pasto ai modelli.")
        with g2:
            fig_dist = px.histogram(df, x="Distanza (km)", nbins=30, color_discrete_sequence=[COLORS['cyan']],
                                     title="Distribuzione delle Distanze Percorse")
            fig_dist.update_layout(height=440)
            st.plotly_chart(fig_dist, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Mostra su quali distanze si allena davvero il gruppo. Una campana troppo stretta intorno a "
                    "pochi km segnala poca varietà di stimoli nel piano stagionale.")

        g3, g4 = st.columns(2)
        with g3:
            fig_sc = px.scatter(df, x="Ore Sonno", y="SMA", color=TARGET, opacity=0.7,
                                 color_continuous_scale=[COLORS['green'], COLORS['red']],
                                 title="Sonno vs Stress Metabolico Accumulato (SMA)")
            st.plotly_chart(fig_sc, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "I punti rossi (sessioni a rischio) tendono ad affollarsi dove il sonno è basso e lo SMA è alto: "
                    "è la prova visiva che il recupero notturno è protettivo tanto quanto il carico è pericoloso.")
        with g4:
            fig_box = px.box(df, x="RPE", y="FC Media", color="RPE", color_discrete_sequence=QUALITATIVE * 3,
                              title="Frequenza Cardiaca per Livello di Fatica Percepita")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Se a parità di RPE la FC media varia molto tra sessioni, la percezione soggettiva della fatica "
                    "non sta raccontando la stessa storia del cuore: un campanello per tarare meglio le scale RPE con l'atleta.")

    # ================================================================ TAB 2: CARICO & ACWR
    with tabs[1]:
        st.markdown("<h3 style='color:white;'>Gestione del Carico nel Tempo (ACWR)</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="runai-info-box warn">
            <strong>Cos'è l'ACWR:</strong> Acute:Chronic Workload Ratio, il rapporto tra il carico allenante delle ultime 7
            giornate e la media delle ultime 4 settimane. In letteratura sportiva un valore tra 0.8 e 1.3 è considerato la
            "sweet spot": sopra 1.5 il rischio di infortunio da sovraccarico aumenta sensibilmente, sotto 0.8 l'atleta
            probabilmente sta perdendo forma (detraining).
        </div>
        """, unsafe_allow_html=True)

        last = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("ACWR attuale", f"{last['ACWR']:.2f}")
        c2.metric("Carico acuto (7g)", f"{last['Carico Acuto (7g)']:.0f}")
        c3.metric("Carico cronico (28g)", f"{last['Carico Cronico (28g)']:.0f}")

        fig_acwr = go.Figure()
        fig_acwr.add_trace(go.Scatter(x=df["Data"], y=df["ACWR"], mode="lines", name="ACWR",
                                       line=dict(color=COLORS['cyan'], width=2)))
        fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor=COLORS['green'], opacity=0.08, line_width=0)
        fig_acwr.add_hrect(y0=1.3, y1=df["ACWR"].max() + 0.3, fillcolor=COLORS['red'], opacity=0.08, line_width=0)
        fig_acwr.update_layout(title="Andamento dell'ACWR nel Tempo (zona verde = carico sicuro)", height=380)
        st.plotly_chart(fig_acwr, use_container_width=True)
        insight("Cosa significa per il Coach",
                "Ogni volta che la linea esce sopra la fascia verde, l'atleta ha aumentato il carico troppo in fretta "
                "rispetto a quanto era abituato: è il momento in cui, storicamente, compaiono stanchezza cronica e infortuni.")

        g1, g2 = st.columns(2)
        with g1:
            fig_load = px.area(df, x="Data", y=["Carico Acuto (7g)", "Carico Cronico (28g)"],
                                title="Carico Acuto vs Cronico", color_discrete_sequence=[COLORS['cyan'], COLORS['purple']])
            st.plotly_chart(fig_load, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "La linea acuta (7 giorni) dovrebbe seguire con calma quella cronica (28 giorni). Se si stacca "
                    "verso l'alto in modo brusco, la settimana è stata uno strappo rispetto alle ultime quattro.")
        with g2:
            df["Settimana"] = df["Data"].dt.to_period("W").astype(str)
            weekly = df.groupby("Settimana")["Distanza (km)"].sum().reset_index()
            fig_week = px.bar(weekly, x="Settimana", y="Distanza (km)", color_discrete_sequence=[COLORS['amber']],
                               title="Volume Settimanale Totale (km)")
            fig_week.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_week, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Una vista da 'diario di allenamento': utile per verificare a occhio le settimane di scarico "
                    "programmate e capire se sono state rispettate davvero.")

    # ================================================================ TAB 3: REGRESSIONE TEMPO
    with tabs[2]:
        st.markdown("<h3 style='color:white;'>Stima Cronometrica e Analisi del Calo Prestativo</h3>", unsafe_allow_html=True)
        preds = reg.predict(X_test_r)

        c1, c2 = st.columns(2)
        c1.metric("Affidabilità Cronometro (R²)", f"{r2_score(y_test_r, preds)*100:.1f}%")
        c2.metric("Errore Medio Stima (MAE)", f"{mean_absolute_error(y_test_r, preds):.1f} min")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter(x=y_test_r, y=preds, opacity=0.6, color_discrete_sequence=[COLORS['cyan']],
                               labels={"x": "Tempo Reale (min)", "y": "Tempo Stimato (min)"},
                               title="Tempo Reale vs Tempo Calcolato dall'IA")
            fig1.add_shape(type="line", x0=y_test_r.min(), y0=y_test_r.min(), x1=y_test_r.max(), y1=y_test_r.max(),
                            line=dict(dash="dash", color=COLORS['text_soft']))
            st.plotly_chart(fig1, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Se i punti formano una linea retta perfetta, l'atleta è 'un orologio': la fatica lo rallenta "
                    "esattamente come calcolato. Punti molto distanti segnalano giornate anomale, per cause esterne non misurate.")

        with g2:
            fig2 = px.histogram(y_test_r - preds, nbins=30, color_discrete_sequence=[COLORS['purple']],
                                 labels={"value": "Errore (min)"}, title="Frequenza degli Errori di Previsione")
            st.plotly_chart(fig2, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Le barre più alte dovrebbero stare al centro (zero errori). Se la campana pende verso destra "
                    "o sinistra, il gruppo squadra tende sistematicamente a rendere meno (o di più) del previsto.")

        with g3:
            fig3 = px.scatter(x=preds, y=(y_test_r - preds), opacity=0.6, color_discrete_sequence=[COLORS['amber']],
                               labels={"x": "Tempo Stimato (min)", "y": "Errore (min)"},
                               title="Decadimento sulle Lunghe Distanze")
            fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['text_soft'])
            st.plotly_chart(fig3, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Guarda come si comporta l'errore man mano che i chilometri aumentano. Se l'errore esplode "
                    "a destra, gli atleti mancano di base aerobica per i lavori lunghi.")

        with g4:
            coefs = pd.DataFrame({"Fattore": X_test_r.columns, "Impatto": reg.coef_}).sort_values("Impatto")
            fig4 = px.bar(coefs, x="Impatto", y="Fattore", orientation="h", color_discrete_sequence=[COLORS['green']],
                           title="Responsabili del Rallentamento")
            st.plotly_chart(fig4, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Mostra i 'pesi' del modello: per ogni punto di fatica percepita (RPE) in più, quanti minuti "
                    "effettivi si perdono sul tempo finale. Aiuta a quantificare il costo della fatica.")

    # ================================================================ TAB 4: RISCHIO OVERLOAD (LR vs RF)
    with tabs[3]:
        st.markdown("<h3 style='color:white;'>Rischio Overload — Confronto tra Modelli</h3>", unsafe_allow_html=True)

        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]

        compare = pd.DataFrame({
            "Modello": ["Regressione Logistica (base)", "Random Forest (avanzato)"],
            "Accuratezza": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
            "AUC (capacità di allarme)": [roc_auc_score(y_test, y_prob_lr), roc_auc_score(y_test, y_prob_rf)],
            "Precisione": [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf)],
        })
        fig_cmp = px.bar(compare.melt(id_vars="Modello", var_name="Metrica", value_name="Valore"),
                          x="Metrica", y="Valore", color="Modello", barmode="group",
                          color_discrete_sequence=[COLORS['purple'], COLORS['cyan']],
                          title="Regressione Logistica vs Random Forest")
        fig_cmp.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_cmp, use_container_width=True)
        insight("Cosa significa per il Coach",
                "Il modello semplice (Logistica) è più facile da spiegare all'atleta; il Random Forest coglie "
                "combinazioni più complesse tra le variabili. Il confronto in tesi giustifica la scelta finale del modello.")

        sub = st.radio("Dettaglio modello", ["Random Forest (avanzato)", "Regressione Logistica (base)"], horizontal=True)
        if sub.startswith("Random"):
            y_pred, y_prob, cmap, feat_vals, feat_label = y_pred_rf, y_prob_rf, "Purp", rf.feature_importances_, "Potere Decisionale"
        else:
            y_pred, y_prob, cmap, feat_vals, feat_label = y_pred_lr, y_prob_lr, "Blues", lr.coef_[0], "Peso sul Rischio"

        c1, c2, c3 = st.columns(3)
        c1.metric("Decisioni corrette", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Sensibilità (AUC)", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Falsi allarmi", f"{(1-precision_score(y_test, y_pred))*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)
        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Efficacia degli Allarmi (Curva ROC)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['text_soft']), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig1, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Valuta se il modello ferma l'atleta al momento giusto. Più la curva sta in alto, più trova "
                    "davvero le sessioni a rischio di strappo muscolare.")
        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale=cmap,
                              labels=dict(x="IA Dice", y="È Successo"),
                              x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Contatore degli Errori")
            st.plotly_chart(fig2, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "La casella in basso a sinistra è il vero nemico: le volte in cui l'algoritmo ha detto 'sicuro' "
                    "e l'atleta è andato in sovraccarico (Falso Negativo).")
        with g3:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig3 = px.line(x=rec, y=prec, title="Qualità della Scelta (Precision-Recall)", color_discrete_sequence=[COLORS['purple']])
            st.plotly_chart(fig3, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Se per essere sicuri al 100% si fermasse l'atleta a ogni minimo segno di fatica, l'affidabilità "
                    "crollerebbe. Il grafico mostra il compromesso migliore tra i due estremi.")
        with g4:
            imp_df = pd.DataFrame({"Metrica": RF_FEATURES, feat_label: feat_vals}).sort_values(feat_label)
            fig4 = px.bar(imp_df, x=feat_label, y="Metrica", orientation="h", color_discrete_sequence=[COLORS['red']],
                           title="Le Variabili più Osservate dal Modello")
            st.plotly_chart(fig4, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Mostra a cosa fa più attenzione il modello per stimare il rischio. Solitamente RPE e Stress "
                    "Metabolico dominano la classifica, mentre il Sonno agisce da fattore protettivo.")

    # ================================================================ TAB 5: CLUSTERING
    with tabs[4]:
        st.markdown("<h3 style='color:white;'>Profilazione Automatica degli Allenamenti (K-Means)</h3>", unsafe_allow_html=True)

        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Separazione netta dei gruppi (silhouette)", f"{sil_score*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster",
                                  color_continuous_scale=list(CLUSTER_COLORS.values()), title="Mappa Fisiologica 3D")
            fig1.update_layout(scene=dict(xaxis_title="Battiti", yaxis_title="Indice Lavoro", zaxis_title="Stress"),
                                margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig1, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "L'IA ha preso mesi di allenamenti e li ha raggruppati da sola per 'impatto sul corpo'. Ogni "
                    "nuvola colorata è un macro-stimolo: rigenerazione, lavoro misto, altissima intensità.")

        with g2:
            centroids = df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
            fig2 = go.Figure()
            for _, row in centroids.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row["FC Media"]/df["FC Media"].max(), row["ISLR"]/df["ISLR"].max(), row["SMA"]/df["SMA"].max()],
                    theta=["FC Media", "Indice Lavoro", "Stress"], fill='toself', name=f'Tipo {int(row["Cluster"])}'
                ))
            fig2.update_layout(title="Identikit del Lavoro (Radar)", polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            st.plotly_chart(fig2, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Definisce 'chi è' ogni gruppo. Un triangolo molto ampio indica la sessione che spreme "
                    "l'atleta al massimo su cuore e stress metabolico insieme.")

        with g3:
            fig3 = px.box(df, x="Cluster", y="SMA", color="Cluster", color_discrete_sequence=list(CLUSTER_COLORS.values()),
                           title="Stress Generato per Tipologia")
            st.plotly_chart(fig3, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "Controllo di coerenza: la sessione chiamata 'scarico' produce davvero poco stress, o l'atleta "
                    "si sta affaticando anche quando non dovrebbe?")

        with g4:
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Tipologia', 'N° Sessioni']
            fig4 = px.bar(cluster_counts, x='Tipologia', y='N° Sessioni', color='Tipologia',
                           color_discrete_sequence=list(CLUSTER_COLORS.values()), title="Bilancio Stagionale")
            st.plotly_chart(fig4, use_container_width=True)
            insight("Cosa significa per il Coach",
                    "La regola dell'80/20: quanti allenamenti 'distruttivi' si fanno rispetto a quelli lenti e "
                    "rigeneranti? Mostra il reale bilanciamento del volume stagionale.")

    # ================================================================ TAB 6: SIMULATORE
    with tabs[5]:
        st.markdown("<h3 style='color:white;'>Tavolo di Controllo del Coach (Pre-Allenamento)</h3>", unsafe_allow_html=True)

        sc1, sc2 = st.columns([1, 2])

        with sc1:
            st.markdown("<h4 style='color:#00e5ff; font-size:1.05rem;'>Valori per la Sessione Odierna</h4>", unsafe_allow_html=True)
            s_dist = st.slider("Chilometri Previsti", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Obiettivo (RPE)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno Atleta", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("Battiti Stimati (BPM)", 100, 190, 150)
            s_temp = st.slider("Gradi Esterni (°C)", 0, 40, 25)

            s_tempo = s_dist * 4.5 + (s_rpe * 2)
            s_lavoro = s_tempo / 60
            s_sma = (s_lavoro * s_rpe) / s_sonno
            s_islr = (s_lavoro * s_rpe) / s_dist
            s_idet = (s_fc * s_temp) / ((s_dist * 1000) / (s_tempo * 60))
            s_iitr = (s_temp * 10) / s_dist

            input_data = pd.DataFrame([[s_dist, s_sonno, s_sma, s_islr, s_idet, s_iitr, s_rpe]], columns=RF_FEATURES)
            prob = rf.predict_proba(input_data)[0][1] * 100
            label, color = risk_band(prob)

            s_carico = s_tempo * s_rpe
            acwr_simulato = (df["Carico Acuto (7g)"].iloc[-1] * 6 + s_carico) / 7 / df["Carico Cronico (28g)"].iloc[-1]
            st.metric("ACWR previsto dopo questa sessione", f"{acwr_simulato:.2f}")

        with sc2:
            st.markdown("<h4 style='color:#ffffff; font-size:1.05rem;'>Decisione Intelligenza Artificiale</h4>", unsafe_allow_html=True)

            c_gauge, c_radar = st.columns(2)

            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob, number={'suffix': "%", 'font': {'color': color}},
                    title={'text': f"Stato: {label}", 'font': {'color': COLORS['text_soft']}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['border']},
                        'bar': {'color': color},
                        'bgcolor': COLORS['surface_2'],
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(163, 230, 53, 0.1)'},
                            {'range': [40, 70], 'color': 'rgba(251, 191, 36, 0.1)'},
                            {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.1)'}],
                    }))
                fig_g.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_g, use_container_width=True)

            with c_radar:
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()

                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"],
                                                 fill='toself', name='Media Squadra', line_color=COLORS['text_soft']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"],
                                                 fill='toself', name='Simulazione Oggi', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=300,
                                     margin=dict(l=20, r=20, t=20, b=20), legend=dict(y=-0.2))
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown(f"""
        <div class="runai-info-box" style="margin-top: 10px;">
            <strong>Istruzioni del Simulatore per il Coach:</strong><br><br>
            <strong>1. Il Tachimetro:</strong> è il semaforo dell'IA. Se imposti 15km ma l'atleta ha dormito solo 4 ore,
            l'ago schizza nel rosso (>70%). Abbassa i cursori a sinistra (riduci RPE o Distanza) finché l'ago non torna
            in zona verde.<br><br>
            <strong>2. Il Radar (Identikit):</strong> compara il carico di oggi con quello che l'atleta sopporta di
            solito. Se una punta esce molto dal grigio, stai somministrando uno stimolo insolito per il suo corpo.<br><br>
            <strong>3. ACWR previsto:</strong> stima come cambierebbe il rapporto tra carico acuto e cronico se questa
            sessione venisse svolta oggi — un secondo parere, indipendente dal modello ML, basato sulla scienza del carico.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_ui()
