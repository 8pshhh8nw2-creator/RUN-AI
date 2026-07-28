"""
Advanced Machine Learning Suite - Dashboard interattiva per tesi magistrale.
File unico. Avvio: streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

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
    initial_sidebar_state="collapsed"
)

# ============================================================================
# COSTANTI E COLORI (STILE IMMAGINE RIFERIMENTO)
# ============================================================================
COLORS = {
    "bg": "#070b12", "surface": "#0f172a", "surface_2": "#1e293b",
    "border": "#1e293b", "border_accent": "#334155",
    "text": "#f8fafc", "text_soft": "#94a3b8", "muted": "#64748b",
    "cyan": "#22d3ee", "cyan_dim": "rgba(34, 211, 238, 0.08)",
    "green": "#22c55e", "amber": "#fbbf24", "red": "#f87171", "purple": "#a78bfa"
}
QUALITATIVE = [COLORS['cyan'], COLORS['purple'], COLORS['amber'], COLORS['green'], COLORS['red']]
CLUSTER_COLORS = {0: COLORS['cyan'], 1: COLORS['purple'], 2: COLORS['amber'], 3: COLORS['green']}

TARGET = "Rischio Overload"
TIME_TARGET = "Tempo (min)"
RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR", "RPE"]
CLUSTER_FEATURES = ["FC Media", "ISLR", "SMA"]

RISK_BANDS = ((40.0, "Basso (Sicuro)", COLORS['green']), (70.0, "Moderato (Attenzione)", COLORS['amber']), (101.0, "Alto (Pericolo)", COLORS['red']))

def risk_band(prob: float) -> tuple[str, str]:
    for thr, label, color in RISK_BANDS:
        if prob < thr:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]

@dataclass(frozen=True)
class Settings:
    seed: int = 42
    test_size: float = 0.25
    n_estimators: int = 200
    max_depth: int = 8
    n_clusters: int = 3
    n_sessions: int = 1000

# ============================================================================
# THEME & CSS (IDENTICO ALLA TUA FOTO)
# ============================================================================
PLOTLY_TEMPLATE = "runai_exact"

def register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS['text_soft'], size=12),
        title=dict(font=dict(size=15, color=COLORS['text'], family="Inter"), x=0.01, xanchor="left", y=0.95),
        margin=dict(t=40, l=10, r=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, linecolor=COLORS['border_accent']),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, linecolor=COLORS['border_accent']),
    )
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {COLORS['bg']};
        color: {COLORS['text']};
    }}
    
    .stApp {{ background-color: {COLORS['bg']}; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    
    /* Hero Card identica alla foto (bordo superiore verde acido) */
    .hero-card {{
        background: {COLORS['surface']};
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border-top: 3px solid {COLORS['green']};
        border-left: 1px solid {COLORS['border']};
        border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']};
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }}
    
    .hero-kicker {{
        color: {COLORS['cyan']};
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}
    
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0 0 0.8rem 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }}
    
    /* Box informativo azzurro laterale */
    .info-box {{
        background: {COLORS['cyan_dim']};
        border-left: 4px solid {COLORS['cyan']};
        border-radius: 4px 8px 8px 4px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
        color: {COLORS['text_soft']};
        font-size: 0.95rem;
        line-height: 1.6;
        border-top: 1px solid rgba(34, 211, 238, 0.1);
        border-right: 1px solid rgba(34, 211, 238, 0.1);
        border-bottom: 1px solid rgba(34, 211, 238, 0.1);
    }}
    .info-box strong {{ color: #ffffff; }}
    
    /* Menu Orizzontale Tech pulito */
    div.row-widget.stRadio > div {{
        display: flex; gap: 0; background: {COLORS['surface']}; 
        border: 1px solid {COLORS['border']}; border-radius: 8px; overflow: hidden;
    }}
    div.row-widget.stRadio > div > label {{
        padding: 12px 20px; border-right: 1px solid {COLORS['border']}; cursor: pointer;
    }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{
        background: {COLORS['cyan_dim']};
        box-shadow: inset 0 -3px 0 {COLORS['cyan']};
    }}
    div.row-widget.stRadio p {{
        font-weight: 700; font-size: 0.8rem; text-transform: uppercase; margin: 0; color: {COLORS['text']};
    }}
    
    .section-title {{
        font-size: 1.5rem; font-weight: 800; margin-top: 1.2rem; margin-bottom: 0.4rem; color: #ffffff;
    }}
    
    /* Box spiegazione sotto i grafici */
    .coach-insight {{
        background: {COLORS['surface_2']};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border: 1px solid {COLORS['border_accent']};
        font-size: 0.88rem;
        color: {COLORS['text_soft']};
        margin-top: -5px;
        margin-bottom: 15px;
        line-height: 1.5;
    }}
    .coach-insight span {{ color: {COLORS['cyan']}; font-weight: 800; text-transform: uppercase; font-size: 0.72rem; display: block; margin-bottom: 4px; letter-spacing: 0.05em; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATI E MODELLI
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
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed)
    df["Cluster"] = kmeans.fit_predict(StandardScaler().fit_transform(X_clust))
    
    return rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_test_r, y_test_r

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================
def render_ui():
    register_plotly_template()
    inject_css()
    cfg = Settings()
    
    # Hero Card stile foto esatto
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-kicker">● MODULO ANALITICO 01 · SPORT DATA SCIENCE</div>
        <h1 class="hero-title">SPORT MACHINE LEARNING<br>& INJURY PREDICTION</h1>
        <p style="color: {COLORS['text_soft']}; font-size: 1rem; line-height: 1.6; max-width: 850px; margin: 0;">
        Piattaforma di supporto decisionale per il coaching. Analizza il carico interno ed esterno per stimare la perdita di performance e prevenire il rischio di sovraccarico (Overload) prima della sessione.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_test_r, y_test_r = train_models(df, cfg)

    # Menu Orizzontale Tech
    sezioni = [
        "1. STIMA PERFORMANCE", 
        "2. RISCHIO BASE", 
        "3. RANDOM FOREST (AI)", 
        "4. PROFILAZIONE", 
        "5. SIMULATORE"
    ]
    scelta = st.radio("Seleziona Modulo", sezioni, horizontal=True, label_visibility="collapsed")

    # ==============================================================
    # 1. REGRESSIONE LINEARE
    # ==============================================================
    if scelta == sezioni[0]:
        st.markdown("<div class='section-title'>Stima della Performance e del Ritmo</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Spiegazione per il Coach:</strong> Questo modulo calcola l'impatto della stanchezza accumulata sul tempo finale dell'atleta. Utilizzando la distanza, lo stress metabolico e la percezione dello sforzo (RPE), l'algoritmo stima il tempo di completamento per evidenziare eventuali cali di rendimento anomali.
        </div>
        """, unsafe_allow_html=True)
        
        preds = reg.predict(X_test_r)
        
        c1, c2 = st.columns(2)
        c1.metric("Accuratezza Previsione (R²)", f"{r2_score(y_test_r, preds)*100:.1f}%")
        c2.metric("Errore Medio Cronometrico", f"{mean_absolute_error(y_test_r, preds):.1f} min")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter(x=y_test_r, y=preds, opacity=0.6, color_discrete_sequence=[COLORS['cyan']], title="Tempo Reale vs Previsto")
            fig1.add_shape(type="line", x0=y_test_r.min(), y0=y_test_r.min(), x1=y_test_r.max(), y1=y_test_r.max(), line=dict(dash="dash", color=COLORS['muted']))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>I punti allineati sulla diagonale mostrano una previsione perfetta. Se un punto si discosta molto, significa che l'atleta ha reso diversamente dal previsto a causa di fattori esterni.</div>", unsafe_allow_html=True)

        with g2:
            residui = y_test_r - preds
            fig2 = px.histogram(residui, nbins=30, color_discrete_sequence=[COLORS['purple']], title="Distribuzione degli Scostamenti")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Evidenzia la regolarità degli errori. Una curva concentrata sullo zero indica stabilità; code larghe segnalano sessioni in cui l'atleta è crollato fisicamente.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.scatter(x=preds, y=residui, opacity=0.6, color_discrete_sequence=[COLORS['amber']], title="Errore in base alla Durata")
            fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['muted'])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Verifica se il modello perde precisione sulle lunghe distanze. Se l'errore cresce linearmente, la resistenza aerobica dell'atleta non è modellata linearmente.</div>", unsafe_allow_html=True)

        with g4:
            coefs = pd.DataFrame({"Fattore": X_test_r.columns, "Impatto": reg.coef_}).sort_values("Impatto")
            fig4 = px.bar(coefs, x="Impatto", y="Fattore", orientation="h", color_discrete_sequence=[COLORS['green']], title="Peso dei Fattori sul Ritmo")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Quantifica l'impatto reale di ogni singola unità di fatica (RPE o Stress) sui minuti persi lungo il percorso.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 2. REGRESSIONE LOGISTICA
    # ==============================================================
    elif scelta == sezioni[1]:
        st.markdown("<div class='section-title'>Classificazione Lineare del Rischio (Baseline)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Spiegazione per il Coach:</strong> Modello di riferimento statistico semplice. Imposta una soglia lineare per dividere gli allenamenti sicuri da quelli a rischio infortunio (Overload), valutando il peso diretto di ogni singola variabile.
        </div>
        """, unsafe_allow_html=True)

        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuratezza", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Capacità Intercetta Rischio (AUC)", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Affidabilità Allarme", f"{precision_score(y_test, y_pred)*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Curva ROC (Efficacia Modello)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Misura la capacità di distinguere il pericolo reale dai falsi allarmi. Più la curva sale verso l'angolo in alto a sinistra, migliore è il modello.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predizione IA", y="Realtà"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Matrice degli Errori")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>In basso a sinistra trovi i falsi negativi: sessioni pericolose classificate per errore come sicure. Vanno ridotti al minimo.</div>", unsafe_allow_html=True)

        with g3:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig3 = px.line(x=rec, y=prec, title="Curva Precision-Recall", color_discrete_sequence=[COLORS['purple']])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Mostra come decade l'affidabilità dell'avviso quando si pretende di intercettare il 100% degli eventi di sovraccarico.</div>", unsafe_allow_html=True)

        with g4:
            coef_df = pd.DataFrame({"Metrica": RF_FEATURES, "Impatto Logistico": lr.coef_[0]}).sort_values("Impatto Logistico")
            fig4 = px.bar(coef_df, x="Impatto Logistico", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['red']], title="Coefficienti di Rischio")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>I valori positivi indicano i fattori che spingono verso l'overload; quelli negativi (come il sonno) proteggono l'atleta.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 3. RANDOM FOREST
    # ==============================================================
    elif scelta == sezioni[2]:
        st.markdown("<div class='section-title'>Previsione Avanzata con Intelligenza Artificiale</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Spiegazione per il Coach:</strong> Modello basato su alberi decisionali multipli in grado di cogliere dinamiche complesse e non lineari. Capisce che determinati carichi di lavoro diventano rischiosi solo se combinati con fattori specifici, come una carenza cronica di sonno.
        </div>
        """, unsafe_allow_html=True)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuratezza", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("AUC-ROC Avanzato", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Affidabilità Allarme", f"{precision_score(y_test, y_pred)*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Curva ROC (Random Forest)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Se questa curva è superiore a quella del modello base, dimostra che il rischio di sovraccarico nello sport segue logiche complesse non lineari.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purp", labels=dict(x="Predizione IA", y="Realtà"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Matrice di Confusione Ottimizzata")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Evidenzia una netta riduzione degli errori di classificazione rispetto al modello precedente grazie all'analisi combinata delle variabili.</div>", unsafe_allow_html=True)

        with g3:
            imp = pd.DataFrame({"Metrica": RF_FEATURES, "Peso Informativo": rf.feature_importances_}).sort_values("Peso Informativo")
            fig3 = px.bar(imp, x="Peso Informativo", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['amber']], title="Importanza delle Variabili (Gini)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Mostra quali metriche l'IA sfrutta di più per decidere se bloccare o approvare la sessione di allenamento.</div>", unsafe_allow_html=True)

        with g4:
            df_prob = pd.DataFrame({"Probabilità %": y_prob * 100, "Stato Reale": ["Overload" if y == 1 else "Sicuro" for y in y_test]})
            fig4 = px.histogram(df_prob, x="Probabilità %", color="Stato Reale", barmode="overlay", nbins=40, color_discrete_sequence=[COLORS['red'], COLORS['green']], title="Distribuzione delle Certezze")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>I picchi separati agli estremi (0% e 100%) indicano che l'algoritmo prende decisioni nette e sicure, riducendo le zone d'ombra.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 4. K-MEANS CLUSTERING
    # ==============================================================
    elif scelta == sezioni[3]:
        st.markdown("<div class='section-title'>Profilazione Latente delle Sessioni</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Spiegazione per il Coach:</strong> Algoritmo non supervisionato che raggruppa automaticamente gli allenamenti in base alle risposte fisiologiche e di carico, senza etichette predefinite. Aiuta a scoprire i veri 'profili di stress' ricorrenti nella stagione.
        </div>
        """, unsafe_allow_html=True)

        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Coerenza dei Gruppi (Silhouette Score)", f"{sil_score*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", color_continuous_scale=list(CLUSTER_COLORS.values()), title="Spazio Latente Tridimensionale")
            fig1.update_layout(scene=dict(xaxis_title="FC Media", yaxis_title="Indice Lavoro", zaxis_title="Stress"), margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Ogni cluster di colore diverso rappresenta una tipologia distinta di sessione allenante identificata autonomamente dall'algoritmo.</div>", unsafe_allow_html=True)

        with g2:
            centroids = df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
            fig2 = go.Figure()
            for i, row in centroids.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row["FC Media"]/df["FC Media"].max(), row["ISLR"]/df["ISLR"].max(), row["SMA"]/df["SMA"].max()],
                    theta=["FC Media", "Indice Lavoro", "Stress"], fill='toself', name=f'Profilo {int(row["Cluster"])}'
                ))
            fig2.update_layout(title="Identikit dei Gruppi (Radar)", polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Tratteggia la forma caratteristica di ogni tipologia di allenamento (es. recupero vs carico massimale).</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.box(df, x="Cluster", y="SMA", color="Cluster", color_discrete_sequence=list(CLUSTER_COLORS.values()), title="Distribuzione dello Stress per Profilo")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Conferma se il livello di stress metabolico è coerente con la tipologia di gruppo assegnata dall'algoritmo.</div>", unsafe_allow_html=True)

        with g4:
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Profilo', 'Sessioni']
            fig4 = px.bar(cluster_counts, x='Profilo', y='Sessioni', color='Profilo', color_continuous_scale=list(CLUSTER_COLORS.values()), title="Volume per Tipologia di Lavoro")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Interpretazione Grafico</span>Mostra la frequenza con cui l'atleta affronta ciascuna tipologia di allenamento all'interno del macrociclo.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 5. SIMULATORE WHAT-IF
    # ==============================================================
    elif scelta == sezioni[4]:
        st.markdown("<div class='section-title'>Simulatore Predittivo Pre-Sessione</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Spiegazione per il Coach:</strong> Imposta i parametri previsti per l'allenamento di oggi. Il sistema calcola in tempo reale le metriche derivate e interroga l'Intelligenza Artificiale per stimare il rischio di sovraccarico prima di mandare in campo l'atleta.
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        
        with sc1:
            st.markdown("<h4 style='color: #22d3ee; font-size: 1.1rem; margin-bottom: 1rem;'>Configurazione Carico Odierno</h4>", unsafe_allow_html=True)
            s_dist = st.slider("Distanza Prevista (km)", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Percepita (RPE 1-10)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno Stanotte", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("FC Media Stimata (bpm)", 100, 190, 150)
            s_temp = st.slider("Temperatura Esterna (°C)", 0, 40, 25)
            
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
            st.markdown("<h4 style='color: #ffffff; font-size: 1.1rem; margin-bottom: 1rem;'>Verdetto dell'Intelligenza Artificiale</h4>", unsafe_allow_html=True)
            
            c_gauge, c_radar = st.columns([1, 1])
            
            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob, number={'suffix': "%", 'font': {'color': color}}, title = {'text': f"Stato: {label}", 'font': {'color': COLORS['text_soft']}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['border_accent']},
                        'bar': {'color': color},
                        'bgcolor': COLORS['surface'],
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.15)'},
                            {'range': [40, 70], 'color': 'rgba(251, 191, 36, 0.15)'},
                            {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.15)'}],
                    }))
                fig_g.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
            
            with c_radar:
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"], fill='toself', name='Media Storica', line_color=COLORS['muted']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"], fill='toself', name='Simulazione', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=320, margin=dict(l=20, r=20, t=20, b=20), legend=dict(y=-0.2))
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown(f"""
        <div class="coach-insight" style="margin-top: 10px;">
            <span>Guida all'utilizzo del Simulatore</span>
            <strong>Tachimetro a Sinistra:</strong> Indica la probabilità stimata di sovraccarico. Se entra in area rossa (>70%), l'IA consiglia di rimodulare i parametri (es. riducendo i km o aumentando le ore di sonno nei giorni precedenti).<br><br>
            <strong>Grafico Radar a Destra:</strong> Mette a confronto la sessione simulata (in colore) con la media storica dell'atleta (in grigio). Evidenzia immediatamente se stai chiedendo un carico sproporzionato rispetto alle sue abitudini metaboliche.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_ui()
