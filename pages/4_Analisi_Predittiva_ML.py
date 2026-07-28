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
# COSTANTI E COLORI (STILE IMMAGINE)
# ============================================================================
COLORS = {
    "bg": "#0a0e17", "bg2": "#111827", "surface": "#162032", "surface_2": "#1e293b",
    "border": "#334155", "border_soft": "rgba(51, 65, 85, 0.5)",
    "text": "#f8fafc", "text_soft": "#cbd5e1", "muted": "#94a3b8",
    "cyan": "#00e5ff", "cyan_dim": "rgba(0, 229, 255, 0.1)",
    "green": "#a3e635", "amber": "#fbbf24", "red": "#f87171", "purple": "#a78bfa"
}
QUALITATIVE = [COLORS['cyan'], COLORS['purple'], COLORS['amber'], COLORS['green'], COLORS['red'], COLORS['text']]
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
# CSS & THEME PLOTLY
# ============================================================================
PLOTLY_TEMPLATE = "runai_tech"

def register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS['text_soft'], size=13),
        title=dict(font=dict(size=16, color=COLORS['text'], family="Inter"), x=0.01, xanchor="left", y=0.96),
        margin=dict(t=50, l=10, r=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border_soft'], zeroline=False, linecolor=COLORS['border'], ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border_soft'], zeroline=False, linecolor=COLORS['border']),
    )
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    
    :root {{
        --bg: {COLORS['bg']}; --surface: {COLORS['surface']}; --cyan: {COLORS['cyan']};
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: var(--bg);
        color: {COLORS['text']};
    }}
    
    /* Sfondo generale scuro simile all'immagine */
    .stApp {{ background-color: var(--bg); }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    
    /* Stile Hero Card (il riquadro principale con riga verde sopra) */
    .hero-card {{
        background: {COLORS['surface']};
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border-top: 3px solid {COLORS['green']};
        border-left: 1px solid {COLORS['border']};
        border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']};
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    
    .hero-kicker {{
        color: {COLORS['cyan']};
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    
    .hero-title {{
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0 0 1rem 0;
        color: #ffffff;
        line-height: 1.1;
    }}
    
    /* Info Box (Riquadro azzurro laterale) */
    .info-box {{
        background: {COLORS['cyan_dim']};
        border-left: 4px solid {COLORS['cyan']};
        border-radius: 4px 8px 8px 4px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: {COLORS['text_soft']};
        font-size: 1rem;
        line-height: 1.6;
    }}
    
    .info-box strong {{ color: #ffffff; }}
    
    /* Menu Orizzontale Tech */
    div.row-widget.stRadio > div {{
        display: flex; gap: 0; background: {COLORS['surface']}; 
        border: 1px solid {COLORS['border']}; border-radius: 8px; overflow: hidden;
    }}
    div.row-widget.stRadio > div > label {{
        padding: 12px 24px; border-right: 1px solid {COLORS['border']}; cursor: pointer;
    }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{
        background: {COLORS['cyan_dim']};
        box-shadow: inset 0 -3px 0 {COLORS['cyan']};
    }}
    div.row-widget.stRadio p {{
        font-weight: 800; font-size: 0.85rem; text-transform: uppercase; margin: 0; color: {COLORS['text']};
    }}
    
    /* Titoli Sezioni */
    .section-title {{
        font-size: 1.8rem; font-weight: 800; margin-top: 1.5rem; margin-bottom: 0.5rem; color: #ffffff;
    }}
    
    /* Box Spiegazioni sotto i grafici */
    .coach-insight {{
        background: {COLORS['surface_2']};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        font-size: 0.9rem;
        color: {COLORS['text_soft']};
        margin-top: -10px;
        margin-bottom: 20px;
    }}
    .coach-insight span {{ color: {COLORS['cyan']}; font-weight: 800; text-transform: uppercase; font-size: 0.75rem; display: block; margin-bottom: 4px; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA GENERATOR (DA SOSTITUIRE IN FUTURO CON df = pd.read_csv("dati.csv"))
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
# UI RENDERING
# ============================================================================
def render_ui():
    register_plotly_template()
    inject_css()
    cfg = Settings()
    
    # ---------------------------------------------------------
    # HEADER (STILE IMMAGINE)
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-kicker">● PERFORMANCE INTELLIGENCE SYSTEM</div>
        <h1 class="hero-title">SPORT MACHINE LEARNING<br>& INJURY PREDICTION</h1>
        <p style="color: {COLORS['text_soft']}; font-size: 1.1rem; max-width: 800px;">
        Questa piattaforma analizza i dati di allenamento per prevedere il rischio di sovraccarico (Overload) e valutare il calo della performance. L'Intelligenza Artificiale impara dallo storico dell'atleta per fornire indicazioni pratiche al coach.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_test_r, y_test_r = train_models(df, cfg)

    # ---------------------------------------------------------
    # MENU ORIZZONTALE
    # ---------------------------------------------------------
    sezioni = [
        "1. STIMA PERFORMANCE", 
        "2. RISCHIO BASE (LOGISTICA)", 
        "3. RISCHIO AVANZATO (AI)", 
        "4. PROFILAZIONE ATLETA", 
        "5. SIMULATORE COACH"
    ]
    scelta = st.radio("Seleziona Analisi", sezioni, horizontal=True, label_visibility="collapsed")

    # ==============================================================
    # 1. REGRESSIONE LINEARE (STIMA PERFORMANCE)
    # ==============================================================
    if scelta == sezioni[0]:
        st.markdown("<div class='section-title'>Analisi del Ritmo e della Fatica</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Come ragiona l'algoritmo:</strong> Immagina questo modello come un cronometro intelligente. Non guarda solo i chilometri da percorrere, ma capisce quanto la stanchezza percepita (RPE) e lo stress fisico accumulato andranno a rallentare l'atleta. Risponde alla domanda del coach: <i>"Se il mio atleta è stanco oggi, quanto tempo in più ci metterà a finire il lavoro?"</i>
        </div>
        """, unsafe_allow_html=True)
        
        preds = reg.predict(X_test_r)
        
        # Risultati Chiari
        st.markdown("### Risultati per il Coach")
        c1, c2 = st.columns(2)
        c1.metric("Affidabilità Previsione (R²)", f"{r2_score(y_test_r, preds)*100:.1f}%", "Più è alto, più il modello ci prende")
        c2.metric("Errore Medio Cronometro (MAE)", f"{mean_absolute_error(y_test_r, preds):.1f} minuti", "Di quanto sbaglia in media")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter(x=y_test_r, y=preds, opacity=0.6, color_discrete_sequence=[COLORS['cyan']], title="Tempo Reale vs Tempo Previsto")
            fig1.add_shape(type="line", x0=y_test_r.min(), y0=y_test_r.min(), x1=y_test_r.max(), y1=y_test_r.max(), line=dict(dash="dash", color=COLORS['text_soft']))
            fig1.update_layout(xaxis_title="Tempo Reale (Cosa è successo)", yaxis_title="Tempo Previsto (Cosa diceva l'IA)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Se i punti sono vicini alla linea tratteggiata, l'algoritmo ha previsto correttamente il tempo di gara/allenamento in base alla stanchezza dell'atleta.</div>", unsafe_allow_html=True)

        with g2:
            residui = y_test_r - preds
            fig2 = px.histogram(residui, nbins=30, color_discrete_sequence=[COLORS['purple']], title="Frequenza degli Errori di Previsione")
            fig2.update_layout(xaxis_title="Errore in Minuti", yaxis_title="Numero di Allenamenti")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>La maggior parte delle barre dovrebbe essere al centro (zero errori). Se ci sono barre a destra o sinistra, significa che a volte l'atleta crolla improvvisamente in modo imprevedibile.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.scatter(x=preds, y=residui, opacity=0.6, color_discrete_sequence=[COLORS['amber']], title="Stabilità su Lunghe Distanze")
            fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['text_soft'])
            fig3.update_layout(xaxis_title="Tempo Previsto", yaxis_title="Errore di Previsione")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Se l'errore (asse verticale) diventa enorme quando il tempo previsto (asse orizzontale) è alto, significa che l'atleta non è prevedibile nelle sessioni molto lunghe (mancanza di resistenza di base).</div>", unsafe_allow_html=True)

        with g4:
            coefs = pd.DataFrame({"Fattore": X_test_r.columns, "Impatto sul Tempo": reg.coef_}).sort_values("Impatto sul Tempo")
            fig4 = px.bar(coefs, x="Impatto sul Tempo", y="Fattore", orientation="h", color_discrete_sequence=[COLORS['green']], title="Cosa fa rallentare l'atleta?")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Mostra i 'pesi'. Ovviamente la Distanza aumenta il tempo, ma guarda le altre barre: ti dice esattamente quanti minuti aggiunge al cronometro un livello di RPE o di Stress elevato.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 2. REGRESSIONE LOGISTICA (RISCHIO BASE)
    # ==============================================================
    elif scelta == sezioni[1]:
        st.markdown("<div class='section-title'>Rischio Overload: Modello Semplice</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Come ragiona l'algoritmo:</strong> Questo è il modello 'semaforo' classico. Traccia una linea dritta: da una parte gli allenamenti sicuri, dall'altra quelli a rischio infortunio (Overload). È utile per capire in modo diretto quali fattori (es. mancanza di sonno) spingono l'atleta oltre il limite in modo matematico e proporzionale.
        </div>
        """, unsafe_allow_html=True)

        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        st.markdown("### Risultati per il Coach")
        c1, c2, c3 = st.columns(3)
        c1.metric("Allenamenti azzeccati", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Sensibilità agli Infortuni", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Falsi Allarmi (100-Precision)", f"{(1-precision_score(y_test, y_pred))*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Curva di Efficacia (ROC)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['text_soft']), x0=0, x1=1, y0=0, y1=1)
            fig1.update_layout(xaxis_title="Falsi Allarmi", yaxis_title="Infortuni Intercettati")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Più la curva blu è 'gonfia' verso l'angolo in alto a sinistra, più il sistema è bravo a trovare gli allenamenti pericolosi senza bloccare l'atleta inutilmente per falsi allarmi.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Cosa dice l'IA", y="Cos'è successo davvero"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Contatore degli Errori")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>In basso a sinistra vedi l'incubo di ogni preparatore: gli allenamenti che l'IA reputava 'Sicuri' ma che in realtà erano 'Overload'. Il nostro scopo è mantenere quel numero a zero.</div>", unsafe_allow_html=True)

        with g3:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig3 = px.line(x=rec, y=prec, title="Qualità degli Allarmi", color_discrete_sequence=[COLORS['purple']])
            fig3.update_layout(xaxis_title="Percentuale di Pericoli Trovati", yaxis_title="Affidabilità dell'Allarme")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Se cerchiamo di intercettare il 100% dei rischi (andando verso destra), quanto diventano inaffidabili i nostri allarmi? Se la linea crolla, significa che stiamo fermando l'atleta troppo spesso per niente.</div>", unsafe_allow_html=True)

        with g4:
            coef_df = pd.DataFrame({"Metrica": RF_FEATURES, "Impatto sul Rischio": lr.coef_[0]}).sort_values("Impatto sul Rischio")
            fig4 = px.bar(coef_df, x="Impatto sul Rischio", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['red']], title="I colpevoli del Sovraccarico")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Barre verso destra (es. Stress, RPE) alzano il rischio di infortunio. Barre verso sinistra (es. Ore Sonno) abbassano il rischio, funzionando come scudo protettivo per l'atleta.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 3. RANDOM FOREST (RISCHIO AVANZATO)
    # ==============================================================
    elif scelta == sezioni[2]:
        st.markdown("<div class='section-title'>Intelligenza Artificiale: Random Forest</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Come ragiona l'algoritmo:</strong> Pensa a un team di 200 allenatori (gli 'alberi'). Ognuno guarda l'atleta da un punto di vista diverso. Questo sistema capisce le 'situazioni complesse' tipiche dello sport. Ad esempio: l'IA capisce che dormire 5 ore non è per forza un rischio se oggi fai scarico, ma diventa un rischio critico se oggi fai le ripetute.
        </div>
        """, unsafe_allow_html=True)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        st.markdown("### Risultati per il Coach")
        c1, c2, c3 = st.columns(3)
        c1.metric("Allenamenti azzeccati", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Sensibilità (rispetto al base)", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Falsi Allarmi", f"{(1-precision_score(y_test, y_pred))*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Efficacia Avanzata (ROC)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['text_soft']), x0=0, x1=1, y0=0, y1=1)
            fig1.update_layout(xaxis_title="Falsi Allarmi", yaxis_title="Infortuni Intercettati")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Confronta questa curva con quella del modello precedente. Se questa è più alta, conferma che la stanchezza umana non è lineare, ma piena di sfumature che solo l'IA complessa può leggere.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purp", labels=dict(x="Cosa dice l'IA", y="Cos'è successo davvero"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Contatore degli Errori (Migliorato)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>I numeri fuori dalla diagonale principale dovrebbero essere molto più bassi qui. Significa che l'Intelligenza Artificiale ha 'salvato' atleti che il modello base non aveva notato.</div>", unsafe_allow_html=True)

        with g3:
            imp = pd.DataFrame({"Metrica": RF_FEATURES, "Importanza": rf.feature_importances_}).sort_values("Importanza")
            fig3 = px.bar(imp, x="Importanza", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['amber']], title="Classifica dei Segnali d'Allarme")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Ti dice a cosa l'IA fa più attenzione per decidere se bloccare l'atleta. Spesso, l'interazione tra Carico Interno (Stress/Battiti) ed Esterno domina questa classifica.</div>", unsafe_allow_html=True)

        with g4:
            df_prob = pd.DataFrame({"Probabilità di Rischio %": y_prob * 100, "Stato Reale": ["Overload" if y == 1 else "Sicuro" for y in y_test]})
            fig4 = px.histogram(df_prob, x="Probabilità di Rischio %", color="Stato Reale", barmode="overlay", nbins=40, color_discrete_sequence=[COLORS['red'], COLORS['green']], title="Sicurezza Decisionale")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Un'IA perfetta avrà i grafici verdi tutti schiacciati a sinistra (0% rischio) e quelli rossi tutti a destra (100% rischio). La zona centrale (50%) rappresenta i dubbi dell'algoritmo.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 4. K-MEANS CLUSTERING (PROFILAZIONE)
    # ==============================================================
    elif scelta == sezioni[3]:
        st.markdown("<div class='section-title'>Classificazione Automatica Allenamenti</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Come ragiona l'algoritmo:</strong> Questo algoritmo lavora senza etichette. Guarda tutti gli allenamenti della stagione e li divide automaticamente in 'cassetti' o tipologie simili. Il coach scopre così come il corpo dell'atleta raggruppa fisiologicamente gli sforzi: magari scopre il cassetto 'Lavori di recupero', quello 'Alta intensità' e quello 'Volume estremo'.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Risultati per il Coach")
        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Qualità della Separazione (Silhouette)", f"{sil_score*100:.1f}%", "Oltre il 50% significa che le tipologie di allenamento sono ben distinte")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", color_continuous_scale=list(CLUSTER_COLORS.values()), title="Mappa 3D della Fatica")
            fig1.update_layout(scene=dict(xaxis_title="Battiti", yaxis_title="Indice Lavoro", zaxis_title="Stress"), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Ogni punto è una sessione. I colori rappresentano le diverse tipologie di impatto sul corpo. Più i 'nuvoloni' colorati sono separati, più i tuoi stimoli allenanti sono diversificati.</div>", unsafe_allow_html=True)

        with g2:
            centroids = df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
            fig2 = go.Figure()
            for i, row in centroids.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row["FC Media"]/df["FC Media"].max(), row["ISLR"]/df["ISLR"].max(), row["SMA"]/df["SMA"].max()],
                    theta=["Battiti", "Indice di Lavoro", "Stress Metabolico"], fill='toself', name=f'Tipologia {int(row["Cluster"])}'
                ))
            fig2.update_layout(title="L'Identikit dei Gruppi (Radar)", polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Ti mostra 'chi è' ogni gruppo. Un poligono enorme su Stress e Battiti identifica la tipologia delle sessioni massimali (le più logoranti per l'atleta).</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.box(df, x="Cluster", y="SMA", color="Cluster", color_discrete_sequence=list(CLUSTER_COLORS.values()), title="Stress Generato per Tipologia")
            fig3.update_layout(xaxis_title="Tipologia Allenamento (Cluster)", yaxis_title="Livello di Stress (SMA)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Verifica se c'è coerenza. Se il gruppo dello scarico attivo ha picchi di stress alti come quello delle ripetute, l'atleta sta sbagliando i ritmi di recupero.</div>", unsafe_allow_html=True)

        with g4:
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Tipologia', 'Numero di Sessioni']
            fig4 = px.bar(cluster_counts, x='Tipologia', y='Numero di Sessioni', color='Tipologia', color_continuous_scale=list(CLUSTER_COLORS.values()), title="Bilancio del Macrociclo")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa guardare</span>Mostra la distribuzione reale del carico. Se hai 80% di sessioni nella tipologia 'Stress Estremo', stai per infortunare l'atleta a causa di una cattiva periodizzazione.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 5. SIMULATORE WHAT-IF
    # ==============================================================
    elif scelta == sezioni[4]:
        st.markdown("<div class='section-title'>Pianificazione Predittiva (Pre-Sessione)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Come usare il Simulatore:</strong> Inserisci i dati dell'allenamento che l'atleta deve svolgere <i>oggi</i>. Il sistema calcola sul momento le metriche biologiche e interroga l'AI (Random Forest) per darti una risposta immediata: "Mandarlo a correre 15km oggi, sapendo che ha dormito male, lo metterà in pericolo?". Se il tachimetro entra in zona rossa, abbassa i km o l'intensità direttamente qui per trovare un compromesso sicuro.
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        
        # COLONNA SINISTRA: INPUT
        with sc1:
            st.markdown("<h4 style='color: #00e5ff;'>Imposta Allenamento Odierno</h4>", unsafe_allow_html=True)
            s_dist = st.slider("Km da percorrere", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Obiettivo (RPE 1-10)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno recuperate stanotte", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("Battiti Medi Previsti", 100, 190, 150)
            s_temp = st.slider("Temperatura Esterna (°C)", 0, 40, 25)
            
            # Calcolo metriche derivate in background
            s_tempo = s_dist * 4.5 + (s_rpe * 2) 
            s_lavoro = s_tempo / 60
            s_sma = (s_lavoro * s_rpe) / s_sonno
            s_islr = (s_lavoro * s_rpe) / s_dist
            s_idet = (s_fc * s_temp) / ((s_dist*1000)/(s_tempo*60))
            s_iitr = (s_temp * 10) / s_dist
            
            input_data = pd.DataFrame([[s_dist, s_sonno, s_sma, s_islr, s_idet, s_iitr, s_rpe]], columns=RF_FEATURES)
            prob = rf.predict_proba(input_data)[0][1] * 100
            label, color = risk_band(prob)

        # COLONNA DESTRA: RISULTATI
        with sc2:
            st.markdown("<h4 style='color: #ffffff;'>Verdetto dell'AI</h4>", unsafe_allow_html=True)
            
            c_gauge, c_radar = st.columns([1, 1])
            
            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob, number={'suffix': "%", 'font': {'color': color}}, title = {'text': f"Livello di Rischio: {label}", 'font': {'color': '#cbd5e1'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['border']},
                        'bar': {'color': color},
                        'bgcolor': COLORS['surface'],
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(163, 230, 53, 0.2)'},  # Green zone
                            {'range': [40, 70], 'color': 'rgba(251, 191, 36, 0.2)'},  # Amber zone
                            {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.2)'}], # Red zone
                    }))
                fig_g.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
            
            with c_radar:
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=["Distanza", "Sonno", "Stress (SMA)", "Lavoro", "Impatto Termico", "Interferenza", "RPE"], fill='toself', name='Media Storica Atleta', line_color=COLORS['border_soft']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=["Distanza", "Sonno", "Stress (SMA)", "Lavoro", "Impatto Termico", "Interferenza", "RPE"], fill='toself', name='Simulazione Odierna', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=350, margin=dict(l=30, r=30, t=30, b=30), legend=dict(y=-0.2))
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown(f"""
        <div class="coach-insight" style="margin-top: 10px;">
            <span>Interpretazione del Simulatore per il Coach</span>
            <strong>Tachimetro a Sinistra:</strong> Se il valore supera il 70% (Area Rossa), l'Intelligenza Artificiale reputa l'allenamento troppo logorante in base alle condizioni attuali (es. poco riposo). È consigliato ridurre lo sforzo agendo sui cursori a sinistra per riportare l'ago in zona Verde o Gialla.<br><br>
            <strong>Radar a Destra:</strong> L'area in grigio è quello a cui l'atleta è "abituato" mediamente. L'area colorata è quello che gli stai chiedendo di fare oggi. Se l'area colorata "sborda" enormemente fuori dal grigio in corrispondenza dello Stress o dell'RPE, significa che stai somministrando uno stimolo eccessivo rispetto alle sue abitudini consolidate, aumentando il rischio infortuni.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_ui()
