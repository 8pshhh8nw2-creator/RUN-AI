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
    page_title="Sport ML Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# COSTANTI E COLORI
# ============================================================================
APP_TITLE = "Advanced Machine Learning Suite"
APP_SUBTITLE = "Dashboard Analitica per la valutazione del rischio e della performance."

COLORS = {
    "bg": "#030712", "bg2": "#0b1221", "surface": "#111827", "surface_2": "#1f2937",
    "border": "#374151", "border_soft": "rgba(55, 65, 81, 0.5)",
    "text": "#f9fafb", "text_soft": "#9ca3af", "muted": "#6b7280",
    "blue": "#3b82f6", "cyan": "#06b6d4", "green": "#10b981", "amber": "#f59e0b",
    "red": "#ef4444", "purple": "#8b5cf6", "pink": "#ec4899",
}
QUALITATIVE = [COLORS['cyan'], COLORS['purple'], COLORS['blue'], COLORS['amber'], COLORS['pink'], COLORS['green']]
CLUSTER_COLORS = {0: COLORS['cyan'], 1: COLORS['purple'], 2: COLORS['blue'], 3: COLORS['pink']}

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
    seed: int = 42
    test_size: float = 0.25
    n_estimators: int = 200
    max_depth: int = 8
    n_clusters: int = 3
    n_sessions: int = 1000

# ============================================================================
# CSS & THEME PLOTLY
# ============================================================================
PLOTLY_TEMPLATE = "ml_suite_tech"

def register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="SF Pro Display, Inter, sans-serif", color=COLORS['text_soft'], size=13),
        title=dict(font=dict(size=16, color=COLORS['text']), x=0.01, xanchor="left", y=0.96),
        margin=dict(t=50, l=10, r=10, b=30),
        hoverlabel=dict(bgcolor=COLORS['surface_2'], bordercolor=COLORS['border'], font=dict(color=COLORS['text'], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor=COLORS['border_soft'], zeroline=False, linecolor=COLORS['border'], ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor=COLORS['border_soft'], zeroline=False, linecolor=COLORS['border']),
    )
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    :root {{
        --bg:{COLORS['bg']}; --bg2:{COLORS['bg2']}; --surface:{COLORS['surface']};
        --border:{COLORS['border']}; --text:{COLORS['text']}; --cyan:{COLORS['cyan']};
        --purple:{COLORS['purple']};
    }}
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); }}
    
    /* Hide Sidebar & Defaults */
    [data-testid="collapsedControl"] {{ display: none !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{ background: radial-gradient(circle at 50% 0%, var(--bg2) 0%, var(--bg) 100%); }}
    
    /* Headers & Typography */
    h1, h2, h3 {{ font-family: 'Inter', sans-serif; font-weight: 800; letter-spacing: -0.03em; }}
    .kicker {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--cyan); text-transform: uppercase; letter-spacing: 0.15em; }}
    
    /* Tech Menu Styling */
    div.row-widget.stRadio > div {{
        display: flex; flex-direction: row; flex-wrap: wrap; gap: 0;
        background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    }}
    div.row-widget.stRadio > div > label {{
        padding: 14px 20px; cursor: pointer; border-right: 1px solid var(--border); background: transparent; transition: all 0.2s;
    }}
    div.row-widget.stRadio > div > label:hover {{ background: rgba(6, 182, 212, 0.1); }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{ background: rgba(6, 182, 212, 0.15); box-shadow: inset 0 -3px 0 var(--cyan); }}
    div.row-widget.stRadio p {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; margin: 0; color: var(--text); }}
    
    /* Callouts & Explanations */
    .theory-box {{ background: var(--surface); border-left: 4px solid var(--cyan); padding: 1.5rem; margin: 1.5rem 0; border-radius: 4px 8px 8px 4px; border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
    .theory-box h4 {{ margin-top: 0; color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; text-transform: uppercase; }}
    .theory-box p {{ line-height: 1.7; color: var(--text_soft); font-size: 0.95rem; margin-bottom: 0; }}
    
    .chart-desc {{ background: rgba(255,255,255,0.02); padding: 1rem; border: 1px dashed var(--border); border-radius: 6px; font-size: 0.85rem; color: var(--text_soft); margin-top: 10px; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA GENERATOR (DA SOSTITUIRE IN FUTURO)
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
    
    st.markdown(f"<div class='kicker'>Master Thesis Analytical Core</div><h1 style='margin-top:0;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # INTRODUZIONE TEORICA
    # ---------------------------------------------------------
    st.markdown("""
    <div class="theory-box">
        <h4>Introduzione al Machine Learning nello Sport</h4>
        <p>Il Machine Learning applicato alla Sport Science segna il passaggio dall'analisi descrittiva (cosa è successo) all'analisi predittiva (cosa succederà). Storicamente, i preparatori atletici si basavano su soglie fisse e intuizione. Oggi, gli algoritmi processano relazioni non lineari tra carico esterno (es. chilometri percorsi, velocità) e carico interno (es. frequenza cardiaca, fatica percepita, qualità del sonno).<br><br>
        In questa suite analizziamo il rischio di sovraccarico (Overload) e il calo prestativo. L'obiettivo non è sostituire il giudizio umano, ma fornire un <b>supporto decisionale oggettivo</b> capace di pesare decine di variabili simultaneamente per individuare modelli di infortunio o fatica latente invisibili all'occhio umano.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # CARICAMENTO DATI
    # ---------------------------------------------------------
    # QUANDO SARAI PRONTO A USARE I TUOI DATI REALI, SOSTITUISCI LA RIGA SEGUENTE CON:
    # df = pd.read_csv("nome_del_tuo_file.csv")
    df = generate_synthetic_data(cfg.n_sessions, cfg.seed)
    
    # Addestramento modelli
    rf, lr, reg, kmeans, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_test_r, y_test_r = train_models(df, cfg)

    # ---------------------------------------------------------
    # MENU ORIZZONTALE TECH
    # ---------------------------------------------------------
    st.write("")
    sezioni = [
        "REGRESSIONE LINEARE", 
        "REGRESSIONE LOGISTICA", 
        "RANDOM FOREST", 
        "CLUSTERING", 
        "SIMULATORE WHAT-IF"
    ]
    scelta = st.radio("MODULI", sezioni, horizontal=True, label_visibility="collapsed")

    # ==============================================================
    # 1. REGRESSIONE LINEARE
    # ==============================================================
    if scelta == sezioni[0]:
        st.markdown("<div class='kicker'>Stima della Performance</div><h2>Regressione Lineare Multipla</h2>", unsafe_allow_html=True)
        st.markdown(r"""
        <div class="theory-box">
            <h4>Teoria del Modello</h4>
            <p>La Regressione Lineare stima una variabile dipendente continua (in questo caso, il <b>Tempo di completamento</b> dell'allenamento) combinando linearmente diverse variabili indipendenti (Distanza, SMA, RPE). L'equazione matematica alla base è:<br><br>
            $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon$<br><br>
            Dove i coefficienti $\beta$ rappresentano il peso di ogni metrica. Questo algoritmo è essenziale per verificare se l'accumulo di fatica (Stress Metabolico Apparente) altera in modo statisticamente significativo il tempo atteso, indicando un calo della performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        preds = reg.predict(X_test_r)
        
        c1, c2 = st.columns(2)
        c1.metric("R-Squared (R²)", f"{r2_score(y_test_r, preds):.3f}")
        c2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test_r, preds):.2f} min")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter(x=y_test_r, y=preds, opacity=0.6, color_discrete_sequence=[COLORS['cyan']], title="1. Reale vs Predetto")
            fig1.add_shape(type="line", x0=y_test_r.min(), y0=y_test_r.min(), x1=y_test_r.max(), y1=y_test_r.max(), line=dict(dash="dash", color=COLORS['muted']))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Valuta la bontà del modello. Più i punti si allineano sulla diagonale tratteggiata, più le stime corrispondono alla realtà. Deviazioni ampie indicano outlier o fattori esterni non misurati.</div>", unsafe_allow_html=True)

        with g2:
            residui = y_test_r - preds
            fig2 = px.histogram(residui, nbins=30, color_discrete_sequence=[COLORS['purple']], title="2. Distribuzione dei Residui")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Un modello sano dovrebbe produrre errori distribuiti normalmente attorno allo zero. Se la distribuzione è asimmetrica, il modello tende a sovrastimare o sottostimare sistematicamente la prestazione.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.scatter(x=preds, y=residui, opacity=0.6, color_discrete_sequence=[COLORS['pink']], title="3. Omoschedasticità (Predetto vs Residui)")
            fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['muted'])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Verifica la varianza dell'errore. Se l'errore aumenta al crescere del tempo previsto (forma a cono), la relazione tra le variabili perde linearità sui lunghi chilometraggi.</div>", unsafe_allow_html=True)

        with g4:
            coefs = pd.DataFrame({"Feature": X_test_r.columns, "Coef": reg.coef_}).sort_values("Coef")
            fig4 = px.bar(coefs, x="Coef", y="Feature", orientation="h", color_discrete_sequence=[COLORS['amber']], title="4. Peso dei Coefficienti")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Indica l'impatto di ogni singola unità. Ad esempio, a parità di distanza, un incremento unitario del RPE aumenterà il tempo finale proporzionalmente al coefficiente qui visualizzato.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 2. REGRESSIONE LOGISTICA
    # ==============================================================
    elif scelta == sezioni[1]:
        st.markdown("<div class='kicker'>Baseline di Classificazione</div><h2>Regressione Logistica</h2>", unsafe_allow_html=True)
        st.markdown(r"""
        <div class="theory-box">
            <h4>Teoria del Modello</h4>
            <p>La Regressione Logistica è impiegata per prevedere un esito binario (1 = Rischio Overload, 0 = Nessun Rischio). Invece di prevedere un valore continuo, modella la probabilità che un'istanza appartenga a una determinata classe utilizzando la funzione logistica (sigmoide):<br><br>
            $p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ...)}}$<br><br>
            Nello sport, funge da eccellente <i>baseline</i>. Poiché assume una relazione proporzionale diretta tra le features e la log-odds del rischio, ci permette di capire quali metriche (es. ISLR o mancanza di sonno) spingono l'atleta verso il sovraccarico in modo lineare.</p>
        </div>
        """, unsafe_allow_html=True)

        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        c2.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
        c3.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob):.3f}")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="1. Curva ROC", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            fig1.update_layout(xaxis_title="Falsi Positivi (FPR)", yaxis_title="Veri Positivi (TPR)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Rappresenta il compromesso tra la capacità di individuare il sovraccarico (Sensibilità) e il rischio di generare falsi allarmi. Più l'area sotto la curva (AUC) è vicina a 1, migliore è il modello.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predizione", y="Realtà"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="2. Matrice di Confusione")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Mostra l'esito delle predizioni. I Falsi Negativi (in basso a sinistra) sono gli errori più gravi nello sport: sessioni a rischio classificate come sicure, esponendo l'atleta all'infortunio.</div>", unsafe_allow_html=True)

        with g3:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig3 = px.line(x=rec, y=prec, title="3. Curva Precision-Recall", color_discrete_sequence=[COLORS['pink']])
            fig3.update_layout(xaxis_title="Recall (Copertura del rischio)", yaxis_title="Precision (Affidabilità dell'allarme)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Cruciale in dataset sbilanciati. Mostra come l'affidabilità dell'allarme (Precision) decada quando cerchiamo di catturare ogni singola sessione a rischio possibile (Recall).</div>", unsafe_allow_html=True)

        with g4:
            coef_df = pd.DataFrame({"Feature": RF_FEATURES, "Logit Weight": lr.coef_[0]}).sort_values("Logit Weight")
            fig4 = px.bar(coef_df, x="Logit Weight", y="Feature", orientation="h", color_discrete_sequence=[COLORS['purple']], title="4. Importanza Lineare delle Variabili")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> A differenza del Random Forest, qui vediamo l'effetto direzionale. Valori negativi riducono la probabilità di overload (es. Ore Sonno), valori positivi la incrementano drasticamente.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 3. RANDOM FOREST
    # ==============================================================
    elif scelta == sezioni[2]:
        st.markdown("<div class='kicker'>Modellazione Non-Lineare</div><h2>Random Forest Classifier</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="theory-box">
            <h4>Teoria del Modello</h4>
            <p>Il Random Forest è un metodo di <i>Ensemble Learning</i> che addestra centinaia di Alberi Decisionistici su porzioni casuali di dati e variabili. La previsione finale è la media (o la classe maggioritaria) delle risposte di tutti gli alberi.<br><br>
            A differenza della Regressione Logistica, il Random Forest eccelle nell'individuare <b>pattern non lineari e interazioni complesse</b>. Nello sport, questo è fondamentale: l'effetto di un RPE elevato potrebbe essere innocuo se l'atleta ha dormito 9 ore, ma critico se ne ha dormite 5. Il Random Forest modella queste condizioni ramificate senza bisogno di specificarle matematicamente a priori.</p>
        </div>
        """, unsafe_allow_html=True)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        c2.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
        c3.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob):.3f}")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="1. Curva ROC", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['muted']), x0=0, x1=1, y0=0, y1=1)
            fig1.update_layout(xaxis_title="Falsi Positivi", yaxis_title="Veri Positivi")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Se quest'area è superiore a quella della regressione logistica, dimostra inequivocabilmente che le cause del sovraccarico allenante contengono interazioni non lineari complesse.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purp", labels=dict(x="Predizione", y="Realtà"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="2. Matrice di Confusione")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Il Random Forest tende a separare meglio le classi, riducendo l'incertezza e minimizzando l'errore misto (spesso azzerando o abbassando drasticamente i Falsi Positivi/Negativi rispetto al modello base).</div>", unsafe_allow_html=True)

        with g3:
            imp = pd.DataFrame({"Feature": RF_FEATURES, "Gini": rf.feature_importances_}).sort_values("Gini")
            fig3 = px.bar(imp, x="Gini", y="Feature", orientation="h", color_discrete_sequence=[COLORS['amber']], title="3. Gini Feature Importance")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Misura quante volte una metrica è stata usata dagli alberi per separare in modo netto i dati. Mostra il 'potere informativo' puro della variabile nel contesto globale della fatica.</div>", unsafe_allow_html=True)

        with g4:
            df_prob = pd.DataFrame({"Probabilità": y_prob, "Classe Reale": ["Overload" if y == 1 else "Sicuro" for y in y_test]})
            fig4 = px.histogram(df_prob, x="Probabilità", color="Classe Reale", barmode="overlay", nbins=40, color_discrete_sequence=[COLORS['cyan'], COLORS['pink']], title="4. Distribuzione delle Certezze Predittive")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Valuta la sicurezza del modello. Un modello eccellente presenterà due picchi distinti (verso lo 0.0 e verso l'1.0), indicando che l'algoritmo non ha dubbi su cosa costituisca un rischio.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 4. K-MEANS CLUSTERING
    # ==============================================================
    elif scelta == sezioni[3]:
        st.markdown("<div class='kicker'>Apprendimento Non Supervisionato</div><h2>K-Means Clustering</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="theory-box">
            <h4>Teoria del Modello</h4>
            <p>Il K-Means è un algoritmo che non tenta di prevedere un risultato, ma esplora i dati alla ricerca di similitudini per dividerli in sottogruppi (Cluster). Funziona calcolando le distanze euclidee nello spazio multidimensionale per minimizzare la varianza interna ai gruppi.<br><br>
            Nella pratica sportiva, questo serve a <b>scoprire profili latenti di allenamento</b>. Non imponiamo noi le categorie (es. "Scarico", "Lavoro Medio", "Alta Intensità"), ma lasciamo che la matematica raggruppi le sessioni che inducono reazioni fisiologiche e biomeccaniche identiche sull'atleta.</p>
        </div>
        """, unsafe_allow_html=True)

        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Silhouette Score", f"{sil_score:.3f}")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", color_continuous_scale=list(CLUSTER_COLORS.values()), title="1. Spazio Latente (3D)")
            fig1.update_layout(scene=dict(xaxis_title="FC Media", yaxis_title="ISLR", zaxis_title="SMA"), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Mappa fisica dei cluster. Più i gruppi sono separati visivamente, più i profili di allenamento indotti sono fisiologicamente distanti tra loro.</div>", unsafe_allow_html=True)

        with g2:
            centroids = df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
            fig2 = go.Figure()
            for i, row in centroids.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row["FC Media"]/df["FC Media"].max(), row["ISLR"]/df["ISLR"].max(), row["SMA"]/df["SMA"].max()],
                    theta=CLUSTER_FEATURES, fill='toself', name=f'Cluster {int(row["Cluster"])}'
                ))
            fig2.update_layout(title="2. Profilazione Centroidi (Radar)", polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Identifica il 'DNA' di ogni cluster. Un cluster esteso sull'asse SMA rappresenta sessioni ad alto impatto metabolico, definendone il profilo latente.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.box(df, x="Cluster", y="SMA", color="Cluster", color_discrete_sequence=list(CLUSTER_COLORS.values()), title="3. Varianza dello Stress Metabolico")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Misura la compattezza dei gruppi rispetto a una singola metrica critica (SMA). Se i box si sovrappongono troppo, la metrica non è discriminante.</div>", unsafe_allow_html=True)

        with g4:
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Conteggio']
            fig4 = px.bar(cluster_counts, x='Cluster', y='Conteggio', color='Cluster', color_continuous_scale=list(CLUSTER_COLORS.values()), title="4. Distribuzione Volumetrica")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='chart-desc'><b>Analisi:</b> Verifica la distribuzione dei carichi nel macrociclo dell'atleta. Un'alta densità di sessioni nei cluster più stressanti indica un'errata periodizzazione del carico.</div>", unsafe_allow_html=True)

    # ==============================================================
    # 5. SIMULATORE
    # ==============================================================
    elif scelta == sezioni[4]:
        st.markdown("<div class='kicker'>Applica i Modelli Predittivi</div><h2>Simulatore Interattivo What-If</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="theory-box">
            <h4>Valutazione in Tempo Reale</h4>
            <p>Questa sezione sfrutta il modello <b>Random Forest</b> addestrato in background. Inserendo ipotetici valori ambientali e fisiologici, il sistema calcola all'istante le metriche derivate (SMA, ISLR, IDET) ed emette una sentenza probabilistica sul rischio di sovraccarico per l'atleta prima che questo scenda in campo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.markdown("#### Imposta Variabili Operative")
            s_dist = st.slider("Distanza Prevista (km)", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Percepita Attesa (RPE)", 1, 10, 7)
            s_sonno = st.slider("Ore di Sonno Notte Precedente", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("FC Media Stimata (bpm)", 100, 190, 150)
            s_temp = st.slider("Temperatura Esterna (C)", 0, 40, 25)
            
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
            st.markdown("#### Verdetto dell'Algoritmo")
            
            c_gauge, c_radar = st.columns([1, 1])
            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob, number={'suffix': "%"}, title = {'text': f"Rischio: {label}"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['border']},
                        'bar': {'color': color},
                        'bgcolor': COLORS['surface'],
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.15)'},
                            {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}],
                    }))
                fig_g.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
            
            with c_radar:
                means = df[RF_FEATURES].mean()
                maxs = df[RF_FEATURES].max()
                norm_input = (input_data.iloc[0] / maxs).tolist()
                norm_mean = (means / maxs).tolist()
                
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=RF_FEATURES, fill='toself', name='Media Storica', line_color=COLORS['border_soft']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=RF_FEATURES, fill='toself', name='Simulazione', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=350, margin=dict(l=30, r=30, t=30, b=30))
                st.plotly_chart(fig_r, use_container_width=True)

if __name__ == "__main__":
    render_ui()
