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
# CONFIGURAZIONE PAGINA (Forza l'estetica a tutto schermo)
# ============================================================================
st.set_page_config(
    page_title="RUN AI - Sport ML Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# COSTANTI E COLORI (Estratti esattamente dalla foto IMG_3331_2.jpg)
# ============================================================================
COLORS = {
    "bg": "#0b0f19",          # Sfondo scuro principale
    "surface": "#111827",     # Sfondo della card (leggermente più chiaro)
    "surface_2": "#1e293b",   # Sfondo per i grafici
    "border": "#1f2937",      # Bordi scuri generici
    "border_light": "rgba(255,255,255,0.05)",
    "text": "#ffffff",        # Testo principale
    "text_soft": "#9ca3af",   # Testo secondario (grigio chiaro)
    "cyan": "#00e5ff",        # Azzurro neon
    "cyan_dim": "#131c2c",    # Sfondo box info
    "green": "#a3e635",       # Verde acido (sfumatura in alto)
    "amber": "#fbbf24",
    "red": "#f87171",
    "purple": "#a78bfa"
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
    n_estimators: int = 200
    max_depth: int = 8
    n_clusters: int = 3
    n_sessions: int = 1000

# ============================================================================
# CSS ESTREMO (Sovrascrive Streamlit e clona il layout della foto)
# ============================================================================
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    /* Reset totale */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {COLORS['bg']} !important;
        color: {COLORS['text']} !important;
    }}
    
    .stApp {{ background-color: {COLORS['bg']} !important; }}
    [data-testid="collapsedControl"], #MainMenu, footer, header {{ display: none !important; }}
    .block-container {{ max-width: 1300px; padding-top: 2rem !important; }}
    
    /* CARD PRINCIPALE IDENTICA ALLA FOTO */
    .runai-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 2.5rem;
        position: relative;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .runai-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['cyan']} 0%, {COLORS['green']} 100%);
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    
    .runai-kicker {{
        color: {COLORS['cyan']};
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}
    
    .runai-title {{
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
    }}
    
    .runai-subtitle {{
        color: {COLORS['text_soft']};
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }}
    
    /* BOX INFORMATIVO CON BORDO AZZURRO */
    .runai-info-box {{
        background-color: {COLORS['cyan_dim']};
        border-left: 3px solid {COLORS['cyan']};
        border-radius: 4px;
        padding: 1.2rem;
        color: {COLORS['text_soft']};
        font-size: 0.9rem;
        border-top: 1px solid {COLORS['border_light']};
        border-right: 1px solid {COLORS['border_light']};
        border-bottom: 1px solid {COLORS['border_light']};
    }}
    .runai-info-box strong {{ color: #ffffff; }}

    /* STILIZZAZIONE DEL MENU A SCELTA (Rimuove i pallini, crea pulsanti tech) */
    div[data-testid="stRadio"] > div {{
        display: flex; flex-direction: row; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem;
    }}
    div[data-testid="stRadio"] > div > label {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 12px 20px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    div[data-testid="stRadio"] > div > label:hover {{
        border-color: {COLORS['text_soft']};
    }}
    div[data-testid="stRadio"] > div > label[data-checked="true"] {{
        background-color: rgba(0, 229, 255, 0.05);
        border-color: {COLORS['cyan']};
    }}
    div[data-testid="stRadio"] > div > label > div:first-child {{
        display: none !important; /* NASCONDE IL PALLINO DEL RADIO BUTTON */
    }}
    div[data-testid="stRadio"] > div > label p {{
        color: {COLORS['text_soft']};
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin: 0;
    }}
    div[data-testid="stRadio"] > div > label[data-checked="true"] p {{
        color: {COLORS['cyan']};
    }}

    /* BOX SPIEGAZIONI GRAFICI */
    .coach-insight {{
        background: {COLORS['surface']};
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid {COLORS['border']};
        font-size: 0.85rem;
        color: {COLORS['text_soft']};
        margin-top: 5px;
        margin-bottom: 25px;
    }}
    .coach-insight span {{
        color: {COLORS['cyan']};
        font-weight: 800;
        text-transform: uppercase;
        font-size: 0.7rem;
        display: block;
        margin-bottom: 5px;
        letter-spacing: 0.05em;
    }}
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
    )
    pio.templates["runai"] = tpl
    pio.templates.default = "runai"

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
    
    return pd.DataFrame({
        "Distanza (km)": distanza, "Tempo (min)": tempo, "Velocità (m/s)": velocita,
        "RPE": rpe, "Ore Sonno": ore_sonno, "FC Media": fc_media,
        "Temperatura": temperatura, "Vento": vento,
        "SMA": sma, "ISLR": islr, "IDET": idet, "IITR": iitr,
        TARGET: rischio
    }).round(2)

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

    # 1. LA CARD PRINCIPALE IDENTICA ALLA FOTO
    st.markdown(f"""
    <div class="runai-card">
        <div class="runai-kicker">● MODULO 06 - SPORT DATA SCIENCE</div>
        <div class="runai-title">AI PERFORMANCE ANALYSIS & INJURY PREDICTION</div>
        <div class="runai-subtitle">
            Questa dashboard analizza i dati storici del tuo team. L'Intelligenza Artificiale impara come il corpo degli atleti reagisce ai carichi, prevedendo cali di rendimento e segnalando il rischio infortuni da sovraccarico.
        </div>
        
        <div class="runai-info-box">
            <strong>Analisi Fisiologica Avanzata:</strong> Estrazione e calcolo dello Stress Metabolico (SMA) e dell'Indice di Lavoro (ISLR). I modelli predittivi valutano come la mancanza di sonno e la percezione della fatica (RPE) si combinano per alterare il ritmo gara e logorare l'atleta.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. IL MENU TECNICO
    sezioni = [
        "1. STIMA DEL RITMO", 
        "2. RISCHIO BASE", 
        "3. AI PREDICTIVE", 
        "4. PROFILAZIONE", 
        "5. SIMULATORE COACH"
    ]
    scelta = st.radio("Seleziona", sezioni, horizontal=True, label_visibility="collapsed")

    # ==============================================================
    # SEZIONE 1: STIMA PERFORMANCE
    # ==============================================================
    if scelta == sezioni[0]:
        st.markdown("<h3 style='color: white;'>Stima Cronometrica e Analisi del Calo Prestativo</h3>", unsafe_allow_html=True)
        preds = reg.predict(X_test_r)
        
        c1, c2 = st.columns(2)
        c1.metric("Affidabilità Cronometro", f"{r2_score(y_test_r, preds)*100:.1f}%")
        c2.metric("Errore Medio Stima", f"{mean_absolute_error(y_test_r, preds):.1f} min")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter(x=y_test_r, y=preds, opacity=0.6, color_discrete_sequence=[COLORS['cyan']], title="Tempo Reale vs Tempo Calcolato dall'IA")
            fig1.add_shape(type="line", x0=y_test_r.min(), y0=y_test_r.min(), x1=y_test_r.max(), y1=y_test_r.max(), line=dict(dash="dash", color=COLORS['text_soft']))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Se i punti formano una linea retta perfetta, l'atleta è 'un orologio': la fatica lo rallenta esattamente come calcolato. Punti molto distanti segnalano giornate in cui l'atleta è crollato (o volato) per cause esterne non misurabili.</div>", unsafe_allow_html=True)

        with g2:
            fig2 = px.histogram(y_test_r - preds, nbins=30, color_discrete_sequence=[COLORS['purple']], title="Frequenza degli Errori di Previsione")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Le barre più alte dovrebbero stare al centro (zero errori). Se la campana pende verso destra o sinistra, significa che il gruppo squadra tende sistematicamente a rendere meno (o di più) del previsto.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.scatter(x=preds, y=(y_test_r - preds), opacity=0.6, color_discrete_sequence=[COLORS['amber']], title="Decadimento sulle Lunghe Distanze")
            fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['text_soft'])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Guarda come si comporta l'errore man mano che i chilometri aumentano (asse orizzontale). Se l'errore esplode a destra, gli atleti mancano di base aerobica per i lavori lunghi.</div>", unsafe_allow_html=True)

        with g4:
            coefs = pd.DataFrame({"Fattore": X_test_r.columns, "Impatto": reg.coef_}).sort_values("Impatto")
            fig4 = px.bar(coefs, x="Impatto", y="Fattore", orientation="h", color_discrete_sequence=[COLORS['green']], title="Responsabili del Rallentamento")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Mostra i 'pesi'. Leggilo così: per ogni punto di stanchezza percepita (RPE) in più, quanti minuti effettivi perdo sul tempo finale? Questo ti aiuta a quantificare la fatica.</div>", unsafe_allow_html=True)

    # ==============================================================
    # SEZIONE 2: REGRESSIONE LOGISTICA
    # ==============================================================
    elif scelta == sezioni[1]:
        st.markdown("<h3 style='color: white;'>Rischio Overload (Analisi Diretta Base)</h3>", unsafe_allow_html=True)
        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Allenamenti Azzeccati", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Pericoli Trovati", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Falsi Allarmi", f"{(1-precision_score(y_test, y_pred))*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Efficacia degli Allarmi (Curva ROC)", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['text_soft']), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Valuta se stiamo fermando l'atleta al momento giusto. La curva deve stare alta per garantire di aver trovato le sessioni in cui l'atleta ha rischiato lo strappo muscolare.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="IA Dice", y="È Successo"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Contatore degli Errori")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>La casella in basso a sinistra è il tuo nemico: indica le volte che l'algoritmo ha valutato l'allenamento 'sicuro' e invece l'atleta è andato in sovraccarico (Falso Negativo).</div>", unsafe_allow_html=True)

        with g3:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig3 = px.line(x=rec, y=prec, title="Qualità della Scelta", color_discrete_sequence=[COLORS['purple']])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Se per essere sicuri al 100% di non infortunare nessuno fermassimo l'atleta ad ogni piccolo segno di fatica, l'affidabilità crollerebbe. Questo grafico trova il compromesso.</div>", unsafe_allow_html=True)

        with g4:
            coef_df = pd.DataFrame({"Metrica": RF_FEATURES, "Peso sul Rischio": lr.coef_[0]}).sort_values("Peso sul Rischio")
            fig4 = px.bar(coef_df, x="Peso sul Rischio", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['red']], title="I Protettori e i Pericoli")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Le barre rosse a sinistra (come le Ore di Sonno) agiscono da 'scudo protettivo', abbassando il rischio totale. Le barre a destra sommano fatica e spingono l'atleta verso il limite.</div>", unsafe_allow_html=True)

    # ==============================================================
    # SEZIONE 3: RANDOM FOREST
    # ==============================================================
    elif scelta == sezioni[2]:
        st.markdown("<h3 style='color: white;'>Intelligenza Artificiale (Random Forest Multiplo)</h3>", unsafe_allow_html=True)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Decisioni Corrette", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        c2.metric("Sensibilità AI", f"{roc_auc_score(y_test, y_prob)*100:.1f}%")
        c3.metric("Falsi Allarmi", f"{(1-precision_score(y_test, y_pred))*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = px.line(x=fpr, y=tpr, title="Efficacia Avanzata AI", color_discrete_sequence=[COLORS['cyan']])
            fig1.add_shape(type='line', line=dict(dash='dash', color=COLORS['text_soft']), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>L'Intelligenza Artificiale unisce migliaia di variabili per capire, ad esempio, che 'dormire poco è rischioso SOLO se fai ripetute al caldo'. Questo grafico conferma se l'IA è più brava del modello base.</div>", unsafe_allow_html=True)

        with g2:
            cm = confusion_matrix(y_test, y_pred)
            fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purp", labels=dict(x="IA Dice", y="È Successo"), x=['Sicuro', 'Overload'], y=['Sicuro', 'Overload'], title="Contatore Errori AI")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Come puoi notare dai numeri ridotti rispetto a prima, la rete neurale complessa riesce ad evitare di farci sbagliare diagnosi, riducendo le sorprese in allenamento.</div>", unsafe_allow_html=True)

        with g3:
            imp = pd.DataFrame({"Metrica": RF_FEATURES, "Potere Decisionale": rf.feature_importances_}).sort_values("Potere Decisionale")
            fig3 = px.bar(imp, x="Potere Decisionale", y="Metrica", orientation="h", color_discrete_sequence=[COLORS['amber']], title="Le Variabili più Osservate")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>A cosa fa più attenzione il 'cervello' dell'AI per capire se l'atleta si farà male? Solitamente l'RPE e lo Stress Metabolico dominano la classifica su tutti gli altri dati.</div>", unsafe_allow_html=True)

        with g4:
            df_prob = pd.DataFrame({"Probabilità": y_prob * 100, "Stato Reale": ["Overload" if y == 1 else "Sicuro" for y in y_test]})
            fig4 = px.histogram(df_prob, x="Probabilità", color="Stato Reale", barmode="overlay", nbins=40, color_discrete_sequence=[COLORS['red'], COLORS['green']], title="Certezze della Macchina")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>L'IA ha dubbi? Se vedi i grafici tutti divisi (tutto verde a sinistra, tutto rosso a destra), l'IA è sicurissima. Se sono ammassati al centro, c'è grande imprevedibilità nell'atleta.</div>", unsafe_allow_html=True)

    # ==============================================================
    # SEZIONE 4: CLUSTERING
    # ==============================================================
    elif scelta == sezioni[3]:
        st.markdown("<h3 style='color: white;'>Profilazione Automatica (K-Means)</h3>", unsafe_allow_html=True)
        
        sil_score = silhouette_score(StandardScaler().fit_transform(df[CLUSTER_FEATURES]), df["Cluster"])
        st.metric("Separazione netta dei gruppi", f"{sil_score*100:.1f}%")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1 = px.scatter_3d(df, x="FC Media", y="ISLR", z="SMA", color="Cluster", color_continuous_scale=list(CLUSTER_COLORS.values()), title="Mappa Fisiologica 3D")
            fig1.update_layout(scene=dict(xaxis_title="Battiti", yaxis_title="Indice Lavoro", zaxis_title="Stress"), margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>L'IA ha preso mesi di allenamenti e li ha raggruppati da sola per 'impatto sul corpo'. Ogni nuvola colorata è un macro-stimolo (es. Rigenerazione, Lavoro Misto, Altissima Intensità).</div>", unsafe_allow_html=True)

        with g2:
            centroids = df.groupby("Cluster")[CLUSTER_FEATURES].mean().reset_index()
            fig2 = go.Figure()
            for i, row in centroids.iterrows():
                fig2.add_trace(go.Scatterpolar(
                    r=[row["FC Media"]/df["FC Media"].max(), row["ISLR"]/df["ISLR"].max(), row["SMA"]/df["SMA"].max()],
                    theta=["FC Media", "Indice Lavoro", "Stress"], fill='toself', name=f'Tipo {int(row["Cluster"])}'
                ))
            fig2.update_layout(title="Identikit del Lavoro (Radar)", polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Definisce 'chi è' ogni gruppo. Un triangolo grandissimo indica la sessione che spreme l'atleta al massimo su cuore e stress metabolico contemporaneamente.</div>", unsafe_allow_html=True)

        with g3:
            fig3 = px.box(df, x="Cluster", y="SMA", color="Cluster", color_discrete_sequence=list(CLUSTER_COLORS.values()), title="Stress Generato per Tipologia")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>Controllo di coerenza: la sessione che tu chiami 'scarico' (tipo 0) produce davvero poco stress? O l'atleta si sta affaticando anche quando non dovrebbe?</div>", unsafe_allow_html=True)

        with g4:
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Tipologia', 'N° Sessioni']
            fig4 = px.bar(cluster_counts, x='Tipologia', y='N° Sessioni', color='Tipologia', color_continuous_scale=list(CLUSTER_COLORS.values()), title="Bilancio Stagionale")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='coach-insight'><span>Cosa significa per il Coach</span>La regola del 80/20. Quanti allenamenti distruttivi stai facendo rispetto a quelli lenti e rigeneranti? Questo grafico ti mostra il reale bilanciamento del volume.</div>", unsafe_allow_html=True)

    # ==============================================================
    # SEZIONE 5: SIMULATORE PRE-ALLENAMENTO
    # ==============================================================
    elif scelta == sezioni[4]:
        st.markdown("<h3 style='color: white;'>Tavolo di Controllo del Coach (Pre-Allenamento)</h3>", unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        
        with sc1:
            st.markdown("<h4 style='color: #00e5ff; font-size: 1.05rem;'>Valori per la Sessione Odierna</h4>", unsafe_allow_html=True)
            s_dist = st.slider("Chilometri Previsti", 5.0, 42.0, 15.0, 0.5)
            s_rpe = st.slider("Fatica Obiettivo (RPE)", 1, 10, 7)
            s_sonno = st.slider("Ore Sonno Atleta", 3.0, 12.0, 6.5, 0.5)
            s_fc = st.slider("Battiti Stimati (BPM)", 100, 190, 150)
            s_temp = st.slider("Gradi Esterni (°C)", 0, 40, 25)
            
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
            st.markdown("<h4 style='color: #ffffff; font-size: 1.05rem;'>Decisione Intelligenza Artificiale</h4>", unsafe_allow_html=True)
            
            c_gauge, c_radar = st.columns([1, 1])
            
            with c_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob, number={'suffix': "%", 'font': {'color': color}}, title = {'text': f"Stato: {label}", 'font': {'color': COLORS['text_soft']}},
                    gauge = {
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
                fig_r.add_trace(go.Scatterpolar(r=norm_mean, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"], fill='toself', name='Media Squadra', line_color=COLORS['text_soft']))
                fig_r.add_trace(go.Scatterpolar(r=norm_input, theta=["Distanza", "Sonno", "Stress", "Lavoro", "Termico", "Interf.", "RPE"], fill='toself', name='Simulazione Oggi', line_color=color))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), height=300, margin=dict(l=20, r=20, t=20, b=20), legend=dict(y=-0.2))
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown(f"""
        <div class="runai-info-box" style="margin-top: 10px;">
            <strong>Istruzioni del Simulatore per il Coach:</strong><br><br>
            <strong>1. Il Tachimetro:</strong> È il semaforo dell'IA. Se imposti 15km ma l'atleta ha dormito solo 4 ore, l'ago schizza nel rosso (>70%). Il modello ha capito che l'Overload metabolico è imminente. Abbassa i cursori a sinistra (riduci RPE o Distanza) finché l'ago non torna in zona Verde.<br><br>
            <strong>2. Il Radar (Identikit):</strong> Compara il carico di oggi (forma colorata) con quello che l'atleta sopporta di solito (forma grigia). Se la punta colorata dello "Stress" o "Lavoro" esce fuori dal grigio in modo esagerato, stai somministrando uno stimolo sconosciuto e traumatico per il corpo del tuo atleta.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_ui()
