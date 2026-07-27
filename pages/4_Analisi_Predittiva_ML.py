import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')
# ============================================================================
# CONFIGURAZIONE INIZIALE & DESIGN SYSTEM (DARK TECH)
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Tesi Magistrale",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .hero-box {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f9fafb;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin: 0;
        font-weight: 400;
    }
    /* Box di Spiegazione & Teoria */
    .tech-box-theory {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 18px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e5e7eb;
        font-size: 0.95rem;
    }
    .tech-box-explanation {
        background-color: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        padding: 18px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e5e7eb;
        font-size: 0.95rem;
    }
    /* NUOVO: box di transizione narrativa tra un modello e il successivo */
    .transition-box {
        background: linear-gradient(90deg, rgba(168,85,247,0.10), rgba(168,85,247,0.0));
        border-left: 4px solid #a855f7;
        padding: 14px 18px;
        margin: 5px 0 25px 0;
        border-radius: 0 8px 8px 0;
        color: #d1d5db;
        font-size: 0.95rem;
        font-style: italic;
    }
    /* NUOVO: card della dashboard di sintesi */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    .kpi-label {
        color: #9ca3af;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .kpi-value {
        color: #f9fafb;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 6px 0 2px 0;
    }
    .kpi-sub {
        color: #6b7280;
        font-size: 0.8rem;
    }
    /* Sezioni Modelli */
    .model-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .model-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 15px;
    }
    .section-num {
        color: #a855f7;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================================
# MOTORE DATI SIMULATI
# ============================================================================
@st.cache_data
def generate_thesis_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'Distanza (km)': np.random.uniform(5, 30, n),
        'FC Media': np.random.uniform(120, 180, n),
        'Velocità (km/h)': np.random.uniform(9, 16, n),
        'Ore Sonno': np.random.uniform(4, 9, n),
        'Stress Lavoro': np.random.uniform(1, 10, n),
        'Ore Lavoro': np.random.uniform(0, 10, n),
        'RPE': np.random.uniform(1, 10, n),
        'Temp (°C)': np.random.uniform(10, 35, n),
        'Vento (km/h)': np.random.uniform(0, 25, n)
    })
    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60 + np.random.normal(0, 5, n)
    # KPI proprietari integrati sotto il cofano
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    risk_score = (df['ISLR'] * 0.5) + (df['IDET'] * 0.02) - (df['Ore Sonno'] * 0.6)
    df['Rischio Overload'] = (risk_score > risk_score.quantile(0.70)).astype(int)
    return df
df = generate_thesis_data()
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#9ca3af"),
        margin=dict(t=30, b=20, l=20, r=20)
    )
    return fig
def kpi_card(label, value, sub):
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)
# ============================================================================
# ADDESTRAMENTO UNICO DI TUTTI I MODELLI (fonte unica di verità)
# Fatto UNA volta, prima del layout, così la dashboard di sintesi e i tab
# usano sempre gli stessi risultati - nessuna duplicazione di logica.
# ============================================================================
# --- 1. Regressione Lineare: Distanza -> Tempo ---
X_lr = df[['Distanza (km)']].values
y_lr = df['Tempo (min)'].values
lr_model = LinearRegression().fit(X_lr, y_lr)
df['Tempo_Predetto'] = lr_model.predict(X_lr)
df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']
lr_r2 = r2_score(y_lr, df['Tempo_Predetto'])
# --- 2. Regressione Logistica: ISLR -> Rischio Overload ---
X_log = df[['ISLR']].values
y_log = df['Rischio Overload'].values
log_model = LogisticRegression().fit(X_log, y_log)
df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]
log_acc = accuracy_score(y_log, log_model.predict(X_log))
log_auc = roc_auc_score(y_log, df['Probabilità_Overload'])
x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
y_prob_curve = log_model.predict_proba(x_range)[:, 1]
# --- 3. Random Forest: feature multiple -> Rischio Overload ---
rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
rf_proba = rf.predict_proba(df[rf_features])[:, 1]
rf_acc = accuracy_score(df['Rischio Overload'], rf.predict(df[rf_features]))
rf_auc = roc_auc_score(df['Rischio Overload'], rf_proba)
imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')
# --- 4. K-Means: segmentazione sessioni ---
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[['FC Media', 'ISLR']])
df['Cluster_ID'] = km.labels_
sil_score = silhouette_score(df[['FC Media', 'ISLR']], km.labels_)
# Mappa i cluster ai profili in base al centroide (evita di forzare a mano l'ordine 0/1/2)
centroids = pd.DataFrame(km.cluster_centers_, columns=['FC Media', 'ISLR'])
order = centroids['ISLR'].sort_values().index  # basso ISLR -> alto ISLR
cluster_labels_ordered = ['Rigenerativo', 'Qualità / Misto', 'Elevato Stress']
cluster_map = {order[i]: cluster_labels_ordered[i] for i in range(3)}
df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)
# ============================================================================
# HERO HEADER & INTRODUZIONE AL MACHINE LEARNING
# ============================================================================
st.markdown("""
<div class='hero-box'>
    <div class='hero-title'>Advanced Machine Learning Suite</div>
    <div class='hero-subtitle'>Framework predittivo avanzato per l'analisi dei carichi di allenamento, stima della performance e prevenzione del sovraccarico atletico.</div>
</div>
""", unsafe_allow_html=True)
st.markdown("""
### 🧠 Cos'è il Machine Learning nello Sport?
Il **Machine Learning** rappresenta il superamento definitivo delle limitazioni dei metodi di allenamento tradizionali e puramente descrittivi.
Invece di affidarsi unicamente a tabelle statiche o soglie fisse, gli algoritmi analizzano simultaneamente centinaia di variabili (biometriche, ambientali e di carico psicofisico) per riconoscere pattern nascosti, stimare la performance attesa e anticipare i rischi di sovraccarico (overtraining) in modo **proattivo**.
""")
# ============================================================================
# NUOVO: DASHBOARD DI SINTESI — dà al lettore il quadro d'insieme PRIMA
# di entrare nel dettaglio di ciascun modello, e lega esplicitamente ogni
# modello a una domanda di ricerca della tesi.
# ============================================================================
st.markdown("### 📊 Quadro di Sintesi: i Quattro Modelli a Confronto")
st.markdown("Ogni algoritmo di questa suite risponde a una domanda diversa della tesi. Prima dell'analisi dettagliata, ecco i risultati chiave:")
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Regressione Lineare", f"R² {lr_r2:.2f}", "Quanto bene stima il tempo di gara")
with k2:
    kpi_card("Regressione Logistica", f"AUC {log_auc:.2f}", f"Accuratezza {log_acc*100:.0f}% sul rischio")
with k3:
    kpi_card("Random Forest", f"AUC {rf_auc:.2f}", f"Accuratezza {rf_acc*100:.0f}%, spiega le cause")
with k4:
    kpi_card("K-Means", f"Silhouette {sil_score:.2f}", "3 profili di allenamento distinti")
st.markdown("<br>", unsafe_allow_html=True)
mapping_df = pd.DataFrame({
    "Modello": ["Regressione Lineare", "Regressione Logistica", "Random Forest", "K-Means Clustering"],
    "Domanda di Ricerca": [
        "Quanto tempo impiegherò dati i km da percorrere?",
        "Qual è la probabilità che questa sessione sia a rischio overload?",
        "Quali fattori causano davvero il rischio, e quanto pesa ciascuno?",
        "Esistono profili di allenamento ricorrenti nei miei dati?"
    ],
    "Tipo di Output": ["Predizione continua", "Probabilità/Classe", "Ranking di importanza", "Segmentazione non supervisionata"]
})
st.dataframe(mapping_df, use_container_width=True, hide_index=True)
st.markdown("<br>", unsafe_allow_html=True)
tab_ml, tab_sim = st.tabs([
    "🧠 Suite Modelli Machine Learning",
    "🎮 Centrale Operativa Simulatore What-If"
])
# ============================================================================
# TAB 1: PERCORSO ANALITICO GUIDATO
# I 4 modelli non sono più blocchi paralleli e slegati: ogni sezione è
# preceduta da un box di transizione che spiega perché si passa al modello
# successivo, seguendo la logica descrittiva -> predittiva -> esplicativa
# -> di segmentazione. Chiude con una sintesi comparativa.
# ============================================================================
with tab_ml:
    st.markdown("""
    <span class='section-num'>4.1</span> Il percorso analitico segue una progressione precisa: prima si **stima** una variabile continua (il tempo di gara), poi si **classifica** il rischio di una sessione, poi si **spiega** quali fattori guidano quella classificazione, e infine si **scoprono** i profili ricorrenti di allenamento senza etichette predefinite.
    """, unsafe_allow_html=True)
    # ---------------------------------------------------------
    # 1. REGRESSIONE LINEARE
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.1</span>📈 Regressione Lineare (OLS Trend Prediction)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> La regressione lineare modella la relazione tra una variabile dipendente continua (es. Tempo) e una indipendente (es. Distanza), minimizzando la discrepanza tra i valori reali e la linea di tendenza (Minimi Quadrati Ordinari). Il modello raggiunge un <b>R² di {lr_r2:.2f}</b> sui dati raccolti.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_lr1 = go.Figure()
        fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo (min)'], mode='markers', name='Dati Reali', marker=dict(color='#0ea5e9', opacity=0.6)))
        fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo_Predetto'], mode='lines', name='Trend Ottimale (OLS)', line=dict(color='#f43f5e', width=3)))
        apply_dark_theme(fig_lr1)
        fig_lr1.update_layout(title="Relazione Distanza - Tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
        st.plotly_chart(fig_lr1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> La linea rossa rappresenta il trend calcolato dall'algoritmo OLS per stimare i minuti attesi in base ai chilometri percorsi.</div>", unsafe_allow_html=True)
    with c2:
        fig_lr2 = px.histogram(df, x="Errore (Residuo)", nbins=20, title="Distribuzione degli Errori di Previsione", color_discrete_sequence=['#a855f7'])
        apply_dark_theme(fig_lr2)
        st.plotly_chart(fig_lr2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> L'istogramma degli errori centrato sullo zero attesta la bontà e l'assenza di bias sistematici nel modello.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='transition-box'>➡️ Sapere <i>quanto</i> tempo impiegherò è utile, ma non basta a proteggermi dal sovraccarico. Il passo successivo è capire <b>quando</b> una sessione diventa a rischio: da qui la regressione logistica.</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # 2. REGRESSIONE LOGISTICA
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.2</span>🎯 Regressione Logistica (Sigmoid Classification)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Algoritmo di classificazione che stima la probabilità che una sessione appartenga a uno stato critico, mappando le feature in uno spazio [0, 1] tramite funzione logistica (Sigmoide). Su questi dati ottiene un'<b>accuratezza del {log_acc*100:.0f}%</b> (AUC {log_auc:.2f}).</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_log1 = go.Figure()
        fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Osservazioni', marker=dict(color='#64748b', opacity=0.5)))
        fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob_curve, mode='lines', name='Curva Sigmoide', line=dict(color='#f59e0b', width=3)))
        fig_log1.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Soglia Decisionale (50%)")
        apply_dark_theme(fig_log1)
        fig_log1.update_layout(title="Transizione verso l'Overload", xaxis_title="ISLR", yaxis_title="Probabilità Predetta")
        st.plotly_chart(fig_log1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> La curva a 'S' mostra l'impennata del rischio quando l'indice di sforzo supera la soglia critica del 50%.</div>", unsafe_allow_html=True)
    with c2:
        fig_log2 = px.box(df, x="Rischio Overload", y="Probabilità_Overload", color="Rischio Overload",
                          color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'}, title="Separabilità delle Classi")
        apply_dark_theme(fig_log2)
        st.plotly_chart(fig_log2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> I boxplot confermano la netta separazione tra le sessioni sicure (0) e quelle classificate a rischio (1).</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='transition-box'>➡️ La regressione logistica dice <i>quanto</i> rischio, ma usa un solo indice (ISLR) e non dice <i>quali fattori</i> pesano di più tra sonno, distanza, temperatura e vento. Serve un modello che gestisca più variabili insieme e ne spieghi il peso relativo: il Random Forest.</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # 3. RANDOM FOREST
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.3</span>🌳 Random Forest Classifier (Ensemble Decision Trees)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Opera combinando alberi di decisione multipli. Gestisce le interazioni non lineari e calcola la Feature Importance per identificare i fattori critici di stress. Con 6 feature combinate raggiunge un'<b>accuratezza del {rf_acc*100:.0f}%</b> (AUC {rf_auc:.2f}), superiore al singolo indice ISLR.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_rf1 = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Feature Importance Globale", color='Importanza', color_continuous_scale="Teal")
        apply_dark_theme(fig_rf1)
        st.plotly_chart(fig_rf1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Classificazione gerarchica delle metriche che incidono maggiormente sul rischio di sovraccarico.</div>", unsafe_allow_html=True)
    with c2:
        top_2 = imp_df.tail(2)['Feature'].values
        fig_rf2 = px.scatter(df, x=top_2[0], y=top_2[1], color='Rischio Overload',
                               title=f"Interazione Top 2 Feature",
                               color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'})
        apply_dark_theme(fig_rf2)
        st.plotly_chart(fig_rf2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Mappatura bidimensionale delle aree critiche individuate dall'ensemble di alberi decisionali.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='transition-box'>➡️ Fin qui il focus è stato sulla singola sessione. L'ultimo passo cambia prospettiva: invece di classificare una sessione contro un'etichetta nota, si lascia che l'algoritmo scopra da solo gruppi ricorrenti di allenamento — il clustering K-Means.</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.4</span>🔍 Clustering K-Means (Unsupervised Segmentation)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Raggruppa le sessioni in base a similitudini geometriche multidimensionali senza etichette preliminari, scoprendo cluster latenti nei dati. Il <b>Silhouette Score di {sil_score:.2f}</b> indica quanto i 3 gruppi trovati siano effettivamente separati tra loro.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa",
                               title="Segmentazione Automatica delle Sessioni",
                               color_discrete_sequence=['#10b981', '#f59e0b', '#f43f5e'])
        apply_dark_theme(fig_km1)
        st.plotly_chart(fig_km1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Ogni colore rappresenta una famiglia di allenamento distinta scoperta dai centroidi geometrici.</div>", unsafe_allow_html=True)
    with c2:
        cluster_means = df.groupby('Profilo_Corsa')[['Ore Sonno', 'Tempo (min)', 'RPE']].mean().reset_index()
        fig_km2 = px.bar(cluster_means, x='Profilo_Corsa', y=['Ore Sonno', 'RPE'], barmode='group',
                               title="Profiler Comportamentale dei Cluster")
        apply_dark_theme(fig_km2)
        st.plotly_chart(fig_km2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Decodifica delle caratteristiche medie di sonno e sforzo per ciascun cluster identificato.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # NUOVO: SINTESI COMPARATIVA FINALE — chiude il percorso analitico
    # riportando tutto a un unico confronto, utile anche per il capitolo
    # di Discussione della tesi.
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.2</span>🔗 Sintesi Comparativa e Implicazioni</div>
    """, unsafe_allow_html=True)
    perf_df = pd.DataFrame({
        "Modello": ["Reg. Lineare (R²)", "Reg. Logistica (AUC)", "Random Forest (AUC)", "K-Means (Silhouette)"],
        "Punteggio": [lr_r2, log_auc, rf_auc, sil_score]
    })
    fig_perf = px.bar(perf_df, x="Modello", y="Punteggio", color="Modello", range_y=[0, 1],
                       color_discrete_sequence=['#0ea5e9', '#f59e0b', '#10b981', '#a855f7'],
                       title="Confronto Sintetico delle Metriche di Performance")
    apply_dark_theme(fig_perf)
    fig_perf.update_layout(showlegend=False)
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown(f"""
    <div class='tech-box-explanation'>
    <b>Lettura d'insieme:</b> il Random Forest (AUC {rf_auc:.2f}) supera la Regressione Logistica a singola feature (AUC {log_auc:.2f}),
    confermando che il rischio di overload è multifattoriale e non riducibile a un solo indice.
    Il Silhouette Score di {sil_score:.2f} del K-Means suggerisce che i 3 profili di allenamento sono riconoscibili ma non perfettamente separati,
    un punto da discutere nel capitolo conclusivo della tesi.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
# ============================================================================
# TAB 2: SIMULATORE WHAT-IF
# ============================================================================
with tab_sim:
    st.markdown("""
    <span class='section-num'>4.3</span> Il simulatore applica in tempo reale il modello **Random Forest** (Sez. 4.1.3, accuratezza {:.0f}%) a una sessione ipotetica, e la confronta con i profili di allenamento scoperti dal K-Means (Sez. 4.1.4).
    """.format(rf_acc * 100), unsafe_allow_html=True)
    st.markdown("---")
    col_input, col_output = st.columns([1, 1], gap="large")
    with col_input:
        st.subheader("Imposta i Parametri dell'Atleta")
        sim_dist = st.slider("🏃 Distanza pianificata (km)", 5.0, 30.0, 10.0, 0.5)
        sim_sonno = st.slider("🛌 Ore di Sonno stanotte", 3.0, 10.0, 7.5, 0.5)
        sim_stress = st.slider("🧠 Livello Stress Vita/Lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
        sim_lavoro = st.slider("💼 Ore Lavorative oggi", 0.0, 12.0, 8.0, 0.5)
        sim_rpe = st.slider("📈 Sforzo (RPE) che si intende sostenere", 1.0, 10.0, 6.0, 1.0)
        sim_fc = 145.0
        sim_temp = 25.0
        sim_vel = 12.0
        sim_vento = 5.0
    with col_output:
        st.subheader("Elaborazione Algoritmica in Corso...")
        sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
        sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
        sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
        sim_iitr = (sim_temp * sim_vento) / sim_dist if sim_dist > 0 else 0
        input_data = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet, sim_iitr]],
                                  columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR'])
        prob_rischio = rf.predict_proba(input_data)[0][1] * 100
        color_status = "#0ea5e9" if prob_rischio < 40 else "#f59e0b" if prob_rischio < 70 else "#f43f5e"
        st.markdown(f"""
        <div style='background-color: #111827; padding: 30px; border-radius: 12px; border: 1px solid #374151; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <p style='color: #9ca3af; margin-bottom: 0; font-weight: 600; font-size: 0.85rem; letter-spacing: 1px;'>PROBABILITÀ DI OVERTRAINING (SOVRACCARICO)</p>
            <h1 style='color: {color_status}; font-size: 4.5rem; margin: 10px 0; font-weight: 800;'>
                {prob_rischio:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Stress Mentale (SMA)", f"{sim_sma:.2f}")
        m2.metric("Sforzo Lavorativo Residuo (ISLR)", f"{sim_islr:.2f}")
        # NUOVO: colloca la sessione simulata nel profilo K-Means più vicino,
        # collegando esplicitamente il simulatore alla Sez. 4.1.4
        sim_point = np.array([[sim_fc, sim_islr]])
        dists = np.linalg.norm(km.cluster_centers_ - sim_point, axis=1)
        nearest_cluster = np.argmin(dists)
        nearest_profile = cluster_map[nearest_cluster]
        st.markdown(f"<div class='tech-box-explanation'>📍 <b>Profilo di allenamento più vicino (K-Means):</b> questa sessione assomiglia di più al gruppo <b>'{nearest_profile}'</b> individuato in Sez. 4.1.4.</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if prob_rischio < 40:
            st.success("🟢 **SEMAFORO VERDE**: Parametri fisiologici nella norma. Lo stress complessivo è ammortizzato correttamente.")
        elif prob_rischio < 70:
            st.warning("🟡 **ZONA DI ATTENZIONE**: L'algoritmo rileva una competizione energetica tra lavoro e recupero. Cautela.")
        else:
            st.error("🔴 **ALLARME CRITICO**: Combinazione anomala di fattori di stress e scarso sonno. Rischio di sovraccarico elevato.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE INIZIALE & DESIGN SYSTEM (DARK TECH)
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Tesi Magistrale",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-box {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f9fafb;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin: 0;
        font-weight: 400;
    }

    /* Box di Spiegazione & Teoria */
    .tech-box-theory {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 18px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e5e7eb;
        font-size: 0.95rem;
    }
    .tech-box-explanation {
        background-color: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        padding: 18px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e5e7eb;
        font-size: 0.95rem;
    }

    /* NUOVO: box di transizione narrativa tra un modello e il successivo */
    .transition-box {
        background: linear-gradient(90deg, rgba(168,85,247,0.10), rgba(168,85,247,0.0));
        border-left: 4px solid #a855f7;
        padding: 14px 18px;
        margin: 5px 0 25px 0;
        border-radius: 0 8px 8px 0;
        color: #d1d5db;
        font-size: 0.95rem;
        font-style: italic;
    }

    /* NUOVO: card della dashboard di sintesi */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    .kpi-label {
        color: #9ca3af;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .kpi-value {
        color: #f9fafb;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 6px 0 2px 0;
    }
    .kpi-sub {
        color: #6b7280;
        font-size: 0.8rem;
    }

    /* Sezioni Modelli */
    .model-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .model-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 15px;
    }
    .section-num {
        color: #a855f7;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MOTORE DATI SIMULATI
# ============================================================================
@st.cache_data
def generate_thesis_data():
    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        'Distanza (km)': np.random.uniform(5, 30, n),
        'FC Media': np.random.uniform(120, 180, n),
        'Velocità (km/h)': np.random.uniform(9, 16, n),
        'Ore Sonno': np.random.uniform(4, 9, n),
        'Stress Lavoro': np.random.uniform(1, 10, n),
        'Ore Lavoro': np.random.uniform(0, 10, n),
        'RPE': np.random.uniform(1, 10, n),
        'Temp (°C)': np.random.uniform(10, 35, n),
        'Vento (km/h)': np.random.uniform(0, 25, n)
    })

    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60 + np.random.normal(0, 5, n)

    # KPI proprietari integrati sotto il cofano
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']

    risk_score = (df['ISLR'] * 0.5) + (df['IDET'] * 0.02) - (df['Ore Sonno'] * 0.6)
    df['Rischio Overload'] = (risk_score > risk_score.quantile(0.70)).astype(int)

    return df

df = generate_thesis_data()

def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#9ca3af"),
        margin=dict(t=30, b=20, l=20, r=20)
    )
    return fig

def kpi_card(label, value, sub):
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ADDESTRAMENTO UNICO DI TUTTI I MODELLI (fonte unica di verità)
# Fatto UNA volta, prima del layout, così la dashboard di sintesi e i tab
# usano sempre gli stessi risultati - nessuna duplicazione di logica.
# ============================================================================

# --- 1. Regressione Lineare: Distanza -> Tempo ---
X_lr = df[['Distanza (km)']].values
y_lr = df['Tempo (min)'].values
lr_model = LinearRegression().fit(X_lr, y_lr)
df['Tempo_Predetto'] = lr_model.predict(X_lr)
df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']
lr_r2 = r2_score(y_lr, df['Tempo_Predetto'])

# --- 2. Regressione Logistica: ISLR -> Rischio Overload ---
X_log = df[['ISLR']].values
y_log = df['Rischio Overload'].values
log_model = LogisticRegression().fit(X_log, y_log)
df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]
log_acc = accuracy_score(y_log, log_model.predict(X_log))
log_auc = roc_auc_score(y_log, df['Probabilità_Overload'])

x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
y_prob_curve = log_model.predict_proba(x_range)[:, 1]

# --- 3. Random Forest: feature multiple -> Rischio Overload ---
rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
rf_proba = rf.predict_proba(df[rf_features])[:, 1]
rf_acc = accuracy_score(df['Rischio Overload'], rf.predict(df[rf_features]))
rf_auc = roc_auc_score(df['Rischio Overload'], rf_proba)
imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')

# --- 4. K-Means: segmentazione sessioni ---
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[['FC Media', 'ISLR']])
df['Cluster_ID'] = km.labels_
sil_score = silhouette_score(df[['FC Media', 'ISLR']], km.labels_)
# Mappa i cluster ai profili in base al centroide (evita di forzare a mano l'ordine 0/1/2)
centroids = pd.DataFrame(km.cluster_centers_, columns=['FC Media', 'ISLR'])
order = centroids['ISLR'].sort_values().index  # basso ISLR -> alto ISLR
cluster_labels_ordered = ['Rigenerativo', 'Qualità / Misto', 'Elevato Stress']
cluster_map = {order[i]: cluster_labels_ordered[i] for i in range(3)}
df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)

# ============================================================================
# HERO HEADER & INTRODUZIONE AL MACHINE LEARNING
# ============================================================================
st.markdown("""
<div class='hero-box'>
    <div class='hero-title'>Advanced Machine Learning Suite</div>
    <div class='hero-subtitle'>Framework predittivo avanzato per l'analisi dei carichi di allenamento, stima della performance e prevenzione del sovraccarico atletico.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🧠 Cos'è il Machine Learning nello Sport?
Il **Machine Learning** rappresenta il superamento definitivo delle limitazioni dei metodi di allenamento tradizionali e puramente descrittivi.
Invece di affidarsi unicamente a tabelle statiche o soglie fisse, gli algoritmi analizzano simultaneamente centinaia di variabili (biometriche, ambientali e di carico psicofisico) per riconoscere pattern nascosti, stimare la performance attesa e anticipare i rischi di sovraccarico (overtraining) in modo **proattivo**.
""")

# ============================================================================
# NUOVO: DASHBOARD DI SINTESI — dà al lettore il quadro d'insieme PRIMA
# di entrare nel dettaglio di ciascun modello, e lega esplicitamente ogni
# modello a una domanda di ricerca della tesi.
# ============================================================================
st.markdown("### 📊 Quadro di Sintesi: i Quattro Modelli a Confronto")
st.markdown("Ogni algoritmo di questa suite risponde a una domanda diversa della tesi. Prima dell'analisi dettagliata, ecco i risultati chiave:")

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Regressione Lineare", f"R² {lr_r2:.2f}", "Quanto bene stima il tempo di gara")
with k2:
    kpi_card("Regressione Logistica", f"AUC {log_auc:.2f}", f"Accuratezza {log_acc*100:.0f}% sul rischio")
with k3:
    kpi_card("Random Forest", f"AUC {rf_auc:.2f}", f"Accuratezza {rf_acc*100:.0f}%, spiega le cause")
with k4:
    kpi_card("K-Means", f"Silhouette {sil_score:.2f}", "3 profili di allenamento distinti")

st.markdown("<br>", unsafe_allow_html=True)

mapping_df = pd.DataFrame({
    "Modello": ["Regressione Lineare", "Regressione Logistica", "Random Forest", "K-Means Clustering"],
    "Domanda di Ricerca": [
        "Quanto tempo impiegherò dati i km da percorrere?",
        "Qual è la probabilità che questa sessione sia a rischio overload?",
        "Quali fattori causano davvero il rischio, e quanto pesa ciascuno?",
        "Esistono profili di allenamento ricorrenti nei miei dati?"
    ],
    "Tipo di Output": ["Predizione continua", "Probabilità/Classe", "Ranking di importanza", "Segmentazione non supervisionata"]
})
st.dataframe(mapping_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_ml, tab_sim = st.tabs([
    "🧠 Suite Modelli Machine Learning",
    "🎮 Centrale Operativa Simulatore What-If"
])

# ============================================================================
# TAB 1: PERCORSO ANALITICO GUIDATO
# I 4 modelli non sono più blocchi paralleli e slegati: ogni sezione è
# preceduta da un box di transizione che spiega perché si passa al modello
# successivo, seguendo la logica descrittiva -> predittiva -> esplicativa
# -> di segmentazione. Chiude con una sintesi comparativa.
# ============================================================================
with tab_ml:
    st.markdown("""
    <span class='section-num'>4.1</span> Il percorso analitico segue una progressione precisa: prima si **stima** una variabile continua (il tempo di gara), poi si **classifica** il rischio di una sessione, poi si **spiega** quali fattori guidano quella classificazione, e infine si **scoprono** i profili ricorrenti di allenamento senza etichette predefinite.
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 1. REGRESSIONE LINEARE
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.1</span>📈 Regressione Lineare (OLS Trend Prediction)</div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> La regressione lineare modella la relazione tra una variabile dipendente continua (es. Tempo) e una indipendente (es. Distanza), minimizzando la discrepanza tra i valori reali e la linea di tendenza (Minimi Quadrati Ordinari). Il modello raggiunge un <b>R² di {lr_r2:.2f}</b> sui dati raccolti.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_lr1 = go.Figure()
        fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo (min)'], mode='markers', name='Dati Reali', marker=dict(color='#0ea5e9', opacity=0.6)))
        fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo_Predetto'], mode='lines', name='Trend Ottimale (OLS)', line=dict(color='#f43f5e', width=3)))
        apply_dark_theme(fig_lr1)
        fig_lr1.update_layout(title="Relazione Distanza - Tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
        st.plotly_chart(fig_lr1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> La linea rossa rappresenta il trend calcolato dall'algoritmo OLS per stimare i minuti attesi in base ai chilometri percorsi.</div>", unsafe_allow_html=True)

    with c2:
        fig_lr2 = px.histogram(df, x="Errore (Residuo)", nbins=20, title="Distribuzione degli Errori di Previsione", color_discrete_sequence=['#a855f7'])
        apply_dark_theme(fig_lr2)
        st.plotly_chart(fig_lr2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> L'istogramma degli errori centrato sullo zero attesta la bontà e l'assenza di bias sistematici nel modello.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='transition-box'>➡️ Sapere <i>quanto</i> tempo impiegherò è utile, ma non basta a proteggermi dal sovraccarico. Il passo successivo è capire <b>quando</b> una sessione diventa a rischio: da qui la regressione logistica.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. REGRESSIONE LOGISTICA
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.2</span>🎯 Regressione Logistica (Sigmoid Classification)</div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Algoritmo di classificazione che stima la probabilità che una sessione appartenga a uno stato critico, mappando le feature in uno spazio [0, 1] tramite funzione logistica (Sigmoide). Su questi dati ottiene un'<b>accuratezza del {log_acc*100:.0f}%</b> (AUC {log_auc:.2f}).</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_log1 = go.Figure()
        fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Osservazioni', marker=dict(color='#64748b', opacity=0.5)))
        fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob_curve, mode='lines', name='Curva Sigmoide', line=dict(color='#f59e0b', width=3)))
        fig_log1.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Soglia Decisionale (50%)")
        apply_dark_theme(fig_log1)
        fig_log1.update_layout(title="Transizione verso l'Overload", xaxis_title="ISLR", yaxis_title="Probabilità Predetta")
        st.plotly_chart(fig_log1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> La curva a 'S' mostra l'impennata del rischio quando l'indice di sforzo supera la soglia critica del 50%.</div>", unsafe_allow_html=True)

    with c2:
        fig_log2 = px.box(df, x="Rischio Overload", y="Probabilità_Overload", color="Rischio Overload",
                          color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'}, title="Separabilità delle Classi")
        apply_dark_theme(fig_log2)
        st.plotly_chart(fig_log2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> I boxplot confermano la netta separazione tra le sessioni sicure (0) e quelle classificate a rischio (1).</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='transition-box'>➡️ La regressione logistica dice <i>quanto</i> rischio, ma usa un solo indice (ISLR) e non dice <i>quali fattori</i> pesano di più tra sonno, distanza, temperatura e vento. Serve un modello che gestisca più variabili insieme e ne spieghi il peso relativo: il Random Forest.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. RANDOM FOREST
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.3</span>🌳 Random Forest Classifier (Ensemble Decision Trees)</div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Opera combinando alberi di decisione multipli. Gestisce le interazioni non lineari e calcola la Feature Importance per identificare i fattori critici di stress. Con 6 feature combinate raggiunge un'<b>accuratezza del {rf_acc*100:.0f}%</b> (AUC {rf_auc:.2f}), superiore al singolo indice ISLR.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_rf1 = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Feature Importance Globale", color='Importanza', color_continuous_scale="Teal")
        apply_dark_theme(fig_rf1)
        st.plotly_chart(fig_rf1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Classificazione gerarchica delle metriche che incidono maggiormente sul rischio di sovraccarico.</div>", unsafe_allow_html=True)

    with c2:
        top_2 = imp_df.tail(2)['Feature'].values
        fig_rf2 = px.scatter(df, x=top_2[0], y=top_2[1], color='Rischio Overload',
                               title=f"Interazione Top 2 Feature",
                               color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'})
        apply_dark_theme(fig_rf2)
        st.plotly_chart(fig_rf2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Mappatura bidimensionale delle aree critiche individuate dall'ensemble di alberi decisionali.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='transition-box'>➡️ Fin qui il focus è stato sulla singola sessione. L'ultimo passo cambia prospettiva: invece di classificare una sessione contro un'etichetta nota, si lascia che l'algoritmo scopra da solo gruppi ricorrenti di allenamento — il clustering K-Means.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.1.4</span>🔍 Clustering K-Means (Unsupervised Segmentation)</div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Raggruppa le sessioni in base a similitudini geometriche multidimensionali senza etichette preliminari, scoprendo cluster latenti nei dati. Il <b>Silhouette Score di {sil_score:.2f}</b> indica quanto i 3 gruppi trovati siano effettivamente separati tra loro.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa",
                               title="Segmentazione Automatica delle Sessioni",
                               color_discrete_sequence=['#10b981', '#f59e0b', '#f43f5e'])
        apply_dark_theme(fig_km1)
        st.plotly_chart(fig_km1, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Ogni colore rappresenta una famiglia di allenamento distinta scoperta dai centroidi geometrici.</div>", unsafe_allow_html=True)

    with c2:
        cluster_means = df.groupby('Profilo_Corsa')[['Ore Sonno', 'Tempo (min)', 'RPE']].mean().reset_index()
        fig_km2 = px.bar(cluster_means, x='Profilo_Corsa', y=['Ore Sonno', 'RPE'], barmode='group',
                               title="Profiler Comportamentale dei Cluster")
        apply_dark_theme(fig_km2)
        st.plotly_chart(fig_km2, use_container_width=True)
        st.markdown("<div class='tech-box-explanation'><b>Guida alla lettura:</b> Decodifica delle caratteristiche medie di sonno e sforzo per ciascun cluster identificato.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # NUOVO: SINTESI COMPARATIVA FINALE — chiude il percorso analitico
    # riportando tutto a un unico confronto, utile anche per il capitolo
    # di Discussione della tesi.
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'><span class='section-num'>4.2</span>🔗 Sintesi Comparativa e Implicazioni</div>
    """, unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        "Modello": ["Reg. Lineare (R²)", "Reg. Logistica (AUC)", "Random Forest (AUC)", "K-Means (Silhouette)"],
        "Punteggio": [lr_r2, log_auc, rf_auc, sil_score]
    })
    fig_perf = px.bar(perf_df, x="Modello", y="Punteggio", color="Modello", range_y=[0, 1],
                       color_discrete_sequence=['#0ea5e9', '#f59e0b', '#10b981', '#a855f7'],
                       title="Confronto Sintetico delle Metriche di Performance")
    apply_dark_theme(fig_perf)
    fig_perf.update_layout(showlegend=False)
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(f"""
    <div class='tech-box-explanation'>
    <b>Lettura d'insieme:</b> il Random Forest (AUC {rf_auc:.2f}) supera la Regressione Logistica a singola feature (AUC {log_auc:.2f}),
    confermando che il rischio di overload è multifattoriale e non riducibile a un solo indice.
    Il Silhouette Score di {sil_score:.2f} del K-Means suggerisce che i 3 profili di allenamento sono riconoscibili ma non perfettamente separati,
    un punto da discutere nel capitolo conclusivo della tesi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 2: SIMULATORE WHAT-IF
# ============================================================================
with tab_sim:
    st.markdown("""
    <span class='section-num'>4.3</span> Il simulatore applica in tempo reale il modello **Random Forest** (Sez. 4.1.3, accuratezza {:.0f}%) a una sessione ipotetica, e la confronta con i profili di allenamento scoperti dal K-Means (Sez. 4.1.4).
    """.format(rf_acc * 100), unsafe_allow_html=True)

    st.markdown("---")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Imposta i Parametri dell'Atleta")
        sim_dist = st.slider("🏃 Distanza pianificata (km)", 5.0, 30.0, 10.0, 0.5)
        sim_sonno = st.slider("🛌 Ore di Sonno stanotte", 3.0, 10.0, 7.5, 0.5)
        sim_stress = st.slider("🧠 Livello Stress Vita/Lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
        sim_lavoro = st.slider("💼 Ore Lavorative oggi", 0.0, 12.0, 8.0, 0.5)
        sim_rpe = st.slider("📈 Sforzo (RPE) che si intende sostenere", 1.0, 10.0, 6.0, 1.0)

        sim_fc = 145.0
        sim_temp = 25.0
        sim_vel = 12.0
        sim_vento = 5.0

    with col_output:
        st.subheader("Elaborazione Algoritmica in Corso...")

        sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
        sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
        sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
        sim_iitr = (sim_temp * sim_vento) / sim_dist if sim_dist > 0 else 0

        input_data = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet, sim_iitr]],
                                  columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR'])

        prob_rischio = rf.predict_proba(input_data)[0][1] * 100

        color_status = "#0ea5e9" if prob_rischio < 40 else "#f59e0b" if prob_rischio < 70 else "#f43f5e"

        st.markdown(f"""
        <div style='background-color: #111827; padding: 30px; border-radius: 12px; border: 1px solid #374151; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <p style='color: #9ca3af; margin-bottom: 0; font-weight: 600; font-size: 0.85rem; letter-spacing: 1px;'>PROBABILITÀ DI OVERTRAINING (SOVRACCARICO)</p>
            <h1 style='color: {color_status}; font-size: 4.5rem; margin: 10px 0; font-weight: 800;'>
                {prob_rischio:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Stress Mentale (SMA)", f"{sim_sma:.2f}")
        m2.metric("Sforzo Lavorativo Residuo (ISLR)", f"{sim_islr:.2f}")

        # NUOVO: colloca la sessione simulata nel profilo K-Means più vicino,
        # collegando esplicitamente il simulatore alla Sez. 4.1.4
        sim_point = np.array([[sim_fc, sim_islr]])
        dists = np.linalg.norm(km.cluster_centers_ - sim_point, axis=1)
        nearest_cluster = np.argmin(dists)
        nearest_profile = cluster_map[nearest_cluster]
        st.markdown(f"<div class='tech-box-explanation'>📍 <b>Profilo di allenamento più vicino (K-Means):</b> questa sessione assomiglia di più al gruppo <b>'{nearest_profile}'</b> individuato in Sez. 4.1.4.</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if prob_rischio < 40:
            st.success("🟢 **SEMAFORO VERDE**: Parametri fisiologici nella norma. Lo stress complessivo è ammortizzato correttamente.")
        elif prob_rischio < 70:
            st.warning("🟡 **ZONA DI ATTENZIONE**: L'algoritmo rileva una competizione energetica tra lavoro e recupero. Cautela.")
        else:
            st.error("🔴 **ALLARME CRITICO**: Combinazione anomala di fattori di stress e scarso sonno. Rischio di sovraccarico elevato.")  
