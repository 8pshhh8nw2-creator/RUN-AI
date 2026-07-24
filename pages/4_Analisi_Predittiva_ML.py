import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
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

    /* Hero Banner Moderno */
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

    /* Sezioni Modelli */
    .model-container {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .model-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 15px;
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

# Helper per applicare il tema scuro pulito ai grafici Plotly
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#9ca3af"),
        margin=dict(t=30, b=20, l=20, r=20)
    )
    return fig

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

st.markdown("<br>", unsafe_allow_html=True)

# Tabs di navigazione principali
tab_ml, tab_sim = st.tabs([
    "🧠 Suite Modelli Machine Learning", 
    "🎮 Centrale Operativa Simulatore What-If"
])

# ============================================================================
# TAB 1: MODELLI ML DISPOSTI CHIARAMENTE (TITOLO -> SPIEGAZIONE -> GRAFICI DI LATO)
# ============================================================================
with tab_ml:
    st.markdown("### Analisi Approfondita dei Modelli Predittivi")
    st.markdown("Ciascun modulo algoritmico implementato nella suite risponde a una specifica esigenza analitica della tesi.")

    # ---------------------------------------------------------
    # 1. REGRESSIONE LINEARE
    # ---------------------------------------------------------
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'>📈 1. Regressione Lineare (OLS Trend Prediction)</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='tech-box-theory'><b>Fondamenti Teorici:</b> La regressione lineare modella la relazione tra una variabile dipendente continua (es. Tempo) e una indipendente (es. Distanza), minimizzando la discrepanza tra i valori reali e la linea di tendenza (Minimi Quadrati Ordinari).</div>", unsafe_allow_html=True)

    X_lr = df[['Distanza (km)']].values
    y_lr = df['Tempo (min)'].values
    lr_model = LinearRegression().fit(X_lr, y_lr)
    df['Tempo_Predetto'] = lr_model.predict(X_lr)
    df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']

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

    # ---------------------------------------------------------
    # 2. REGRESSIONE LOGISTICA
    # ---------------------------------------------------------
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'>🎯 2. Regressione Logistica (Sigmoid Classification)</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Algoritmo di classificazione che stima la probabilità che una sessione appartenga a uno stato critico, mappando le feature in uno spazio [0, 1] tramite funzione logistica (Sigmoide).</div>", unsafe_allow_html=True)

    X_log = df[['ISLR']].values
    y_log = df['Rischio Overload'].values
    log_model = LogisticRegression().fit(X_log, y_log)
    
    x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
    y_prob = log_model.predict_proba(x_range)[:, 1]
    df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]

    c1, c2 = st.columns(2)
    with c1:
        fig_log1 = go.Figure()
        fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Osservazioni', marker=dict(color='#64748b', opacity=0.5)))
        fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines', name='Curva Sigmoide', line=dict(color='#f59e0b', width=3)))
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

    # ---------------------------------------------------------
    # 3. RANDOM FOREST
    # ---------------------------------------------------------
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'>🌳 3. Random Forest Classifier (Ensemble Decision Trees)</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Opera combinando alberi di decisione multipli. Gestisce le interazioni non lineari e calcola la Feature Importance per identificare i fattori critici di stress.</div>", unsafe_allow_html=True)

    rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
    imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')

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

    # ---------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # ---------------------------------------------------------
    st.markdown("""
    <div class='model-container'>
        <div class='model-title'>🔍 4. Clustering K-Means (Unsupervised Segmentation)</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='tech-box-theory'><b>Fondamenti Teorici:</b> Raggruppa le sessioni in base a similitudini geometriche multidimensionali senza etichette preliminari, scoprendo cluster latenti nei dati.</div>", unsafe_allow_html=True)

    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[['FC Media', 'ISLR']])
    df['Cluster_ID'] = km.labels_
    cluster_map = {0: 'Rigenerativo', 1: 'Elevato Stress', 2: 'Qualità / Misto'}
    df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)

    c1, c2 = st.columns(2)
    with c1:
        fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa", 
                               title="Segmentazione Automatica delle Sessioni",
                               color_discrete_sequence=['#10b981', '#f43f5e', '#f59e0b'])
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

# ============================================================================
# TAB 2: SIMULATORE WHAT-IF
# ============================================================================
with tab_sim:
    st.markdown("### 🎮 Centrale Operativa: Simulatore What-If")
    st.markdown("Regola i cursori per simulare i parametri di una sessione. Il modello Random Forest calcolerà in tempo reale la probabilità di overtraining e fornirà una diagnostica istantanea.")
    
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        if prob_rischio < 40:
            st.success("🟢 **SEMAFORO VERDE**: Parametri fisiologici nella norma. Lo stress complessivo è ammortizzato correttamente.")
        elif prob_rischio < 70:
            st.warning("🟡 **ZONA DI ATTENZIONE**: L'algoritmo rileva una competizione energetica tra lavoro e recupero. Cautela.")
        else:
            st.error("🔴 **ALLARME CRITICO**: Combinazione anomala di fattori di stress e scarso sonno. Rischio di sovraccarico elevato.")
