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
# CONFIGURAZIONE INIZIALE E DESIGN SYSTEM (UI/UX REFINED)
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Tesi Magistrale",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Styling generale e font moderno */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Hero Title Refined */
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 40px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #9ca3af;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Box informativi e teorici */
    .explanation-box {
        background-color: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 16px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
    }
    .theory-box {
        background-color: rgba(251, 191, 36, 0.08);
        border-left: 4px solid #fbbf24;
        padding: 16px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
    }
    
    /* Miglioramento visivo degli expander */
    .streamlit-expanderHeader {
        background-color: #1f2937 !important;
        border-radius: 8px !important;
        border: 1px solid #374151 !important;
        font-weight: 600 !important;
        color: #f3f4f6 !important;
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
    
    # Performance di base
    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60
    df['Tempo (min)'] += np.random.normal(0, 5, n) 
    
    # Calcolo KPI Proprietari di supporto nascosti/integrati nei modelli
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    # Overload dinamico
    risk_score = (df['ISLR'] * 0.5) + (df['IDET'] * 0.02) - (df['Ore Sonno'] * 0.6)
    df['Rischio Overload'] = (risk_score > risk_score.quantile(0.70)).astype(int)
    
    return df

df = generate_thesis_data()

# ============================================================================
# TITOLO ED HEADER REFINED
# ============================================================================
st.markdown("""
<div class='hero-container'>
    <div class='hero-title'>Advanced Machine Learning Suite</div>
    <div class='hero-subtitle'>Framework predittivo di tesi magistrale per l'analisi dei carichi atletici, stima della performance e mitigazione proattiva del sovraccarico.</div>
</div>
""", unsafe_allow_html=True)

# Definiamo i tab principali puliti (senza la tab separata dei KPI)
tab_ml, tab_sim = st.tabs([
    "🧠 Suite Algoritmica & Modelli ML (Tendine)", 
    "🎮 Simulatore Predittivo What-If"
])

# ============================================================================
# TAB 1: MODELLI MACHINE LEARNING INTERAMENTE A TENDINA (EXPANDER)
# ============================================================================
with tab_ml:
    st.markdown("### Esplora l'Architettura dei Modelli Predittivi")
    st.markdown("Seleziona e apri i singoli moduli sottostanti per analizzare la matematica sottostante, le visualizzazioni grafiche interattive e l'interpretazione dei risultati.")

    # ---------------------------------------------------------
    # 1. REGRESSIONE LINEARE
    # ---------------------------------------------------------
    with st.expander("📈 1. Regressione Lineare (OLS Trend Prediction)", expanded=True):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Modella la relazione continua tra distanza percorsa e tempo di esecuzione, minimizzando l'errore quadratico medio per stimare la prestazione attesa.</div>", unsafe_allow_html=True)
        
        X_lr = df[['Distanza (km)']].values
        y_lr = df['Tempo (min)'].values
        lr_model = LinearRegression().fit(X_lr, y_lr)
        df['Tempo_Predetto'] = lr_model.predict(X_lr)
        df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']

        c1, c2 = st.columns(2)
        with c1:
            fig_lr1 = go.Figure()
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo (min)'], mode='markers', name='Dati Reali', marker=dict(color='#38bdf8', opacity=0.7)))
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo_Predetto'], mode='lines', name='Trend OLS', line=dict(color='#f43f5e', width=3)))
            fig_lr1.update_layout(title="Relazione Distanza - Tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lr1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> La linea rossa esprime il trend ottimale calcolato dall'algoritmo OLS.</div>", unsafe_allow_html=True)

        with c2:
            fig_lr2 = px.histogram(df, x="Errore (Residuo)", nbins=20, title="Distribuzione Residui / Errori", color_discrete_sequence=['#a78bfa'])
            fig_lr2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lr2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> La distribuzione gaussiana centrata sullo zero conferma l'assenza di bias sistematici nel modello.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. REGRESSIONE LOGISTICA
    # ---------------------------------------------------------
    with st.expander("🎯 2. Regressione Logistica (Sigmoid Classification)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Mappa le variabili di sforzo in uno spazio di probabilità [0, 1] tramite funzione logistica, attivando la soglia di guardia al superamento del 50%.</div>", unsafe_allow_html=True)
        
        X_log = df[['ISLR']].values
        y_log = df['Rischio Overload'].values
        log_model = LogisticRegression().fit(X_log, y_log)
        
        x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
        y_prob = log_model.predict_proba(x_range)[:, 1]
        df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]

        c1, c2 = st.columns(2)
        with c1:
            fig_log1 = go.Figure()
            fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Sessioni', marker=dict(color='#94a3b8', opacity=0.5)))
            fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines', name='Curva Sigmoide', line=dict(color='#fbbf24', width=3)))
            fig_log1.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Soglia Critica (50%)")
            fig_log1.update_layout(title="Curva di Transizione all'Overload", xaxis_title="ISLR", yaxis_title="Probabilità Predetta", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_log1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Oltrepassata la linea tratteggiata, la probabilità di sovraccarico cresce esponenzialmente.</div>", unsafe_allow_html=True)

        with c2:
            fig_log2 = px.box(df, x="Rischio Overload", y="Probabilità_Overload", color="Rischio Overload", 
                              color_discrete_map={0: '#38bdf8', 1: '#f43f5e'}, title="Analisi di Separabilità")
            fig_log2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_log2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> I boxplot evidenziano la netta separazione tra classi sicure (0) e a rischio (1).</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. RANDOM FOREST
    # ---------------------------------------------------------
    with st.expander("🌳 3. Random Forest Classifier (Ensemble Decision Trees)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Combina alberi decisionali multipli per valutare la feature importance e catturare interazioni complesse non lineari.</div>", unsafe_allow_html=True)
        
        rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
        imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')

        c1, c2 = st.columns(2)
        with c1:
            fig_rf1 = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Feature Importance Globale", color='Importanza', color_continuous_scale="Teal")
            fig_rf1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rf1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Gerarchia delle variabili che guidano le decisioni dell'ensemble.</div>", unsafe_allow_html=True)

        with c2:
            top_2 = imp_df.tail(2)['Feature'].values
            fig_rf2 = px.scatter(df, x=top_2[0], y=top_2[1], color='Rischio Overload', 
                                   title=f"Spazio d'Interazione ({top_2[0]} vs {top_2[1]})",
                                   color_discrete_map={0: '#38bdf8', 1: '#f43f5e'})
            fig_rf2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rf2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Gli alberi mappano regioni di rischio non lineari nello spazio bidimensionale.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # ---------------------------------------------------------
    with st.expander("🔍 4. Clustering K-Means (Unsupervised Profiling)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Raggruppa le sessioni nello spazio multidimensionale in base a somiglianze geometriche, scoprendo pattern latenti senza etichette preliminari.</div>", unsafe_allow_html=True)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[['FC Media', 'ISLR']])
        df['Cluster_ID'] = km.labels_
        cluster_map = {0: 'Rigenerativo', 1: 'Elevato Stress', 2: 'Qualità / Misto'}
        df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)

        c1, c2 = st.columns(2)
        with c1:
            fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa", 
                                   title="Cluster Spaziali delle Sessioni",
                                   color_discrete_sequence=['#34d399', '#f43f5e', '#fbbf24'])
            fig_km1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_km1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> I centroidi identificano tre macro-famiglie di carico distinte.</div>", unsafe_allow_html=True)

        with c2:
            cluster_means = df.groupby('Profilo_Corsa')[['Ore Sonno', 'Tempo (min)', 'RPE']].mean().reset_index()
            fig_km2 = px.bar(cluster_means, x='Profilo_Corsa', y=['Ore Sonno', 'RPE'], barmode='group', 
                                   title="Caratteristiche Medie dei Cluster")
            fig_km2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_km2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Profilazione comportamentale dei gruppi identificati.</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 2: SIMULATORE PREDITTIVO WHAT-IF
# ============================================================================
with tab_sim:
    st.markdown("### 🎮 Centrale Operativa: Simulatore What-If")
    st.markdown("Regola i cursori per simulare una sessione di allenamento in tempo reale. Il modello Random Forest calcolerà l'indice di rischio e fornirà un riscontro diagnostico immediato.")
    
    st.markdown("---")
    
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("#### Parametri di Carico & Recupero")
        sim_dist = st.slider("🏃 Distanza pianificata (km)", 5.0, 30.0, 10.0, 0.5)
        sim_sonno = st.slider("🛌 Ore di Sonno", 3.0, 10.0, 7.5, 0.5)
        sim_stress = st.slider("🧠 Livello Stress Vita/Lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
        sim_lavoro = st.slider("💼 Ore Lavorative giornaliere", 0.0, 12.0, 8.0, 0.5)
        sim_rpe = st.slider("📈 Sforzo Percepito (RPE 1-10)", 1.0, 10.0, 6.0, 1.0)
        
        sim_fc = 145.0
        sim_temp = 25.0
        sim_vel = 12.0
        sim_vento = 5.0
        
    with col_output:
        st.markdown("#### Inferenza in Tempo Reale")
        
        # Calcolo KPI interni per il modello
        sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
        sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
        sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
        sim_iitr = (sim_temp * sim_vento) / sim_dist if sim_dist > 0 else 0
        
        input_data = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet, sim_iitr]], 
                                  columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR'])
        
        prob_rischio = rf.predict_proba(input_data)[0][1] * 100
        
        color_risk = "#38bdf8" if prob_rischio < 40 else "#fbbf24" if prob_rischio < 70 else "#f43f5e"
        
        st.markdown(f"""
        <div style='background-color: #1f2937; padding: 30px; border-radius: 12px; border: 1px solid #374151; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <p style='color: #9ca3af; margin-bottom: 5px; font-weight: 600; font-size: 0.9rem;'>PROBABILITÀ DI OVERTRAINING STIMATA</p>
            <h1 style='color: {color_risk}; font-size: 4.5rem; margin: 0; font-weight: 800;'>
                {prob_rischio:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Indice SMA", f"{sim_sma:.2f}")
        m2.metric("Indice ISLR", f"{sim_islr:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if prob_rischio < 40:
            st.success("🟢 **SEMAFORO VERDE**: Carico fisiologico ottimale. Le risorse energetiche e il riposo ammortizzano ampiamente lo stress imposto.")
        elif prob_rischio < 70:
            st.warning("🟡 **ZONA DI ATTENZIONE**: L'algoritmo rileva segnali di affaticamento o competizione metabolica. Si consiglia cautela.")
        else:
            st.error("🔴 **ALLARME CRITICO**: Combinazione di fattori a rischio elevato (es. stress elevato unito a scarso sonno). Rischio infortunio imminente.")
