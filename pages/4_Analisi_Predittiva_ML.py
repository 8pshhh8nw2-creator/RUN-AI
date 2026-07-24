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
# CONFIGURAZIONE INIZIALE & HIGH-TECH DESIGN SYSTEM
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Cyber-Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts High-Tech */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Global Base */
    .stApp {
        background-color: #030712;
        color: #f3f4f6;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Cyber Hero Banner */
    .cyber-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 35px 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .cyber-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(180deg, #38bdf8, #818cf8);
    }
    .cyber-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
    }
    .cyber-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }

    /* Code & Theory Boxes */
    .cyber-box-theory {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-left: 4px solid #fbbf24;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    .cyber-box-guide {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38bdf8;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #e2e8f0;
    }

    /* Expander High-Tech Styling */
    .streamlit-expanderHeader {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
        transition: all 0.3s ease;
    }
    .streamlit-expanderHeader:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
    }

    /* Code Fonts for data specs */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MOTORE DATI SIMULATI (CACHED)
# ============================================================================
@st.cache_data
def generate_thesis_data():
    np.random.seed(42)
    n = 350
    
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
    
    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60 + np.random.normal(0, 4, n)
    
    # KPI proprietari
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    risk_score = (df['ISLR'] * 0.5) + (df['IDET'] * 0.02) - (df['Ore Sonno'] * 0.6)
    df['Rischio Overload'] = (risk_score > risk_score.quantile(0.70)).astype(int)
    
    return df

df = generate_thesis_data()

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class='cyber-hero'>
    <div class='cyber-title'>ADVANCED ML SUITE // 01</div>
    <div class='cyber-subtitle'>CORE TELEMETRY & PREDICTIVE ANALYTICS ENGINE // THESIS MASTER SUITE</div>
</div>
""", unsafe_allow_html=True)

# Layout Tabs Principali
tab_ml, tab_sim = st.tabs([
    "⚡ MODULI ALGORITMICI & MACHINE LEARNING", 
    "🎯 CENTRALE OPERATIVA WHAT-IF SIMULATOR"
])

# Plotly dark template helper
def apply_cyber_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
        margin=dict(t=40, b=30, l=30, r=30),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="rgba(56, 189, 248, 0.2)")
    )
    return fig

# ============================================================================
# TAB 1: MODELLI MACHINE LEARNING (EXPANDER HIGH-TECH)
# ============================================================================
with tab_ml:
    st.markdown("### 🧬 Architectural Pipeline & Model Breakdown")
    st.markdown("Seleziona i moduli sottostanti per ispezionare il comportamento dei modelli predittivi nello spazio multidimensionale.")

    # 1. REGRESSIONE LINEARE
    with st.expander("📈 [MOD_01] Linear Regression (OLS Trend Prediction)", expanded=True):
        st.markdown("<div class='cyber-box-theory'><b>FONDAENTO MATEMATICO:</b> Ottimizzazione dei minimi quadrati ordinari (OLS) per stimare il tempo atteso di percorrenza in funzione del volume chilometrico.</div>", unsafe_allow_html=True)
        
        X_lr = df[['Distanza (km)']].values
        y_lr = df['Tempo (min)'].values
        lr_model = LinearRegression().fit(X_lr, y_lr)
        df['Tempo_Predetto'] = lr_model.predict(X_lr)
        df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']

        c1, c2 = st.columns(2)
        with c1:
            fig_lr1 = go.Figure()
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo (min)'], mode='markers', name='Telemetry Data', marker=dict(color='#38bdf8', opacity=0.6, size=7)))
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo_Predetto'], mode='lines', name='OLS Regression Fit', line=dict(color='#f43f5e', width=3)))
            apply_cyber_theme(fig_lr1)
            fig_lr1.update_layout(title="Distance vs Time Regressed Curve", xaxis_title="Distanza [km]", yaxis_title="Tempo [min]")
            st.plotly_chart(fig_lr1, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> La curva lineare evidenzia l'andamento atteso della prestazione escludendo le anomalie di sforzo.</div>", unsafe_allow_html=True)

        with c2:
            fig_lr2 = px.histogram(df, x="Errore (Residuo)", nbins=20, title="Residues Distribution (Gauss Error)", color_discrete_sequence=['#818cf8'])
            apply_cyber_theme(fig_lr2)
            st.plotly_chart(fig_lr2, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> La distribuzione centrata sullo zero conferma la correttezza omoschedastica del modello lineare.</div>", unsafe_allow_html=True)

    # 2. REGRESSIONE LOGISTICA
    with st.expander("🎯 [MOD_02] Logistic Regression (Sigmoid Overload Classification)", expanded=False):
        st.markdown("<div class='cyber-box-theory'><b>FONDAENTO MATEMATICO:</b> Mappatura probabilistica non lineare tramite funzione sigmoidea per la classificazione binaria dello stato critico di fatica.</div>", unsafe_allow_html=True)
        
        X_log = df[['ISLR']].values
        y_log = df['Rischio Overload'].values
        log_model = LogisticRegression().fit(X_log, y_log)
        
        x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
        y_prob = log_model.predict_proba(x_range)[:, 1]
        df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]

        c1, c2 = st.columns(2)
        with c1:
            fig_log1 = go.Figure()
            fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Sessions', marker=dict(color='#64748b', opacity=0.5, size=6)))
            fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines', name='Sigmoid Probability Curve', line=dict(color='#fbbf24', width=3)))
            fig_log1.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Decision Threshold [50%]")
            apply_cyber_theme(fig_log1)
            fig_log1.update_layout(title="Sigmoid Transition State", xaxis_title="ISLR Metric", yaxis_title="Probability [0-1]")
            st.plotly_chart(fig_log1, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Superato il valore di soglia ISLR critico, la probabilità di sovraccarico impenna verso l'area di attenzione.</div>", unsafe_allow_html=True)

        with c2:
            fig_log2 = px.box(df, x="Rischio Overload", y="Probabilità_Overload", color="Rischio Overload", 
                              color_discrete_map={0: '#38bdf8', 1: '#f43f5e'}, title="Class Separation Spread")
            apply_cyber_theme(fig_log2)
            st.plotly_chart(fig_log2, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Netta separazione tra distribuzioni di probabilità delle sessioni safe (0) e risk (1).</div>", unsafe_allow_html=True)

    # 3. RANDOM FOREST
    with st.expander("🌳 [MOD_03] Random Forest Classifier (Ensemble Feature Importance)", expanded=False):
        st.markdown("<div class='cyber-box-theory'><b>FONDAENTO MATEMATICO:</b> Modello d'insieme a bagging di alberi decisionali per l'estrazione delle interazioni non lineari e della feature importance.</div>", unsafe_allow_html=True)
        
        rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
        imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')

        c1, c2 = st.columns(2)
        with c1:
            fig_rf1 = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Global Feature Importance", color='Importanza', color_continuous_scale="Teal")
            apply_cyber_theme(fig_rf1)
            st.plotly_chart(fig_rf1, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Gerarchizzazione dei fattori biologici e contestuali che determinano il rischio d'infortunio.</div>", unsafe_allow_html=True)

        with c2:
            top_2 = imp_df.tail(2)['Feature'].values
            fig_rf2 = px.scatter(df, x=top_2[0], y=top_2[1], color='Rischio Overload', 
                                   title=f"Non-Linear Space ({top_2[0]} vs {top_2[1]})",
                                   color_discrete_map={0: '#38bdf8', 1: '#f43f5e'})
            apply_cyber_theme(fig_rf2)
            st.plotly_chart(fig_rf2, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Mappatura geometrica bidimensionale dei cluster di rischio rilevati dalla foresta.</div>", unsafe_allow_html=True)

    # 4. K-MEANS CLUSTERING
    with st.expander("🔍 [MOD_04] Unsupervised K-Means Clustering", expanded=False):
        st.markdown("<div class='cyber-box-theory'><b>FONDAENTO MATEMATICO:</b> Partizionamento geometrico non supervisionato dello spazio feature basato sulla minimizzazione della varianza intra-cluster.</div>", unsafe_allow_html=True)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df[['FC Media', 'ISLR']])
        df['Cluster_ID'] = km.labels_
        cluster_map = {0: 'Rigenerativo', 1: 'Elevato Stress', 2: 'Qualità / Misto'}
        df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)

        c1, c2 = st.columns(2)
        with c1:
            fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa", 
                                   title="Unsupervised Cluster Map",
                                   color_discrete_sequence=['#34d399', '#f43f5e', '#fbbf24'])
            apply_cyber_theme(fig_km1)
            st.plotly_chart(fig_km1, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Raggruppamento automatico delle sessioni in macro-famiglie funzionali prive di etichettatura preventiva.</div>", unsafe_allow_html=True)

        with c2:
            cluster_means = df.groupby('Profilo_Corsa')[['Ore Sonno', 'Tempo (min)', 'RPE']].mean().reset_index()
            fig_km2 = px.bar(cluster_means, x='Profilo_Corsa', y=['Ore Sonno', 'RPE'], barmode='group', 
                                   title="Cluster Behavioral Fingerprint")
            apply_cyber_theme(fig_km2)
            st.plotly_chart(fig_km2, use_container_width=True)
            st.markdown("<div class='cyber-box-guide'><b>ANALISI:</b> Profilazione energetica media dei cluster individuati dall'algoritmo.</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 2: SIMULATORE WHAT-IF
# ============================================================================
with tab_sim:
    st.markdown("### 🎯 Real-Time Predictive Control Unit (What-If Engine)")
    st.markdown("Modifica i parametri atletici per interrogare istantaneamente il modello Random Forest addestrato.")
    
    st.markdown("---")
    
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("#### [INPUT] Session Telemetry & Variables")
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
        st.markdown("#### [OUTPUT] AI Diagnostic Inference")
        
        sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
        sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
        sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
        sim_iitr = (sim_temp * sim_vento) / sim_dist if sim_dist > 0 else 0
        
        input_data = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet, sim_iitr]], 
                                  columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR'])
        
        prob_rischio = rf.predict_proba(input_data)[0][1] * 100
        
        color_glow = "#38bdf8" if prob_rischio < 40 else "#fbbf24" if prob_rischio < 70 else "#f43f5e"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                    padding: 35px; border-radius: 16px; border: 1px solid {color_glow}; 
                    text-align: center; box-shadow: 0 0 25px rgba(56, 189, 248, 0.1);'>
            <p style='color: #94a3b8; margin: 0 0 10px 0; font-family: "JetBrains Mono", monospace; font-size: 0.85rem; letter-spacing: 1px;'>OVERTRAINING PROBABILITY INDEX</p>
            <h1 style='color: {color_glow}; font-size: 5rem; margin: 0; font-weight: 800; font-family: "JetBrains Mono", monospace;'>
                {prob_rischio:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Calculated SMA", f"{sim_sma:.2f}")
        m2.metric("Calculated ISLR", f"{sim_islr:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if prob_rischio < 40:
            st.success("🟢 **SYSTEM STATUS: SAFE**: Parametri biologici e di carico perfettamente bilanciati.")
        elif prob_rischio < 70:
            st.warning("🟡 **SYSTEM STATUS: WARNING**: Rilevata competizione metabolica/nervosa. Procedere con cautela.")
        else:
            st.error("🔴 **SYSTEM STATUS: CRITICAL ERROR**: Rischio di sovraccarico elevato. Modificare i parametri di sessione.")
