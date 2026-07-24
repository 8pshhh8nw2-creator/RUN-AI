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
# CONFIGURAZIONE INIZIALE E STILI
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Explainability Suite | Tesi Magistrale",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# CSS Custom per il layout "Dark/Tech" originale
st.markdown("""
<style>
    .hero-container {
        background: linear-gradient(135deg, rgba(0,229,255,0.1) 0%, rgba(32,40,58,0.8) 100%);
        border-radius: 12px; padding: 24px; margin-bottom: 28px;
        border: 1px solid rgba(0,229,255,0.15);
    }
    .kpi-card {
        background: rgba(32,40,58,0.5); border-left: 4px solid; padding: 15px; 
        border-radius: 8px; margin-bottom: 15px; height: 100%;
    }
    .highlight-text { color: #00E5FF; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GENERAZIONE DATI (ROBUSTA E SICURA)
# ============================================================================
@st.cache_data
def generate_safe_data():
    """Genera un dataset robusto senza valori nulli per evitare crash nei modelli"""
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'Distanza (km)': np.random.uniform(5, 25, n),
        'FC Media': np.random.uniform(125, 175, n),
        'Velocità (km/h)': np.random.uniform(9, 15, n),
        'Ore Sonno': np.random.uniform(4, 9, n),
        'Stress Lavoro': np.random.uniform(1, 10, n),
        'Ore Lavoro': np.random.uniform(0, 10, n),
        'RPE': np.random.uniform(1, 10, n),
        'Temp (°C)': np.random.uniform(15, 35, n),
        'Vento (km/h)': np.random.uniform(0, 20, n)
    })
    
    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60
    
    # Calcolo KPI Proprietari
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    # Logica per l'Overload (Target 0/1)
    # Alto ISLR, basso sonno e alto IDET aumentano drasticamente il rischio
    risk_score = (df['ISLR'] * 0.4) + (df['IDET'] * 0.01) - (df['Ore Sonno'] * 0.5)
    df['Rischio Overload'] = (risk_score > risk_score.quantile(0.65)).astype(int)
    
    return df

df = generate_safe_data()

# ============================================================================
# HEADER & SPIEGAZIONE MACHINE LEARNING
# ============================================================================
st.markdown("""
<div class='hero-container'>
    <h1 style='color: #00E5FF; margin-top: 0; font-size: 2.2em;'>L'era dell'allenamento data-driven</h1>
    <h3 style='color: #FFFFFF; font-weight: 400;'>Ottimizzazione della performance atletica attraverso il Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("### Che cos'è il Machine Learning (Apprendimento Automatico)?")
col_ml1, col_ml2 = st.columns([2, 1])

with col_ml1:
    st.markdown("""
    Nello sport tradizionale, l'allenamento si basava su modelli standardizzati e tabelle rigide che faticavano ad adattarsi alla variabilità individuale dell'atleta[cite: 2]. Il **Machine Learning (ML)** cambia questo paradigma.
    
    Il ML non è "magia", ma statistica computazionale avanzata. È una branca dell'Intelligenza Artificiale che permette ai sistemi informatici di imparare dai dati storici senza essere esplicitamente programmati. In ambito sportivo, il Machine Learning funge da **assistente decisionale**[cite: 2]:
    
    * **Trova pattern invisibili:** Riesce a calcolare relazioni complesse (spesso non lineari) tra il lavoro fisico oggettivo (velocità, distanza) e la risposta biologica del corpo (battito cardiaco, sforzo percepito)[cite: 2].
    * **Da Reattivo a Proattivo:** Invece di accorgersi dell'affaticamento quando l'infortunio è già avvenuto, i modelli predittivi analizzano lo storico dell'atleta per anticipare il rischio di sovrallenamento (overtraining)[cite: 2].
    * **Personalizzazione:** Transforma i dati grezzi raccolti da smartwatch e sensori IoT in conoscenze pratiche[cite: 2], creando profili di carico su misura per l'atleta amatore, democratizzando metodologie riservate finora solo agli atleti d'élite[cite: 2].
    """)
with col_ml2:
    st.info("""
    **I 4 Algoritmi della Suite:**
    1. **Regressione Lineare:** Per le proiezioni di performance (es. Volume vs Tempo).
    2. **Regressione Logistica:** Per il calcolo delle probabilità critiche (Sano vs Infortunio).
    3. **Random Forest:** Per capire l'importanza delle variabili (Cosa causa lo stress?).
    4. **K-Means Clustering:** Per raggruppare e scoprire i tipi di allenamento.
    """)

st.markdown("---")

# ============================================================================
# TABS PRINCIPALI
# ============================================================================
tab_kpi, tab_models, tab_sim = st.tabs([
    "⭐ 1. I KPI Proprietari", 
    "🧠 2. Modelli Predittivi ML", 
    "🎮 3. Simulatore What-If"
])

# ----------------------------------------------------------------------------
# TAB 1: I KPI PROPRIETARI (Il Tuo Core)
# ----------------------------------------------------------------------------
with tab_kpi:
    st.markdown("### L'Ingegnerizzazione delle Feature (I Pilastri della Tesi)")
    st.markdown("Per superare i limiti degli algoritmi tradizionali che ignorano lo stress della vita reale, ho sviluppato quattro indicatori (KPI) che combinano metriche soggettive, oggettive e ambientali[cite: 2].")
    
    k1, k2 = st.columns(2)
    
    with k1:
        st.markdown("""
        <div class='kpi-card' style='border-color: #00E5FF;'>
            <h4 style='margin:0; color:#00E5FF;'>SMA (Stress Mentale dell'Allenamento)</h4>
            <p style='font-size:0.9em; color:#B8C2D0;'>Quantifica l'impatto psicofisico integrando la stanchezza cognitiva giornaliera e la qualità del recupero notturno[cite: 2].</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
        
        st.markdown("""
        <div class='kpi-card' style='border-color: #FFB020; margin-top:20px;'>
            <h4 style='margin:0; color:#FFB020;'>ISLR (Indice di Sforzo Lavorativo Residuo)</h4>
            <p style='font-size:0.9em; color:#B8C2D0;'>Rapporto tra il carico cognitivo lavorativo (che compete con le risorse energetiche) e la distanza coperta[cite: 2]. Perfetto per l'atleta amatore.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")

    with k2:
        st.markdown("""
        <div class='kpi-card' style='border-color: #FF6A3D;'>
            <h4 style='margin:0; color:#FF6A3D;'>IITR (Indice Impatto Termico e Resistenza)</h4>
            <p style='font-size:0.9em; color:#B8C2D0;'>Misura la severità ambientale combinando le forze resistive esogene: il calore e la resistenza aerodinamica del vento[cite: 2].</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"IITR = \frac{\text{Gradi Celsius} \times \text{Velocità Vento (km/h)}}{\text{Distanza (km)}}")
        
        st.markdown("""
        <div class='kpi-card' style='border-color: #6AFF87; margin-top:20px;'>
            <h4 style='margin:0; color:#6AFF87;'>IDET (Indice di Degradazione Termica)</h4>
            <p style='font-size:0.9em; color:#B8C2D0;'>Mappa l'efficienza meccanica per evitare che il modello scambi la deriva cardiaca estiva per una reale perdita di forma fisica[cite: 2].</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"IDET = \frac{\text{FC Media} \times \text{Gradi Celsius}}{\text{Velocità (km/h)}}")

    st.markdown("#### Analisi Visiva dei KPI Proprietari")
    scelta_kpi = st.radio("Scegli un KPI per visualizzarne la distribuzione e l'impatto sul rischio:", ['SMA', 'ISLR', 'IITR', 'IDET'], horizontal=True)
    
    fig_kpi = px.histogram(
        df, x=scelta_kpi, color='Rischio Overload', nbins=25,
        color_discrete_map={0: '#00E5FF', 1: '#FF6A3D'},
        labels={'Rischio Overload': 'Stato'},
        title=f"Distribuzione {scelta_kpi}: Sessioni Sicure (Azzurro) vs Rischio Overload (Rosso)"
    )
    fig_kpi.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_kpi, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: MACHINE LEARNING (Pulito, Funzionante, Professionale)
# ----------------------------------------------------------------------------
with tab_models:
    st.markdown("### L'Applicazione Pratica del Machine Learning")
    
    m1, m2 = st.columns(2)
    
    # 1. REGRESSIONE LINEARE
    with m1:
        st.markdown("#### 📈 1. Regressione Lineare")
        st.markdown("*Obiettivo: Modellare la relazione tra volumi di allenamento e tempi[cite: 2].*")
        
        fig_lr = px.scatter(
            df, x="Distanza (km)", y="Tempo (min)", trendline="ols",
            color_discrete_sequence=['#00E5FF']
        )
        fig_lr.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_lr, use_container_width=True)

    # 2. REGRESSIONE LOGISTICA
    with m2:
        st.markdown("#### 🎯 2. Regressione Logistica")
        st.markdown("*Obiettivo: Classificazione dicotomica per prevedere lo stato di Overload[cite: 2].*")
        
        # Prep veloce per la sigmoide
        X_log = df[['ISLR']].values
        y_log = df['Rischio Overload'].values
        clf = LogisticRegression().fit(X_log, y_log)
        x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 200).reshape(-1, 1)
        y_prob = clf.predict_proba(x_range)[:, 1]
        
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Dati', marker_color='#8792A3'))
        fig_log.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines', name='Probabilità', line_color='#FF6A3D'))
        fig_log.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Probabilità (0 a 1)")
        st.plotly_chart(fig_log, use_container_width=True)

    m3, m4 = st.columns(2)
    
    # 3. RANDOM FOREST
    with m3:
        st.markdown("#### 🌳 3. Random Forest (Feature Importance)")
        st.markdown("*Obiettivo: Gestire le non linearità e gerarchizzare l'importanza delle variabili[cite: 2].*")
        
        rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET']
        rf_model = RandomForestClassifier(random_state=42).fit(df[rf_features], df['Rischio Overload'])
        imp = pd.DataFrame({'Feature': rf_features, 'Importanza': rf_model.feature_importances_}).sort_values('Importanza')
        
        fig_rf = px.bar(imp, x='Importanza', y='Feature', orientation='h', color_discrete_sequence=['#FFB020'])
        fig_rf.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_rf, use_container_width=True)

    # 4. K-MEANS
    with m4:
        st.markdown("#### 🔍 4. Clustering K-Means")
        st.markdown("*Obiettivo: Raggruppare le sessioni simili minimizzando la distanza geometrica[cite: 2].*")
        
        km = KMeans(n_clusters=3, random_state=42).fit(df[['FC Media', 'ISLR']])
        df['Cluster'] = km.labels_.astype(str)
        
        fig_km = px.scatter(
            df, x="ISLR", y="FC Media", color="Cluster", 
            color_discrete_sequence=['#00E5FF', '#FF6A3D', '#6AFF87']
        )
        fig_km.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_km, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 3: SIMULATORE WHAT-IF (Sicuro e Immediato)
# ----------------------------------------------------------------------------
with tab_sim:
    st.markdown("### 🎮 Simulatore dell'Allenamento")
    st.markdown("Questo simulatore utilizza il modello **Random Forest** appena addestrato per calcolare in tempo reale il rischio in base ai parametri che inserisci. Prova a diminuire le ore di sonno o aumentare lo stress per vedere l'impatto sui KPI e sul rischio.")
    
    s1, s2 = st.columns(2)
    
    with s1:
        st.markdown("#### Parametri Base")
        sim_dist = st.slider("Distanza (km)", 5.0, 30.0, 10.0, 0.5)
        sim_sonno = st.slider("Ore Sonno", 4.0, 10.0, 7.5, 0.5)
        sim_stress = st.slider("Stress Lavoro (1-10)", 1.0, 10.0, 5.0, 1.0)
    
    with s2:
        st.markdown("#### Parametri Avanzati")
        sim_lavoro = st.slider("Ore Lavoro", 0.0, 12.0, 8.0, 0.5)
        sim_rpe = st.slider("RPE Atteso (1-10)", 1.0, 10.0, 6.0, 1.0)
        sim_fc = st.slider("FC Media Stimata", 120.0, 180.0, 145.0, 1.0)
        
    # Variabili fisse per il calcolo IDET
    sim_temp = 25.0
    sim_vel = 12.0
        
    # Calcolo Live dei KPI
    sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
    sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
    sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
    
    # Preparazione array predizione
    # Deve combaciare con rf_features: ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET']
    input_df = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet]], 
                            columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET'])
    
    prob_rischio = rf_model.predict_proba(input_df)[0][1] * 100
    
    st.markdown("---")
    
    # Risultato Visivo
    res_text = "OTTIMALE" if prob_rischio < 35 else "ATTENZIONE" if prob_rischio < 65 else "PERICOLO OVERLOAD"
    res_color = "#00E5FF" if prob_rischio < 35 else "#FFB020" if prob_rischio < 65 else "#FF6A3D"
    
    r1, r2, r3 = st.columns([1, 1, 2])
    
    with r1:
        st.metric("Valore SMA Ricalcolato", f"{sim_sma:.1f}")
        st.metric("Valore ISLR Ricalcolato", f"{sim_islr:.1f}")
        
    with r2:
        st.markdown(f"""
        <div style='text-align:center; padding:15px; border-radius:10px; background:rgba(255,255,255,0.05);'>
            <div style='font-size:3em; font-weight:bold; color:{res_color};'>{prob_rischio:.0f}%</div>
            <div style='color:#B8C2D0;'>Probabilità Infortunio</div>
        </div>
        """, unsafe_allow_html=True)
        
    with r3:
        st.markdown(f"<h3 style='color:{res_color};'>{res_text}</h3>", unsafe_allow_html=True)
        if prob_rischio < 35:
            st.write("Il profilo di carico è bilanciato. I tuoi indici (ISLR, SMA) indicano che lo stress lavorativo e la fatica sono gestibili in base alle ore di sonno.")
        elif prob_rischio < 65:
            st.write("Sei in una zona grigia. L'algoritmo rileva un accumulo di stress. Cerca di non aumentare ulteriormente la distanza se non migliori le ore di sonno.")
        else:
            st.write("L'interazione non lineare tra il poco sonno, l'alto stress lavorativo e i parametri di corsa scatenano un forte rischio biologico. Il modello suggerisce uno scarico immediato.")
