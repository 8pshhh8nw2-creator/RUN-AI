import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Data-Driven Sports Analytics", layout="wide", page_icon="🏃")

st.title("🏃 Data-Driven Sports Analytics")
st.markdown("""
Benvenuto nella piattaforma di analisi predittiva applicata alla performance sportiva. 
Questa dashboard dimostra come superare gli approcci tradizionali integrando metodologie data-driven per valorizzare i dati biometrici degli atleti amatori.
""")

# ==========================================
# GENERAZIONE DATI SIMULATI (MOCK DATA)
# ==========================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_sessions = 150
    
    # Variabili Base
    distanza = np.random.uniform(5, 21, n_sessions)
    fc_media = np.random.uniform(120, 180, n_sessions)
    velocita = np.random.uniform(8, 16, n_sessions)
    ore_sonno = np.random.uniform(4, 9, n_sessions)
    stress_lavoro = np.random.uniform(1, 10, n_sessions)
    ore_lavoro = np.random.uniform(0, 10, n_sessions)
    rpe = np.random.uniform(1, 10, n_sessions)
    gradi = np.random.uniform(15, 35, n_sessions)
    vento = np.random.uniform(0, 20, n_sessions)
    
    df = pd.DataFrame({
        'Distanza (km)': distanza,
        'FC Media (BPM)': fc_media,
        'Velocità (km/h)': velocita,
        'Tempo (min)': (distanza / velocita) * 60,
        'Ore Sonno': ore_sonno,
        'Stress Lavoro (1-10)': stress_lavoro,
        'Ore Lavoro': ore_lavoro,
        'RPE (1-10)': rpe,
        'Temperatura (°C)': gradi,
        'Vento (km/h)': vento
    })
    
    # Calcolo KPI Proprietari
    df['SMA'] = (df['Stress Lavoro (1-10)'] * df['RPE (1-10)']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro (1-10)']) / df['Distanza (km)']
    df['IITR'] = (df['Temperatura (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media (BPM)'] * df['Temperatura (°C)']) / df['Velocità (km/h)']
    
    # Creazione Variabile Target: Overload (1) o Normale (0)
    # Simuliamo che un alto ISLR e un basso sonno portino all'overload
    prob_overload = 1 / (1 + np.exp(-(0.5 * df['ISLR'] - 0.8 * df['Ore Sonno'] + 3)))
    df['Overload'] = (np.random.rand(n_sessions) < prob_overload).astype(int)
    
    return df

df = generate_data()

# ==========================================
# TABS NAVIGAZIONE
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 I KPI Proprietari", "🧠 Algoritmi di Machine Learning", "🎮 Simulatore Predittivo"])

# ------------------------------------------
# TAB 1: I KPI PROPRIETARI
# ------------------------------------------
with tab1:
    st.header("I 4 Pilastri dell'Analisi Personalizzata")
    st.markdown("Gli algoritmi standard falliscono perché ignorano il *carico allostatico* (lo stress della vita reale). Ecco i 4 indicatori ingegnerizzati per questa tesi:")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("**SMA (Stress Mentale dell'Allenamento)**")
        st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
        st.markdown("Misura l'impatto psicofisico integrando la fatica mentale e il recupero notturno.")
        
        st.warning("**IITR (Indice Impatto Termico e Resistenza)**")
        st.latex(r"IITR = \frac{\text{Temperatura} \times \text{Vento}}{\text{Distanza}}")
        st.markdown("Misura la severità ambientale subita dall'atleta durante la sessione.")

    with c2:
        st.success("**ISLR (Indice Sforzo Lavorativo Residuo)**")
        st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza}}")
        st.markdown("Rapporto tra lo stress non-atletico (lavorativo) e il chilometraggio percorso.")
        
        st.error("**IDET (Indice di Degradazione Termica)**")
        st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temperatura}}{\text{Velocità}}")
        st.markdown("Isola la *deriva cardiaca* causata dal caldo per evitare falsi allarmi sui cali di forma.")

    st.subheader("Distribuzione dei KPI nel Dataset")
    kpi_scelto = st.selectbox("Seleziona un KPI da visualizzare:", ['SMA', 'ISLR', 'IITR', 'IDET'])
    fig_kpi = px.histogram(df, x=kpi_scelto, color="Overload", nbins=30, 
                           title=f"Distribuzione di {kpi_scelto} (Suddiviso per stato di Overload)",
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    st.plotly_chart(fig_kpi, use_container_width=True)


# ------------------------------------------
# TAB 2: ALGORITMI DI MACHINE LEARNING
# ------------------------------------------
with tab2:
    st.header("L'Intelligenza Artificiale applicata allo Sport")
    
    # REGRESSIONE LINEARE
    st.subheader("1. Regressione Lineare: Volume vs Performance")
    st.markdown("Modella la relazione tra volume di allenamento e tempi di percorrenza riducendo l'errore al minimo (OLS).")
    fig_lr = px.scatter(df, x="Distanza (km)", y="Tempo (min)", trendline="ols", 
                        title="Impatto del Volume sul Tempo Totale")
    st.plotly_chart(fig_lr, use_container_width=True)
    
    st.divider()

    # REGRESSIONE LOGISTICA
    st.subheader("2. Regressione Logistica: Rischio Overload")
    st.markdown("Calcola la probabilità esatta (Curva Sigmoide) che l'atleta si trovi in uno stato di sovraccarico.")
    
    # Prep modello
    X_log = df[['ISLR']].values
    y_log = df['Overload'].values
    log_reg = LogisticRegression().fit(X_log, y_log)
    X_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
    y_prob = log_reg.predict_proba(X_range)[:, 1]
    
    fig_log = go.Figure()
    fig_log.add_trace(go.Scatter(x=df['ISLR'], y=df['Overload'], mode='markers', name='Dati Reali', marker=dict(color='orange')))
    fig_log.add_trace(go.Scatter(x=X_range.flatten(), y=y_prob, mode='lines', name='Curva Logistica (Probabilità)', line=dict(color='red')))
    fig_log.update_layout(title="Probabilità di Overload basata sull'ISLR", xaxis_title="ISLR", yaxis_title="Probabilità")
    st.plotly_chart(fig_log, use_container_width=True)

    st.divider()

    # RANDOM FOREST
    st.subheader("3. Random Forest: Feature Importance")
    st.markdown("Algoritmo ad alberi decisionali multipli. Gestisce relazioni non lineari e ci dice *quali* variabili pesano di più sul rischio infortuni.")
    
    features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
    X_rf = df[features]
    y_rf = df['Overload']
    rf = RandomForestClassifier(random_state=42).fit(X_rf, y_rf)
    
    imp_df = pd.DataFrame({'Feature': features, 'Importanza': rf.feature_importances_}).sort_values(by='Importanza', ascending=True)
    fig_rf = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Cosa causa maggiormente l'Overload?")
    st.plotly_chart(fig_rf, use_container_width=True)

    st.divider()

    # K-MEANS
    st.subheader("4. Clustering K-Means: Segmentazione Automatica")
    st.markdown("Raggruppa le sessioni simili tra loro (es. Recupero, Qualità, Affaticamento) calcolando le distanze geometriche (Centroidi).")
    
    km = KMeans(n_clusters=3, random_state=42).fit(df[['FC Media (BPM)', 'ISLR']])
    df['Cluster'] = km.labels_
    
    fig_km = px.scatter(df, x="ISLR", y="FC Media (BPM)", color=df['Cluster'].astype(str),
                        title="Segmentazione K-Means delle Corse",
                        labels={"color": "Tipo di Sessione (Cluster)"},
                        color_discrete_sequence=['#3498db', '#e67e22', '#9b59b6'])
    st.plotly_chart(fig_km, use_container_width=True)


# ------------------------------------------
# TAB 3: SIMULATORE
# ------------------------------------------
with tab3:
    st.header("🎮 Simulatore Prescrittivo Real-Time")
    st.markdown("""
    Cosa succede se un atleta corre 15 km dopo aver dormito solo 5 ore e con una giornata lavorativa stressante? 
    **Usa gli slider qui sotto per scoprirlo.** Il modello *Random Forest* elaborerà i dati all'istante.
    """)
    
    # Layout a colonne per gli input
    col1, col2 = st.columns(2)
    
    with col1:
        sim_distanza = st.slider("Distanza da percorrere (km)", 5.0, 30.0, 10.0)
        sim_sonno = st.slider("Ore di Sonno notte precedente", 3.0, 10.0, 7.0)
        sim_stress = st.slider("Stress Lavorativo (1-10)", 1.0, 10.0, 5.0)
        
    with col2:
        sim_lavoro = st.slider("Ore di Lavoro", 0.0, 12.0, 8.0)
        sim_rpe = st.slider("Sforzo percepito atteso (RPE 1-10)", 1.0, 10.0, 6.0)
        sim_fc = st.slider("Frequenza Cardiaca stimata (BPM)", 110.0, 190.0, 145.0)

    # Parametri fissi per semplificare il simulatore
    sim_gradi = 25.0
    sim_vento = 5.0
    sim_vel = 10.0
    
    # Calcolo dei KPI on the fly
    calc_sma = (sim_stress * sim_rpe) / sim_sonno
    calc_islr = (sim_lavoro * sim_stress) / sim_distanza
    calc_iitr = (sim_gradi * sim_vento) / sim_distanza
    calc_idet = (sim_fc * sim_gradi) / sim_vel
    
    # Creazione Array per la predizione
    input_data = pd.DataFrame([[sim_distanza, sim_sonno, calc_sma, calc_islr, calc_idet, calc_iitr]], columns=features)
    
    # Predizione
    prob = rf.predict_proba(input_data)[0][1]
    
    st.divider()
    st.subheader("Risultato dell'Analisi AI")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        color = "green" if prob < 0.4 else "orange" if prob < 0.7 else "red"
        st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 60px;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Probabilità di Overload</p>", unsafe_allow_html=True)
        
    with res_col2:
        if prob < 0.4:
            st.success("✅ **Luce Verde!** Il carico allostatico è sotto controllo. Puoi procedere con l'allenamento previsto.")
        elif prob < 0.7:
            st.warning("⚠️ **Attenzione.** Il rischio è moderato. Valuta di ridurre i chilometri o diminuire l'intensità (RPE) per non stressare troppo il sistema nervoso centrale.")
        else:
            st.error("🚨 **Pericolo di Overtraining!** Le tue condizioni psicofisiche o ambientali (Sonno, Stress, Lavoro) suggeriscono fortemente un giorno di Rest attivo o passivo.")
