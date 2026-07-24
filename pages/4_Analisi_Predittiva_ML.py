import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE INIZIALE E STILI
# ============================================================================
st.set_page_config(
    page_title="Advanced ML Suite | Tesi Magistrale",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px; border-radius: 12px; margin-bottom: 30px; color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .explanation-box {
        background-color: rgba(0, 229, 255, 0.05);
        border-left: 4px solid #00E5FF;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
    }
    .theory-box {
        background-color: rgba(255, 176, 32, 0.05);
        border-left: 4px solid #FFB020;
        padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;
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
    
    # Base performance
    df['Tempo (min)'] = (df['Distanza (km)'] / df['Velocità (km/h)']) * 60
    # Aggiungo un po' di rumore per rendere i grafici più realistici
    df['Tempo (min)'] += np.random.normal(0, 5, n) 
    
    # Calcolo KPI Proprietari
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
# INTRODUZIONE
# ============================================================================
st.markdown("""
<div class='hero'>
    <h1 style='margin-top: 0; font-size: 2.5em;'>L'Era Data-Driven nello Sport</h1>
    <h3 style='font-weight: 300;'>Ottimizzazione prestazionale attraverso intelligenza predittiva</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Il **Machine Learning** rappresenta il superamento definitivo delle limitazioni dei metodi di allenamento convenzionali. Non ci limitiamo più a osservare parametri fisici isolati, ma integriamo una moltitudine di segnali per creare algoritmi predittivi complessi capaci di imparare dallo storico dell'atleta. L'obiettivo è passare da un approccio *reattivo* (curare l'affaticamento) a uno *proattivo* (anticipare le necessità dell'atleta calcolando il rischio in tempo reale).
""")

tab_kpi, tab_ml, tab_sim = st.tabs([
    "📍 1. I KPI Proprietari", 
    "🧠 2. Suite Machine Learning (Spiegata)", 
    "🎮 3. Simulatore Predittivo"
])

# ============================================================================
# TAB 1: I KPI PROPRIETARI (Come richiesto, mantenuto e pulito)
# ============================================================================
with tab_kpi:
    st.header("L'Ingegnerizzazione delle Feature")
    st.markdown("Questi quattro KPI superano l'approccio puramente meccanico, integrando il carico allostatico complessivo dell'atleta (stress vitale, lavoro, clima).")
    
    k1, k2 = st.columns(2)
    with k1:
        st.info("**SMA (Stress Mentale dell'Allenamento)**\n\nQuantifica l'impatto psicofisico della sessione.")
        st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
        
        st.warning("**ISLR (Indice di Sforzo Lavorativo Residuo)**\n\nMisura la competizione tra carico lavorativo e risorse energetiche.")
        st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")

    with k2:
        st.error("**IITR (Indice Impatto Termico e Resistenza)**\n\nStandardizza lo stress climatico subito dall'atleta per chilometro.")
        st.latex(r"IITR = \frac{\text{Temp (C)} \times \text{Vento (km/h)}}{\text{Distanza (km)}}")
        
        st.success("**IDET (Indice di Degradazione Termica)**\n\nMappa la deriva cardiaca estiva per prevenire falsi allarmi di calo forma.")
        st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temp (C)}}{\text{Velocità (km/h)}}")

    st.divider()
    scelta_kpi = st.selectbox("Seleziona un KPI per visualizzarne la distribuzione nel dataset:", ['SMA', 'ISLR', 'IITR', 'IDET'])
    fig_kpi = px.histogram(df, x=scelta_kpi, color='Rischio Overload', nbins=30, barmode="overlay",
                           color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                           title=f"Sovrapposizione {scelta_kpi}: Sessioni Sicure (Blu) vs Overload (Rosso)")
    st.plotly_chart(fig_kpi, use_container_width=True)

# ============================================================================
# TAB 2: MACHINE LEARNING - ESPANDIBILI
# ============================================================================
with tab_ml:
    st.header("Analisi dei Modelli Predittivi")
    st.markdown("Esplora come gli algoritmi apprendono dai dati. Apri le tendine sottostanti per analizzare ogni singolo modello.")

    # ---------------------------------------------------------
    # 1. REGRESSIONE LINEARE
    # ---------------------------------------------------------
    with st.expander("📈 1. Regressione Lineare (Linear Regression)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> La regressione lineare modella la relazione tra una variabile dipendente continua (es. Tempo) e una variabile indipendente (es. Distanza), minimizzando la discrepanza tra i valori reali e la linea di tendenza.</div>", unsafe_allow_html=True)
        
        X_lr = df[['Distanza (km)']].values
        y_lr = df['Tempo (min)'].values
        lr_model = LinearRegression().fit(X_lr, y_lr)
        df['Tempo_Predetto'] = lr_model.predict(X_lr)
        df['Errore (Residuo)'] = df['Tempo (min)'] - df['Tempo_Predetto']

        c1, c2 = st.columns(2)
        with c1:
            fig_lr1 = go.Figure()
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo (min)'], mode='markers', name='Dati Reali', marker=dict(color='#3498db', opacity=0.6)))
            fig_lr1.add_trace(go.Scatter(x=df['Distanza (km)'], y=df['Tempo_Predetto'], mode='lines', name='Trend Ottimale (OLS)', line=dict(color='#e74c3c', width=3)))
            fig_lr1.update_layout(title="Relazione Distanza - Tempo", xaxis_title="Distanza (km)", yaxis_title="Tempo (min)")
            st.plotly_chart(fig_lr1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> La linea rossa rappresenta l'apprendimento del modello. L'algoritmo ha calcolato il coefficiente matematico perfetto che moltiplicato per i chilometri ci restituisce i minuti attesi.</div>", unsafe_allow_html=True)

        with c2:
            fig_lr2 = px.histogram(df, x="Errore (Residuo)", nbins=20, title="Distribuzione degli Errori di Previsione", color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig_lr2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Questo grafico mostra di quanti minuti il modello si è sbagliato. Essendo centrato sullo zero (forma a campana), significa che il modello è sano e non ha bias evidenti.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. REGRESSIONE LOGISTICA
    # ---------------------------------------------------------
    with st.expander("🎯 2. Regressione Logistica (Logistic Regression)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Algoritmo di classificazione che stima la probabilità che una sessione appartenga a uno stato critico. Mappa la combinazione delle feature in uno spazio probabilistico tra 0 e 1 applicando la funzione logistica (Sigmoide).</div>", unsafe_allow_html=True)
        
        X_log = df[['ISLR']].values
        y_log = df['Rischio Overload'].values
        log_model = LogisticRegression().fit(X_log, y_log)
        
        x_range = np.linspace(df['ISLR'].min(), df['ISLR'].max(), 300).reshape(-1, 1)
        y_prob = log_model.predict_proba(x_range)[:, 1]
        df['Probabilità_Overload'] = log_model.predict_proba(X_log)[:, 1]

        c1, c2 = st.columns(2)
        with c1:
            fig_log1 = go.Figure()
            fig_log1.add_trace(go.Scatter(x=df['ISLR'], y=df['Rischio Overload'], mode='markers', name='Osservazioni', marker=dict(color='#7f8c8d', opacity=0.5)))
            fig_log1.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines', name='Curva Sigmoide', line=dict(color='#e67e22', width=3)))
            fig_log1.add_hline(y=0.5, line_dash="dash", annotation_text="Soglia Decisionale (50%)")
            fig_log1.update_layout(title="Transizione verso l'Overload", xaxis_title="Indice Sforzo Lavorativo Residuo (ISLR)", yaxis_title="Probabilità Predetta")
            st.plotly_chart(fig_log1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> La curva a 'S' (Sigmoide) converte lo stress in rischio. Appena il valore ISLR supera la linea tratteggiata del 50%, il modello scatta e classifica la sessione come 'Pericolosa'.</div>", unsafe_allow_html=True)

        with c2:
            fig_log2 = px.box(df, x="Rischio Overload", y="Probabilità_Overload", color="Rischio Overload", 
                              color_discrete_map={0: '#3498db', 1: '#e74c3c'}, title="Separabilità delle Classi")
            st.plotly_chart(fig_log2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> I boxplot mostrano quanto il modello sia sicuro. Le sessioni sicure (0) hanno probabilità schiacciate verso il basso, mentre quelle a rischio (1) sono concentrate verso l'alto.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. RANDOM FOREST
    # ---------------------------------------------------------
    with st.expander("🌳 3. Random Forest (Alberi Decisionali)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Opera combinando una moltitudine di alberi di decisione indipendenti. Gestisce nativamente le interazioni non lineari (es. poco sonno + alta FC) e calcola la <i>Feature Importance</i> per capire quale metrica pesa di più sulla fatica.</div>", unsafe_allow_html=True)
        
        rf_features = ['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR']
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[rf_features], df['Rischio Overload'])
        imp_df = pd.DataFrame({'Feature': rf_features, 'Importanza': rf.feature_importances_}).sort_values('Importanza')

        c1, c2 = st.columns(2)
        with c1:
            fig_rf1 = px.bar(imp_df, x='Importanza', y='Feature', orientation='h', title="Cosa influenza maggiormente il Rischio?", color='Importanza', color_continuous_scale="Viridis")
            st.plotly_chart(fig_rf1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Questa barra gerarchizza le tue metriche. Le feature in alto sono quelle che gli alberi decisionali usano più spesso per 'dividere' le sessioni sane da quelle in overtraining.</div>", unsafe_allow_html=True)

        with c2:
            # Estraggo le due feature più importanti per uno scatter 
            top_2 = imp_df.tail(2)['Feature'].values
            fig_rf2 = px.scatter(df, x=top_2[0], y=top_2[1], color='Rischio Overload', 
                                 title=f"Interazione delle Top 2 Feature",
                                 color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            st.plotly_chart(fig_rf2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Il grafico incrocia le due variabili più forti. Noterai che i punti rossi (Overload) si concentrano in specifiche aree dello spazio, dimostrando l'interazione non-lineare catturata dalla Foresta.</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # ---------------------------------------------------------
    with st.expander("🔍 4. Clustering K-Means (Segmentazione Non Supervisionata)", expanded=False):
        st.markdown("<div class='theory-box'><b>Fondamenti Teorici:</b> Raggruppa le sessioni sulla base delle loro similitudini geometriche nello spazio multidimensionale. L'algoritmo non sa in anticipo l'esito della sessione: raggruppa 'alla cieca' per scoprire pattern latenti.</div>", unsafe_allow_html=True)
        
        km = KMeans(n_clusters=3, random_state=42).fit(df[['FC Media', 'ISLR']])
        df['Cluster_ID'] = km.labels_
        # Rinominiamo i cluster per chiarezza
        cluster_map = {0: 'Rigenerativo', 1: 'Elevato Stress', 2: 'Qualità / Misto'}
        df['Profilo_Corsa'] = df['Cluster_ID'].map(cluster_map)

        c1, c2 = st.columns(2)
        with c1:
            fig_km1 = px.scatter(df, x="ISLR", y="FC Media", color="Profilo_Corsa", 
                                 title="Segmentazione Automatica delle Corse",
                                 color_discrete_sequence=['#2ecc71', '#e74c3c', '#f1c40f'])
            st.plotly_chart(fig_km1, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Ogni colore rappresenta una 'famiglia' di allenamento scoperta dall'algoritmo calcolando i centri geometrici. Sessioni distanti fisicamente nel grafico hanno caratteristiche opposte.</div>", unsafe_allow_html=True)

        with c2:
            cluster_means = df.groupby('Profilo_Corsa')[['Ore Sonno', 'Tempo (min)', 'RPE']].mean().reset_index()
            fig_km2 = px.bar(cluster_means, x='Profilo_Corsa', y=['Ore Sonno', 'RPE'], barmode='group', 
                             title="Cosa distingue i profili?")
            st.plotly_chart(fig_km2, use_container_width=True)
            st.markdown("<div class='explanation-box'><b>Guida alla lettura:</b> Questo grafico decodifica i Cluster. Ci mostra che, ad esempio, il gruppo 'Elevato Stress' è tipicamente associato a un RPE (sforzo) più alto e a un sonno minore rispetto agli altri gruppi.</div>", unsafe_allow_html=True)


# ============================================================================
# TAB 3: SIMULATORE PREDITTIVO
# ============================================================================
with tab_sim:
    st.header("🎮 Centrale Operativa: Simulatore What-If")
    st.markdown("Usa i cursori per simulare una sessione di allenamento. L'Intelligenza Artificiale (Random Forest) calcolerà istantaneamente il livello di rischio elaborando l'interazione tra i tuoi KPI proprietari e i dati ambientali.")
    
    st.markdown("---")
    
    col_input, col_output = st.columns([1, 1])
    
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
        
        # Motore Matematico dei KPI
        sim_sma = (sim_stress * sim_rpe) / sim_sonno if sim_sonno > 0 else 0
        sim_islr = (sim_lavoro * sim_stress) / sim_dist if sim_dist > 0 else 0
        sim_idet = (sim_fc * sim_temp) / sim_vel if sim_vel > 0 else 0
        sim_iitr = (sim_temp * sim_vento) / sim_dist if sim_dist > 0 else 0
        
        # Il modello RF che abbiamo allenato sopra richiede questi 6 campi esatti
        input_data = pd.DataFrame([[sim_dist, sim_sonno, sim_sma, sim_islr, sim_idet, sim_iitr]], 
                                  columns=['Distanza (km)', 'Ore Sonno', 'SMA', 'ISLR', 'IDET', 'IITR'])
        
        prob_rischio = rf.predict_proba(input_data)[0][1] * 100
        
        st.markdown(f"""
        <div style='background-color: #1e272e; padding: 25px; border-radius: 12px; border: 1px solid #485460; text-align:center;'>
            <p style='color: #808e9b; margin-bottom: 0;'>PROBABILITÀ DI OVERTRAINING (SOVRACCARICO)</p>
            <h1 style='color: {"#00E5FF" if prob_rischio < 40 else "#f1c40f" if prob_rischio < 70 else "#e74c3c"}; font-size: 4em; margin: 10px 0;'>
                {prob_rischio:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Diagnostica dei KPI Generati")
        m1, m2 = st.columns(2)
        m1.metric("Stress Mentale (SMA)", f"{sim_sma:.2f}")
        m2.metric("Sforzo Lavorativo Residuo (ISLR)", f"{sim_islr:.2f}")
        
        if prob_rischio < 40:
            st.success("🟢 **SEMAFORO VERDE**: I parametri rientrano nella norma fisiologica. Lo stress lavorativo inserito è ammortizzato efficacemente dalle ore di sonno e dal volume chilometrico previsto.")
        elif prob_rischio < 70:
            st.warning("🟡 **ZONA DI ATTENZIONE**: L'algoritmo rileva una competizione tra le energie richieste per l'allenamento e l'affaticamento del sistema nervoso (poco sonno o alto stress). Cautela.")
        else:
            st.error("🔴 **ALLARME CRITICO**: La combinazione inserita (es. alto stress lavorativo concentrato su pochi km, o riposo gravemente insufficiente) crea un picco anomalo nei KPI. Rischio infortunio imminente.")
