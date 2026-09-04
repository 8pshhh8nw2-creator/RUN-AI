import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score, recall_score,
    f1_score, r2_score, mean_absolute_error, silhouette_score
)

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, get_svg_url, style_fig, SVG_ML
from utils.sidebar import sidebar_comune

# 1. Configurazione pagina
st.set_page_config(page_title="Analisi Predittiva ML", layout="wide")
carica_css()

# 2. Inizializzazione stato base (se non già fatto)
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
if 'device_connected' not in st.session_state:
    st.session_state.device_connected = False
if 'diario_note' not in st.session_state:
    st.session_state.diario_note = []
if 'analisi_fatta' not in st.session_state:
    st.session_state.analisi_fatta = False
if 'risultati_analisi' not in st.session_state:
    st.session_state.risultati_analisi = {}

# 3. Chiamata sidebar comune per ottenere dati e filtri sincronizzati
sidebar_result = sidebar_comune()
if sidebar_result and isinstance(sidebar_result, tuple) and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati.copy()
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

IMG_HERO_ML = get_svg_url(SVG_ML)

# ---------------------------------------------------------
# PAGINA 4: ANALISI PREDITTIVA ML
# ---------------------------------------------------------
header_block(
    "Modulo 04 — Model Explainability",
    "ANALISI PREDITTIVA ML",
    "Esplora i modelli di Machine Learning avanzati addestrati sul tuo storico biometrico e comportamentale.",
    IMG_HERO_ML, "Machine Learning Engine"
)

df_base = st.session_state.dati.copy()

st.markdown("""
<div class='info-box'>
<h3>Come opera il Machine Learning in RUN AI?</h3>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema analizza i tuoi dati storici mediante algoritmi di classificazione, regressione e clustering non supervisionato per individuare pattern invisibili e stimare con precisione la tua risposta biologica agli stimoli. In parole semplici: i modelli "imparano" dai tuoi allenamenti passati per prevedere cosa succederà con quelli futuri.</p>
</div>
""", unsafe_allow_html=True)

try:
    # =========================================================
    # PREPARAZIONE DATI CONDIVISA
    # =========================================================
    feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']
    X_train_class = df_base[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
    y_train_class = df_base['Rischio Infortunio'].values
    scaler = StandardScaler()
    X_scaled_class = scaler.fit_transform(X_train_class)

    # Modelli addestrati una sola volta, riutilizzati in tutte le tab
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled_class, y_train_class)
    y_pred_rf = rf_model.predict(X_scaled_class)
    y_proba_rf = rf_model.predict_proba(X_scaled_class)[:, 1]

    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_scaled_class, y_train_class)
    y_proba_log = log_model.predict_proba(X_scaled_class)[:, 1]

    # =========================================================
    # KPI PANORAMICA — colpo d'occhio prima di entrare nel dettaglio
    # =========================================================
    st.subheader("KPI Panoramica Modelli")
    acc_rf = (y_pred_rf == y_train_class).mean() * 100
    prec_rf = precision_score(y_train_class, y_pred_rf, zero_division=0) * 100
    rec_rf = recall_score(y_train_class, y_pred_rf, zero_division=0) * 100
    giorni_rischio_pct = (df_base['Rischio Infortunio'].sum() / len(df_base)) * 100

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Accuratezza Random Forest", f"{acc_rf:.1f}%", help="Su quanti giorni il modello ha indovinato correttamente se ci fosse rischio o meno.")
    kc2.metric("Precisione (Precision)", f"{prec_rf:.1f}%", help="Quando il modello segnala 'rischio', quante volte ha ragione davvero.")
    kc3.metric("Sensibilità (Recall)", f"{rec_rf:.1f}%", help="Su tutti i giorni realmente a rischio, quanti ne ha individuati il modello.")
    kc4.metric("Giorni a Rischio Storici", f"{giorni_rischio_pct:.1f}%", help="Percentuale di giorni nel tuo storico classificati come a rischio infortunio.")

    st.markdown("---")

    t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6, t_ml7 = st.tabs([
        "Random Forest", "Logistic Regression", "Linear Regression",
        "Cluster K-Means", "Stress Prediction", "Simulatore What-If", "Confronto Modelli"
    ])

    # =========================================================
    # TAB 1 — RANDOM FOREST
    # =========================================================
    with t_ml1:
        st.markdown("### Random Forest Classifier (Infortunio)")
        st.markdown("<div class='explain-text'>Immagina 100 piccoli 'esperti' (alberi decisionali) che votano indipendentemente se un giorno è a rischio infortunio o no. La Random Forest prende la decisione finale per maggioranza di voto: per questo è uno dei modelli più affidabili e resistenti agli errori isolati.</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto', name="Importanza Feature"))
            fig_imp.update_traces(hovertemplate="Feature: %{y}<br>Peso: %{x:.1f}%<extra></extra>")
            fig_imp.update_layout(height=320, yaxis=dict(autorange="reversed"), title="Importanza delle Variabili")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            top_feat = imp_data[0][0]
            st.markdown(f"<div class='explain-text'><strong>Cosa conta di più:</strong> tra tutte le metriche, <strong>{top_feat}</strong> è quella che pesa di più nella decisione del modello. Tienila d'occhio prima di aumentare i carichi.</div>", unsafe_allow_html=True)

        with c2:
            cm = confusion_matrix(y_train_class, y_pred_rf)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False, name="Matrice"))
            fig_cm.update_traces(hovertemplate="Reale: %{y}<br>Predetto: %{x}<br>Casi: %{z}<extra></extra>")
            fig_cm.update_layout(height=320, title="Matrice di Confusione")
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Come leggerla:</strong> le due caselle in diagonale (in alto a sinistra e in basso a destra) sono le previsioni corrette. Più sono 'piene' rispetto alle altre due, più il modello è affidabile.</div>", unsafe_allow_html=True)

        st.markdown("#### Curva ROC — Capacità Discriminante del Modello")
        c3, c4 = st.columns(2)
        with c3:
            fpr, tpr, _ = roc_curve(y_train_class, y_proba_rf)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#00E5FF', width=3), name=f'Random Forest (AUC={roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#8792A3', dash='dash'), name='Modello Casuale'))
            fig_roc.update_traces(hovertemplate="Falsi Positivi: %{x:.2f}<br>Veri Positivi: %{y:.2f}<extra></extra>")
            fig_roc.update_layout(height=320, title="Curva ROC", xaxis_title="Tasso Falsi Positivi", yaxis_title="Tasso Veri Positivi")
            st.plotly_chart(style_fig(fig_roc), use_container_width=True)
        with c4:
            f1_rf = f1_score(y_train_class, y_pred_rf, zero_division=0) * 100
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
                <h3 style='color:#FFB020; margin-bottom:15px;'>Pagella del Modello</h3>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Area Sotto la Curva (AUC)</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{roc_auc:.2f}</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>F1-Score</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{f1_rf:.1f}%</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Precisione</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{prec_rf:.1f}%</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Sensibilità</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{rec_rf:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='explain-text'><strong>AUC in parole povere:</strong> un valore vicino a 1.0 significa che il modello distingue quasi perfettamente i giorni a rischio da quelli sicuri. Un valore vicino a 0.5 equivale a tirare a indovinare.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 2 — LOGISTIC REGRESSION
    # =========================================================
    with t_ml2:
        st.markdown("### Logistic Regression (Probabilità Lineare)")
        st.markdown("<div class='explain-text'>A differenza della Random Forest, questo modello è più 'trasparente': assegna un peso preciso e diretto a ciascuna metrica, dicendoti esattamente quanto ogni fattore aumenta o riduce il rischio.</div>", unsafe_allow_html=True)

        coefs = log_model.coef_[0]
        colors = ['#FF6A3D' if c > 0 else '#00F5A0' for c in coefs]
        c1, c2 = st.columns(2)
        with c1:
            fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors, name="Coefficiente"))
            fig_log.update_traces(hovertemplate="Feature: %{x}<br>Impatto Lineare: %{y:.2f}<extra></extra>")
            fig_log.update_layout(height=350, title="Coefficienti di Impatto (Logistic Regression)", yaxis_title="Peso Coefficiente")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Regressione Logistica:</strong> i coefficienti verdi agiscono come fattori protettivi (riducono il rischio), quelli arancioni aumentano le probabilità di sovraccarico.</div>", unsafe_allow_html=True)
        with c2:
            fpr_log, tpr_log, _ = roc_curve(y_train_class, y_proba_log)
            auc_log = auc(fpr_log, tpr_log)
            odds = np.exp(coefs)
            fig_odds = go.Figure(go.Bar(x=feature_names, y=odds, marker_color='#00B8D4', name="Odds Ratio"))
            fig_odds.update_traces(hovertemplate="Feature: %{x}<br>Odds Ratio: %{y:.2f}x<extra></extra>")
            fig_odds.add_hline(y=1, line_dash="dash", line_color="#FFB020", annotation_text="Nessun effetto")
            fig_odds.update_layout(height=350, title="Odds Ratio — Quante Volte Cambia il Rischio")
            st.plotly_chart(style_fig(fig_odds), use_container_width=True)
            st.markdown(f"<div class='explain-text'><strong>Odds Ratio spiegato:</strong> un valore sopra 1 significa che quella variabile moltiplica il rischio; sotto 1, lo riduce. Modello valutato con AUC = {auc_log:.2f}.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 3 — LINEAR REGRESSION
    # =========================================================
    with t_ml3:
        st.markdown("### Linear Regression (Previsione FC Media)")
        st.markdown("<div class='explain-text'>Questo modello impara la relazione tra velocità, temperatura e distanza per prevedere quale dovrebbe essere la tua frequenza cardiaca media in condizioni normali. Se il valore reale si discosta molto da quello previsto, potrebbe essere un segnale di stanchezza latente.</div>", unsafe_allow_html=True)

        X_lr = df_base[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']]
        y_lr = df_base['FC Media']
        lr_model = LinearRegression()
        lr_model.fit(X_lr, y_lr)
        df_base['FC_Predetta'] = lr_model.predict(X_lr)
        df_base['Residuo'] = df_base['FC Media'] - df_base['FC_Predetta']
        r2 = r2_score(y_lr, df_base['FC_Predetta'])
        mae = mean_absolute_error(y_lr, df_base['FC_Predetta'])

        c1, c2 = st.columns(2)
        with c1:
            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], labels={'FC_Predetta':'FC Predetta Modello', 'FC Media':'FC Reale'})
            fig_lr.update_traces(hovertemplate="FC Reale: %{x} bpm<br>FC Predetta: %{y:.1f} bpm<extra></extra>")
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(), x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(), line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=320, title="FC Reale vs FC Predetta")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Previsione Lineare:</strong> la linea verde rappresenta la previsione perfetta. Deviazioni eccessive segnalano un affaticamento non spiegato dal passo o dal clima.</div>", unsafe_allow_html=True)
        with c2:
            fig_resid = px.scatter(df_base, x='Giorno', y='Residuo', color='Residuo', color_continuous_scale=[[0,'#00F5A0'],[0.5,'#8792A3'],[1,'#FF6A3D']], labels={'Residuo':'Scostamento (bpm)'})
            fig_resid.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            fig_resid.update_traces(hovertemplate="Data: %{x}<br>Scostamento: %{y:.1f} bpm<extra></extra>")
            fig_resid.update_layout(height=320, title="Andamento degli Scostamenti nel Tempo")
            st.plotly_chart(style_fig(fig_resid), use_container_width=True)
            st.markdown(f"<div class='explain-text'><strong>Precisione del modello:</strong> spiega circa il <strong>{r2*100:.0f}%</strong> della variazione della tua FC, con un errore medio di <strong>±{mae:.1f} bpm</strong>. Punti sopra lo zero (arancioni) indicano giorni in cui il cuore ha lavorato più del previsto.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 4 — CLUSTER K-MEANS
    # =========================================================
    with t_ml4:
        st.markdown("### Cluster Analysis (K-Means)")
        st.markdown("<div class='explain-text'>Il modello raggruppa da solo i tuoi allenamenti in categorie simili tra loro, senza che tu gli dica nulla in anticipo. È utile per scoprire se ti stai davvero allenando in modo 'polarizzato' (facile + duro) o se resti sempre nella stessa zona intermedia, poco efficace.</div>", unsafe_allow_html=True)

        X_clust = df_base[['Distanza (km)', 'FC Media']]
        c1, c2 = st.columns(2)
        with c1:
            inertias = []
            k_range = range(2, 7)
            for k in k_range:
                km_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_clust)
                inertias.append(km_test.inertia_)
            fig_elbow = go.Figure(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', line=dict(color='#00E5FF', width=3), marker=dict(size=9)))
            fig_elbow.add_vline(x=3, line_dash="dash", line_color="#FFB020", annotation_text="Scelto: 3")
            fig_elbow.update_traces(hovertemplate="N. Cluster: %{x}<br>Inerzia: %{y:.0f}<extra></extra>")
            fig_elbow.update_layout(height=320, title="Metodo del Gomito — Perché 3 Cluster?", xaxis_title="Numero di Cluster", yaxis_title="Inerzia")
            st.plotly_chart(style_fig(fig_elbow), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Metodo del gomito:</strong> si sceglie il punto dove la curva smette di scendere ripidamente — qui succede intorno a 3, motivo per cui usiamo quel numero di gruppi.</div>", unsafe_allow_html=True)
        with c2:
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_base['Cluster_ID'] = km.fit_predict(X_clust)
            df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")
            sil = silhouette_score(X_clust, df_base['Cluster_ID'])
            fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type', color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='RPE')
            fig_km.update_traces(hovertemplate="Distanza: %{x} km<br>FC: %{y} bpm<extra></extra>")
            fig_km.update_layout(height=320, title=f"Segmentazione Allenamenti (Silhouette: {sil:.2f})")
            st.plotly_chart(style_fig(fig_km), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Silhouette Score:</strong> più è vicino a 1, più i gruppi trovati sono ben separati e coerenti tra loro.</div>", unsafe_allow_html=True)

        st.markdown("#### Identikit di Ogni Cluster")
        cluster_profile = df_base.groupby('Cluster_Type')[['Distanza (km)', 'FC Media', 'RPE', 'Ore Sonno']].mean().reset_index()
        fig_profile = go.Figure()
        for _, row in cluster_profile.iterrows():
            fig_profile.add_trace(go.Scatterpolar(
                r=[row['Distanza (km)'], row['FC Media']/2, row['RPE']*4, row['Ore Sonno']*4],
                theta=['Distanza', 'FC Media (scala)', 'RPE (scala)', 'Sonno (scala)'],
                fill='toself', name=row['Cluster_Type']
            ))
        fig_profile.update_layout(height=380, title="Profilo Medio per Cluster", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(style_fig(fig_profile), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Come leggere il radar:</strong> ogni forma colorata rappresenta il 'ritratto tipico' di un gruppo di allenamenti. Se le forme sono molto diverse tra loro, significa che alterni davvero stili di allenamento differenti — un buon segno di polarizzazione.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 5 — STRESS PREDICTION
    # =========================================================
    with t_ml5:
        st.markdown("### Stress / Overload Prediction (Time Series)")
        st.markdown("<div class='explain-text'>Questo modulo traccia l'accumulo cronico di fatica nel tempo, aiutandoti a individuare in anticipo i periodi in cui il carico complessivo sta diventando eccessivo.</div>", unsafe_allow_html=True)

        df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').copy()
        df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()

        c1, c2 = st.columns(2)
        with c1:
            fig_sp = px.area(df_stress, x='Giorno', y='SMA_Rolling', color_discrete_sequence=['#FF6A3D'], labels={'SMA_Rolling': 'Media Mobile Stress'})
            fig_sp.update_traces(hovertemplate="Data: %{x}<br>SMA Rolling: %{y:.1f}<extra></extra>")
            fig_sp.add_hline(y=15, line_dash="dash", line_color="#FFB020", annotation_text="Soglia Critica")
            fig_sp.update_layout(height=320, title="Media Mobile Stress Sistemico (7 Giorni)")
            st.plotly_chart(style_fig(fig_sp), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Analisi Serie Temporali:</strong> superare la soglia critica indica alto rischio di sovrallenamento cronico, non solo affaticamento passeggero.</div>", unsafe_allow_html=True)
        with c2:
            giorni_sopra_soglia = int((df_stress['SMA_Rolling'] > 15).sum())
            pct_sopra_soglia = (giorni_sopra_soglia / len(df_stress)) * 100
            fig_dist_stress = px.histogram(df_stress, x='SMA', nbins=15, color_discrete_sequence=['#00E5FF'], labels={'SMA': 'Valore Stress Giornaliero'})
            fig_dist_stress.add_vline(x=15, line_dash="dash", line_color="#FFB020")
            fig_dist_stress.update_traces(hovertemplate="Stress: %{x}<br>Frequenza: %{y}<extra></extra>")
            fig_dist_stress.update_layout(height=320, title="Distribuzione dello Stress Giornaliero")
            st.plotly_chart(style_fig(fig_dist_stress), use_container_width=True)
            st.markdown(f"<div class='explain-text'><strong>In numeri:</strong> hai trascorso circa <strong>{pct_sopra_soglia:.0f}%</strong> del periodo sopra la soglia critica di stress cronico ({giorni_sopra_soglia} giorni su {len(df_stress)}).</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 6 — SIMULATORE WHAT-IF
    # =========================================================
    with t_ml6:
        st.markdown("### Simulatore What-If (Random Forest Live)")
        st.markdown("<div class='explain-text'>Muovi gli slider per simulare uno scenario futuro e scoprire in tempo reale, secondo il modello, quanto sarebbe rischioso allenarsi con quei parametri.</div>", unsafe_allow_html=True)

        base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6}

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            sim_dist = st.slider("Distanza simulata (km)", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
            sim_sonno = st.slider("Ore di sonno simulate", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
        with col_sim2:
            sim_stress = st.slider("Stress simulato", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
            sim_rpe = st.slider("RPE simulato", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")

        sim_fc = 100 + sim_rpe * 10
        sim_input = np.array([[sim_dist, sim_sonno, sim_stress, sim_fc, sim_rpe]])
        sim_prob = rf_model.predict_proba(scaler.transform(sim_input))[0][1] * 100
        sim_color = "#FF6A3D" if sim_prob >= 60 else "#FFB020" if sim_prob >= 25 else "#00F5A0"

        if sim_prob >= 60:
            safe_dist = max(0, sim_dist * 0.4)
            advice_msg = f"🔴 <strong>RISCHIO ELEVATO ({sim_prob:.1f}%)</strong>: Con questi alti valori di stress e fatica, i {sim_dist} km impostati sono molto pericolosi. Il modello consiglia di <strong>ridurre drasticamente la distanza a {safe_dist:.1f} km</strong> (o riposo completo) per evitare infortuni acuti."
            adv_col = "#FF6A3D"
        elif sim_prob >= 25:
            safe_dist = max(0, sim_dist * 0.7)
            advice_msg = f"🟡 <strong>RISCHIO MODERATO ({sim_prob:.1f}%)</strong>: C'è un sovraccarico latente. Considera di <strong>scalare il volume da {sim_dist} km a circa {safe_dist:.1f} km</strong> per rientrare nella fascia di totale sicurezza."
            adv_col = "#FFB020"
        else:
            advice_msg = f"🟢 <strong>RISCHIO BASSO ({sim_prob:.1f}%)</strong>: I tuoi parametri supportano perfettamente i {sim_dist} km simulati. Nessuna restrizione raccomandata, puoi procedere al 100%."
            adv_col = "#00F5A0"

        st.markdown(f"<div class='info-box' style='border-left-color: {adv_col};'>{advice_msg}</div>", unsafe_allow_html=True)

        col_simg1, col_simg2 = st.columns(2)
        with col_simg1:
            fig_sim_gauge = go.Figure(go.Indicator(mode="gauge+number", value=sim_prob, title={'text': "Rischio Simulato", 'font': {'color': '#8792A3'}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color}, 'bgcolor': "#111827", 'borderwidth': 0}, number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}))
            fig_sim_gauge.update_layout(height=300)
            st.plotly_chart(style_fig(fig_sim_gauge), use_container_width=True)
        with col_simg2:
            sonno_range = np.linspace(4, 10, 20)
            probs_range = [rf_model.predict_proba(scaler.transform(np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100 for s in sonno_range]
            fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'}, title="Sensibilità: Rischio vs Ore di Sonno")
            fig_sens.update_traces(line_color="#00E5FF", line_width=3, name="Sensibilità", hovertemplate="Sonno: %{x:.1f}h<br>Rischio: %{y:.1f}%<extra></extra>")
            fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#FF6A3D")
            fig_sens.update_layout(height=300)
            st.plotly_chart(style_fig(fig_sens), use_container_width=True)

        st.markdown("#### Sensibilità Incrociata: Distanza")
        dist_range = np.linspace(0, 42, 20)
        probs_dist_range = [rf_model.predict_proba(scaler.transform(np.array([[d, sim_sonno, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100 for d in dist_range]
        fig_sens_dist = px.area(x=dist_range, y=probs_dist_range, labels={'x': 'Distanza (km)', 'y': 'Rischio %'})
        fig_sens_dist.update_traces(line_color="#FFB020", fillcolor="rgba(255,176,32,0.15)", hovertemplate="Distanza: %{x:.1f} km<br>Rischio: %{y:.1f}%<extra></extra>")
        fig_sens_dist.add_vline(x=sim_dist, line_dash="dash", line_color="#FF6A3D")
        fig_sens_dist.update_layout(height=280, title="Come Cambia il Rischio all'Aumentare della Distanza")
        st.plotly_chart(style_fig(fig_sens_dist), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Come usare questo grafico:</strong> mostra a quale distanza il rischio inizia a salire rapidamente, tenendo fissi gli altri tuoi parametri attuali — utile per capire il tuo 'punto di rottura' personale di oggi.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 7 — CONFRONTO MODELLI (NUOVO)
    # =========================================================
    with t_ml7:
        st.markdown("### Confronto tra Modelli")
        st.markdown("<div class='explain-text'>Non tutti i modelli sono uguali: ognuno ha punti di forza diversi. Ecco un confronto diretto per capire quale approccio si adatta meglio ai tuoi dati.</div>", unsafe_allow_html=True)

        acc_log = ((y_proba_log >= 0.5).astype(int) == y_train_class).mean() * 100
        auc_rf_final = auc(*roc_curve(y_train_class, y_proba_rf)[:2])
        auc_log_final = auc(*roc_curve(y_train_class, y_proba_log)[:2])

        comp_data = pd.DataFrame({
            'Modello': ['Random Forest', 'Logistic Regression'],
            'Accuratezza (%)': [acc_rf, acc_log],
            'AUC': [auc_rf_final, auc_log_final]
        })
        c1, c2 = st.columns(2)
        with c1:
            fig_comp1 = px.bar(comp_data, x='Modello', y='Accuratezza (%)', color='Modello', color_discrete_sequence=['#00E5FF', '#FFB020'], text='Accuratezza (%)')
            fig_comp1.update_traces(texttemplate='%{text:.1f}%', hovertemplate="Modello: %{x}<br>Accuratezza: %{y:.1f}%<extra></extra>")
            fig_comp1.update_layout(height=320, title="Accuratezza a Confronto", showlegend=False)
            st.plotly_chart(style_fig(fig_comp1), use_container_width=True)
        with c2:
            fig_comp2 = px.bar(comp_data, x='Modello', y='AUC', color='Modello', color_discrete_sequence=['#00E5FF', '#FFB020'], text='AUC')
            fig_comp2.update_traces(texttemplate='%{text:.2f}', hovertemplate="Modello: %{x}<br>AUC: %{y:.2f}<extra></extra>")
            fig_comp2.add_hline(y=0.5, line_dash="dash", line_color="#8792A3", annotation_text="Livello Casuale")
            fig_comp2.update_layout(height=320, title="Capacità Discriminante (AUC) a Confronto", showlegend=False)
            st.plotly_chart(style_fig(fig_comp2), use_container_width=True)

        vincitore = "Random Forest" if auc_rf_final >= auc_log_final else "Logistic Regression"
        st.markdown(f"""
        <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
            <h3 style='color:#FFB020; margin-bottom:15px;'>Verdetto Finale</h3>
            <p style='color:#B8C2D0;'>Sul tuo storico attuale, il modello più affidabile risulta essere <strong style='color:#fff;'>{vincitore}</strong>. La Random Forest tende a catturare meglio relazioni complesse e non lineari tra le variabili, mentre la Logistic Regression offre maggiore trasparenza su "quanto" pesa ciascun fattore. Usali insieme: uno per prevedere, l'altro per capire il "perché".</p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Errore caricamento modelli ML: {str(e)}")
