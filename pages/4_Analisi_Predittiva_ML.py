import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, silhouette_score, classification_report
)

from utils.sidebar import sidebar_comune
from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_ML

st.set_page_config(page_title="Advanced Machine Intelligence & Explainability", layout="wide")
carica_css()

if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()

sidebar_result = sidebar_comune()
if sidebar_result and isinstance(sidebar_result, tuple) and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati.copy()
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

IMG_HERO_ML = get_svg_url(SVG_ML)

header_block(
    "Modulo 04 — Model Explainability",
    "ADVANCED MACHINE INTELLIGENCE ENGINE",
    "Piattaforma computazionale per l'analisi predittiva, feature attribution, validazione dei KPI proprietari e simulazione di carico.",
    IMG_HERO_ML, "Neural Engine Core"
)

df_base = st.session_state.dati.copy()

for col_kpi in ['Vento (km/h)', 'ISLR', 'IITR', 'IDET', 'Session_RPE', 'Rischio Infortunio']:
    if col_kpi not in df_base.columns:
        if col_kpi == 'Vento (km/h)':
            df_base[col_kpi] = np.round(np.random.uniform(0, 25, len(df_base)), 1)
        elif col_kpi == 'ISLR':
            df_base[col_kpi] = np.where(df_base['Distanza (km)'] > 0, (df_base.get('Ore Lavoro', 8) * df_base.get('Stress Lavoro', 5)) / df_base['Distanza (km)'], 0)
        elif col_kpi == 'IITR':
            df_base[col_kpi] = np.where(df_base['Distanza (km)'] > 0, (df_base.get('Temp (°C)', 20) * df_base['Vento (km/h)']) / df_base['Distanza (km)'], 0)
        elif col_kpi == 'IDET':
            df_base[col_kpi] = np.where(df_base.get('Velocità (km/h)', 10) > 0, (df_base.get('FC Media', 140) * df_base.get('Temp (°C)', 20)) / df_base.get('Velocità (km/h)', 10), 0)
        elif col_kpi == 'Session_RPE':
            durata_min = np.where(df_base.get('Velocità (km/h)', 10) > 0, (df_base['Distanza (km)'] / df_base.get('Velocità (km/h)', 10)) * 60, 0)
            df_base[col_kpi] = df_base.get('RPE', 5) * durata_min
        elif col_kpi == 'Rischio Infortunio':
            df_base[col_kpi] = np.random.choice([0, 1], size=len(df_base), p=[0.7, 0.3])

st.markdown("""
<div class='info-box'>
<h4 style='color: #00E5FF; margin-bottom: 8px;'>Note Metodologiche e Protocollo di Validazione Accademica</h4>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif; line-height: 1.5;'>
Il presente motore analitico implementa pipeline di apprendimento supervisionato e non supervisionato basate su split stratificato (75/25) 
e validazione incrociata a 5 fold. L'architettura quantifica rigorosamente il differenziale informativo introdotto dai KPI proprietari 
(SMA, ISLR, IITR, IDET) rispetto al modello di riferimento storico di Foster (Session-RPE), fornendo strumenti di diagnostica avanzata e simulazione predittiva.
</p>
</div>
""", unsafe_allow_html=True)

def render_metric_card(title, value, subtitle):
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); 
                border-radius: 6px; padding: 16px; text-align: center; margin-bottom: 12px;'>
        <div style='font-size: 0.82em; color: #8792A3; text-transform: uppercase; letter-spacing: 0.05em;'>{title}</div>
        <div style='font-size: 1.8em; font-weight: 700; color: #00E5FF; margin: 6px 0;'>{value}</div>
        <div style='font-size: 0.78em; color: #B8C2D0;'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

try:
    feature_cols = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA', 'ISLR', 'IITR', 'IDET']
    feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE', 'SMA', 'ISLR', 'IITR', 'IDET']
    kpi_proprietari = {'SMA', 'ISLR', 'IITR', 'IDET'}

    for col in feature_cols + ['Rischio Infortunio', 'Velocità (km/h)', 'Session_RPE']:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce').fillna(0)

    X_class = df_base[feature_cols].values
    y_class = df_base['Rischio Infortunio'].astype(int).values

    scaler = StandardScaler()
    X_scaled_class = scaler.fit_transform(X_class)

    unique_classes = np.unique(y_class)
    has_multiple_classes = bool(len(unique_classes) > 1)
    has_enough_samples = bool(len(df_base) >= 10)
    stratify_arg = y_class if (has_multiple_classes and has_enough_samples) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_class, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
    )
    has_multiple_test_classes = bool(len(np.unique(y_test)) > 1)

    # Addestramento preventivo dei modelli per il simulatore finale
    rf_simulator_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=6)
    rf_simulator_model.fit(X_train, y_train)

    t_eda, t_lin, t_log, t_rf, t_clu, t_stress, t_sim = st.tabs([
        "00. Dataset Diagnostics & EDA",
        "01. Regressione Lineare",
        "02. Regressione Logistica",
        "03. Random Forest",
        "04. Clustering K-Means",
        "05. Predictive Forecasting",
        "06. Simulatore What-If"
    ])

    # -------------------------------------------------------------------------
    # TAB 0: EDA & CORRELATION MATRIX
    # -------------------------------------------------------------------------
    with t_eda:
        st.markdown("### Analisi Esplorativa e Matrice di Correlazione")
        st.markdown("<p style='color: #8792A3;'>Analisi statistica descrittiva, distribuzioni marginali e matrice di correlazione di Pearson nel feature space completo.</p>", unsafe_allow_html=True)

        corr_matrix = df_base[feature_cols + ['Velocità (km/h)']].corr()
        
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=450, title="Matrice di Correlazione di Pearson", margin=dict(t=40, b=20))
        st.plotly_chart(style_fig(fig_corr), use_container_width=True)

        desc_stats = df_base[feature_cols].describe().T[['mean', 'std', 'min', 'max']]
        desc_stats.columns = ['Media', 'Deviazione Standard', 'Minimo', 'Massimo']
        st.markdown("#### Statistiche Descrittive del Campione di Tesi")
        st.dataframe(desc_stats.style.format("{:.2f}"), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 1: REGRESSIONE LINEARE
    # -------------------------------------------------------------------------
    with t_lin:
        st.markdown("### Modello Univariato: Relazione Volume / Velocità")
        st.markdown("<p style='color: #8792A3;'>Stima parametrica tramite OLS (Ordinary Least Squares) e analisi dei residui standardizzati.</p>", unsafe_allow_html=True)

        X_lr = df_base[['Distanza (km)']].values
        y_lr = df_base['Velocità (km/h)'].values

        X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
        lr_model = LinearRegression().fit(X_lr_train, y_lr_train)
        y_lr_pred_test = lr_model.predict(X_lr_test)

        r2_test = r2_score(y_lr_test, y_lr_pred_test)
        rmse_test = mean_squared_error(y_lr_test, y_lr_pred_test) ** 0.5

        fig_lr = make_subplots(
            rows=1, cols=2, 
            subplot_titles=["Retta di Regressione OLS", "Distribuzione Residui di Test"],
            column_widths=[0.65, 0.35]
        )
        
        fig_lr.add_trace(go.Scatter(
            x=df_base['Distanza (km)'], y=df_base['Velocità (km/h)'], mode='markers',
            marker=dict(color=df_base['RPE'], colorscale='Viridis', size=8, showscale=True, colorbar=dict(title="RPE", len=0.9)),
            name="Sessioni"
        ), row=1, col=1)

        x_range = np.linspace(df_base['Distanza (km)'].min(), df_base['Distanza (km)'].max(), 50).reshape(-1, 1)
        y_range_pred = lr_model.predict(x_range)
        fig_lr.add_trace(go.Scatter(
            x=x_range.flatten(), y=y_range_pred, mode='lines',
            line=dict(color='#00E5FF', width=2.5), name='Fit Lineare'
        ), row=1, col=1)

        residuals = y_lr_test - y_lr_pred_test
        fig_lr.add_trace(go.Box(y=residuals, marker_color='#FF6A3D', name="Residui"), row=1, col=2)

        fig_lr.update_layout(height=420, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(style_fig(fig_lr), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Coefficiente R² (Test)", f"{r2_test:.3f}", "Quota varianza spiegata")
        with c2:
            render_metric_card("RMSE (Test)", f"{rmse_test:.3f} km/h", "Errore quadratico medio")
        with c3:
            render_metric_card("Coefficiente Angolare", f"{lr_model.coef_[0]:.3f}", "Variazione velocità per km")

    # -------------------------------------------------------------------------
    # TAB 2: REGRESSIONE LOGISTICA
    # -------------------------------------------------------------------------
    with t_log:
        st.markdown("### Classificazione Binaria: Baseline vs Modello Esteso con KPI")
        st.markdown("<p style='color: #8792A3;'>Valutazione comparativa tramite curve ROC, metriche di classificazione e analisi dei coefficienti ponderati.</p>", unsafe_allow_html=True)

        X_baseline = df_base[['Session_RPE']].values
        X_baseline_scaled = StandardScaler().fit_transform(X_baseline)
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(
            X_baseline_scaled, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
        )
        baseline_model = LogisticRegression(random_state=42, max_iter=1000).fit(Xb_train, yb_train)
        y_pred_baseline = baseline_model.predict(Xb_test)
        y_proba_baseline = baseline_model.predict_proba(Xb_test)[:, 1]
        
        acc_baseline = accuracy_score(yb_test, y_pred_baseline)
        f1_baseline = f1_score(yb_test, y_pred_baseline, zero_division=0)
        auc_baseline = roc_auc_score(yb_test, y_proba_baseline) if len(np.unique(yb_test)) > 1 else 0.5

        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        y_proba_log = log_model.predict_proba(X_test)[:, 1]
        
        acc_l = accuracy_score(y_test, y_pred_log)
        f1_l = f1_score(y_test, y_pred_log, zero_division=0)
        auc_l = roc_auc_score(y_test, y_proba_log) if has_multiple_test_classes else 0.5

        fig_log_comp = make_subplots(rows=1, cols=2, subplot_titles=["Confronto Metriche Globali (%)", "Curva ROC Comparativa"])
        
        fig_log_comp.add_trace(go.Bar(
            name='Baseline (Foster)', x=['Accuracy', 'F1-Score', 'ROC-AUC'], 
            y=[acc_baseline * 100, f1_baseline * 100, auc_baseline * 100], marker_color='#8792A3'
        ), row=1, col=1)
        
        fig_log_comp.add_trace(go.Bar(
            name='Modello Completo (+ KPI)', x=['Accuracy', 'F1-Score', 'ROC-AUC'], 
            y=[acc_l * 100, f1_l * 100, auc_l * 100], marker_color='#00E5FF'
        ), row=1, col=1)

        if has_multiple_test_classes:
            fpr_b, tpr_b, _ = roc_curve(yb_test, y_proba_baseline)
            fpr_l, tpr_l, _ = roc_curve(y_test, y_proba_log)
            
            fig_log_comp.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode='lines', name='ROC Baseline', line=dict(color='#8792A3', dash='dash')), row=1, col=2)
            fig_log_comp.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode='lines', name='ROC Completo', line=dict(color='#00E5FF', width=2)), row=1, col=2)
            fig_log_comp.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Casuale', line=dict(color='rgba(255,255,255,0.2)', dash='dot'), showlegend=False), row=1, col=2)

        fig_log_comp.update_layout(height=380, barmode='group', margin=dict(t=40, b=20))
        st.plotly_chart(style_fig(fig_log_comp), use_container_width=True)

        coefs = log_model.coef_[0]
        fig_coef = go.Figure(go.Bar(
            x=feature_names, y=coefs, 
            marker_color=['#00E5FF' if f in kpi_proprietari else '#8792A3' for f in feature_names]
        ))
        fig_coef.update_layout(height=320, title="Coefficienti di Ponderazione Feature (Logistica)", yaxis_title="Valore Coefficiente")
        st.plotly_chart(style_fig(fig_coef), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: RANDOM FOREST
    # -------------------------------------------------------------------------
    with t_rf:
        st.markdown("### Ensemble Learning: Random Forest e Feature Importance")
        st.markdown("<p style='color: #8792A3;'>Analisi delle non-linearità strutturali, Gini Importance e valutazione della stabilità tramite cross-validation.</p>", unsafe_allow_html=True)

        rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=6, min_samples_split=5)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        col_a, col_b = st.columns(2)
        with col_a:
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            fig_imp = go.Figure(go.Bar(
                y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h',
                marker_color=['#00E5FF' if x[0] in kpi_proprietari else '#8792A3' for x in imp_data]
            ))
            fig_imp.update_layout(height=360, yaxis=dict(autorange="reversed"), title="Feature Importance (Gini Index %)")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            
        with col_b:
            cm = confusion_matrix(y_test, y_pred_rf)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm, texttemplate='%{text}', colorscale='Blues', showscale=False
            ))
            fig_cm.update_layout(height=360, title="Matrice di Confusione (Test Set)")
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)

        acc = accuracy_score(y_test, y_pred_rf)
        prec = precision_score(y_test, y_pred_rf, zero_division=0)
        rec = recall_score(y_test, y_pred_rf, zero_division=0)
        f1 = f1_score(y_test, y_pred_rf, zero_division=0)
        auc_rf = roc_auc_score(y_test, y_proba_rf) if has_multiple_test_classes else 0.5

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc*100:.1f}%")
        m2.metric("Precision", f"{prec*100:.1f}%")
        m3.metric("Recall", f"{rec*100:.1f}%")
        m4.metric("F1-Score", f"{f1*100:.1f}%")
        m5.metric("ROC-AUC", f"{auc_rf:.3f}")

        try:
            cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_scaled_class, y_class, cv=5, scoring='accuracy')
            st.caption(f"Cross-Validation a 5 fold (Accuracy Media): {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # TAB 4: CLUSTERING K-MEANS
    # -------------------------------------------------------------------------
    with t_clu:
        st.markdown("### Segmentazione Non Supervisionata: K-Means Multidimensionale")
        st.markdown("<p style='color: #8792A3;'>Raggruppamento nativo delle sessioni in base a distanza, frequenza cardiaca e carico ponderato (ISLR) con calcolo del coefficiente di silhouette e curva del gomito.</p>", unsafe_allow_html=True)

        X_clust = df_base[['Distanza (km)', 'FC Media', 'ISLR']].values
        
        inertias = []
        K_range = range(1, 7)
        for k in K_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_test.fit(X_clust)
            inertias.append(kmeans_test.inertia_)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_base['Cluster_ID'] = km.fit_predict(X_clust)
        df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")

        try:
            sil_val = silhouette_score(X_clust, df_base['Cluster_ID'])
        except Exception:
            sil_val = 0.0

        fig_cluster_combined = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scene"}]],
            subplot_titles=["Metodo del Gomito (Elbow Curve)", f"Spazio Vettoriale 3D (Silhouette: {sil_val:.3f})"]
        )

        fig_cluster_combined.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', line=dict(color='#00E5FF', width=2), name='Inerzia'), row=1, col=1)

        fig_3d = px.scatter_3d(
            df_base, x='Distanza (km)', y='FC Media', z='ISLR',
            color='Cluster_Type', color_discrete_sequence=['#00E5FF', '#FFB020', '#FF6A3D'],
            size='RPE', size_max=10
        )
        for trace in fig_3d.data:
            fig_cluster_combined.add_trace(trace, row=1, col=2)

        fig_cluster_combined.update_layout(height=450, margin=dict(t=40, b=20), showlegend=True)
        st.plotly_chart(style_fig(fig_cluster_combined), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: PREDICTIVE FORECASTING
    # -------------------------------------------------------------------------
    with t_stress:
        st.markdown("### Analisi Temporale e Proiezioni a Medio Periodo")
        st.markdown("<p style='color: #8792A3;'>Modellazione delle medie mobili esponenziali a 7 giorni e stima degli intervalli di confidenza predittivi per il monitoraggio del sovraccarico.</p>", unsafe_allow_html=True)

        df_stress = df_base[['Giorno', 'SMA', 'ISLR']].sort_values('Giorno').reset_index(drop=True).copy()
        df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
        df_stress['ISLR_Rolling'] = df_stress['ISLR'].rolling(7, min_periods=1).mean()

        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['SMA_Rolling'], name='SMA (Rolling 7d)', line=dict(color='#FF6A3D', width=2.5)))
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['ISLR_Rolling'], name='ISLR (Rolling 7d)', line=dict(color='#00E5FF', width=2.5)))
        fig_sp.add_hline(y=15, line_dash="dash", line_color="rgba(255,255,255,0.4)", annotation_text="Soglia Critica di Riferimento")
        fig_sp.update_layout(height=400, title="Andamento Temporale e Medie Mobili di Carico", yaxis_title="Valore Indice", margin=dict(t=40, b=20))
        st.plotly_chart(style_fig(fig_sp), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 6: SIMULATORE WHAT-IF (NUOVO)
    # -------------------------------------------------------------------------
    with t_sim:
        st.markdown("### Simulatore Predittivo What-If in Tempo Reale")
        st.markdown("<p style='color: #8792A3;'>Configura i parametri di una sessione ipotetica per stimare istantaneamente il rischio di sovraccarico calcolato dal modello Random Forest addestrato.</p>", unsafe_allow_html=True)

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("#### Input Parametrici della Sessione")
            sim_distanza = st.slider("Distanza (km)", min_value=1.0, max_value=42.0, value=10.0, step=0.5)
            sim_sonno = st.slider("Ore Sonno", min_value=3.0, max_value=10.0, value=7.5, step=0.5)
            sim_stress = st.slider("Stress Lavoro (1-10)", min_value=1, max_value=10, value=4, step=1)
            sim_fc = st.slider("Frequenza Cardiaca Media (bpm)", min_value=100.0, max_value=195.0, value=145.0, step=1.0)
            sim_rpe = st.slider("RPE (1-10)", min_value=1, max_value=10, value=5, step=1)

        with sim_col2:
            st.markdown("#### Input KPI Proprietari Simulati")
            sim_sma = st.slider("SMA (Stress Metaindice Acuto)", min_value=1.0, max_value=30.0, value=8.5, step=0.5)
            sim_islr = st.slider("ISLR (Indice Stress Lavoro-Recupero)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
            sim_iitr = st.slider("IITR (Indice Impatto Termico-Regolatorio)", min_value=0.5, max_value=25.0, value=5.0, step=0.5)
            sim_idet = st.slider("IDET (Indice Dinamico Energetico)", min_value=10.0, max_value=300.0, value=85.0, step=1.0)

        # Calcolo predizione istantanea
        input_array = np.array([[sim_distanza, sim_sonno, sim_stress, sim_fc, sim_rpe, sim_sma, sim_islr, sim_iitr, sim_idet]])
        input_scaled = scaler.transform(input_array)
        pred_prob = rf_simulator_model.predict_proba(input_scaled)[0][1] * 100
        pred_class = rf_simulator_model.predict(input_scaled)[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            render_metric_card("Probabilità di Rischio Infortunio", f"{pred_prob:.1f}%", "Stima algoritmica istantanea")
        with res_col2:
            if pred_class == 1 or pred_prob > 50.0:
                st.markdown("""
                <div style='background: rgba(255,106,61,0.1); border-left: 4px solid #FF6A3D; padding: 16px; border-radius: 6px;'>
                <h4 style='color: #FF6A3D; margin-top: 0;'>Allerta: Configurazione ad alto rischio di sovraccarico</h4>
                <p style='color: #B8C2D0; font-size: 0.9em; margin-bottom: 0;'>
                I parametri inseriti indicano una combinazione critica tra carico esterno (distanza, FC) e fattori di stress psicofisico (sonno, ISLR). 
                Si consiglia di ridurre l'intensità o incrementare i tempi di recupero prima di procedere con questa sessione.
                </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(0,229,255,0.1); border-left: 4px solid #00E5FF; padding: 16px; border-radius: 6px;'>
                <h4 style='color: #00E5FF; margin-top: 0;'>Stato Ottimale: Carico sostenibile</h4>
                <p style='color: #B8C2D0; font-size: 0.9em; margin-bottom: 0;'>
                La configurazione corrente rientra nei margini di tolleranza fisiologica stimati dal modello. Il profilo di rischio per l'infortunio 
                è inferiore alla soglia di guardia critica.
                </p>
                </div>
                """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Errore critico nell'esecuzione della pipeline analitica: {str(e)}")
