import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, silhouette_score
)

from utils.sidebar import sidebar_comune
from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_ML

st.set_page_config(page_title="Analisi Predittiva ML", layout="wide")
carica_css()

# Inizializzazione sicura dello stato
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()

# Chiamata sicura alla sidebar
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
    "ANALISI PREDITTIVA ML",
    "I 5 modelli di Machine Learning descritti nella tesi, applicati al tuo storico di allenamento.",
    IMG_HERO_ML, "Machine Learning Engine"
)

df_base = st.session_state.dati.copy()

# Allineamento dataset ai KPI proprietari se mancanti
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
<h3>Come leggere questa pagina (Core della Tesi)</h3>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>
Ogni scheda qui sotto corrisponde a uno dei 5 modelli descritti nel Capitolo 2 della tesi.
Per ciascuno trovi: spiegazione in linguaggio semplice, grafici dedicati, metriche di validazione
e il confronto con una <strong>baseline</strong> (Session-RPE di Foster) per dimostrare il reale valore aggiunto
dei KPI proprietari (SMA, ISLR, IITR, IDET).
</p>
</div>
""", unsafe_allow_html=True)

# Helper visivi
def verdetto_box(valore_pct, soglie=(50, 75), testo_basso="Debole", testo_medio="Discreto", testo_alto="Solido", spiegazione=""):
    if valore_pct >= soglie[1]:
        colore, emoji, etichetta = "#00F5A0", "✅", testo_alto
    elif valore_pct >= soglie[0]:
        colore, emoji, etichetta = "#FFB020", "⚠️", testo_medio
    else:
        colore, emoji, etichetta = "#FF6A3D", "🔴", testo_basso
    st.markdown(f"""
    <div style='border-left: 4px solid {colore}; background: rgba(255,255,255,0.03);
                padding: 12px 16px; border-radius: 8px; margin: 10px 0;'>
        <span style='font-size: 1.05em; color:{colore}; font-weight:700;'>{emoji} {etichetta}</span>
        <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>{spiegazione}</p>
    </div>
    """, unsafe_allow_html=True)

def in_pratica(testo):
    st.markdown(f"""
    <div style='background: rgba(0,229,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
    <span style='color:#00E5FF; font-weight:600;'>💡 In parole semplici:</span>
    <span style='color:#B8C2D0; font-family:"Inter",sans-serif;'> {testo}</span>
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

    n_sessioni = len(df_base)
    if n_sessioni < 60:
        st.markdown(f"""
        <div style='border-left: 4px solid #FFB020; background: rgba(255,176,32,0.08);
                    padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;'>
            <span style='color:#FFB020; font-weight:700;'>⚠️ Campione di tesi ({n_sessioni} sessioni)</span>
            <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>
            Le metriche vengono validate tramite split train/test e cross-validation a 5 fold per garantire stabilità scientifica.
            </p>
        </div>
        """, unsafe_allow_html=True)

    unique_classes = np.unique(y_class)
    has_multiple_classes = bool(len(unique_classes) > 1)
    has_enough_samples = bool(len(df_base) >= 10)
    stratify_arg = y_class if (has_multiple_classes and has_enough_samples) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_class, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
    )
    has_multiple_test_classes = bool(len(np.unique(y_test)) > 1)

    t_lin, t_log, t_rf, t_clu, t_stress = st.tabs([
        "📈 Regressione Lineare", "📊 Regressione Logistica",
        "🌲 Random Forest", "🎯 Clustering K-Means", "⏳ Stress / Overload Prediction"
    ])

    # =====================================================
    # TAB 1 — REGRESSIONE LINEARE
    # =====================================================
    with t_lin:
        st.markdown("### Regressione Lineare — Volume di Allenamento vs Performance")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> stima la relazione lineare tra il volume di allenamento (km) e la velocità media.</div>", unsafe_allow_html=True)
        in_pratica("Più km fai, in genere corri più veloce nel tempo — questo modello traccia la retta di tendenza nei tuoi dati.")

        X_lr = df_base[['Distanza (km)']].values
        y_lr = df_base['Velocità (km/h)'].values

        X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
        lr_model = LinearRegression().fit(X_lr_train, y_lr_train)
        y_lr_pred_test = lr_model.predict(X_lr_test)

        r2_test = r2_score(y_lr_test, y_lr_pred_test)
        rmse_test = mean_squared_error(y_lr_test, y_lr_pred_test) ** 0.5

        fig_lr = px.scatter(df_base, x='Distanza (km)', y='Velocità (km/h)', color='RPE',
                            color_continuous_scale=[[0, '#00E5FF'], [1, '#FF6A3D']])
        x_range = np.linspace(df_base['Distanza (km)'].min(), df_base['Distanza (km)'].max(), 50).reshape(-1, 1)
        y_range_pred = lr_model.predict(x_range)
        fig_lr.add_trace(go.Scatter(x=x_range.flatten(), y=y_range_pred, mode='lines',
                                    line=dict(color='#00F5A0', width=3, dash='dash'), name='Retta di regressione'))
        fig_lr.update_layout(height=380, title="Distanza vs Velocità media, con retta di tendenza")
        st.plotly_chart(style_fig(fig_lr), use_container_width=True)

        rc1, rc2 = st.columns(2)
        rc1.metric("R² (test)", f"{r2_test:.2f}", help="Quota di variabilità spiegata")
        rc2.metric("RMSE (test)", f"{rmse_test:.2f} km/h", help="Errore medio di previsione")

        verdetto_box(
            max(r2_test, 0) * 100, soglie=(30, 60),
            testo_basso=f"Il volume da solo spiega parzialmente la velocità (errore ±{rmse_test:.1f} km/h)",
            testo_medio=f"Il volume ha una buona correlazione con la velocità",
            testo_alto=f"Il volume spiega fortemente la velocità",
            spiegazione="Modello basilare di riferimento per la tesi."
        )

    # =====================================================
    # TAB 2 — REGRESSIONE LOGISTICA
    # =====================================================
    with t_log:
        st.markdown("### Regressione Logistica — Predizione dello Stato di Overload")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> stima la probabilità di sovraccarico confrontandosi con la baseline classica (Session-RPE).</div>", unsafe_allow_html=True)
        in_pratica("Vediamo quanto i KPI proprietari migliorano la predizione rispetto al classico metodo di Foster.")

        # Baseline
        X_baseline = df_base[['Session_RPE']].values
        X_baseline_scaled = StandardScaler().fit_transform(X_baseline)
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(
            X_baseline_scaled, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
        )
        baseline_model = LogisticRegression(random_state=42, max_iter=1000).fit(Xb_train, yb_train)
        y_pred_baseline = baseline_model.predict(Xb_test)
        acc_baseline = accuracy_score(yb_test, y_pred_baseline)
        f1_baseline = f1_score(yb_test, y_pred_baseline, zero_division=0)

        # Modello Completo
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        y_proba_log = log_model.predict_proba(X_test)[:, 1]
        acc_l = accuracy_score(y_test, y_pred_log)
        f1_l = f1_score(y_test, y_pred_log, zero_division=0)
        auc_l = roc_auc_score(y_test, y_proba_log) if has_multiple_test_classes else float('nan')

        fig_confronto = go.Figure()
        fig_confronto.add_trace(go.Bar(name='Baseline (solo Session-RPE)', x=['Accuracy', 'F1-Score'], y=[acc_baseline * 100, f1_baseline * 100], marker_color='#8792A3'))
        fig_confronto.add_trace(go.Bar(name='Modello Completo (+ KPI)', x=['Accuracy', 'F1-Score'], y=[acc_l * 100, f1_l * 100], marker_color='#00E5FF'))
        fig_confronto.update_layout(height=350, barmode='group', yaxis_title="%", title="Confronto Baseline vs Modello con KPI Proprietari")
        st.plotly_chart(style_fig(fig_confronto), use_container_width=True)

        delta_acc = (acc_l - acc_baseline) * 100
        in_pratica(f"Il modello con i KPI proprietari {'migliora' if delta_acc > 0 else 'varia'} l'accuracy di {abs(delta_acc):.1f} punti percentuali rispetto alla sola Session-RPE.")

        st.markdown("#### Coefficienti del modello completo")
        coefs = log_model.coef_[0]
        colors = ['#FF6A3D' if fname in kpi_proprietari and c > 0 else '#00F5A0' if fname in kpi_proprietari else ('#FFB020' if c > 0 else '#8792A3') for fname, c in zip(feature_names, coefs)]
        fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors, text=[f'{c:+.2f}' for c in coefs], textposition='auto'))
        fig_log.update_layout(height=380, title="Peso di ciascuna variabile (evidenziati i KPI proprietari)", yaxis_title="Coefficiente")
        fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
        st.plotly_chart(style_fig(fig_log), use_container_width=True)

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Accuracy", f"{acc_l*100:.1f}%")
        lc2.metric("F1-Score", f"{f1_l*100:.1f}%")
        lc3.metric("ROC-AUC", f"{auc_l:.2f}" if not np.isnan(auc_l) else "N/D")

    # =====================================================
    # TAB 3 — RANDOM FOREST
    # =====================================================
    with t_rf:
        st.markdown("### Random Forest — Classificazione del Rischio e Feature Importance")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> unisce alberi decisionali per catturare relazioni non lineari e individuare i fattori critici.</div>", unsafe_allow_html=True)
        in_pratica("Il modello ideale per pesare scientificamente l'importanza di ciascun indice nel determinare il rischio.")

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, min_samples_split=5)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        c1, c2 = st.columns(2)
        with c1:
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            colors_imp = ['#FF6A3D' if name in kpi_proprietari else '#00E5FF' for name, _ in imp_data]
            fig_imp = go.Figure(go.Bar(
                y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h',
                marker_color=colors_imp, text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'
            ))
            fig_imp.update_layout(height=380, yaxis=dict(autorange="reversed"), title="Importanza delle variabili (Arancione = KPI)", xaxis_title="%")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
        with c2:
            cm = confusion_matrix(y_test, y_pred_rf)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"},
                colorscale=[[0, '#0E1420'], [1, '#00E5FF']], showscale=False
            ))
            fig_cm.update_layout(height=380, title="Matrice di Confusione (Test Set)")
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)

        acc = accuracy_score(y_test, y_pred_rf)
        prec = precision_score(y_test, y_pred_rf, zero_division=0)
        rec = recall_score(y_test, y_pred_rf, zero_division=0)
        f1 = f1_score(y_test, y_pred_rf, zero_division=0)
        roc_auc_rf = roc_auc_score(y_test, y_proba_rf) if has_multiple_test_classes else float('nan')

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy", f"{acc*100:.1f}%")
        mc2.metric("Precision", f"{prec*100:.1f}%")
        mc3.metric("Recall", f"{rec*100:.1f}%")
        mc4.metric("F1-Score", f"{f1*100:.1f}%")
        mc5.metric("ROC-AUC", f"{roc_auc_rf:.2f}" if not np.isnan(roc_auc_rf) else "N/D")

        try:
            cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6), X_scaled_class, y_class, cv=5, scoring='accuracy')
            st.caption(f"Cross-validation a 5 fold: accuracy media {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
        except Exception:
            pass

    # =====================================================
    # TAB 4 — CLUSTERING K-MEANS
    # =====================================================
    with t_clu:
        st.markdown("### Clustering K-Means — Segmentazione dei Profili di Allenamento")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> raggruppa in autonomia le sessioni in base a distanza, frequenza cardiaca e sforzo.</div>", unsafe_allow_html=True)
        in_pratica("Permette di scoprire se esite una clusterizzazione naturale tra i diversi tipi di carichi di lavoro.")

        X_clust = df_base[['Distanza (km)', 'FC Media', 'ISLR']].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_base['Cluster_ID'] = km.fit_predict(X_clust)
        df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")

        try:
            sil = silhouette_score(X_clust, df_base['Cluster_ID'])
        except Exception:
            sil = float('nan')

        fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type',
                            color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='ISLR')
        fig_km.update_layout(height=420, title="Sessioni raggruppate in 3 Cluster (dimensione bolla = ISLR)")
        st.plotly_chart(style_fig(fig_km), use_container_width=True)

        if not np.isnan(sil):
            st.caption(f"Silhouette Score del clustering: **{sil:.2f}**")

    # =====================================================
    # TAB 5 — STRESS / OVERLOAD PREDICTION (Time Series)
    # =====================================================
    with t_stress:
        st.markdown("### Stress / Overload Prediction — Analisi Temporale e Proiezioni")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> analizza la media mobile a 7 giorni per anticipare i trend di sovraccarico prima che diventino critici.</div>", unsafe_allow_html=True)
        in_pratica("Monitora la continuità dello stress neurale e fisico nel medio periodo.")

        df_stress = df_base[['Giorno', 'SMA', 'ISLR']].sort_values('Giorno').reset_index(drop=True).copy()
        df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
        df_stress['ISLR_Rolling'] = df_stress['ISLR'].rolling(7, min_periods=1).mean()

        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['SMA_Rolling'], name='SMA (media 7gg)', line=dict(color='#FF6A3D', width=2.5), fill='tozeroy', fillcolor='rgba(255,106,61,0.12)'))
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['ISLR_Rolling'], name='ISLR (media 7gg)', line=dict(color='#FFB020', width=2.5)))
        fig_sp.add_hline(y=15, line_dash="dash", line_color="#8792A3", annotation_text="Soglia critica SMA")
        fig_sp.update_layout(height=400, title="Media Mobile a 7 giorni di SMA e ISLR", xaxis_title="Giorno", yaxis_title="Livello")
        st.plotly_chart(style_fig(fig_sp), use_container_width=True)

        # Forecast semplice
        n_forecast = 7
        x_idx = np.arange(len(df_stress))
        valid_mask = df_stress['SMA_Rolling'].notna().to_numpy()

        if int(valid_mask.sum()) >= 3:
            coeffs = np.polyfit(x_idx[valid_mask], df_stress['SMA_Rolling'][valid_mask], deg=1)
            trend_fn = np.poly1d(coeffs)
            future_idx = np.arange(len(df_stress), len(df_stress) + n_forecast)
            future_vals = trend_fn(future_idx)
            residual_std = np.std(df_stress['SMA_Rolling'][valid_mask] - trend_fn(x_idx[valid_mask]))

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=list(range(len(df_stress))), y=df_stress['SMA_Rolling'], mode='lines', line=dict(color='#00E5FF'), name="Storico"))
            fig_forecast.add_trace(go.Scatter(x=list(future_idx), y=future_vals, mode='lines', line=dict(color='#FF6A3D', dash='dash'), name="Proiezione"))
            fig_forecast.add_trace(go.Scatter(
                x=list(future_idx) + list(future_idx)[::-1],
                y=list(future_vals + residual_std) + list(future_vals - residual_std)[::-1],
                fill='toself', fillcolor='rgba(255,106,61,0.15)', line=dict(color='rgba(0,0,0,0)'), name="Incertezza"
            ))
            fig_forecast.update_layout(height=380, title=f"Proiezione SMA — Prossimi {n_forecast} giorni")
            st.plotly_chart(style_fig(fig_forecast), use_container_width=True)

except Exception as e:
    st.error(f"Errore nell'esecuzione dei modelli ML: {str(e)}")
