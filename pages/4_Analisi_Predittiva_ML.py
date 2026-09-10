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
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
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
    "Modulo 04 — Come funzionano i modelli",
    "ANALISI PREDITTIVA ML",
    "Scopri come i modelli imparano dal tuo storico di allenamenti per prevedere il rischio e aiutarti a decidere.",
    IMG_HERO_ML, "Machine Learning Engine"
)

df_base = st.session_state.dati.copy()

st.markdown("""
<div class='info-box'>
<h3>Come opera il Machine Learning in RUN AI?</h3>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema analizza il tuo storico di allenamenti con algoritmi di classificazione, regressione e clustering per individuare pattern nascosti e stimare come il tuo corpo reagisce ai carichi di lavoro. In parole semplici: i modelli "imparano" dai tuoi allenamenti passati per prevedere cosa succederà con quelli futuri.</p>
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

    # ---------------------------------------------------------------
    # Modelli addestrati sui dati completi: servono per la feature
    # importance, i coefficienti interpretativi e il simulatore
    # What-If (qui è corretto usare tutti i dati disponibili, perché
    # l'obiettivo non è misurare le performance ma dare al modello
    # la massima informazione possibile per generare la spiegazione).
    # ---------------------------------------------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled_class, y_train_class)

    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_scaled_class, y_train_class)

    # ---------------------------------------------------------------
    # Le METRICHE DI PERFORMANCE (accuratezza, precisione, AUC, ecc.)
    # vengono invece calcolate con cross-validation: ogni previsione
    # arriva da una versione del modello che NON ha visto quella
    # sessione durante l'addestramento. Coerente con quanto dichiarato
    # nel Capitolo 1 della tesi (validazione su dati mai visti).
    # Il numero di fold si adatta automaticamente se le classi sono
    # sbilanciate (poche sessioni a rischio rispetto a quelle sicure).
    # ---------------------------------------------------------------
    n_pos = int(y_train_class.sum())
    n_neg = len(y_train_class) - n_pos
    if n_pos > 0 and n_neg > 0:
        n_splits = max(2, min(5, n_pos, n_neg))
        cv_class = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv_class = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_rf = cross_val_predict(rf_model, X_scaled_class, y_train_class, cv=cv_class, method='predict')
    y_proba_rf = cross_val_predict(rf_model, X_scaled_class, y_train_class, cv=cv_class, method='predict_proba')[:, 1]

    y_pred_log = cross_val_predict(log_model, X_scaled_class, y_train_class, cv=cv_class, method='predict')
    y_proba_log = cross_val_predict(log_model, X_scaled_class, y_train_class, cv=cv_class, method='predict_proba')[:, 1]

    # =========================================================
    # TOKEN DI DESIGN E COMPONENTI RIUTILIZZABILI ("Data Lab" theme)
    # Nomi di classe con prefisso mlx- per non entrare mai in conflitto
    # con le classi globali definite altrove (info-box, explain-text, ecc.)
    # =========================================================
    BG_DARK   = "#0B1017"
    BG_DARK2  = "#0E1420"
    BD        = "#202B3D"
    TXT_PRI   = "#FFFFFF"
    TXT_SEC   = "#B8C2D0"
    TXT_TER   = "#8792A3"

    C_CYAN   = "#00E5FF"
    C_CYAN2  = "#00B8D4"
    C_AMBER  = "#FFB020"
    C_ORANGE = "#FF6A3D"
    C_GREEN  = "#00F5A0"

    st.markdown(f"""
    <style>
    @keyframes mlxReveal {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .mlx-hero {{
        background: linear-gradient(135deg, {BG_DARK2} 0%, #101A2E 100%);
        border: 1px solid {BD}; border-radius: 18px; padding: 30px 34px;
        position: relative; overflow: hidden; margin: 6px 0 30px 0;
        animation: mlxReveal .6s ease-out;
        box-shadow: 0 8px 26px rgba(0,0,0,0.28);
    }}
    .mlx-hero::after {{
        content:""; position:absolute; top:-40%; right:-6%; width:340px; height:340px;
        background: radial-gradient(circle, {C_CYAN} 0%, transparent 70%); opacity:.10; pointer-events:none;
    }}
    .mlx-hero-top {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:18px; position:relative; z-index:1; margin-bottom:24px; }}
    .mlx-eyebrow {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TER}; font-weight:700; margin:0 0 8px 0; }}
    .mlx-hero-title {{ font-family:'Inter', sans-serif; font-weight:700; font-size:1.5rem; color:{TXT_PRI}; margin:0; }}
    .mlx-hero-msg {{ font-family:'Inter', sans-serif; color:{TXT_SEC}; font-size:.92rem; max-width:480px; line-height:1.6; margin:0; }}

    .mlx-stat-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; position:relative; z-index:1; }}
    .mlx-stat-card {{ background: rgba(255,255,255,0.025); border:1px solid {BD}; border-radius:12px; padding:16px 18px; }}
    .mlx-stat-card .v {{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.55rem; color: var(--stat-color, {TXT_PRI}); }}
    .mlx-stat-card .l {{ font-family:'Inter', sans-serif; font-size:.76rem; color:{TXT_TER}; margin-top:7px; line-height:1.4; min-height:44px; }}
    .mlx-stat-card .bar-track {{ height:4px; border-radius:4px; background:{BD}; margin-top:10px; overflow:hidden; }}
    .mlx-stat-card .bar-fill {{ height:100%; border-radius:4px; background: var(--stat-color, {C_CYAN}); }}

    .mlx-section-head {{ display:flex; align-items:baseline; gap:14px; margin: 30px 0 14px 0; }}
    .mlx-section-head .kicker {{ font-family:'JetBrains Mono', monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color: var(--kicker-color, {C_CYAN}); font-weight:700; white-space:nowrap; }}
    .mlx-section-head h3 {{ margin:0; font-family:'Inter',sans-serif; font-weight:700; color:{TXT_PRI}; font-size:1.12rem; }}
    .mlx-section-head .rule {{ flex:1; height:1px; background: linear-gradient(90deg, {BD} 0%, transparent 100%); align-self:center; }}

    .mlx-status-chip {{ display:inline-block; padding:3px 11px; border-radius:20px; font-family:'JetBrains Mono', monospace; font-size:.7rem; font-weight:700; letter-spacing:.03em; text-transform:uppercase; margin-right:8px; vertical-align:middle; }}

    .mlx-insight {{ border-left:3px solid var(--ic-color, {C_CYAN}); background: rgba(255,255,255,0.02); padding:12px 16px; border-radius: 0 8px 8px 0; margin-top:10px; font-family:'Inter',sans-serif; font-size:.92rem; color:{TXT_SEC}; line-height:1.6; }}
    .mlx-insight strong {{ color:{TXT_PRI}; }}
    </style>
    """, unsafe_allow_html=True)

    def mlx_section(kicker, title, color=C_CYAN):
        st.markdown(f"""
        <div class='mlx-section-head'>
            <span class='kicker' style='--kicker-color:{color};'>{kicker}</span>
            <h3>{title}</h3>
            <div class='rule'></div>
        </div>
        """, unsafe_allow_html=True)

    def mlx_insight(html_text, color=C_CYAN):
        st.markdown(f"<div class='mlx-insight' style='--ic-color:{color};'>{html_text}</div>", unsafe_allow_html=True)

    def mlx_chip(text, color):
        return f"<span class='mlx-status-chip' style='color:{color}; background:{color}22; border:1px solid {color}55;'>{text}</span>"

    # =========================================================
    # KPI PANORAMICA — colpo d'occhio prima di entrare nel dettaglio
    # =========================================================
    acc_rf = (y_pred_rf == y_train_class).mean() * 100
    prec_rf = precision_score(y_train_class, y_pred_rf, zero_division=0) * 100
    rec_rf = recall_score(y_train_class, y_pred_rf, zero_division=0) * 100
    giorni_rischio_pct = (df_base['Rischio Infortunio'].sum() / len(df_base)) * 100

    st.markdown(f"""
    <div class='mlx-hero'>
        <div class='mlx-hero-top'>
            <div>
                <p class='mlx-eyebrow'>Panoramica in tempo reale</p>
                <h2 class='mlx-hero-title'>Come si comportano i modelli sul tuo storico</h2>
            </div>
            <p class='mlx-hero-msg'>Ogni numero qui sotto arriva da una validazione incrociata: il modello viene giudicato solo su allenamenti che non ha mai visto durante l'addestramento, per una stima onesta della sua reale capacità predittiva.</p>
        </div>
        <div class='mlx-stat-grid'>
            <div class='mlx-stat-card' style='--stat-color:{C_CYAN};'>
                <div class='v'>{acc_rf:.1f}%</div>
                <div class='l'>Accuratezza Random Forest — su quanti giorni ha indovinato se c'era rischio o no.</div>
                <div class='bar-track'><div class='bar-fill' style='width:{acc_rf:.0f}%;'></div></div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_GREEN};'>
                <div class='v'>{prec_rf:.1f}%</div>
                <div class='l'>Precisione — quando segnala "rischio", quante volte ha ragione davvero.</div>
                <div class='bar-track'><div class='bar-fill' style='width:{prec_rf:.0f}%;'></div></div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_AMBER};'>
                <div class='v'>{rec_rf:.1f}%</div>
                <div class='l'>Sensibilità — su tutti i giorni davvero a rischio, quanti ne ha trovati.</div>
                <div class='bar-track'><div class='bar-fill' style='width:{rec_rf:.0f}%;'></div></div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_ORANGE};'>
                <div class='v'>{giorni_rischio_pct:.1f}%</div>
                <div class='l'>Giorni a rischio nel tuo storico — percentuale di sessioni segnate come pericolose.</div>
                <div class='bar-track'><div class='bar-fill' style='width:{giorni_rischio_pct:.0f}%;'></div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mlx_section("Esplora nel dettaglio", "Scegli un modello per vedere come ragiona", C_CYAN)

    t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6, t_ml7 = st.tabs([
        "Random Forest", "Logistic Regression", "Linear Regression",
        "Cluster K-Means", "Stress Prediction", "Simulatore What-If", "Confronto Modelli"
    ])

    # =========================================================
    # TAB 1 — RANDOM FOREST
    # =========================================================
    with t_ml1:
        mlx_section("Classificazione — Ensemble", "Random Forest Classifier (Infortunio)", C_CYAN)
        mlx_insight("Immagina 100 piccoli 'esperti' (alberi decisionali) che votano indipendentemente se un giorno è a rischio infortunio o no. La Random Forest prende la decisione finale per maggioranza di voto: per questo è uno dei modelli più affidabili e resistenti agli errori isolati.", C_CYAN)

        c1, c2 = st.columns(2)
        with c1:
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto', name="Importanza Feature"))
            fig_imp.update_traces(hovertemplate="Feature: %{y}<br>Peso: %{x:.1f}%<extra></extra>")
            fig_imp.update_layout(height=320, yaxis=dict(autorange="reversed"), title="Quali fattori pesano di più")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            top_feat = imp_data[0][0]
            mlx_insight(f"<strong>Cosa conta di più:</strong> tra tutte le metriche, <strong>{top_feat}</strong> è quella che pesa di più nella decisione del modello. Tienila d'occhio prima di aumentare i carichi.", C_AMBER)

        with c2:
            cm = confusion_matrix(y_train_class, y_pred_rf)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False, name="Matrice"))
            fig_cm.update_traces(hovertemplate="Reale: %{y}<br>Predetto: %{x}<br>Casi: %{z}<extra></extra>")
            fig_cm.update_layout(height=320, title="Quante volte ha indovinato")
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)
            mlx_insight("<strong>Come leggerla:</strong> le due caselle in diagonale (in alto a sinistra e in basso a destra) sono le previsioni corrette. Più sono 'piene' rispetto alle altre due, più il modello è affidabile.", C_CYAN)

        mlx_section("Diagnostica", "Curva ROC — Capacità Discriminante del Modello", C_CYAN2)
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
            if roc_auc >= 0.8:
                auc_label, auc_col = "Ottimo", C_GREEN
            elif roc_auc >= 0.7:
                auc_label, auc_col = "Buono", C_CYAN
            elif roc_auc >= 0.6:
                auc_label, auc_col = "Discreto", C_AMBER
            else:
                auc_label, auc_col = "Debole", C_ORANGE
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;'>
                    <h3 style='color:#FFB020; margin:0;'>Pagella del Modello</h3>
                    {mlx_chip(auc_label, auc_col)}
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Area Sotto la Curva (AUC)</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{roc_auc:.2f}</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>F1-Score</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{f1_rf:.1f}%</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Precisione</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{prec_rf:.1f}%</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Sensibilità</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{rec_rf:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            mlx_insight("<strong>AUC in parole povere:</strong> un valore vicino a 1.0 significa che il modello distingue quasi perfettamente i giorni a rischio da quelli sicuri. Un valore vicino a 0.5 equivale a tirare a indovinare. Questo valore, come tutti gli altri in questa pagina, è calcolato su dati che il modello non ha usato per imparare.", C_CYAN)

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
            fig_log.update_layout(height=350, title="Quanto pesa ogni fattore", yaxis_title="Peso Coefficiente")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Come leggerlo:</strong> le barre verdi agiscono come fattori protettivi (riducono il rischio), quelle arancioni aumentano le probabilità di sovraccarico.</div>", unsafe_allow_html=True)
        with c2:
            fpr_log, tpr_log, _ = roc_curve(y_train_class, y_proba_log)
            auc_log = auc(fpr_log, tpr_log)
            odds = np.exp(coefs)
            fig_odds = go.Figure(go.Bar(x=feature_names, y=odds, marker_color='#00B8D4', name="Odds Ratio"))
            fig_odds.update_traces(hovertemplate="Feature: %{x}<br>Odds Ratio: %{y:.2f}x<extra></extra>")
            fig_odds.add_hline(y=1, line_dash="dash", line_color="#FFB020", annotation_text="Nessun effetto")
            fig_odds.update_layout(height=350, title="Di quante volte cambia il rischio")
            st.plotly_chart(style_fig(fig_odds), use_container_width=True)
            st.markdown(f"<div class='explain-text'><strong>Come leggerlo:</strong> un valore sopra 1 significa che quella variabile moltiplica il rischio; sotto 1, lo riduce. Modello valutato su dati mai visti, con AUC = {auc_log:.2f}.</div>", unsafe_allow_html=True)

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

        # Il grafico degli scostamenti usa le previsioni "sui dati di addestramento":
        # qui va bene, perché lo scopo è individuare le sessioni anomale rispetto
        # al pattern generale, non misurare quanto il modello generalizza.
        df_base['FC_Predetta'] = lr_model.predict(X_lr)
        df_base['Residuo'] = df_base['FC Media'] - df_base['FC_Predetta']

        # R² e MAE, invece, vengono calcolati con cross-validation: indicano quanto
        # bene il modello prevede la FC su sessioni che non ha mai visto, una stima
        # onesta e coerente con la metodologia dichiarata nella tesi.
        cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv_lr = cross_val_predict(lr_model, X_lr, y_lr, cv=cv_reg)
        r2 = r2_score(y_lr, y_pred_cv_lr)
        mae = mean_absolute_error(y_lr, y_pred_cv_lr)

        c1, c2 = st.columns(2)
        with c1:
            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], labels={'FC_Predetta':'FC Predetta Modello', 'FC Media':'FC Reale'})
            fig_lr.update_traces(hovertemplate="FC Reale: %{x} bpm<br>FC Predetta: %{y:.1f} bpm<extra></extra>")
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(), x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(), line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=320, title="FC Reale vs FC Prevista")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Come leggerlo:</strong> la linea verde rappresenta la previsione perfetta. Deviazioni eccessive segnalano un affaticamento non spiegato dal passo o dal clima.</div>", unsafe_allow_html=True)
        with c2:
            fig_resid = px.scatter(df_base, x='Giorno', y='Residuo', color='Residuo', color_continuous_scale=[[0,'#00F5A0'],[0.5,'#8792A3'],[1,'#FF6A3D']], labels={'Residuo':'Scostamento (bpm)'})
            fig_resid.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            fig_resid.update_traces(hovertemplate="Data: %{x}<br>Scostamento: %{y:.1f} bpm<extra></extra>")
            fig_resid.update_layout(height=320, title="Andamento degli Scostamenti nel Tempo")
            st.plotly_chart(style_fig(fig_resid), use_container_width=True)
            st.markdown(f"<div class='explain-text'><strong>Quanto è preciso il modello:</strong> su sessioni mai viste, spiega circa il <strong>{r2*100:.0f}%</strong> della variazione della tua FC, con un errore medio di <strong>±{mae:.1f} bpm</strong>. Nel grafico qui sopra, punti sopra lo zero (arancioni) indicano giorni in cui il cuore ha lavorato più del previsto.</div>", unsafe_allow_html=True)

    # =========================================================
    # TAB 4 — CLUSTER K-MEANS
    # =========================================================
    with t_ml4:
        st.markdown("### Cluster Analysis (K-Means)")
        st.markdown("<div class='explain-text'>Il modello raggruppa da solo i tuoi allenamenti in categorie simili tra loro, senza che tu gli dica nulla in anticipo. È utile per scoprire se ti stai davvero allenando in modo 'polarizzato' (facile + duro) o se resti sempre nella stessa zona intermedia, poco efficace.</div>", unsafe_allow_html=True)

        # Distanza (km, range ~0-42) e FC Media (bpm, range ~100-180) hanno scale
        # molto diverse: senza standardizzare, il raggruppamento finirebbe per
        # dare più peso alla FC solo perché i numeri sono più grandi, non perché
        # conta davvero di più. Standardizziamo prima di raggruppare.
        X_clust_raw = df_base[['Distanza (km)', 'FC Media']]
        scaler_clust = StandardScaler()
        X_clust = scaler_clust.fit_transform(X_clust_raw)

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
            fig_elbow.update_layout(height=320, title="Metodo del Gomito — Perché 3 Gruppi?", xaxis_title="Numero di Gruppi", yaxis_title="Inerzia (compattezza dei gruppi)")
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
            st.markdown("<div class='explain-text'><strong>Silhouette Score:</strong> più è vicino a 1, più i gruppi trovati sono ben separati e coerenti tra loro (i valori tipici in dati reali sono spesso tra 0.3 e 0.6).</div>", unsafe_allow_html=True)

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
            st.markdown("<div class='explain-text'><strong>Come leggerlo:</strong> superare la soglia critica indica alto rischio di sovrallenamento cronico, non solo affaticamento passeggero.</div>", unsafe_allow_html=True)
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

        # NOTA METODOLOGICA: questa è una stima approssimata della FC attesa in
        # base allo sforzo percepito simulato, non una previsione del modello di
        # Linear Regression (che richiederebbe velocità e temperatura, non
        # presenti tra gli slider di questo simulatore). Va trattata come un
        # proxy plausibile, non come una previsione di precisione clinica.
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
        st.caption("La frequenza cardiaca usata in questa simulazione è una stima approssimata basata sullo sforzo percepito impostato, non una misura reale né una previsione del modello di regressione lineare.")

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
    # TAB 7 — CONFRONTO MODELLI
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
            <p style='color:#B8C2D0;'>Sul tuo storico attuale, valutato su dati mai visti durante l'addestramento, il modello più affidabile risulta essere <strong style='color:#fff;'>{vincitore}</strong>. La Random Forest tende a catturare meglio relazioni complesse e non lineari tra le variabili, mentre la Logistic Regression offre maggiore trasparenza su "quanto" pesa ciascun fattore. Usali insieme: uno per prevedere, l'altro per capire il "perché".</p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Errore caricamento modelli ML: {str(e)}")
