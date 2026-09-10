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
    # TAB 7 — CONFRONTO MODELLI (versione avanzata)
    # =========================================================
    with t_ml7:
        mlx_section("Testa a testa", "Random Forest vs Logistic Regression", C_AMBER)
        mlx_insight(
            "Due modelli, due filosofie diverse. La <strong>Random Forest</strong> è come una giuria di 100 esperti che votano: "
            "cattura relazioni complesse ma è più difficile da spiegare in due parole. La <strong>Logistic Regression</strong> è "
            "come una formula matematica trasparente: ogni fattore ha un peso preciso e dichiarato. Qui li mettiamo a confronto "
            "su ogni metrica, tutte calcolate su dati che i modelli non hanno mai visto durante l'addestramento.",
            C_AMBER
        )

        # ---------------------------------------------------
        # Calcolo di TUTTE le metriche per entrambi i modelli
        # ---------------------------------------------------
        y_pred_log_bin = (y_proba_log >= 0.5).astype(int)

        acc_log = (y_pred_log_bin == y_train_class).mean() * 100
        prec_log = precision_score(y_train_class, y_pred_log_bin, zero_division=0) * 100
        rec_log = recall_score(y_train_class, y_pred_log_bin, zero_division=0) * 100
        f1_log = f1_score(y_train_class, y_pred_log_bin, zero_division=0) * 100

        f1_rf_final = f1_score(y_train_class, y_pred_rf, zero_division=0) * 100
        auc_rf_final = auc(*roc_curve(y_train_class, y_proba_rf)[:2])
        auc_log_final = auc(*roc_curve(y_train_class, y_proba_log)[:2])

        metriche_labels = ["Accuratezza", "Precisione", "Sensibilità", "F1-Score", "AUC ×100"]
        rf_values = [acc_rf, prec_rf, rec_rf, f1_rf_final, auc_rf_final * 100]
        log_values = [acc_log, prec_log, rec_log, f1_log, auc_log_final * 100]

        # Punteggio complessivo: media delle 5 metriche, usato per il verdetto finale
        score_rf = float(np.mean(rf_values))
        score_log = float(np.mean(log_values))
        vincitore = "Random Forest" if score_rf >= score_log else "Logistic Regression"
        margine = abs(score_rf - score_log)

        # ---------------------------------------------------
        # HERO: badge vincitore + punteggio complessivo
        # (tutte le stringhe calcolate PRIMA, mai letterali dentro
        # le graffe delle f-string, per evitare conflitti di virgolette)
        # ---------------------------------------------------
        col_win = C_CYAN if vincitore == "Random Forest" else C_AMBER
        punteggio_vincente = score_rf if vincitore == "Random Forest" else score_log
        punteggio_altro = score_log if vincitore == "Random Forest" else score_rf
        nota_scarto = "netto" if margine > 8 else "risicato, i due modelli si equivalgono quasi"

        st.markdown(f"""
        <div class='mlx-hero' style='padding:26px 30px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:18px; position:relative; z-index:1;'>
                <div>
                    <p class='mlx-eyebrow'>Verdetto complessivo</p>
                    <h2 class='mlx-hero-title' style='margin-bottom:6px;'>🏆 {vincitore} vince sul tuo storico</h2>
                    <p class='mlx-hero-msg'>Punteggio medio su 5 metriche: <strong style='color:#fff;'>{punteggio_vincente:.1f}/100</strong>
                    contro <strong style='color:#fff;'>{punteggio_altro:.1f}/100</strong> dell'altro modello
                    (scarto di {margine:.1f} punti — {nota_scarto}).</p>
                </div>
                <div style='text-align:center; background: rgba(255,255,255,0.03); border:1px solid {BD}; border-radius:14px; padding:14px 26px;'>
                    <div style='font-family:"JetBrains Mono",monospace; font-size:2.1rem; font-weight:700; color:{col_win};'>{max(score_rf, score_log):.0f}<span style='font-size:1.1rem; color:{TXT_TER};'>/100</span></div>
                    <div style='font-family:"Inter",sans-serif; font-size:.72rem; color:{TXT_TER}; text-transform:uppercase; letter-spacing:.08em;'>Punteggio {vincitore}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # RADAR: le 5 metriche fianco a fianco
        # ---------------------------------------------------
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=rf_values + [rf_values[0]], theta=metriche_labels + [metriche_labels[0]],
                fill='toself', name='Random Forest', line=dict(color=C_CYAN, width=2),
                fillcolor='rgba(0,229,255,0.15)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=log_values + [log_values[0]], theta=metriche_labels + [metriche_labels[0]],
                fill='toself', name='Logistic Regression', line=dict(color=C_AMBER, width=2),
                fillcolor='rgba(255,176,32,0.15)'
            ))
            fig_radar.update_traces(hovertemplate="%{theta}: %{r:.1f}<extra></extra>")
            fig_radar.update_layout(
                height=380, title="Il profilo completo dei due modelli",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100]))
            )
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
            mlx_insight(
                "<strong>Come leggere la ragnatela:</strong> più l'area colorata è estesa, più quel modello è forte "
                "su tutti i fronti insieme. Se una forma è più larga in un punto ma più stretta in un altro, significa "
                "che i modelli hanno punti di forza diversi, non che uno è semplicemente 'migliore'.",
                C_CYAN
            )

        with c2:
            comp_data_full = pd.DataFrame({
                'Metrica': metriche_labels * 2,
                'Valore': rf_values + log_values,
                'Modello': ['Random Forest'] * 5 + ['Logistic Regression'] * 5
            })
            fig_comp_full = px.bar(
                comp_data_full, x='Metrica', y='Valore', color='Modello', barmode='group',
                color_discrete_map={'Random Forest': C_CYAN, 'Logistic Regression': C_AMBER}
            )
            fig_comp_full.update_traces(hovertemplate="%{x}: %{y:.1f}<extra></extra>")
            fig_comp_full.update_layout(height=380, title="Ogni metrica, fianco a fianco", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(style_fig(fig_comp_full), use_container_width=True)
            mlx_insight(
                "<strong>Qual è la metrica più importante per te?</strong> Se preferisci non farti sorprendere da un "
                "infortunio (meglio un falso allarme in più), guarda la <strong>Sensibilità</strong>. Se invece vuoi "
                "fidarti degli allarmi senza esagerare, guarda la <strong>Precisione</strong>.",
                C_AMBER
            )

        # ---------------------------------------------------
        # TABELLA PRO / CONTRO — linguaggio semplice
        # ---------------------------------------------------
        mlx_section("Guida alla scelta", "Pro, contro e quando usarli", C_GREEN)

        chip_rf = mlx_chip("Precisione", C_CYAN) if prec_rf >= prec_log else ""
        chip_log = mlx_chip("Trasparenza", C_AMBER)

        col_rf, col_log = st.columns(2)
        with col_rf:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; background: linear-gradient(135deg, #0E1420 0%, #0F1C24 100%); border:1px solid {C_CYAN}44;'>
                <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
                    <h3 style='color:{C_CYAN}; margin:0;'>🌲 Random Forest</h3>
                    {chip_rf}
                </div>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Punti di forza:</strong> cattura pattern complessi e non lineari (es. il rischio esplode solo se poco sonno e alto stress si combinano insieme). Robusta agli outlier.</p>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Limiti:</strong> è una scatola nera — più difficile spiegare esattamente perché ha dato un certo responso in un singolo caso.</p>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Usala quando:</strong> vuoi la previsione più accurata possibile e ti fidi del modello come consulente esperto.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_log:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; background: linear-gradient(135deg, #0E1420 0%, #241a0f 100%); border:1px solid {C_AMBER}44;'>
                <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
                    <h3 style='color:{C_AMBER}; margin:0;'>📐 Logistic Regression</h3>
                    {chip_log}
                </div>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Punti di forza:</strong> ogni fattore ha un peso dichiarato e leggibile (vedi tab Logistic Regression). Facile da spiegare a chiunque.</p>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Limiti:</strong> assume relazioni lineari — se il rischio dipende da combinazioni complesse di fattori, può perdersele.</p>
                <p style='color:{TXT_SEC}; font-size:.88rem;'><strong style='color:#fff;'>Usala quando:</strong> vuoi capire il perché dietro un consiglio, non solo il risultato finale.</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # VERDETTO FINALE ESTESO
        # ---------------------------------------------------
        differenza_auc = abs(auc_rf_final - auc_log_final)
        nota_auc = (
            "una differenza minima: sul piano puramente predittivo i due modelli si equivalgono quasi del tutto"
            if differenza_auc < 0.03 else
            f"una differenza di {differenza_auc:.2f} punti di AUC, non trascurabile sul tuo storico"
        )
        st.markdown(f"""
        <div class='mlx-insight' style='--ic-color:{C_GREEN}; margin-top:22px;'>
            <strong>Verdetto pratico:</strong> sul tuo storico attuale, valutato solo su allenamenti mai usati per l'addestramento,
            il modello più accurato è <strong style='color:#fff;'>{vincitore}</strong> ({nota_auc}).
            Il consiglio migliore però non è "usarne solo uno": lascia che la <strong>Random Forest</strong> ti dia l'allarme più
            affidabile, e usa la <strong>Logistic Regression</strong> per capire subito quale fattore specifico (sonno, stress,
            distanza...) sta spingendo il rischio verso l'alto — insieme coprono sia la previsione che la spiegazione.
        </div>
        """, unsafe_allow_html=True)

     
