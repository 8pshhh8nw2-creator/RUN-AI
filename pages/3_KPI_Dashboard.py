"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET).
Ogni KPI ha la sua sezione dedicata con spiegazione teorica, formula, grafici
dedicati e analisi visiva, culminando nell'ultima sezione di unione con il Machine Learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.sidebar import sidebar_comune
from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_KPI

from utils.kpi_ui_components import (
    verdetto_box, in_pratica, azione_consigliata,
    kpi_card_sparkline, feature_importance_chart,
)
from utils.kpi_engine import (
    calcola_kpi_giornalieri,
    calcola_risk_score_pesato,
    COL_SONNO, COL_DISTANZA
)

st.set_page_config(page_title="Centro KPI & Masterclass Intelligence", layout="wide")
carica_css()

# ==================================================================
# STILE CUSTOM MASTERCLASS
# ==================================================================
st.markdown("""
<style>
    .kpi-main-container {
        background: linear-gradient(135deg, rgba(32,40,58,0.85) 0%, rgba(15,20,30,0.98) 100%);
        border: 1px solid rgba(0,229,255,0.3);
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .explain-text {
        color: #B8C2D0;
        font-size: 0.95em;
        line-height: 1.6;
        background: rgba(255,255,255,0.03);
        padding: 14px 18px;
        border-radius: 8px;
        border-left: 3px solid #00E5FF;
        margin-top: 12px;
    }
    .theory-box {
        background: rgba(255, 176, 32, 0.05);
        border-left: 4px solid #FFB020;
        padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0;
        color: #D1D5DB; font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# INIZIALIZZAZIONE STATO
# ==================================================================
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
st.session_state.setdefault('analisi_fatta', False)
st.session_state.setdefault('risultati_analisi', {})

# ==================================================================
# SIDEBAR
# ==================================================================
sidebar_result = sidebar_comune()
if sidebar_result and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

IMG_HERO_KPI = get_svg_url(SVG_KPI)

header_block(
    "Modulo 04 — Centro KPI & Masterclass Intelligence",
    "I TUOI 4 INDICI PROPRIETARI",
    "Analisi dedicata per ciascun KPI con grafici avanzati, spiegazioni teoriche e unione finale con il Machine Learning predittivo.",
    IMG_HERO_KPI, "Proprietary KPI & AI Engine"
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("⚠️ Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA' per sbloccare i tuoi KPI di oggi.")
    st.stop()


# ==================================================================
# FUNZIONI DI SUPPORTO
# ==================================================================
def _colore_e_stato(valore, soglia_verde, soglia_gialla):
    if valore is None or pd.isna(valore):
        return "#566178", "N/D"
    if valore < soglia_verde:
        return "#00F5A0", "OTTIMALE"
    elif valore < soglia_gialla:
        return "#FFB020", "MODERATO"
    return "#FF6A3D", "ATTENZIONE"


def _delta_vs_storico(valore_oggi, serie_storica):
    if serie_storica is None or len(serie_storica) == 0 or pd.isna(valore_oggi):
        return None
    ultimo_storico = serie_storica.iloc[-1]
    if pd.isna(ultimo_storico):
        return None
    return valore_oggi - ultimo_storico


def _badge_delta(delta, positivo_e_meglio=False):
    if delta is None:
        return ""
    peggiora = (delta > 0) if not positivo_e_meglio else (delta < 0)
    freccia = "▲" if delta > 0 else "▼" if delta < 0 else "→"
    colore = "#FF6A3D" if peggiora and abs(delta) > 0.01 else "#00F5A0" if abs(delta) > 0.01 else "#566178"
    return f"<span style='color:{colore}; font-size:0.8em; margin-left:6px; font-weight:600;'>{freccia} {abs(delta):.1f} vs ultima sessione</span>"


def _calcola_percentile(valore, serie_storica):
    if serie_storica is None or len(serie_storica) < 5 or pd.isna(valore):
        return None
    serie_pulita = serie_storica.dropna()
    return (serie_pulita < valore).mean() * 100


# ==================================================================
# CALCOLO DATI DI OGGI + STORICO
# ==================================================================
r = st.session_state.risultati_analisi
df_base = st.session_state.dati.copy()

kpi_oggi = calcola_kpi_giornalieri(r)

kpi_storico = None
if len(df_base) > 0:
    try:
        kpi_storico = df_base.apply(calcola_kpi_giornalieri, axis=1, result_type="expand")
    except Exception:
        kpi_storico = None

try:
    risk_score, dettaglio_scores = calcola_risk_score_pesato(
        oggi={
            "ISLR": kpi_oggi["ISLR"],
            "IDET": kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0,
            "Ore Sonno": r.get(COL_SONNO, r.get("ore_sonno", 7.0)),
            "Volume Settimanale": r.get("volume_settimanale_km", df_base[COL_DISTANZA].tail(7).sum() if COL_DISTANZA in df_base else 25.0),
            "Passo Medio": r.get("passo_medio", 5.0),
        },
        storico=kpi_storico if kpi_storico is not None else pd.DataFrame(),
    )
except Exception:
    risk_score, dettaglio_scores = 50.0, {}

status_color = "#00F5A0" if risk_score < 25 else "#FFB020" if risk_score < 60 else "#FF6A3D"
status_text = "OTTIMALE" if risk_score < 25 else "MODERATO" if risk_score < 60 else "CRITICO"

# ==================================================================
# HEADER PRINCIPALE
# ==================================================================
st.markdown(f"""
<div class='kpi-main-container'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;'>
        <div>
            <div style='color: #8792A3; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;'>Stato di Prontezza Odierno</div>
            <div style='font-size: 2.3em; font-weight: 800; color: {status_color}; margin-top: 4px;'>
                RISCHIO {status_text} <span style='font-size: 0.7em; font-weight: 400; color: #FFFFFF;'>({risk_score:.0f}%)</span>
            </div>
        </div>
        <div style='background: rgba(255,255,255,0.04); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(0,229,255,0.2); max-width: 480px;'>
            <div style='color: #00E5FF; font-weight: 600; font-size: 0.9em; margin-bottom: 4px;'>🧠 Architettura di Tesi</div>
            <div style='color: #B8C2D0; font-size: 0.85em; line-height: 1.4;'>
                Esplora singolarmente ciascun KPI proprietario con grafici dedicati, per poi scoprire come convergono nell'algoritmo di Machine Learning.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS PRINCIPALI CON I 4 KPI E L'UNIONE FINALE CON IL ML
tab_sma, tab_islr, tab_iitr, tab_idet, tab_ml = st.tabs([
    "⭐ 1. SMA", 
    "⭐ 2. ISLR", 
    "⭐ 3. IITR", 
    "⭐ 4. IDET", 
    "🧠 5. Unione con Machine Learning"
])

# Comune asse x per i grafici temporali individuali
giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

# ==================================================================
# TAB 1 — SMA (Stress Mentale dell'Allenamento)
# ==================================================================
with tab_sma:
    st.markdown("### 🧠 SMA — Stress Mentale dell'Allenamento")
    st.markdown("Analisi approfondita del primo indicatore proprietario focalizzato sulla vulnerabilità neurale e psicofisica.")
    
    col_s1, col_s2 = st.columns([1, 1.2])
    with col_s1:
        colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
        delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
        perc_sma = _calcola_percentile(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
        
        st.markdown(f"**Valore Odierno:** <span style='color:{colore_sma}; font-size:1.8em; font-weight:bold;'>{kpi_oggi['SMA']:.2f}</span>", unsafe_allow_html=True)
        st.markdown(_badge_delta(delta_sma), unsafe_allow_html=True)
        if perc_sma is not None:
            st.caption(f"📊 Posizione nel **{perc_sma:.0f}° percentile** del tuo storico personale.")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
        in_pratica("SMA alto significa che hai affrontato la sessione con stanchezza cognitiva e sonno ridotto: il corpo lavora in svantaggio neurale[cite: 2].")

    with col_s2:
        st.markdown("#### 📈 Trend Storico Individuale (SMA)")
        if kpi_storico is not None and 'SMA' in kpi_storico.columns:
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(
                x=giorni_asse, y=kpi_storico['SMA'].tail(14),
                mode='lines+markers', line=dict(color='#00E5FF', width=3),
                marker=dict(size=8)
            ))
            fig_sma.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma), use_container_width=True)
        else:
            st.info("Storico non disponibile per il grafico.")

    st.markdown("""
    <div class='explain-text'>
    <b>Razionale Scientifico (Tesi):</b> L'indicatore SMA unisce la sfera cognitiva (stress lavorativo/giornaliero) e la percezione dello sforzo (RPE) come fattori moltiplicativi del carico interno, normalizzandoli per la quantità di sonno notturno (fattore di recupero)[cite: 2]. Questo impedisce ai modelli predittivi di trattare un allenamento come puramente meccanico.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 2 — ISLR (Indice di Sforzo Lavorativo Residuo)
# ==================================================================
with tab_islr:
    st.markdown("### 💼 ISLR — Indice di Sforzo Lavorativo Residuo")
    st.markdown("Il KPI core della tesi, progettato specificamente per l'atleta amatore (*worker-athlete*).")

    col_i1, col_i2 = st.columns([1, 1.2])
    with col_i1:
        colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
        delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
        perc_islr = _calcola_percentile(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
        
        st.markdown(f"**Valore Odierno:** <span style='color:{colore_islr}; font-size:1.8em; font-weight:bold;'>{kpi_oggi['ISLR']:.2f}</span>", unsafe_allow_html=True)
        st.markdown(_badge_delta(delta_islr), unsafe_allow_html=True)
        if perc_islr is not None:
            st.caption(f"📊 Posizione nel **{perc_islr:.0f}° percentile** del tuo storico personale.")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")
        in_pratica("ISLR misura quanta densità di stress non-atletico grava su ogni singolo chilometro percorso[cite: 2]. Sopra la soglia di 6.3, il rischio di sovraccarico impenna.")

    with col_i2:
        st.markdown("#### 📈 Trend Storico Individuale (ISLR)")
        if kpi_storico is not None and 'ISLR' in kpi_storico.columns:
            fig_islr = go.Figure()
            fig_islr.add_trace(go.Scatter(
                x=giorni_asse, y=kpi_storico['ISLR'].tail(14),
                mode='lines+markers', line=dict(color='#FF6A3D', width=3),
                marker=dict(size=8)
            ))
            fig_islr.add_hline(y=6.3, line_dash="dash", line_color="#FFB020", annotation_text="Soglia Critica (6.3)")
            fig_islr.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr), use_container_width=True)
        else:
            st.info("Storico non disponibile per il grafico.")

    if kpi_oggi["ISLR"] >= 6.3:
        azione_consigliata("⚠️ Il carico lavorativo odierno sta 'mangiando' risorse preziose: valuta una sessione più corta o di scarico.")

    st.markdown("""
    <div class='explain-text'>
    <b>Razionale Scientifico (Tesi):</b> Basato sul principio del <i>Total Life Stress</i>, l'ISLR riconosce che lo stress occupazionale compete direttamente con le risorse metaboliche e neuromuscolari necessarie alla supercompensazione sportiva[cite: 2]. Non a caso, risulta essere la variabile con il peso predittivo più alto nel Random Forest.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 3 — IITR (Indice Impatto Termico e Resistenza)
# ==================================================================
with tab_iitr:
    st.markdown("### 🌡️ IITR — Indice Impatto Termico e Resistenza")
    st.markdown("Indicatore ambientale progettato per pesare le forze resistive esogene durante la sessione estiva.")

    col_t1, col_t2 = st.columns([1, 1.2])
    with col_t1:
        colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
        delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
        perc_iitr = _calcola_percentile(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
        
        st.markdown(f"**Valore Odierno:** <span style='color:{colore_iitr}; font-size:1.8em; font-weight:bold;'>{kpi_oggi['IITR']:.2f}</span>", unsafe_allow_html=True)
        st.markdown(_badge_delta(delta_iitr), unsafe_allow_html=True)
        if perc_iitr is not None:
            st.caption(f"📊 Posizione nel **{perc_iitr:.0f}° percentile** del tuo storico personale.")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.latex(r"IITR = \frac{\text{Temperatura} \times \text{Vento}}{\text{Distanza (km)}}")
        in_pratica("IITR alto indica che il mix di calore e resistenza aerodinamica ha reso lo sforzo complessivo notevolmente più severo[cite: 2].")

    with col_t2:
        st.markdown("#### 📈 Trend Storico Individuale (IITR)")
        if kpi_storico is not None and 'IITR' in kpi_storico.columns:
            fig_iitr = go.Figure()
            fig_iitr.add_trace(go.Scatter(
                x=giorni_asse, y=kpi_storico['IITR'].tail(14),
                mode='lines+markers', line=dict(color='#FFB020', width=3),
                marker=dict(size=8)
            ))
            fig_iitr.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr), use_container_width=True)
        else:
            st.info("Storico non disponibile per il grafico.")

    st.markdown("""
    <div class='explain-text'>
    <b>Razionale Scientifico (Tesi):</b> Poiché la raccolta dati si è sviluppata nei mesi estivi, l'IITR consente di isolare la componente climatica, quantificando il costo energetico aggiuntivo imposto dalle condizioni atmosferiche avverse[cite: 2].
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 4 — IDET (Indice di Degradazione Termica)
# ==================================================================
with tab_idet:
    st.markdown("### 💧 IDET — Indice di Degradazione Termica")
    st.markdown("Indicatore chiave per mappare l'efficienza meccanica e prevenire falsi allarmi nei modelli.")

    val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0

    col_d1, col_d2 = st.columns([1, 1.2])
    with col_d1:
        colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
        delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
        perc_idet = _calcola_percentile(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
        
        st.markdown(f"**Valore Odierno:** <span style='color:{colore_idet}; font-size:1.8em; font-weight:bold;'>{val_idet:.2f}</span>", unsafe_allow_html=True)
        st.markdown(_badge_delta(delta_idet), unsafe_allow_html=True)
        if perc_idet is not None:
            st.caption(f"📊 Posizione nel **{perc_idet:.0f}° percentile** del tuo storico personale.")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temperatura}}{\text{Velocità (km/h)}}")
        in_pratica("IDET alto evidenzia la 'deriva cardiaca': il cuore batte più forte per termoregolazione, non per un calo della tua forma fisica[cite: 2].")

    with col_d2:
        st.markdown("#### 📈 Trend Storico Individuale (IDET)")
        if kpi_storico is not None and 'IDET' in kpi_storico.columns:
            fig_idet = go.Figure()
            fig_idet.add_trace(go.Scatter(
                x=giorni_asse, y=kpi_storico['IDET'].tail(14),
                mode='lines+markers', line=dict(color='#00F5A0', width=3),
                marker=dict(size=8)
            ))
            fig_idet.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet), use_container_width=True)
        else:
            st.info("Storico non disponibile per il grafico.")

    st.markdown("""
    <div class='explain-text'>
    <b>Razionale Scientifico (Tesi):</b> Basato sui principi di Galloway e Maughan (1997), l'IDET corregge i dati cardiaci in base al calore[cite: 2]. Senza questo indice, un modello di machine learning interpreterebbe erroneamente la sessione estiva come un sintomo di imminente overtraining.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 5 — UNIONE DEFINITIVA CON IL MACHINE LEARNING (CAPOLAVORO FINALE)
# ==================================================================
with tab_ml:
    st.markdown("### 🧠 Sinergia Finale: Dai KPI Proprietari al Machine Learning")
    st.markdown("""
    I quattro indicatori sviluppati non sono metriche passive, ma costituiscono le **features ingegnerizzate** d'eccellenza che alimentano gli algoritmi predittivi della tesi (Random Forest e regressioni)[cite: 2]. 
    """)

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("#### 1. Ponderazione nel Modello (Feature Importance)")
        st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Il legame predittivo:</strong> L'algoritmo apprende autonomamente che l'<b>ISLR</b> e il recupero notturno guidano il rischio di overload con oltre il 50% del peso decisionale complessivo[cite: 2].</div>", unsafe_allow_html=True)

    with c_m2:
        st.markdown("#### 2. Scomposizione Analitica del Rischio Odierno")
        if dettaglio_scores:
            nomi = list(dettaglio_scores.keys())
            valori = [dettaglio_scores[k] for k in nomi]

            fig_breakdown = go.Figure(go.Bar(
                x=valori, y=nomi, orientation="h",
                marker=dict(color=valori, colorscale=[[0, "#00F5A0"], [0.5, "#FFB020"], [1, "#FF6A3D"]])
            ))
            fig_breakdown.update_layout(
                height=350, xaxis_title="Contributo al Rischio Complessivo (0-100)",
                margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(style_fig(fig_breakdown), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Trasparenza Algoritmica:</strong> Scomponendo l'output del modello ML, azzeriamo l'effetto 'scatola nera', mostrando esattamente quale KPI sta spingendo il punteggio di rischio per la sessione di oggi[cite: 2].</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🏆 Sintesi Metodologica per la Discussione di Tesi
    L'unione strutturata tra l'ingegneria delle feature (SMA, ISLR, IITR, IDET) e il machine learning supervisionato trasforma la raccolta dati in un vero **sistema di supporto decisionale proattivo**[cite: 2]. L'atleta amatore ottiene così una panoramica scientificamente rigorosa ed estremamente intuitiva della propria prontezza atletica.
    """)
