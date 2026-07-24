"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET),
tendine dettagliate per ciascun indice, grafici avanzati (Radar Chart, Breakdown)
e l'integrazione finale con i modelli di Machine Learning.
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

st.set_page_config(page_title="Centro KPI & ML Integration", layout="wide")
carica_css()

# ==================================================================
# STILE CUSTOM MASTERCLASS
# ==================================================================
st.markdown("""
<style>
    .kpi-main-container {
        background: linear-gradient(135deg, rgba(32,40,58,0.8) 0%, rgba(15,20,30,0.95) 100%);
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
    "Modulo 04 — Centro KPI & ML Integration",
    "I TUOI 4 INDICI PROPRIETARI",
    "Analisi approfondita per singola tendina, profili radar multidimensionali e sinergia diretta con i modelli di Machine Learning.",
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
            <div style='color: #00E5FF; font-weight: 600; font-size: 0.9em; margin-bottom: 4px;'>🧠 Ponderazione Basata su Random Forest</div>
            <div style='color: #B8C2D0; font-size: 0.85em; line-height: 1.4;'>
                Il risk score aggrega i KPI pesandoli direttamente sulla reale <b>Feature Importance</b> appresa dal modello predittivo[cite: 2].
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TABS PRINCIPALI
tab_kpi_tendine, tab_radar, tab_storico, tab_ml_integration = st.tabs([
    "📍 1. Tendine Dettaglio KPI", 
    "🕸️ 2. Profilo Radar Multidimensionale", 
    "📈 3. Andamento Storico & Trend", 
    "🧠 4. Unione con Machine Learning"
])

# ==================================================================
# TAB 1 — TENDINE APPROFONDITE PER OGNI KPI (NOVITÀ RICHIESTA)
# ==================================================================
with tab_kpi_tendine:
    st.markdown("### 🔍 Analisi Approfondita dei Singoli Indicatori")
    st.markdown("Espandi le sezioni sottostanti per esaminare la formulazione matematica, il razionale teorico, i valori odierni e l'interpretazione pratica di ciascun KPI proprietario[cite: 2].")
    
    giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

    # --- TENDINA 1: SMA ---
    colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
    delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
    perc_sma = _calcola_percentile(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
    
    with st.expander("⭐ SMA — Stress Mentale dell'Allenamento (Espandi per dettagli)", expanded=True):
        c_t1, c_t2 = st.columns([1, 1.2])
        with c_t1:
            st.markdown(f"**Valore Odierno:** <span style='color:{colore_sma}; font-size:1.5em; font-weight:bold;'>{kpi_oggi['SMA']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(_badge_delta(delta_sma), unsafe_allow_html=True)
            if perc_sma is not None:
                st.caption(f"📊 Posizione nel **{perc_sma:.0f}° percentile** del tuo storico.")
            st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
        with c_t2:
            st.markdown("""
            <div class='theory-box'>
            <b>Razionale Teorico (Tesi):</b> Quantifica l'impatto psicofisico integrando la stanchezza cognitiva accumulata durante il giorno e la percezione dello sforzo (RPE), rapportate al fattore di recupero notturno (ore di sonno al denominatore)[cite: 2]. Un valore elevato indica vulnerabilità neurale.
            </div>
            """, unsafe_allow_html=True)
        in_pratica("SMA alto = hai corso con poco sonno e molto stress: oggi il sistema nervoso centrale lavora in forte svantaggio[cite: 2].")

    # --- TENDINA 2: ISLR ---
    colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
    delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
    perc_islr = _calcola_percentile(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)

    with st.expander("⭐ ISLR — Indice di Sforzo Lavorativo Residuo (Espandi per dettagli)", expanded=False):
        c_t1, c_t2 = st.columns([1, 1.2])
        with c_t1:
            st.markdown(f"**Valore Odierno:** <span style='color:{colore_islr}; font-size:1.5em; font-weight:bold;'>{kpi_oggi['ISLR']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(_badge_delta(delta_islr), unsafe_allow_html=True)
            if perc_islr is not None:
                st.caption(f"📊 Posizione nel **{perc_islr:.0f}° percentile** del tuo storico.")
            st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")
        with c_t2:
            st.markdown("""
            <div class='theory-box'>
            <b>Razionale Teorico (Tesi):</b> Dedicato all'atleta amatore (worker-athlete), isola lo stress occupazionale che compete con le risorse energetiche e neuromuscolari, rapportandolo al chilometraggio[cite: 2]. È la feature con la massima importanza nel modello di machine learning.
            </div>
            """, unsafe_allow_html=True)
        in_pratica("ISLR sopra 6.3 = l'affaticamento lavorativo erode pesantemente le tue capacità di recupero atletico[cite: 2].")
        if kpi_oggi["ISLR"] >= 6.3:
            azione_consigliata("⚠️ Il carico lavorativo odierno rende rischiosa una sessione intensa: preferisci un lavoro breve o di scarico.")

    # --- TENDINA 3: IITR ---
    colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
    delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
    perc_iitr = _calcola_percentile(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)

    with st.expander("⭐ IITR — Indice Impatto Termico e Resistenza (Espandi per dettagli)", expanded=False):
        c_t1, c_t2 = st.columns([1, 1.2])
        with c_t1:
            st.markdown(f"**Valore Odierno:** <span style='color:{colore_iitr}; font-size:1.5em; font-weight:bold;'>{kpi_oggi['IITR']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(_badge_delta(delta_iitr), unsafe_allow_html=True)
            if perc_iitr is not None:
                st.caption(f"📊 Posizione nel **{perc_iitr:.0f}° percentile** del tuo storico.")
            st.latex(r"IITR = \frac{\text{Temperatura} \times \text{Vento}}{\text{Distanza (km)}}")
        with c_t2:
            st.markdown("""
            <div class='theory-box'>
            <b>Razionale Teorico (Tesi):</b> Pesa la severità ambientale combinando le forze resistive esogene (calore e resistenza aerodinamica del vento) standardizzate per chilometro[cite: 2].
            </div>
            """, unsafe_allow_html=True)
        in_pratica("IITR alto = l'ambiente estivo o ventoso aumenta esponenzialmente il costo energetico della sessione[cite: 2].")

    # --- TENDINA 4: IDET ---
    val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0
    colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
    delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
    perc_idet = _calcola_percentile(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)

    with st.expander("⭐ IDET — Indice di Degradazione Termica (Espandi per dettagli)", expanded=False):
        c_t1, c_t2 = st.columns([1, 1.2])
        with c_t1:
            st.markdown(f"**Valore Odierno:** <span style='color:{colore_idet}; font-size:1.5em; font-weight:bold;'>{val_idet:.2f}</span>", unsafe_allow_html=True)
            st.markdown(_badge_delta(delta_idet), unsafe_allow_html=True)
            if perc_idet is not None:
                st.caption(f"📊 Posizione nel **{perc_idet:.0f}° percentile** del tuo storico.")
            st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temperatura}}{\text{Velocità (km/h)}}")
        with c_t2:
            st.markdown("""
            <div class='theory-box'>
            <b>Razionale Teorico (Tesi):</b> Mappa la deriva cardiaca estiva[cite: 2]. Evita che i modelli di Machine Learning interpretino l'innalzamento dei battiti dovuto al caldo come una perdita di forma fisica dell'atleta.
            </div>
            """, unsafe_allow_html=True)
        in_pratica("IDET alto = la frequenza cardiaca è pompata dal caldo e non da un calo di condizione[cite: 2].")

# ==================================================================
# TAB 2 — RADAR CHART & PROFILO MULTIDIMENSIONALE
# ==================================================================
with tab_radar:
    st.markdown("### 🕸️ Profilo di Rischio Multidimensionale (Radar Chart)")
    st.markdown("Il grafico a ragnatela mostra geometricamente il tuo carico allostatico odierno confrontato con la **Media Storica** e la **Soglia Critica**.")

    try:
        max_sma = max(15.0, kpi_storico["SMA"].max() if kpi_storico is not None else 20.0)
        max_islr = max(10.0, kpi_storico["ISLR"].max() if kpi_storico is not None else 12.0)
        max_iitr = max(30.0, kpi_storico["IITR"].max() if kpi_storico is not None else 40.0)
        max_idet = max(200.0, kpi_storico["IDET"].max() if kpi_storico is not None else 250.0)

        oggi_norm = [
            min(100, (kpi_oggi["SMA"] / max_sma) * 100),
            min(100, (kpi_oggi["ISLR"] / max_islr) * 100),
            min(100, (kpi_oggi["IITR"] / max_iitr) * 100),
            min(100, (val_idet / max_idet) * 100)
        ]

        if kpi_storico is not None and len(kpi_storico) > 0:
            storico_norm = [
                min(100, (kpi_storico["SMA"].mean() / max_sma) * 100),
                min(100, (kpi_storico["ISLR"].mean() / max_islr) * 100),
                min(100, (kpi_storico["IITR"].mean() / max_iitr) * 100),
                min(100, (kpi_storico["IDET"].mean() / max_idet) * 100)
            ]
        else:
            storico_norm = [50, 50, 50, 50]

        soglia_critica = [70, 70, 70, 70]
        categorie = ['SMA (Stress Mentale)', 'ISLR (Lavoro Residuo)', 'IITR (Impatto Termico)', 'IDET (Degradazione)']

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=storico_norm + [storico_norm[0]], theta=categorie + [categorie[0]],
            fill='toself', name='Media Storica Personale', line=dict(color='#00E5FF', width=2), fillcolor='rgba(0, 229, 255, 0.1)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=oggi_norm + [oggi_norm[0]], theta=categorie + [categorie[0]],
            fill='toself', name='Profilo Odierno', line=dict(color='#FF6A3D', width=3), fillcolor='rgba(255, 106, 61, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=soglia_critica + [soglia_critica[0]], theta=categorie + [categorie[0]],
            mode='lines', name='Soglia Critica di Allarme', line=dict(color='#FFB020', width=1.5, dash='dash')
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)")),
            height=480, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=20, b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(style_fig(fig_radar), use_container_width=True)
    except Exception:
        st.info("Impossibile generare il radar chart con i dati attuali.")

    st.markdown("<div class='explain-text'><strong>Analisi geometrica:</strong> Eventuali sbilanciamenti verso l'esterno evidenziano quale fattore specifico stia saturando le risorse dell'atleta.</div>", unsafe_allow_html=True)

# ==================================================================
# TAB 3 — ANDAMENTO STORICO & TREND
# ==================================================================
with tab_storico:
    if kpi_storico is None or len(kpi_storico) < 3:
        st.info("ℹ️ Servono almeno 3 sessioni storiche per mostrare l'andamento temporale.")
    else:
        st.markdown("### 📈 Dinamica Longitudinale dei KPI")
        n_storico = min(30, len(kpi_storico))
        finestra = kpi_storico.tail(n_storico).reset_index(drop=True)
        asse_x = df_base['Giorno'].tail(n_storico).tolist() if 'Giorno' in df_base.columns else list(range(n_storico))

        fig = make_subplots(rows=2, cols=2, subplot_titles=("SMA", "ISLR", "IITR", "IDET"))
        posizioni = {"SMA": (1, 1), "ISLR": (1, 2), "IITR": (2, 1), "IDET": (2, 2)}
        colori_kpi = {"SMA": "#00E5FF", "ISLR": "#FF6A3D", "IITR": "#FFB020", "IDET": "#00F5A0"}

        for kpi_nome, (riga, colonna) in posizioni.items():
            if kpi_nome in finestra.columns:
                fig.add_trace(go.Scatter(x=asse_x, y=finestra[kpi_nome], mode="lines+markers", name=kpi_nome, line=dict(color=colori_kpi[kpi_nome], width=2)), row=riga, col=colonna)

        fig.update_layout(height=520, showlegend=False, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ==================================================================
# TAB 4 — UNIONE DEFINITIVA CON IL MACHINE LEARNING (RICHIESTA SPECIALE)
# ==================================================================
with tab_ml_integration:
    st.markdown("### 🧠 Sinergia tra KPI Proprietari e Machine Learning")
    st.markdown("""
    Questa sezione unisce i concetti chiave della tua tesi: i KPI non vivono isolati, ma alimentano direttamente il **Random Forest** e i modelli di classificazione per determinare la probabilità di overload[cite: 2].
    """)

    c_ml1, c_ml2 = st.columns(2)
    with c_ml1:
        st.markdown("#### 1. Feature Importance del Modello")
        st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Il legame:</strong> L'ISLR (Sforzo Lavorativo Residuo) guida la predizione con oltre il 31% del peso. Il machine learning riconosce che lo stress lavorativo è il principale fattore di rischio per l'overtraining nell'amatore[cite: 2].</div>", unsafe_allow_html=True)

    with c_ml2:
        st.markdown("#### 2. Breakdown del Rischio Odierno (Ponderato)")
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
            st.markdown("<div class='explain-text'><strong>Trasparenza Predittiva:</strong> Scomponendo l'output del modello ML, vediamo esattamente quale KPI sta spingendo il punteggio di rischio verso la soglia critica per la sessione di oggi[cite: 2].</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🏆 Conclusione Metodologica per la Discussione di Tesi
    L'integrazione tra la **Feature Engineering** (i 4 KPI proprietari) e il **Machine Learning** (Random Forest e regressioni) consente di superare l'era dei dati grezzi passivi[cite: 2]. La dashboard non si limita a registrare la fatica, ma la interpreta, offrendo all'atleta amatore un supporto decisionale proattivo di livello enterprise[cite: 2].
    """)
