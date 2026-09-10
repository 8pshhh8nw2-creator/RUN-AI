"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET).
Design High-Tech rigoroso, privo di emoji, con layout verticale esteso,
cruscotto indicatore grafico a stanghetta/gauge ingrandito per il rischio e griglia 2x2 sottostante.

NOTA: testi rivisti per essere comprensibili anche a chi non ha background
tecnico/scientifico. Il rigore accademico resta nei pannelli "theory-panel"
(Razionale Scientifico), che restano riservati a chi vuole approfondire.
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
# STILE CUSTOM HIGH-TECH (DARK ENTERPRISE, NESSUNA EMOJI)
# ==================================================================
st.markdown("""
<style>
    .kpi-main-container {
        background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.98) 100%);
        border: 1px solid rgba(0,229,255,0.25);
        padding: 24px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .tech-box {
        background: rgba(0,229,255,0.02);
        border-left: 3px solid #00E5FF;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-top: 10px;
        color: #B8C2D0;
        font-size: 0.9em;
        line-height: 1.5;
    }
    .theory-panel {
        background: rgba(255,176,32,0.02);
        border-left: 3px solid #FFB020;
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin: 15px 0;
        color: #D1D5DB;
        font-size: 0.95em;
        line-height: 1.6;
    }
    .metric-card-horizontal {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
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
    "Modulo 04 — Centro KPI",
    "I TUOI 4 INDICATORI DI RISCHIO",
    "Qui trovi i tuoi 4 indicatori personali, uno per scheda, con i grafici che ne spiegano l'andamento nel tempo.",
    IMG_HERO_KPI, "Centro KPI"
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("Prima di vedere i tuoi KPI, completa il questionario nella pagina 'ANALISI STATO DI FORMA'.")
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
    return "#FF6A3D", "CRITICO"


def _delta_vs_storico(valore_oggi, serie):
    """Calcola la differenza tra il valore di oggi e l'ultimo storico."""
    if serie is None or len(serie) == 0 or pd.isna(valore_oggi):
        return None
    try:
        ultimo_storico = serie.iloc[-1]
        if pd.isna(ultimo_storico):
            return None
        return float(valore_oggi - ultimo_storico)
    except Exception:
        return None


def _badge_delta(delta, positivo_e_meglio=False):
    if delta is None:
        return ""
    peggiora = (delta > 0) if not positivo_e_meglio else (delta < 0)
    freccia = "▲" if delta > 0 else "▼" if delta < 0 else "→"
    colore = "#FF6A3D" if peggiora and abs(delta) > 0.01 else "#00F5A0" if abs(delta) > 0.01 else "#566178"
    return f"<span style='color:{colore}; font-size:0.9em; font-weight:600;'>{freccia} {abs(delta):.1f} vs ultima sessione</span>"


def _calcola_percentile(valore, serie):
    """Calcola in che percentile si posiziona il dato odierno."""
    if serie is None or len(serie) < 5 or pd.isna(valore):
        return None
    try:
        serie_pulita = serie.dropna()
        if len(serie_pulita) == 0:
            return None
        return float((serie_pulita < valore).mean() * 100)
    except Exception:
        return None


def _testo_percentile(perc):
    """Trasforma il percentile in una frase semplice da leggere."""
    if perc is None:
        return "N/D — servono più sessioni per un confronto"
    return f"Più alto del {perc:.0f}% delle tue sessioni passate"


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
# HEADER PRINCIPALE CON CRUSCOTTO GAUGE INGRANDITO
# ==================================================================
col_head_testo, col_head_gauge = st.columns([1.2, 1.2], gap="large")

with col_head_testo:
    st.markdown(f"""
    <div class='kpi-main-container'>
        <div style='color: #8792A3; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1.5px;'>Come stai oggi</div>
        <div style='font-size: 2.2em; font-weight: 800; color: {status_color}; margin-top: 4px;'>
            RISCHIO {status_text} <span style='font-size: 0.65em; font-weight: 400; color: #FFFFFF;'>({risk_score:.0f}%)</span>
        </div>
        <div style='color: #B8C2D0; font-size: 0.85em; line-height: 1.4; margin-top: 10px;'>
            Questo punteggio combina i tuoi 4 indicatori, dando più peso a quelli che si sono dimostrati più importanti nel prevedere il rischio di sovraccarico.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_head_gauge:
    # Cruscotto gauge ingrandito (altezza portata a 200px e font ridimensionati)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "%", 'font': {'color': "#FFFFFF", 'size': 36}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#8792A3", 'tickfont': {'size': 14}},
            'bar': {'color': status_color, 'thickness': 0.65},
            'bgcolor': "rgba(255,255,255,0.02)",
            'borderwidth': 1,
            'bordercolor': "rgba(0,229,255,0.25)",
            'steps': [
                {'range': [0, 25], 'color': "rgba(0,245,160,0.1)"},
                {'range': [25, 60], 'color': "rgba(255,176,32,0.1)"},
                {'range': [60, 100], 'color': "rgba(255,106,61,0.1)"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 4},
                'thickness': 0.8,
                'value': risk_score
            }
        }
    ))
    fig_gauge.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(style_fig(fig_gauge), use_container_width=True)

st.markdown("---")

# TABS PRINCIPALI PER I 4 KPI E IL MACHINE LEARNING
tab_sma, tab_islr, tab_iitr, tab_idet, tab_ml = st.tabs([
    "01. SMA",
    "02. ISLR",
    "03. IITR",
    "04. IDET",
    "05. Dai KPI alle previsioni"
])

giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

# ==================================================================
# TAB 1 — SMA (Stress Mentale dell'Allenamento)
# ==================================================================
with tab_sma:
    st.markdown("### SMA — Stress Mentale dell'Allenamento")
    st.markdown("Quanto la tua mente e il tuo corpo sono sotto pressione oggi.")

    colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
    delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
    perc_sma = _calcola_percentile(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)

    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore di Oggi</div>
            <div style='color: {colore_sma}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['SMA']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto all'ultima sessione</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_sma)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto al tuo storico</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{_testo_percentile(perc_sma)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
    st.markdown("<div class='tech-box'><strong>Cosa significa:</strong> se questo numero è alto, vuol dire che ti sei stancato più di quanto il sonno sia riuscito a recuperarti.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### I tuoi ultimi 14 giorni, in 4 grafici")

    if kpi_storico is not None and 'SMA' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_sma_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['SMA'].tail(14), mode='lines+markers', line=dict(color='#00E5FF', width=2.5)))
            fig_sma_1.update_layout(title="1. Andamento nel tempo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> se vedi un picco improvviso, di solito vuol dire poco riposo unito a un giorno di lavoro pesante.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_sma_2 = go.Figure(go.Histogram(x=kpi_storico['SMA'], marker_color='#00E5FF', opacity=0.8, nbinsx=20))
            fig_sma_2.update_layout(title="2. Quanto spesso ti capita questo valore", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> ti mostra se il valore di oggi è nella norma per te, o è un'eccezione.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_sma_3 = go.Figure(go.Box(y=kpi_storico['SMA'], marker_color='#00E5FF', boxmean=True))
            fig_sma_3.update_layout(title="3. Dove ti collochi di solito", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> evidenzia le sessioni fuori dal tuo range abituale di stress e recupero.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_sma_4 = go.Figure(go.Scatter(y=kpi_storico['SMA'].rolling(3).mean(), mode='lines', line=dict(color='#FFB020', width=2)))
            fig_sma_4.update_layout(title="4. Tendenza delle ultime 3 sessioni", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> mostra se la fatica mentale si sta accumulando piano piano nelle ultime uscite.</div>", unsafe_allow_html=True)
    else:
        st.info("Servono ancora più sessioni per generare questi grafici di confronto.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> L'indicatore SMA unisce la sfera cognitiva (stress lavorativo/giornaliero) e la percezione dello sforzo (RPE) come fattori moltiplicativi del carico interno, normalizzandoli per la quantità di sonno notturno (fattore di recupero)[cite: 2]. Questo impedisce ai modelli predittivi di trattare un allenamento come un sistema puramente meccanico.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 2 — ISLR (Indice di Sforzo Lavorativo Residuo)
# ==================================================================
with tab_islr:
    st.markdown("### ISLR — Indice di Sforzo Lavorativo Residuo")
    st.markdown("Quanto il lavoro sta 'rubando' energie alla tua corsa.")

    colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
    delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
    perc_islr = _calcola_percentile(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)

    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore di Oggi</div>
            <div style='color: {colore_islr}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['ISLR']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto all'ultima sessione</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_islr)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto al tuo storico</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{_testo_percentile(perc_islr)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")
    st.markdown("<div class='tech-box'><strong>Cosa significa:</strong> misura quanto lo stress da lavoro ha pesato su questa corsa, chilometro per chilometro. Sopra 6.3 sei in zona di attenzione.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### I tuoi ultimi 14 giorni, in 4 grafici")

    if kpi_storico is not None and 'ISLR' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_islr_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['ISLR'].tail(14), mode='lines+markers', line=dict(color='#FF6A3D', width=2.5)))
            fig_islr_1.update_layout(title="1. Andamento nel tempo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> evidenzia i giorni in cui il lavoro ha pesato di più sulla corsa.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_islr_2 = go.Figure(go.Box(y=kpi_storico['ISLR'], marker_color='#FF6A3D', boxmean=True))
            fig_islr_2.update_layout(title="2. Quanto varia questo valore", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> utile per individuare le sessioni corse sotto forte stress da lavoro.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_islr_3 = go.Figure(go.Histogram(x=kpi_storico['ISLR'], marker_color='#FF6A3D', opacity=0.8, nbinsx=20))
            fig_islr_3.update_layout(title="3. Quanto spesso ti capita questo valore", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> mostra dove si concentrano i tuoi valori rispetto alla soglia di attenzione.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_islr_4 = go.Figure(go.Scatter(y=kpi_storico['ISLR'].rolling(3).mean(), mode='lines', line=dict(color='#00E5FF', width=2)))
            fig_islr_4.update_layout(title="4. Tendenza delle ultime 3 sessioni", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> valuta se il carico di lavoro extra-sportivo sta persistendo nel breve periodo.</div>", unsafe_allow_html=True)
    else:
        st.info("Servono ancora più sessioni per generare questi grafici di confronto.")

    if kpi_oggi["ISLR"] >= 6.3:
        st.markdown("<div style='background: rgba(255,106,61,0.08); border-left: 3px solid #FF6A3D; padding: 12px; border-radius: 0 6px 6px 0; margin-top: 10px; color: #FF6A3D;'><strong>Attenzione:</strong> oggi il carico di lavoro sta pesando molto sulle tue energie. Valuta di alleggerire l'allenamento.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Basato sul principio del <em>Total Life Stress</em>, l'ISLR riconosce che lo stress occupazionale compete direttamente con le risorse metaboliche e neuromuscolari necessarie alla supercompensazione sportiva[cite: 2]. Non a caso, risulta essere la variabile con il peso predittivo più alto nel Random Forest.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 3 — IITR (Indice Impatto Termico e Resistenza)
# ==================================================================
with tab_iitr:
    st.markdown("### IITR — Indice Impatto Termico e Resistenza")
    st.markdown("Quanto caldo e vento hanno reso più dura la sessione.")

    colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
    delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
    perc_iitr = _calcola_percentile(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)

    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore di Oggi</div>
            <div style='color: {colore_iitr}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['IITR']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto all'ultima sessione</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_iitr)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto al tuo storico</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{_testo_percentile(perc_iitr)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"IITR = \frac{\text{Temperatura} \times \text{Vento}}{\text{Distanza (km)}}")
    st.markdown("<div class='tech-box'><strong>Cosa significa:</strong> misura quanto caldo e vento hanno reso più faticosa la corsa, per ogni chilometro percorso.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### I tuoi ultimi 14 giorni, in 4 grafici")

    if kpi_storico is not None and 'IITR' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_iitr_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['IITR'].tail(14), mode='lines+markers', line=dict(color='#FFB020', width=2.5)))
            fig_iitr_1.update_layout(title="1. Andamento nel tempo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> evidenzia quanto sono state impegnative le condizioni meteo nelle tue sessioni.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_iitr_2 = go.Figure(go.Scatter(y=kpi_storico['IITR'], fill='tozeroy', marker_color='#FFB020', opacity=0.3))
            fig_iitr_2.update_layout(title="2. Accumulo nel tempo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> mostra come si è accumulato lo stress da caldo e vento nel tempo.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_iitr_3 = go.Figure(go.Histogram(x=kpi_storico['IITR'], marker_color='#FFB020', opacity=0.8, nbinsx=20))
            fig_iitr_3.update_layout(title="3. Quanto spesso ti capita questo valore", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> aiuta a individuare le sessioni corse nelle condizioni climatiche più difficili.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_iitr_4 = go.Figure(go.Box(y=kpi_storico['IITR'], marker_color='#FFB020', boxmean=True))
            fig_iitr_4.update_layout(title="4. Dove ti collochi di solito", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> valuta quanto variano le condizioni climatiche affrontate nel periodo considerato.</div>", unsafe_allow_html=True)
    else:
        st.info("Servono ancora più sessioni per generare questi grafici di confronto.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Poiché la raccolta dati si è sviluppata nei mesi estivi, l'IITR consente di isolare la componente climatica, quantificando il costo energetico aggiuntivo imposto dalle condizioni atmosferiche avverse[cite: 2].
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 4 — IDET (Indice di Degradazione Termica)
# ==================================================================
with tab_idet:
    st.markdown("### IDET — Indice di Degradazione Termica")
    st.markdown("Se il cuore ha lavorato più del normale a causa del caldo.")

    val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0

    colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
    delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
    perc_idet = _calcola_percentile(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)

    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore di Oggi</div>
            <div style='color: {colore_idet}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{val_idet:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto all'ultima sessione</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_idet)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Rispetto al tuo storico</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{_testo_percentile(perc_idet)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temperatura}}{\text{Velocità (km/h)}}")
    st.markdown("<div class='tech-box'><strong>Cosa significa:</strong> misura quanto il tuo cuore ha lavorato più del dovuto per via del caldo, evitando falsi allarmi di sovrallenamento nei giorni più caldi.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### I tuoi ultimi 14 giorni, in 4 grafici")

    if kpi_storico is not None and 'IDET' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_idet_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['IDET'].tail(14), mode='lines+markers', line=dict(color='#00F5A0', width=2.5)))
            fig_idet_1.update_layout(title="1. Andamento nel tempo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> evidenzia i giorni in cui il caldo ha fatto lavorare di più il tuo cuore.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_idet_2 = go.Figure(go.Violin(y=kpi_storico['IDET'], marker_color='#00F5A0', box_visible=True, meanline_visible=True))
            fig_idet_2.update_layout(title="2. Distribuzione dei tuoi valori", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> mostra dove si concentra di solito il costo cardiaco legato al caldo.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_idet_3 = go.Figure(go.Histogram(x=kpi_storico['IDET'], marker_color='#00F5A0', opacity=0.8, nbinsx=20))
            fig_idet_3.update_layout(title="3. Quanto spesso ti capita questo valore", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> aiuta a valutare quanto sei stabile nella gestione del caldo nel lungo periodo.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_idet_4 = go.Figure(go.Scatter(y=kpi_storico['IDET'].rolling(3).mean(), mode='lines', line=dict(color='#FF6A3D', width=2)))
            fig_idet_4.update_layout(title="4. Tendenza delle ultime 3 sessioni", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> mostra se ti stai adattando progressivamente alle condizioni di caldo.</div>", unsafe_allow_html=True)
    else:
        st.info("Servono ancora più sessioni per generare questi grafici di confronto.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Basato sui principi di Galloway e Maughan (1997), l'IDET corregge i dati cardiaci in base al calore[cite: 2]. Senza questo indice, un modello di machine learning interpreterebbe erroneamente la sessione estiva come un sintomo di imminente overtraining.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 5 — UNIONE CON MACHINE LEARNING (VERSIONE ARRICCHITA)
# ==================================================================
with tab_ml:
    st.markdown("### Dai KPI alle previsioni")
    st.markdown("""
    I quattro indicatori non restano numeri isolati: alimentano i modelli che provano a prevedere
    il rischio di sovraccarico prima che si presenti. Qui sotto trovi il quadro completo: cosa pesa
    di più, come si muovono nel tempo insieme, dove potrebbero andare nei prossimi giorni, e cosa
    cambierebbe se modificassi le tue abitudini.
    """)

    # ---------------------------------------------------------
    # RIGA 1 — Feature importance + breakdown del rischio odierno
    # ---------------------------------------------------------
    c_m1, c_m2 = st.columns(2, gap="large")
    with c_m1:
        st.markdown("#### Quali fattori contano di più nel calcolo del rischio")
        st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
        st.markdown("<div class='tech-box'><strong>Cosa dicono i dati:</strong> l'<strong>ISLR</strong> (stress da lavoro) e la qualità del sonno sono i fattori che pesano di più nel prevedere il rischio di sovraccarico.</div>", unsafe_allow_html=True)

    with c_m2:
        st.markdown("#### Da cosa dipende il rischio di oggi")
        if dettaglio_scores:
            nomi = list(dettaglio_scores.keys())
            valori = [dettaglio_scores[k] for k in nomi]

            fig_breakdown = go.Figure(go.Bar(
                x=valori, y=nomi, orientation="h",
                marker=dict(color=valori, colorscale=[[0, "#00F5A0"], [0.5, "#FFB020"], [1, "#FF6A3D"]])
            ))
            fig_breakdown.update_layout(
                height=350, xaxis_title="Contributo al rischio complessivo (0-100)",
                margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(style_fig(fig_breakdown), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Perché è utile:</strong> vedi subito quale dei 4 indicatori sta pesando di più sul tuo rischio di oggi, invece di ricevere solo un numero senza spiegazione.</div>", unsafe_allow_html=True)
        else:
            st.info("Dettaglio non disponibile per questa sessione.")

    st.markdown("---")

    # ---------------------------------------------------------
    # RIGA 2 — Radar del profilo + correlazione tra indicatori
    # ---------------------------------------------------------
    st.markdown("#### Il tuo profilo a colpo d'occhio")
    col_radar, col_corr = st.columns([1.1, 1], gap="large")

    percentili_disponibili = {
        "SMA": perc_sma if 'perc_sma' in dir() else None,
        "ISLR": perc_islr if 'perc_islr' in dir() else None,
        "IITR": perc_iitr if 'perc_iitr' in dir() else None,
        "IDET": perc_idet if 'perc_idet' in dir() else None,
    }

    with col_radar:
        categorie = ["SMA", "ISLR", "IITR", "IDET"]
        valori_radar = [percentili_disponibili[c] if percentili_disponibili[c] is not None else 50 for c in categorie]
        categorie_chiuse = categorie + [categorie[0]]
        valori_chiusi = valori_radar + [valori_radar[0]]
        media_chiusa = [50] * len(categorie_chiuse)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=media_chiusa, theta=categorie_chiuse, mode='lines',
            line=dict(color='#566178', width=1, dash='dot'), name='Il tuo storico medio'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=valori_chiusi, theta=categorie_chiuse, fill='toself',
            fillcolor='rgba(0,229,255,0.18)', line=dict(color='#00E5FF', width=2.5),
            name='Oggi'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.08)', tickfont=dict(size=10)),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(size=12, color="#D1D5DB")),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True, legend=dict(orientation="h", y=-0.1),
            height=360, margin=dict(l=50, r=50, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(style_fig(fig_radar), use_container_width=True)
        st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> ogni punta è il percentile di oggi rispetto al tuo storico su quell'indicatore. Più ti allontani dal centro, più quel valore è alto rispetto al tuo normale — non rispetto ad altre persone.</div>", unsafe_allow_html=True)

    with col_corr:
        if kpi_storico is not None and all(c in kpi_storico.columns for c in ["SMA", "ISLR", "IITR", "IDET"]) and len(kpi_storico.dropna(how="all")) >= 5:
            corr_df = kpi_storico[["SMA", "ISLR", "IITR", "IDET"]].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr_df.values, x=list(corr_df.columns), y=list(corr_df.columns),
                colorscale=[[0, "#FF6A3D"], [0.5, "#0F172A"], [1, "#00E5FF"]], zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_df.values, 2), texttemplate="%{text}",
                textfont=dict(size=13, color="#FFFFFF"), showscale=False
            ))
            fig_corr.update_layout(
                height=360, margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(style_fig(fig_corr), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> quando due indicatori si muovono spesso insieme (colore acceso), un peggioramento nell'uno tende ad accompagnarsi all'altro.</div>", unsafe_allow_html=True)
        else:
            st.info("Servono più sessioni storiche per calcolare le correlazioni tra indicatori.")

    st.markdown("---")

    # ---------------------------------------------------------
    # RIGA 3 — Andamento combinato storico + proiezione
    # ---------------------------------------------------------
    st.markdown("#### Dove sta andando il tuo rischio")

    if kpi_storico is not None and len(kpi_storico.dropna(how="all")) >= 5:
        cols_kpi = [c for c in ["SMA", "ISLR", "IITR", "IDET"] if c in kpi_storico.columns]
        norm_df = pd.DataFrame(index=kpi_storico.index)
        for c in cols_kpi:
            serie = kpi_storico[c]
            rng = serie.max() - serie.min()
            norm_df[c] = ((serie - serie.min()) / rng * 100) if pd.notna(rng) and rng > 0 else 50

        indice_combinato = norm_df.mean(axis=1).tail(14).reset_index(drop=True)
        x_storico = list(range(len(indice_combinato)))

        proiezione_x, proiezione_y = [], []
        y_validi = indice_combinato.dropna()
        if len(y_validi) >= 3:
            x_validi = np.arange(len(y_validi))
            coeff = np.polyfit(x_validi, y_validi.values, 1)
            ultimo_x = x_storico[-1]
            proiezione_x = list(range(ultimo_x, ultimo_x + 6))
            proiezione_y = [float(np.clip(coeff[0] * px + coeff[1], 0, 100)) for px in range(len(x_validi) - 1, len(x_validi) + 5)]

        fig_proj = go.Figure()
        fig_proj.add_hrect(y0=0, y1=33, fillcolor="rgba(0,245,160,0.05)", line_width=0)
        fig_proj.add_hrect(y0=33, y1=66, fillcolor="rgba(255,176,32,0.05)", line_width=0)
        fig_proj.add_hrect(y0=66, y1=100, fillcolor="rgba(255,106,61,0.05)", line_width=0)
        fig_proj.add_trace(go.Scatter(
            x=x_storico, y=indice_combinato, mode='lines+markers',
            line=dict(color='#00E5FF', width=2.5), name='Storico (ultimi 14gg)'
        ))
        if proiezione_x:
            fig_proj.add_trace(go.Scatter(
                x=proiezione_x, y=proiezione_y, mode='lines',
                line=dict(color='#FFB020', width=2.5, dash='dash'), name='Proiezione stimata'
            ))
        fig_proj.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Indice combinato (0-100)", yaxis_range=[0, 100],
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(style_fig(fig_proj), use_container_width=True)
        st.markdown("<div class='tech-box'><strong>Come leggerlo:</strong> la linea tratteggiata è una stima basata sull'andamento recente, non una previsione certa: se continui con lo stesso ritmo di sonno, lavoro e allenamento, è lì che potresti dirigerti nei prossimi giorni.</div>", unsafe_allow_html=True)
    else:
        st.info("Servono più sessioni storiche per calcolare una proiezione affidabile.")

    st.markdown("---")

    # ---------------------------------------------------------
    # RIGA 4 — Simulatore what-if
    # ---------------------------------------------------------
    st.markdown("#### Simulatore: cosa succederebbe se...")
    st.markdown("Muovi i cursori per vedere come cambierebbe il tuo rischio con abitudini diverse, a parità di tutto il resto.")

    sonno_attuale = float(r.get(COL_SONNO, r.get("ore_sonno", 7.0)))
    volume_attuale = float(r.get("volume_settimanale_km", df_base[COL_DISTANZA].tail(7).sum() if COL_DISTANZA in df_base else 25.0))
    passo_attuale = float(r.get("passo_medio", 5.0))

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sonno_sim = st.slider("Ore di sonno", 3.0, 10.0, sonno_attuale, 0.5, key="sim_sonno")
    with sim_c2:
        volume_sim = st.slider("Volume settimanale (km)", 0.0, 100.0, min(volume_attuale, 100.0), 1.0, key="sim_volume")
    with sim_c3:
        passo_sim = st.slider("Passo medio (min/km)", 3.5, 8.0, min(max(passo_attuale, 3.5), 8.0), 0.1, key="sim_passo")

    try:
        risk_sim, _ = calcola_risk_score_pesato(
            oggi={
                "ISLR": kpi_oggi["ISLR"],
                "IDET": kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0,
                "Ore Sonno": sonno_sim,
                "Volume Settimanale": volume_sim,
                "Passo Medio": passo_sim,
            },
            storico=kpi_storico if kpi_storico is not None else pd.DataFrame(),
        )
    except Exception:
        risk_sim = risk_score

    delta_sim = risk_sim - risk_score
    colore_sim = "#00F5A0" if risk_sim < 25 else "#FFB020" if risk_sim < 60 else "#FF6A3D"

    sim_res1, sim_res2 = st.columns([1, 1.4], gap="large")
    with sim_res1:
        fig_gauge_sim = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_sim,
            number={'suffix': "%", 'font': {'color': "#FFFFFF", 'size': 30}},
            delta={'reference': risk_score, 'increasing': {'color': '#FF6A3D'}, 'decreasing': {'color': '#00F5A0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 11}},
                'bar': {'color': colore_sim, 'thickness': 0.65},
                'bgcolor': "rgba(255,255,255,0.02)",
                'steps': [
                    {'range': [0, 25], 'color': "rgba(0,245,160,0.1)"},
                    {'range': [25, 60], 'color': "rgba(255,176,32,0.1)"},
                    {'range': [60, 100], 'color': "rgba(255,106,61,0.1)"}
                ],
            }
        ))
        fig_gauge_sim.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(style_fig(fig_gauge_sim), use_container_width=True)

    with sim_res2:
        if delta_sim > 0.5:
            verso, colore_verso = "aumenterebbe", "#FF6A3D"
        elif delta_sim < -0.5:
            verso, colore_verso = "diminuirebbe", "#00F5A0"
        else:
            verso, colore_verso = "resterebbe stabile", "#8792A3"
        st.markdown(f"""
        <div class='tech-box' style='font-size:1em; border-left-color: {colore_verso};'>
            Con queste abitudini, il tuo rischio <strong style='color:{colore_verso};'>{verso}</strong>
            di circa <strong>{abs(delta_sim):.1f} punti</strong> rispetto a oggi
            (<strong>{risk_score:.0f}%</strong> → <strong>{risk_sim:.0f}%</strong>).
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='tech-box'>
            Stai simulando: <strong>{sonno_sim:.1f}h</strong> di sonno,
            <strong>{volume_sim:.0f} km</strong> di volume settimanale,
            passo medio <strong>{passo_sim:.1f} min/km</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # RIGA 5 — Raccomandazione basata sul driver principale
    # ---------------------------------------------------------
    if dettaglio_scores:
        driver_principale = max(dettaglio_scores, key=dettaglio_scores.get)
        mappa_consigli = {
            "ISLR": "Il lavoro sta pesando più del solito sulle tue energie: se puoi, oggi valuta un allenamento più corto o più leggero.",
            "IITR": "Le condizioni climatiche stanno rendendo le uscite più dure del normale: idratati di più e considera orari più freschi.",
            "IDET": "Il tuo cuore sta lavorando più del dovuto per via del caldo: non è un allarme di forma, ma di temperatura.",
            "SMA": "Stress e fatica percepita sono alti rispetto al sonno recuperato: dai priorità al riposo prima del prossimo allenamento.",
        }
        st.markdown(f"""
        <div class='metric-card-horizontal' style='border-left: 3px solid #00E5FF;'>
            <div>
                <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Indicatore che sta pesando di più oggi</div>
                <div style='color: #00E5FF; font-size: 1.8em; font-weight: 800; margin: 4px 0;'>{driver_principale}</div>
            </div>
            <div style='flex: 1; min-width: 250px;'>
                <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Cosa vuol dire in pratica</div>
                <div style='color: #FFFFFF; font-size: 1em; margin-top: 4px; line-height: 1.4;'>{mappa_consigli.get(driver_principale, "Continua a monitorare i tuoi indicatori nei prossimi giorni.")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### In sintesi
    Unendo i 4 indicatori (SMA, ISLR, IITR, IDET) ai modelli predittivi, i dati grezzi diventano un aiuto concreto per decidere quando allenarsi e quando riposare[cite: 2].
    """)
