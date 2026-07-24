"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET).
Design High-Tech rigoroso, privo di emoji, con layout verticale esteso,
cruscotto indicatore grafico a stanghetta/gauge ingrandito per il rischio e griglia 2x2 sottostante.
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
    "Modulo 04 — Centro KPI & Masterclass Intelligence",
    "PROPRIETARY KPI ENGINE",
    "Analisi metrica avanzata per singolo indicatore con disposizione verticale estesa e griglia analitica 2x2 sottostante.",
    IMG_HERO_KPI, "Proprietary KPI & AI Engine"
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("Completare preliminarmente il questionario nella pagina 'ANALISI STATO DI FORMA' per inizializzare il calcolo dei KPI.")
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
        <div style='color: #8792A3; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1.5px;'>Stato di Prontezza Operativa</div>
        <div style='font-size: 2.2em; font-weight: 800; color: {status_color}; margin-top: 4px;'>
            INDICE DI RISCHIO {status_text} <span style='font-size: 0.65em; font-weight: 400; color: #FFFFFF;'>({risk_score:.0f}%)</span>
        </div>
        <div style='color: #B8C2D0; font-size: 0.85em; line-height: 1.4; margin-top: 10px;'>
            Il punteggio aggrega i vettori KPI pesandoli direttamente sulla reale Feature Importance estratta dal Random Forest.
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
    "05. Machine Learning Integration"
])

giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

# ==================================================================
# TAB 1 — SMA (Stress Mentale dell'Allenamento)
# ==================================================================
with tab_sma:
    st.markdown("### Modulo Analitico — SMA (Stress Mentale dell'Allenamento)")
    st.markdown("Analisi metrica avanzata finalizzata alla quantificazione della vulnerabilità neurale e psicofisica.")
    
    colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
    delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
    perc_sma = _calcola_percentile(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
    
    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore Istantaneo</div>
            <div style='color: {colore_sma}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['SMA']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Variazione Temporale</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_sma)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Posizione Statistica</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{f"{perc_sma:.0f}° percentile" if perc_sma is not None else "N/D"} rispetto allo storico</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"SMA = \frac{\text{Stress Giornata} \times \text{RPE}}{\text{Ore Sonno}}")
    st.markdown("<div class='tech-box'><strong>Interpretazione Operativa:</strong> Valori elevati di SMA evidenziano una sproporzione tra la fatica cognitiva accumulata e la capacità di ripristino sistemico garantita dal sonno.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Scomposizione Analitica (Griglia di Controllo 2x2)")
    
    if kpi_storico is not None and 'SMA' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_sma_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['SMA'].tail(14), mode='lines+markers', line=dict(color='#00E5FF', width=2.5)))
            fig_sma_1.update_layout(title="1. Trend Longitudinale", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Evoluzione temporale dell'indice SMA. Picchi improvvisi segnalano scarso riposo associato a stress lavorativo intenso.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_sma_2 = go.Figure(go.Histogram(x=kpi_storico['SMA'], marker_color='#00E5FF', opacity=0.8, nbinsx=20))
            fig_sma_2.update_layout(title="2. Densità di Popolazione", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Distribuzione statistica sull'intero set di dati per verificare se il valore odierno si colloca nella norma o nelle code.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_sma_3 = go.Figure(go.Box(y=kpi_storico['SMA'], marker_color='#00E5FF', boxmean=True))
            fig_sma_3.update_layout(title="3. Analisi dei Quartili", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Boxplot di controllo per isolare eventuali valori anomali nella risposta neurale e di recupero dell'atleta.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_sma_4 = go.Figure(go.Scatter(y=kpi_storico['SMA'].rolling(3).mean(), mode='lines', line=dict(color='#FFB020', width=2)))
            fig_sma_4.update_layout(title="4. Media Mobile (3 Sessioni)", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_sma_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Trend smussato a breve termine per evidenziare l'accumulo latente della fatica cognitiva nelle ultime sedute.</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset storico insufficiente per la generazione dei grafici multi-asse.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> L'indicatore SMA unisce la sfera cognitiva (stress lavorativo/giornaliero) e la percezione dello sforzo (RPE) come fattori moltiplicativi del carico interno, normalizzandoli per la quantità di sonno notturno (fattore di recupero)[cite: 2]. Questo impedisce ai modelli predittivi di trattare un allenamento come un sistema puramente meccanico.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 2 — ISLR (Indice di Sforzo Lavorativo Residuo)
# ==================================================================
with tab_islr:
    st.markdown("### Modulo Analitico — ISLR (Indice di Sforzo Lavorativo Residuo)")
    st.markdown("Indicatore core per la profilazione dell'atleta amatore (*worker-athlete*).")

    colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
    delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
    perc_islr = _calcola_percentile(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
    
    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore Istantaneo</div>
            <div style='color: {colore_islr}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['ISLR']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Variazione Temporale</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_islr)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Posizione Statistica</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{f"{perc_islr:.0f}° percentile" if perc_islr is not None else "N/D"} rispetto allo storico</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"ISLR = \frac{\text{Ore Lavoro} \times \text{Stress Mentale}}{\text{Distanza (km)}}")
    st.markdown("<div class='tech-box'><strong>Interpretazione Operativa:</strong> Quantifica la densità di stress non-atletico per unità di distanza percorsa[cite: 2]. Superamento della soglia critica fissata a 6.3 unità.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Scomposizione Analitica (Griglia di Controllo 2x2)")
    
    if kpi_storico is not None and 'ISLR' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_islr_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['ISLR'].tail(14), mode='lines+markers', line=dict(color='#FF6A3D', width=2.5)))
            fig_islr_1.update_layout(title="1. Trend Longitudinale", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Monitora l'andamento nel tempo dell'ISLR, evidenziando i giorni in cui i carichi lavorativi professionali hanno pesato sul chilometraggio.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_islr_2 = go.Figure(go.Box(y=kpi_storico['ISLR'], marker_color='#FF6A3D', boxmean=True))
            fig_islr_2.update_layout(title="2. Analisi di Dispersione", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Boxplot della variabilità dell'ISLR, utile per individuare le sessioni eseguite sotto estremo stress occupazionale.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_islr_3 = go.Figure(go.Histogram(x=kpi_storico['ISLR'], marker_color='#FF6A3D', opacity=0.8, nbinsx=20))
            fig_islr_3.update_layout(title="3. Distribuzione di Frequenza", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Istogramma che mostra la concentrazione dei valori di sforzo residuo rispetto alla soglia limite di sicurezza.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_islr_4 = go.Figure(go.Scatter(y=kpi_storico['ISLR'].rolling(3).mean(), mode='lines', line=dict(color='#00E5FF', width=2)))
            fig_islr_4.update_layout(title="4. Media Mobile (3 Sessioni)", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_islr_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Trend mobile per valutare la persistenza del carico lavorativo extra-sportivo nel breve periodo.</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset storico insufficiente per la generazione dei grafici multi-asse.")

    if kpi_oggi["ISLR"] >= 6.3:
        st.markdown("<div style='background: rgba(255,106,61,0.08); border-left: 3px solid #FF6A3D; padding: 12px; border-radius: 0 6px 6px 0; margin-top: 10px; color: #FF6A3D;'><strong>Avviso di Sistema:</strong> Il carico lavorativo odierno satura le risorse energetiche. Si raccomanda la rimodulazione dell'intensità.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Basato sul principio del <em>Total Life Stress</em>, l'ISLR riconosce che lo stress occupazionale compete direttamente con le risorse metaboliche e neuromuscolari necessarie alla supercompensazione sportiva[cite: 2]. Non a caso, risulta essere la variabile con il peso predittivo più alto nel Random Forest.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 3 — IITR (Indice Impatto Termico e Resistenza)
# ==================================================================
with tab_iitr:
    st.markdown("### Modulo Analitico — IITR (Indice Impatto Termico e Resistenza)")
    st.markdown("Indicatore ambientale finalizzato alla misurazione della severità meteorologica esogena.")

    colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
    delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
    perc_iitr = _calcola_percentile(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
    
    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore Istantaneo</div>
            <div style='color: {colore_iitr}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{kpi_oggi['IITR']:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Variazione Temporale</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_iitr)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Posizione Statistica</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{f"{perc_iitr:.0f}° percentile" if perc_iitr is not None else "N/D"} rispetto allo storico</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"IITR = \frac{\text{Temperatura} \times \text{Vento}}{\text{Distanza (km)}}")
    st.markdown("<div class='tech-box'><strong>Interpretazione Operativa:</strong> Pesa le forze resistive esterne (termiche e aerodinamiche) per chilometro[cite: 2], evidenziando contesti ad alto attrito metabolico.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Scomposizione Analitica (Griglia di Controllo 2x2)")
    
    if kpi_storico is not None and 'IITR' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_iitr_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['IITR'].tail(14), mode='lines+markers', line=dict(color='#FFB020', width=2.5)))
            fig_iitr_1.update_layout(title="1. Trend Longitudinale", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Serie temporale dell'IITR per evidenziare la gravità delle condizioni meteorologiche incontrate nelle sedute.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_iitr_2 = go.Figure(go.Scatter(y=kpi_storico['IITR'], fill='tozeroy', marker_color='#FFB020', opacity=0.3))
            fig_iitr_2.update_layout(title="2. Profilo Cumulativo", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Area plot che illustra l'accumulo di stress esogeno derivato da temperature elevate e ventilazione contraria.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_iitr_3 = go.Figure(go.Histogram(x=kpi_storico['IITR'], marker_color='#FFB020', opacity=0.8, nbinsx=20))
            fig_iitr_3.update_layout(title="3. Distribuzione di Frequenza", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Istogramma della popolazione termica per identificare le sessioni esposte a microclimi critici.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_iitr_4 = go.Figure(go.Box(y=kpi_storico['IITR'], marker_color='#FFB020', boxmean=True))
            fig_iitr_4.update_layout(title="4. Analisi dei Quartili", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_iitr_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Boxplot di controllo per valutare la dispersione dei fattori climatici resistivi nel corso del periodo considerato.</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset storico insufficiente per la generazione dei grafici multi-asse.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Poiché la raccolta dati si è sviluppata nei mesi estivi, l'IITR consente di isolare la componente climatica, quantificando il costo energetico aggiuntivo imposto dalle condizioni atmosferiche avverse[cite: 2].
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 4 — IDET (Indice di Degradazione Termica)
# ==================================================================
with tab_idet:
    st.markdown("### Modulo Analitico — IDET (Indice di Degradazione Termica)")
    st.markdown("Indicatore cinematico e termoregolatorio per il controllo della deriva cardiaca.")

    val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0

    colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
    delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
    perc_idet = _calcola_percentile(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
    
    st.markdown(f"""
    <div class='metric-card-horizontal'>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Valore Istantaneo</div>
            <div style='color: {colore_idet}; font-size: 2.5em; font-weight: 800; margin: 4px 0;'>{val_idet:.2f}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Variazione Temporale</div>
            <div style='margin-top: 8px;'>{_badge_delta(delta_idet)}</div>
        </div>
        <div>
            <div style='color: #8792A3; font-size: 0.8em; text-transform: uppercase;'>Posizione Statistica</div>
            <div style='color: #FFFFFF; font-size: 1.1em; font-weight: 600; margin-top: 4px;'>{f"{perc_idet:.0f}° percentile" if perc_idet is not None else "N/D"} rispetto allo storico</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"IDET = \frac{\text{FC Media} \times \text{Temperatura}}{\text{Velocità (km/h)}}")
    st.markdown("<div class='tech-box'><strong>Interpretazione Operativa:</strong> Mappa l'efficienza cardiaca in regime di stress termico, prevenendo falsi positivi nei modelli di overtraining[cite: 2].</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Scomposizione Analitica (Griglia di Controllo 2x2)")
    
    if kpi_storico is not None and 'IDET' in kpi_storico.columns:
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        row2_c1, row2_c2 = st.columns(2, gap="medium")

        with row1_c1:
            fig_idet_1 = go.Figure(go.Scatter(x=giorni_asse, y=kpi_storico['IDET'].tail(14), mode='lines+markers', line=dict(color='#00F5A0', width=2.5)))
            fig_idet_1.update_layout(title="1. Trend Longitudinale", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_1), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Andamento temporale dell'IDET per evidenziare i giorni in cui la frequenza cardiaca ha subito alterazioni per il calore.</div>", unsafe_allow_html=True)

        with row1_c2:
            fig_idet_2 = go.Figure(go.Violin(y=kpi_storico['IDET'], marker_color='#00F5A0', box_visible=True, meanline_visible=True))
            fig_idet_2.update_layout(title="2. Analisi di Densità", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_2), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Grafico a violino che unisce il boxplot alla densità di probabilità, mostrando la concentrazione dei costi cardiaci-termici.</div>", unsafe_allow_html=True)

        with row2_c1:
            fig_idet_3 = go.Figure(go.Histogram(x=kpi_storico['IDET'], marker_color='#00F5A0', opacity=0.8, nbinsx=20))
            fig_idet_3.update_layout(title="3. Distribuzione di Frequenza", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_3), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Istogramma di frequenza dell'indice di degradazione per valutare la stabilità termoregolatoria nel lungo periodo.</div>", unsafe_allow_html=True)

        with row2_c2:
            fig_idet_4 = go.Figure(go.Scatter(y=kpi_storico['IDET'].rolling(3).mean(), mode='lines', line=dict(color='#FF6A3D', width=2)))
            fig_idet_4.update_layout(title="4. Media Mobile (3 Sessioni)", height=160, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(style_fig(fig_idet_4), use_container_width=True)
            st.markdown("<div class='tech-box'><strong>Spiegazione:</strong> Trend mobile della deriva cardiaca per monitorare l'adattamento progressivo dell'atleta alle condizioni climatiche.</div>", unsafe_allow_html=True)
    else:
        st.info("Dataset storico insufficiente per la generazione dei grafici multi-asse.")

    st.markdown("""
    <div class='theory-panel'>
    <strong>Razionale Scientifico (Tesi):</strong> Basato sui principi di Galloway e Maughan (1997), l'IDET corregge i dati cardiaci in base al calore[cite: 2]. Senza questo indice, un modello di machine learning interpreterebbe erroneamente la sessione estiva come un sintomo di imminente overtraining.
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 5 — UNIONE CON MACHINE LEARNING
# ==================================================================
with tab_ml:
    st.markdown("### Convergenza Architetturale: Dai KPI Proprietari al Machine Learning")
    st.markdown("""
    I quattro indicatori non operano in modo isolato, ma costituiscono l'architettura di *feature engineering* che alimenta i modelli predittivi supervisionati (Random Forest e regressioni)[cite: 2].
    """)

    c_m1, c_m2 = st.columns(2, gap="large")
    with c_m1:
        st.markdown("#### Ponderazione Algoritmica (Feature Importance)")
        st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
        st.markdown("<div class='tech-box'><strong>Evidenza di Tesi:</strong> L'algoritmo individua nell'<strong>ISLR</strong> e nella qualità del sonno i driver decisionali primari, gestendo oltre il 50% della varianza associata al rischio di sovraccarico[cite: 2].</div>", unsafe_allow_html=True)

    with c_m2:
        st.markdown("#### Scomposizione Analitica del Rischio Odierno")
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
            st.markdown("<div class='tech-box'><strong>Trasparenza Computazionale:</strong> L'output probabilistico viene disaggregato per isolare il contributo specifico di ciascun KPI, eliminando la criticità della 'scatola nera' algoritmica[cite: 2].</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Sintesi Metodologica per la Discussione di Tesi
    L'integrazione strutturata tra l'ingegneria delle feature (SMA, ISLR, IITR, IDET) e il machine learning supervisionato trasforma la raccolta dati in un sistema di supporto decisionale proattivo di livello enterprise[cite: 2].
    """)
