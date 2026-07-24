"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET),
il risk_score pesato sulla Feature Importance reale, e visualizzazioni avanzate
inclusi radar chart multidimensionali e analisi di densità statistica.
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
    calcola_sma,
    calcola_islr,
    calcola_iitr,
    calcola_idet,
    calcola_kpi_giornalieri,
    calcola_risk_score_pesato,
    FEATURE_IMPORTANCE_CHART_DATA,
    COL_SONNO, COL_DISTANZA
)

st.set_page_config(page_title="Centro KPI & Advanced Intelligence", layout="wide")
carica_css()

# ==================================================================
# STILE CUSTOM AGGIUNTIVO PER IL LOOK ENTERPRISE
# ==================================================================
st.markdown("""
<style>
    .kpi-main-container {
        background: linear-gradient(135deg, rgba(32,40,58,0.7) 0%, rgba(15,20,30,0.9) 100%);
        border: 1px solid rgba(0,229,255,0.25);
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .explain-text {
        color: #B8C2D0;
        font-size: 0.95em;
        line-height: 1.5;
        background: rgba(255,255,255,0.03);
        padding: 14px 18px;
        border-radius: 8px;
        border-left: 3px solid #00E5FF;
        margin-top: 12px;
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
    "Modulo 04 — Centro KPI & Advanced Intelligence",
    "I TUOI 4 INDICI PROPRIETARI",
    "Analisi multidimensionale avanzata: SMA, ISLR, IITR, IDET integrati con radar chart di profilo e breakdown di rischio.",
    IMG_HERO_KPI, "Proprietary KPI Engine"
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
    percentile = (serie_pulita < valore).mean() * 100
    return percentile


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

tab_oggi, tab_radar, tab_storico, tab_metodo = st.tabs([
    "📍 Situazione di Oggi", 
    "🕸️ Radar & Profilo Multidimensionale", 
    "📈 Andamento Storico & Trend", 
    "🔬 Metodologia & Pesi"
])

# ==================================================================
# TAB 1 — SITUAZIONE DI OGGI
# ==================================================================
with tab_oggi:
    st.markdown("### 🎯 Cruscotto di Sintesi dei 4 KPI Proprietari")
    st.markdown("Valutazione multidimensionale in tempo reale basata sullo stato di forma inserito e confrontata con il tuo storico comportamentale.")

    giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

    col1, col2 = st.columns(2)
    
    with col1:
        colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
        delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
        perc_sma = _calcola_percentile(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
        
        kpi_card_sparkline("SMA — Stress Mentale Allenamento", kpi_oggi["SMA"], colore_sma,
                           kpi_storico["SMA"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_sma), unsafe_allow_html=True)
        if perc_sma is not None:
            st.caption(f"📊 Valore nel **{perc_sma:.0f}° percentile** del tuo storico personale.")
        in_pratica("SMA alto = hai corso con poco sonno e molto stress accumulato: oggi il corpo lavora in svantaggio neurale[cite: 2].")

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
        delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
        perc_iitr = _calcola_percentile(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
        
        kpi_card_sparkline("IITR — Impatto Termico e Resistenza", kpi_oggi["IITR"], colore_iitr,
                           kpi_storico["IITR"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_iitr), unsafe_allow_html=True)
        if perc_iitr is not None:
            st.caption(f"📊 Valore nel **{perc_iitr:.0f}° percentile** del tuo storico personale.")
        in_pratica("IITR alto = caldo e vento hanno reso la corsa di oggi più dura del solito per ogni km percorso[cite: 2].")

    with col2:
        colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
        delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
        perc_islr = _calcola_percentile(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
        
        kpi_card_sparkline("ISLR — Sforzo Lavorativo Residuo", kpi_oggi["ISLR"], colore_islr,
                           kpi_storico["ISLR"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_islr), unsafe_allow_html=True)
        if perc_islr is not None:
            st.caption(f"📊 Valore nel **{perc_islr:.0f}° percentile** del tuo storico personale.")
        in_pratica("ISLR è il KPI più predittivo del tuo modello (31,5%): sopra 6,3 il tuo Random Forest classifica le sessioni come overload nel 50%+ dei casi[cite: 2].")
        
        if kpi_oggi["ISLR"] >= 6.3:
            azione_consigliata("⚠️ Oggi il tuo carico lavorativo sta 'mangiando' risorse che servirebbero alla corsa. Valuta una sessione più corta o rigenerativa invece che qualitativa.")

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0
        colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
        delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
        perc_idet = _calcola_percentile(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
        
        kpi_card_sparkline("IDET — Degradazione Termica", val_idet, colore_idet,
                           kpi_storico["IDET"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_idet), unsafe_allow_html=True)
        if perc_idet is not None:
            st.caption(f"📊 Valore nel **{perc_idet:.0f}° percentile** del tuo storico personale.")
        in_pratica("IDET alto = il caldo sta facendo salire i tuoi battiti più di quanto la velocità giustifichi (deriva cardiaca): non è un calo di forma, è il clima[cite: 2].")

    st.markdown("---")
    verdetto_box(
        100 - risk_score, soglie=(40, 75),
        testo_basso=f"🚨 Rischio elevato ({risk_score:.0f}%): la combinazione dei tuoi 4 KPI oggi indica una condizione di vulnerabilità — valuta di ridurre intensità o rimandare.",
        testo_medio=f"⚠️ Rischio moderato ({risk_score:.0f}%): allenati con attenzione, monitorando in particolare l'ISLR se lavori molto in questo periodo.",
        testo_alto=f"✅ Rischio basso ({risk_score:.0f}%): condizioni favorevoli su tutti e 4 gli indici.",
        spiegazione="Ogni KPI viene confrontato con il tuo storico personale, non con soglie generiche — coerente con l'approccio single-subject (N-of-1) della tua ricerca."
    )

# ==================================================================
# TAB 2 — RADAR CHART & PROFILO MULTIDIMENSIONALE (NOVITÀ RICHIESTA)
# ==================================================================
with tab_radar:
    st.markdown("### 🕸️ Profilo di Rischio Multidimensionale (Radar Chart)")
    st.markdown("Il grafico a ragnatela permette di visualizzare la forma geometrica del tuo carico allostatico odierno confrontato con la **Media del tuo Storico** e con la **Soglia Critica di Allarme**.")

    # Normalizzazione dei KPI su scala 0-100 per renderli comparabili sul radar
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

        soglia_critica = [70, 70, 70, 70] # Rappresenta il limite di guardia
        categorie = ['SMA (Stress Mentale)', 'ISLR (Lavoro Residuo)', 'IITR (Impatto Termico)', 'IDET (Degradazione)']

        fig_radar = go.Figure()

        # Traccia Storico
        fig_radar.add_trace(go.Scatterpolar(
            r=storico_norm + [storico_norm[0]],
            theta=categorie + [categorie[0]],
            fill='toself',
            name='Media Storica Personale',
            line=dict(color='#00E5FF', width=2),
            fillcolor='rgba(0, 229, 255, 0.1)'
        ))

        # Traccia di Oggi
        fig_radar.add_trace(go.Scatterpolar(
            r=oggi_norm + [oggi_norm[0]],
            theta=categorie + [categorie[0]],
            fill='toself',
            name='Profilo Odierno',
            line=dict(color='#FF6A3D', width=3),
            fillcolor='rgba(255, 106, 61, 0.25)'
        ))

        # Traccia Soglia Critica
        fig_radar.add_trace(go.Scatterpolar(
            r=soglia_critica + [soglia_critica[0]],
            theta=categorie + [categorie[0]],
            mode='lines',
            name='Soglia Critica di Allarme',
            line=dict(color='#FFB020', width=1.5, dash='dash')
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)")
            ),
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(style_fig(fig_radar), use_container_width=True)

    except Exception as e:
        st.info("ℹ️ Dati insufficienti per generare il grafico radar multidimensionale.")

    st.markdown(
        "<div class='explain-text'>"
        "<strong>🧠 Lettura Avanzata (Tesi Magistrale):</strong> Il radar chart permette di cogliere asimmetrie strutturali nel carico. "
        "Se la poligonale di oggi (in arancione) si estende oltre la linea tratteggiata della soglia critica o si discosta marcatamente "
        "dalla media storica (in azzurro), evidenzia una specifica fonte di sbilanciamento (es. stress lavorativo o termico elevato) "
        "anche quando il volume chilometrico è basso."
        "</div>", unsafe_allow_html=True
    )

# ==================================================================
# TAB 3 — ANDAMENTO STORICO REALE
# ==================================================================
with tab_storico:
    if kpi_storico is None or len(kpi_storico) < 3:
        st.info("ℹ️ Servono almeno 3 sessioni storiche per mostrare un andamento significativo. Continua a registrare le tue corse.")
    else:
        st.markdown("### 📈 Dinamica Temporale dei 4 KPI")
        st.caption("Ogni pannello mantiene la propria scala naturale: i 4 indici descrivono l'evoluzione longitudinale del carico allostatico.")

        n_storico = min(30, len(kpi_storico))
        finestra = kpi_storico.tail(n_storico).reset_index(drop=True)
        asse_x = df_base['Giorno'].tail(n_storico).tolist() if 'Giorno' in df_base.columns else list(range(n_storico))

        fig = make_subplots(rows=2, cols=2, subplot_titles=("SMA (Stress Mentale)", "ISLR (Sforzo Lavorativo)", "IITR (Impatto Termico)", "IDET (Degradazione Termica)"))
        posizioni = {"SMA": (1, 1), "ISLR": (1, 2), "IITR": (2, 1), "IDET": (2, 2)}
        colori_kpi = {"SMA": "#00E5FF", "ISLR": "#FF6A3D", "IITR": "#FFB020", "IDET": "#00F5A0"}

        for kpi_nome, (riga, colonna) in posizioni.items():
            if kpi_nome in finestra.columns:
                fig.add_trace(
                    go.Scatter(x=asse_x, y=finestra[kpi_nome], mode="lines+markers",
                               name=kpi_nome, line=dict(color=colori_kpi[kpi_nome], width=2.5),
                               marker=dict(size=6)),
                    row=riga, col=colonna
                )

        fig.update_layout(
            height=560, 
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        
        st.markdown(
            "<div class='explain-text'>"
            "<strong>💡 Interpretazione per la Discussione di Tesi:</strong> Picchi ricorrenti di ISLR o IDET "
            "in corrispondenza di cali successivi di performance costituiscono l'evidenza empirica a supporto "
            "dell'approccio <em>single-subject (N-of-1)</em>, dimostrando come il monitoraggio personalizzato intercetti "
            "i rischi prima che si trasformino in overtraining cronico[cite: 2]."
            "</div>", unsafe_allow_html=True
        )

# ==================================================================
# TAB 4 — METODOLOGIA E TRASPARENZA DEL MODELLO
# ==================================================================
with tab_metodo:
    st.markdown("### 🔬 Architettura di Ponderazione: Feature Importance")
    st.markdown("Il calcolo del rischio non si basa su stime soggettive, ma sui pesi decisionali appresi dall'algoritmo[cite: 2].")
    
    st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
    
    st.markdown(
        "<div class='explain-text'>"
        "<strong>📊 Rigore Scientifico:</strong> Questo grafico non è puramente decorativo. Rappresenta i pesi "
        "reali con cui il modello Random Forest della tesi discrimina le sessioni a rischio. L'ISLR e la qualità del sonno "
        "guidano la classificazione con oltre il 50% del peso complessivo."
        "</div>",
        unsafe_allow_html=True
    )

    if dettaglio_scores:
        st.markdown("---")
        st.markdown("### 🧬 Breakdown Analitico del Rischio Odierno")
        st.caption("Scomposizione matematica del punteggio di rischio calcolato specificamente per la tua sessione odierna.")

        nomi = list(dettaglio_scores.keys())
        valori = [dettaglio_scores[k] for k in nomi]

        fig_breakdown = go.Figure(go.Bar(
            x=valori, y=nomi, orientation="h",
            marker=dict(color=valori, colorscale=[[0, "#00F5A0"], [0.5, "#FFB020"], [1, "#FF6A3D"]])
        ))
        fig_breakdown.update_layout(
            height=320, 
            xaxis_title="Punteggio di Contributo al Rischio (0-100)",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(style_fig(fig_breakdown), use_container_width=True)
        
        st.markdown(
            "<div class='explain-text'>"
            "<strong>🔍 Trasparenza Algoritmica:</strong> Questa vista elimina l'effetto 'scatola nera' tipico del machine learning, "
            "permettendo all'atleta o al coach di identificare immediatamente <em>quale</em> fattore specifico (es. stress lavorativo o carico termico) "
            "stia spingendo verso l'alto il rischio di sovraccarico."
            "</div>",
            unsafe_allow_html=True
        )
