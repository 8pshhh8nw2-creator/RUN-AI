"""
pages/04_Centro_KPI.py
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET),
il risk_score pesato sulla Feature Importance reale, e il grafico che mostra
quali variabili contano davvero nel tuo modello.
"""

import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Centro KPI", layout="wide")
carica_css()

# ==================================================================
# INIZIALIZZAZIONE STATO — unica fonte di verità, nessuna duplicazione
# ==================================================================
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
st.session_state.setdefault('analisi_fatta', False)
st.session_state.setdefault('risultati_analisi', {})

# ==================================================================
# SIDEBAR — fallback sicuro, nessun ricalcolo superfluo
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
    "I TUOI 4 INDICI PROPRIETARI",
    "SMA, ISLR, IITR, IDET calcolati sui parametri di oggi e confrontati col tuo storico personale.",
    IMG_HERO_KPI, "Proprietary KPI Engine"
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA' per sbloccare i tuoi KPI di oggi.")
    st.stop()


# ==================================================================
# FUNZIONI DI SUPPORTO
# ==================================================================
def _colore_e_stato(valore, soglia_verde, soglia_gialla):
    """Restituisce (colore, etichetta) di stato per un valore rispetto a due soglie crescenti."""
    if valore is None or pd.isna(valore):
        return "#566178", "N/D"
    if valore < soglia_verde:
        return "#00F5A0", "OTTIMALE"
    elif valore < soglia_gialla:
        return "#FFB020", "MODERATO"
    return "#FF6A3D", "ATTENZIONE"


def _delta_vs_storico(valore_oggi, serie_storica):
    """Calcola la variazione rispetto all'ultima sessione storica disponibile. None se non calcolabile."""
    if serie_storica is None or len(serie_storica) == 0 or pd.isna(valore_oggi):
        return None
    ultimo_storico = serie_storica.iloc[-1]
    if pd.isna(ultimo_storico):
        return None
    return valore_oggi - ultimo_storico


def _badge_delta(delta, positivo_e_meglio=False):
    """Renderizza una piccola etichetta di trend (▲/▼) accanto al valore del KPI."""
    if delta is None:
        return ""
    peggiora = (delta > 0) if not positivo_e_meglio else (delta < 0)
    freccia = "▲" if delta > 0 else "▼" if delta < 0 else "→"
    colore = "#FF6A3D" if peggiora and abs(delta) > 0.01 else "#00F5A0" if abs(delta) > 0.01 else "#566178"
    return f"<span style='color:{colore}; font-size:0.8em; margin-left:6px;'>{freccia} {abs(delta):.1f} vs ultima sessione</span>"


# ==================================================================
# CALCOLO KPI DI OGGI + STORICO (per confronto e percentile)
# ==================================================================
r = st.session_state.risultati_analisi
df_base = st.session_state.dati.copy()

kpi_oggi = calcola_kpi_giornalieri(r)

kpi_storico = None
if len(df_base) > 0:
    try:
        kpi_storico = df_base.apply(calcola_kpi_giornalieri, axis=1, result_type="expand")
    except Exception:
        st.info("Alcune sessioni storiche non sono calcolabili: i confronti col passato saranno parziali.")
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
    st.info("Non è stato possibile calcolare il risk score pesato con i dati disponibili: mostrato un valore neutro (50%).")

status_color = "#00F5A0" if risk_score < 25 else "#FFB020" if risk_score < 60 else "#FF6A3D"
status_text = "OTTIMALE" if risk_score < 25 else "MODERATO" if risk_score < 60 else "CRITICO"

st.markdown(
    f"<h3 style='text-align:center; color:{status_color}; font-size:2em; letter-spacing:4px;'>"
    f"RISCHIO {status_text} — {risk_score:.0f}%</h3>", unsafe_allow_html=True
)
st.caption("Calcolato pesando ogni KPI secondo la sua reale Feature Importance nel tuo Random Forest, non su pesi arbitrari.")
st.markdown("---")

tab_oggi, tab_storico, tab_metodo = st.tabs(["📍 Oggi", "📈 Andamento Storico", "🔬 Metodologia"])

# ==================================================================
# TAB 1 — SITUAZIONE DI OGGI
# ==================================================================
with tab_oggi:
    st.markdown("### I Tuoi 4 KPI Proprietari")

    giorni_asse = df_base['Giorno'].tail(14).tolist() if (kpi_storico is not None and 'Giorno' in df_base.columns and len(df_base) >= 14) else list(range(14))

    col1, col2 = st.columns(2)
    with col1:
        colore_sma, _ = _colore_e_stato(kpi_oggi["SMA"], 10, 15)
        delta_sma = _delta_vs_storico(kpi_oggi["SMA"], kpi_storico["SMA"] if kpi_storico is not None else None)
        kpi_card_sparkline("SMA — Stress Mentale Allenamento", kpi_oggi["SMA"], colore_sma,
                           kpi_storico["SMA"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_sma), unsafe_allow_html=True)
        in_pratica("SMA alto = hai corso con poco sonno e molto stress accumulato: oggi il corpo lavora in svantaggio neurale.")

        colore_iitr, _ = _colore_e_stato(dettaglio_scores.get("IITR", 50), 40, 70)
        delta_iitr = _delta_vs_storico(kpi_oggi["IITR"], kpi_storico["IITR"] if kpi_storico is not None else None)
        kpi_card_sparkline("IITR — Impatto Termico e Resistenza", kpi_oggi["IITR"], colore_iitr,
                           kpi_storico["IITR"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_iitr), unsafe_allow_html=True)
        in_pratica("IITR alto = caldo e vento hanno reso la corsa di oggi più dura del solito per ogni km percorso.")

    with col2:
        colore_islr, _ = _colore_e_stato(kpi_oggi["ISLR"], 4.5, 6.3)
        delta_islr = _delta_vs_storico(kpi_oggi["ISLR"], kpi_storico["ISLR"] if kpi_storico is not None else None)
        kpi_card_sparkline("ISLR — Sforzo Lavorativo Residuo", kpi_oggi["ISLR"], colore_islr,
                           kpi_storico["ISLR"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_islr), unsafe_allow_html=True)
        in_pratica("ISLR è il KPI più predittivo del tuo modello (31,5%): sopra 6,3 il tuo Random Forest classifica le sessioni come overload nel 50%+ dei casi.")
        if kpi_oggi["ISLR"] >= 6.3:
            azione_consigliata("Oggi il tuo carico lavorativo sta 'mangiando' risorse che servirebbero alla corsa. Valuta una sessione più corta o rigenerativa invece che qualitativa.")

        val_idet = kpi_oggi["IDET"] if pd.notna(kpi_oggi["IDET"]) else 0.0
        colore_idet, _ = _colore_e_stato(dettaglio_scores.get("IDET", 50), 40, 70)
        delta_idet = _delta_vs_storico(val_idet, kpi_storico["IDET"] if kpi_storico is not None else None)
        kpi_card_sparkline("IDET — Degradazione Termica", val_idet, colore_idet,
                           kpi_storico["IDET"].tail(14).tolist() if kpi_storico is not None else [], giorni_asse)
        st.markdown(_badge_delta(delta_idet), unsafe_allow_html=True)
        in_pratica("IDET alto = il caldo sta facendo salire i tuoi battiti più di quanto la velocità giustifichi (deriva cardiaca): non è un calo di forma, è il clima.")

    st.markdown("---")
    verdetto_box(
        100 - risk_score, soglie=(40, 75),
        testo_basso=f"Rischio elevato ({risk_score:.0f}%): la combinazione dei tuoi 4 KPI oggi indica una condizione di vulnerabilità — valuta di ridurre intensità o rimandare.",
        testo_medio=f"Rischio moderato ({risk_score:.0f}%): allenati con attenzione, monitorando in particolare l'ISLR se lavori molto in questo periodo.",
        testo_alto=f"Rischio basso ({risk_score:.0f}%): condizioni favorevoli su tutti e 4 gli indici.",
        spiegazione="Ogni KPI viene confrontato con il tuo storico personale, non con soglie generiche — coerente con l'approccio single-subject della tua ricerca."
    )

# ==================================================================
# TAB 2 — ANDAMENTO STORICO REALE (non solo sparkline nelle card)
# ==================================================================
with tab_storico:
    if kpi_storico is None or len(kpi_storico) < 3:
        st.info("Servono almeno 3 sessioni storiche per mostrare un andamento significativo. Continua a registrare le tue corse.")
    else:
        st.markdown("### Andamento dei 4 KPI nel Tempo")
        st.caption("Ogni pannello mantiene la propria scala naturale: i 4 indici non sono comparabili in valore assoluto, solo in tendenza.")

        n_storico = min(30, len(kpi_storico))
        finestra = kpi_storico.tail(n_storico).reset_index(drop=True)
        asse_x = df_base['Giorno'].tail(n_storico).tolist() if 'Giorno' in df_base.columns else list(range(n_storico))

        fig = make_subplots(rows=2, cols=2, subplot_titles=("SMA", "ISLR", "IITR", "IDET"))
        posizioni = {"SMA": (1, 1), "ISLR": (1, 2), "IITR": (2, 1), "IDET": (2, 2)}
        colori_kpi = {"SMA": "#00E5FF", "ISLR": "#FF6A3D", "IITR": "#FFB020", "IDET": "#00F5A0"}

        for kpi_nome, (riga, colonna) in posizioni.items():
            if kpi_nome in finestra.columns:
                fig.add_trace(
                    go.Scatter(x=asse_x, y=finestra[kpi_nome], mode="lines+markers",
                               name=kpi_nome, line=dict(color=colori_kpi[kpi_nome], width=2)),
                    row=riga, col=colonna
                )

        fig.update_layout(height=520, showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown(
            "<div class='explain-text'><strong>Lettura del grafico:</strong> picchi ricorrenti di ISLR o IDET "
            "in corrispondenza di cali successivi di performance sono l'evidenza empirica che sostiene "
            "l'approccio single-subject (N-of-1) della tesi.</div>", unsafe_allow_html=True
        )

# ==================================================================
# TAB 3 — METODOLOGIA E TRASPARENZA DEL MODELLO
# ==================================================================
with tab_metodo:
    st.markdown("### Cosa Pesa Davvero: la Feature Importance del tuo Random Forest")
    st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
    st.markdown(
        "<div class='explain-text'><strong>Spiegazione Grafico:</strong> "
        "questo non è un grafico decorativo: sono i pesi reali con cui il tuo modello "
        "predice l'overload, estratti dal Random Forest della tesi. Il risk_score qui sopra "
        "usa esattamente questi numeri come pesi, invece di valori arbitrari.</div>",
        unsafe_allow_html=True
    )

    if dettaglio_scores:
        st.markdown("---")
        st.markdown("### Come si Compone il Rischio di Oggi")
        st.caption("Contributo di ciascuna variabile al risk_score complessivo, calcolato sui dati che hai inserito oggi.")

        nomi = list(dettaglio_scores.keys())
        valori = [dettaglio_scores[k] for k in nomi]

        fig_breakdown = go.Figure(go.Bar(
            x=valori, y=nomi, orientation="h",
            marker=dict(color=valori, colorscale=[[0, "#00F5A0"], [0.5, "#FFB020"], [1, "#FF6A3D"]])
        ))
        fig_breakdown.update_layout(height=320, xaxis_title="Punteggio di rischio (0-100)")
        st.plotly_chart(style_fig(fig_breakdown), use_container_width=True)
        st.markdown(
            "<div class='explain-text'><strong>Nota:</strong> questa scomposizione rende trasparente "
            "il calcolo del risk_score aggregato mostrato in cima alla pagina, invece di lasciarlo come "
            "\"scatola nera\".</div>", unsafe_allow_html=True
        )
