"""
pages/04_Centro_KPI.py (adatta il nome/percorso alla tua struttura di pagine)
--------------------------------------------------------------------------------
Dashboard unificata con i 4 KPI proprietari della tesi (SMA, ISLR, IITR, IDET),
il risk_score pesato sulla Feature Importance reale, e il grafico che mostra
quali variabili contano davvero nel tuo modello. Segue lo stesso stile/CSS
della tua Pagina 3 originale (header_block, style_fig, carica_css, kpi-card).
"""

from utils.sidebar import sidebar_comune
import streamlit as st

st.set_page_config(page_title="Centro KPI", layout="wide")

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_KPI

from kpi_ui_components import (
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
    FEATURE_IMPORTANCE_CHART_DATA
)
carica_css()
df, df_full, filtro_tempo = sidebar_comune()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

IMG_HERO_KPI = get_svg_url(SVG_KPI)

if not st.session_state.analisi_fatta:
    st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
else:
    r = st.session_state.risultati_analisi
    df_base = st.session_state.dati.copy()

    header_block(
        "Modulo 04 — Centro KPI",
        "I TUOI 4 INDICI PROPRIETARI",
        "SMA, ISLR, IITR, IDET calcolati sui parametri di oggi e confrontati col tuo storico personale.",
        IMG_HERO_KPI, "Proprietary KPI Engine"
    )

    # ============================================================
    # CALCOLO KPI DI OGGI + STORICO (per confronto percentile)
    # ============================================================
    kpi_oggi = calcola_kpi_giornalieri(r)  # r deve contenere le stesse chiavi di COL_* in kpi_engine.py
    kpi_storico = df_base.apply(calcola_kpi_giornalieri, axis=1, result_type="expand") if len(df_base) > 0 else None

    risk_score, dettaglio_scores = calcola_risk_score_pesato(
        oggi={
            "ISLR": kpi_oggi["ISLR"],
            "IDET": kpi_oggi["IDET"] if kpi_oggi["IDET"] == kpi_oggi["IDET"] else 0,  # gestisce eventuale NaN
            "Ore Sonno": r.get(COL_SONNO, r.get("ore_sonno", 0)),
            "Volume Settimanale": r.get("volume_settimanale_km", df_base[COL_DISTANZA].tail(7).sum() if COL_DISTANZA in df_base else 0),
            "Passo Medio": r.get("passo_medio", 0),
        },
        storico=kpi_storico if kpi_storico is not None else {},
    )

    status_color = "#00F5A0" if risk_score < 25 else "#FFB020" if risk_score < 60 else "#FF6A3D"
    status_text = "OTTIMALE" if risk_score < 25 else "MODERATO" if risk_score < 60 else "CRITICO"
    st.markdown(
        f"<h3 style='text-align:center; color:{status_color}; font-size:2em; letter-spacing:4px;'>"
        f"RISCHIO {status_text} — {risk_score:.0f}%</h3>", unsafe_allow_html=True
    )
    st.caption("Calcolato pesando ogni KPI secondo la sua reale Feature Importance nel tuo Random Forest, non su pesi arbitrari.")
    st.markdown("---")

    # ============================================================
    # 4 CARD KPI CON SPARKLINE
    # ============================================================
    st.markdown("### I Tuoi 4 KPI Proprietari")

    giorni_asse = df_base['Giorno'].tail(14).tolist() if kpi_storico is not None and len(df_base) >= 14 else list(range(14))

    col1, col2 = st.columns(2)
    with col1:
        colore_sma = "#00F5A0" if kpi_oggi["SMA"] < 10 else "#FFB020" if kpi_oggi["SMA"] < 15 else "#FF6A3D"
        kpi_card_sparkline("SMA — Stress Mentale Allenamento", kpi_oggi["SMA"], colore_sma,
                            kpi_storico["SMA"].tail(14) if kpi_storico is not None else [], giorni_asse)
        in_pratica("SMA alto = hai corso con poco sonno e molto stress accumulato: oggi il corpo lavora in svantaggio neurale.")

        colore_iitr = "#00F5A0" if dettaglio_scores.get("IITR", 50) < 40 else "#FFB020" if dettaglio_scores.get("IITR", 50) < 70 else "#FF6A3D"
        kpi_card_sparkline("IITR — Impatto Termico e Resistenza", kpi_oggi["IITR"], colore_iitr,
                            kpi_storico["IITR"].tail(14) if kpi_storico is not None else [], giorni_asse)
        in_pratica("IITR alto = caldo e vento hanno reso la corsa di oggi più dura del solito per ogni km percorso.")

    with col2:
        colore_islr = "#00F5A0" if kpi_oggi["ISLR"] < 4.5 else "#FFB020" if kpi_oggi["ISLR"] < 6.3 else "#FF6A3D"
        kpi_card_sparkline("ISLR — Sforzo Lavorativo Residuo", kpi_oggi["ISLR"], colore_islr,
                            kpi_storico["ISLR"].tail(14) if kpi_storico is not None else [], giorni_asse)
        in_pratica("ISLR è il KPI più predittivo del tuo modello (31,5%): sopra 6,3 il tuo Random Forest classifica le sessioni come overload nel 50%+ dei casi.")
        if kpi_oggi["ISLR"] >= 6.3:
            azione_consigliata("Oggi il tuo carico lavorativo sta 'mangiando' risorse che servirebbero alla corsa. Valuta una sessione più corta o rigenerativa invece che qualitativa.")

        colore_idet = "#00F5A0" if dettaglio_scores.get("IDET", 50) < 40 else "#FFB020" if dettaglio_scores.get("IDET", 50) < 70 else "#FF6A3D"
        kpi_card_sparkline("IDET — Degradazione Termica", kpi_oggi["IDET"] if kpi_oggi["IDET"] == kpi_oggi["IDET"] else 0, colore_idet,
                            kpi_storico["IDET"].tail(14) if kpi_storico is not None else [], giorni_asse)
        in_pratica("IDET alto = il caldo sta facendo salire i tuoi battiti più di quanto la velocità giustifichi (deriva cardiaca): non è un calo di forma, è il clima.")

    verdetto_box(
        100 - risk_score, soglie=(40, 75),
        testo_basso=f"Rischio elevato ({risk_score:.0f}%): la combinazione dei tuoi 4 KPI oggi indica una condizione di vulnerabilità — valuta di ridurre intensità o rimandare.",
        testo_medio=f"Rischio moderato ({risk_score:.0f}%): allenati con attenzione, monitorando in particolare l'ISLR se lavori molto in questo periodo.",
        testo_alto=f"Rischio basso ({risk_score:.0f}%): condizioni favorevoli su tutti e 4 gli indici.",
        spiegazione="Ogni KPI viene confrontato con il tuo storico personale, non con soglie generiche — coerente con l'approccio single-subject della tua ricerca."
    )

    st.markdown("---")

    # ============================================================
    # FEATURE IMPORTANCE — LA "PROVA SCIENTIFICA" DIETRO I PESI
    # ============================================================
    st.markdown("### Cosa Pesa Davvero: la Feature Importance del tuo Random Forest")
    st.plotly_chart(feature_importance_chart(style_fig), use_container_width=True)
    st.markdown(
        "<div class='explain-text'><strong>Spiegazione Grafico:</strong> "
        "questo non è un grafico decorativo: sono i pesi reali con cui il tuo modello "
        "predice l'overload, estratti dal Random Forest della tesi. Il risk_score qui sopra "
        "usa esattamente questi numeri come pesi, invece di valori arbitrari.</div>",
        unsafe_allow_html=True
    )
