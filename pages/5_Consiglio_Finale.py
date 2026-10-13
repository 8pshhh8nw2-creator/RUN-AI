import math
import html as _html
import re as _re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.components import header_block
from utils.data import genera_dati
from utils.kpi_engine import calcola_kpi_giornalieri
from utils.ml_engine import (
    analisi_completa_oggi,
    get_bundle_classificazione,
    get_bundle_cluster,
    get_bundle_regressione,
    punteggio_rischio_combinato,
)
from utils.sidebar import sidebar_comune
from utils.style import carica_css

st.set_page_config(page_title="Consiglio Finale", layout="wide")
carica_css()


def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_text(value, default="n/d"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value)


def _plain(value):
    if value is None:
        return ""
    return _re.sub(r'<[^>]+>', '', str(value)).replace('&nbsp;', ' ').strip()


def md(text):
    clean = "\n".join(line.strip() for line in text.split("\n"))
    st.markdown(clean, unsafe_allow_html=True)


if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
    st.session_state.dati_simulati = True
else:
    st.session_state.dati_simulati = st.session_state.get('dati_simulati', False)

if 'analisi_fatta' not in st.session_state:
    st.session_state.analisi_fatta = False
if 'risultati_analisi' not in st.session_state:
    st.session_state.risultati_analisi = {}

sidebar_result = sidebar_comune()
if sidebar_result and isinstance(sidebar_result, tuple) and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati.copy()
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

header_block(
    "Modulo 05 — Sintesi Operativa",
    "CONSIGLIO FINALE E REPORT GIORNALIERO",
    "Il riassunto di oggi: quanto puoi spingere, perché, e cosa fare passo dopo passo.",
    None,
    "Executive Summary",
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA' per sbloccare i consigli personalizzati.")
else:
    r = st.session_state.risultati_analisi
    df_base = st.session_state.dati.copy()

    for col in ['Ore Sonno', 'Stress Lavoro', 'RPE', 'Distanza', 'FC Media', 'Temperatura', 'Vento', 'Velocità']:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce')

    date_col = next((c for c in df_base.columns if c.lower() in ['data', 'date', 'giorno', 'time']), None)

    if st.session_state.get('dati_simulati', False):
        st.info("I dati attualmente mostrati sono dimostrativi: usa dataset reali per decisioni operative.")

    try:
        class_bundle = get_bundle_classificazione(df_base)
        cluster_bundle = get_bundle_cluster(df_base)
        reg_bundle = get_bundle_regressione(df_base)
    except Exception as exc:
        st.error(f"Errore durante il caricamento dei modelli: {exc}")
        class_bundle = {}
        cluster_bundle = {}
        reg_bundle = {}

    try:
        analisi_ml = analisi_completa_oggi(
            df_base,
            distanza=safe_float(r.get('distanza_oggi', 10.0)),
            ore_sonno=safe_float(r.get('ore_sonno', 7.5)),
            stress=safe_float(r.get('stress_lavoro', 5.0)),
            rpe=safe_float(r.get('rpe_previsto', 5.0)),
        )
    except Exception as exc:
        st.error(f"Errore nel calcolo dell'analisi odierna: {exc}")
        analisi_ml = {
            'componenti': {
                'random_forest': 50.0,
                'logistic': 50.0,
                'cluster': 50.0,
                'trend_fisiologico': 50.0,
            },
            'fattore_principale_rf': 'dati insufficienti',
        }

    SOGLIA_BASSA = 40
    SOGLIA_MEDIA = 70

    ore_sonno = safe_float(r.get('ore_sonno', 7.5))
    stress_lavoro = safe_float(r.get('stress_lavoro', 5.0))
    rpe_previsto = safe_float(r.get('rpe_previsto', 5.0))

    risk_score_euristico = min(
        100.0,
        (40 if ore_sonno < 6 else 25 if ore_sonno < 6.5 else 10)
        + (35 if stress_lavoro >= 8 else 20 if stress_lavoro >= 6 else 5)
        + (30 if rpe_previsto >= 8 else 15 if rpe_previsto >= 6 else 5)
        + (20 if ore_sonno < 6.5 and stress_lavoro >= 7 and rpe_previsto >= 7 else 0),
    )

    componenti = analisi_ml.get('componenti', {})
    if componenti:
        risk_index = float(punteggio_rischio_combinato(componenti, risk_score_euristico))
    else:
        risk_index = float(risk_score_euristico)

    rischio_ml = float(componenti.get('random_forest', 0.0)) if componenti else float(risk_score_euristico)
    fattore_ml = safe_text(analisi_ml.get('fattore_principale_rf', 'n/d'))

    if ore_sonno >= 7.5:
        recovery_score = 100.0
    else:
        recovery_score = max(0.0, 100.0 - (7.5 - ore_sonno) * 16.0)

    sma = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0.0

    try:
        kpi_oggi = calcola_kpi_giornalieri(r)
    except Exception as exc:
        st.warning(f"KPI personalizzati non disponibili: {exc}")
        kpi_oggi = {'SMA': sma, 'ISLR': None, 'IITR': None, 'IDET': None}

    islr_val = kpi_oggi.get('ISLR')
    iitr_val = kpi_oggi.get('IITR')
    idet_val = kpi_oggi.get('IDET')

    distanza_target = safe_float(r.get('distanza_oggi', 10.0), 10.0)
    if risk_index < SOGLIA_BASSA:
        tit = "Consigliato"
        col = "#30D158"
        liv = "basso"
        distanza_consigliata = distanza_target
    elif risk_index < SOGLIA_MEDIA:
        tit = "Recupero attivo"
        col = "#FF9F0A"
        liv = "medio"
        distanza_consigliata = max(0.0, distanza_target * 0.6)
    else:
        tit = "Riposo consigliato"
        col = "#FF453A"
        liv = "alto"
        distanza_consigliata = 0.0

    cadenza_target = "170-180 spm" if liv != 'alto' else "165-172 spm (ritmo molto rilassato)"
    zona_consigliata = "Zona 2-3 (aerobico)" if liv == 'basso' else "Zona 1-2 (sforzo basso)" if liv == 'medio' else "Mobilità / camminata"

    hero_messages = {
        'Consigliato': "Il corpo appare sufficientemente recuperato: puoi seguire la seduta prevista con attenzione alle sensazioni.",
        'Recupero attivo': "Valuta di ridurre intensità e volume; la prudenza oggi dà più valore al recupero.",
        'Riposo consigliato': "Il quadro suggerisce di evitare sforzi intensi oggi: preferibile un recupero attivo e una seduta leggera.",
    }
    hero_msg = hero_messages.get(tit, "") + f" Fattore rilevante in questo scenario: <strong>{_html.escape(fattore_ml)}</strong>."

    scarto = abs((rischio_ml or 0.0) - risk_score_euristico)
    nota_coerenza = "I segnali sono globalmente coerenti." if scarto < 15 else "I segnali non sono completamente allineati: preferisci prudenza e monitoraggio."

    df_adv = df_base.copy()
    if date_col:
        df_adv['Data_Chart'] = pd.to_datetime(df_adv[date_col], errors='coerce')
        dates_artificial = False
    else:
        df_adv['Data_Chart'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_adv))
        dates_artificial = True
    df_adv = df_adv.sort_values('Data_Chart').dropna(subset=['Data_Chart']).reset_index(drop=True)

    PANEL_BG = '#0D1117'
    PANEL_BD = '#1E2633'
    PANEL_BD_H = '#2A3546'
    TXT_PRIMARY = '#F8F9FA'
    TXT_SECONDARY = '#8B949E'
    TXT_TERTIARY = '#485363'
    C_SONNO = '#2E90FF'
    C_STRESS = '#FF453A'
    C_RPE = '#30D158'
    C_AMBRA = '#FF9F0A'
    C_VIOLA = '#BF5AF2'
    C_NEUTRO = '#1F2733'
    C_ARANCIO = '#FF6B35'

    def rgba(hex_color, alpha):
        try:
            h = hex_color.lstrip('#')
            if len(h) == 3:
                h = ''.join(ch * 2 for ch in h)
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'
        except Exception:
            return hex_color

    md(f"""
    <style>
    .panel {{ background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 18px rgba(0,0,0,0.16); }}
    .panel-flush {{ padding: 0; overflow: hidden; }}
    .panel-body {{ padding: 16px 20px; }}
    .chart-top-rule {{ height: 4px; width: 100%; background: {C_NEUTRO}; }}
    .eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {TXT_TERTIARY}; margin: 0 0 8px 0; font-weight: 700; }}
    .section-head {{ display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; margin: 12px 0 20px; }}
    .section-head h3 {{ margin: 0; color: {TXT_PRIMARY}; font-size: 1.4rem; font-weight: 700; text-transform: uppercase; letter-spacing: .02em; }}
    .section-head .sub {{ color: {TXT_SECONDARY}; font-size: .9rem; margin-top: 4px; }}
    .section-head .rule {{ flex: 1; height: 1px; background: linear-gradient(90deg, {PANEL_BD} 0%, transparent 100%); }}
    .status-badge {{ display: inline-flex; align-items: center; }}
    .status-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 9px; }}
    .value-pill {{ display: inline-flex; align-items: baseline; border-radius: 8px; padding: 4px 12px; }}
    .value-pill .unit {{ font-size: .45em; color: {TXT_SECONDARY}; margin-left: 4px; text-transform: uppercase; }}
    .mini-caption {{ color: {TXT_SECONDARY}; font-size: .8rem; line-height: 1.5; }}
    .split-sheet {{ padding: 0; overflow: hidden; border-radius: 12px; }}
    .split-head, .split-row {{ display: grid; grid-template-columns: 1.2fr 1fr 1fr 1.6fr; align-items: center; gap: 10px; }}
    .split-head {{ background: #161C24; padding: 12px 24px; border-bottom: 1px solid {PANEL_BD}; font-family: 'JetBrains Mono', monospace; font-size: .65rem; letter-spacing: .12em; text-transform: uppercase; color: {TXT_TERTIARY}; font-weight: 700; }}
    .split-row {{ padding: 18px 24px; border-bottom: 1px solid {PANEL_BD}; }}
    .split-row:last-child {{ border-bottom: none; }}
    .split-row .sr-label {{ font-family: 'Oswald', sans-serif; font-size: 1rem; letter-spacing: .02em; text-transform: uppercase; color: {TXT_PRIMARY}; font-weight: 500; }}
    .split-row .sr-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; }}
    .split-row .sr-ref {{ font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: {TXT_SECONDARY}; }}
    .split-row .sr-note {{ font-family: 'Inter', sans-serif; font-size: .86rem; color: {TXT_SECONDARY}; line-height: 1.5; border-left: 1px solid {PANEL_BD}; padding-left: 12px; }}
    .zone-card {{ background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 14px; overflow: hidden; height: 100%; box-shadow: 0 4px 18px rgba(0,0,0,0.14); }}
    .zone-card .zone-card-top {{ height: 4px; width: 100%; background: var(--zc); }}
    .zone-card .zone-card-body {{ padding: 20px; }}
    .zone-card .zone-card-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }}
    .zone-card .zone-badge {{ width: 38px; height: 38px; border-radius: 50%; border: 2px solid var(--zc); color: var(--zc); background: rgba(255,255,255,0.04); display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
    .zone-card .zone-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: .68rem; line-height: 1; letter-spacing: .1em; text-transform: uppercase; color: {TXT_TERTIARY}; font-weight: 700; }}
    .zone-card .zone-title {{ font-family: 'Oswald', sans-serif; font-size: 1.05rem; letter-spacing: .02em; text-transform: uppercase; color: {TXT_PRIMARY}; }}
    .zone-card .zone-desc {{ color: {TXT_SECONDARY}; font-size: .88rem; line-height: 1.6; margin: 0; }}
    .hero-panel {{ background: linear-gradient(135deg, {PANEL_BG} 0%, #131A24 100%); border: 1px solid {PANEL_BD}; border-radius: 18px; padding: 28px 30px; position: relative; overflow: hidden; }}
    .hero-panel::after {{ content: ""; position: absolute; top: -40%; right: -8%; width: 360px; height: 360px; background: radial-gradient(circle, var(--hero-color) 0%, transparent 70%); opacity: .14; pointer-events: none; }}
    .hero-kicker {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; font-weight:700; margin:0 0 10px 0; }}
    .hero-title {{ font-family:'Oswald',sans-serif; font-weight:700; font-size:2.2rem; letter-spacing:.02em; margin:0; text-transform:uppercase; line-height:1.08; }}
    .hero-msg {{ color: {TXT_SECONDARY}; line-height: 1.6; margin: 12px 0 0; max-width: 620px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 18px; }}
    .kpi {{ background: rgba(255,255,255,0.02); border: 1px solid {PANEL_BD}; border-radius: 10px; padding: 14px; }}
    .kpi .val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; color: {TXT_PRIMARY}; font-weight: 700; }}
    .kpi .label {{ color: {TXT_SECONDARY}; font-size: .75rem; margin-top: 6px; }}
    .coach-block {{ margin-bottom: 18px; background: rgba(255,255,255,0.02); padding: 16px; border-left: 3px solid var(--block-color); border-radius: 8px; }}
    .coach-block:last-child {{ margin-bottom: 0; }}
    .coach-block .label {{ font-family: 'Oswald', sans-serif; font-size: .9rem; letter-spacing: .05em; text-transform: uppercase; color: var(--block-color); font-weight: 600; margin-bottom: 10px; }}
    .coach-block ul {{ margin: 0; padding-left: 18px; }}
    .coach-block li {{ color: {TXT_SECONDARY}; line-height: 1.6; margin-bottom: 8px; }}
    .model-glyph {{ display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--gc); color: var(--gc); font-family: 'JetBrains Mono', monospace; font-size: .78rem; letter-spacing: .02em; font-weight: 700; }}
    @media (max-width: 900px) {{ .split-head, .split-row {{ grid-template-columns: 1fr 1fr; }} .split-row .sr-note {{ grid-column: 1 / -1; border-left: none; border-top: 1px solid {PANEL_BD}; padding-left: 0; padding-top: 8px; margin-top: 8px; }} .card-grid {{ grid-template-columns: 1fr; }} }}
    </style>
    """)

    def section_head(eyebrow, title, sub=None):
        sub_html = f"<div class='sub'>{_html.escape(sub)}</div>" if sub else ""
        md(f"""
        <div class='section-head'>
            <div>
                <p class='eyebrow'>{_html.escape(eyebrow)}</p>
                <h3>{_html.escape(title)}</h3>
                {sub_html}
            </div>
            <div class='rule'></div>
        </div>
        """)

    def gauge_svg(score, color, size=210, label='INDICE', show_pct=False):
        stroke_w = max(12, int(size * 0.07))
        radius = (size / 2) - stroke_w - 6
        cx = size / 2
        cy = size / 2
        circumference = 2 * math.pi * radius
        offset = circumference * (1 - min(max(score, 0.0), 100.0) / 100.0)
        text_value = f"{int(score)}%" if show_pct else str(int(score))
        return f"""
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
            <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{C_NEUTRO}" stroke-width="{stroke_w}" />
            <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_w}"
                stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                transform="rotate(-90 {cx} {cy})" />
            <text x="{cx}" y="{cy - 4}" text-anchor="middle" font-family="Oswald, sans-serif" font-size="{int(size * 0.22)}" font-weight="700" fill="{TXT_PRIMARY}">{text_value}</text>
            <text x="{cx}" y="{cy + int(size * 0.12)}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="{int(size * 0.045)}" letter-spacing="2" fill="{TXT_SECONDARY}">{label}</text>
        </svg>
        """

    def telemetry_svg(score):
        w, h = 800, 100
        bars = 60
        gap = 2
        bar_w = (w - (bars * gap)) / bars
        pieces = []
        for i in range(bars):
            x = i * (bar_w + gap)
            progression = i / bars
            bar_h = 15 + (progression ** 2.5) * (h - 15)
            y = h - bar_h
            pct = (i / bars) * 100
            active = pct <= score
            color = C_RPE if pct < SOGLIA_BASSA else C_AMBRA if pct < SOGLIA_MEDIA else C_STRESS
            fill = color if active else C_NEUTRO
            opacity = '1.0' if active else '0.3'
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{fill}" opacity="{opacity}" rx="2"/>')
        marker_x = (score / 100.0) * w
        return f"""
        <svg viewBox="0 -28 {w} {h + 53}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; display:block;">
            <line x1="0" y1="{h}" x2="{w}" y2="{h}" stroke="{TXT_TERTIARY}" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
            {''.join(pieces)}
            <g transform="translate({marker_x}, 0)">
                <line x1="0" y1="0" x2="0" y2="{h + 10}" stroke="{TXT_PRIMARY}" stroke-width="2" />
                <polygon points="-6,{h + 10} 6,{h + 10} 0,{h + 18}" fill="{TXT_PRIMARY}" />
                <rect x="-24" y="-20" width="48" height="20" rx="4" fill="{TXT_PRIMARY}" />
                <text x="0" y="-6" font-family="JetBrains Mono, monospace" font-size="12" font-weight="bold" fill="{PANEL_BG}" text-anchor="middle">{int(score)}%</text>
            </g>
            <text x="0" y="{h + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_RPE}" opacity="0.8">OK (0-{SOGLIA_BASSA-1})</text>
            <text x="{w/2}" y="{h + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_AMBRA}" text-anchor="middle" opacity="0.8">ATTENZIONE ({SOGLIA_BASSA}-{SOGLIA_MEDIA-1})</text>
            <text x="{w}" y="{h + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_STRESS}" text-anchor="end" opacity="0.8">PERICOLO ({SOGLIA_MEDIA}-100)</text>
        </svg>
        """

    def embed_svg(svg_code, height=220, extra_padding=6):
        st.components.v1.html(
            f"""
            <html>
            <head>
            <style>
                html, body {{ margin:0; padding:0; background:transparent; height:100%; overflow:visible; }}
                .svg-wrap {{ display:flex; align-items:center; justify-content:center; width:100%; height:100%; background:transparent; padding:{extra_padding}px; box-sizing:border-box; overflow:visible; }}
                .svg-wrap svg {{ max-width:100%; max-height:100%; width:auto; height:auto; overflow:visible; }}
            </style>
            </head>
            <body><div class="svg-wrap">{svg_code}</div></body>
            </html>
            """,
            height=height + 16,
            scrolling=False,
        )

    gauge = gauge_svg(risk_index, col, size=220, label='INDICE', show_pct=True)
    telem = telemetry_svg(risk_index)

    md(f"""
    <div class='hero-panel' style='--hero-color:{col};'>
      <p class='hero-kicker'>Il verdetto di oggi</p>
      <h1 class='hero-title' style='color:{col};'>{_html.escape(tit)}</h1>
      <p class='hero-msg'>{hero_msg}</p>
      <div class='card-grid' style='margin-top:22px;'>
        <div class='kpi'><div class='val' style='color:{TXT_PRIMARY};'>{distanza_consigliata:.1f}<span style='font-size:.5em; color:{TXT_SECONDARY}; margin-left:4px; text-transform:uppercase;'>km</span></div><div class='label'>Distanza consigliata</div></div>
        <div class='kpi'><div class='val' style='color:{C_SONNO};'>{recovery_score:.0f}<span style='font-size:.5em; color:{TXT_SECONDARY}; margin-left:4px; text-transform:uppercase;'>%</span></div><div class='label'>Recovery score</div></div>
        <div class='kpi'><div class='val' style='color:{C_AMBRA};'>{sma:.1f}</div><div class='label'>SMA proxy</div></div>
      </div>
    </div>
    """)

    md(f"""
    <div class='mini-caption' style='margin: 12px 0 20px 0;'>
      Indice totale 0-100: <strong style='color:{TXT_PRIMARY}'>{risk_index:.0f}</strong> ·
      Stima RF: <strong style='color:{TXT_PRIMARY}'>{rischio_ml:.0f}%</strong> ·
      Euristica: <strong style='color:{TXT_PRIMARY}'>{risk_score_euristico:.0f}</strong> ·
      {nota_coerenza}
    </div>
    """)

    c1, c2 = st.columns([1, 2.2])
    with c1:
        embed_svg(gauge, 240, 8)
    with c2:
        md("<p class='eyebrow' style='margin-top: 8px;'>Dettaglio soglie di rischio</p>")
        md("<p class='mini-caption'>Più la barra è a destra, più il rischio di andare oltre i limiti aumenta. L'indice è un punteggio sintetico, non una probabilità clinica certificata.</p>")
        embed_svg(telem, 180, 10)

    section_head("Numeri chiave", "I 6 indicatori di oggi", "Ogni riga mostra il numero e il suo significato pratico del giorno.")
    md(f"""
    <div class='panel split-sheet'>
      <div class='split-head'><span>Indicatore</span><span>Valore</span><span>Riferimento</span><span>Cosa significa</span></div>
      <div class='split-row'>
        <div class='sr-label'>Distanza Target</div>
        <div class='sr-value'><span class='value-pill' style='color:{TXT_PRIMARY}; background:{rgba(TXT_PRIMARY, 0.08)};'>{distanza_consigliata:.1f}<span class='unit'>km</span></span></div>
        <div class='sr-ref'>Piano originale: {distanza_target:.1f} km</div>
        <div class='sr-note'>Riduzione % applicata dal sistema in base all'indice oggi. Se {distanza_consigliata:.1f} km = 0, si raccomanda riposo.</div>
      </div>
      <div class='split-row'>
        <div class='sr-label'>Recovery Score</div>
        <div class='sr-value'><span class='value-pill' style='color:{C_SONNO}; background:{rgba(C_SONNO,0.08)};'>{recovery_score:.0f}<span class='unit'>%</span></span></div>
        <div class='sr-ref'>Base: {ore_sonno:.1f}h di sonno</div>
        <div class='sr-note'>Valore parziale: considera il sonno e non sostituisce una valutazione completa del recupero.</div>
      </div>
      <div class='split-row'>
        <div class='sr-label'>Carico mentale (SMA)</div>
        <div class='sr-value'><span class='value-pill' style='color:{C_AMBRA}; background:{rgba(C_AMBRA,0.08)};'>{sma:.1f}</span></div>
        <div class='sr-ref'>(Stress × RPE) / Sonno</div>
        <div class='sr-note'>Proxy non validata, utile come indicatore di carico complessivo ma non come misura clinica ufficiale.</div>
      </div>
      <div class='split-row'>
        <div class='sr-label'>Sforzo lavorativo (ISLR)</div>
        <div class='sr-value'><span class='value-pill' style='color:{C_STRESS}; background:{rgba(C_STRESS,0.08)};'>{f'{islr_val:.1f}' if islr_val is not None and not pd.isna(islr_val) else 'N/D'}</span></div>
        <div class='sr-ref'>(Ore lavoro × stress) / distanza</div>
        <div class='sr-note'>Proxy: segnala se il lavoro sta sottraendo troppo energia alla seduta. Le soglie vanno validate.</div>
      </div>
      <div class='split-row'>
        <div class='sr-label'>Impatto termico (IITR)</div>
        <div class='sr-value'><span class='value-pill' style='color:{TXT_PRIMARY}; background:{rgba(TXT_PRIMARY,0.08)};'>{f'{iitr_val:.1f}' if iitr_val is not None and not pd.isna(iitr_val) else 'N/D'}</span></div>
        <div class='sr-ref'>(Temperatura × vento) / distanza</div>
        <div class='sr-note'>Proxy meteorologico: aiuta a leggere se le condizioni del giorno rendono la seduta più dura, senza attribuire causalità certa.</div>
      </div>
      <div class='split-row'>
        <div class='sr-label'>Degradazione termica (IDET)</div>
        <div class='sr-value'><span class='value-pill' style='color:{C_VIOLA}; background:{rgba(C_VIOLA,0.08)};'>{f'{idet_val:.1f}' if idet_val is not None and not pd.isna(idet_val) else 'N/D'}</span></div>
        <div class='sr-ref'>(FC media × temperatura) / velocità</div>
        <div class='sr-note'>Proxy: non distingue automaticamente la causa dell'aumento della frequenza cardiaca; va interpretata con cautela.</div>
      </div>
    </div>
    """)

    section_head("Coach personalizzato", "Il tuo protocollo passo dopo passo", "Cosa fare prima, durante e dopo la corsa in base ai segnali di oggi.")

    if liv == 'basso':
        pacing = "Puoi tenere il tuo ritmo normale; se vuoi rallentare, aggiungi 10-15 secondi al km rispetto al tuo passo abituale."
        breath = "Respira in modo naturale (schema 3:3 in riscaldamento, poi 2:2 durante la seduta)."
        recovery = "Recupero passivo leggero; l'alimentazione e la doccia dopo l'uscita possono essere normali."
    elif liv == 'medio':
        pacing = "Vai con prudenza: rallenta aggiungendo 10-15 secondi al km e evita salite ripide."
        breath = "Allunga l'espirazione: aiuta a mantenere più basso il battito nelle parti più dure."
        recovery = "Reintegro liquidi e sali più attento del solito; evita shock termici immediati dopo l'allenamento."
    else:
        pacing = "Oggi è preferibile evitare una seduta intensa: fai 30-40 minuti di camminata dinamica su terreno morbido."
        breath = "Respira con calma e profondamente; il recupero oggi ha più valore dell'allenamento."
        recovery = "Stretching passivo e riposo attivo; se i sintomi persistono, valuta un controllo professionale."

    coach_content = {
        '1. Prima della seduta': {
            'colore': C_SONNO,
            'blocchi': [
                ('Mobilità ed attivazione', [
                    'Cammina sui talloni e poi sulle punte per attivare caviglie e arco plantare.',
                    '5 minuti di mobilità dinamica: affondi, slanci, rotazioni delle anche.',
                    'Evita stretching statico troppo intenso prima di corse di qualità.',
                ]),
                ('Preparazione mentale', [
                    f'Zona consigliata: {zona_consigliata}. I primi 10 minuti devono essere più tranquilli del normale.',
                    'Fai 10 respiri profondi prima di partire.',
                    pacing,
                ]),
            ],
        },
        '2. Durante la seduta': {
            'colore': C_AMBRA,
            'blocchi': [
                ('Postura e ritmo', [
                    f'Cadenza indicativa: {cadenza_target}.',
                    'Mantieni busto leggermente inclinato in avanti; guarda davanti a te.',
                    'Rilassa mani e mascella per ridurre tensioni inutili.',
                ]),
                ('Sforzo e sicurezza', [
                    breath,
                    'Se compaiono fitte acute o dolori localizzati, interrompi e valuta le condizioni.',
                    'Se la sensazione di fatica peggiora marcatamente dopo i primi 15 minuti, riduci il carico.',
                ]),
            ],
        },
        '3. Dopo la seduta': {
            'colore': C_RPE,
            'blocchi': [
                ('Rientro alla calma', [
                    'Cammina almeno 3-5 minuti per riportare il battito a valori più bassi.',
                    "Entro 45 minuti, priorizza carboidrati + un po' di proteine.",
                ]),
                ('Cura muscolare', [
                    "Stretching passivo 45-60 s per polpacci, quadricipiti e flessori dell'anca.",
                    'Foam rolling lento, senza forzare punti molto dolorosi.',
                ]),
            ],
        },
        '4. Sera e sonno': {
            'colore': C_VIOLA,
            'blocchi': [
                ('Raffreddamento e recupero', [
                    recovery,
                    "Migliora la regolarità del sonno: evita schermi nell'ultima ora prima di dormire.",
                ]),
                ('Obiettivo notte', [
                    f'Obiettivo: raggiungere un sonno regolare e non fare di oggi un giorno di deficit accumulato.',
                    'Se il carico era elevato, la qualità del sonno conta più di “forzare” l\'allenamento.',
                ]),
            ],
        },
    }

    tabs = st.tabs(list(coach_content.keys()))
    for tab, key in zip(tabs, coach_content.keys()):
        with tab:
            content = coach_content[key]
            blocks = ''
            for label, items in content['blocchi']:
                bullets = ''.join(f'<li>{_html.escape(item)}</li>' for item in items)
                blocks += f"<div class='coach-block' style='--block-color: {content['colore']};'><div class='label'>{_html.escape(label)}</div><ul>{bullets}</ul></div>"
            md(f"<div class='panel' style='padding: 20px;'>{blocks}</div>")

    section_head("Trend storici", "Come cambiano le tue abitudini", "L'ultimo andamento di sonno, stress e fatica aiuta a leggere il quadro di oggi.")
    df_plot = df_adv.tail(90).copy()

    def trend_delta(series):
        if len(series) < 15:
            return 0.0
        last = series.tail(14).mean()
        prev = series.head(len(series) - 14).mean()
        return float(last - prev)

    if 'Ore Sonno' in df_plot.columns:
        fig = go.Figure(go.Scatter(x=df_plot['Data_Chart'], y=df_plot['Ore Sonno'], mode='lines', line=dict(color=C_SONNO, width=2), fill='tozeroy', fillcolor=rgba(C_SONNO, 0.08)))
        fig.update_layout(paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG, margin=dict(l=30, r=20, t=10, b=30), height=260, showlegend=False, font=dict(color=TXT_SECONDARY, family='Inter, sans-serif'))
        fig.update_xaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        fig.update_yaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    if 'Stress Lavoro' in df_plot.columns:
        fig = go.Figure(go.Scatter(x=df_plot['Data_Chart'], y=df_plot['Stress Lavoro'], mode='lines', line=dict(color=C_STRESS, width=2), fill='tozeroy', fillcolor=rgba(C_STRESS, 0.08)))
        fig.update_layout(paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG, margin=dict(l=30, r=20, t=10, b=30), height=260, showlegend=False, font=dict(color=TXT_SECONDARY, family='Inter, sans-serif'))
        fig.update_xaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        fig.update_yaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    if 'RPE' in df_plot.columns:
        fig = go.Figure(go.Scatter(x=df_plot['Data_Chart'], y=df_plot['RPE'], mode='lines', line=dict(color=C_RPE, width=2), fill='tozeroy', fillcolor=rgba(C_RPE, 0.08)))
        fig.update_layout(paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG, margin=dict(l=30, r=20, t=10, b=30), height=260, showlegend=False, font=dict(color=TXT_SECONDARY, family='Inter, sans-serif'))
        fig.update_xaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        fig.update_yaxes(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    section_head("Export", "Porta il report dal tuo coach", "Scarica l'analisi in testo semplice e un file HTML parsabile.")
    coach_txt = ''
    for tab_key, content in coach_content.items():
        coach_txt += f'\n[{tab_key.upper()}]\n'
        for label, items in content['blocchi']:
            coach_txt += f'  {label}:\n'
            for item in items:
                coach_txt += f'    - {item}\n'

    report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (target originale: {distanza_target:.1f} km)
Indice 0-100: {risk_index:.0f}
Recovery Score: {recovery_score:.0f}%
SMA proxy: {sma:.1f}

VERDETTO: {_plain(hero_msg)}
{_plain(nota_coerenza)}

PROTOCOLLO COACH COMPLETO{coach_txt}
--------------------------------"""

    col1, col2 = st.columns(2)
    with col1:
        st.download_button('Scarica TXT', data=report_testo, file_name='runai_report.txt', mime='text/plain', use_container_width=True)
    with col2:
        html_report = f"""<html><body><h1>{_html.escape(tit)}</h1><p>{_html.escape(hero_msg)}</p><p>Indice 0-100: {risk_index:.0f}</p></body></html>"""
        st.download_button('Scarica HTML', data=html_report, file_name='runai_report.html', mime='text/html', use_container_width=True)

    if dates_artificial:
        st.warning("Le date dei grafici sono state generate artificialmente perché manca una colonna temporale reale; i trend vanno interpretati con cautela.")

    md(f"""
    <div class='panel' style='margin-top: 22px;'>
      <p class='eyebrow'>Nota tecnica</p>
      <div class='mini-caption'>
        L'indice di rischio e la distanza consigliata sono decisioni sintetiche, non diagnosi mediche. Le regole sono euristiche, non validazioni cliniche. I pesi, le soglie e le penalità vanno verificati e documentati con dati reali prima di un uso operativo esteso.
      </div>
    </div>
    """)
