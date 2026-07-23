from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_HOME
from utils.sidebar import sidebar_comune

st.set_page_config(page_title="RUN AI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")
carica_css()

IMG_HERO_HOME = get_svg_url(SVG_HOME)

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}
    st.session_state.device_connected = False
    st.session_state.diario_note = []

df, df_full, filtro_tempo = sidebar_comune()

# =========================================================
# CONTENUTO HOME
# =========================================================
header_block(
    "RUNAI // Master Thesis Project",
    "PERFORMANCE INTELLIGENCE SYSTEM",
    "Piattaforma avanzata di Sport Data Science e Machine Learning per l'analisi predittiva e la prevenzione del rischio infortuni nei runner amatori.",
    IMG_HERO_HOME, "Executive Dashboard"
)

st.markdown("""
<div style='display:flex; align-items:center; gap:8px; margin-bottom:18px;'>
    <span style='width:8px; height:8px; border-radius:50%; background:#00F5A0;
                 box-shadow:0 0 8px #00F5A0; display:inline-block;'></span>
    <span style='font-family:"JetBrains Mono",monospace; font-size:0.75em; color:#00F5A0;
                 letter-spacing:0.1em;'>SISTEMA ONLINE — DATI SINCRONIZZATI</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
<strong>Benvenuto in RUNAI.</strong> Questo sistema integra IoT, Wearable Analytics e modelli di Machine Learning supervisionati per ottimizzare i carichi di allenamento e supportare le decisioni del preparatore o dell'atleta.
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

tot_km = df_full['Distanza (km)'].sum()
n_sessioni = len(df_full)
km_ultimi_15 = df_full['Distanza (km)'].tail(15).sum()
km_precedenti_15 = df_full['Distanza (km)'].tail(30).head(15).sum() if len(df_full) >= 30 else 0
delta_km = km_ultimi_15 - km_precedenti_15 if km_precedenti_15 > 0 else 0
delta_pct = (delta_km / km_precedenti_15 * 100) if km_precedenti_15 > 0 else 0

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("KM Totali (90gg)", f"{tot_km:.0f} km",
               f"{delta_pct:+.0f}% ultime 2 sett." if km_precedenti_15 > 0 else None)
col_h2.metric("Sessioni Monitorate", f"{n_sessioni}")
col_h3.metric("Modelli ML Attivi", "5 Algoritmi", "Random Forest, Regressioni, K-Means")
col_h4.metric("Stato Sistema", "Online", "Sync attivo")

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

st.markdown("<p style='font-size:0.8em; color:#8792A3; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.08em; margin-bottom:4px;'>TREND DISTANZA — ULTIME SESSIONI</p>", unsafe_allow_html=True)
serie_km = df_full['Distanza (km)'].tail(30).reset_index(drop=True)
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    y=serie_km, mode='lines', fill='tozeroy', name='Distanza',
    line=dict(color='#00E5FF', width=2.5),
    fillcolor='rgba(0,229,255,0.12)',
    hovertemplate='%{y:.1f} km<extra></extra>'
))
fig_trend.update_layout(
    height=140, margin=dict(l=0, r=0, t=6, b=0),
    showlegend=False,
    xaxis=dict(visible=False, title=None),
    yaxis=dict(visible=False, title=None),
    title=None,
)
fig_trend_finale = style_fig(fig_trend)
fig_trend_finale.update_layout(showlegend=False, title=None)
st.plotly_chart(fig_trend_finale, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
st.subheader("Panoramica Moduli Principali")

def icona_svg(tipo, colore, colore_id):
    """Genera un'icona vettoriale astratta sfumata in stile sport-tech (no emoji, no foto)."""
    base = f"""
    <svg viewBox="0 0 64 64" width="52" height="52" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="glow{colore_id}" cx="50%" cy="45%" r="65%">
          <stop offset="0%" stop-color="{colore}" stop-opacity="0.55"/>
          <stop offset="100%" stop-color="{colore}" stop-opacity="0"/>
        </radialGradient>
        <filter id="blur{colore_id}" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="1.4"/>
        </filter>
      </defs>
      <circle cx="32" cy="32" r="30" fill="url(#glow{colore_id})"/>
      <circle cx="32" cy="32" r="29" fill="none" stroke="{colore}" stroke-width="1" opacity="0.25"/>
      <g stroke="{colore}" fill="{colore}">
        {tipo}
      </g>
    </svg>
    """
    return base

forme = {
    "pulse": f'<polyline points="12,34 22,34 26,20 32,46 37,28 41,34 52,34" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" filter="url(#blur1)" />',
    "bars": '<g stroke-linecap="round"><rect x="16" y="30" width="7" height="18" rx="2" opacity="0.55"/><rect x="28" y="20" width="7" height="28" rx="2" opacity="0.8"/><rect x="40" y="12" width="7" height="36" rx="2"/></g>',
    "radar": '<g fill="none" stroke-width="1.6"><circle cx="32" cy="32" r="18" opacity="0.35"/><circle cx="32" cy="32" r="11" opacity="0.55"/><circle cx="32" cy="32" r="3.5" opacity="1" fill-opacity="1"/><line x1="32" y1="6" x2="32" y2="14" stroke-width="2.2"/><line x1="32" y1="50" x2="32" y2="58" stroke-width="2.2" opacity="0.4"/><line x1="6" y1="32" x2="14" y2="32" stroke-width="2.2" opacity="0.4"/><line x1="50" y1="32" x2="58" y2="32" stroke-width="2.2" opacity="0.4"/></g>',
    "network": '<g stroke-width="1.4"><line x1="18" y1="22" x2="32" y2="14" opacity="0.5"/><line x1="18" y1="22" x2="18" y2="42" opacity="0.5"/><line x1="18" y1="42" x2="32" y2="50" opacity="0.5"/><line x1="32" y1="14" x2="46" y2="22" opacity="0.5"/><line x1="46" y1="22" x2="46" y2="42" opacity="0.5"/><line x1="46" y1="42" x2="32" y2="50" opacity="0.5"/><line x1="32" y1="14" x2="32" y2="50" opacity="0.3"/><circle cx="18" cy="22" r="3.4"/><circle cx="32" cy="14" r="3.4"/><circle cx="46" cy="22" r="3.4"/><circle cx="18" cy="42" r="3.4"/><circle cx="46" cy="42" r="3.4"/><circle cx="32" cy="50" r="3.4"/></g>',
    "check": '<g fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="32" cy="32" r="16" opacity="0.3"/><polyline points="23,32 29,39 42,24"/></g>',
    "skeleton": '<g stroke-width="1.6" stroke-linecap="round"><line x1="32" y1="14" x2="32" y2="30" opacity="0.6"/><line x1="32" y1="18" x2="20" y2="26" opacity="0.6"/><line x1="32" y1="18" x2="44" y2="26" opacity="0.6"/><line x1="32" y1="30" x2="22" y2="42" opacity="0.6"/><line x1="32" y1="30" x2="42" y2="42" opacity="0.6"/><line x1="22" y1="42" x2="19" y2="54" opacity="0.6"/><line x1="42" y1="42" x2="45" y2="54" opacity="0.6"/><circle cx="32" cy="10" r="4"/><circle cx="32" cy="18" r="2"/><circle cx="20" cy="26" r="2"/><circle cx="44" cy="26" r="2"/><circle cx="32" cy="30" r="2"/><circle cx="22" cy="42" r="2"/><circle cx="42" cy="42" r="2"/></g>',
}

moduli = [
    {"num": "01", "titolo": "Stato di Forma", "colore": "#00E5FF", "id": 1, "icona": forme["pulse"], "desc": "Configura i parametri biologici giornalieri, sonno e stress per avviare il calcolo predittivo dell'allenamento.", "tag": "Input Giornaliero"},
    {"num": "02", "titolo": "Statistiche & Analisi", "colore": "#00F5A0", "id": 2, "icona": forme["bars"], "desc": "Esplora lo storico delle sessioni con grafici e statistiche descrittive su volumi, ritmi e andamento nel tempo.", "tag": "Storico Sessioni"},
    {"num": "03", "titolo": "KPI Dashboard", "colore": "#FFB020", "id": 3, "icona": forme["radar"], "desc": "Colpo d'occhio sugli indicatori chiave di performance: carico, recupero e stato generale dell'atleta.", "tag": "Metriche Chiave"},
    {"num": "04", "titolo": "Analisi Predittiva ML", "colore": "#FF6A3D", "id": 4, "icona": forme["network"], "desc": "Random Forest, Regressioni e Cluster K-Means per comprendere i pattern nascosti nel tuo storico e prevedere il rischio.", "tag": "5 Modelli Attivi"},
    {"num": "05", "titolo": "Consiglio Finale", "colore": "#00E5FF", "id": 5, "icona": forme["check"], "desc": "Sintesi operativa: distanza consigliata, zone cardiache e raccomandazioni per la sessione odierna.", "tag": "Report Giornaliero"},
    {"num": "06", "titolo": "Computer Vision", "colore": "#00F5A0", "id": 6, "icona": forme["skeleton"], "desc": "Analisi biomeccanica della falcata tramite video e stima del rischio associato ai sovraccarichi articolari.", "tag": "Pose Estimation"},
]

riga1 = st.columns(3)
riga2 = st.columns(3)
colonne_moduli = riga1 + riga2

for col, m in zip(colonne_moduli, moduli):
    with col:
        svg_icona = icona_svg(m["icona"], m["colore"], m["id"])
        svg_icona = svg_icona.replace('<g ', f'<g stroke="{m["colore"]}" fill="{m["colore"]}" ', 1) if '<g ' in svg_icona else svg_icona
        st.markdown(f"""
        <div class='kpi-card' style='text-align:left; height: 240px; position:relative;
                    border-left: 3px solid {m['colore']};
                    transition: box-shadow 0.2s ease;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
                    margin-bottom: 16px;'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div style='width:52px; height:52px;'>{svg_icona}</div>
                <span style='font-family:"JetBrains Mono",monospace; font-size:0.7em; color:#566178;
                             letter-spacing:0.1em;'>{m['num']}</span>
            </div>
            <h3 style='color:{m['colore']}; margin-top:10px; margin-bottom:8px; font-size:1.15em;'>{m['titolo']}</h3>
            <p style='color:#8792A3; font-size:0.88em; line-height:1.5; margin-bottom:14px;'>{m['desc']}</p>
            <div style='position:absolute; bottom:16px; left:20px; right:20px;'>
                <span style='font-family:"JetBrains Mono",monospace; font-size:0.7em; color:{m['colore']};
                             background:rgba(255,255,255,0.04); padding:4px 10px; border-radius:20px;
                             letter-spacing:0.05em;'>● {m['tag']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:18px 0 4px 0; border-top:1px solid #1c2333;'>
    <p style='color:#566178; font-size:0.75em; font-family:"JetBrains Mono",monospace; letter-spacing:0.08em;'>
        RUNAI PERFORMANCE INTELLIGENCE SYSTEM — Master Thesis Project
    </p>
</div>
""", unsafe_allow_html=True)
