import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.sidebar import sidebar_comune
from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, get_svg_url
from utils.kpi_engine import calcola_kpi_giornalieri
from utils.ml_engine import (
    get_bundle_classificazione, get_bundle_regressione, get_bundle_cluster,
    stima_rischio_oggi, analisi_completa_oggi, punteggio_rischio_combinato,
    PESI_RISCHIO, FEATURE_LABELS_CLASS,
)

st.set_page_config(page_title="Consiglio Finale", layout="wide")
carica_css()

# Funzione di utilità per renderizzare HTML in modo sicuro
def md(text):
    # Rimuove gli spazi iniziali di ogni riga per evitare che 
    # Streamlit lo interpreti come un blocco di codice Markdown
    testo_pulito = "\n".join([line.strip() for line in text.split("\n")])
    st.markdown(testo_pulito, unsafe_allow_html=True)

# Inizializzazione sicura dello stato se manca
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
if 'analisi_fatta' not in st.session_state:
    st.session_state.analisi_fatta = False
if 'risultati_analisi' not in st.session_state:
    st.session_state.risultati_analisi = {}

# Chiamata sicura alla sidebar con controllo sul risultato
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
    None, "Executive Summary"
)

if not st.session_state.get('analisi_fatta', False):
    st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA' per sbloccare i consigli personalizzati.")
else:
    r = st.session_state.risultati_analisi
    df_base = st.session_state.dati.copy()

    # Il bundle ML e la stima del rischio richiedono 'r' e 'df_base',
    # quindi vanno calcolati qui (non in cima al file, dove non esistono ancora)
    # I singoli bundle sono cache_resource in ml_engine: chiamarli qui non
    # riallena nulla (il training è già avvenuto altrove), ci dà solo accesso
    # diretto ai modelli per disegnare i grafici più sotto.
    class_bundle = get_bundle_classificazione(df_base)
    cluster_bundle = get_bundle_cluster(df_base)
    reg_bundle = get_bundle_regressione(df_base)

    # analisi_completa_oggi interroga TUTTI i modelli (RF, Logistic, K-Means,
    # Linear Regression) per lo scenario di oggi: stessa funzione usata dalla
    # pagina "Analisi Predittiva ML", così il verdetto è sempre coerente.
    analisi_ml = analisi_completa_oggi(
        df_base,
        distanza=r.get('distanza_oggi', 10.0),
        ore_sonno=r.get('ore_sonno', 7.5),
        stress=r.get('stress_lavoro', 5),
        rpe=r.get('rpe_previsto', 5),
    )
    rischio_ml = analisi_ml["componenti"]["random_forest"]
    fattore_ml = analisi_ml["fattore_principale_rf"]

    # =========================================================
    # TOKEN DI DESIGN (High-Tech Sports Theme)
    # =========================================================
    PANEL_BG    = "#0D1117"
    PANEL_BD    = "#1E2633"
    PANEL_BD_H  = "#2A3546"
    TXT_PRIMARY   = "#F8F9FA"
    TXT_SECONDARY = "#8B949E"
    TXT_TERTIARY  = "#485363"

    C_SONNO  = "#2E90FF"
    C_STRESS = "#FF453A"
    C_RPE    = "#30D158"
    C_AMBRA  = "#FF9F0A"
    C_VIOLA  = "#BF5AF2"
    C_NEUTRO = "#1F2733"
    C_ARANCIO = "#FF6B35"

    # Badge testuali (non-emoji) usati al posto delle emoji nei messaggi e
    # nelle card dei modelli: un pallino colorato + un'etichetta breve.
    def badge(colore, testo):
        return f"<span class='status-badge'><span class='status-dot' style='background:{colore};'></span>{testo}</span>"

    def model_glyph(colore, sigla):
        return f"<span class='model-glyph' style='--gc:{colore};'>{sigla}</span>"

    md(f"""
    <style>
    .panel {{
        background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 12px;
        padding: 24px; transition: border-color .2s ease, box-shadow .2s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}
    .panel:hover {{ border-color: {PANEL_BD_H}; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }}
    
    .panel-flush {{ padding: 0; overflow: hidden; }}
    .panel-flush .panel-body {{ padding: 16px 20px; }}
    .panel-flush .chart-top-rule {{ height: 4px; width: 100%; }}

    .eyebrow {{
        font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.12em;
        text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 6px 0; font-weight:700;
    }}
    .section-head {{ margin: 12px 0 20px 0; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
    .section-head h3 {{
        font-family:'Oswald',sans-serif; font-weight:600; color:{TXT_PRIMARY};
        margin:0; font-size:1.4rem; letter-spacing:.02em; text-transform:uppercase;
    }}
    .section-head .sub {{ color:{TXT_SECONDARY}; font-size:.9rem; font-family:'Inter',sans-serif; margin-top:4px; }}
    .section-head .rule {{ flex:1; height:1px; background: linear-gradient(90deg, {PANEL_BD} 0%, transparent 100%); align-self:center; }}

    .coach-block {{ margin-bottom: 22px; background: rgba(255,255,255,0.02); padding: 16px; border-radius: 8px; border-left: 3px solid var(--block-color); }}
    .coach-block:last-child {{ margin-bottom: 0; }}
    .coach-block .label {{
        font-family:'Oswald',sans-serif; font-size:.9rem; letter-spacing:.05em;
        text-transform:uppercase; margin-bottom:10px; font-weight:600; color: var(--block-color);
    }}
    .coach-block ul {{ margin:0; padding-left:0; list-style:none; }}
    .coach-block li {{
        position:relative; padding-left:18px; margin-bottom:10px; color:{TXT_SECONDARY};
        font-family:'Inter',sans-serif; font-size:.95rem; line-height:1.6;
    }}
    .coach-block li::before {{ content:"▸"; position:absolute; left:0; top:1px; color:var(--block-color); font-size: 1rem; }}

    .split-sheet {{ padding:0; overflow:hidden; border-radius: 12px; }}
    .split-row {{
        display:grid; grid-template-columns: 1.2fr 1fr 1fr 1.6fr;
        align-items:center; padding:18px 24px; border-bottom:1px solid {PANEL_BD}; gap:12px;
    }}
    .split-row:last-child {{ border-bottom:none; }}
    .split-row .sr-label {{ font-family:'Oswald',sans-serif; font-size:1rem; letter-spacing:.02em; text-transform:uppercase; color:{TXT_PRIMARY}; font-weight:500; }}
    .split-row .sr-value {{ font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; }}
    .split-row .sr-value .unit {{ font-family:'Inter',sans-serif; font-size:.45em; color:{TXT_SECONDARY}; margin-left:4px; text-transform:uppercase; }}
    .split-row .sr-ref {{ font-family:'JetBrains Mono',monospace; font-size:.85rem; color:{TXT_SECONDARY}; }}
    .split-row .sr-note {{ font-family:'Inter',sans-serif; font-size:.85rem; color:{TXT_SECONDARY}; line-height:1.5; border-left: 1px solid {PANEL_BD}; padding-left: 12px; }}
    .split-head {{
        display:grid; grid-template-columns: 1.2fr 1fr 1fr 1.6fr; padding:12px 24px;
        border-bottom:1px solid {PANEL_BD}; background:#161C24;
    }}
    .split-head span {{ font-family:'JetBrains Mono',monospace; font-size:.65rem; letter-spacing:.12em; text-transform:uppercase; color:{TXT_TERTIARY}; font-weight:700; }}

    .hud-grid {{ display: flex; gap: 20px; align-items: flex-end; margin-bottom: 10px; }}
    .hud-stat {{ flex: 1; }}
    .hud-stat h2 {{ margin:0; font-family:"Oswald",sans-serif; font-weight:700; font-size:1.8rem; text-transform:uppercase; letter-spacing:.02em; line-height:1; }}

    .lane-chip {{
        background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:10px;
        padding:14px 16px 16px 16px; position:relative; overflow:hidden;
    }}
    .lane-chip::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--zc); }}
    .lane-chip .lane-num {{ font-family:'Oswald',sans-serif; font-weight:600; font-size:1.6em; color:var(--zc); line-height:1; margin-bottom:2px; }}
    .lane-chip .zt {{ font-family:'JetBrains Mono',monospace; font-size:.68em; letter-spacing:.08em; color:{TXT_TERTIARY}; text-transform:uppercase; font-weight:600; }}
    .lane-chip .zn {{ font-family:'Inter',sans-serif; font-weight:600; color:{TXT_PRIMARY}; margin:6px 0 6px 0; font-size:.95em; }}
    .lane-chip .zd {{ font-family:'Inter',sans-serif; color:{TXT_SECONDARY}; font-size:.85em; line-height:1.4; }}
    
    .chart-caption {{ border-top: 1px solid {PANEL_BD}; margin-top: 10px; padding-top: 10px; color:{TXT_SECONDARY}; font-family:'Inter',sans-serif; font-size:.9rem; line-height:1.55; }}

    /* ===================== ZONE FC (nuovo stile) ===================== */
    .zone-track-wrap {{ margin-bottom: 22px; }}
    .zone-track {{
        height: 10px; border-radius: 6px; overflow: hidden; display: flex;
        border: 1px solid {PANEL_BD}; box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
    }}
    .zone-track .seg {{ height: 100%; }}
    .zone-track-labels {{
        display:flex; justify-content:space-between; margin-top:8px;
        font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.06em;
        text-transform:uppercase; color:{TXT_TERTIARY};
    }}
    .zone-card {{
        background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 14px;
        overflow: hidden; height: 100%;
        transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
        box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    }}
    .zone-card:hover {{
        border-color: {PANEL_BD_H}; transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.32);
    }}
    .zone-card.is-active {{ border-color: var(--zc); }}
    .zone-card-top {{ height: 4px; width: 100%; background: var(--zc); }}
    .zone-card-body {{ padding: 20px 20px 22px 20px; }}
    .zone-card-head {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
    .zone-badge {{
        flex-shrink:0; display:flex; align-items:center; justify-content:center;
        width:38px; height:38px; border-radius:50%;
        border:2px solid var(--zc); color:var(--zc); background: var(--zc)14;
        font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.05rem;
    }}
    .zone-head-txt {{ display:flex; flex-direction:column; gap:3px; }}
    .zone-eyebrow {{
        font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.1em;
        text-transform:uppercase; color:{TXT_TERTIARY}; font-weight:700;
    }}
    .zone-title {{
        font-family:'Oswald',sans-serif; font-weight:600; font-size:1.05rem;
        text-transform:uppercase; letter-spacing:.02em; color:{TXT_PRIMARY};
    }}
    .zone-desc {{ font-family:'Inter',sans-serif; color:{TXT_SECONDARY}; font-size:.88rem; line-height:1.6; margin:0; }}

    .chart-caption {{ border-top: 1px solid {PANEL_BD}; margin-top: 10px; padding-top: 10px; color:{TXT_SECONDARY}; font-family:'Inter',sans-serif; font-size:.9rem; line-height:1.55; }}

    /* ===================== NUOVI STILI "WOW" ===================== */
    @keyframes heroReveal {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .hero-panel {{
        background: linear-gradient(135deg, {PANEL_BG} 0%, #131A24 100%);
        border: 1px solid {PANEL_BD}; border-radius: 18px; padding: 34px;
        position: relative; overflow: hidden; animation: heroReveal .6s ease-out;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }}
    .hero-panel::after {{
        content:""; position:absolute; top:-45%; right:-8%; width:360px; height:360px;
        background: radial-gradient(circle, var(--hero-color) 0%, transparent 70%);
        opacity:.14; pointer-events:none;
    }}
    .hero-grid {{ display:flex; align-items:center; gap:40px; flex-wrap:wrap; position:relative; z-index:1; }}
    .hero-info {{ flex:1; min-width:260px; }}
    .hero-kicker {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; font-weight:700; margin:0 0 8px 0; }}
    .hero-title {{ font-family:'Oswald',sans-serif; font-weight:700; font-size:2.5rem; letter-spacing:.02em; margin:0 0 12px 0; text-transform:uppercase; line-height:1.05; }}
    .hero-msg {{ font-family:'Inter',sans-serif; color:{TXT_SECONDARY}; font-size:1.02rem; line-height:1.65; margin:0 0 22px 0; max-width:540px; }}
    .hero-stats {{ display:flex; gap:32px; flex-wrap:wrap; }}
    .hero-stat {{ display:flex; flex-direction:column; }}
    .hero-stat .hs-val {{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.75rem; }}
    .hero-stat .hs-val .unit {{ font-family:'Inter',sans-serif; font-size:.48em; color:{TXT_SECONDARY}; margin-left:3px; text-transform:uppercase; }}
    .hero-stat .hs-label {{ font-family:'Inter',sans-serif; font-size:.78rem; color:{TXT_SECONDARY}; margin-top:5px; }}

    .value-pill {{ display:inline-flex; align-items:baseline; padding:4px 12px; border-radius:8px; }}
    .row-icon {{ margin-right:9px; font-size:1.05em; }}
    .dot-flag {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; vertical-align:middle; flex-shrink:0; }}
    .mini-caption {{ font-family:'Inter',sans-serif; font-size:.78rem; color:{TXT_TERTIARY}; margin:6px 0 18px 0; line-height:1.5; }}

    /* Titolo dei grafici (mancava: causava testo "attaccato" al dot-flag) */
    .panel-title {{
        display:flex; align-items:center; gap:2px; margin:0; font-size:1rem;
        font-family:'Inter',sans-serif; line-height:1.4;
    }}

    /* Badge non-emoji per i messaggi semaforici (verde/ambra/arancio/rosso) */
    .status-badge {{ display:inline-flex; align-items:center; }}
    .status-dot {{
        display:inline-block; width:11px; height:11px; border-radius:50%;
        margin-right:9px; vertical-align:middle; flex-shrink:0;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.05);
    }}

    /* Piccolo distintivo circolare (sostituisce le emoji dei modelli ML) */
    .model-glyph {{
        display:inline-flex; align-items:center; justify-content:center;
        width:40px; height:40px; border-radius:50%; margin-bottom:8px;
        border:2px solid var(--gc); color:var(--gc);
        font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.78rem;
        letter-spacing:.02em;
    }}

    /* ===================== RESPONSIVITÀ / ANTI-TAGLIO ===================== */
    /* Evita che titoli e numeri lunghi vengano tagliati o si sovrappongano */
    .hero-title {{ word-break: break-word; }}
    .hero-msg, .chart-caption, .mini-caption, .sr-note, .zd {{ overflow-wrap: break-word; }}
    .split-row, .split-head {{ overflow-wrap: break-word; }}
    .hud-grid {{ flex-wrap: wrap; row-gap: 18px; }}
    .hud-stat {{ min-width: 130px; }}
    .lane-chip {{ min-height: 0; }}
    .panel {{ overflow: visible; }}
    .panel-flush {{ overflow: hidden; }} /* solo qui serve per l'angolo arrotondato del top-rule */

    @media (max-width: 900px) {{
        .split-row, .split-head {{ grid-template-columns: 1fr 1fr; row-gap: 6px; }}
        .split-row .sr-note {{ grid-column: 1 / -1; border-left:none; padding-left:0; border-top:1px solid {PANEL_BD}; padding-top:8px; margin-top:4px; }}
        .hero-title {{ font-size: 1.9rem; }}
        .hero-stats {{ gap: 20px; }}
    }}
    </style>
    """)

    def section_head(eyebrow, title, sub=None):
        sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
        md(f"""
        <div class='section-head'>
            <div class='head-txt'>
                <p class='eyebrow'>{eyebrow}</p>
                <h3>{title}</h3>
                {sub_html}
            </div>
            <div class='rule'></div>
        </div>
        """)

    # =========================================================
    # CALCOLI BASE E GESTIONE ROBUSTA DATASET
    # =========================================================
    risk_score_euristico = min(100,
        (40 if r.get('ore_sonno', 7.5) < 6 else 25 if r.get('ore_sonno', 7.5) < 6.5 else 10) +
        (35 if r.get('stress_lavoro', 5) >= 8 else 20 if r.get('stress_lavoro', 5) >= 6 else 5) +
        (30 if r.get('rpe_previsto', 5) >= 8 else 15 if r.get('rpe_previsto', 5) >= 6 else 5) +
        (20 if r.get('ore_sonno', 7.5) < 6.5 and r.get('stress_lavoro', 5) >= 7 and r.get('rpe_previsto', 5) >= 7 else 0)
    )

    # Punteggio combinato "ufficiale": media pesata di tutti e 4 i modelli ML
    # più l'euristica clinica (PESI_RISCHIO in ml_engine.py), la stessa
    # formula usata nella pagina "Analisi Predittiva ML".
    risk_score = punteggio_rischio_combinato(analisi_ml["componenti"], risk_score_euristico)

    recovery_score = max(0, 100 - abs(r.get('ore_sonno', 7.5) - 7.5) * 13.33)
    sma = (r.get('stress_lavoro', 5) * r.get('rpe_previsto', 5)) / r.get('ore_sonno', 7.5) if r.get('ore_sonno', 7.5) > 0 else 0

    # Recupero di tutti i KPI proprietari della tesi (non solo la SMA)
    try:
        kpi_oggi = calcola_kpi_giornalieri(r)
    except Exception:
        kpi_oggi = {"SMA": sma, "ISLR": None, "IITR": None, "IDET": None}
    islr_val = kpi_oggi.get("ISLR")
    iitr_val = kpi_oggi.get("IITR")
    idet_val = kpi_oggi.get("IDET")

    distanza_target = r.get('distanza_oggi', 10.0)
    distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0

    if risk_score < 25:
        tit, col, liv = "AUTORIZZATO", C_RPE, "basso"
    elif risk_score < 60:
        tit, col, liv = "RECUPERO ATTIVO", C_AMBRA, "medio"
    else:
        tit, col, liv = "RIPOSO OBBLIGATORIO", C_STRESS, "alto"

    cadenza_target = "170-180 spm" if liv != "alto" else "165-172 spm (passo rilassato per abbassare l'impatto articolare)"
    zona_consigliata = "Zona 2-3 (aerobico puro)" if liv == "basso" else "Zona 1-2 (sforzo percepito bassissimo)" if liv == "medio" else "Solo mobilità o camminata veloce"

    # Messaggio "in una frase" pensato per essere capito al primo sguardo
    hero_messaggi = {
        "AUTORIZZATO": "Hai dormito e recuperato bene: oggi il corpo ti dà il via libera. Puoi allenarti come da programma.",
        "RECUPERO ATTIVO": "Il corpo è un po' scarico: oggi meglio abbassare l'intensità e ascoltare le sensazioni.",
        "RIPOSO OBBLIGATORIO": "I segnali dicono chiaramente stop: oggi il riposo vale più di qualsiasi allenamento.",
    }
    hero_msg = hero_messaggi.get(tit, "") + f" Il modello segnala che oggi il fattore più determinante è: <strong>{fattore_ml}</strong>."
    scarto = abs(rischio_ml - risk_score_euristico)
    nota_coerenza = (
        "Il modello statistico e le regole cliniche sono d'accordo."
        if scarto < 15 else
        "Attenzione: il modello statistico e le regole cliniche non sono del tutto allineati oggi — vale la pena essere prudenti."
    )

    date_col = next((c for c in df_base.columns if c.lower() in ['data', 'date', 'giorno', 'time']), None)
    df_adv = df_base.copy()
    if date_col:
        df_adv['Data_Chart'] = pd.to_datetime(df_adv[date_col], errors='coerce')
    else:
        df_adv['Data_Chart'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_adv))
    
    df_adv = df_adv.sort_values('Data_Chart').dropna(subset=['Data_Chart'])

    # =========================================================
    # EFFORT EQUALIZER (TELEMETRIA VETTORIALE)
    # =========================================================
    def disegna_telemetria_rischio(score):
        svg_width, svg_height = 800, 100
        bars = 60
        gap = 2
        bar_w = (svg_width - (bars * gap)) / bars
        
        elements = []
        for i in range(bars):
            x = i * (bar_w + gap)
            progression = i / bars
            h = 15 + (progression ** 2.5) * (svg_height - 15)
            y = svg_height - h
            
            threshold_pct = (i / bars) * 100
            is_active = threshold_pct <= score
            
            bar_col = C_RPE if threshold_pct < 25 else C_AMBRA if threshold_pct < 60 else C_STRESS
            fill = bar_col if is_active else C_NEUTRO
            opacity = "1.0" if is_active else "0.3"
            
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" opacity="{opacity}" rx="2"/>')

        marker_x = (score / 100) * svg_width
        
        return f"""
        <svg viewBox="0 -28 {svg_width} {svg_height + 53}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; display:block;">
            <line x1="0" y1="{svg_height}" x2="{svg_width}" y2="{svg_height}" stroke="{TXT_TERTIARY}" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
            {"".join(elements)}
            <g transform="translate({marker_x}, 0)">
                <line x1="0" y1="0" x2="0" y2="{svg_height + 10}" stroke="{TXT_PRIMARY}" stroke-width="2" />
                <polygon points="-6,{svg_height + 10} 6,{svg_height + 10} 0,{svg_height + 18}" fill="{TXT_PRIMARY}" />
                <rect x="-24" y="-20" width="48" height="20" rx="4" fill="{TXT_PRIMARY}" />
                <text x="0" y="-6" font-family="JetBrains Mono, monospace" font-size="12" font-weight="bold" fill="{PANEL_BG}" text-anchor="middle">{int(score)}%</text>
            </g>
            <text x="0" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_RPE}" opacity="0.8">OK (0-25)</text>
            <text x="{svg_width/2}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_AMBRA}" text-anchor="middle" opacity="0.8">ATTENZIONE (26-59)</text>
            <text x="{svg_width}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_STRESS}" text-anchor="end" opacity="0.8">PERICOLO (60-100)</text>
        </svg>
        """

    # Nuovo: gauge circolare "a colpo d'occhio" per la hero section (riutilizzabile
    # anche per altri indicatori 0-100, come la concordanza tra i modelli)
    def disegna_gauge_circolare(score, color, size=210, label="RISCHIO %"):
        radius = 82
        stroke_width = 15
        center = size / 2
        circumference = 2 * 3.14159265 * radius
        offset = circumference * (1 - min(max(score, 0), 100) / 100)

        # width/height fissi (oltre al viewBox): senza questi il browser
        # tratta l'SVG come "responsivo" e lo stira per riempire tutta la
        # larghezza della colonna che lo ospita. Se quella colonna è più
        # larga dell'altezza fissa dell'iframe (embed_svg), il cerchio
        # cresce oltre lo spazio disponibile e viene tagliato in alto/basso.
        # Con width/height espliciti l'SVG mantiene sempre le sue
        # dimensioni naturali e si rimpicciolisce (mai ingrandisce) per
        # adattarsi, restando sempre dentro l'iframe.
        return f"""
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{C_NEUTRO}" stroke-width="{stroke_width}"/>
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
                stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 {center} {center})"/>
            <text x="{center}" y="{center - 4}" text-anchor="middle" font-family="Oswald, sans-serif" font-size="44" font-weight="700" fill="{TXT_PRIMARY}">{int(score)}</text>
            <text x="{center}" y="{center + 24}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" letter-spacing="2" fill="{TXT_SECONDARY}">{label}</text>
        </svg>
        """

    radar_svg = disegna_telemetria_rischio(risk_score)
    gauge_svg = disegna_gauge_circolare(risk_score, col)

    # Inserisce un SVG a schermo trasparente, così si fonde con lo sfondo
    # della pagina invece di creare un riquadro grigio separato, e con
    # spazio extra così non taglia mai nulla ai bordi.
    def embed_svg(svg_code, height, extra_padding=6):
        # height include un margine di sicurezza extra: senza margine, in
        # alcune finestre l'SVG (specie le etichette sopra/sotto la barra)
        # veniva tagliato dall'iframe a altezza fissa.
        # max-width E max-height (mai solo max-width) impediscono all'SVG di
        # essere ingrandito oltre le sue dimensioni naturali quando la
        # colonna che lo ospita è più larga dell'altezza fissa dell'iframe:
        # senza max-height un SVG quadrato "responsivo" può crescere in
        # altezza più dello spazio disponibile e finire tagliato.
        st.components.v1.html(f"""
        <html>
        <head>
        <style>
            html, body {{ margin:0; padding:0; background:transparent; height:100%; overflow:visible; }}
            .svg-wrap {{
                display:flex; align-items:center; justify-content:center;
                width:100%; height:100%; background:transparent;
                padding:{extra_padding}px; box-sizing:border-box; overflow:visible;
            }}
            .svg-wrap svg {{ max-width:100%; max-height:100%; width:auto; height:auto; overflow:visible; }}
        </style>
        </head>
        <body>
            <div class="svg-wrap">{svg_code}</div>
        </body>
        </html>
        """, height=height + 14, scrolling=False)

    # =========================================================
    # HERO SECTION — IL PRIMO COLPO D'OCCHIO
    # =========================================================
    md(f"""
    <div class='hero-panel' style='--hero-color:{col};'>
        <div class='hero-grid'>
            <div class='hero-info'>
                <p class='hero-kicker'>Il verdetto di oggi</p>
                <h1 class='hero-title' style='color:{col};'>{tit}</h1>
                <p class='hero-msg'>{hero_msg}</p>
                <div class='hero-stats'>
                    <div class='hero-stat'>
                        <span class='hs-val' style='color:{TXT_PRIMARY};'>{distanza_consigliata:.1f}<span class='unit'>km</span></span>
                        <span class='hs-label'>Distanza consigliata oggi</span>
                    </div>
                    <div class='hero-stat'>
                        <span class='hs-val' style='color:{C_SONNO};'>{recovery_score:.0f}<span class='unit'>%</span></span>
                        <span class='hs-label'>Quanto sei recuperato</span>
                    </div>
                    <div class='hero-stat'>
                        <span class='hs-val' style='color:{C_AMBRA};'>{sma:.1f}</span>
                        <span class='hs-label'>Carico mentale (SMA)</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    md(f"""
    <div class='mini-caption' style='margin-top:10px; margin-bottom:20px;'>
        Rischio da modello ML: <strong style='color:{TXT_PRIMARY}'>{rischio_ml:.0f}%</strong> ·
        Rischio da regole cliniche: <strong style='color:{TXT_PRIMARY}'>{risk_score_euristico:.0f}%</strong> ·
        {nota_coerenza}
    </div>
    """)

    gc1, gc2 = st.columns([1, 2.2])
    with gc1:
        embed_svg(gauge_svg, height=245, extra_padding=8)
    with gc2:
        md("<p class='eyebrow' style='margin-top:6px;'>Dettaglio soglie di rischio</p>")
        md("<p class='mini-caption'>Più la barra bianca è a destra, più alto è il rischio di allenarti oggi.</p>")
        embed_svg(radar_svg, height=185, extra_padding=10)

    md("<div style='height:24px;'></div>")

    
    # =========================================================
    # SPLIT SHEET (Cronometraggio)
    # =========================================================
    ore_s = r.get('ore_sonno', 7.5)
    sonno_delta_txt = f"{'+' if (ore_s-7.5) >= 0 else ''}{ore_s-7.5:.1f}h rispetto alle 7.5h ideali per notte"

    section_head("Numeri chiave", "I 6 indicatori di oggi", "Ogni riga ti dice cosa significa il numero e cosa cambia per il tuo allenamento.")

    md(f"""
    <div class='panel split-sheet'>
        <div class='split-head'>
            <span>Indicatore</span><span>Valore</span><span>Riferimento</span><span>Cosa significa</span>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Distanza Target</div>
            <div class='sr-value'><span class='value-pill' style='color:{TXT_PRIMARY}; background:{TXT_PRIMARY}14;'>{distanza_consigliata:.1f}<span class='unit'>km</span></span></div>
            <div class='sr-ref'>Piano originale: {distanza_target} km</div>
            <div class='sr-note'>La distanza è stata ridotta del {100 - (distanza_consigliata/distanza_target*100 if distanza_target>0 else 0):.0f}% per non stressare troppo muscoli e articolazioni.</div>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Recovery Score</div>
            <div class='sr-value'><span class='value-pill' style='color:{C_SONNO}; background:{C_SONNO}14;'>{recovery_score:.0f}<span class='unit'>%</span></span></div>
            <div class='sr-ref'>Base: {ore_s:.1f}h di sonno</div>
            <div class='sr-note'>{sonno_delta_txt}. Il sonno è il motore principale del recupero: dormire bene vuol dire correre meglio.</div>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Carico Mentale (SMA)</div>
            <div class='sr-value'><span class='value-pill' style='color:{C_AMBRA}; background:{C_AMBRA}14;'>{sma:.1f}</span></div>
            <div class='sr-ref'>(Stress × Fatica prevista) / Sonno</div>
            <div class='sr-note'>Misura quanto sei "carico" tra testa e corpo insieme. Livello di oggi: {liv.upper()}.</div>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Sforzo Lavorativo (ISLR)</div>
            <div class='sr-value'><span class='value-pill' style='color:{C_STRESS}; background:{C_STRESS}14;'>{f"{islr_val:.1f}" if islr_val is not None and not pd.isna(islr_val) else "N/D"}</span></div>
            <div class='sr-ref'>(Ore Lavoro × Stress) / Distanza</div>
            <div class='sr-note'>Quanto il lavoro "ruba" energie alla corsa. Sopra 6.3 vuol dire che lo stress da lavoro sta consumando troppe risorse per allenarti bene.</div>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Impatto Termico (IITR)</div>
            <div class='sr-value'><span class='value-pill' style='color:{TXT_PRIMARY}; background:{TXT_PRIMARY}14;'>{f"{iitr_val:.1f}" if iitr_val is not None and not pd.isna(iitr_val) else "N/D"}</span></div>
            <div class='sr-ref'>(Temperatura × Vento) / Distanza</div>
            <div class='sr-note'>Dice quanto il meteo di oggi (caldo, vento) rende la corsa più dura. Più alto il numero, più conviene rallentare.</div>
        </div>
        <div class='split-row'>
            <div class='sr-label'><span class='row-icon'></span>Degradazione Termica (IDET)</div>
            <div class='sr-value'><span class='value-pill' style='color:{C_VIOLA}; background:{C_VIOLA}14;'>{f"{idet_val:.1f}" if idet_val is not None and not pd.isna(idet_val) else "N/D"}</span></div>
            <div class='sr-ref'>(FC Media × Temperatura) / Velocità</div>
            <div class='sr-note'>Capisce se il cuore batte più forte solo per il caldo, così non scambi un normale adattamento per un segnale di troppo allenamento.</div>
        </div>
    </div>
    """)

    md("<div style='height:34px;'></div>")

    # =========================================================
    # COACH PERSONALIZZATO
    # =========================================================
    section_head("Coach personalizzato", "Il tuo protocollo passo dopo passo", "Cosa fare prima, durante e dopo la corsa — spiegato in modo semplice, in base ai tuoi dati di oggi.")

    if liv == "basso":
        dinamica_pacing = "Puoi tenere il tuo ritmo normale. Nessuna restrizione sui cambi di velocità: via libera anche a qualche scatto, se previsto."
        dinamica_resp = "Respira in modo naturale (schema 3:3 in riscaldamento, poi 2:2 al ritmo gara)."
        dinamica_rec = "Recupero passivo normale o massaggio leggero. Dopo l'allenamento va bene anche la doccia contrasto caldo/freddo."
    elif liv == "medio":
        dinamica_pacing = "Vai con prudenza: togli 10-15 secondi al km rispetto al tuo ritmo abituale. Evita salite ripide."
        dinamica_resp = "Allunga l'espirazione (schema 3:4): aiuta a tenere basso il battito ed evitare picchi di sforzo."
        dinamica_rec = "Bevi e reintegra sali minerali più del solito. Evita l'acqua fredda dopo la corsa: meglio temperatura neutra, per non stressare il sistema nervoso."
    else:
        dinamica_pacing = "Oggi meglio non correre. Trasforma la seduta in 30-40 minuti di camminata dinamica su terreno piatto e morbido (prato o terra)."
        dinamica_resp = "Respira solo con il naso, in modo profondo: aiuta il corpo a rilassarsi e recuperare."
        dinamica_rec = "Yoga nidra o stretching passivo lungo. Se hai muscoli tesi, meglio il calore del ghiaccio."

    coach_content = {
        " 1. Prima di Correre (Warm-Up)": {
            "colore": C_SONNO,
            "blocchi": [
                ("Attivazione Neurale e Meccanica", [
                    "Cammina sui talloni e poi sulle punte (30 secondi ciascuno) per svegliare caviglie e arco plantare.",
                    "5 minuti di mobilità dinamica: slanci di gamba avanti/lato, rotazioni delle anche, affondi controllati.",
                    "Evita lo stretching statico prima di correre: rende il tendine d'Achille meno reattivo del 5-8%."
                ]),
                ("Preparare Mente e Corpo", [
                    f"Zona di partenza consigliata: {zona_consigliata}. I primi 10 minuti devono sembrarti 'quasi troppo lenti'.",
                    "Fai 10 respiri profondi prima di partire con il cronometro: aiuta a scaricare lo stress della giornata.",
                    f"Indicazione di oggi: {dinamica_pacing}"
                ])
            ],
        },
        " 2. Durante la Corsa": {
            "colore": C_AMBRA,
            "blocchi": [
                ("Postura e Ritmo", [
                    f"Passi al minuto (cadenza): {cadenza_target}. Alzare la cadenza del 5% riduce il carico sulle ginocchia del 20%.",
                    "Busto leggermente inclinato in avanti partendo dalle caviglie, non dalla schiena. Guarda avanti, non i piedi.",
                    "Rilassa mani e mascella: la tensione sul viso si trasmette subito ai muscoli della schiena e delle gambe."
                ]),
                ("Sforzo e Sicurezza", [
                    f"Respirazione: {dinamica_resp}",
                    "Distingui la 'fatica normale' (va bene) dal 'dolore acuto o fitta' (segnale di allarme, fermati).",
                    "Regola dei 15 minuti: se dopo 15 minuti fai più fatica del previsto, taglia subito la distanza a metà."
                ])
            ],
        },
        " 3. Dopo la Corsa (Recupero)": {
            "colore": C_RPE,
            "blocchi": [
                ("Rientro alla Calma", [
                    "Non fermarti di colpo. Cammina 3-5 minuti finché il battito non scende comodamente sotto i 110 bpm.",
                    "Entro 45 minuti mangia carboidrati e un po' di proteine: aiuta il corpo a smettere di 'consumarsi' e a ripartire con il recupero."
                ]),
                ("Cura dei Muscoli", [
                    "Fai stretching passivo tenendo ogni posizione 45-60 secondi (polpacci, dietro coscia, flessori dell'anca).",
                    "Usa il foam roller lentamente. Se trovi un punto dolente, fermati lì e respira per 30 secondi."
                ])
            ],
        },
        " 4. Sera e Sonno": {
            "colore": C_VIOLA,
            "blocchi": [
                ("Rilassare il Sistema Nervoso", [
                    "Doccia o calore: " + dinamica_rec,
                    "10 minuti di meditazione guidata o rilassamento corporeo (NSDR) prima di dormire, se oggi lo stress era sopra 7/10."
                ]),
                ("Dormire Meglio Stanotte", [
                    f"Obiettivo per stanotte: recuperare il sonno arretrato arrivando a {max(7.5, ore_s+0.5):.1f} ore.",
                    "Evita smartphone e TV nell'ultima ora prima di dormire: la luce blu blocca la melatonina, l'ormone che ti fa addormentare."
                ])
            ],
        }
    }

    tabs = st.tabs(list(coach_content.keys()))
    for tab, (nome_tab, contenuto) in zip(tabs, coach_content.items()):
        with tab:
            blocchi_html = ""
            for label, bullets in contenuto["blocchi"]:
                bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                blocchi_html += f"""
                <div class='coach-block' style='--block-color: {contenuto["colore"]};'>
                    <div class='label'>{label}</div>
                    <ul>{bullets_html}</ul>
                </div>
                """
            md(f"<div class='panel' style='padding: 20px;'>{blocchi_html}</div>")

    md("<div style='height:34px;'></div>")

    # =========================================================
    # PREPARAZIONE GRAFICI E STILI COMUNI
    # (spostata qui perché serve già per il grafico delle zone cardiache
    # subito sotto, non solo per i grafici ML più in basso)
    # =========================================================
    CHART_HEIGHT = 280
    layout_base = dict(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TXT_SECONDARY, family="Inter, sans-serif", size=11),
        margin=dict(l=38, r=16, t=10, b=32),
        height=CHART_HEIGHT,
        showlegend=False,
        hoverlabel=dict(bgcolor="#1A2233", font_size=12, font_family="Inter, sans-serif", bordercolor=PANEL_BD),
    )
    axis_style = dict(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
    config_pulita = {'displayModeBar': False}

    media_sonno_90 = df_base['Ore Sonno'].mean() if 'Ore Sonno' in df_base.columns else 7.0
    media_stress_90 = df_base['Stress Lavoro'].mean() if 'Stress Lavoro' in df_base.columns else 5.0

    figs_per_export = []
    insights_export = []

    def chart_card(container, titolo, fig, spiegazione, rule_color=C_NEUTRO):
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        with container:
            md(f"""
            <div class='panel panel-flush'>
                <div class='chart-top-rule' style='background:{rule_color};'></div>
                <div class='panel-body'>
                    <p class='panel-title' style='color:{TXT_PRIMARY}; font-weight:600;'>
                        <span class='dot-flag' style='background:{rule_color};'></span>{titolo}
                    </p>
                </div>
            """)
            st.plotly_chart(fig, use_container_width=True, config=config_pulita)
            md(f"""
                <div class='panel-body' style='padding-top:0;'><div class='chart-caption'><strong>Cosa significa questo grafico?</strong><br>{spiegazione}</div></div>
            </div>
            """)
        figs_per_export.append(fig)
        insights_export.append((titolo, spiegazione))

    # =========================================================
    # CORSIE E ZONE
    # =========================================================
    section_head("Riferimento", "Le tue Zone di Frequenza Cardiaca", "A quale intensità corrispondono le zone che vedi nei grafici qui sotto.")

    corsie = [
        ("1", "Zona 1-2", "Recupero / Base Aerobica", "Sforzo bassissimo: riesci a parlare senza fatica. L'energia arriva dai grassi. Perfetta per costruire resistenza senza accumulare stanchezza.", C_RPE, 34),
        ("2", "Zona 3", "Soglia Aerobica / Tempo", "Ritmo sostenuto, respiro più profondo, poco acido lattico. Serve a rendere il cuore più forte ed efficiente.", C_AMBRA, 33),
        ("3", "Zona 4-5", "Soglia Lattacida / VO2Max", "Sforzo massimo: parlare diventa difficile. Le fibre muscolari lavorano al limite per poi rinforzarsi. Da usare con moderazione se il rischio infortunio è medio o alto.", C_STRESS, 33),
    ]

    # Barra continua che riassume visivamente la progressione di intensità
    # tra le tre zone, in stile "telemetria", coerente con il resto della
    # pagina (invece delle vecchie card isolate senza un filo conduttore).
    segmenti_track = "".join(
        f"<div class='seg' style='width:{pct}%; background:{zcol};'></div>"
        for _, _, _, _, zcol, pct in corsie
    )
    md(f"""
    <div class='zone-track-wrap'>
        <div class='zone-track'>{segmenti_track}</div>
        <div class='zone-track-labels'>
            <span>Sforzo minimo</span><span>Sforzo massimo</span>
        </div>
    </div>
    """)

    cc1, cc2, cc3 = st.columns(3)
    for c, (num, zt, zn, zd, zcol, _pct) in zip([cc1, cc2, cc3], corsie):
        c.markdown(f"""
        <div class='zone-card' style='--zc:{zcol};'>
            <div class='zone-card-top'></div>
            <div class='zone-card-body'>
                <div class='zone-card-head'>
                    <div class='zone-badge'>{num}</div>
                    <div class='zone-head-txt'>
                        <span class='zone-eyebrow'>{zt}</span>
                        <span class='zone-title'>{zn}</span>
                    </div>
                </div>
                <p class='zone-desc'>{zd}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    md("<div style='height:24px;'></div>")

    # ---------------------------------------------------------------
    # GRAFICO "TERMOMETRO CARDIACO" — dove hai vissuto negli ultimi 90 giorni
    # (mancava un grafico sotto le zone: qui trasformiamo lo storico di FC
    # Media in una distribuzione per corsia, con la corsia consigliata per
    # oggi evidenziata da un marcatore).
    # ---------------------------------------------------------------
    if 'FC Media' in df_base.columns:
        fc_serie = df_base['FC Media'].dropna()
        fc_media_glob = fc_serie.mean()
        fc_std_glob = fc_serie.std() if fc_serie.std() > 0 else 1.0
        soglia_bassa = fc_media_glob - 0.5 * fc_std_glob
        soglia_alta = fc_media_glob + 0.5 * fc_std_glob

        n_tot = len(fc_serie)
        pct_z1 = float((fc_serie < soglia_bassa).sum()) / n_tot * 100 if n_tot else 0
        pct_z2 = float(((fc_serie >= soglia_bassa) & (fc_serie <= soglia_alta)).sum()) / n_tot * 100 if n_tot else 0
        pct_z3 = float((fc_serie > soglia_alta).sum()) / n_tot * 100 if n_tot else 0

        zona_oggi_map = {"basso": 0, "medio": 0, "alto": None}
        indice_zona_oggi = zona_oggi_map.get(liv, 0)

        fig_hr_zone = go.Figure()
        segmenti = [
            ("Corsia 1 · Recupero", pct_z1, C_RPE),
            ("Corsia 2 · Soglia Aerobica", pct_z2, C_AMBRA),
            ("Corsia 3 · Soglia Lattacida", pct_z3, C_STRESS),
        ]
        for i, (nome_seg, pct_seg, colore_seg) in enumerate(segmenti):
            fig_hr_zone.add_trace(go.Bar(
                y=["Ultimi 90 giorni"], x=[pct_seg], name=nome_seg, orientation='h',
                marker=dict(
                    color=colore_seg,
                    line=dict(color=TXT_PRIMARY, width=3 if i == indice_zona_oggi else 0),
                ),
                text=[f"{pct_seg:.0f}%"], textposition='inside',
                textfont=dict(color=PANEL_BG if pct_seg > 8 else colore_seg, size=12, family="JetBrains Mono, monospace"),
                insidetextanchor='middle',
            ))
        fig_hr_zone.update_layout(
            barmode='stack', paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
            font=dict(color=TXT_SECONDARY, family="Inter, sans-serif", size=11),
            margin=dict(l=16, r=16, t=10, b=10), height=130,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, font=dict(size=10)),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False),
            hoverlabel=dict(bgcolor="#1A2233", font_size=12, font_family="Inter, sans-serif", bordercolor=PANEL_BD),
        )

        if indice_zona_oggi is not None:
            nome_zona_oggi = segmenti[indice_zona_oggi][0]
            marcatore_txt = f"Il riquadro con il bordo bianco mostra <strong>{nome_zona_oggi}</strong>, la corsia consigliata per l'allenamento di oggi."
        else:
            marcatore_txt = "Oggi nessuna corsia è evidenziata: le condizioni indicano riposo, non una sessione di corsa vera e propria."
        hr_zone_txt = (
            f"Negli ultimi 90 giorni hai passato circa il <strong>{pct_z1:.0f}%</strong> delle sedute in Corsia 1, "
            f"il <strong>{pct_z2:.0f}%</strong> in Corsia 2 e il <strong>{pct_z3:.0f}%</strong> in Corsia 3 "
            f"(soglie calcolate sulla tua frequenza cardiaca media storica). {marcatore_txt}"
        )
        chart_card(st.container(), "Come si distribuisce il tuo storico tra le corsie", fig_hr_zone, hr_zone_txt, col)
    else:
        st.info("Colonna 'FC Media' non trovata: impossibile calcolare la distribuzione storica per corsia.")

    md("<div style='height:34px;'></div>")

    # =========================================================
    # SEZIONE ML: SOTTO IL COFANO — COSA DICONO I MODELLI
    # =========================================================
    section_head("Intelligenza Artificiale", "Sotto il cofano: cosa dicono i modelli",
                 "Il Consiglio Finale non nasce da una sola formula: nasce dal voto di 4 modelli diversi più le regole cliniche. Ecco cosa vede ciascuno.")

    modelli_spiegazione = [
        ("Random Forest", C_RPE,
         "Immagina 100 piccoli allenatori che guardano il tuo storico da angolazioni diverse e votano 'rischio' o 'sicuro'. La Random Forest è la somma dei loro voti: è il modello più usato per il verdetto perché coglie relazioni complesse, tipo 'poco sonno conta doppio se lo stress è già alto'."),
        ("Logistic Regression", C_SONNO,
         "Un modello più semplice e trasparente: assegna un peso fisso a ogni fattore (sonno, stress, RPE...) e li somma. Meno potente della Random Forest, ma più facile da spiegare: 'ogni ora di sonno in meno aumenta il rischio di una quantità precisa'."),
        ("K-Means (Clustering)", C_VIOLA,
         "Raggruppa i tuoi allenamenti passati in 'famiglie' simili per distanza e frequenza cardiaca. Oggi il modello guarda a quale famiglia assomiglia il tuo allenamento previsto, e quanto spesso quella famiglia ha storicamente portato a un giorno a rischio."),
        ("Linear Regression", C_AMBRA,
         "Stima quale dovrebbe essere la tua frequenza cardiaca media in base a velocità, temperatura e distanza. Se il cuore batte più forte del previsto in modo continuativo, è un segnale di affaticamento che ritmo e meteo da soli non spiegano."),
    ]
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col_widget, (nome_m, colore_m, spieg_m) in zip([mc1, mc2, mc3, mc4], modelli_spiegazione):
        col_widget.markdown(f"""
        <div class='lane-chip' style='--zc:{colore_m}; min-height:200px;'>
            <div class='zt'>Modello</div>
            <div class='zn' style='color:{colore_m}; font-size:1.05em;'>{nome_m}</div>
            <div class='zd'>{spieg_m}</div>
        </div>
        """, unsafe_allow_html=True)

    md("<div style='height:24px;'></div>")

    componenti_oggi = {
        "Euristica clinica": risk_score_euristico,
        "Random Forest": analisi_ml["componenti"]["random_forest"],
        "Logistic Regression": analisi_ml["componenti"]["logistic"],
        "Profilo storico (Cluster)": analisi_ml["componenti"]["cluster"],
        "Trend fisiologico (FC)": analisi_ml["componenti"]["trend_fisiologico"],
    }
    pesi_map = {
        "Euristica clinica": PESI_RISCHIO["euristica"],
        "Random Forest": PESI_RISCHIO["random_forest"],
        "Logistic Regression": PESI_RISCHIO["logistic"],
        "Profilo storico (Cluster)": PESI_RISCHIO["cluster"],
        "Trend fisiologico (FC)": PESI_RISCHIO["trend_fisiologico"],
    }
    colori_comp = {
        "Euristica clinica": TXT_TERTIARY, "Random Forest": C_RPE, "Logistic Regression": C_SONNO,
        "Profilo storico (Cluster)": C_VIOLA, "Trend fisiologico (FC)": C_AMBRA,
    }
    sigle_comp = {
        "Euristica clinica": "EU", "Random Forest": "RF", "Logistic Regression": "LR",
        "Profilo storico (Cluster)": "KM", "Trend fisiologico (FC)": "LM",
    }

    # =========================================================
    # IL CONSIGLIO DEGLI ALGORITMI — 5 modelli votano indipendentemente
    # =========================================================
    def verdetto_da_score(v):
        if v < 40:
            return ("VIA LIBERA", C_RPE)
        elif v < 70:
            return ("ATTENZIONE", C_AMBRA)
        else:
            return ("STOP", C_STRESS)

    verdetti = {k: verdetto_da_score(v) for k, v in componenti_oggi.items()}
    conteggio_verdetti = {}
    for _, (etichetta, _) in verdetti.items():
        conteggio_verdetti[etichetta] = conteggio_verdetti.get(etichetta, 0) + 1
    verdetto_maggioranza = max(conteggio_verdetti, key=conteggio_verdetti.get)
    n_maggioranza = conteggio_verdetti[verdetto_maggioranza]
    colore_maggioranza = {"VIA LIBERA": C_RPE, "ATTENZIONE": C_AMBRA, "STOP": C_STRESS}[verdetto_maggioranza]

    md(f"""
    <div class='hero-panel' style='--hero-color:{colore_maggioranza}; padding:28px 34px;'>
        <p class='hero-kicker'>Il Consiglio degli Algoritmi</p>
        <div style='display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:22px;'>
            <span style='font-family:"Oswald",sans-serif; font-weight:700; font-size:2.1rem; text-transform:uppercase; color:{colore_maggioranza};'>{n_maggioranza}/5 dicono {verdetto_maggioranza}</span>
        </div>
        <div style='display:flex; gap:14px; flex-wrap:wrap;'>
            {"".join(f'''
            <div style='flex:1; min-width:140px; background:rgba(255,255,255,0.02); border:1px solid {PANEL_BD}; border-radius:10px; padding:14px; text-align:center;'>
                {model_glyph(verdetti[nome][1], sigle_comp[nome])}
                <div style='font-family:"Inter",sans-serif; font-size:.78em; color:{TXT_SECONDARY}; margin-bottom:8px;'>{nome}</div>
                <div class='value-pill' style='color:{verdetti[nome][1]}; background:{verdetti[nome][1]}1A; font-family:"JetBrains Mono",monospace; font-size:.78em; font-weight:700;'>{verdetti[nome][0]}</div>
            </div>
            ''' for nome in componenti_oggi.keys())}
        </div>
    </div>
    """)

    md("<div style='height:24px;'></div>")

    c_wow1, c_wow2 = st.columns([1.4, 1])
    with c_wow1:
        md("<div class='panel' style='padding:20px 24px 6px 24px;'>")
        md("<p class='eyebrow'>La forma del rischio di oggi</p>")
        nomi_radar = list(componenti_oggi.keys())
        valori_radar = [componenti_oggi[k] for k in nomi_radar] + [componenti_oggi[nomi_radar[0]]]
        categorie_radar = nomi_radar + [nomi_radar[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=valori_radar, theta=categorie_radar, fill='toself',
            line=dict(color=col, width=2), fillcolor=f"{col}33",
            marker=dict(size=7, color=col),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=PANEL_BG,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=PANEL_BD, linecolor=PANEL_BD, tickfont=dict(color=TXT_TERTIARY, size=9)),
                angularaxis=dict(gridcolor=PANEL_BD, linecolor=PANEL_BD, tickfont=dict(color=TXT_SECONDARY, size=10)),
            ),
            paper_bgcolor=PANEL_BG, showlegend=False,
            margin=dict(l=70, r=70, t=20, b=20), height=320,
            font=dict(color=TXT_SECONDARY, family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config=config_pulita)
        md(f"<div class='chart-caption' style='padding-bottom:16px;'>Più il poligono si allarga verso i bordi, più quel segnale spinge verso il rischio. Una forma piccola e compatta vicino al centro è il segnale migliore.</div>")
        md("</div>")
    with c_wow2:
        valori_componenti = list(componenti_oggi.values())
        std_componenti = float(np.std(valori_componenti))
        concordanza_pct = max(0, min(100, 100 - std_componenti * 1.8))
        gauge_concordanza_svg = disegna_gauge_circolare(concordanza_pct, col, size=210, label="CONCORDANZA %")
        md("<div class='panel' style='padding:20px 24px; text-align:center;'>")
        md("<p class='eyebrow' style='text-align:left;'>Quanto sono d'accordo i modelli</p>")
        embed_svg(gauge_concordanza_svg, height=230, extra_padding=6)
        if concordanza_pct >= 75:
            conc_txt = "Alta concordanza: i 5 segnali raccontano più o meno la stessa storia. Puoi fidarti del verdetto di oggi."
        elif concordanza_pct >= 50:
            conc_txt = "Concordanza media: qualche segnale si discosta dagli altri. Vale la pena leggere il dettaglio qui sotto prima di decidere."
        else:
            conc_txt = "Bassa concordanza: i modelli non sono allineati oggi. Meglio essere prudenti e dare più peso al buon senso."
        md(f"<div class='chart-caption' style='text-align:left;'>{conc_txt}</div>")
        md("</div>")

    md("<div style='height:24px;'></div>")

    # ---- Breakdown dei contributi al punteggio finale ----
    c_ml1, c_ml2 = st.columns(2)

    nomi_comp = list(componenti_oggi.keys())
    contributi_pesati = [componenti_oggi[k] * pesi_map[k] for k in nomi_comp]
    fig_contrib = go.Figure(go.Bar(
        y=nomi_comp, x=contributi_pesati, orientation='h',
        marker_color=[colori_comp[k] for k in nomi_comp],
        text=[f"{v:.1f} pt" for v in contributi_pesati], textposition='outside',
        textfont=dict(color=TXT_SECONDARY, size=10),
    ))
    fig_contrib.update_layout(**layout_base, xaxis_title="Punti sul totale (0-100)")
    fig_contrib.update_layout(margin=dict(l=150, r=40, t=10, b=32))
    fattore_dominante = nomi_comp[contributi_pesati.index(max(contributi_pesati))]
    contrib_txt = f"Il tuo punteggio di rischio di oggi (<strong>{risk_score:.0f}%</strong>) è la somma pesata di questi 5 segnali. Quello che pesa di più oggi è: <strong>{fattore_dominante}</strong>."
    chart_card(c_ml1, "Da dove nasce il punteggio di oggi", fig_contrib, contrib_txt, C_AMBRA)

    # ---- Feature importance Random Forest ----
    importanze = sorted(zip(FEATURE_LABELS_CLASS, class_bundle["rf_model"].feature_importances_), key=lambda t: t[1])
    fig_imp = go.Figure(go.Bar(
        y=[i[0] for i in importanze], x=[i[1] * 100 for i in importanze], orientation='h',
        marker_color=C_RPE, text=[f"{i[1]*100:.0f}%" for i in importanze], textposition='outside',
        textfont=dict(color=TXT_SECONDARY, size=10),
    ))
    fig_imp.update_layout(**layout_base, xaxis_title="Importanza relativa")
    fig_imp.update_layout(margin=dict(l=100, r=40, t=10, b=32))
    imp_txt = f"Mostra quali fattori la Random Forest guarda di più per decidere se un giorno è a rischio. Nel tuo caso di oggi, il fattore più determinante è: <strong>{fattore_ml}</strong>."
    chart_card(c_ml2, "Cosa guarda di più la Random Forest", fig_imp, imp_txt, C_RPE)

    md("<div style='height:24px;'></div>")

    c_ml3, c_ml4 = st.columns(2)

    # ---- Coefficienti Logistic Regression ----
    coefs = sorted(zip(FEATURE_LABELS_CLASS, class_bundle["log_model"].coef_[0]), key=lambda t: t[1])
    fig_log = go.Figure(go.Bar(
        y=[c[0] for c in coefs], x=[c[1] for c in coefs], orientation='h',
        marker_color=[C_STRESS if c[1] > 0 else C_RPE for c in coefs],
        text=[f"{c[1]:+.2f}" for c in coefs], textposition='outside',
        textfont=dict(color=TXT_SECONDARY, size=10),
    ))
    fig_log.add_vline(x=0, line_color=PANEL_BD_H, line_width=1)
    fig_log.update_layout(**layout_base, xaxis_title="Coefficiente (verde = riduce rischio, rosso = aumenta)")
    fig_log.update_layout(margin=dict(l=100, r=40, t=10, b=32))
    log_txt = "La Logistic Regression assegna un peso fisso a ogni fattore: barra rossa verso destra vuol dire che quel fattore, quando sale, aumenta il rischio; barra verde vuol dire che lo riduce. A differenza della Random Forest, qui la relazione è sempre lineare e diretta."
    chart_card(c_ml3, "Come pesa i fattori la Logistic Regression", fig_log, log_txt, C_SONNO)

    # ---- Cluster storico ----
    rischio_cluster_series = cluster_bundle["rischio_per_cluster"]
    cluster_oggi_id = analisi_ml["cluster_oggi"] - 1
    fig_clust = go.Figure(go.Bar(
        x=[f"Profilo {int(cid)+1}" for cid in rischio_cluster_series.index], y=rischio_cluster_series.values,
        marker_color=[C_VIOLA if int(cid) == cluster_oggi_id else TXT_TERTIARY for cid in rischio_cluster_series.index],
        text=[f"{v:.0f}%" for v in rischio_cluster_series.values], textposition='outside',
        textfont=dict(color=TXT_SECONDARY, size=10),
    ))
    fig_clust.update_layout(**layout_base, yaxis_title="% giorni a rischio (storico)")
    rischio_cluster_oggi_pct = float(rischio_cluster_series.get(cluster_oggi_id, 0.0))
    clust_txt = f"Il K-Means ha raggruppato i tuoi allenamenti passati in {len(rischio_cluster_series)} profili simili per distanza e frequenza cardiaca. L'allenamento previsto per oggi assomiglia al <strong>Profilo {analisi_ml['cluster_oggi']}</strong> (evidenziato in viola), che storicamente ha portato a un giorno a rischio nel <strong>{rischio_cluster_oggi_pct:.0f}%</strong> dei casi."
    chart_card(c_ml4, "In quale 'famiglia' di allenamenti rientra oggi", fig_clust, clust_txt, C_VIOLA)

    md("<div style='height:24px;'></div>")

    # ---- Storico onesto: come avrebbe giudicato la Random Forest ogni giorno passato ----
    # y_proba_rf viene da cross_val_predict: per ogni giorno storico, la previsione
    # arriva da un modello che NON ha mai visto quel giorno in fase di addestramento.
    # È quindi una stima onesta, non "letta con il senno di poi".
    df_rf_hist = df_base.copy()
    df_rf_hist['Rischio_RF_Storico'] = class_bundle["y_proba_rf"] * 100
    if date_col:
        df_rf_hist['Data_Chart'] = pd.to_datetime(df_rf_hist[date_col], errors='coerce')
    else:
        df_rf_hist['Data_Chart'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_rf_hist))
    df_rf_hist = df_rf_hist.sort_values('Data_Chart').dropna(subset=['Data_Chart']).tail(60)

    fig_rf_hist = go.Figure()
    fig_rf_hist.add_hrect(y0=0, y1=25, fillcolor=f"{C_RPE}1A", line_width=0, layer="below")
    fig_rf_hist.add_hrect(y0=25, y1=60, fillcolor=f"{C_AMBRA}1A", line_width=0, layer="below")
    fig_rf_hist.add_hrect(y0=60, y1=100, fillcolor=f"{C_STRESS}1A", line_width=0, layer="below")
    fig_rf_hist.add_trace(go.Scatter(
        x=df_rf_hist['Data_Chart'], y=df_rf_hist['Rischio_RF_Storico'], mode='lines',
        line=dict(color=TXT_PRIMARY, width=2, shape='spline'), fill='tozeroy',
        fillcolor='rgba(248,249,250,0.06)',
    ))
    fig_rf_hist.add_hline(
        y=rischio_ml, line_dash="dash", line_color=col, line_width=2,
        annotation_text=f"Oggi: {rischio_ml:.0f}%", annotation_font_color=col,
        annotation_position="top left", annotation_font_size=12,
    )
    fig_rf_hist.update_layout(**layout_base, yaxis_title="Rischio RF stimato (%)", yaxis=dict(range=[0, 100]))
    rf_hist_txt = f"Questa è la stima di rischio della Random Forest ricalcolata onestamente su ogni giorno del tuo storico (validazione incrociata: il modello non ha mai 'sbirciato' quel giorno durante l'addestramento). La linea tratteggiata mostra dove ti trovi oggi ({rischio_ml:.0f}%) rispetto a tutto il tuo storico recente."
    chart_card(st.container(), "Come avrebbe giudicato la Random Forest ogni giorno passato", fig_rf_hist, rf_hist_txt, TXT_PRIMARY)

    # ---- Linear Regression: FC reale vs prevista ----
    df_reg_chart = df_base.copy()
    df_reg_chart['FC_Prevista'] = reg_bundle["fc_predetta"]
    if date_col:
        df_reg_chart['Data_Chart'] = pd.to_datetime(df_reg_chart[date_col], errors='coerce')
    else:
        df_reg_chart['Data_Chart'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_reg_chart))
    df_reg_chart = df_reg_chart.sort_values('Data_Chart').dropna(subset=['Data_Chart']).tail(60)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=df_reg_chart['Data_Chart'], y=df_reg_chart['FC Media'], mode='lines',
        line=dict(color=C_STRESS, width=2), name="FC Reale"
    ))
    fig_reg.add_trace(go.Scatter(
        x=df_reg_chart['Data_Chart'], y=df_reg_chart['FC_Prevista'], mode='lines',
        line=dict(color=TXT_TERTIARY, width=2, dash='dot'), name="FC Prevista"
    ))
    fig_reg.update_layout(**layout_base)
    fig_reg.update_layout(
        yaxis_title="Battiti al minuto", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10))
    )
    residuo_oggi = analisi_ml["residuo_recente_bpm"]
    if residuo_oggi > 3:
        reg_txt = f"Negli ultimi 14 giorni il tuo cuore ha battuto in media <strong>{residuo_oggi:+.1f} bpm</strong> più del previsto rispetto a ritmo e meteo. È un segnale di affaticamento che i numeri esterni da soli non spiegano: vale la pena tenerlo d'occhio."
    elif residuo_oggi < -3:
        reg_txt = f"Negli ultimi 14 giorni il tuo cuore ha battuto in media <strong>{residuo_oggi:+.1f} bpm</strong> meno del previsto: probabilmente stai migliorando la tua efficienza cardiovascolare."
    else:
        reg_txt = f"Negli ultimi 14 giorni la differenza tra battito reale e previsto è stata di <strong>{residuo_oggi:+.1f} bpm</strong>, nella norma. Il tuo cuore risponde come atteso a ritmo e condizioni esterne."
    chart_card(st.container(), "FC reale vs FC prevista (Linear Regression)", fig_reg, reg_txt, C_AMBRA)

    # ---- Affidabilità dei modelli ----
    md("<div style='height:10px;'></div>")
    metr_rf = class_bundle["metriche"]["rf"]
    metr_log = class_bundle["metriche"]["log"]
    md(f"""
    <div class='panel' style='padding:24px;'>
        <p class='eyebrow' style='margin-bottom:16px;'>Quanto ci si può fidare dei modelli</p>
        <div class='hud-grid'>
            <div class='hud-stat'>
                <h2 style='color:{C_RPE};'>{metr_rf['auc']:.2f}</h2>
                <div class='mini-caption' style='margin:4px 0 0 0;'>AUC Random Forest<br>(1.0 = perfetto, 0.5 = a caso)</div>
            </div>
            <div class='hud-stat'>
                <h2 style='color:{C_SONNO};'>{metr_log['auc']:.2f}</h2>
                <div class='mini-caption' style='margin:4px 0 0 0;'>AUC Logistic Regression</div>
            </div>
            <div class='hud-stat'>
                <h2 style='color:{C_AMBRA};'>{reg_bundle['r2']:.2f}</h2>
                <div class='mini-caption' style='margin:4px 0 0 0;'>R² Linear Regression<br>(quanto bene predice la FC)</div>
            </div>
            <div class='hud-stat'>
                <h2 style='color:{C_VIOLA};'>{cluster_bundle['silhouette']:.2f}</h2>
                <div class='mini-caption' style='margin:4px 0 0 0;'>Silhouette K-Means<br>(quanto sono distinti i profili)</div>
            </div>
        </div>
        <div class='chart-caption' style='margin-top:16px;'>
            Questi numeri misurano quanto sono affidabili i modelli sul tuo storico, non la previsione di oggi in sé stessa. Più giorni alleni e registri, più questi indicatori tendono a migliorare.
        </div>
    </div>
    """)

    md("<div style='height:34px;'></div>")

    # =========================================================
    # SEZIONE 1: DINAMICHE AVANZATE E CARICO 
    # =========================================================
    section_head("Analisi Avanzata", "Come sta cambiando il tuo carico", "Uniamo i dati storici a modelli usati per prevedere il rischio infortunio (metodo ACWR).")
        
    c_adv1, c_adv2, c_adv3 = st.columns(3)

    # 1. MATRICE DI PRONTEZZA
    fig_matrix = go.Figure()
    fig_matrix.add_trace(go.Scatter(
        x=df_adv['Ore Sonno'], y=df_adv['Stress Lavoro'], mode='markers',
        marker=dict(color=TXT_TERTIARY, size=6, opacity=0.4), hoverinfo='skip'
    ))
    fig_matrix.add_trace(go.Scatter(
        x=[ore_s], y=[r.get('stress_lavoro', 5)], mode='markers+text',
        marker=dict(color=col, size=15, symbol='diamond', line=dict(width=2, color=TXT_PRIMARY)),
        text=["OGGI"], textposition="top center",
        textfont=dict(color=TXT_PRIMARY, size=11, family="JetBrains Mono, monospace"),
        name="Oggi"
    ))
    fig_matrix.add_hline(y=media_stress_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
    fig_matrix.add_vline(x=media_sonno_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
    
    fig_matrix.update_layout(**layout_base, xaxis_title="Ore di Sonno", yaxis_title="Stress Lavoro", xaxis=dict(range=[4, 10]), yaxis=dict(range=[0, 10]))
    
    stress_oggi = r.get('stress_lavoro', 5)
    if ore_s >= media_sonno_90 and stress_oggi <= media_stress_90:
        quad_txt = badge(C_RPE, "<strong>Situazione ottimale:</strong> hai dormito bene e sei poco stressato. Il corpo è pronto per un allenamento anche impegnativo.")
    elif ore_s >= media_sonno_90 and stress_oggi > media_stress_90:
        quad_txt = badge(C_AMBRA, "<strong>Attenzione:</strong> dormi abbastanza, ma lo stress da lavoro è alto. Meglio non esagerare con l'intensità oggi.")
    elif ore_s < media_sonno_90 and stress_oggi <= media_stress_90:
        quad_txt = badge(C_ARANCIO, "<strong>Recupero parziale:</strong> sei poco stressato ma hai dormito poco. I muscoli non sono al 100%: meglio un allenamento più leggero.")
    else:
        quad_txt = badge(C_STRESS, "<strong>Situazione critica:</strong> poco sonno e molto stress insieme. Il rischio infortunio è alto: oggi conviene riposare o solo camminare.")
        
    chart_card(c_adv1, "Sonno vs Stress: dove sei oggi", fig_matrix, quad_txt, col)

    # 2. ACUTE TO CHRONIC WORKLOAD RATIO (ACWR Proxy)
    if 'RPE' in df_adv.columns:
        df_adv['RPE_7'] = df_adv['RPE'].rolling(7, min_periods=1).mean()
        df_adv['RPE_28'] = df_adv['RPE'].rolling(28, min_periods=1).mean()
        df_adv['ACWR'] = df_adv['RPE_7'] / df_adv['RPE_28'].replace(0, 0.1)
        
        fig_acwr = go.Figure()
        fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(48,209,88,0.1)", opacity=1, layer="below", line_width=0)
        fig_acwr.add_trace(go.Scatter(
            x=df_adv['Data_Chart'].tail(60), y=df_adv['ACWR'].tail(60), mode='lines',
            line=dict(color=C_AMBRA, width=2, shape='spline')
        ))
        fig_acwr.add_hline(y=1.3, line_dash="dash", line_color=C_STRESS, line_width=1)

        acwr_attuale = df_adv['ACWR'].iloc[-1] if not df_adv['ACWR'].empty else 1.0
        ultima_data = df_adv['Data_Chart'].tail(60).iloc[-1] if not df_adv['Data_Chart'].tail(60).empty else None
        if ultima_data is not None:
            fig_acwr.add_trace(go.Scatter(
                x=[ultima_data], y=[acwr_attuale], mode='markers+text',
                marker=dict(color=TXT_PRIMARY, size=10, line=dict(width=2, color=C_AMBRA)),
                text=[f"{acwr_attuale:.2f}"], textposition="top center",
                textfont=dict(color=TXT_PRIMARY, size=11, family="JetBrains Mono, monospace"),
                hoverinfo='skip'
            ))

        fig_acwr.update_layout(**layout_base, yaxis_title="Rapporto Fatica", yaxis=dict(range=[0.5, 2.0]))
        
        if acwr_attuale > 1.3:
            acwr_txt = badge(C_STRESS, "Sei sopra la zona verde: <strong>ti stai affaticando troppo in fretta</strong> rispetto al mese scorso. Rallenta, o rischi un infortunio da sovraccarico (es. tendinite).")
        elif acwr_attuale < 0.8:
            acwr_txt = badge(C_SONNO, "Sei sotto la zona verde: ti stai allenando meno o più piano del solito. Se continua così, rischi di perdere un po' di forma.")
        else:
            acwr_txt = badge(C_RPE, "Perfetto: sei dentro la zona verde. Stai aumentando (o mantenendo) la fatica in modo <strong>giusto e graduale</strong>.")
            
        chart_card(c_adv2, "Carico recente vs mese scorso", fig_acwr, acwr_txt, C_AMBRA)
    else:
        c_adv2.warning("Dati di fatica insufficienti per questo grafico.")

    # 3. PATTERN SETTIMANALE DELLO STRESS
    if 'Stress Lavoro' in df_adv.columns:
        df_adv['Giorno'] = df_adv['Data_Chart'].dt.dayofweek
        giorni_map = {0:'Lun', 1:'Mar', 2:'Mer', 3:'Gio', 4:'Ven', 5:'Sab', 6:'Dom'}
        
        stress_giornaliero = df_adv.groupby('Giorno')['Stress Lavoro'].mean().reset_index()
        stress_giornaliero['Nome_Giorno'] = stress_giornaliero['Giorno'].map(giorni_map)
        
        if not stress_giornaliero.empty:
            fig_week = go.Figure(go.Bar(
                x=stress_giornaliero['Nome_Giorno'], y=stress_giornaliero['Stress Lavoro'],
                marker_color=TXT_TERTIARY, text=stress_giornaliero['Stress Lavoro'].round(1),
                textposition='outside', textfont=dict(color=TXT_SECONDARY, size=10)
            ))
            giorno_max = stress_giornaliero.loc[stress_giornaliero['Stress Lavoro'].idxmax()]
            fig_week.add_trace(go.Bar(
                x=[giorno_max['Nome_Giorno']], y=[giorno_max['Stress Lavoro']],
                marker_color=C_STRESS, text=[giorno_max['Stress Lavoro'].round(1)],
                textposition='outside', textfont=dict(color=C_STRESS, size=10)
            ))
            
            fig_week.update_layout(**layout_base, yaxis=dict(range=[0, 10]), xaxis_title="Giorno Settimana", barmode='overlay')
            
            adv_week = f"Guardando la tua storia, il <strong>{giorno_max['Nome_Giorno']}</strong> è di solito il giorno in cui accumuli più stress mentale. Prova a tenerlo per il riposo o per corse molto leggere."
            chart_card(c_adv3, "Stress medio per giorno", fig_week, adv_week, C_STRESS)
        else:
            c_adv3.warning("Dati storici insufficienti.")
    else:
        c_adv3.warning("Colonna Stress Lavoro non trovata.")

    md("<div style='height:34px;'></div>")

    # =========================================================
    # SEZIONE 2: TREND STORICI
    # =========================================================
    section_head("Trend Storici", "Come cambiano le tue abitudini", "Gli ultimi 3 mesi di sonno, stress e fatica, con l'ultimo valore evidenziato.")

    df_plot = df_adv.tail(90)

    def calcola_trend(serie):
        if len(serie) < 15: return 0
        recente = serie.tail(14).mean()
        precedente = serie.head(len(serie) - 14).mean()
        return recente - precedente

    def aggiungi_punto_finale(fig, df_sorgente, colonna, colore):
        if df_sorgente.empty:
            return
        ultimo = df_sorgente.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[ultimo['Data_Chart']], y=[ultimo[colonna]], mode='markers+text',
            marker=dict(color=TXT_PRIMARY, size=9, line=dict(width=2, color=colore)),
            text=[f"{ultimo[colonna]:.1f}"], textposition="top center",
            textfont=dict(color=TXT_PRIMARY, size=11, family="JetBrains Mono, monospace"),
            hoverinfo='skip'
        ))

    r1c1, r1c2, r1c3 = st.columns(3)

    if 'Ore Sonno' in df_plot.columns:
        trend_sonno = calcola_trend(df_plot['Ore Sonno'])
        fig_t1 = go.Figure(go.Scatter(
            x=df_plot['Data_Chart'], y=df_plot['Ore Sonno'], mode='lines',
            line=dict(color=C_SONNO, width=2), fill='tozeroy', fillcolor='rgba(46,144,255,0.08)'
        ))
        aggiungi_punto_finale(fig_t1, df_plot, 'Ore Sonno', C_SONNO)
        fig_t1.update_layout(**layout_base, yaxis_title="Ore a notte")
        spieg_sonno = badge(C_AMBRA, "Attenzione: la linea scende. Ultimamente dormi meno del solito. Prova ad andare a letto un po' prima per far recuperare i muscoli.") if trend_sonno < -0.3 else badge(C_RPE, "Bene: le tue ore di sonno sono costanti. Stai dando al corpo il tempo giusto per ricaricarsi.")
        chart_card(r1c1, "Andamento del Sonno", fig_t1, spieg_sonno, C_SONNO)

    if 'Stress Lavoro' in df_plot.columns:
        trend_stress = calcola_trend(df_plot['Stress Lavoro'])
        fig_t2 = go.Figure(go.Scatter(
            x=df_plot['Data_Chart'], y=df_plot['Stress Lavoro'], mode='lines',
            line=dict(color=C_STRESS, width=2), fill='tozeroy', fillcolor='rgba(255,69,58,0.08)'
        ))
        aggiungi_punto_finale(fig_t2, df_plot, 'Stress Lavoro', C_STRESS)
        fig_t2.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="Livello Stress (0-10)")
        spieg_stress = badge(C_AMBRA, "La linea sale: il tuo stress generale sta aumentando. Quando la mente è stanca, il corpo si infortuna più facilmente: abbassa l'intensità della corsa.") if trend_stress > 0.5 else badge(C_RPE, "Il tuo stress da lavoro e vita quotidiana è stabile e sotto controllo.")
        chart_card(r1c2, "Andamento dello Stress", fig_t2, spieg_stress, C_STRESS)

    if 'RPE' in df_plot.columns:
        trend_rpe = calcola_trend(df_plot['RPE'])
        fig_t3 = go.Figure(go.Scatter(
            x=df_plot['Data_Chart'], y=df_plot['RPE'], mode='lines',
            line=dict(color=C_RPE, width=2), fill='tozeroy', fillcolor='rgba(48,209,88,0.08)'
        ))
        aggiungi_punto_finale(fig_t3, df_plot, 'RPE', C_RPE)
        fig_t3.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="Fatica Percepita (0-10)")
        spieg_rpe = badge(C_AMBRA, "La linea sale: fai più fatica del solito negli allenamenti. È il segnale che serve scaricare: fai un paio di giorni leggeri.") if trend_rpe > 0.5 else badge(C_RPE, "La fatica che senti dopo gli allenamenti è costante. Il corpo gestisce bene i chilometri.")
        chart_card(r1c3, "Andamento della Fatica", fig_t3, spieg_rpe, C_RPE)

    md("<div style='height:34px;'></div>")

    # =========================================================
    # EXPORT REPORT
    # =========================================================
    section_head("Export", "Porta il report dal tuo coach", "Scarica l'analisi completa in formato testo o come pagina HTML con grafici e design.")

    coach_txt = ""
    for nome_tab, contenuto in coach_content.items():
        coach_txt += f"\n[{nome_tab.upper()}]\n"
        for label, bullets in contenuto["blocchi"]:
            coach_txt += f"  {label}:\n"
            for b in bullets:
                coach_txt += f"    - {b}\n"

    import re as _re
    def _plain(html_txt):
        # Rimuove qualsiasi tag HTML (badge, strong, br, span...) per il
        # report testuale, così restano solo le parole, senza codice visibile.
        senza_tag = _re.sub(r'<[^>]+>', '', html_txt)
        return senza_tag.replace('&nbsp;', ' ').strip()

    grafici_txt = "\n".join(f"  - {t}: {_plain(s)}" for t, s in insights_export)

    report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (Target Originale: {distanza_target} km)
Indice Rischio: {risk_score:.0f}%
Recovery Score: {recovery_score:.0f}%
Stress Mentale (SMA): {sma:.1f}

VERDETTO: {_plain(hero_msg)}
{_plain(nota_coerenza)}

NOTE CLINICHE E ANALISI AVANZATA:
{grafici_txt}

PROTOCOLLO COACH COMPLETO{coach_txt}
--------------------------------"""

    colb1, colb2 = st.columns(2)
    with colb1:
        st.download_button("Scarica TXT", data=report_testo, file_name="runai_report.txt", mime="text/plain", use_container_width=True)

    with colb2:
        # Ogni grafico esportato porta con sé il proprio titolo e la propria
        # spiegazione (prese da insights_export, popolata in parallelo a
        # figs_per_export dentro chart_card): senza questo, il file scaricato
        # mostrava solo i grafici nudi, senza alcuna scritta di contesto.
        charts_html = ""
        for i, (fig, (titolo_c, spiegazione_c)) in enumerate(zip(figs_per_export, insights_export)):
            include_js = 'cdn' if i == 0 else False
            grafico_html = fig.to_html(full_html=False, include_plotlyjs=include_js, config={'displayModeBar': False})
            charts_html += f"""
            <div class='chart-block'>
                <p class='chart-block-title'>{titolo_c}</p>
                {grafico_html}
                <div class='chart-block-caption'><strong>Cosa significa questo grafico?</strong><br>{spiegazione_c}</div>
            </div>
            """

        coach_html = ""
        for nome_tab, contenuto in coach_content.items():
            blocchi_html = ""
            for label, bullets in contenuto["blocchi"]:
                bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                blocchi_html += f"<div class='coach-block' style='--block-color:{contenuto['colore']};'><div class='label'>{label}</div><ul>{bullets_html}</ul></div>"
            coach_html += f"<div class='panel' style='margin-bottom:14px;'><h3 style='margin-bottom:14px;'>{nome_tab}</h3>{blocchi_html}</div>"

        verdetti_html = "".join(f"""
        <div class='verdict-chip'>
            <div class='vc-name'>{nome}</div>
            <div class='vc-badge' style='color:{verdetti[nome][1]}; background:{verdetti[nome][1]}1A;'>{verdetti[nome][0]}</div>
        </div>
        """ for nome in componenti_oggi.keys())

        report_html_completo = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>RunAI Performance Report — {tit}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');
  * {{ box-sizing: border-box; }}
  body {{ background:#0A0E15; color:{TXT_SECONDARY}; font-family: Inter, sans-serif; padding: 36px; max-width:1100px; margin:0 auto; line-height:1.6; }}
  h1 {{ color:{col}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.7em; margin-bottom:4px; }}
  h2 {{ color:{TXT_PRIMARY}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.15em; margin:34px 0 14px 0; border-bottom:1px solid {PANEL_BD}; padding-bottom:10px; }}
  h3 {{ color:{TXT_PRIMARY}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1em; }}
  p {{ margin: 0 0 10px 0; }}
  .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7em; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 8px 0; font-weight:600; }}
  .hero-msg {{ font-size:1.02em; color:{TXT_SECONDARY}; max-width:720px; margin-bottom:8px; }}
  .coerenza-note {{ font-size:.85em; color:{TXT_TERTIARY}; margin-bottom:20px; }}
  .panel {{ background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:14px; padding:20px 22px; margin-bottom:14px; }}
  .hero-row {{ display:flex; align-items:center; gap:28px; flex-wrap:wrap; margin: 14px 0 6px 0; }}
  .hero-row svg {{ max-width: 260px; height:auto; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
  .kpi-row .panel {{ flex:1 1 30%; min-width:200px; }}
  .kpi-row .val {{ font-family:'JetBrains Mono',monospace; font-size:1.7em; color:{TXT_PRIMARY}; font-weight:600; }}
  .coach-block {{ margin-bottom:14px; border-left:3px solid var(--block-color); padding-left:14px; }}
  .coach-block .label {{ font-family:'Oswald',sans-serif; font-size:.85em; letter-spacing:.05em; text-transform:uppercase; margin-bottom:8px; font-weight:600; color:var(--block-color); }}
  .coach-block ul {{ margin:0; padding-left:18px; }}
  .coach-block li {{ margin-bottom:6px; font-size:.92em; }}
  .verdict-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
  .verdict-chip {{ flex:1; min-width:130px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:10px; padding:12px; text-align:center; }}
  .verdict-chip .vc-name {{ font-size:.72em; color:{TXT_SECONDARY}; margin-bottom:8px; }}
  .verdict-chip .vc-badge {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.78em; padding:4px 10px; border-radius:6px; }}
  .charts-grid {{ display:flex; flex-wrap:wrap; gap:18px; }}
  .chart-block {{ flex: 1 1 44%; min-width:320px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:12px; padding:14px; }}
  .chart-block-title {{ font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; letter-spacing:.02em; color:{TXT_PRIMARY}; font-size:.95em; margin-bottom:6px; }}
  .chart-block-caption {{ border-top:1px solid {PANEL_BD}; margin-top:10px; padding-top:10px; font-size:.85em; color:{TXT_SECONDARY}; }}
  .status-badge {{ display:inline-flex; align-items:center; }}
  .status-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
  footer {{ margin-top:40px; padding-top:16px; border-top:1px solid {PANEL_BD}; font-size:.75em; color:{TXT_TERTIARY}; }}
</style>
</head>
<body>
  <p class="eyebrow">RunAI Performance Report — Generato il {pd.Timestamp.today().strftime('%d/%m/%Y')}</p>
  <h1>{tit}</h1>
  <p class="hero-msg">{hero_msg}</p>
  <p class="coerenza-note">Rischio da modello ML: {rischio_ml:.0f}% · Rischio da regole cliniche: {risk_score_euristico:.0f}% · {nota_coerenza}</p>
  <div class="hero-row">{gauge_svg}{radar_svg}</div>
  <div class="kpi-row">
    <div class="panel"><p class="eyebrow">Distanza Consigliata</p><div class="val">{distanza_consigliata:.1f} km</div><p style="font-size:.8em; margin-top:6px;">Piano originale: {distanza_target} km</p></div>
    <div class="panel"><p class="eyebrow">Indice Rischio</p><div class="val" style="color:{col};">{risk_score:.0f}%</div><p style="font-size:.8em; margin-top:6px;">Somma pesata di 4 modelli ML più le regole cliniche</p></div>
    <div class="panel"><p class="eyebrow">Recovery Score</p><div class="val" style="color:{C_SONNO};">{recovery_score:.0f}%</div><p style="font-size:.8em; margin-top:6px;">Basato su {ore_s:.1f}h di sonno (ideale: 7.5h)</p></div>
  </div>

  <h2>Il consiglio degli algoritmi</h2>
  <p>{n_maggioranza} modelli su 5 dicono <strong style="color:{colore_maggioranza};">{verdetto_maggioranza}</strong>. Ogni modello vota in modo indipendente sul rischio di oggi:</p>
  <div class="verdict-row">{verdetti_html}</div>

  <h2>Protocollo coach completo</h2>{coach_html}
  <h2>Grafici analitici</h2>
  <p style="font-size:.88em;">Ogni grafico qui sotto è accompagnato dalla spiegazione mostrata nella dashboard, così il report resta comprensibile anche senza aprire l'app.</p>
  <div class="charts-grid">{charts_html}</div>
  <footer>Report generato automaticamente da RunAI. I punteggi si basano sui dati inseriti e sui modelli allenati sul tuo storico personale.</footer>
</body>
</html>"""

        st.download_button("Scarica HTML (Grafici e Design)", data=report_html_completo, file_name="runai_report_completo.html", mime="text/html", use_container_width=True)
