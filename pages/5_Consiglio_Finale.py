from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_PLAN
st.set_page_config(page_title="Consiglio Finale", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_PLAN = get_svg_url(SVG_PLAN)

elif pagina == "CONSIGLIO FINALE":
    def md(html):
        """Renderizza HTML in modo sicuro. Rimuove l'indentazione python."""
        html_pulito = "\n".join([line.strip() for line in html.split("\n")])
        st.markdown(html_pulito, unsafe_allow_html=True)

    header_block(
        "Modulo 05 — Action Plan",
        "CONSIGLIO FINALE",
        "Protocollo operativo, proiezioni fisiologiche ed export report per la sessione odierna.",
        IMG_HERO_PLAN, "Coach Protocol"
    )

    if not st.session_state.get('analisi_fatta', False):
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        df_base = st.session_state.dati.copy()

        # =========================================================
        # TOKEN DI DESIGN (High-Tech Sports Theme)
        # =========================================================
        PANEL_BG   = "#0D1117"
        PANEL_BD   = "#1E2633"
        PANEL_BD_H = "#2A3546"
        TXT_PRIMARY   = "#F8F9FA"
        TXT_SECONDARY = "#8B949E"
        TXT_TERTIARY  = "#485363"

        C_SONNO  = "#2E90FF"
        C_STRESS = "#FF453A"
        C_RPE    = "#30D158"
        C_AMBRA  = "#FF9F0A"
        C_VIOLA  = "#BF5AF2"
        C_NEUTRO = "#1F2733"

        md(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

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

        .kv-num {{ font-family:'JetBrains Mono',monospace; color:{TXT_PRIMARY}; font-weight:700; }}
        .badge {{
            display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.65rem;
            letter-spacing:.1em; text-transform:uppercase; padding:4px 10px; border-radius:4px; font-weight:700;
        }}

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

        .hud-grid {{ display: flex; gap: 20px; align-items: flex-end; margin-bottom: 20px; }}
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
        
        .chart-caption {{ border-top: 1px solid {PANEL_BD}; margin-top: 10px; padding-top: 10px; color:{TXT_SECONDARY}; font-family:'Inter',sans-serif; font-size:.85rem; line-height:1.55; }}
        
        .delta-track {{ position:relative; height:6px; border-radius:3px; background:{PANEL_BD}; margin:10px 0 4px 0; }}
        .delta-fill {{ position:absolute; top:0; height:6px; border-radius:3px; }}
        .delta-mid {{ position:absolute; top:-3px; left:50%; width:1px; height:12px; background:{TXT_TERTIARY}; }}
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
        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        distanza_target = r.get('distanza_oggi', 10.0)
        distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0

        if risk_score < 25:
            tit, col, liv = "AUTORIZZATO", C_RPE, "basso"
        elif risk_score < 60:
            tit, col, liv = "RECUPERO ATTIVO", C_AMBRA, "medio"
        else:
            tit, col, liv = "RIPOSO OBBLIGATORIO", C_STRESS, "alto"

        tipo_all = r.get('tipo_allenamento', 'Easy Run')
        cadenza_target = "170-180 spm" if liv != "alto" else "165-172 spm (passo rilassato per abbassare l'impatto articolare)"
        zona_consigliata = "Zona 2-3 (aerobico puro)" if liv == "basso" else "Zona 1-2 (sforzo percepito bassissimo)" if liv == "medio" else "Solo mobilità o camminata veloce"

        # RICERCA COLONNA DATA (Fallback intelligente)
        date_col = next((c for c in df_base.columns if c.lower() in ['data', 'date', 'giorno', 'time']), None)
        df_adv = df_base.copy()
        if date_col:
            df_adv['Data_Chart'] = pd.to_datetime(df_adv[date_col], errors='coerce')
        else:
            # Se non c'è una data, generiamo un asse fittizio per mostrare i grafici
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
            <svg viewBox="0 0 {svg_width} {svg_height + 25}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.3));">
                <line x1="0" y1="{svg_height}" x2="{svg_width}" y2="{svg_height}" stroke="{TXT_TERTIARY}" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
                {"".join(elements)}
                <g transform="translate({marker_x}, 0)">
                    <line x1="0" y1="0" x2="0" y2="{svg_height + 10}" stroke="{TXT_PRIMARY}" stroke-width="2" />
                    <polygon points="-6,{svg_height + 10} 6,{svg_height + 10} 0,{svg_height + 18}" fill="{TXT_PRIMARY}" />
                    <rect x="-24" y="-20" width="48" height="20" rx="4" fill="{TXT_PRIMARY}" />
                    <text x="0" y="-6" font-family="JetBrains Mono, monospace" font-size="12" font-weight="bold" fill="{PANEL_BG}" text-anchor="middle">{int(score)}%</text>
                </g>
                <text x="0" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_RPE}" opacity="0.8">OPTIMAL (0-25)</text>
                <text x="{svg_width/2}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_AMBRA}" text-anchor="middle" opacity="0.8">WARNING (26-59)</text>
                <text x="{svg_width}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_STRESS}" text-anchor="end" opacity="0.8">DANGER (60-100)</text>
            </svg>
            """

        radar_svg = disegna_telemetria_rischio(risk_score)

        md(f"""
        <div class='panel'>
            <div class='hud-grid'>
                <div class='hud-stat'>
                    <p class='eyebrow'>System Status</p>
                    <h2 style='color:{col};'>{tit}</h2>
                </div>
                <div class='hud-stat' style='text-align:right;'>
                    <p class='eyebrow'>Risk Load</p>
                    <h2 style='color:{TXT_PRIMARY};'>{risk_score:.0f}<span style='font-size:0.6em; color:{TXT_SECONDARY};'>%</span></h2>
                </div>
            </div>
            <div style='margin-top: 10px;'>
                {radar_svg}
            </div>
        </div>
        """)

        md("<div style='height:24px;'></div>")

        # =========================================================
        # SPLIT SHEET (Cronometraggio)
        # =========================================================
        sonno_delta_txt = f"{'+' if (r['ore_sonno']-7.5) >= 0 else ''}{r['ore_sonno']-7.5:.1f}h rispetto al target fisiologico ottimale"

        md(f"""
        <div class='panel split-sheet'>
            <div class='split-head'>
                <span>Metric</span><span>Value</span><span>Reference</span><span>Notes / Adjustments</span>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Distanza Target</div>
                <div class='sr-value' style='color:{TXT_PRIMARY};'>{distanza_consigliata:.1f}<span class='unit'>km</span></div>
                <div class='sr-ref'>Da piano originale: {distanza_target} km</div>
                <div class='sr-note'>La distanza è stata scalata algoritmicamente del {100 - (distanza_consigliata/distanza_target*100 if distanza_target>0 else 0):.0f}% per proteggere i tuoi tessuti.</div>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Recovery Score</div>
                <div class='sr-value' style='color:{C_SONNO};'>{recovery_score:.0f}<span class='unit'>%</span></div>
                <div class='sr-ref'>Qualità base: {r['ore_sonno']:.1f}h di sonno</div>
                <div class='sr-note'>{sonno_delta_txt}. Il sonno governa il recupero del sistema nervoso autonomo.</div>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Mental Load (SMA)</div>
                <div class='sr-value' style='color:{C_AMBRA};'>{sma:.1f}</div>
                <div class='sr-ref'>(Stress × RPE) / Sonno</div>
                <div class='sr-note'>Indice composito di sovraccarico del sistema nervoso centrale. Livello odierno: {liv.upper()}.</div>
            </div>
        </div>
        """)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # COACH PERSONALIZZATO - ESTESO CON PIÙ CONSIGLI
        # =========================================================
        section_head("Coach personalizzato", "Protocollo Operativo Dettagliato", "Linee guida biomeccaniche, di pacing e bio-hacking basate sulla telemetria odierna.")

        # Logica dei testi dinamici
        if liv == "basso":
            dinamica_pacing = "Mantenimento del ritmo standard. Nessuna restrizione fisiologica sui cambi di velocità, via libera a scatti se previsti."
            dinamica_resp = "Respirazione naturale (suggerito schema 3:3 in riscaldamento, passando a 2:2 a ritmo gara)."
            dinamica_rec = "Recupero passivo standard o massaggio leggero. Doccia contrasto (caldo/freddo) autorizzata post-workout."
        elif liv == "medio":
            dinamica_pacing = "Gestione conservativa (Pacing difensivo). Togli 10-15 secondi al km dal tuo ritmo abituale. Evita pendenze severe."
            dinamica_resp = "Forza un'espirazione lunga (schema 3:4) per mantenere bassa la frequenza cardiaca ed evitare picchi di sforzo."
            dinamica_rec = "Focus massimo su idratazione e reintegro elettrolitico. Niente acqua fredda post-corsa, preferisci temperature neutre per non stressare il SNC."
        else:
            dinamica_pacing = "Sforzo sconsigliato. Converti la seduta in 30-40 minuti di camminata dinamica su terreno piatto e morbido (prato/terra)."
            dinamica_resp = "Respirazione diaframmatica esclusiva naso-naso per favorire il recupero parasimpatico."
            dinamica_rec = "Yoga nidra o stretching passivo lungo. Applica calore (non freddo) su eventuali zone muscolari tese."

        coach_content = {
            "1. Warm-Up & Prep (Pre)": {
                "colore": C_SONNO,
                "blocchi": [
                    ("Attivazione Neurale e Meccanica", [
                        "Camminata sui talloni e sulle punte (30 secondi per tipo) per attivare l'arco plantare e le caviglie.",
                        "5 minuti di mobilità dinamica: leg swings frontali/laterali, rotazioni delle anche, affondi controllati.",
                        "Evita lo stretching statico pre-corsa: riduce la reattività elastica del tendine d'Achille del 5-8%."
                    ]),
                    ("Setup Mentale e Fisiologico", [
                        f"Target zona Iniziale: {zona_consigliata}. I primi 10 minuti devono sembrare 'troppo lenti'.",
                        "Esegui 10 cicli di respirazione profonda prima di far partire il cronometro per azzerare lo stress lavorativo.",
                        f"Direttiva Rischio: {dinamica_pacing}"
                    ])
                ],
            },
            "2. In Azione (Durante)": {
                "colore": C_AMBRA,
                "blocchi": [
                    ("Gestione Biomeccanica e Pacing", [
                        f"Frequenza passi (Cadenza): {cadenza_target}. Aumentare la frequenza del 5% riduce il carico sulle ginocchia del 20%.",
                        "Postura: Busto leggermente inclinato in avanti partendo dalle caviglie (non dalla vita). Sguardo a 20 metri, mai ai piedi.",
                        "Rilassa mani e mascella: la tensione nel volto si trasmette istantaneamente alla catena muscolare posteriore."
                    ]),
                    ("Gestione Sforzo e Sicurezza", [
                        f"Respirazione: {dinamica_resp}",
                        "Check Infortuni: Differenzia la 'fatica sorda' (normale) dal 'dolore acuto/fitta' (allarme articolare).",
                        "Regola dei 15 minuti: Se dopo 15 minuti l'RPE percepito è più alto di 2 punti rispetto al previsto, taglia la distanza a metà."
                    ])
                ],
            },
            "3. Protocollo di Recupero (Post)": {
                "colore": C_RPE,
                "blocchi": [
                    ("Cooldown Sistemico", [
                        "Non fermarti bruscamente. Cammina per 3-5 minuti finché il battito non scende comodamente in Zona 1 (Sotto i 110 bpm).",
                        "Finestra anabolica: assumi carboidrati e proteine leggere entro 45 minuti per arrestare il catabolismo indotto dallo stress."
                    ]),
                    ("Gestione Tessuti", [
                        "Stretching passivo: mantieni ogni posizione per almeno 45-60 secondi (polpacci, ischiocrurali, psoas).",
                        "Usa il foam roller lentamente. Se trovi un 'trigger point' (punto di dolore), fermati su di esso respirando per 30 secondi."
                    ])
                ],
            },
            "4. Bio-Hacking Serale (Mindset)": {
                "colore": C_VIOLA,
                "blocchi": [
                    ("Regolazione Sistema Nervoso", [
                        "Doccia / Termoterapia: " + dinamica_rec,
                        "NSDR (Non-Sleep Deep Rest): 10 minuti di meditazione guidata o body-scan prima di dormire se lo stress oggi era sopra il 7/10."
                    ]),
                    ("Ottimizzazione Sonno", [
                        f"Target di stanotte: recuperare il deficit arrivando a {max(7.5, r['ore_sonno']+0.5):.1f} ore.",
                        "Riduci l'esposizione alla luce blu intensa (smartphone/TV) 60 minuti prima del sonno per permettere il rilascio di melatonina naturale."
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
        # CORSIE E ZONE
        # =========================================================
        section_head("Riferimento", "Corsie di Frequenza Cardiaca")

        corsie = [
            ("Corsia 1", "Zona 1-2", "Recupero / Base Aerobica", "Sforzo bassissimo (test del parlato superato facilmente). L'energia deriva dai grassi. Ideale per costruire resistenza senza accumulare fatica.", C_RPE),
            ("Corsia 2", "Zona 3", "Soglia Aerobica / Tempo", "Ritmo sostenuto, respirazione più profonda. Accumulo minimo di acido lattico. Sviluppa la potenza cardiaca.", C_AMBRA),
            ("Corsia 3", "Zona 4-5", "Soglia Lattacida / VO2Max", "Sforzo massimale (test del parlato fallito). Distrugge le fibre muscolari per super-compensare. Da usare col contagocce se il rischio infortunio è medio/alto.", C_STRESS),
        ]
        cc1, cc2, cc3 = st.columns(3)
        for c, (num, zt, zn, zd, zcol) in zip([cc1, cc2, cc3], corsie):
            c.markdown(f"""
            <div class='lane-chip' style='--zc:{zcol};'>
                <div class='lane-num'>{num}</div>
                <div class='zt'>{zt}</div>
                <div class='zn'>{zn}</div>
                <div class='zd'>{zd}</div>
            </div>
            """, unsafe_allow_html=True)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # PREPARAZIONE GRAFICI E STILI COMUNI
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
        media_rpe_90 = df_base['RPE'].mean() if 'RPE' in df_base.columns else 5.0

        sonno_vs_media = r['ore_sonno'] - media_sonno_90
        stress_vs_media = r['stress_lavoro'] - media_stress_90
        rpe_vs_media = r['rpe_previsto'] - media_rpe_90

        figs_per_export = []
        insights_export = []

        def chart_card(container, titolo, fig, spiegazione, rule_color=C_NEUTRO):
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
            with container:
                md(f"""
                <div class='panel panel-flush'>
                    <div class='chart-top-rule' style='background:{rule_color};'></div>
                    <div class='panel-body'><p class='panel-title' style='color:{TXT_PRIMARY}; font-weight:600;'>{titolo}</p></div>
                """)
                st.plotly_chart(fig, use_container_width=True, config=config_pulita)
                md(f"""
                    <div class='panel-body' style='padding-top:0;'><div class='chart-caption'><strong>Insight Clinico:</strong><br>{spiegazione}</div></div>
                </div>
                """)
            figs_per_export.append(fig)
            insights_export.append((titolo, spiegazione))


        # =========================================================
        # SEZIONE 1: DINAMICHE AVANZATE E CARICO 
        # (Usa df_adv con data garantita)
        # =========================================================
        section_head("Analisi Avanzata", "Dinamiche di Carico", "Integrazione dei dati storici con modelli di readiness e previsione infortuni (Metodologia ACWR).")
            
        c_adv1, c_adv2, c_adv3 = st.columns(3)

        # --- 1. MATRICE DI PRONTEZZA ---
        fig_matrix = go.Figure()
        fig_matrix.add_trace(go.Scatter(
            x=df_adv['Ore Sonno'], y=df_adv['Stress Lavoro'], mode='markers',
            marker=dict(color=TXT_TERTIARY, size=6, opacity=0.4), hoverinfo='skip'
        ))
        fig_matrix.add_trace(go.Scatter(
            x=[r['ore_sonno']], y=[r['stress_lavoro']], mode='markers',
            marker=dict(color=col, size=14, symbol='diamond', line=dict(width=2, color=TXT_PRIMARY)),
            name="Oggi"
        ))
        fig_matrix.add_hline(y=media_stress_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
        fig_matrix.add_vline(x=media_sonno_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
        
        fig_matrix.update_layout(**layout_base, xaxis_title="Ore di Sonno", yaxis_title="Stress Lavoro", xaxis=dict(range=[4, 10]), yaxis=dict(range=[0, 10]))
        
        if r['ore_sonno'] >= media_sonno_90 and r['stress_lavoro'] <= media_stress_90:
            quad = "Ottimale (Alto recupero, Basso stress)."
            azione = "Sei nel quadrante d'oro. Il corpo è pronto per carichi pesanti e adattamenti strutturali."
        elif r['ore_sonno'] >= media_sonno_90 and r['stress_lavoro'] > media_stress_90:
            quad = "Compensato (Alto recupero, Alto stress)."
            azione = "Stai dormendo abbastanza per tamponare lo stress lavorativo, ma il sistema nervoso è sotto pressione. Non eccedere in Z4/Z5."
        elif r['ore_sonno'] < media_sonno_90 and r['stress_lavoro'] <= media_stress_90:
            quad = "Vulnerabile (Basso recupero, Basso stress)."
            azione = "Il livello di stress è OK, ma manca materia prima (sonno) per riparare i muscoli. Allenamento leggero consigliato."
        else:
            quad = "Critico (Basso recupero, Alto stress)."
            azione = "Zona rossa di infortunio. Cortisolo alto e recupero cellulare assente. Altamente consigliato riposo totale o sola mobilità."
            
        chart_card(c_adv1, "Matrice di Prontezza", fig_matrix, f"Stato: {quad}<br>{azione}", col)


        # --- 2. ACUTE TO CHRONIC WORKLOAD RATIO (ACWR Proxy) ---
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
            fig_acwr.update_layout(**layout_base, yaxis_title="Ratio (Acuto/Cronico)", yaxis=dict(range=[0.5, 2.0]))
            
            acwr_attuale = df_adv['ACWR'].iloc[-1] if not df_adv['ACWR'].empty else 1.0
            if acwr_attuale > 1.3:
                acwr_txt = "ATTENZIONE: Ratio sopra 1.3. Stai aumentando il carico (fatica) troppo in fretta rispetto alla tua abitudine mensile. Rischio tendiniti alle stelle. Usa i prossimi giorni per scaricare."
            elif acwr_attuale < 0.8:
                acwr_txt = "Ratio sotto 0.8. Sei in fase di de-training (scarico prolungato). Il tuo corpo sta perdendo adattamenti atletici. Ottimo pre-gara, ma se non hai gare, torna a spingere moderatamente."
            else:
                acwr_txt = "Ratio nella 'Sweet Spot' (0.8 - 1.3). Progression del carico perfettamente bilanciata. Stai costruendo fitness senza sovraccaricare le articolazioni."
                
            chart_card(c_adv2, "ACWR (Carico Acuto vs Cronico)", fig_acwr, acwr_txt, C_AMBRA)
        else:
            c_adv2.warning("Dati RPE insufficienti per calcolo ACWR.")


        # --- 3. PATTERN SETTIMANALE DELLO STRESS ---
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
                
                adv_week = f"Il {giorno_max['Nome_Giorno']} è mediamente la tua giornata con maggior carico mentale. Il consiglio d'oro è spostare gli allenamenti di 'Lavori Specifici' (Ripetute, Soglia, Lunghi) lontano da questo giorno, tenendolo per riposo o Easy Run."
                chart_card(c_adv3, "Pattern Stress Settimanale", fig_week, adv_week, C_STRESS)
            else:
                c_adv3.warning("Dati storici insufficienti per il calcolo settimanale.")
        else:
            c_adv3.warning("Colonna Stress Lavoro non trovata.")

        md("<div style='height:34px;'></div>")

        # =========================================================
        # SEZIONE 2: GRAFICI DI TREND BASE
        # =========================================================
        section_head("Trend Storici", "Andamento Base (90 giorni)", "Evoluzione fisiologica per individuare sovrallenamento cronico o deficit di recupero.")

        df_plot = df_adv.tail(90)

        def calcola_trend(serie):
            if len(serie) < 15: return 0
            recente = serie.tail(14).mean()
            precedente = serie.head(len(serie) - 14).mean()
            return recente - precedente

        r1c1, r1c2, r1c3 = st.columns(3)

        if 'Ore Sonno' in df_plot.columns:
            trend_sonno = calcola_trend(df_plot['Ore Sonno'])
            fig_t1 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['Ore Sonno'], mode='lines',
                line=dict(color=C_SONNO, width=2), fill='tozeroy', fillcolor='rgba(46,144,255,0.08)'
            ))
            fig_t1.update_layout(**layout_base, yaxis_title="ore")
            spieg_sonno = "Un trend in calo indica che non stai compensando il volume di allenamento. Dormire di più (o inserire Power Naps) è l'unica via naturale per innalzare il picco di forma." if trend_sonno < -0.3 else "Il tuo corpo sta ricevendo input di sonno stabili. Ottimo substrato per adattamenti positivi."
            chart_card(r1c1, "Trend Sonno Temporale", fig_t1, spieg_sonno, C_SONNO)

        if 'Stress Lavoro' in df_plot.columns:
            trend_stress = calcola_trend(df_plot['Stress Lavoro'])
            fig_t2 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['Stress Lavoro'], mode='lines',
                line=dict(color=C_STRESS, width=2), fill='tozeroy', fillcolor='rgba(255,69,58,0.08)'
            ))
            fig_t2.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="punti")
            spieg_stress = "Stress cronico in ascesa. Il cervello non distingue tra stress da Excel o da Squat: tutto si somma. Se la linea rossa sale, il volume di allenamento DEVE scendere per evitare crac sistemici." if trend_stress > 0.5 else "Carico mentale sotto controllo, via libera per concentrarsi sulla performance."
            chart_card(r1c2, "Trend Stress Lavorativo", fig_t2, spieg_stress, C_STRESS)

        if 'RPE' in df_plot.columns:
            trend_rpe = calcola_trend(df_plot['RPE'])
            fig_t3 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['RPE'], mode='lines',
                line=dict(color=C_RPE, width=2), fill='tozeroy', fillcolor='rgba(48,209,88,0.08)'
            ))
            fig_t3.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="punti")
            spieg_rpe = "Se l'RPE medio si alza continuamente a parità di passi, sei in over-reaching non funzionale. Necessario valutare una settimana di deload (dimezzare km e intensità)." if trend_rpe > 0.5 else "Mantenimento ottimale dello sforzo, il fitness sta compensando la fatica generata."
            chart_card(r1c3, "Trend Sforzo Percepito (RPE)", fig_t3, spieg_rpe, C_RPE)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # EXPORT REPORT
        # =========================================================
        section_head("Export", "Generazione report per coach", "Scarica l'analisi completa.")

        coach_txt = ""
        for nome_tab, contenuto in coach_content.items():
            coach_txt += f"\n[{nome_tab.upper()}]\n"
            for label, bullets in contenuto["blocchi"]:
                coach_txt += f"  {label}:\n"
                for b in bullets:
                    coach_txt += f"    - {b}\n"

        grafici_txt = "\n".join(f"  - {t}: {s}" for t, s in insights_export)

        report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (Target Originale: {distanza_target} km)
Indice Rischio: {risk_score:.0f}%
Recovery Score: {recovery_score:.0f}%
Stress Mentale (SMA): {sma:.1f}

NOTE CLINICHE E ANALISI AVANZATA:
{grafici_txt.replace('<br>', ' ')}

PROTOCOLLO COACH COMPLETO{coach_txt}
--------------------------------"""

        colb1, colb2 = st.columns(2)
        with colb1:
            st.download_button("Scarica TXT", data=report_testo, file_name="runai_report.txt", mime="text/plain", use_container_width=True)

        with colb2:
            charts_html = ""
            for i, f in enumerate(figs_per_export):
                include_js = 'cdn' if i == 0 else False
                charts_html += f.to_html(full_html=False, include_plotlyjs=include_js)

            coach_html = ""
            for nome_tab, contenuto in coach_content.items():
                blocchi_html = ""
                for label, bullets in contenuto["blocchi"]:
                    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                    blocchi_html += f"<div class='coach-block' style='--block-color:{contenuto['colore']};'><div class='label'>{label}</div><ul>{bullets_html}</ul></div>"
                coach_html += f"<div class='panel' style='margin-bottom:14px;'><h3 style='margin-bottom:14px;'>{nome_tab}</h3>{blocchi_html}</div>"

            report_html_completo = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');
  body {{ background:#0A0E15; color:{TXT_SECONDARY}; font-family: Inter, sans-serif; padding: 36px; max-width:1100px; margin:0 auto; }}
  h1 {{ color:{col}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.7em; margin-bottom:4px; }}
  h2 {{ color:{TXT_PRIMARY}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.15em; margin:34px 0 14px 0; }}
  .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7em; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 8px 0; font-weight:600; }}
  .panel {{ background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:14px; padding:20px 22px; margin-bottom:14px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
  .kpi-row .panel {{ flex:1 1 30%; min-width:200px; }}
  .kpi-row .val {{ font-family:'JetBrains Mono',monospace; font-size:1.7em; color:{TXT_PRIMARY}; font-weight:600; }}
  .coach-block {{ margin-bottom:14px; border-left:3px solid var(--block-color); padding-left:14px; }}
  .coach-block .label {{ font-family:'Oswald',sans-serif; font-size:.85em; letter-spacing:.05em; text-transform:uppercase; margin-bottom:8px; font-weight:600; color:var(--block-color); }}
  .charts-grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .charts-grid > div {{ flex: 1 1 30%; min-width:280px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:12px; padding:10px; }}
</style>
</head>
<body>
  <p class="eyebrow">RunAI Performance Report</p>
  <h1>{tit}</h1>
  <div style="margin:20px 0;">{radar_svg}</div>
  <div class="kpi-row">
    <div class="panel"><p class="eyebrow">Distanza Consigliata</p><div class="val">{distanza_consigliata:.1f} km</div></div>
    <div class="panel"><p class="eyebrow">Indice Rischio</p><div class="val" style="color:{col};">{risk_score:.0f}%</div></div>
    <div class="panel"><p class="eyebrow">Recovery Score</p><div class="val" style="color:{C_SONNO};">{recovery_score:.0f}%</div></div>
  </div>
  <h2>Protocollo coach completo</h2>{coach_html}
  <h2>Grafici analitici</h2><div class="charts-grid">{charts_html}</div>
</body>
</html>"""

            st.download_button("Scarica HTML (Grafici e Design)", data=report_html_completo, file_name="runai_report_completo.html", mime="text/html", use_container_width=True)

