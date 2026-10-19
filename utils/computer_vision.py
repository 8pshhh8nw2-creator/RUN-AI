import os
import tempfile

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_CV
from utils.computer_vision import analizza_running_video
from utils.sidebar import sidebar_comune

st.set_page_config(page_title="Computer Vision", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_CV = get_svg_url(SVG_CV)

df, df_full, filtro_tempo = sidebar_comune()

# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION - ANALISI CINEMATICA DELLA CORSA
# ---------------------------------------------------------
header_block(
    "Modulo 06 — Computer Vision",
    "AI RUNNING FORM ANALYSIS",
    "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro, individua ogni passo e misura angoli, overstride, cadenza e tipo di appoggio, con media e deviazione standard sui passi validi.",
    IMG_HERO_CV, "Pose Estimation 2D"
)
st.markdown("""
<div class='info-box'>
<strong>Analisi cinematica passo per passo:</strong> contatto iniziale e stacco sono individuati da cinematica, poi ogni misura è calcolata sul singolo passo e riassunta come media ± SD. Non vengono prodotti indici di rischio infortunio: non sono stimabili in modo valido da un video 2D.
</div>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    video_file = st.file_uploader("Carica video della corsa (profilo laterale, camera ferma, MP4/MOV)", type=["mp4", "mov", "avi"])
with col_h2:
    altezza_cm = st.number_input("La tua altezza (cm) — serve a calibrare le misure", min_value=120, max_value=220, value=175, step=1)

if video_file is not None:
    # Se cambia il video o l'altezza, invalida l'analisi precedente
    chiave_video = (hash(video_file.getvalue()), int(altezza_cm))
    if st.session_state.get('cv_video_hash') != chiave_video:
        st.session_state['cv_analizzato'] = False
        st.session_state['cv_video_hash'] = chiave_video

    if not st.session_state.get('cv_analizzato', False):
        if st.button("ELABORA SCHELETRO E ANALIZZA LA CORSA", use_container_width=True):
            with st.spinner("Estrazione scheletro (MediaPipe Pose) e analisi dei passi in corso..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                try:
                    tfile.write(video_file.getvalue())
                    tfile.close()
                    dati_estratti = analizza_running_video(tfile.name, altezza_cm=altezza_cm)
                    st.session_state.cv_dati = dati_estratti
                    st.session_state.cv_analizzato = True
                    ok = True
                except ValueError as e:
                    ok = False
                    st.error(f"Analisi non riuscita: {e}")
                except Exception as e:
                    ok = False
                    st.error(f"Errore durante l'elaborazione del video: {e}")
                finally:
                    if os.path.exists(tfile.name):
                        os.remove(tfile.name)
            if ok:
                st.rerun()

    if st.session_state.get('cv_analizzato', False):
        d = st.session_state.cv_dati
        q = d['qualita']
        ov_cm = d['overstride']['cm']
        tibia = d['angolo_tibia_contatto_gradi']
        contatto = d['tempo_contatto_ms']
        fsa = d['foot_strike_angle_gradi']
        busto = d['inclinazione_busto_gradi']
        vo = d['oscillazione_verticale_cm']

        st.success(
            f"Analisi completata. Lato {d['lato_analizzato']}, corsa {d['direzione_corsa']}, "
            f"{d['fps_video']} FPS, {q['passi_validi']} passi validi ({q['passi_scartati']} scartati)."
        )
        for avviso in d['avvisi']:
            st.warning(avviso)
        st.markdown("---")

        st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:4px; letter-spacing:0.1em;'>KINEMATIC WIREFRAME // GAIT ANALYSIS</p>", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Frame Rate", f"{d['fps_video']} FPS", f"±{q['risoluzione_temporale_ms']} ms per frame")
        mc2.metric("Lato Analizzato", d['lato_analizzato'], "MediaPipe Pose")
        mc3.metric("Tracking Valido", f"{q['frame_tracking_valido_pct']}%", f"{q['frame_totali']} frame")
        mc4.metric("Passi Validi", f"{q['passi_validi']}", f"{q['passi_scartati']} scartati")

        col_out1, col_out2 = st.columns([1, 1.1])

        with col_out1:
            st.video(video_file)
            st.markdown("<p style='font-size:0.75em; color:#00F5A0; text-align:center; font-family:\"JetBrains Mono\",monospace; margin-top:10px;'>OUTPUT: TRACKING COMPLETATO</p>", unsafe_allow_html=True)

        with col_out2:
            st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:6px; letter-spacing:0.1em;'>DIGITAL TWIN // MISURE AL CONTATTO INIZIALE</p>", unsafe_allow_html=True)

            txt_ginocchio = f"{d['angolo_ginocchio_appoggio']:.1f}"
            txt_overstride = f"{ov_cm['media']:+.1f}"
            txt_tibia = f"{tibia['media']:+.1f}"
            txt_cadenza = f"{d['cadenza_spm']:.0f}"
            txt_contatto = f"{contatto['media']:.0f}"

            digital_twin_svg = f"""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 320" style="background: radial-gradient(circle at center, #0B111A 0%, #04070B 100%); border-radius: 12px; border: 1px solid #1c2333; width: 100%; box-shadow: 0 8px 30px rgba(0,229,255,0.08);">
              <defs>
                <filter id="glow-real" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <pattern id="grid-real" width="25" height="25" patternUnits="userSpaceOnUse">
                  <path d="M 25 0 L 0 0 0 25" fill="none" stroke="#1c2333" stroke-width="0.5" opacity="0.3"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid-real)" />
              <g transform="translate(40, -15)">
                  <path d="M 180 60 C 190 100, 205 140, 215 180 C 200 190, 185 160, 175 110 Z" fill="#00E5FF" opacity="0.25"/>
                  <path d="M 180 60 C 190 100, 205 140, 215 180" fill="none" stroke="#00E5FF" stroke-width="3"/>
                  <path d="M 215 195 Q 205 260, 220 310" fill="none" stroke="#00E5FF" stroke-width="10" opacity="0.2" stroke-linecap="round"/>
                  <circle cx="215" cy="188" r="14" fill="#0E1420" stroke="#00E5FF" stroke-width="2"/>
                  <path d="M 200 240 Q 190 280, 205 310" fill="none" stroke="#FFB020" stroke-width="6" opacity="0.7"/>
                  <polygon points="210,310 230,325 260,325 250,305" fill="#0E1420" stroke="#00E5FF" stroke-width="1.5" opacity="0.8"/>
              </g>
              <circle cx="255" cy="173" r="16" fill="#00E5FF" opacity="0.45" filter="url(#glow-real)"/>
              <circle cx="255" cy="173" r="5" fill="#FFFFFF"/>
              <polyline points="255,173 320,115 560,115" fill="none" stroke="#00E5FF" stroke-width="1.5"/>
              <rect x="330" y="93" width="235" height="42" fill="#0A0F17" stroke="#00E5FF" stroke-width="1" rx="4"/>
              <text x="342" y="109" fill="#00E5FF" font-family="monospace" font-size="10" font-weight="bold">GINOCCHIO AL CONTATTO: {txt_ginocchio}°</text>
              <text x="342" y="123" fill="#8792A3" font-family="monospace" font-size="8">Angolo incluso, media sui passi</text>
              <circle cx="240" cy="285" r="12" fill="#FFB020" opacity="0.6" filter="url(#glow-real)"/>
              <circle cx="240" cy="285" r="4" fill="#FFFFFF"/>
              <polyline points="240,285 320,225 560,225" fill="none" stroke="#FFB020" stroke-width="1.5"/>
              <rect x="330" y="203" width="235" height="42" fill="#0A0F17" stroke="#FFB020" stroke-width="1" rx="4"/>
              <text x="342" y="219" fill="#FFB020" font-family="monospace" font-size="10" font-weight="bold">OVERSTRIDE: {txt_overstride} CM</text>
              <text x="342" y="233" fill="#8792A3" font-family="monospace" font-size="8">Tibia {txt_tibia}° dalla verticale</text>
              <rect x="20" y="240" width="150" height="70" fill="#0E1420" stroke="#1c2333" stroke-width="1" rx="6"/>
              <text x="28" y="256" fill="#8792A3" font-family="monospace" font-size="7">CADENZA MEDIA</text>
              <text x="28" y="282" fill="#00F5A0" font-family="monospace" font-size="20" font-weight="bold">{txt_cadenza} SPM</text>
              <text x="28" y="298" fill="#8792A3" font-family="monospace" font-size="7">CONTATTO {txt_contatto} MS</text>
            </svg>
            """
            st.components.v1.html(digital_twin_svg, height=330, scrolling=False)
            st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px; margin-bottom:12px;'><strong>Spiegazione:</strong> Sagoma illustrativa con le misure reali del video. Il disegno è schematico: i numeri sono le medie sui passi validi.</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>Report Biomeccanico Dettagliato</h2>", unsafe_allow_html=True)
        st.caption("Valori come media ± deviazione standard sui passi validi.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Angolo Ginocchio (contatto)", f"{d['angolo_ginocchio_appoggio']:.1f}°", f"± {d['flessione_ginocchio_gradi']['Contatto iniziale']['sd']}°")
        c2.metric("Inclinazione Busto", f"{busto['media']:.1f}°", f"± {busto['sd']}° (+ = avanti)")
        c3.metric("Overstride", f"{ov_cm['media']:+.1f} cm", f"± {ov_cm['sd']} cm")
        c4.metric("Oscillazione Verticale", f"{vo['media']:.1f} cm", f"± {vo['sd']} cm")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Cadenza", f"{d['cadenza_spm']:.1f} spm")
        c6.metric("Tempo di Contatto", f"{contatto['media']:.0f} ms", f"± {contatto['sd']} ms")
        c7.metric("Tibia al Contatto", f"{tibia['media']:+.1f}°", f"± {tibia['sd']}°")
        c8.metric("Foot Strike Angle", f"{fsa['media']:+.1f}°", f"± {fsa['sd']}°")
        st.markdown("<br>", unsafe_allow_html=True)

        cg1, cg2, cg3 = st.columns(3)

        with cg1:
            st.markdown("### 1. Tipo di Appoggio (%)")
            distr = d['distribuzione_appoggio_pct']
            fig_app = px.bar(
                x=["Tallone", "Mesopiede", "Avampiede"],
                y=[distr['tallone'], distr['mesopiede'], distr['avampiede']],
                labels={'x': 'Appoggio', 'y': '% dei passi'},
                color=[distr['tallone'], distr['mesopiede'], distr['avampiede']],
                color_continuous_scale=[[0, '#1c2333'], [1, '#00E5FF']]
            )
            fig_app.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_app), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Distribuzione:</strong> percentuale di passi classificati per tipo di appoggio dal foot strike angle (soglie di Altman & Davis 2012).</div>", unsafe_allow_html=True)

        with cg2:
            st.markdown("### 2. Angoli Articolari (Falcata)")
            fig_radar = go.Figure(go.Scatterpolar(
                r=d['angoli_fase'] + [d['angoli_fase'][0]],
                theta=d['fasi_gait'] + [d['fasi_gait'][0]],
                fill='toself',
                marker=dict(color='#00F5A0'), line=dict(color='#00F5A0')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[40, 190], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')),
                height=320
            )
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Angolo del ginocchio</strong> (180° = gamba estesa) nelle quattro fasi del passo, individuate automaticamente dal tracking.</div>", unsafe_allow_html=True)

        with cg3:
            st.markdown("### 3. Flessione Ginocchio (± SD)")
            fasi = list(d['flessione_ginocchio_gradi'].keys())
            medie = [d['flessione_ginocchio_gradi'][f]['media'] for f in fasi]
            sds = [d['flessione_ginocchio_gradi'][f]['sd'] for f in fasi]
            fig_flex = go.Figure(go.Bar(
                x=fasi, y=medie,
                error_y=dict(type='data', array=sds, color='#FFB020'),
                marker=dict(color='#00E5FF')
            ))
            fig_flex.update_layout(height=320, yaxis_title="Flessione (°)")
            st.plotly_chart(style_fig(fig_flex), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Variabilità:</strong> le barre d'errore mostrano quanto cambia la flessione da un passo all'altro. Una SD alta indica una ripresa instabile o poco regolare.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3>Lettura dei Risultati</h3>", unsafe_allow_html=True)
        st.info(
            f"{d['tipo_appoggio']} (FSA medio {fsa['media']:+.1f}°). "
            f"Overstride {ov_cm['media']:+.1f} cm (positivo = piede davanti all'anca) e tibia a {tibia['media']:+.1f}° dalla verticale al contatto."
        )
        st.markdown(f"""
        <div class='info-box'>
            <strong>Cadenza e contatto:</strong> {d['cadenza_spm']:.0f} passi/min e {contatto['media']:.0f} ms di contatto. Il lavoro di Heiderscheit et al. (2011) associa un aumento della frequenza di passo a minori carichi articolari: è uno spunto da valutare con un tecnico, non una prescrizione.
        </div>
        """, unsafe_allow_html=True)
        st.caption("Analisi 2D indicativa: non è una diagnosi medica e non stima il rischio di infortunio. Per valutazioni cliniche rivolgiti a un fisioterapista o a un tecnico di corsa.")

        with st.expander("Note metodologiche"):
            for nota in d['note_metodologiche']:
                st.markdown(f"- {nota}")
else:
    st.info("Suggerimento: carica un video registrato lateralmente (camera ferma e perpendicolare alla corsa, corridore intero visibile per almeno 5-10 secondi, meglio a 120+ fps) per attivare l'analisi.")
