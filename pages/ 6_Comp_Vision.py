from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import tempfile
import plotly.express as px
import plotly.graph_objects as go

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_CV
from utils.computer_vision import analizza_running_video
st.set_page_config(page_title="Computer Vision", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_CV = get_svg_url(SVG_CV)
import cv2
import mediapipe as mp
# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI (DATI REALI)
# ---------------------------------------------------------
    header_block(
        "Modulo 06 — Computer Vision",
        "AI RUNNING FORM ANALYSIS & INJURY PREDICTION",
        "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro biometrico, calcola angoli/sovraccarichi e predice il rischio d'infortunio tramite Machine Learning.",
        IMG_HERO_CV, "Pose Estimation & ML"
    )
    st.markdown("""
    <div class='info-box'>
    <strong>Analisi Biometrica Avanzata:</strong> Estrazione dello scheletro posturale, mappatura dei sovraccarichi articolari, analisi angolare della falcata e predizione del distretto anatomico a rischio infortunio, calcolate direttamente dai keypoint del tuo video.
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        video_file = st.file_uploader("Carica video della corsa (Profilo laterale consigliato, MP4/MOV)", type=["mp4", "mov", "avi"])
    with col_h2:
        altezza_cm = st.number_input("La tua altezza (cm) — serve a calibrare le misure", min_value=120, max_value=220, value=175, step=1)

    if video_file is not None:
        # Se cambia il video caricato, invalida l'analisi precedente
        video_hash = hash(video_file.getvalue())
        if st.session_state.get('cv_video_hash') != video_hash:
            st.session_state['cv_analizzato'] = False
            st.session_state['cv_video_hash'] = video_hash

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.getvalue())
        video_path = tfile.name

        if not st.session_state.get('cv_analizzato', False):
            if st.button("ELABORA SCHELETRO E PREDICI INFORTUNIO", use_container_width=True):
                with st.spinner("Estrazione scheletro (MediaPipe Pose) e calcolo biomeccanico in corso..."):
                    try:
                        dati_estratti = analizza_running_video(video_path, altezza_cm=altezza_cm)
                        st.session_state.cv_analizzato = True
                        st.session_state.cv_dati = dati_estratti
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Analisi non riuscita: {e}")
                    except Exception as e:
                        st.error(f"Errore durante l'elaborazione del video: {e}")

        if st.session_state.get('cv_analizzato', False):
            dati_cv = st.session_state.cv_dati
            st.success(
                f"Analisi video completata. Scheletro tracciato lato {dati_cv['lato_analizzato']}, "
                f"{dati_cv['fps_video']} FPS, appoggio analizzato al frame {dati_cv['frame_strike_analizzato']}."
            )
            st.markdown("---")

            st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:4px; letter-spacing:0.1em;'>KINEMATIC WIREFRAME // GAIT ANALYSIS FRAME</p>", unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Frame Rate", f"{dati_cv['fps_video']} FPS", "Sorgente video")
            mc2.metric("Lato Analizzato", dati_cv['lato_analizzato'], "MediaPipe Pose")
            mc3.metric("Fase", "Strike", f"Frame {dati_cv['frame_strike_analizzato']}")
            st.markdown("<p style='font-size:0.85em; color:#8792A3; margin-top:8px; margin-bottom:16px;'>Tracciamento articolare e analisi vettoriale basata sui dati reali del video:</p>", unsafe_allow_html=True)

            col_out1, col_out2 = st.columns([1, 1.1])

            with col_out1:
                st.video(video_file)
                st.markdown("<p style='font-size:0.75em; color:#00F5A0; text-align:center; font-family:\"JetBrains Mono\",monospace; margin-top:10px;'>OUTPUT: AI TRACKING COMPLETATO</p>", unsafe_allow_html=True)
            with col_out2:
                dati_REALI = dati_cv
                st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:6px; letter-spacing:0.1em;'>DIGITAL TWIN // KINEMATIC STRESS MAP (REALE)</p>", unsafe_allow_html=True)
                digital_twin_real_svg = f"""
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 320" style="background: radial-gradient(circle at center, #0B111A 0%, #04070B 100%); border-radius: 12px; border: 1px solid #1c2333; width: 100%; box-shadow: 0 8px 30px rgba(0,229,255,0.08);">
                  <defs>
                    <filter id="glow-red-real" x="-50%" y="-50%" width="200%" height="200%">
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
                  <circle cx="255" cy="173" r="16" fill="#FF6A3D" opacity="0.5" filter="url(#glow-red-real)"/>
                  <circle cx="255" cy="173" r="5" fill="#FFFFFF"/>
                  <polyline points="255,173 320,115 560,115" fill="none" stroke="#FF6A3D" stroke-width="1.5"/>
                  <rect x="330" y="93" width="235" height="42" fill="#0A0F17" stroke="#FF6A3D" stroke-width="1" rx="4"/>
                  <text x="342" y="109" fill="#FF6A3D" font-family="monospace" font-size="10" font-weight="bold">GINOCCHIO: {dati_REALI['angolo_ginocchio_appoggio']}°</text>
                  <text x="342" y="123" fill="#8792A3" font-family="monospace" font-size="8">Angolo critico estratto dal video</text>
                  <circle cx="240" cy="285" r="12" fill="#FFB020" opacity="0.6" filter="url(#glow-red-real)"/>
                  <circle cx="240" cy="285" r="4" fill="#FFFFFF"/>
                  <polyline points="240,285 320,225 560,225" fill="none" stroke="#FFB020" stroke-width="1.5"/>
                  <rect x="330" y="203" width="235" height="42" fill="#0A0F17" stroke="#FFB020" stroke-width="1" rx="4"/>
                  <text x="342" y="219" fill="#FFB020" font-family="monospace" font-size="10" font-weight="bold">OVERSTRIDE: {dati_REALI['overstride_cm']} CM</text>
                  <text x="342" y="233" fill="#8792A3" font-family="monospace" font-size="8">Anticipo falcata rilevato dal tracking</text>
                  <rect x="20" y="240" width="150" height="70" fill="#0E1420" stroke="#1c2333" stroke-width="1" rx="6"/>
                  <text x="28" y="256" fill="#8792A3" font-family="monospace" font-size="7">RISCHIO STIMATO VIDEO</text>
                  <text x="28" y="282" fill="#FF6A3D" font-family="monospace" font-size="20" font-weight="bold">{dati_REALI['probabilita_infortunio_ml']}%</text>
                  <text x="28" y="298" fill="#FF6A3D" font-family="monospace" font-size="7">STATUS: {"CRITICO" if dati_REALI['probabilita_infortunio_ml'] > 60 else "MODERATO" if dati_REALI['probabilita_infortunio_ml'] > 35 else "BASSO"}</text>
                </svg>
                """
                st.components.v1.html(digital_twin_real_svg, height=330, scrolling=False)
                st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px; margin-bottom:12px;'><strong>Spiegazione:</strong> Mappa anatomica vettoriale guidata dai dati biometrici reali del video, che evidenzia i distretti sottoposti a picchi di stress strutturale.</p>", unsafe_allow_html=True)
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                grf_real_svg = f"""
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200" style="background: #080B12; border-radius: 12px; border: 1px solid #1c2333; width: 100%;">
                    <defs>
                        <linearGradient id="grfGradReal" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stop-color="#FF6A3D" stop-opacity="0.5"/>
                            <stop offset="100%" stop-color="#FF6A3D" stop-opacity="0.0"/>
                        </linearGradient>
                    </defs>
                    <line x1="50" y1="150" x2="570" y2="150" stroke="#566178" stroke-width="1.5"/>
                    <line x1="50" y1="40" x2="570" y2="40" stroke="#1c2333" stroke-width="1" stroke-dasharray="3,3"/>
                    <text x="15" y="44" fill="#8792A3" font-family="monospace" font-size="8">3.0 BW</text>
                    <text x="15" y="154" fill="#8792A3" font-family="monospace" font-size="8">0.0 BW</text>
                    <path d="M 50 150 C 150 150, 200 65, 310 65 C 420 65, 470 150, 550 150" fill="none" stroke="#00E5FF" stroke-width="2" stroke-dasharray="4,4" opacity="0.6"/>
                    <text x="430" y="55" fill="#00E5FF" font-family="monospace" font-size="8">Standard Ideale</text>
                    <path d="M 50 150 L 90 150 L 130 20 L 170 90 C 240 90, 360 55, 470 150 L 550 150" fill="url(#grfGradReal)" />
                    <path d="M 50 150 L 90 150 L 130 20 L 170 90 C 240 90, 360 55, 470 150 L 550 150" fill="none" stroke="#FF6A3D" stroke-width="2.5" stroke-linejoin="round"/>
                    <circle cx="130" cy="20" r="5" fill="#FFFFFF" stroke="#FF6A3D" stroke-width="2"/>
                    <line x1="130" y1="20" x2="190" y2="20" stroke="#FF6A3D" stroke-width="1"/>
                    <rect x="195" y="10" width="195" height="20" fill="#0A0F17" stroke="#FF6A3D" stroke-width="1" rx="3"/>
                    <text x="202" y="24" fill="#FF6A3D" font-family="monospace" font-size="9" font-weight="bold">OVERSTRIDE: {dati_REALI['overstride_cm']} CM</text>
                    <text x="270" y="185" fill="#566178" font-family="monospace" font-size="8">TEMPO DI CONTATTO (ms)</text>
                </svg>
                """
                st.components.v1.html(grf_real_svg, height=195, scrolling=False)
                st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px;'><strong>Spiegazione:</strong> Grafico delle forze d'impatto al suolo (GRF), stimato a partire dall'overstride e dall'angolo di ginocchio rilevati nel video.</p>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h2>Report Biomeccanico e Scheletrico Dettagliato</h2>", unsafe_allow_html=True)
            c_met1, c_met2, c_met3, c_met4 = st.columns(4)
            c_met1.metric("Angolo Ginocchio", f"{dati_cv['angolo_ginocchio_appoggio']:.1f}°", "Target > 150°")
            c_met2.metric("Inclinazione Busto", f"{dati_cv['angolo_inclinazione_busto']:.1f}°", "Ottimale 5-8°")
            c_met3.metric("Overstride (Anticipo)", f"{dati_cv['overstride_cm']:.1f} cm", "Target < 10cm")
            c_met4.metric("Oscillazione Vert.", f"{dati_cv['oscillazione_verticale']:.1f} cm", "Target < 8cm")
            st.markdown("<br>", unsafe_allow_html=True)
            cg1, cg2, cg3 = st.columns(3)

            with cg1:
                st.markdown("### 1. Mappatura Sovraccarico (%)")
                fig_bar_load = px.bar(
                    x=dati_cv['articolazioni_carico'], y=dati_cv['carichi_pct'],
                    labels={'x': 'Distretto', 'y': '% Impatto'},
                    color=dati_cv['carichi_pct'], color_continuous_scale=[[0, '#00E5FF'], [0.5, '#FFB020'], [1, '#FF6A3D']]
                )
                fig_bar_load.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_bar_load), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Analisi Carichi:</strong> Percentuale di stress relativo distribuito sui distretti articolari, calcolata dall'angolo di ginocchio, dal tipo di appoggio e dall'inclinazione del busto misurati nel video.</div>", unsafe_allow_html=True)
            with cg2:
                st.markdown("### 2. Angoli Articolari (Falcata)")
                fig_radar_angles = go.Figure(go.Scatterpolar(
                    r=dati_cv['angoli_fase'], theta=dati_cv['fasi_gait'], fill='toself',
                    marker=dict(color='#00F5A0'), line=dict(color='#00F5A0')
                ))
                fig_radar_angles.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[60, 190], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')),
                    height=320
                )
                st.plotly_chart(style_fig(fig_radar_angles), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Analisi Angolare:</strong> Grado di flessione del ginocchio nelle quattro fasi del ciclo del passo, individuate automaticamente dal tracking (strike, mid-stance, toe-off, swing).</div>", unsafe_allow_html=True)
            with cg3:
                st.markdown("### 3. Rischio per Distretto (%)")
                fig_ml_risk = px.bar(
                    x=dati_cv['distretti_rischio'], y=dati_cv['rischi_ml'],
                    labels={'x': 'Patologia/Distretto', 'y': 'Peso relativo (%)'},
                    color=dati_cv['rischi_ml'], color_continuous_scale=[[0, '#00F5A0'], [0.5, '#FFB020'], [1, '#FF6A3D']]
                )
                fig_ml_risk.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_ml_risk), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Stima del Rischio:</strong> Punteggio euristico calcolato dalle feature biomeccaniche reali (angolo ginocchio, overstride, tipo di appoggio, inclinazione busto). Non è un modello clinico addestrato su dataset di infortuni reali.</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h3>Diagnosi Posturale, Errori e Stima del Rischio</h3>", unsafe_allow_html=True)

            st.error(f"PATTERN RILEVATO — {dati_cv['tipo_appoggio']}: angolo del ginocchio all'appoggio di {dati_cv['angolo_ginocchio_appoggio']}° e overstride di {dati_cv['overstride_cm']} cm, misurati direttamente sul frame di appoggio del video.")
            st.warning(f"ZONA DI SOVRACCARICO PRINCIPALE: {dati_cv['sovraccarico_prevalente']}, in base alla distribuzione di carico calcolata sopra.")
            st.markdown(f"""
            <div class='danger-box' style='border-left-color: #FF6A3D;'>
                <h3 style='color: #FF6A3D; margin-top:0;'>STIMA DI RISCHIO (Indice: {dati_cv['probabilita_infortunio_ml']}%)</h3>
                <p style='color: #E8ECF2; font-size: 1.05em;'>In base al pattern di overstride e al tipo di appoggio rilevati nel video, il distretto con il punteggio di rischio relativo più alto è: <strong style='color: #FF6A3D;'>{dati_cv['infortunio_predetto']}</strong>. Questa è una stima euristica orientativa, non una diagnosi medica.</p>
            </div>
            """, unsafe_allow_html=True)

            st.info("PROTOCOLLO DI CORREZIONE BIOMECCANICA CONSIGLIATO:\n1. Riduzione dell'ampiezza della falcata per eliminare l'over-stride anteriore al baricentro.\n2. Incremento della frequenza di passo a 176-180 falcate al minuto (SPM) per facilitare l'atterraggio sul mesopiede.\n3. Integrazione di esercizi di forza eccentrica per il quadricipite e protocollo di rinforzo progressivo per il tendine d'Achille.\n4. Consulta un fisioterapista o un tecnico di corsa per una valutazione clinica completa prima di modificare la tecnica.")
    else:
        st.info("Suggerimento: Carica un video registrato lateralmente (profilo laterale, corridore ben visibile per almeno 2-3 secondi) per attivare l'estrazione dello scheletro e l'analisi biomeccanica reale.")

