import os
import hashlib
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_CV
from utils.computer_vision import analizza_running_video, AnalisiNonRiuscita
from utils.sidebar import sidebar_comune

st.set_page_config(page_title="Computer Vision", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_CV = get_svg_url(SVG_CV)

df, df_full, filtro_tempo = sidebar_comune()

# Chiavi che la pagina legge dal risultato dell'analisi.
CHIAVI_CV = {
    "lato_analizzato", "direzione_corsa", "fps_video", "frame_strike_analizzato",
    "frame_contatto", "frame_stacco", "qualita", "avvisi", "cadenza_spm",
    "tempo_contatto_ms", "overstride_cm", "overstride",
    "angolo_tibia_contatto_gradi", "flessione_ginocchio_gradi",
    "angolo_ginocchio_appoggio", "angolo_inclinazione_busto",
    "inclinazione_busto_gradi", "oscillazione_verticale",
    "oscillazione_verticale_cm", "foot_strike_angle_gradi", "tipo_appoggio",
    "distribuzione_appoggio_pct", "fasi_gait", "angoli_fase",
    "diagnostica", "note_metodologiche",
}


def fmt_stat(s, unita="", cifre=1):
    """Formatta un dict {'media','sd','n'} come 'media ± sd'."""
    if not s or s.get("media") is None:
        return "n.d."
    return f"{s['media']:.{cifre}f} ± {s['sd']:.{cifre}f}{unita}"


def grafico_segnale_piede(y_px, fps, picchi, contatti, stacchi, titolo_y="Quota piede (px)"):
    """Serie del punto piu' basso del piede con picchi, contatti e stacchi."""
    if not y_px:
        return None
    t = [i / fps for i in range(len(y_px))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_px, mode="lines", name="Piede (y)",
                             line=dict(color="#00E5FF", width=2)))

    def marker(frames, nome, colore, simbolo):
        fr = [f for f in frames if 0 <= f < len(y_px)]
        if fr:
            fig.add_trace(go.Scatter(
                x=[f / fps for f in fr], y=[y_px[f] for f in fr],
                mode="markers", name=nome,
                marker=dict(color=colore, size=9, symbol=simbolo)))

    marker(picchi, "Picchi sollevamento", "#FFB020", "triangle-up")
    marker(contatti, "Contatto iniziale (IC)", "#00F5A0", "circle")
    marker(stacchi, "Stacco (TO)", "#FF6A3D", "x")
    # y immagine cresce verso il basso: invertita per leggere "su = piede alto"
    fig.update_yaxes(autorange="reversed", title=titolo_y)
    fig.update_xaxes(title="Tempo (s)")
    fig.update_layout(height=340, legend=dict(orientation="h", y=-0.25))
    return style_fig(fig)


# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION - ANALISI CINEMATICA DELLA CORSA
# ---------------------------------------------------------
header_block(
    "Modulo 06 — Computer Vision",
    "ANALISI CINEMATICA DELLA CORSA",
    "Carica un video di corsa (profilo laterale): MediaPipe Pose estrae lo scheletro e la pagina calcola cadenza, tempo di contatto, overstride, angoli articolari e tipo di appoggio.",
    IMG_HERO_CV, "Pose Estimation 2D"
)
st.markdown("""
<div class='info-box'>
<strong>Analisi cinematica 2D:</strong> gli eventi del passo (contatto e stacco) sono stime ricavate dal segnale del piede, gli angoli sono proiezioni sul piano immagine e la scala in cm deriva dall'altezza dichiarata. Non vengono stimati rischio di infortunio o carichi articolari.
</div>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    video_file = st.file_uploader(
        "Carica video della corsa (profilo laterale, MP4/MOV/AVI)",
        type=["mp4", "mov", "avi"]
    )
with col_h2:
    altezza_cm = st.number_input(
        "La tua altezza (cm) — serve a stimare la scala",
        min_value=120, max_value=220, value=175, step=1
    )

if video_file is None:
    st.info("Suggerimento: carica un video girato di lato, con il corridore ben visibile per almeno 5-10 secondi "
            "e camera ferma. Se possibile usa 120 fps o più: a 30 fps contatto e stacco hanno un errore di almeno un frame.")
    st.stop()

# --- Invalida l'analisi se cambia il video o l'altezza ---
contenuto = video_file.getvalue()
firma = hashlib.md5(contenuto).hexdigest() + f"|{altezza_cm}"
if st.session_state.get('cv_firma') != firma:
    st.session_state['cv_firma'] = firma
    st.session_state['cv_analizzato'] = False
    st.session_state.pop('cv_dati', None)
    st.session_state.pop('cv_errore', None)

# --- Invalida dati in sessione di una versione precedente del modulo ---
if st.session_state.get('cv_analizzato', False):
    mancanti = CHIAVI_CV - set(st.session_state.get('cv_dati', {}).keys())
    if mancanti:
        st.session_state['cv_analizzato'] = False
        st.session_state.pop('cv_dati', None)
        st.warning("Risultati salvati di una versione precedente: rielabora il video.")

# --- Elaborazione ---
if not st.session_state.get('cv_analizzato', False):
    if st.button("ELABORA SCHELETRO E CALCOLA METRICHE", use_container_width=True):
        st.session_state.pop('cv_errore', None)
        video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(contenuto)
                video_path = tfile.name
            with st.spinner("Estrazione scheletro (MediaPipe Pose) e analisi dei cicli di corsa in corso..."):
                st.session_state.cv_dati = analizza_running_video(video_path, altezza_cm=altezza_cm)
                st.session_state.cv_analizzato = True
        except AnalisiNonRiuscita as e:
            st.session_state.cv_errore = {"msg": str(e), "dati": e.dati}
        except ValueError as e:
            st.session_state.cv_errore = {"msg": str(e), "dati": {}}
        except Exception as e:
            st.session_state.cv_errore = {"msg": f"Errore durante l'elaborazione del video: {e}", "dati": {}}
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        if st.session_state.get('cv_analizzato', False):
            st.rerun()

# --- Analisi non riuscita: mostra il motivo e il segnale per capire cosa non va ---
errore = st.session_state.get('cv_errore')
if errore and not st.session_state.get('cv_analizzato', False):
    st.error(f"Analisi non riuscita: {errore['msg']}")
    d = errore["dati"]
    if d.get("y_piede_px"):
        st.markdown("**Segnale del piede (diagnostica)** — utile per capire se il tracking o l'inquadratura sono il problema:")
        fig = grafico_segnale_piede(
            d["y_piede_px"], d.get("fps", 30.0),
            d.get("picchi_piede_frame", []),
            d.get("frame_contatto", []), d.get("frame_stacco", [])
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    st.info("Consigli: camera ferma e di profilo, corridore interamente inquadrato, buona luce, "
            "almeno 5-10 secondi di corsa continua, abiti non troppo larghi.")

# --- Risultati ---
if st.session_state.get('cv_analizzato', False):
    r = st.session_state.cv_dati
    q = r["qualita"]
    diag = r["diagnostica"]
    fps = r["fps_video"]

    st.success(
        f"Analisi completata. Lato {r['lato_analizzato'].lower()}, corsa {r['direzione_corsa']}, "
        f"{fps} FPS, {q['cicli_validi']} cicli validi su {q['cicli_validi'] + q['cicli_scartati']}."
    )

    for avviso in r["avvisi"]:
        st.warning(avviso)

    st.markdown("---")
    col_v, col_m = st.columns([1, 1.2])
    with col_v:
        st.video(contenuto)
        st.markdown(
            "<p style='font-size:0.75em; color:#8792A3; text-align:center;'>Video originale. "
            f"Frame di contatto centrale analizzato: {r['frame_strike_analizzato']}.</p>",
            unsafe_allow_html=True
        )
    with col_m:
        st.markdown("<h3>Metriche principali</h3>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Cadenza", f"{r['cadenza_spm']:.0f} spm")
        m2.metric("Tempo di contatto", fmt_stat(r["tempo_contatto_ms"], " ms", 0))
        m3, m4 = st.columns(2)
        m3.metric("Overstride", fmt_stat(r["overstride"]["cm"], " cm"),
                  help="Distanza orizzontale caviglia-anca al contatto (positiva = piede davanti all'anca). Scala indicativa.")
        m4.metric("Angolo tibia al contatto", fmt_stat(r["angolo_tibia_contatto_gradi"], "°"),
                  help="Inclinazione della tibia rispetto alla verticale; positivo = piede davanti al ginocchio.")
        m5, m6 = st.columns(2)
        m5.metric("Inclinazione busto", fmt_stat(r["inclinazione_busto_gradi"], "°"))
        m6.metric("Oscillazione verticale", fmt_stat(r["oscillazione_verticale_cm"], " cm"),
                  help="Escursione verticale dell'anca nel ciclo, scala indicativa.")
        m7, m8 = st.columns(2)
        m7.metric("Angolo ginocchio al contatto", f"{r['angolo_ginocchio_appoggio']:.1f}°",
                  help="Angolo incluso anca-ginocchio-caviglia (180° = gamba estesa).")
        m8.metric("Tipo di appoggio", r["tipo_appoggio"].split(" (")[0])

    st.markdown("---")
    st.markdown("<h3>Segnale del piede e rilevamento degli eventi</h3>", unsafe_allow_html=True)
    fig_sig = grafico_segnale_piede(
        diag["y_piede_px"], fps, diag["picchi_piede_frame"],
        r["frame_contatto"], r["frame_stacco"]
    )
    if fig_sig is not None:
        st.plotly_chart(fig_sig, use_container_width=True)
    st.markdown(
        "<div class='explain-text'><strong>Come leggerlo:</strong> la curva è la quota del punto più basso del piede "
        "(in alto = piede sollevato). I triangoli sono i picchi di sollevamento che delimitano i cicli; "
        "i cerchi verdi sono i contatti iniziali e le croci arancioni gli stacchi, stimati dal segnale. "
        "Se i marcatori non cadono sui passi reali, le metriche non sono affidabili.</div>",
        unsafe_allow_html=True
    )

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown("### Flessione del ginocchio per fase")
        fasi = r["fasi_gait"]
        angoli = r["angoli_fase"]
        coppie = [(f, a) for f, a in zip(fasi, angoli) if a is not None]
        if coppie:
            fig_radar = go.Figure(go.Scatterpolar(
                r=[a for _, a in coppie] + [coppie[0][1]],
                theta=[f for f, _ in coppie] + [coppie[0][0]],
                fill='toself',
                marker=dict(color='#00F5A0'), line=dict(color='#00F5A0')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[60, 190], gridcolor='#1c2333'),
                           angularaxis=dict(gridcolor='#1c2333')),
                height=340, showlegend=False
            )
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
        st.markdown(
            "<div class='explain-text'><strong>Angoli:</strong> angolo incluso anca-ginocchio-caviglia (media sui cicli validi) "
            "al contatto, alla massima flessione in appoggio, allo stacco e alla massima flessione in oscillazione. "
            "Valori più bassi = ginocchio più flesso.</div>",
            unsafe_allow_html=True
        )
    with cg2:
        st.markdown("### Distribuzione del tipo di appoggio")
        distr = r["distribuzione_appoggio_pct"]
        fig_app = px.bar(
            x=["Tallone", "Mesopiede", "Avampiede"],
            y=[distr["tallone"], distr["mesopiede"], distr["avampiede"]],
            labels={'x': 'Appoggio', 'y': '% dei cicli validi'},
            color=[distr["tallone"], distr["mesopiede"], distr["avampiede"]],
            color_continuous_scale=[[0, '#00E5FF'], [1, '#00F5A0']]
        )
        fig_app.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_app), use_container_width=True)
        st.markdown(
            f"<div class='explain-text'><strong>Foot strike angle:</strong> {fmt_stat(r['foot_strike_angle_gradi'], '°')}. "
            "La classificazione usa soglie sull'angolo del piede al contatto; con camera a bassa risoluzione "
            "o a 30 fps è indicativa.</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<h3>Qualità dell'analisi</h3>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Frame totali", q["frame_totali"])
    k2.metric("Tracking valido", f"{q['frame_tracking_valido_pct']:.0f}%")
    k3.metric("Cicli validi / scartati", f"{q['cicli_validi']} / {q['cicli_scartati']}")
    k4.metric("Risoluzione temporale", f"{q['risoluzione_temporale_ms']:.0f} ms")

    righe = diag.get("validazione_cicli", [])
    if righe:
        with st.expander("Dettaglio dei cicli (validi e scartati)"):
            tab = pd.DataFrame(righe)
            tab["valido"] = tab["valido"].map({True: "Sì", False: "No"})
            tab = tab.rename(columns={
                "ciclo": "Ciclo", "inizio_frame": "Inizio", "fine_frame": "Fine",
                "ic_frame": "IC", "to_frame": "TO", "stride_s": "Durata (s)",
                "contatto_s": "Contatto (s)", "valido": "Valido", "motivo_scarto": "Motivo scarto",
            })
            st.dataframe(tab, use_container_width=True, hide_index=True)

    with st.expander("Note metodologiche e limiti"):
        for nota in r["note_metodologiche"]:
            st.markdown(f"- {nota}")
        st.markdown(
            f"- Lunghezza arto stimata: {diag['lunghezza_arto_px']} px; "
            f"scala: {q['scala_cm_per_pixel']} cm/px."
        )
        cop = diag.get("copertura_landmark_pct", {})
        if cop:
            st.markdown("**Copertura landmark:** " + ", ".join(f"{k} {v:.0f}%" for k, v in cop.items()))

    st.info("Questi risultati sono misure cinematiche 2D a scopo descrittivo, non una valutazione clinica. "
            "Per modificare la tecnica o in presenza di dolore rivolgiti a un fisioterapista o a un tecnico di corsa.")
