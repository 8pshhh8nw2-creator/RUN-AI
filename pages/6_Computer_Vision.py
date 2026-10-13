import hashlib
import html
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import (
    header_block,
    style_fig,
    get_svg_url,
    SVG_CV,
)
from utils.computer_vision import analizza_running_video
from utils.sidebar import sidebar_comune


# =========================================================
# CONFIGURAZIONE
# =========================================================
st.set_page_config(
    page_title="Computer Vision",
    layout="wide",
)

carica_css()
logger = logging.getLogger(__name__)

C_CYAN = "#00E5FF"
C_GREEN = "#00F5A0"
C_AMBER = "#FFB020"
C_ORANGE = "#FF6A3D"
C_TEXT = "#E8ECF2"
C_DIM = "#8792A3"

MAX_VIDEO_MB = 200

st.markdown(
    """
    <style>
    .cv-card {
        background: linear-gradient(135deg, #0E1420, #080B12);
        border: 1px solid #1c2333;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.18);
    }
    .cv-eyebrow {
        color: #00E5FF;
        font-family: "JetBrains Mono", monospace;
        font-size: .73rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin: 0 0 10px;
    }
    .cv-description {
        color: #8792A3;
        font-size: .88rem;
        line-height: 1.6;
        margin: 0;
    }
    .cv-notice {
        background: rgba(0,229,255,.045);
        border: 1px solid rgba(0,229,255,.18);
        border-left: 3px solid #00E5FF;
        border-radius: 10px;
        padding: 16px 20px;
        color: #B8C2D0;
        line-height: 1.6;
        margin: 12px 0 20px;
    }
    .cv-caption {
        color: #8792A3;
        font-size: .82rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# UTILITÀ
# =========================================================
def numero(valore):
    if valore is None or isinstance(valore, (bool, np.bool_)):
        return None

    try:
        valore = float(valore)
    except (TypeError, ValueError):
        return None

    return valore if np.isfinite(valore) else None


def formato(valore, decimali=1, unita=""):
    valore = numero(valore)

    if valore is None:
        return "N/D"

    testo = f"{valore:.{decimali}f}"
    return f"{testo} {unita}".strip()


def formato_stat(statistica, unita=""):
    if not isinstance(statistica, dict):
        return "N/D"

    media = numero(statistica.get("media"))
    sd = numero(statistica.get("sd"))

    if media is None:
        return "N/D"

    if sd is None:
        return formato(media, 1, unita)

    return f"{media:.1f} ± {sd:.1f} {unita}".strip()


def reset_analisi():
    for chiave in (
        "cv_dati",
        "cv_errore",
        "cv_diagnostica",
        "cv_analisi_firma",
    ):
        st.session_state.pop(chiave, None)

    st.session_state["cv_analizzato"] = False


def figura_stilizzata(fig, altezza=320):
    fig = style_fig(fig)
    fig.update_layout(
        height=altezza,
        paper_bgcolor="#080B12",
        plot_bgcolor="#080B12",
        font=dict(color=C_TEXT),
        margin=dict(l=45, r=25, t=35, b=55),
    )
    return fig


def render_svg(svg, altezza=350):
    st.components.v1.html(
        f"""
        <html>
        <head>
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        .wrap {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: {altezza}px;
        }}
        svg {{
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
        }}
        </style>
        </head>
        <body>
            <div class="wrap">{svg}</div>
        </body>
        </html>
        """,
        height=altezza,
        scrolling=False,
    )


def svg_scheletro(dati):
    punti = dati.get("scheletro_frame")

    if not isinstance(punti, dict):
        return None

    parti = (
        "spalla",
        "anca",
        "ginocchio",
        "caviglia",
        "tallone",
        "punta",
    )

    coordinate = {}

    for parte in parti:
        punto = punti.get(parte)

        if not isinstance(punto, dict):
            return None

        x = numero(punto.get("x"))
        y = numero(punto.get("y"))

        if x is None or y is None:
            return None

        coordinate[parte] = (x, y)

    xs = [p[0] for p in coordinate.values()]
    ys = [p[1] for p in coordinate.values()]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    larghezza = max(xmax - xmin, 1.0)
    altezza = max(ymax - ymin, 1.0)

    # Scala uniforme: conserva le proporzioni della proiezione.
    scala = min(210 / larghezza, 260 / altezza)

    centro_x = (xmin + xmax) / 2
    centro_y = (ymin + ymax) / 2

    proiettati = {
        parte: (
            160 + (x - centro_x) * scala,
            170 + (y - centro_y) * scala,
        )
        for parte, (x, y) in coordinate.items()
    }

    segmenti = [
        ("spalla", "anca"),
        ("anca", "ginocchio"),
        ("ginocchio", "caviglia"),
        ("caviglia", "tallone"),
        ("tallone", "punta"),
        ("punta", "caviglia"),
    ]

    linee = []

    for a, b in segmenti:
        x1, y1 = proiettati[a]
        x2, y2 = proiettati[b]

        linee.append(
            f"<line x1='{x1:.2f}' y1='{y1:.2f}' "
            f"x2='{x2:.2f}' y2='{y2:.2f}' "
            f"stroke='{C_CYAN}' stroke-width='4' "
            "stroke-linecap='round'/>"
        )

    nodi = []

    for parte, (x, y) in proiettati.items():
        colore = C_AMBER if parte == "ginocchio" else C_GREEN

        nodi.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='6' "
            f"fill='#080B12' stroke='{colore}' stroke-width='2'/>"
        )

    ginocchio = html.escape(
        formato(dati.get("angolo_ginocchio_appoggio"), 1, "°")
    )
    overstride = html.escape(
        formato(dati.get("overstride_cm"), 1, "cm")
    )
    frame = html.escape(str(dati.get("frame_strike_analizzato", "N/D")))

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="600" height="340" viewBox="0 0 600 340">
        <defs>
            <linearGradient id="cvBackground" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#0E1420"/>
                <stop offset="100%" stop-color="#04070B"/>
            </linearGradient>
            <pattern id="cvGrid" width="25" height="25"
                     patternUnits="userSpaceOnUse">
                <path d="M25 0H0V25" fill="none"
                      stroke="#1c2333" stroke-width=".5"/>
            </pattern>
        </defs>
        <rect x="1" y="1" width="598" height="338" rx="14"
              fill="url(#cvBackground)" stroke="#1c2333"/>
        <rect x="1" y="1" width="598" height="338" rx="14"
              fill="url(#cvGrid)" opacity=".55"/>

        {''.join(linee)}
        {''.join(nodi)}

        <text x="320" y="68" fill="{C_CYAN}"
              font-family="monospace" font-size="12">
            PROIEZIONE CINEMATICA 2D
        </text>
        <text x="320" y="94" fill="{C_DIM}"
              font-family="monospace" font-size="11">
            Frame rappresentativo: {frame}
        </text>

        <rect x="310" y="125" width="265" height="65"
              rx="8" fill="#0E1420" stroke="#1c2333"/>
        <text x="325" y="148" fill="{C_DIM}"
              font-family="monospace" font-size="10">
            ANGOLO GINOCCHIO — MEDIA CICLI
        </text>
        <text x="325" y="176" fill="{C_AMBER}"
              font-family="monospace" font-size="24">
            {ginocchio}
        </text>

        <rect x="310" y="208" width="265" height="65"
              rx="8" fill="#0E1420" stroke="#1c2333"/>
        <text x="325" y="231" fill="{C_DIM}"
              font-family="monospace" font-size="10">
            CAVIGLIA–ANCA ALL'IC — MEDIA
        </text>
        <text x="325" y="259" fill="{C_GREEN}"
              font-family="monospace" font-size="24">
            {overstride}
        </text>

        <text x="25" y="320" fill="{C_DIM}"
              font-family="monospace" font-size="9">
            Punti filtrati del lato analizzato · non una mappa di carico
        </text>
    </svg>
    """


def json_sicuro(valore):
    if isinstance(valore, dict):
        return {str(k): json_sicuro(v) for k, v in valore.items()}

    if isinstance(valore, (list, tuple)):
        return [json_sicuro(v) for v in valore]

    if isinstance(valore, np.ndarray):
        return json_sicuro(valore.tolist())

    if isinstance(valore, np.generic):
        return json_sicuro(valore.item())

    if isinstance(valore, float) and not np.isfinite(valore):
        return None

    return valore


# =========================================================
# STATO E SIDEBAR
# =========================================================
if st.session_state.get("dati") is None:
    st.session_state.dati = genera_dati()

st.session_state.setdefault("cv_analizzato", False)

sidebar_result = sidebar_comune()

if (
    isinstance(sidebar_result, tuple)
    and len(sidebar_result) == 3
):
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati
    df = df_full
    filtro_tempo = st.session_state.get("filtro_tempo", "Tutto")


# =========================================================
# HEADER
# =========================================================
header_block(
    "Modulo 06 — Computer Vision",
    "AI RUNNING FORM ANALYSIS",
    (
        "Carica una ripresa laterale della corsa: il sistema traccia "
        "i punti articolari e stima parametri cinematici ciclo per ciclo."
    ),
    get_svg_url(SVG_CV),
    "Pose Estimation & Kinematics",
)

st.markdown(
    """
    <div class="cv-notice">
        Analisi cinematica 2D sperimentale: angoli, posizione
        caviglia–anca, tempi e distribuzione dell'appoggio.
        Non misura forze articolari e non produce diagnosi o
        probabilità di infortunio.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT VIDEO
# =========================================================
col_upload, col_parametri = st.columns([2, 1])

with col_upload:
    video_file = st.file_uploader(
        "Carica il video della corsa",
        type=["mp4", "mov", "avi"],
        key="cv_upload_video",
        help=(
            "Ripresa laterale, corridore intero e camera ferma. "
            "Il motore richiede almeno tre cicli validi dello stesso piede."
        ),
    )

with col_parametri:
    altezza_cm = st.number_input(
        "Altezza dichiarata (cm)",
        min_value=120,
        max_value=220,
        value=175,
        step=1,
        key="cv_altezza",
        help=(
            "Usata per una scala antropometrica indicativa, "
            "non per una calibrazione metrica."
        ),
    )

    complessita = st.selectbox(
        "Modello di tracking",
        options=[0, 1, 2],
        index=1,
        format_func=lambda v: {
            0: "Leggero",
            1: "Standard",
            2: "Completo",
        }[v],
        key="cv_complessita",
    )

if video_file is None:
    reset_analisi()
    st.session_state.pop("cv_input_firma", None)

    st.info(
        "Carica un video per iniziare. La ripresa deve contenere "
        "abbastanza cicli completi per consentire l'analisi."
    )
    st.stop()

video_bytes = video_file.getvalue()

if not video_bytes:
    st.error("Il file caricato è vuoto.")
    st.stop()

dimensione_mb = len(video_bytes) / (1024 * 1024)

if dimensione_mb > MAX_VIDEO_MB:
    st.error(
        f"Il video occupa {dimensione_mb:.1f} MB. "
        f"Il limite impostato per questa pagina è {MAX_VIDEO_MB} MB."
    )
    st.stop()

firma = (
    hashlib.sha256(video_bytes).hexdigest(),
    int(altezza_cm),
    int(complessita),
    "pagina-cv-v2",
)

if st.session_state.get("cv_input_firma") != firma:
    reset_analisi()
    st.session_state["cv_input_firma"] = firma

st.caption(
    f"File: {video_file.name} · {dimensione_mb:.1f} MB · "
    f"Altezza: {altezza_cm} cm"
)

etichetta_bottone = (
    "RIELABORA IL VIDEO"
    if st.session_state.get("cv_analizzato")
    else "ELABORA SCHELETRO E CINEMATICA"
)

elabora = st.button(
    etichetta_bottone,
    use_container_width=True,
    key="cv_elabora_video",
)


# =========================================================
# ELABORAZIONE
# =========================================================
if elabora:
    reset_analisi()
    video_path = None

    with st.spinner(
        "Tracking MediaPipe e analisi dei cicli in corso..."
    ):
        try:
            suffix = Path(video_file.name).suffix.lower()

            if suffix not in {".mp4", ".mov", ".avi"}:
                suffix = ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tfile:
                tfile.write(video_bytes)
                video_path = tfile.name

            risultato = analizza_running_video(
                video_path,
                altezza_cm=altezza_cm,
                model_complexity=complessita,
            )

            if not isinstance(risultato, dict):
                raise ValueError(
                    "Il motore non ha restituito un dizionario."
                )

            richieste = {
                "lato_analizzato",
                "fps_video",
                "frame_strike_analizzato",
                "qualita",
                "cadenza_spm",
                "tempo_contatto_ms",
                "overstride_cm",
                "angolo_ginocchio_appoggio",
                "angolo_inclinazione_busto",
                "oscillazione_verticale",
                "tipo_appoggio",
                "distribuzione_appoggio_pct",
                "fasi_gait",
                "angoli_fase",
            }

            mancanti = richieste - risultato.keys()

            if mancanti:
                raise ValueError(
                    "Motore non compatibile. Chiavi mancanti: "
                    + ", ".join(sorted(mancanti))
                )

            st.session_state["cv_dati"] = risultato
            st.session_state["cv_analizzato"] = True
            st.session_state["cv_analisi_firma"] = firma

        except Exception as exc:
            logger.exception("Errore nell'analisi del video")

            st.session_state["cv_errore"] = str(exc)
            st.session_state["cv_diagnostica"] = getattr(
                exc,
                "diagnostica",
                {},
            )
            st.session_state["cv_analizzato"] = False

        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    logger.warning(
                        "Impossibile eliminare il video temporaneo.",
                        exc_info=True,
                    )


# =========================================================
# ERRORI E DIAGNOSTICA
# =========================================================
if st.session_state.get("cv_errore"):
    st.error(
        "Analisi non riuscita: "
        + st.session_state["cv_errore"]
    )

    diagnostica = st.session_state.get("cv_diagnostica", {})

    if diagnostica:
        with st.expander(
            "Perché i cicli sono stati scartati?",
            expanded=True,
        ):
            motivi = diagnostica.get("motivi_scarto", {})

            if motivi:
                df_motivi = pd.DataFrame([
                    {
                        "Motivo": motivo,
                        "Numero": conteggio,
                    }
                    for motivo, conteggio in motivi.items()
                ])

                st.dataframe(
                    df_motivi,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Uno stesso ciclo può avere più motivi di scarto."
                )

            dettagli = diagnostica.get("dettagli_cicli", [])

            if dettagli:
                df_dettagli = pd.DataFrame(dettagli)

                if "motivi_scarto" in df_dettagli.columns:
                    df_dettagli["motivi_scarto"] = (
                        df_dettagli["motivi_scarto"].apply(
                            lambda v: "; ".join(v)
                            if isinstance(v, list)
                            else str(v)
                        )
                    )

                st.dataframe(
                    df_dettagli,
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander("Diagnostica completa"):
                st.json(json_sicuro(diagnostica))

    with st.expander("Anteprima del video"):
        st.video(video_bytes)

    st.stop()

if not st.session_state.get("cv_analizzato", False):
    st.video(video_bytes)
    st.info("Premi il pulsante per avviare l'elaborazione.")
    st.stop()

if st.session_state.get("cv_analisi_firma") != firma:
    st.warning("I parametri sono cambiati: rielabora il video.")
    st.stop()


# =========================================================
# RISULTATI
# =========================================================
dati_cv = st.session_state["cv_dati"]
qualita = dati_cv.get("qualita", {})

cicli_validi = qualita.get(
    "cicli_validi",
    qualita.get("passi_validi", 0),
)
cicli_scartati = qualita.get(
    "cicli_scartati",
    qualita.get("passi_scartati", 0),
)

st.success(
    f"Analisi completata · lato {dati_cv['lato_analizzato']} · "
    f"{formato(dati_cv['fps_video'], 1, 'FPS')} · "
    f"{cicli_validi} cicli validi."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Frame rate",
    formato(dati_cv.get("fps_video"), 1, "FPS"),
)
m2.metric("Lato analizzato", dati_cv["lato_analizzato"])
m3.metric("Cicli validi", str(cicli_validi))
m4.metric(
    "Tracking completo",
    formato(qualita.get("frame_tracking_valido_pct"), 1, "%"),
)

avvisi = dati_cv.get("avvisi", [])

if avvisi:
    with st.expander("Qualità e limiti della misura", expanded=True):
        for avviso in avvisi:
            st.write(f"- {avviso}")

st.markdown("---")


# =========================================================
# VIDEO E SCHELETRO
# =========================================================
col_video, col_scheletro = st.columns([1, 1.1])

with col_video:
    st.markdown(
        "<p class='cv-eyebrow'>Sorgente video</p>",
        unsafe_allow_html=True,
    )
    st.video(video_bytes)

    st.caption(
        "Video originale: non contiene una sovrapposizione "
        "animata dello scheletro."
    )

with col_scheletro:
    st.markdown(
        "<p class='cv-eyebrow'>Kinematic wireframe · tracking 2D</p>",
        unsafe_allow_html=True,
    )

    scheletro = svg_scheletro(dati_cv)

    if scheletro:
        render_svg(scheletro, altezza=350)
    else:
        st.info(
            "Coordinate dello scheletro non disponibili. "
            "Usa il motore aggiornato per visualizzarle."
        )

    st.caption(
        "Scheletro del frame rappresentativo, costruito dalle "
        "coordinate filtrate del lato analizzato. "
        "Le metriche a destra sono medie dei cicli validi, "
        "non misure del solo frame mostrato."
    )


# =========================================================
# METRICHE
# =========================================================
st.header("Report cinematico")

a1, a2, a3, a4 = st.columns(4)

a1.metric(
    "Angolo ginocchio all'IC",
    formato(dati_cv.get("angolo_ginocchio_appoggio"), 1, "°"),
)
a2.metric(
    "Inclinazione busto",
    formato(dati_cv.get("angolo_inclinazione_busto"), 1, "°"),
)
a3.metric(
    "Caviglia–anca all'IC",
    formato(dati_cv.get("overstride_cm"), 1, "cm"),
)
a4.metric(
    "Oscillazione verticale",
    formato(dati_cv.get("oscillazione_verticale"), 1, "cm"),
)

b1, b2, b3 = st.columns(3)

b1.metric(
    "Cadenza stimata",
    formato(dati_cv.get("cadenza_spm"), 1, "passi/min"),
)
b2.metric(
    "Contatto stimato",
    formato_stat(dati_cv.get("tempo_contatto_ms"), "ms"),
)
b3.metric(
    "Foot strike angle",
    formato_stat(dati_cv.get("foot_strike_angle_gradi"), "°"),
)

st.caption(
    "IC = contatto iniziale stimato. Un valore positivo della "
    "distanza caviglia–anca indica una caviglia davanti all'anca "
    "nella direzione di corsa. Le misure in cm usano una scala "
    "antropometrica indicativa. Media ± SD, dove disponibile."
)

st.info(f"Classificazione dell'appoggio: {dati_cv['tipo_appoggio']}")

st.caption(
    "L'etichetta deriva dalle soglie FSA configurate nel prototipo. "
    "Non è una diagnosi e non indica da sola un errore tecnico."
)


# =========================================================
# GRAFICI
# =========================================================
grafici_export = []
g1, g2, g3 = st.columns(3)

with g1:
    st.subheader("1. Appoggio rilevato")

    distribuzione = dati_cv["distribuzione_appoggio_pct"]

    df_appoggio = pd.DataFrame([
        {
            "Appoggio": nome.capitalize(),
            "Percentuale": numero(valore),
        }
        for nome, valore in distribuzione.items()
    ]).dropna(subset=["Percentuale"])

    if not df_appoggio.empty:
        fig_appoggio = px.bar(
            df_appoggio,
            x="Appoggio",
            y="Percentuale",
            color="Appoggio",
            color_discrete_map={
                "Tallone": C_CYAN,
                "Mesopiede": C_GREEN,
                "Avampiede": C_AMBER,
            },
            text="Percentuale",
        )

        fig_appoggio.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="auto",
        )
        fig_appoggio.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Cicli validi (%)",
        )
        fig_appoggio.update_yaxes(range=[0, 100])
        fig_appoggio = figura_stilizzata(fig_appoggio)

        st.plotly_chart(fig_appoggio, use_container_width=True)
        grafici_export.append(
            ("Distribuzione dell'appoggio", fig_appoggio)
        )

    st.caption(
        "Distribuzione delle classi di appoggio nei cicli validi "
        "del lato analizzato, non percentuali di carico."
    )

with g2:
    st.subheader("2. Angoli nelle fasi")

    fasi = list(dati_cv["fasi_gait"])
    angoli = [numero(v) for v in dati_cv["angoli_fase"]]

    if (
        len(fasi) == len(angoli)
        and len(fasi) > 0
        and all(v is not None for v in angoli)
    ):
        fig_angoli = go.Figure(go.Scatterpolar(
            r=angoli + [angoli[0]],
            theta=fasi + [fasi[0]],
            fill="toself",
            fillcolor="rgba(0,245,160,0.12)",
            line=dict(color=C_GREEN, width=2),
            marker=dict(color=C_GREEN, size=6),
            hovertemplate="%{theta}: %{r:.1f}°<extra></extra>",
        ))

        fig_angoli = figura_stilizzata(fig_angoli)
        fig_angoli.update_layout(
            showlegend=False,
            polar=dict(
                bgcolor="#080B12",
                radialaxis=dict(
                    visible=True,
                    range=[0, 180],
                    gridcolor="#1c2333",
                    tickfont=dict(color=C_DIM),
                ),
                angularaxis=dict(
                    gridcolor="#1c2333",
                    tickfont=dict(color=C_DIM, size=9),
                ),
            ),
        )

        st.plotly_chart(fig_angoli, use_container_width=True)
        grafici_export.append(
            ("Angolo incluso del ginocchio nelle fasi", fig_angoli)
        )

    st.caption(
        "Angolo incluso anca–ginocchio–caviglia. "
        "180° corrisponde alla gamba estesa. "
        "La flessione è 180° meno questo angolo."
    )

with g3:
    st.subheader("3. Qualità dell'analisi")

    fig_qualita = go.Figure(go.Bar(
        x=["Validi", "Scartati"],
        y=[cicli_validi, cicli_scartati],
        marker_color=[C_GREEN, C_ORANGE],
        text=[cicli_validi, cicli_scartati],
        textposition="auto",
    ))

    fig_qualita.update_layout(
        xaxis_title="",
        yaxis_title="Numero di cicli",
    )
    fig_qualita = figura_stilizzata(fig_qualita)

    st.plotly_chart(fig_qualita, use_container_width=True)
    grafici_export.append(
        ("Cicli validi e scartati", fig_qualita)
    )

    st.caption(
        "I cicli sono intervalli tra contatti successivi dello "
        "stesso piede. Non sono singoli passi alternati."
    )


# =========================================================
# VARIABILITÀ CICLO PER CICLO
# =========================================================
st.header("Misure ciclo per ciclo")

per_ciclo = dati_cv.get("metriche_per_ciclo", {})

df_cicli = pd.DataFrame()

if isinstance(per_ciclo, dict) and per_ciclo:
    lunghezze = {
        len(v)
        for v in per_ciclo.values()
        if isinstance(v, (list, tuple))
    }

    if (
        len(lunghezze) == 1
        and all(
            isinstance(v, (list, tuple))
            for v in per_ciclo.values()
        )
    ):
        df_cicli = pd.DataFrame(per_ciclo)
        df_cicli.insert(
            0,
            "Ciclo",
            np.arange(1, len(df_cicli) + 1),
        )

if not df_cicli.empty:
    vc1, vc2 = st.columns(2)

    with vc1:
        fig_overstride = px.line(
            df_cicli,
            x="Ciclo",
            y="over_cm",
            markers=True,
            labels={
                "over_cm": "Caviglia–anca all'IC (cm)",
                "Ciclo": "Ciclo valido",
            },
        )
        fig_overstride.update_traces(
            line_color=C_CYAN,
            marker_color=C_CYAN,
        )
        fig_overstride = figura_stilizzata(
            fig_overstride,
            altezza=280,
        )

        st.plotly_chart(fig_overstride, use_container_width=True)
        grafici_export.append(
            ("Distanza caviglia–anca per ciclo", fig_overstride)
        )

    with vc2:
        fig_contatto = px.line(
            df_cicli,
            x="Ciclo",
            y="contatto",
            markers=True,
            labels={
                "contatto": "Contatto stimato (ms)",
                "Ciclo": "Ciclo valido",
            },
        )
        fig_contatto.update_traces(
            line_color=C_AMBER,
            marker_color=C_AMBER,
        )
        fig_contatto = figura_stilizzata(
            fig_contatto,
            altezza=280,
        )

        st.plotly_chart(fig_contatto, use_container_width=True)
        grafici_export.append(
            ("Tempo di contatto per ciclo", fig_contatto)
        )

    colonne_tabella = {
        "stride": "Durata ciclo (s)",
        "contatto": "Contatto (ms)",
        "over_cm": "Caviglia–anca (cm)",
        "over_pct": "Caviglia–anca (% arto)",
        "tibia": "Tibia all'IC (°)",
        "flex_ic": "Flessione IC (°)",
        "flex_stance": "Max flessione appoggio (°)",
        "flex_to": "Flessione stacco (°)",
        "flex_swing": "Max flessione oscillazione (°)",
        "fsa": "FSA (°)",
        "busto": "Busto (°)",
        "vo": "Oscillazione verticale (cm)",
    }

    with st.expander("Tabella delle misure"):
        st.dataframe(
            df_cicli.rename(columns=colonne_tabella).round(2),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info("Le misure per ciclo non sono disponibili nel risultato.")


# =========================================================
# DIAGNOSTICA E METODOLOGIA
# =========================================================
st.header("Come leggere il risultato")

st.markdown(
    """
    <div class="cv-card">
        <p class="cv-eyebrow">Interpretazione</p>
        <p class="cv-description">
            Il sistema descrive il movimento osservato, non decide
            se la tua tecnica sia corretta o se svilupperai un
            infortunio. Eventuali modifiche tecniche vanno valutate
            nel contesto dell'allenamento e delle tue condizioni.
            Il video e le metriche possono essere condivisi con
            un tecnico o un professionista per un confronto.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Diagnostica dei cicli"):
    diagnostica = dati_cv.get("diagnostica", {})
    dettagli = diagnostica.get("dettagli_cicli", [])

    if dettagli:
        df_diagnostica = pd.DataFrame(dettagli)

        if "motivi_scarto" in df_diagnostica.columns:
            df_diagnostica["motivi_scarto"] = (
                df_diagnostica["motivi_scarto"].apply(
                    lambda v: "; ".join(v)
                    if isinstance(v, list)
                    else str(v)
                )
            )

        st.dataframe(
            df_diagnostica,
            hide_index=True,
            use_container_width=True,
        )

    st.json(json_sicuro(diagnostica))

with st.expander("Definizioni e note metodologiche"):
    for nota in dati_cv.get("note_metodologiche", []):
        st.write(f"- {nota}")

    st.write(
        "Lo scheletro mostrato rappresenta un singolo frame "
        "filtrato. Le metriche principali riassumono tutti "
        "i cicli accettati."
    )


# =========================================================
# EXPORT
# =========================================================
st.header("Esporta l'analisi")

generato = datetime.now().astimezone().isoformat(
    timespec="seconds"
)

report_righe = [
    "RUNAI — ANALISI CINEMATICA 2D",
    f"Generato: {generato}",
    f"Video: {video_file.name}",
    f"Altezza dichiarata: {altezza_cm} cm",
    f"Lato analizzato: {dati_cv['lato_analizzato']}",
    f"Direzione: {dati_cv.get('direzione_corsa', 'N/D')}",
    f"FPS: {formato(dati_cv.get('fps_video'))}",
    f"Cicli validi: {cicli_validi}",
    f"Cicli scartati: {cicli_scartati}",
    (
        "Tracking completo: "
        + formato(
            qualita.get("frame_tracking_valido_pct"),
            1,
            "%",
        )
    ),
    "",
    "METRICHE",
    (
        "Cadenza stimata: "
        + formato(dati_cv.get("cadenza_spm"), 1, "passi/min")
    ),
    (
        "Contatto stimato: "
        + formato_stat(dati_cv.get("tempo_contatto_ms"), "ms")
    ),
    (
        "Angolo ginocchio all'IC: "
        + formato(
            dati_cv.get("angolo_ginocchio_appoggio"),
            1,
            "°",
        )
    ),
    (
        "Caviglia–anca all'IC: "
        + formato_stat(
            dati_cv.get("overstride", {}).get("cm"),
            "cm",
        )
    ),
    (
        "Inclinazione busto: "
        + formato_stat(
            dati_cv.get("inclinazione_busto_gradi"),
            "°",
        )
    ),
    (
        "Oscillazione verticale: "
        + formato_stat(
            dati_cv.get("oscillazione_verticale_cm"),
            "cm",
        )
    ),
    (
        "Foot strike angle: "
        + formato_stat(
            dati_cv.get("foot_strike_angle_gradi"),
            "°",
        )
    ),
    f"Appoggio prevalente: {dati_cv['tipo_appoggio']}",
    "",
    "DISTRIBUZIONE APPOGGIO",
]

for nome, valore in dati_cv["distribuzione_appoggio_pct"].items():
    report_righe.append(
        f"{nome}: {formato(valore, 1, '%')}"
    )

report_righe.extend(["", "AVVISI"])
report_righe.extend(
    f"- {avviso}"
    for avviso in dati_cv.get("avvisi", [])
)

report_righe.extend(["", "METODOLOGIA"])
report_righe.extend(
    f"- {nota}"
    for nota in dati_cv.get("note_metodologiche", [])
)

report_righe.extend([
    "",
    "Analisi sperimentale e descrittiva, non diagnostica.",
    "Non vengono stimate probabilità di infortunio, "
    "forze al suolo o percentuali di carico articolare.",
    "Il video originale non è incluso nel report HTML.",
])

report_txt = "\n".join(report_righe)

blocchi_html = []

for i, (titolo, fig) in enumerate(grafici_export):
    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True if i == 0 else False,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    blocchi_html.append(
        "<section>"
        f"<h2>{html.escape(titolo)}</h2>"
        f"{fig_html}"
        "</section>"
    )

scheletro_export = svg_scheletro(dati_cv) or ""

tabella_export = (
    df_cicli.round(3).to_html(
        index=False,
        escape=True,
        border=0,
    )
    if not df_cicli.empty
    else ""
)

report_html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RunAI — Analisi cinematica</title>
<style>
body {{
    margin: 0 auto;
    padding: 24px;
    max-width: 1150px;
    background: #080B12;
    color: #E8ECF2;
    font-family: Arial, sans-serif;
    line-height: 1.6;
}}
h1 {{ color: #00E5FF; }}
h2 {{ color: #00F5A0; font-size: 1.2rem; }}
section {{
    background: #0E1420;
    border: 1px solid #1c2333;
    border-radius: 12px;
    padding: 18px;
    margin: 18px 0;
    overflow-x: auto;
}}
pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: inherit;
}}
svg {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; font-size: .8rem; }}
th, td {{
    padding: 8px;
    border-bottom: 1px solid #263044;
    text-align: right;
}}
th {{ color: #00E5FF; }}
</style>
</head>
<body>
<h1>RunAI — Analisi cinematica 2D</h1>
<section><pre>{html.escape(report_txt)}</pre></section>
<section>
<h2>Proiezione cinematica del frame rappresentativo</h2>
{scheletro_export}
</section>
{''.join(blocchi_html)}
<section>
<h2>Misure per ciclo</h2>
{tabella_export}
</section>
</body>
</html>
"""

e1, e2, e3 = st.columns(3)

with e1:
    st.download_button(
        "Scarica report TXT",
        data=report_txt,
        file_name="runai_analisi_cinematica.txt",
        mime="text/plain; charset=utf-8",
        use_container_width=True,
    )

with e2:
    st.download_button(
        "Scarica HTML con grafici",
        data=report_html,
        file_name="runai_analisi_cinematica.html",
        mime="text/html; charset=utf-8",
        use_container_width=True,
    )

with e3:
    st.download_button(
        "Scarica risultati JSON",
        data=json.dumps(
            json_sicuro(dati_cv),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        file_name="runai_analisi_cinematica.json",
        mime="application/json",
        use_container_width=True,
    )
