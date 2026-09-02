import streamlit as st
import pandas as pd
from datetime import timedelta

# =========================================================
#   CSS CONDIVISO (design system RUNAI)
# =========================================================
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12; --panel: #0E1420; --line: #1c2333;
        --cyan: #00E5FF; --mint: #00F5A0; --amber: #FFB020;
        --text: #E8ECF2; --text-dim: #8792A3;
    }
    .stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }

    /* --- Contenitore sidebar: flex column, per poter riordinare i blocchi --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070B12 0%, #060910 100%) !important;
        border-right: 1px solid #161D2B;
    }
    section[data-testid="stSidebar"] > div:first-child {
        display: flex; flex-direction: column; min-height: 100vh; padding-top: 4px;
    }

    /* --- Riordino: il contenuto scritto da noi (device + filtro) va PRIMA,
           la nav automatica delle pagine va DOPO --- */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { order: 1; }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { order: 2; margin-top: 4px; }

    /* --- Selectbox più curata --- */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        transition: border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #2a3348 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 1px rgba(0,229,255,0.25);
    }

    /* --- Bottone connetti --- */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(0,245,160,0.12));
        border: 1px solid rgba(0,229,255,0.35);
        color: var(--cyan);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.78em;
        letter-spacing: 0.08em;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,229,255,0.22), rgba(0,245,160,0.22));
        border-color: var(--cyan);
        color: #ffffff;
    }

    .runai-card {
        background: linear-gradient(180deg, #0E1420 0%, #0A0F18 100%);
        border: 1px solid var(--line); border-radius: 10px; padding: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .runai-label {
        color: #566178; font-size: 0.68em; font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 8px 2px;
    }
    .runai-row { display: flex; justify-content: space-between; margin: 7px 0; font-family: "JetBrains Mono", monospace; font-size: 0.9em; }
    .runai-row span:first-child { color: var(--text-dim); font-family: "Inter", sans-serif; }
    .runai-row span:last-child { color: var(--text); font-weight: 600; }

    /* --- Badge live sync con pulse --- */
    .runai-live-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: var(--mint); margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(0,245,160,0.6);
        animation: runai-pulse 1.8s infinite;
    }
    @keyframes runai-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0,245,160,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(0,245,160,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,245,160,0); }
    }

    /* --- Restyle della nav automatica delle pagine --- */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        border-top: 1px solid var(--line);
        padding-top: 14px;
        padding-left: 0; padding-right: 0;
        position: relative;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "Selezione Pagine";
        display: block;
        color: #566178;
        font-size: 0.68em;
        font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 10px 2px;
    }
    /* nasconde l'eventuale intestazione "app" generata di default */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > span,
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div > span {
        display: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        padding: 0; margin: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        list-style: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        display: flex; align-items: center;
        color: var(--text-dim) !important;
        font-family: "Inter", sans-serif;
        font-size: 0.88em;
        font-weight: 500;
        padding: 8px 10px;
        margin: 2px 0;
        border-radius: 8px;
        border-left: 2px solid transparent;
        text-decoration: none !important;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(0,229,255,0.06);
        color: var(--text) !important;
        border-left-color: rgba(0,229,255,0.4);
    }
    /* pagina attiva */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(0,245,160,0.03));
        color: var(--cyan) !important;
        font-weight: 600;
        border-left-color: var(--cyan);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        max-height: 300px;
        overflow-y: auto;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::-webkit-scrollbar { width: 4px; }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::-webkit-scrollbar-thumb {
        background: var(--line); border-radius: 4px;
    }

    .runai-footer {
        order: 3;
        margin-top: auto; padding-top: 18px; border-top: 1px solid var(--line);
        color: #3d4658; font-family: "JetBrains Mono", monospace; font-size: 0.68em;
        letter-spacing: 0.08em; text-transform: uppercase;
        padding: 18px 0 4px 2px;
    }
</style>
"""


def _init_state():
    """Inizializza in modo sicuro le chiavi di session_state usate dalla sidebar."""
    if "device_connected" not in st.session_state:
        st.session_state.device_connected = False
    if "device_info" not in st.session_state:
        st.session_state.device_info = None
    if "filtro_tempo" not in st.session_state:
        st.session_state.filtro_tempo = "Ultimi 30 giorni"


def _filtra_per_tempo(df_full: pd.DataFrame, filtro_tempo: str) -> pd.DataFrame:
    """Applica il filtro temporale al DataFrame, se presente una colonna data riconoscibile."""
    if df_full is None or df_full.empty:
        return df_full if df_full is not None else pd.DataFrame()

    colonna_data = next(
        (c for c in ["data", "date", "Data", "timestamp"] if c in df_full.columns),
        None,
    )
    if colonna_data is None or filtro_tempo == "Tutto":
        return df_full.copy()

    df = df_full.copy()
    df[colonna_data] = pd.to_datetime(df[colonna_data])
    giorni = 30 if filtro_tempo == "Ultimi 30 giorni" else 60
    soglia = df[colonna_data].max() - timedelta(days=giorni)
    return df[df[colonna_data] >= soglia]


def sidebar_comune():
    """
    Disegna la sidebar comune a tutte le pagine (logo, connessione device,
    filtro temporale, nav pagine in fondo), applica il CSS del design system
    e ritorna i dati filtrati in base al periodo selezionato.

    Va chiamata all'inizio di ogni file dentro pages/, DOPO aver popolato
    st.session_state.dati con il DataFrame generato da genera_dati().

    Ritorna:
        df (DataFrame filtrato), df_full (DataFrame completo), filtro_tempo (str)
    """
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
                <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
                <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 26px; "
            "font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>"
            "Performance Intelligence</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<p class='runai-label'>Dispositivo</p>", unsafe_allow_html=True)
        device_scelto = st.selectbox(
            "Device",
            ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3"],
            label_visibility="collapsed",
            key="sb_device_select",
        )

        if st.button("CONNETTI DISPOSITIVO", use_container_width=True, key="sb_connect_btn"):
            st.session_state.device_connected = True
            st.session_state.device_info = {
                "nome": device_scelto,
                "fc": 72,
                "battery": 88,
            }

        info = st.session_state.get("device_info")
        if st.session_state.get("device_connected", False) and info:
            st.markdown(
                f"""
                <div class='runai-card' style='margin-top: 12px;'>
                    <div style='color: #00F5A0; font-family:"JetBrains Mono",monospace; font-size:0.75em; margin-bottom:6px;'>
                        <span class='runai-live-dot'></span>LIVE SYNC ACTIVE
                    </div>
                    <div class='runai-row'><span>Dispositivo</span><span>{info['nome']}</span></div>
                    <div class='runai-row'><span>FC</span><span>{info['fc']} bpm</span></div>
                    <div class='runai-row'><span>Batteria</span><span>{info['battery']}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
        st.markdown("<p class='runai-label'>Filtro Temporale</p>", unsafe_allow_html=True)
        filtro_tempo = st.selectbox(
            "Intervallo",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Tutto"],
            label_visibility="collapsed",
            key="sb_filtro_tempo",
        )
        st.session_state.filtro_tempo = filtro_tempo

        st.markdown("<div class='runai-footer'>RUNAI · Data-Driven Training</div>", unsafe_allow_html=True)

    df_full = st.session_state.get("dati", pd.DataFrame())
    df = _filtra_per_tempo(df_full, filtro_tempo)

    return df, df_full, filtro_tempo
