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
        --bg: #080B12; --panel: #0E1420; --line: #1a2130;
        --cyan: #00E5FF; --mint: #00F5A0; --amber: #FFB020;
        --text: #E8ECF2; --text-dim: #7A8499; --text-faint: #4A5568;
        --ease: cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070A10 0%, #05070C 100%) !important;
        border-right: 1px solid #12151f;
    }
    section[data-testid="stSidebar"] > div:first-child {
        display: flex; flex-direction: column; min-height: 100vh;
        padding-top: 6px; padding-bottom: 8px;
    }

    /* Ordine: contenuto nostro (device + filtro) sopra, nav pagine sotto */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { order: 1; }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { order: 2; }

    /* --- Selectbox --- */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.02) !important;
        border: 1px solid var(--line) !important;
        border-radius: 9px !important;
        transition: border-color 0.2s var(--ease), background-color 0.2s var(--ease);
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #2a3348 !important;
        background-color: rgba(255,255,255,0.035) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 3px rgba(0,229,255,0.10);
    }

    /* --- Bottone connetti --- */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(0,229,255,0.06);
        border: 1px solid rgba(0,229,255,0.28);
        color: var(--cyan);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.74em;
        letter-spacing: 0.08em;
        border-radius: 9px;
        padding: 0.5em 0.8em;
        transition: all 0.2s var(--ease);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,229,255,0.14);
        border-color: var(--cyan);
        color: #ffffff;
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
    }

    .runai-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%);
        border: 1px solid var(--line); border-radius: 12px; padding: 16px;
        animation: runai-fadein 0.3s var(--ease);
    }
    @keyframes runai-fadein {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .runai-label {
        color: var(--text-faint); font-size: 0.66em; font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.16em; text-transform: uppercase; margin: 0 0 10px 2px;
    }
    .runai-row { display: flex; justify-content: space-between; margin: 8px 0; font-family: "JetBrains Mono", monospace; font-size: 0.88em; }
    .runai-row span:first-child { color: var(--text-dim); font-family: "Inter", sans-serif; }
    .runai-row span:last-child { color: var(--text); font-weight: 600; }

    .runai-live-dot {
        display: inline-block; width: 6px; height: 6px; border-radius: 50%;
        background: var(--mint); margin-right: 7px;
        box-shadow: 0 0 0 0 rgba(0,245,160,0.6);
        animation: runai-pulse 2s infinite;
    }
    @keyframes runai-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0,245,160,0.45); }
        70%  { box-shadow: 0 0 0 6px rgba(0,245,160,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,245,160,0); }
    }

    /* =========================================================
       NAV PAGINE — restyle fluido
    ========================================================= */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        margin-top: 28px;
        padding: 18px 6px 0 6px;
        position: relative;
        max-height: none;
        overflow: visible;
    }
    /* linea sottile invece di un bordo netto */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "";
        position: absolute; top: 0; left: 6px; right: 6px; height: 1px;
        background: linear-gradient(90deg, rgba(0,229,255,0.25), transparent 70%);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::after {
        content: "Naviga";
        display: block;
        color: var(--text-faint);
        font-size: 0.66em;
        font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 14px 0 6px 8px;
        order: -1;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > span,
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div > span {
        display: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        padding: 0; margin: 0;
        display: flex; flex-direction: column; gap: 1px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        list-style: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        position: relative;
        display: flex; align-items: center;
        color: var(--text-dim) !important;
        font-family: "Inter", sans-serif;
        font-size: 0.87em;
        font-weight: 500;
        padding: 9px 12px 9px 16px;
        border-radius: 8px;
        text-decoration: none !important;
        transition: color 0.2s var(--ease), background-color 0.25s var(--ease), padding-left 0.2s var(--ease);
    }
    /* indicatore attivo: barretta verticale che appare/scorre invece di un bordo fisso */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a::before {
        content: "";
        position: absolute; left: 0; top: 50%;
        width: 3px; height: 0;
        background: linear-gradient(180deg, var(--cyan), var(--mint));
        border-radius: 0 3px 3px 0;
        transform: translateY(-50%);
        transition: height 0.25s var(--ease);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(255,255,255,0.03);
        color: var(--text) !important;
        padding-left: 20px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(90deg, rgba(0,229,255,0.09), transparent 85%);
        color: #ffffff !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]::before {
        height: 60%;
    }

    .runai-footer {
        order: 3;
        margin-top: auto;
        padding: 18px 8px 2px 8px;
        color: var(--text-faint);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.65em;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.7;
    }

    /* scroll generale più morbido se il contenuto della sidebar eccede */
    section[data-testid="stSidebar"] > div:first-child {
        scrollbar-width: thin;
        scrollbar-color: #232b3d transparent;
    }
    section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { width: 4px; }
    section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: #232b3d; border-radius: 4px;
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

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
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
