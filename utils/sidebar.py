import streamlit as st
import pandas as pd
from datetime import timedelta

# =========================================================
#   CSS CONDIVISO (design system RUNAI)
# =========================================================
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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

    /* --- Bottoni sidebar (connetti / disconnetti) --- */
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
    /* variante "disconnetti", più discreta */
    section[data-testid="stSidebar"] .runai-disconnect .stButton > button {
        background: transparent;
        border: 1px solid var(--line);
        color: var(--text-faint);
    }
    section[data-testid="stSidebar"] .runai-disconnect .stButton > button:hover {
        border-color: #E85D5D;
        color: #E85D5D;
        background: rgba(232,93,93,0.06);
    }

    .runai-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%);
        border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
    }
    /* L'animazione di ingresso si applica SOLO al primo render della card
       (vedi flag _device_card_shown_once in Python) cosi' non "sfarfalla"
       ad ogni interazione/rerun: e' quello che dava la sensazione di lentezza. */
    .runai-card.runai-card-enter {
        animation: runai-fadein 0.35s var(--ease);
    }
    @keyframes runai-fadein {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .runai-label {
        color: var(--text-faint); font-size: 0.66em; font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.16em; text-transform: uppercase; margin: 0 0 10px 2px;
    }

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

    /* --- Griglia stat compatta (FC / batteria affiancate) --- */
    .runai-stat-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
        margin-top: 12px;
    }
    .runai-stat {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--line); border-radius: 9px;
        padding: 8px 10px;
    }
    .runai-stat-top {
        display: flex; align-items: center; gap: 6px;
        color: var(--text-dim); font-size: 0.66em;
        font-family: "JetBrains Mono", monospace; letter-spacing: 0.06em;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .runai-stat-top svg { width: 12px; height: 12px; flex-shrink: 0; }
    .runai-stat-value {
        font-family: "JetBrains Mono", monospace; font-weight: 600;
        font-size: 1.05em; color: var(--text);
    }
    .runai-battery-track {
        width: 100%; height: 4px; border-radius: 2px;
        background: rgba(255,255,255,0.06); margin-top: 6px; overflow: hidden;
    }
    .runai-battery-fill {
        height: 100%; border-radius: 2px;
        transition: width 0.4s var(--ease);
    }
    .runai-device-name {
        color: var(--text); font-size: 0.86em; font-weight: 600;
        margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    /* =========================================================
       NAV PAGINE - pulita, leggibile, stabile al click
    ========================================================= */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        margin-top: 20px;
        padding: 14px 0 0 0;
        position: relative;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: rgba(0,229,255,0.18);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > span,
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div > span {
        display: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        padding: 0; margin: 0;
        display: flex; flex-direction: column; gap: 2px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        list-style: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        display: flex; align-items: center;
        color: var(--text) !important;
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.12em;
        font-weight: 600;
        letter-spacing: -0.01em;
        padding: 11px 12px;
        margin: 0;
        border-radius: 8px;
        text-decoration: none !important;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(255,255,255,0.05);
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(0,229,255,0.10);
        color: #ffffff !important;
        font-weight: 700;
        box-shadow: inset 3px 0 0 var(--cyan);
    }

    /* --- Transizione fluida al cambio pagina ---
       Ogni volta che una pagina viene (ri)renderizzata, il contenuto
       principale entra con una dissolvenza morbida invece di comparire
       di scatto: e' questo che rende il cambio pagina "fluido". */
    [data-testid="stAppViewContainer"] .main .block-container {
        animation: runai-page-enter 0.28s var(--ease);
    }
    @keyframes runai-page-enter {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

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

_ICON_BOLT = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>"""
_ICON_HEART = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l1 1L12 21l7.8-7.8 1-1a5.5 5.5 0 0 0 0-7.6z"/></svg>"""
_ICON_BATTERY = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="18" height="10" rx="2"/><line x1="22" y1="10" x2="22" y2="14"/></svg>"""


def _init_state():
    """Inizializza in modo sicuro le chiavi di session_state usate dalla sidebar."""
    if "device_connected" not in st.session_state:
        st.session_state.device_connected = False
    if "device_info" not in st.session_state:
        st.session_state.device_info = None
    if "filtro_tempo" not in st.session_state:
        st.session_state.filtro_tempo = "Ultimi 30 giorni"
    if "_device_card_shown_once" not in st.session_state:
        st.session_state._device_card_shown_once = False


@st.cache_data(show_spinner=False)
def _filtra_per_tempo(df_full: pd.DataFrame, filtro_tempo: str) -> pd.DataFrame:
    """Applica il filtro temporale al DataFrame, se presente una colonna data riconoscibile.
    Cachata: evita di ripetere il filtro ad ogni rerun quando dati e filtro non cambiano."""
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


def _battery_color(pct: int) -> str:
    if pct >= 50:
        return "var(--mint)"
    if pct >= 20:
        return "var(--amber)"
    return "#E85D5D"


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
            "<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; "
            "font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>"
            "Performance Intelligence</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<p class='runai-label'>Dispositivo</p>", unsafe_allow_html=True)

        connesso = st.session_state.get("device_connected", False)
        info = st.session_state.get("device_info")

        if not connesso:
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
                st.session_state._device_card_shown_once = False  # rifai l'animazione una volta
                st.rerun()
        elif info:
            # la card entra in animazione solo la prima volta che viene mostrata
            card_class = "runai-card"
            if not st.session_state._device_card_shown_once:
                card_class += " runai-card-enter"
                st.session_state._device_card_shown_once = True

            batt = info["battery"]
            st.markdown(
                f"""
                <div class='{card_class}'>
                    <div style='color: #00F5A0; font-family:"JetBrains Mono",monospace; font-size:0.75em; display:flex; align-items:center;'>
                        <span class='runai-live-dot'></span>LIVE SYNC ACTIVE
                    </div>
                    <div class='runai-device-name'>{info['nome']}</div>
                    <div class='runai-stat-grid'>
                        <div class='runai-stat'>
                            <div class='runai-stat-top'>{_ICON_HEART}FC</div>
                            <div class='runai-stat-value'>{info['fc']} <span style='font-size:0.6em; color:var(--text-dim); font-weight:500;'>bpm</span></div>
                        </div>
                        <div class='runai-stat'>
                            <div class='runai-stat-top'>{_ICON_BATTERY}Batteria</div>
                            <div class='runai-stat-value'>{batt}%</div>
                            <div class='runai-battery-track'>
                                <div class='runai-battery-fill' style='width:{batt}%; background:{_battery_color(batt)};'></div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='runai-disconnect' style='margin-top:8px;'>", unsafe_allow_html=True)
            if st.button("DISCONNETTI", use_container_width=True, key="sb_disconnect_btn"):
                st.session_state.device_connected = False
                st.session_state.device_info = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown("<p class='runai-label'>Filtro Temporale</p>", unsafe_allow_html=True)
        filtro_tempo = st.selectbox(
            "Intervallo",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Tutto"],
            label_visibility="collapsed",
            key="sb_filtro_tempo",
        )
        st.session_state.filtro_tempo = filtro_tempo

    df_full = st.session_state.get("dati", pd.DataFrame())
    df = _filtra_per_tempo(df_full, filtro_tempo)

    return df, df_full, filtro_tempo
