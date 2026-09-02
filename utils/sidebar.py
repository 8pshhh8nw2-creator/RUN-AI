import streamlit as st

# =========================================================
#   CSS CONDIVISO (design system RUNAI)
# =========================================================
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12; --panel: #0E1420; --line: #1c2333;
        --cyan: #00E5FF; --mint: #00F5A0; --text: #E8ECF2; --text-dim: #8792A3;
    }
    .stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] { background-color: #070B12 !important; border-right: 1px solid #161D2B; }
    section[data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; min-height: 100vh; }

    .runai-card {
        background: linear-gradient(180deg, #0E1420 0%, #0A0F18 100%);
        border: 1px solid #1c2333; border-radius: 10px; padding: 16px;
    }
    .runai-label {
        color: #566178; font-size: 0.68em; font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 8px 2px;
    }
    .runai-row { display: flex; justify-content: space-between; margin: 7px 0; font-family: "JetBrains Mono", monospace; font-size: 0.9em; }
    .runai-row span:first-child { color: #8792A3; font-family: "Inter", sans-serif; }
    .runai-row span:last-child { color: #E8ECF2; font-weight: 600; }
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


def sidebar_comune():
    """
    Disegna la sidebar comune a tutte le pagine (logo, connessione device,
    filtro temporale) e applica il CSS del design system.
    Va chiamata all'inizio di ogni file dentro pages/.
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
                    <div style='color: #00F5A0; font-family:"JetBrains Mono",monospace; font-size:0.75em; margin-bottom:6px;'>&#9679; LIVE SYNC ACTIVE</div>
                    <div class='runai-row'><span>Dispositivo</span><span>{info['nome']}</span></div>
                    <div class='runai-row'><span>FC</span><span>{info['fc']} bpm</span></div>
                    <div class='runai-row'><span>Batteria</span><span>{info['battery']}%</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown("<p class='runai-label'>Filtro Temporale</p>", unsafe_allow_html=True)
        filtro_tempo = st.selectbox(
            "Intervallo",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Tutto"],
            label_visibility="collapsed",
            key="sb_filtro_tempo",
        )
        st.session_state.filtro_tempo = filtro_tempo

    return filtro_tempo
