import streamlit as st
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RUN AI | Performance Intelligence", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# =========================================================
#   DESIGN SYSTEM & CSS SIDEBAR
# =========================================================
st.markdown("""
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
""", unsafe_allow_html=True)

@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    fc_media = np.clip(100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n), 80, 200)
    rpe = np.clip(np.round((distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4 + np.random.normal(0, 1, n)), 1, 10)
    
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1), 'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media), 'RPE': rpe, 'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro, 'Rischio Infortunio': np.where((rpe > 7) & (ore_sonno < 6.5), 1, 0)
    })
    return df

# Inizializzazione Stato
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
if 'device_connected' not in st.session_state:
    st.session_state.device_connected = False

# =========================================================
#   DEFINIZIONE DELLE PAGINE (Funzioni o File)
# =========================================================
def home_page():
    st.title("🏠 Home / Panoramica")
    st.markdown("Benvenuto nel sistema di performance intelligence RUNAI.")
    st.metric("Sessioni Totali", len(st.session_state.dati))

def stato_forma_page():
    st.title("📈 Analisi Stato di Forma")
    st.line_chart(st.session_state.dati[['Distanza (km)', 'RPE', 'Ore Sonno']])

def statistiche_page():
    st.title("📊 Statistiche e Storico Analisi")
    st.dataframe(st.session_state.dati.tail(15), use_container_width=True)

def kpi_page():
    st.title("🧭 KPI Dashboard")
    col1, col2 = st.columns(2)
    col1.metric("KM Totali", f"{st.session_state.dati['Distanza (km)'].sum():.1f} km")
    col2.metric("Rischio Medio", f"{st.session_state.dati['Rischio Infortunio'].mean()*100:.1f}%")

def ml_page():
    st.title("🤖 Analisi Predittiva ML")
    st.info("Moduli di Machine Learning attivi per la prevenzione infortuni.")

def consiglio_page():
    st.title("💡 Consiglio Finale & Coaching")
    st.success("Mantieni costante il carico di lavoro settimanale e cura il sonno.")

def cv_page():
    st.title("👁️ Computer Vision & Postura")
    st.file_uploader("Carica video della corsa", type=["mp4", "mov"])

# =========================================================
#   NAVIGAZIONE NATIVA STREAMLIT
# =========================================================
pg = st.navigation({
    "Navigazione": [
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page(stato_forma_page, title="Stato Forma", icon="📈"),
        st.Page(statistiche_page, title="Statistiche Analisi", icon="📊"),
        st.Page(kpi_page, title="KPI Dashboard", icon="🧭"),
        st.Page(ml_page, title="Analisi Predittiva ML", icon="🤖"),
        st.Page(consiglio_page, title="Consiglio Finale", icon="💡"),
        st.Page(cv_page, title="Computer Vision", icon="👁️"),
    ]
])

# =========================================================
#   SIDEBAR PERSONALIZZATA (Comune a tutte le pagine)
# =========================================================
with st.sidebar:
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
            <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
            <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 26px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence</p>", unsafe_allow_html=True)

    st.markdown("<p class='runai-label'>Dispositivo</p>", unsafe_allow_html=True)
    device_scelto = st.selectbox("Device", ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3"], label_visibility="collapsed")

    if st.button("CONNETTI DISPOSITIVO", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_info = {'nome': device_scelto, 'fc': 72, 'battery': 88}

    if st.session_state.get('device_connected', False):
        info = st.session_state.device_info
        st.markdown(f"""
        <div class='runai-card' style='margin-top: 12px;'>
            <div style='color: #00F5A0; font-family:"JetBrains Mono",monospace; font-size:0.75em; margin-bottom:6px;'>● LIVE SYNC ACTIVE</div>
            <div class='runai-row'><span>Dispositivo</span><span>{info['nome']}</span></div>
            <div class='runai-row'><span>FC</span><span>{info['fc']} bpm</span></div>
            <div class='runai-row'><span>Batteria</span><span>{info['battery']}%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<p class='runai-label'>Filtro Temporale</p>", unsafe_allow_html=True)
    filtro_tempo = st.selectbox("Intervallo", ["Ultimi 30 giorni", "Ultimi 60 giorni", "Tutto"], label_visibility="collapsed")

# Esecuzione della pagina attiva selezionata nella navigazione nativa
pg.run()
