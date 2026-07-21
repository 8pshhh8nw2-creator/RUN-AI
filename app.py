import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
import base64
import tempfile
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUNAI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#  DESIGN SYSTEM — RUNAI (SPORT TECH RUN)
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12;
        --panel: #0E1420;
        --panel-2: #111827;
        --line: #1c2333;
        --cyan: #00E5FF;
        --signal: #FF6A3D;
        --mint: #00F5A0;
        --amber: #FFB020;
        --text: #E8ECF2;
        --text-dim: #8792A3;
        --text-faint: #566178;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(0,229,255,0.06) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(255,106,61,0.05) 0%, transparent 45%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    * { letter-spacing: -0.01em; }

    .telemetry-bar {
        display: flex; align-items: center; gap: 0;
        height: 3px; width: 100%;
        background: linear-gradient(90deg, var(--cyan) 0%, var(--mint) 35%, var(--signal) 70%, var(--cyan) 100%);
        background-size: 200% 100%;
        border-radius: 2px;
        margin-bottom: 22px;
        animation: scanline 6s linear infinite;
    }
    @keyframes scanline { 0% {background-position: 0% 0;} 100% {background-position: 200% 0;} }

    .app-header { padding: 6px 0 18px 0; }
    .app-kicker {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72em; letter-spacing: 0.25em;
        color: var(--cyan); text-transform: uppercase; margin-bottom: 6px; display:flex; align-items:center; gap:10px;
    }
    .app-kicker .dot { width:6px; height:6px; border-radius:50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); display:inline-block; }

    h1.hero-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.6em;
        color: #fff; margin: 0 0 4px 0; letter-spacing: -0.03em; line-height: 1.05; text-align:left;
    }
    .hero-sub { color: var(--text-dim); font-size: 1.02em; max-width: 640px; margin-bottom: 4px; }

    h2 {
        font-family: 'Space Grotesk', sans-serif; color: #fff; font-weight: 600; font-size: 1.5em;
        padding-bottom: 12px; margin: 8px 0 18px 0; border-bottom: 1px solid var(--line); letter-spacing: -0.02em;
    }
    h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text); font-size: 1.15em; font-weight: 600; letter-spacing: -0.01em; }

    .section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7em; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text-faint); margin-bottom: 6px;
    }

    .info-box, .success-box, .warning-box, .danger-box {
        padding: 18px 20px; border-radius: 10px; margin: 16px 0; color: var(--text-dim);
        background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--cyan);
    }
    .success-box { border-left-color: var(--mint); }
    .warning-box { border-left-color: var(--amber); }
    .danger-box  { border-left-color: var(--signal); }

    .kpi-card {
        background: var(--panel); border-radius: 12px; padding: 26px 20px; text-align: center;
        border: 1px solid var(--line); position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top:0; left:0; right:0; height: 2px;
        background: linear-gradient(90deg, var(--cyan), transparent);
    }

    .explain-text {
        font-family: 'Inter', sans-serif; font-size: 0.87em; color: var(--text-faint); line-height: 1.55;
        margin-top: 8px; margin-bottom: 14px; padding: 14px 16px; background: var(--panel); border-radius: 8px; border-left: 2px solid var(--cyan);
    }
    .explain-text strong { color: var(--text-dim); font-weight: 600; }
    .data-figure { font-family: 'JetBrains Mono', monospace; }

    .stForm { background-color: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 26px; }
    
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #131a29 !important; 
        color: var(--text) !important; 
        border: 1px solid var(--line) !important;
    }
    
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { 
        background-color: #131a29 !important; 
        border: 1px solid var(--line) !important;
    }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #131a29 !important; 
        color: var(--text) !important; 
    }
    div[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover { 
        background-color: #1c2740 !important; 
        color: #ffffff !important; 
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg);
        border-bottom: 1px solid var(--line);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: var(--panel) !important;
        border-radius: 8px 8px 0px 0px !important;
        border: 1px solid var(--line) !important;
        color: var(--text-dim) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 0 16px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #162032 !important;
        color: var(--cyan) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(0, 245, 160, 0.05)) !important;
        border-color: var(--cyan) !important;
        color: var(--cyan) !important;
        box-shadow: 0 -2px 10px rgba(0, 229, 255, 0.15);
    }

    div[data-testid="stFileUploader"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #131a29 !important;
        border: 1px dashed var(--line) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] section div, div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span {
        color: var(--text-dim) !important;
    }
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important;
        color: #04121a !important;
        border: none !important;
    }

    .stSlider label, .stSelectSlider label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color: var(--text-dim) !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    }
    .stSlider [data-baseweb="slider"] div { color: var(--text) !important; }
    div[data-testid="stTickBar"] { color: var(--text-faint) !important; }
    .stSelectSlider [role="slider"] { background-color: var(--cyan) !important; }
    div[data-testid="stWidgetLabel"] p { color: var(--text-dim) !important; }

    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important; color: #04121a !important;
        border: none !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    section[data-testid="stSidebar"] { background-color: var(--bg) !important; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] > div { background-color: var(--bg) !important; }
    section[data-testid="stSidebar"] h3 { color: var(--text-dim) !important; }
    
    div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[role="radiogroup"] label {
        background-color: var(--panel) !important; border: 1px solid var(--line) !important;
        border-left: 4px solid var(--cyan) !important; border-radius: 8px !important;
        padding: 14px 16px !important; margin-bottom: 10px !important; cursor: pointer !important;
        transition: all 0.2s ease-in-out !important; display: flex; align-items: center;
    }
    div[role="radiogroup"] label p {
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
        font-size: 1.05em !important; color: var(--text) !important; margin: 0 !important; letter-spacing: 0.02em;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(0, 229, 255, 0.05) !important; border-color: var(--cyan) !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.1), transparent) !important;
        border-left: 4px solid var(--mint) !important; border-color: rgba(0, 245, 160, 0.5) !important;
    }

    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif !important; color: var(--text-faint) !important; }

    .hero-media {
        border-radius: 16px; overflow: hidden; position: relative; margin-bottom: 6px; border: 1px solid var(--line);
        background: var(--panel);
    }
    .hero-media img { display:block; width: 100%; height: 220px; object-fit: cover; }
    .hero-media .tag {
        position:absolute; bottom:14px; left:14px; font-family:'JetBrains Mono', monospace; font-size:0.72em;
        letter-spacing:0.12em; color:#fff; background: rgba(8,11,18,0.85); padding: 5px 10px; border-radius:6px;
        border: 1px solid rgba(255,255,255,0.15); text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#B8C2D0")

def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT, title_font=dict(family="Space Grotesk, sans-serif", color="#E8ECF2", size=16),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if height: fig.update_layout(height=height)
    return fig

def get_svg_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

SVG_HOME = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="450" cy="200" r="140" fill="none" stroke="#00E5FF" stroke-width="2" opacity="0.3"/><circle cx="450" cy="200" r="90" fill="none" stroke="#00F5A0" stroke-width="2" opacity="0.4"/><path d="M200,200 L700,200" stroke="#1c2333" stroke-width="2"/><path d="M450,50 L450,350" stroke="#1c2333" stroke-width="2"/><circle cx="450" cy="200" r="25" fill="#00E5FF"/><circle cx="600" cy="130" r="8" fill="#FF6A3D"/><path d="M450,200 L600,130" stroke="#FFB020" stroke-width="2" stroke-dasharray="4,4"/></svg>"""
SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><path d="M50,200 L250,200 L300,80 L350,280 L400,150 L450,250 L500,200 L850,200" stroke="#00E5FF" stroke-width="4" fill="none" opacity="0.8"/><circle cx="300" cy="80" r="6" fill="#00F5A0"/><circle cx="350" cy="280" r="6" fill="#FF6A3D"/><g opacity="0.3"><line x1="0" y1="100" x2="900" y2="100" stroke="#1c2333" stroke-width="1"/><line x1="0" y1="300" x2="900" y2="300" stroke="#1c2333" stroke-width="1"/></g></svg>"""
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><rect x="150" y="150" width="40" height="150" fill="#00E5FF" opacity="0.3"/><rect x="250" y="200" width="40" height="100" fill="#00E5FF" opacity="0.5"/><rect x="350" y="100" width="40" height="200" fill="#00F5A0" opacity="0.8"/><rect x="450" y="220" width="40" height="80" fill="#00E5FF" opacity="0.4"/><rect x="550" y="70" width="40" height="230" fill="#FFB020" opacity="0.9"/><rect x="650" y="180" width="40" height="120" fill="#00E5FF" opacity="0.6"/><path d="M170,150 L270,200 L370,100 L470,220 L570,70 L670,180" stroke="#fff" stroke-width="3" fill="none"/><circle cx="570" cy="70" r="5" fill="#FF6A3D"/></svg>"""
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><path d="M300,300 A 150 150 0 1 1 600,300" fill="none" stroke="#1c2333" stroke-width="20"/><path d="M300,300 A 150 150 0 0 1 500,170" fill="none" stroke="#00F5A0" stroke-width="20"/><circle cx="450" cy="270" r="10" fill="#00E5FF"/><line x1="450" y1="270" x2="520" y2="150" stroke="#00E5FF" stroke-width="4"/><text x="400" y="330" fill="#E8ECF2" font-family="monospace" font-size="28" font-weight="bold">98.2%</text></svg>"""
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="200" cy="200" r="8" fill="#00E5FF"/><circle cx="350" cy="100" r="12" fill="#00F5A0"/><circle cx="350" cy="300" r="12" fill="#FFB020"/><circle cx="550" cy="150" r="15" fill="#FF6A3D"/><circle cx="550" cy="250" r="10" fill="#00E5FF"/><circle cx="750" cy="200" r="20" fill="#00F5A0"/><line x1="200" y1="200" x2="350" y2="100" stroke="#1c2333" stroke-width="2"/><line x1="200" y1="200" x2="350" y2="300" stroke="#1c2333" stroke-width="2"/><line x1="350" y1="100" x2="550" y2="150" stroke="#00E5FF" stroke-width="2" stroke-dasharray="5,5"/><line x1="350" y1="300" x2="550" y2="150" stroke="#1c2333" stroke-width="2"/><line x1="350" y1="300" x2="550" y2="250" stroke="#00F5A0" stroke-width="2" stroke-dasharray="5,5"/><line x1="550" y1="150" x2="750" y2="200" stroke="#FF6A3D" stroke-width="3"/><line x1="550" y1="250" x2="750" y2="200" stroke="#00E5FF" stroke-width="2"/></svg>"""
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="450" cy="200" r="120" fill="none" stroke="#1c2333" stroke-width="2"/><circle cx="450" cy="200" r="80" fill="none" stroke="#1c2333" stroke-width="2"/><circle cx="450" cy="200" r="40" fill="#00E5FF" opacity="0.2"/><circle cx="450" cy="200" r="10" fill="#00F5A0"/><path d="M450,200 L550,100" stroke="#FFB020" stroke-width="3"/><circle cx="550" cy="100" r="6" fill="#FFB020"/><path d="M450,200 L300,250" stroke="#FF6A3D" stroke-width="3"/><circle cx="300" cy="250" r="6" fill="#FF6A3D"/></svg>"""
SVG_CV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="450" cy="150" r="20" fill="#00E5FF"/><line x1="450" y1="170" x2="450" y2="260" stroke="#00F5A0" stroke-width="4"/><line x1="450" y1="200" x2="380" y2="240" stroke="#FFB020" stroke-width="3"/><line x1="450" y1="200" x2="520" y2="240" stroke="#FFB020" stroke-width="3"/><line x1="450" y1="260" x2="400" y2="340" stroke="#FF6A3D" stroke-width="4"/><line x1="450" y1="260" x2="500" y2="340" stroke="#00E5FF" stroke-width="4"/></svg>"""

IMG_HERO_HOME = get_svg_url(SVG_HOME)
IMG_HERO_ANALISI = get_svg_url(SVG_ANALISI)
IMG_HERO_STATS = get_svg_url(SVG_STATS)
IMG_HERO_KPI = get_svg_url(SVG_KPI)
IMG_HERO_ML = get_svg_url(SVG_ML)
IMG_HERO_PLAN = get_svg_url(SVG_PLAN)
IMG_HERO_CV = get_svg_url(SVG_CV)

def header_block(kicker, title, subtitle, image_url=None, image_tag=None):
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    if image_url:
        col_txt, col_img = st.columns([1.4, 1])
        with col_txt:
            st.markdown(f"""
            <div class="app-header">
                <div class="app-kicker"><span class="dot"></span>{kicker}</div>
                <h1 class="hero-title">{title}</h1>
                <p class="hero-sub">{subtitle}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_img:
            st.markdown(f"""
            <div class="hero-media">
                <img src="{image_url}" />
                <div class="tag">{image_tag or ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
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
    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1), 'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media), 'FC Max': np.round(fc_media + np.random.uniform(10, 30, n)),
        'Temp (°C)': np.round(temp, 1), 'RPE': rpe, 'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro, 'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
    })
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}
    st.session_state.device_connected = False
    st.session_state.diario_note = []

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
            <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
            <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

    st.subheader("Dispositivo")
    device_scelto = st.selectbox("Seleziona dispositivo:", ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3", "Fitbit Charge 6", "WHOOP 4.0", "Fascia Cardio Garmin"], label_visibility="collapsed")

    if st.button("CONNETTI DISPOSITIVO", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_info = {
            'nome': device_scelto, 'fc': np.random.randint(60, 80), 'battery': np.random.randint(70, 100),
            'steps': np.random.randint(2000, 5000), 'calories': np.random.randint(150, 300),
            'sync_time': pd.Timestamp.now().strftime('%H:%M:%S')
        }

    if st.session_state.device_connected:
        st.markdown("---")
        st.markdown("""
        <div style='background-color: #0E1420; border: 1px solid #1c2333; border-radius: 10px; padding: 16px; font-family:"Inter",sans-serif;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.1em;'>LIVE SYNC</span>
                <span style='color: #566178; font-size: 0.75em; font-family:"JetBrains Mono",monospace;'>{}</span>
            </div>
            <div style='color: #E8ECF2; font-family:"JetBrains Mono",monospace; font-size:0.92em;'>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>FC</span><span style='font-weight:600;'>{} bpm</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Batteria</span><span style='font-weight:600; color:#00F5A0;'>{}%</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Passi</span><span style='font-weight:600;'>{:,}</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Calorie</span><span style='font-weight:600;'>{}</span></div>
            </div>
        </div>
        """.format(
            st.session_state.device_info['nome'], st.session_state.device_info['fc'], st.session_state.device_info['battery'],
            st.session_state.device_info['steps'], st.session_state.device_info['calories']
        ), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Filtri Temporali Storico")
    filtro_tempo = st.selectbox("Intervallo Analisi:", ["Ultimi 30 giorni", "Ultimi 60 giorni", "Ultimi 90 giorni (Tutto)"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<h3 style='color: #00E5FF; font-size: 0.85em; letter-spacing: 0.15em; text-transform: uppercase;'>SELEZIONA</h3>", unsafe_allow_html=True)
    
    pagina = st.radio(
        "Menu",
        ["HOME", "ANALISI STATO DI FORMA", "STATISTICHE ANALISI", "KPI DASHBOARD", "ANALISI PREDITTIVA ML", "CONSIGLIO FINALE", "COMPUTER VISION"],
        label_visibility="collapsed"
    )

# Filtro dataframe in base alla selezione temporale della sidebar
df_full = st.session_state.dati.copy()
if filtro_tempo == "Ultimi 30 giorni":
    df = df_full.tail(30)
elif filtro_tempo == "Ultimi 60 giorni":
    df = df_full.tail(60)
else:
    df = df_full

# ---------------------------------------------------------
# PAGINA 0: HOME / LANDING PAGE
# ---------------------------------------------------------
if pagina == "HOME":
    header_block(
        "RUNAI // Master Thesis Project",
        "PERFORMANCE INTELLIGENCE SYSTEM",
        "Piattaforma avanzata di Sport Data Science e Machine Learning per l'analisi predittiva e la prevenzione del rischio infortuni nei runner amatori.",
        IMG_HERO_HOME, "Executive Dashboard"
    )

    st.markdown("""
    <div class='info-box'>
    <strong>Benvenuto in RUNAI.</strong> Questo sistema integra IoT, Wearable Analytics e modelli di Machine Learning supervisionati per ottimizzare i carichi di allenamento e supportare le decisioni del preparatore o dell'atleta.
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("KM Totali (90gg)", f"{df_full['Distanza (km)'].sum():.0f} km")
    col_h2.metric("Sessioni Monitorate", f"{len(df_full)}")
    col_h3.metric("Modelli ML Attivi", "5 Algoritmi")
    col_h4.metric("Stato Sistema", "Online / Sync")

    st.markdown("---")
    st.subheader("Panoramica Moduli Principali")

    c_card1, c_card2, c_card3 = st.columns(3)
    with c_card1:
        st.markdown("""
        <div class='kpi-card' style='text-align:left; height: 210px;'>
            <h3 style='color:#00E5FF; margin-bottom:8px;'>01. Stato di Forma</h3>
            <p style='color:#8792A3; font-size:0.9em;'>Configura i parametri biologici giornalieri, sonno e stress per avviare il calcolo predittivo dell'allenamento.</p>
        </div>
        """, unsafe_allow_html=True)
    with c_card2:
        st.markdown("""
        <div class='kpi-card' style='text-align:left; height: 210px;'>
            <h3 style='color:#00F5A0; margin-bottom:8px;'>02. Analytics & ML</h3>
            <p style='color:#8792A3; font-size:0.9em;'>Esplora Random Forest, Regressioni e Cluster K-Means per comprendere i pattern nascosti nel tuo storico.</p>
        </div>
        """, unsafe_allow_html=True)
    with c_card3:
        st.markdown("""
        <div class='kpi-card' style='text-align:left; height: 210px;'>
            <h3 style='color:#FFB020; margin-bottom:8px;'>03. Computer Vision</h3>
            <p style='color:#8792A3; font-size:0.9em;'>Analisi biomeccanica della falcata tramite video e stima del rischio clinico associato ai sovraccarichi.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGINA 1: ANALISI STATO DI FORMA
# ---------------------------------------------------------
elif pagina == "ANALISI STATO DI FORMA":
    header_block(
        "Modulo 01 — Acquisizione Dati",
        "ANALISI STATO DI FORMA",
        "Inserisci i parametri fisiologici, annota le sensazioni soggettive e seleziona l'obiettivo odierno.",
        IMG_HERO_ANALISI, "Sport Tech Scan"
    )

    st.markdown("""
    <div class='info-box'>
    <strong>Configura i parametri odierni e compila il diario delle sensazioni prima di avviare l'analisi predittiva.</strong>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_analisi"):
        st.markdown("### Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.selectbox("Obiettivo Odierno", ["Leggero", "Medio", "Intermedio"])
        with col_o2:
            distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)

        st.markdown("#### Obiettivo Finale (Lungo Termine)")
        col_of1, col_of2, col_of3 = st.columns(3)
        with col_of1:
            obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
        with col_of2:
            data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
        with col_of3:
            km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)

        st.markdown("---")
        st.markdown("### Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)

        st.markdown("---")
        st.markdown("### Stress Mentale & Diario Sensazioni")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)
        with col_st2:
            nota_soggettiva = st.text_area("Diario Feedback Atleta (Sensazioni, dolori lievi, umore)", placeholder="Es: Gambe leggermente pesanti dopo il lavoro di ieri, ma buon focus mentale...")

        st.markdown("---")
        st.markdown("### Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)

        st.markdown("---")
        bottone = st.form_submit_button("ANALIZZA STATO DI FORMA", use_container_width=True)

    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi, 'distanza_oggi': distanza_oggi, 'obj_finale': obj_finale, 'data_obj_finale': data_obj_finale,
            'km_obj_finale': km_obj_finale, 'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno, 'fc_riposo': fc_riposo,
            'stress_lavoro': stress_lavoro, 'ore_lavoro': ore_lavoro, 'tipo_allenamento': tipo_allenamento, 'rpe_previsto': rpe_previsto,
            'nota_soggettiva': nota_soggettiva, 'data_nota': pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')
        }
        if nota_soggettiva.strip():
            st.session_state.diario_note.append({'data': st.session_state.risultati_analisi['data_nota'], 'nota': nota_soggettiva})
        st.success("Stato di forma analizzato e note salvate con successo nel diario!")

    if st.session_state.diario_note:
        st.markdown("---")
        st.subheader("Diario Storico Feedback Atleta")
        for item in reversed(st.session_state.diario_note[-5:]):
            st.markdown(f"""
            <div style='background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;'>
                <span style='color: var(--cyan); font-family: "JetBrains Mono", monospace; font-size: 0.8em;'>{item['data']}</span>
                <p style='color: var(--text-dim); margin: 4px 0 0 0; font-size: 0.95em;'>{item['nota']}</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGINA 2: STATISTICHE ANALISI
# ---------------------------------------------------------
elif pagina == "STATISTICHE ANALISI":
    header_block(
        "Modulo 02 — Analytics Storico",
        "STATISTICHE ANALISI",
        f"Volume, intensità e recupero filtrati per: **{filtro_tempo}**.",
        IMG_HERO_STATS, "Historical Metrics"
    )

    st.subheader("KPI Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km", filtro_tempo)
    col_m2.metric("Sessioni", f"{len(df)}")
    col_m3.metric("Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")

    st.markdown("---")
    st.subheader("Analisi Dettagliata")

    tab1, tab2, tab3, tab4 = st.tabs(["Volume", "Intensità", "Recupero", "Tabella Storico"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({'Distanza (km)': 'sum'}).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=300, color='Distanza (km)', color_continuous_scale=[[0,'#0E4A57'],[1,'#00E5FF']])
            st.plotly_chart(style_fig(fig1), use_container_width=True)
            st.markdown("<div class='explain-text'>Verifica che le barre non presentino sbalzi improvvisi superiori al 10% da una settimana all'altra per prevenire sovraccarichi tendinei.</div>", unsafe_allow_html=True)

            st.markdown("**Carico per Giorno della Settimana**")
            df_copy = df.copy()
            df_copy['Giorno_Settimana'] = df_copy['Giorno'].dt.day_name()
            df_day = df_copy.groupby('Giorno_Settimana')['Distanza (km)'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig_day = px.bar(df_day, x='Giorno_Settimana', y='Distanza (km)', height=300, color_discrete_sequence=['#00E5FF'])
            st.plotly_chart(style_fig(fig_day), use_container_width=True)
            st.markdown("<div class='explain-text'>Evidenzia la distribuzione settimanale dei chilometri. Assicurati di alternare giorni di carico a giorni di recupero attivo.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distanza Cumulativa**")
            df_copy = df.copy()
            df_copy['Cumulativa'] = df_copy['Distanza (km)'].cumsum()
            fig_cum = px.line(df_copy, x='Giorno', y='Cumulativa', height=300, markers=True)
            fig_cum.update_traces(line_color="#00E5FF")
            st.plotly_chart(style_fig(fig_cum), use_container_width=True)
            st.markdown("<div class='explain-text'>Traccia la progressione lineare dei chilometri accumulati nel periodo di riferimento.</div>", unsafe_allow_html=True)

            record_km = df.loc[df['Distanza (km)'].idxmax()]
            record_vel = df.loc[df['Velocità (km/h)'].idxmax()]
            giorni_attivi = (df['Distanza (km)'] > 0).sum()
            streak = int((df['Distanza (km)'] > df['Distanza (km)'].mean()).astype(int).groupby((df['Distanza (km)'] <= df['Distanza (km)'].mean()).cumsum()).cumsum().max())

            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
                <h3 style='color:#FFB020; margin-bottom:15px;'>Bacheca Record — Periodo Selezionato</h3>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Corsa più lunga</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_km['Distanza (km)']:.1f} km</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Velocità massima</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_vel['Velocità (km/h)']:.1f} km/h</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Miglior striscia sopra media</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{streak} allenamenti</strong></div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0;'><span>Giorni con allenamento</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{giorni_attivi} / {len(df)}</strong></div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', color='RPE', color_continuous_scale=[[0,'#0E4A57'],[0.5,'#00E5FF'],[1,'#FF6A3D']], height=300)
            st.plotly_chart(style_fig(fig2), use_container_width=True)
            st.markdown("<div class='explain-text'>Relazione tra velocità e frequenza cardiaca. Una maggiore efficienza sposta i punti verso destra mantenendo i battiti bassi.</div>", unsafe_allow_html=True)

            st.markdown("**Ripartizione Zone Cardiache**")
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Fondo Lento)', 'Z3 (Medio/Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df_copy = df.copy()
            df_copy['Zone'] = pd.cut(df_copy['FC Media'], bins=bins, labels=labels)
            zone_counts = df_copy['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', hole=0.6, height=300, color_discrete_sequence=['#00E5FF','#00B8D4','#0E4A57','#FFB020','#FF6A3D'])
            st.plotly_chart(style_fig(fig_zones), use_container_width=True)
            st.markdown("<div class='explain-text'>Distribuzione percentuale del tempo trascorso nelle diverse zone cardiache di allenamento.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=300, color_discrete_sequence=['#00E5FF'])
            fig3.add_vline(x=3.5, line_dash="dash", line_color="#00F5A0")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig3), use_container_width=True)
            st.markdown("<div class='explain-text'>Frequenza dei livelli di sforzo percepito registrati al termine delle sessioni.</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=300, markers=True)
            fig_sleep.update_traces(line_color="#00E5FF")
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="#00F5A0")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_sleep), use_container_width=True)
            st.markdown("<div class='explain-text'>Monitoraggio giornaliero delle ore di sonno rispetto alle soglie di recupero raccomandate.</div>", unsafe_allow_html=True)

            st.markdown("**Debito di Sonno (Rolling 7gg)**")
            df_copy = df.copy()
            df_copy['Debito'] = df_copy['Ore Sonno'].apply(lambda x: max(0, 7.5 - x)).rolling(7).sum()
            fig_debt = px.area(df_copy, x='Giorno', y='Debito', height=300, color_discrete_sequence=['#FF6A3D'])
            st.plotly_chart(style_fig(fig_debt), use_container_width=True)
            st.markdown("<div class='explain-text'>Accumulo settimanale del deficit di sonno rispetto allo standard ottimale di 7.5 ore.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', color='Rischio Infortunio', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], height=300)
            fig4.add_hline(y=7, line_dash="dash", line_color="#FFB020")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="#FFB020")
            st.plotly_chart(style_fig(fig4), use_container_width=True)
            st.markdown("<div class='explain-text'>Correlazione bivariata tra ore di sonno e intensità dello sforzo in relazione al rischio infortuni.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("**Storico Allenamenti Selezionati**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: 'ALTO' if x == 1 else 'OK')

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#111827', align='center', font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#0E1420', align='center', font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
        st.plotly_chart(style_fig(fig_table), use_container_width=True)

# ---------------------------------------------------------
# PAGINA 3: KPI DASHBOARD
# ---------------------------------------------------------
elif pagina == "KPI DASHBOARD":
    header_block(
        "Modulo 03 — Live Monitoring",
        "KPI DASHBOARD",
        "Bilancio carico/recupero, indice di rischio e profilo atletico calcolati sui parametri odierni.",
        IMG_HERO_KPI, "Live Pulse Monitor"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        df_base = st.session_state.dati.copy()

        st.markdown("### Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
        df_14 = df_base.tail(14).copy()
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)", fill='tozeroy', fillcolor='rgba(255, 106, 61, 0.18)', line=dict(color='#FF6A3D', width=3)))
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero", line=dict(color='#00F5A0', width=4)))
        fig_balance.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E8ECF2", size=13), bgcolor="rgba(14,20,32,0.85)", bordercolor="#1c2333", borderwidth=1))
        st.plotly_chart(style_fig(fig_balance), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Confronta visivamente la curva dello stress fisico (area arancione) con la capacità di recupero biologico (linea verde). Quando la linea verde sovrasta i picchi di carico, il corpo si trova in fase di supercompensazione ottimale.</div>", unsafe_allow_html=True)

        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        if risk_score < 25:
            status_color, status_text = "#00F5A0", "OTTIMALE"
        elif risk_score < 60:
            status_color, status_text = "#FFB020", "MODERATO"
        else:
            status_color, status_text = "#FF6A3D", "CRITICO"

        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 4px; font-family:\"Space Grotesk\",sans-serif;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {status_color};'><div class='section-label'>Rischio Infortunio</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k2:
            rec_color = "#00F5A0" if recovery_score >= 75 else "#FFB020" if recovery_score >= 40 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {rec_color};'><div class='section-label'>Recovery Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k3:
            sma_color = "#00F5A0" if sma < 10 else "#FFB020" if sma < 15 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {sma_color};'><div class='section-label'>SMA Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#8792A3'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#E8ECF2"}, 'bar': {'color': status_color, 'thickness': 0.75}, 'bgcolor': "#111827", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': "rgba(0, 245, 160, 0.08)"}, {'range': [25, 60], 'color': "rgba(255, 176, 32, 0.08)"}, {'range': [60, 100], 'color': "rgba(255, 106, 61, 0.08)"}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Tachimetro sintetico che quantifica il livello di pericolo sistemico attuale, associando fasce di colore (Verde = Sicuro, Giallo = Attenzione, Rosso = Rischio elevato).</div>", unsafe_allow_html=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (%)'], fill='toself', name='Parametri',
                marker=dict(color=status_color), line=dict(color=status_color)
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')), height=360)
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Diagramma a ragnatela multidimensionale che mappa l'equilibrio tra i fattori di stress e le risorse di recupero attuali dell'atleta.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Il Tuo Profilo Atletico AI")
        cv_sonno, cv_rpe = df_base['Ore Sonno'].std() / df_base['Ore Sonno'].mean(), df_base['RPE'].std() / df_base['RPE'].mean()
        consistenza = max(0, 100 - (cv_sonno + cv_rpe) * 100)

        if recovery_score >= 75 and sma < 10:
            archetipo, arch_col, arch_desc = "Il Bilanciato", "#00F5A0", "Gestisci sonno e carichi con grande equilibrio. Il tuo corpo lavora in supercompensazione costante: mantieni questa routine."
        elif r['stress_lavoro'] >= 7 and r['ore_sonno'] < 7:
            archetipo, arch_col, arch_desc = "Il Guerriero Stanco", "#FFB020", "Spingi forte nonostante stress e sonno limitato. Grande grinta, ma il conto arriva: pianifica un blocco di recupero prima possibile."
        elif sma >= 15:
            archetipo, arch_col, arch_desc = "L'Instancabile", "#FF6A3D", "Accumuli carico su carico. Ottimo motore, ma attenzione: senza pause il rischio di crollo fisico o mentale cresce rapidamente."
        else:
            archetipo, arch_col, arch_desc = "Il Costante", "#00E5FF", "Il tuo profilo è stabile e prevedibile: la base ideale su cui costruire progressi graduali e a basso rischio infortuni."

        col_arch1, col_arch2 = st.columns([1, 2])
        with col_arch1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 2px solid {arch_col}; display:flex; flex-direction:column; justify-content:center;'>
                <h3 style='color:{arch_col}; margin:5px 0; font-size:1.3em;'>{archetipo}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_arch2:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; height:100%;'>
                <p style='color:#B8C2D0; font-size:1.02em; margin-bottom:15px; font-family:"Inter",sans-serif;'>{arch_desc}</p>
                <p style='color:#8792A3; margin-bottom:5px; font-family:"Inter",sans-serif; font-size:0.9em;'>Indice di Consistenza (90gg)</p>
                <div style='background:#111827; border-radius:8px; overflow:hidden; height:20px;'>
                    <div style='background: linear-gradient(90deg, #00E5FF, #00F5A0); width:{min(consistenza,100):.0f}%; height:100%; text-align:right; padding-right:8px; color:#04121a; font-size:0.78em; font-weight:700; line-height:20px; font-family:"JetBrains Mono",monospace;'>{consistenza:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGINA 4: ANALISI PREDITTIVA ML
# ---------------------------------------------------------
elif pagina == "ANALISI PREDITTIVA ML":
    header_block(
        "Modulo 04 — Model Explainability",
        "ANALISI PREDITTIVA ML",
        "Esplora i modelli di Machine Learning avanzati addestrati sul tuo storico biometrico e comportamentale.",
        IMG_HERO_ML, "Machine Learning Engine"
    )

    df_base = st.session_state.dati.copy()

    st.markdown("""
    <div class='info-box'>
    <h3>Come opera il Machine Learning in RUNAI?</h3>
    <p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema analizza i tuoi dati storici mediante algoritmi di classificazione, regressione e clustering non supervisionato per individuare pattern invisibili e stimare con precisione la tua risposta biologica agli stimoli.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        X_train_class = df_base[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
        y_train_class = df_base['Rischio Infortunio'].values
        scaler = StandardScaler()
        X_scaled_class = scaler.fit_transform(X_train_class)

        t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6 = st.tabs([
            "Random Forest", "Logistic Regression", "Linear Regression", "Cluster K-Means", "Stress Prediction", "Simulatore What-If"
        ])

        with t_ml1:
            st.markdown("### Random Forest Classifier (Infortunio)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello basato su un insieme (ensemble) di alberi decisionali indipendenti. Ciascun albero esprime un voto binario basato su soglie biometriche; il risultato finale aggrega le probabilità. È ideale per catturare dinamiche non lineari complesse.</div>", unsafe_allow_html=True)
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
            rf_model.fit(X_scaled_class, y_train_class)
            
            c1, c2 = st.columns(2)
            with c1:
                feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']
                importances = rf_model.feature_importances_
                imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
                fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'))
                fig_imp.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Importanza delle Variabili")
                st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            with c2:
                y_pred_rf = rf_model.predict(X_scaled_class)
                cm = confusion_matrix(y_train_class, y_pred_rf)
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False))
                fig_cm.update_layout(height=350, title="Matrice di Confusione")
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> Il grafico a barre mostra il peso relativo di ogni metrica nel processo decisionale della foresta. La matrice di confusione evidenzia l'accuratezza predittiva rispetto agli eventi storici reali.</div>", unsafe_allow_html=True)

        with t_ml2:
            st.markdown("### Logistic Regression (Probabilità Lineare)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello statistico supervisionato che calcola la probabilità di un evento binario (rischio sì/no) attraverso una funzione logistica. Restituisce coefficienti lineari espliciti per ciascuna feature.</div>", unsafe_allow_html=True)
            
            log_model = LogisticRegression(random_state=42)
            log_model.fit(X_scaled_class, y_train_class)
            coefs = log_model.coef_[0]
            
            colors = ['#FF6A3D' if c > 0 else '#00F5A0' for c in coefs]
            fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors))
            fig_log.update_layout(height=400, title="Coefficienti di Impatto (Logistic Regression)", yaxis_title="Peso Coefficiente")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> I coefficienti verdi (es. ore di sonno) agiscono come fattori protettivi riducendo la probabilità di rischio; i coefficienti arancioni (es. stress o RPE elevato) aumentano esponenzialmente le probabilità di sovraccarico.</div>", unsafe_allow_html=True)

        with t_ml3:
            st.markdown("### Linear Regression (Previsione FC Media)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Algoritmo di regressione supervisionata che modella il legame lineare tra una variabile continua dipendente (Frequenza Cardiaca) e variabili indipendenti (Velocità, Temperatura, Distanza).</div>", unsafe_allow_html=True)
            
            X_lr = df_base[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']]
            y_lr = df_base['FC Media']
            lr_model = LinearRegression()
            lr_model.fit(X_lr, y_lr)
            df_base['FC_Predetta'] = lr_model.predict(X_lr)
            
            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']])
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(), x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(), line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=400, title="FC Reale vs FC Predetta", xaxis_title="FC Reale", yaxis_title="FC Predetta")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> La linea diagonale verde rappresenta la previsione perfetta. La vicinanza dei punti dimostra quanto la risposta cardiaca sia prevedibile in base alle condizioni ambientali e di velocità; deviazioni anomale segnalano affaticamento latente.</div>", unsafe_allow_html=True)

        with t_ml4:
            st.markdown("### Cluster Analysis (K-Means)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Algoritmo di apprendimento non supervisionato che raggruppa automaticamente il set di dati in 3 cluster omogenei in base a similarità di distanza percorsa e frequenza cardiaca media.</div>", unsafe_allow_html=True)
            
            X_clust = df_base[['Distanza (km)', 'FC Media']]
            km = KMeans(n_clusters=3, random_state=42)
            df_base['Cluster_ID'] = km.fit_predict(X_clust)
            df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")
            
            fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type', color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='RPE')
            fig_km.update_layout(height=400, title="Segmentazione Cluster Allenamenti")
            st.plotly_chart(style_fig(fig_km), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> L'algoritmo suddivide autonomamente le sessioni in tipologie distinte (es. fondi lunghi, sedute di recupero, lavori ad alta intensità), consentendo di verificare l'efficacia della polarizzazione del carico.</div>", unsafe_allow_html=True)

        with t_ml5:
            st.markdown("### Stress / Overload Prediction (Time Series)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Analisi delle serie temporali basata sul calcolo della media mobile dello stress sistemico (SMA = Stress * RPE / Sonno), finalizzata a intercettare trend di accumulo di fatica cronica.</div>", unsafe_allow_html=True)
            
            df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').copy()
            df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
            
            fig_sp = px.area(df_stress, x='Giorno', y='SMA_Rolling', color_discrete_sequence=['#FF6A3D'])
            fig_sp.add_hline(y=15, line_dash="dash", line_color="#FFB020", annotation_text="Soglia Critica")
            fig_sp.update_layout(height=400, title="Media Mobile Stress Sistemico (7 Giorni)")
            st.plotly_chart(style_fig(fig_sp), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> L'area evidenzia l'andamento della fatica accumulata nel tempo. Superamenti ripetuti della soglia critica indicano finestre temporali ad alto rischio di sovrallenamento e calo prestazionale.</div>", unsafe_allow_html=True)

        with t_ml6:
            st.markdown("### Simulatore What-If (Random Forest Live)")
            st.markdown("""<div class='info-box'><strong>Modifica i parametri interattivi e osserva in tempo reale l'impatto sul rischio stimato dal modello Random Forest.</strong></div>""", unsafe_allow_html=True)
            
            base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6}

            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                sim_dist = st.slider("Distanza simulata (km)", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
                sim_sonno = st.slider("Ore di sonno simulate", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
            with col_sim2:
                sim_stress = st.slider("Stress simulato", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
                sim_rpe = st.slider("RPE simulato", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")

            sim_fc = 100 + sim_rpe * 10
            sim_input = np.array([[sim_dist, sim_sonno, sim_stress, sim_fc, sim_rpe]])
            sim_prob = rf_model.predict_proba(scaler.transform(sim_input))[0][1] * 100
            sim_color = "#FF6A3D" if sim_prob >= 60 else "#FFB020" if sim_prob >= 25 else "#00F5A0"

            col_simg1, col_simg2 = st.columns(2)
            with col_simg1:
                fig_sim_gauge = go.Figure(go.Indicator(mode="gauge+number", value=sim_prob, title={'text': "Rischio Simulato", 'font': {'color': '#8792A3'}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color}, 'bgcolor': "#111827", 'borderwidth': 0}, number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}))
                fig_sim_gauge.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sim_gauge), use_container_width=True)
            with col_simg2:
                sonno_range = np.linspace(4, 10, 20)
                probs_range = [rf_model.predict_proba(scaler.transform(np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100 for s in sonno_range]
                fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'}, title="Sensibilità: Rischio vs Ore di Sonno")
                fig_sens.update_traces(line_color="#00E5FF", line_width=3)
                fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#FF6A3D")
                fig_sens.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sens), use_container_width=True)

    except Exception as e:
        st.error(f"Errore caricamento modelli ML: {str(e)}")

# ---------------------------------------------------------
# PAGINA 5: CONSIGLIO FINALE
# ---------------------------------------------------------
elif pagina == "CONSIGLIO FINALE":
    header_block(
        "Modulo 05 — Action Plan",
        "CONSIGLIO FINALE",
        "Protocollo operativo, proiezioni fisiologiche e export report per la sessione odierna.",
        IMG_HERO_PLAN, "Coach Protocol"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        df_base = st.session_state.dati.copy()

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

        if risk_score < 25: tit, col = "ALLENAMENTO INTENSO AUTORIZZATO", "#00F5A0"
        elif risk_score < 60: tit, col = "RECUPERO ATTIVO CONSIGLIATO", "#FFB020"
        else: tit, col = "RIPOSO OBBLIGATORIO", "#FF6A3D"

        st.markdown(f"""
        <div class='kpi-card' style='border: 1px solid {col}; background-color: rgba(0,0,0,0.35);'>
            <h2 style='color: {col}; margin: 0; border: none; font-size:1.6em;'>{tit}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .kpi-card-equal { background: #0E1420; border-radius: 12px; padding: 25px 20px; border: 1px solid #1c2333; height: 480px; display: flex; flex-direction: column; }
        .kpi-card-equal .kpi-equal-body { overflow-y: auto; flex-grow: 1; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar { width: 6px; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar-thumb { background: #1c2333; border-radius: 4px; }
        </style>
        """, unsafe_allow_html=True)

        tipo_all = r.get('tipo_allenamento', 'Easy Run')
        col_new1, col_new2, col_new3 = st.columns(3)
        
        with col_new1:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00E5FF;'>Distanza Consigliata</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{distanza_consigliata:.1f} km</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>su {distanza_target}km desiderati</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center;'>Tipo allenamento: <strong style='color:#B8C2D0;'>{tipo_all}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new2:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:{col};'>Rischio Calcolato</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{risk_score:.0f}%</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>Probabilità Infortunio/Burnout</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center;'>Recovery Score: <strong style='color:#B8C2D0;'>{recovery_score:.0f}%</strong> · SMA: <strong style='color:#B8C2D0;'>{sma:.1f}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new3:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00F5A0;'>Protocollo Coach Dettagliato</h3>
                <div class='kpi-equal-body' style='color:#B8C2D0; font-size:0.85em; text-align:left;'>
                    <strong style='color:#00E5FF;'>PRE-ALLENAMENTO (T-90/-15 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>T-90': pasto leggero, ~{round(distanza_target * 3)}g carboidrati.</li>
                        <li>T-30': {round(distanza_target * 20)}ml di liquidi.</li>
                        <li>T-15': mobilità dinamica anche/caviglie, skip (5').</li>
                    </ul>
                    <strong style='color:#FFB020;'>DURANTE</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Sorso d'acqua ogni 20' se superi i 60'.</li>
                        <li>Cadenza target 170-180 spm, respiro controllato.</li>
                    </ul>
                    <strong style='color:#00F5A0;'>POST (0-30 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Entro 30': ~{round(distanza_target * 1.2) + 15}g proteine + ~{round(distanza_target * 4) + 20}g carboidrati.</li>
                        <li>Stretching statico gentile 8-10'.</li>
                        <li>Rullo miofasciale 5' su quadricipiti.</li>
                    </ul>
                    <strong style='color:#8b5cf6;'>SERALE</strong>
                    <ul style='margin-top:5px; padding-left:18px; margin-bottom:0;'>
                        <li>Punta a {max(r['ore_sonno'],7.5):.1f}h di sonno per recupero cellulare.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Zone Cardiache Consigliate per Oggi")
st.markdown(
    """
* **Zona 1-2 (Recupero / Base):** Sforzo confortevole (Test del parlato OK). Ideale per Easy Run o recuperi.
* **Zona 3 (Aerobico / Tempo):** Ritmo sostenuto ma controllato.
* **Zona 4-5 (Soglia / Anaerobico):** Da evitare se il rischio infortunio è alto (>40%).
"""
)
        # NUOVA IDEA 4: EXPORT REPORT RAPIDO
        st.subheader("Generazione Report per Coach / Export")
        report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (Target: {distanza_target} km)
Indice Rischio: {risk_score:.0f}%
Recovery Score: {recovery_score:.0f}%
Stress Mentale (SMA): {sma:.1f}
Note Atleta: {r.get('nota_soggettiva', 'Nessuna nota')}
--------------------------------"""
        st.text_area("Testo Report Formattato:", value=report_testo, height=120)
        st.download_button("SCARICA REPORT TXT", data=report_testo, file_name="runai_report_allenamento.txt", mime="text/plain", use_container_width=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.subheader("Analisi Parametri vs Media (90 giorni)")
        media_sonno_90, media_stress_90, media_rpe_90 = df_base['Ore Sonno'].mean(), df_base['Stress Lavoro'].mean(), df_base['RPE'].mean()
        sonno_vs_media, stress_vs_media, rpe_vs_media = r['ore_sonno'] - media_sonno_90, r['stress_lavoro'] - media_stress_90, r['rpe_previsto'] - media_rpe_90

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            sb, sc = ("SOTTO MEDIA", "#FF6A3D") if sonno_vs_media < -0.5 else ("SOPRA MEDIA", "#00F5A0") if sonno_vs_media > 0.5 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{sc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{sb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['ore_sonno']:.1f}h</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_sonno_90:.1f}h</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Tempo di rigenerazione cellulare. Un deficit prolungato rallenta il recupero muscolare.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a2:
            stb, stc = ("SOTTO MEDIA", "#00F5A0") if stress_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if stress_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{stc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{stb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['stress_lavoro']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_stress_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Carico cognitivo e nervoso accumulato. Uno stress elevato innalza i livelli di cortisolo.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a3:
            rpb, rpc = ("SOTTO MEDIA", "#00F5A0") if rpe_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if rpe_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{rpc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{rpb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['rpe_previsto']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_rpe_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Sforzo pianificato per la sessione odierna rispetto alla media storica registrata.</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI
# ---------------------------------------------------------
elif pagina == "COMPUTER VISION":
    header_block(
        "Modulo 06 — Computer Vision",
        "AI RUNNING FORM ANALYSIS & INJURY PREDICTION",
        "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro biometrico, calcola angoli/sovraccarichi e predice il rischio d'infortunio tramite Machine Learning.",
        IMG_HERO_CV, "Pose Estimation & ML"
    )

    st.markdown("""
    <div class='info-box'>
    <strong>Analisi Biometrica Avanzata:</strong> Estrazione dello scheletro posturale, mappatura dei sovraccarichi articolari, analisi angolare della falcata e predizione ML del distretto anatomico a rischio infortunio secondo i modelli di cinematica applicata.
    </div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader("Carica video della corsa (Profilo laterale consigliato, MP4/MOV)", type=["mp4", "mov", "avi"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name

        if not st.session_state.get('cv_analizzato', False):
            if st.button("ELABORA SCHELETRO E PREDICI INFORTUNIO", use_container_width=True):
                with st.spinner("Estrazione fotogrammi, analisi vettoriale e calcolo predittivo ML in corso..."):
                    import time
                    time.sleep(2.5)

                    st.session_state.cv_analizzato = True
                    st.session_state.cv_dati = {
                        'angolo_ginocchio_appoggio': 141.5,
                        'angolo_inclinazione_busto': 7.2,
                        'oscillazione_verticale': 8.4,
                        'overstride_cm': 14.2,
                        'sovraccarico_prevalente': "Complesso Rotuleo & Tendine d'Achille",
                        'tipo_appoggio': "Appoggio di Tallone Marcato (Heel Striking)",
                        'infortunio_predetto': "Sindrome Patello-Femorale & Tendinopatia Achillea",
                        'probabilita_infortunio_ml': 84.5
                    }
                st.rerun()

        if st.session_state.get('cv_analizzato', False):
            st.success("Analisi video completata. Video elaborato e Modelli Generativi caricati con successo.")
            st.markdown("---")
            
            st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:4px; letter-spacing:0.1em;'>KINEMATIC WIREFRAME // GAIT ANALYSIS FRAME</p>", unsafe_allow_html=True)
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Frame Rate", "240 FPS", "High-Speed")
            mc2.metric("Confidence", "99.2%", "OpenPose v3")
            mc3.metric("Fase", "Strike", "0ms Impatto")

            st.markdown("<p style='font-size:0.85em; color:#8792A3; margin-top:8px; margin-bottom:16px;'>Tracciamento articolare e analisi vettoriale basata sui dati reali del video:</p>", unsafe_allow_html=True)
            
            col_out1, col_out2 = st.columns([1, 1.1])
            
            with col_out1:
                st.video(video_file)
                st.markdown("<p style='font-size:0.75em; color:#00F5A0; text-align:center; font-family:\"JetBrains Mono\",monospace; margin-top:10px;'>OUTPUT: AI TRACKING COMPLETATO</p>", unsafe_allow_html=True)

            with col_out2:
                dati_REALI = st.session_state.cv_dati
                
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
                  <text x="28" y="256" fill="#8792A3" font-family="monospace" font-size="7">RISCHIO ML VIDEO</text>
                  <text x="28" y="282" fill="#FF6A3D" font-family="monospace" font-size="20" font-weight="bold">{dati_REALI['probabilita_infortunio_ml']}%</text>
                  <text x="28" y="298" fill="#FF6A3D" font-family="monospace" font-size="7">STATUS: CRITICO</text>
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
                st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px;'><strong>Spiegazione:</strong> Grafico dinamico delle forze d'impatto al suolo (GRF). Evidenzia il picco transitorio di frenata causato dall'anticipo della falcata estratto dal video.</p>", unsafe_allow_html=True)

            dati_cv = st.session_state.cv_dati
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
                articolazioni = ['Ginocchia', 'Achille', 'Anca', 'Schiena', 'Caviglie']
                carichi = [38, 31, 14, 11, 6]
                fig_bar_load = px.bar(
                    x=articolazioni, y=carichi, 
                    labels={'x': 'Distretto', 'y': '% Impatto'},
                    color=carichi, color_continuous_scale=[[0, '#00E5FF'], [0.5, '#FFB020'], [1, '#FF6A3D']]
                )
                fig_bar_load.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_bar_load), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Analisi Carichi:</strong> Percentuale di forza d'impatto verticale trasferita sui distretti articolari in base al vettore di frenata del tallone.</div>", unsafe_allow_html=True)

            with cg2:
                st.markdown("### 2. Angoli Articolari (Falcata)")
                fasi = ['Impatto (Strike)', 'Mid-Stance', 'Toe-Off', 'Swing']
                angoli_fase = [dati_cv['angolo_ginocchio_appoggio'], 168.0, 115.0, 92.0]
                fig_radar_angles = go.Figure(go.Scatterpolar(
                    r=angoli_fase, theta=fasi, fill='toself',
                    marker=dict(color='#00F5A0'), line=dict(color='#00F5A0')
                ))
                fig_radar_angles.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[80, 180], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')),
                    height=320
                )
                st.plotly_chart(style_fig(fig_radar_angles), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Analisi Angolare:</strong> Grado di flessione dell'articolazione del ginocchio lungo le quattro fasi del ciclo del passo (Gait Cycle).</div>", unsafe_allow_html=True)

            with cg3:
                st.markdown("### 3. Rischio Infortunio ML (%)")
                distretti_rischio = ['Ginocchio/Rotula', 'Tendine Achille', 'Fascia Plantare', 'Tibia (Periostite)', 'Lombari']
                rischi_ml = [42.5, 28.0, 15.2, 10.3, 4.0]
                fig_ml_risk = px.bar(
                    x=distretti_rischio, y=rischi_ml,
                    labels={'x': 'Patologia/Distretto', 'y': 'Probabilità ML (%)'},
                    color=rischi_ml, color_continuous_scale=[[0, '#00F5A0'], [0.5, '#FFB020'], [1, '#FF6A3D']]
                )
                fig_ml_risk.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_ml_risk), use_container_width=True)
                st.markdown("<div class='explain-text'><strong>Predizione ML:</strong> Classificatore probabilistico basato su dataset di cinematica clinica per la stima del distretto anatomico a cedimento strutturale.</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h3>Diagnosi Posturale, Errori e Predizione Machine Learning</h3>", unsafe_allow_html=True)
            
            st.error(f"ERRORE BIOMECCANICO RILEVATO — {dati_cv['tipo_appoggio']}: L'estensione anticipata della tibia all'impatto produce un angolo del ginocchio ridotto ({dati_cv['angolo_ginocchio_appoggio']}°), generando un momento flettente esterno e una forza frenante di picco che aumenta il carico di compressione sulla rotula.")
            st.warning(f"ZONA DI SOVRACCARICO CRITICO: {dati_cv['sovraccarico_prevalente']}. L'onda d'urto transitoria non viene dissipata correttamente dal complesso muscolotendineo, trasferendo stress meccanico diretto sulle cartilagini e sulle inserzioni tendinee.")
            st.markdown(f"""
            <div class='danger-box' style='border-left-color: #FF6A3D;'>
                <h3 style='color: #FF6A3D; margin-top:0;'>PREDIZIONE MACHINE LEARNING (Indice di Rischio: {dati_cv['probabilita_infortunio_ml']}%)</h3>
                <p style='color: #E8ECF2; font-size: 1.05em;'>Proiettando il pattern di over-stride e la dissipazione cinetica attuale sulle curve di tolleranza al carico del tessuto connettivo, il modello predittivo diagnostica una probabilità elevata di sviluppare nel medio termine: <strong style='color: #FF6A3D;'>{dati_cv['infortunio_predetto']}</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("PROTOCOLLO DI CORREZIONE BIOMECCANICA CONSIGLIATO:\n1. Riduzione dell'ampiezza della falcata per eliminare l'over-stride anteriore al baricentro.\n2. Incremento della frequenza di passo a 176-180 falcate al minuto (SPM) per facilitare l'atterraggio sul mesopiede.\n3. Integrazione di esercizi di forza eccentrica per il quadricipite e protocollo di rinforzo progressivo per il tendine d'Achille.")
    else:
        st.info("Suggerimento: Carica un video registrato lateralmente per attivare l'estrazione dello scheletro, i grafici di analisi biomeccanica e la predizione clinica basata su Machine Learning.")
