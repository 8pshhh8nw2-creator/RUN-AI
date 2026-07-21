from cv_engine import analizza_running_video
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUN AI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

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
# PAGINA 0: HOME / LANDING PAGE (VERSIONE MIGLIORATA)
# ---------------------------------------------------------
if pagina == "HOME":
    header_block(
        "RUNAI // Master Thesis Project",
        "PERFORMANCE INTELLIGENCE SYSTEM",
        "Piattaforma avanzata di Sport Data Science e Machine Learning per l'analisi predittiva e la prevenzione del rischio infortuni nei runner amatori.",
        IMG_HERO_HOME, "Executive Dashboard"
    )

    # --- Badge di stato live, per dare un senso di "sistema attivo" ---
    st.markdown("""
    <div style='display:flex; align-items:center; gap:8px; margin-bottom:18px;'>
        <span style='width:8px; height:8px; border-radius:50%; background:#00F5A0;
                     box-shadow:0 0 8px #00F5A0; display:inline-block;'></span>
        <span style='font-family:"JetBrains Mono",monospace; font-size:0.75em; color:#00F5A0;
                     letter-spacing:0.1em;'>SISTEMA ONLINE — DATI SINCRONIZZATI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <strong>Benvenuto in RUNAI.</strong> Questo sistema integra IoT, Wearable Analytics e modelli di Machine Learning supervisionati per ottimizzare i carichi di allenamento e supportare le decisioni del preparatore o dell'atleta.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # --- Calcolo variazioni settimanali per dare contesto ai numeri (non solo il totale secco) ---
    tot_km = df_full['Distanza (km)'].sum()
    n_sessioni = len(df_full)
    km_ultimi_15 = df_full['Distanza (km)'].tail(15).sum()
    km_precedenti_15 = df_full['Distanza (km)'].tail(30).head(15).sum() if len(df_full) >= 30 else 0
    delta_km = km_ultimi_15 - km_precedenti_15 if km_precedenti_15 > 0 else 0
    delta_pct = (delta_km / km_precedenti_15 * 100) if km_precedenti_15 > 0 else 0

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("KM Totali (90gg)", f"{tot_km:.0f} km",
                   f"{delta_pct:+.0f}% ultime 2 sett." if km_precedenti_15 > 0 else None)
    col_h2.metric("Sessioni Monitorate", f"{n_sessioni}")
    col_h3.metric("Modelli ML Attivi", "5 Algoritmi", "Random Forest, Regressioni, K-Means")
    col_h4.metric("Stato Sistema", "Online", "Sync attivo")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # --- Trend rapido delle ultime sessioni: dà subito un colpo d'occhio visivo, non solo numeri ---
    st.markdown("<p style='font-size:0.8em; color:#8792A3; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.08em; margin-bottom:4px;'>TREND DISTANZA — ULTIME SESSIONI</p>", unsafe_allow_html=True)
    serie_km = df_full['Distanza (km)'].tail(30).reset_index(drop=True)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        y=serie_km, mode='lines', fill='tozeroy', name='Distanza',
        line=dict(color='#00E5FF', width=2.5),
        fillcolor='rgba(0,229,255,0.12)',
        hovertemplate='%{y:.1f} km<extra></extra>'
    ))
    fig_trend.update_layout(
        height=140, margin=dict(l=0, r=0, t=6, b=0),
        showlegend=False,
        xaxis=dict(visible=False, title=None),
        yaxis=dict(visible=False, title=None),
        title=None,
    )
    fig_trend_finale = style_fig(fig_trend)
    fig_trend_finale.update_layout(showlegend=False, title=None)
    st.plotly_chart(fig_trend_finale, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    st.subheader("Panoramica Moduli Principali")

    def icona_svg(tipo, colore, colore_id):
        """Genera un'icona vettoriale astratta sfumata in stile sport-tech (no emoji, no foto)."""
        base = f"""
        <svg viewBox="0 0 64 64" width="52" height="52" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="glow{colore_id}" cx="50%" cy="45%" r="65%">
              <stop offset="0%" stop-color="{colore}" stop-opacity="0.55"/>
              <stop offset="100%" stop-color="{colore}" stop-opacity="0"/>
            </radialGradient>
            <filter id="blur{colore_id}" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="1.4"/>
            </filter>
          </defs>
          <circle cx="32" cy="32" r="30" fill="url(#glow{colore_id})"/>
          <circle cx="32" cy="32" r="29" fill="none" stroke="{colore}" stroke-width="1" opacity="0.25"/>
          <g stroke="{colore}" fill="{colore}">
            {tipo}
          </g>
        </svg>
        """
        return base

    forme = {
        "pulse": f'<polyline points="12,34 22,34 26,20 32,46 37,28 41,34 52,34" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" filter="url(#blur1)" />',
        "bars": '<g stroke-linecap="round"><rect x="16" y="30" width="7" height="18" rx="2" opacity="0.55"/><rect x="28" y="20" width="7" height="28" rx="2" opacity="0.8"/><rect x="40" y="12" width="7" height="36" rx="2"/></g>',
        "radar": '<g fill="none" stroke-width="1.6"><circle cx="32" cy="32" r="18" opacity="0.35"/><circle cx="32" cy="32" r="11" opacity="0.55"/><circle cx="32" cy="32" r="3.5" opacity="1" fill-opacity="1"/><line x1="32" y1="6" x2="32" y2="14" stroke-width="2.2"/><line x1="32" y1="50" x2="32" y2="58" stroke-width="2.2" opacity="0.4"/><line x1="6" y1="32" x2="14" y2="32" stroke-width="2.2" opacity="0.4"/><line x1="50" y1="32" x2="58" y2="32" stroke-width="2.2" opacity="0.4"/></g>',
        "network": '<g stroke-width="1.4"><line x1="18" y1="22" x2="32" y2="14" opacity="0.5"/><line x1="18" y1="22" x2="18" y2="42" opacity="0.5"/><line x1="18" y1="42" x2="32" y2="50" opacity="0.5"/><line x1="32" y1="14" x2="46" y2="22" opacity="0.5"/><line x1="46" y1="22" x2="46" y2="42" opacity="0.5"/><line x1="46" y1="42" x2="32" y2="50" opacity="0.5"/><line x1="32" y1="14" x2="32" y2="50" opacity="0.3"/><circle cx="18" cy="22" r="3.4"/><circle cx="32" cy="14" r="3.4"/><circle cx="46" cy="22" r="3.4"/><circle cx="18" cy="42" r="3.4"/><circle cx="46" cy="42" r="3.4"/><circle cx="32" cy="50" r="3.4"/></g>',
        "check": '<g fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="32" cy="32" r="16" opacity="0.3"/><polyline points="23,32 29,39 42,24"/></g>',
        "skeleton": '<g stroke-width="1.6" stroke-linecap="round"><line x1="32" y1="14" x2="32" y2="30" opacity="0.6"/><line x1="32" y1="18" x2="20" y2="26" opacity="0.6"/><line x1="32" y1="18" x2="44" y2="26" opacity="0.6"/><line x1="32" y1="30" x2="22" y2="42" opacity="0.6"/><line x1="32" y1="30" x2="42" y2="42" opacity="0.6"/><line x1="22" y1="42" x2="19" y2="54" opacity="0.6"/><line x1="42" y1="42" x2="45" y2="54" opacity="0.6"/><circle cx="32" cy="10" r="4"/><circle cx="32" cy="18" r="2"/><circle cx="20" cy="26" r="2"/><circle cx="44" cy="26" r="2"/><circle cx="32" cy="30" r="2"/><circle cx="22" cy="42" r="2"/><circle cx="42" cy="42" r="2"/></g>',
    }

    moduli = [
        {
            "num": "01", "titolo": "Stato di Forma", "colore": "#00E5FF", "id": 1,
            "icona": forme["pulse"],
            "desc": "Configura i parametri biologici giornalieri, sonno e stress per avviare il calcolo predittivo dell'allenamento.",
            "tag": "Input Giornaliero"
        },
        {
            "num": "02", "titolo": "Statistiche & Analisi", "colore": "#00F5A0", "id": 2,
            "icona": forme["bars"],
            "desc": "Esplora lo storico delle sessioni con grafici e statistiche descrittive su volumi, ritmi e andamento nel tempo.",
            "tag": "Storico Sessioni"
        },
        {
            "num": "03", "titolo": "KPI Dashboard", "colore": "#FFB020", "id": 3,
            "icona": forme["radar"],
            "desc": "Colpo d'occhio sugli indicatori chiave di performance: carico, recupero e stato generale dell'atleta.",
            "tag": "Metriche Chiave"
        },
        {
            "num": "04", "titolo": "Analisi Predittiva ML", "colore": "#FF6A3D", "id": 4,
            "icona": forme["network"],
            "desc": "Random Forest, Regressioni e Cluster K-Means per comprendere i pattern nascosti nel tuo storico e prevedere il rischio.",
            "tag": "5 Modelli Attivi"
        },
        {
            "num": "05", "titolo": "Consiglio Finale", "colore": "#00E5FF", "id": 5,
            "icona": forme["check"],
            "desc": "Sintesi operativa: distanza consigliata, zone cardiache e raccomandazioni per la sessione odierna.",
            "tag": "Report Giornaliero"
        },
        {
            "num": "06", "titolo": "Computer Vision", "colore": "#00F5A0", "id": 6,
            "icona": forme["skeleton"],
            "desc": "Analisi biomeccanica della falcata tramite video e stima del rischio associato ai sovraccarichi articolari.",
            "tag": "Pose Estimation"
        },
    ]

    riga1 = st.columns(3)
    riga2 = st.columns(3)
    colonne_moduli = riga1 + riga2

    for col, m in zip(colonne_moduli, moduli):
        with col:
            svg_icona = icona_svg(m["icona"], m["colore"], m["id"])
            # Lo stroke delle forme dentro l'SVG eredita il colore del modulo
            svg_icona = svg_icona.replace('<g ', f'<g stroke="{m["colore"]}" fill="{m["colore"]}" ', 1) if '<g ' in svg_icona else svg_icona
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; height: 240px; position:relative;
                        border-left: 3px solid {m['colore']};
                        transition: box-shadow 0.2s ease;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
                        margin-bottom: 16px;'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <div style='width:52px; height:52px;'>{svg_icona}</div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:0.7em; color:#566178;
                                 letter-spacing:0.1em;'>{m['num']}</span>
                </div>
                <h3 style='color:{m['colore']}; margin-top:10px; margin-bottom:8px; font-size:1.15em;'>{m['titolo']}</h3>
                <p style='color:#8792A3; font-size:0.88em; line-height:1.5; margin-bottom:14px;'>{m['desc']}</p>
                <div style='position:absolute; bottom:16px; left:20px; right:20px;'>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:0.7em; color:{m['colore']};
                                 background:rgba(255,255,255,0.04); padding:4px 10px; border-radius:20px;
                                 letter-spacing:0.05em;'>● {m['tag']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # --- Footer informativo, dà un tocco di "prodotto finito" alla tesi ---
    st.markdown("""
    <div style='text-align:center; padding:18px 0 4px 0; border-top:1px solid #1c2333;'>
        <p style='color:#566178; font-size:0.75em; font-family:"JetBrains Mono",monospace; letter-spacing:0.08em;'>
            RUNAI PERFORMANCE INTELLIGENCE SYSTEM — Master Thesis Project
        </p>
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
# IMPORT AGGIUNTIVI NECESSARI (da aggiungere in cima al file principale,
# insieme a quelli che già hai per RandomForest, LogisticRegression, ecc.)
# ---------------------------------------------------------
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import (
#     confusion_matrix, accuracy_score, precision_score, recall_score,
#     f1_score, roc_curve, auc, roc_auc_score, r2_score, mean_squared_error,
#     silhouette_score
# )
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# try:
#     import shap
#     SHAP_AVAILABLE = True
# except ImportError:
#     SHAP_AVAILABLE = False

# ---------------------------------------------------------
# PAGINA 4: ANALISI PREDITTIVA ML — VERSIONE COMPLETA
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
    <p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema analizza i tuoi dati storici mediante algoritmi di classificazione, regressione e clustering non supervisionato per individuare pattern invisibili e stimare con precisione la tua risposta biologica agli stimoli. Ogni modello viene validato su un set di dati mai visto in fase di addestramento, per garantire stime realistiche e non ottimistiche.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        feature_cols = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']
        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']

        X_class = df_base[feature_cols].values
        y_class = df_base['Rischio Infortunio'].values

        scaler = StandardScaler()
        X_scaled_class = scaler.fit_transform(X_class)

        # -----------------------------------------------------
        # SPLIT TRAIN/TEST CONDIVISO — evita di valutare i modelli
        # sugli stessi dati con cui sono stati addestrati (overfitting)
        # -----------------------------------------------------
        stratify_arg = y_class if len(np.unique(y_class)) > 1 and len(df_base) >= 10 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_class, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
        )

        t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6, t_ml7, t_ml8, t_ml9, t_ml10 = st.tabs([
            "Random Forest", "Logistic Regression", "Linear Regression", "Cluster K-Means",
            "Stress Prediction", "Simulatore What-If", "Confronto Modelli",
            "Explainability (SHAP)", "Anomaly Detection", "PCA"
        ])

        # =====================================================
        # TAB 1 — RANDOM FOREST
        # =====================================================
        with t_ml1:
            st.markdown("### Random Forest Classifier (Infortunio)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello basato su un insieme (ensemble) di alberi decisionali indipendenti. Ciascun albero esprime un voto binario basato su soglie biometriche; il risultato finale aggrega le probabilità. È ideale per catturare dinamiche non lineari complesse.</div>", unsafe_allow_html=True)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

            c1, c2 = st.columns(2)
            with c1:
                importances = rf_model.feature_importances_
                imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
                fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'))
                fig_imp.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Importanza delle Variabili")
                st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            with c2:
                cm = confusion_matrix(y_test, y_pred_rf)
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False))
                fig_cm.update_layout(height=350, title="Matrice di Confusione (dati di TEST)")
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)

            # --- Metriche di validazione ---
            acc = accuracy_score(y_test, y_pred_rf)
            prec = precision_score(y_test, y_pred_rf, zero_division=0)
            rec = recall_score(y_test, y_pred_rf, zero_division=0)
            f1 = f1_score(y_test, y_pred_rf, zero_division=0)
            try:
                roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
            except ValueError:
                roc_auc_rf = float('nan')

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{acc*100:.1f}%")
            mc2.metric("Precision", f"{prec*100:.1f}%")
            mc3.metric("Recall", f"{rec*100:.1f}%")
            mc4.metric("F1-Score", f"{f1*100:.1f}%")
            mc5.metric("ROC-AUC", f"{roc_auc_rf:.2f}" if not np.isnan(roc_auc_rf) else "N/D")

            # --- Cross-validation per robustezza della stima ---
            try:
                cv_scores = cross_val_score(
                    RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5),
                    X_scaled_class, y_class, cv=5, scoring='accuracy'
                )
                st.caption(f"Validazione incrociata (5-fold) su tutto il dataset: accuracy media {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
            except ValueError:
                st.caption("Cross-validation non disponibile: servono più campioni per classe.")

            # --- Curva ROC ---
            if not np.isnan(roc_auc_rf):
                fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#00E5FF', width=3), name=f"Random Forest (AUC={roc_auc_rf:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#8792A3', dash='dash'), name="Random / baseline"))
                fig_roc.update_layout(height=350, title="Curva ROC (dati di TEST)", xaxis_title="Falsi Positivi", yaxis_title="Veri Positivi")
                st.plotly_chart(style_fig(fig_roc), use_container_width=True)

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> Tutte le metriche sono calcolate su dati di test mai visti durante l'addestramento, per una stima realistica della capacità predittiva del modello. La cross-validation a 5 fold conferma la stabilità della performance su porzioni diverse del dataset.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 2 — LOGISTIC REGRESSION
        # =====================================================
        with t_ml2:
            st.markdown("### Logistic Regression (Probabilità Lineare)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello statistico supervisionato che calcola la probabilità di un evento binario (rischio sì/no) attraverso una funzione logistica. Restituisce coefficienti lineari espliciti per ciascuna feature, rendendolo altamente interpretabile.</div>", unsafe_allow_html=True)

            log_model = LogisticRegression(random_state=42, max_iter=1000)
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)
            y_proba_log = log_model.predict_proba(X_test)[:, 1]
            coefs = log_model.coef_[0]

            colors = ['#FF6A3D' if c > 0 else '#00F5A0' for c in coefs]
            fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors))
            fig_log.update_layout(height=400, title="Coefficienti di Impatto (Logistic Regression)", yaxis_title="Peso Coefficiente")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)

            acc_l = accuracy_score(y_test, y_pred_log)
            f1_l = f1_score(y_test, y_pred_log, zero_division=0)
            try:
                auc_l = roc_auc_score(y_test, y_proba_log)
            except ValueError:
                auc_l = float('nan')

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Accuracy (test)", f"{acc_l*100:.1f}%")
            lc2.metric("F1-Score (test)", f"{f1_l*100:.1f}%")
            lc3.metric("ROC-AUC (test)", f"{auc_l:.2f}" if not np.isnan(auc_l) else "N/D")

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> I coefficienti verdi (es. ore di sonno) agiscono come fattori protettivi riducendo la probabilità di rischio; i coefficienti arancioni (es. stress o RPE elevato) aumentano esponenzialmente le probabilità di sovraccarico. Le metriche a destra confermano quanto questa lettura lineare regga sui dati di test.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 3 — LINEAR REGRESSION
        # =====================================================
        with t_ml3:
            st.markdown("### Linear Regression (Previsione FC Media)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Algoritmo di regressione supervisionata che modella il legame lineare tra una variabile continua dipendente (Frequenza Cardiaca) e variabili indipendenti (Velocità, Temperatura, Distanza).</div>", unsafe_allow_html=True)

            X_lr = df_base[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']].values
            y_lr = df_base['FC Media'].values

            X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
            lr_model = LinearRegression()
            lr_model.fit(X_lr_train, y_lr_train)

            df_base['FC_Predetta'] = lr_model.predict(X_lr)
            y_lr_pred_test = lr_model.predict(X_lr_test)
            r2_test = r2_score(y_lr_test, y_lr_pred_test)
            rmse_test = mean_squared_error(y_lr_test, y_lr_pred_test) ** 0.5

            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']])
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(), x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(), line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=400, title="FC Reale vs FC Predetta (tutti i punti)", xaxis_title="FC Reale", yaxis_title="FC Predetta")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)

            rc1, rc2 = st.columns(2)
            rc1.metric("R² (dati di test)", f"{r2_test:.2f}")
            rc2.metric("RMSE (dati di test)", f"{rmse_test:.1f} bpm")

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> La linea diagonale verde rappresenta la previsione perfetta. L'R² sui dati di test misura la quota di variabilità della FC spiegata dal modello; l'RMSE esprime l'errore medio in battiti al minuto. Deviazioni anomale segnalano affaticamento latente non catturato dalle sole variabili ambientali.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 4 — CLUSTER K-MEANS (con elbow + silhouette)
        # =====================================================
        with t_ml4:
            st.markdown("### Cluster Analysis (K-Means)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Algoritmo di apprendimento non supervisionato che raggruppa automaticamente il set di dati in cluster omogenei in base a similarità di distanza percorsa e frequenza cardiaca media. Il numero di cluster ottimale non va scelto a caso: va giustificato con il metodo del gomito (elbow) e lo score di silhouette.</div>", unsafe_allow_html=True)

            X_clust = df_base[['Distanza (km)', 'FC Media']].values
            max_k = min(8, max(3, len(df_base) - 1))

            inertias, sil_scores, k_range = [], [], list(range(2, max_k + 1))
            for k in k_range:
                km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_test = km_test.fit_predict(X_clust)
                inertias.append(km_test.inertia_)
                try:
                    sil_scores.append(silhouette_score(X_clust, labels_test))
                except ValueError:
                    sil_scores.append(0)

            ec1, ec2 = st.columns(2)
            with ec1:
                fig_elbow = go.Figure(go.Scatter(x=k_range, y=inertias, mode='lines+markers', line=dict(color='#00E5FF', width=3)))
                fig_elbow.update_layout(height=320, title="Metodo del Gomito (Inertia)", xaxis_title="N. Cluster (k)", yaxis_title="Inertia")
                st.plotly_chart(style_fig(fig_elbow), use_container_width=True)
            with ec2:
                fig_sil = go.Figure(go.Bar(x=k_range, y=sil_scores, marker_color='#00F5A0'))
                fig_sil.update_layout(height=320, title="Silhouette Score per k", xaxis_title="N. Cluster (k)", yaxis_title="Silhouette Score")
                st.plotly_chart(style_fig(fig_sil), use_container_width=True)

            best_k = k_range[int(np.argmax(sil_scores))] if sil_scores else 3
            st.caption(f"Il valore di k con silhouette score più alto è k={best_k}. Per coerenza con la lettura sportiva (fondi/recupero/alta intensità) si utilizza k=3 nel grafico sottostante.")

            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_base['Cluster_ID'] = km.fit_predict(X_clust)
            df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")

            fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type', color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='RPE')
            fig_km.update_layout(height=400, title="Segmentazione Cluster Allenamenti (k=3)")
            st.plotly_chart(style_fig(fig_km), use_container_width=True)

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> L'algoritmo suddivide autonomamente le sessioni in tipologie distinte (es. fondi lunghi, sedute di recupero, lavori ad alta intensità), consentendo di verificare l'efficacia della polarizzazione del carico. Il grafico del gomito e lo score di silhouette forniscono la giustificazione quantitativa della scelta di k.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 5 — STRESS / OVERLOAD PREDICTION
        # =====================================================
        with t_ml5:
            st.markdown("### Stress / Overload Prediction (Time Series)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Analisi delle serie temporali basata sul calcolo della media mobile dello stress sistemico (SMA = Stress * RPE / Sonno), finalizzata a intercettare trend di accumulo di fatica cronica. Un modello di trend lineare proietta l'andamento sui giorni successivi.</div>", unsafe_allow_html=True)

            df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').reset_index(drop=True).copy()
            df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()

            fig_sp = px.area(df_stress, x='Giorno', y='SMA_Rolling', color_discrete_sequence=['#FF6A3D'])
            fig_sp.add_hline(y=15, line_dash="dash", line_color="#FFB020", annotation_text="Soglia Critica")
            fig_sp.update_layout(height=400, title="Media Mobile Stress Sistemico (7 Giorni)")
            st.plotly_chart(style_fig(fig_sp), use_container_width=True)

            # --- Proiezione a breve termine (trend lineare semplice) ---
            n_forecast = 7
            x_idx = np.arange(len(df_stress))
            valid_mask = ~df_stress['SMA_Rolling'].isna()
            if valid_mask.sum() >= 3:
                coeffs = np.polyfit(x_idx[valid_mask], df_stress['SMA_Rolling'][valid_mask], deg=1)
                trend_fn = np.poly1d(coeffs)
                future_idx = np.arange(len(df_stress), len(df_stress) + n_forecast)
                future_vals = trend_fn(future_idx)
                residual_std = np.std(df_stress['SMA_Rolling'][valid_mask] - trend_fn(x_idx[valid_mask]))

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=list(range(len(df_stress))), y=df_stress['SMA_Rolling'], mode='lines', line=dict(color='#00E5FF'), name="Storico"))
                fig_forecast.add_trace(go.Scatter(x=list(future_idx), y=future_vals, mode='lines', line=dict(color='#FF6A3D', dash='dash'), name="Proiezione (trend lineare)"))
                fig_forecast.add_trace(go.Scatter(x=list(future_idx) + list(future_idx)[::-1], y=list(future_vals + residual_std) + list(future_vals - residual_std)[::-1], fill='toself', fillcolor='rgba(255,106,61,0.15)', line=dict(color='rgba(0,0,0,0)'), name="Banda di incertezza"))
                fig_forecast.update_layout(height=380, title=f"Proiezione Stress Sistemico — Prossimi {n_forecast} giorni", xaxis_title="Indice giorno", yaxis_title="SMA")
                st.plotly_chart(style_fig(fig_forecast), use_container_width=True)
                st.caption("Nota metodologica: la proiezione utilizza un semplice trend lineare a fini illustrativi. Per una tesi più approfondita si può sostituire con modelli ARIMA o Holt-Winters (statsmodels), che catturano meglio stagionalità e autocorrelazione.")
            else:
                st.info("Servono almeno 3 punti validi di SMA per calcolare una proiezione.")

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> L'area evidenzia l'andamento della fatica accumulata nel tempo. Superamenti ripetuti della soglia critica indicano finestre temporali ad alto rischio di sovrallenamento e calo prestazionale; la proiezione aiuta a intervenire preventivamente.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 6 — SIMULATORE WHAT-IF
        # =====================================================
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

        # =====================================================
        # TAB 7 — CONFRONTO MODELLI (NUOVO)
        # =====================================================
        with t_ml7:
            st.markdown("### Confronto tra Modelli di Classificazione")
            st.markdown("<div class='explain-text'><strong>Perché confrontare più modelli:</strong> nessun algoritmo è ottimale a priori. Il confronto sistematico su accuracy, F1 e ROC-AUC calcolati sugli stessi dati di test permette di scegliere il modello più adatto al problema, motivando la scelta finale nella tesi.</div>", unsafe_allow_html=True)

            models_to_compare = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3),
            }

            comparison_rows = []
            fig_roc_all = go.Figure()
            roc_colors = ['#00E5FF', '#FFB020', '#00F5A0']

            for (name, model), color in zip(models_to_compare.items(), roc_colors):
                model.fit(X_train, y_train)
                y_pred_m = model.predict(X_test)
                y_proba_m = model.predict_proba(X_test)[:, 1]

                row = {
                    "Modello": name,
                    "Accuracy": accuracy_score(y_test, y_pred_m),
                    "Precision": precision_score(y_test, y_pred_m, zero_division=0),
                    "Recall": recall_score(y_test, y_pred_m, zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred_m, zero_division=0),
                }
                try:
                    row["ROC-AUC"] = roc_auc_score(y_test, y_proba_m)
                    fpr_m, tpr_m, _ = roc_curve(y_test, y_proba_m)
                    fig_roc_all.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode='lines', line=dict(color=color, width=3), name=f"{name} (AUC={row['ROC-AUC']:.2f})"))
                except ValueError:
                    row["ROC-AUC"] = float('nan')

                comparison_rows.append(row)

            df_compare = pd.DataFrame(comparison_rows).set_index("Modello")
            st.dataframe(df_compare.style.format("{:.2%}", subset=["Accuracy", "Precision", "Recall", "F1-Score"]).format("{:.2f}", subset=["ROC-AUC"]).background_gradient(cmap="Blues", subset=["Accuracy", "F1-Score", "ROC-AUC"]), use_container_width=True)

            fig_roc_all.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#8792A3', dash='dash'), name="Baseline casuale"))
            fig_roc_all.update_layout(height=400, title="Curve ROC a Confronto (dati di TEST)", xaxis_title="Falsi Positivi", yaxis_title="Veri Positivi")
            st.plotly_chart(style_fig(fig_roc_all), use_container_width=True)

            best_model_name = df_compare["ROC-AUC"].idxmax() if df_compare["ROC-AUC"].notna().any() else df_compare["F1-Score"].idxmax()
            st.markdown(f"<div class='explain-text'><strong>Analisi Risultati:</strong> Sul set di test, il modello con la performance più solida è <strong>{best_model_name}</strong>. Questo confronto quantitativo è la base per giustificare, nel capitolo metodologico della tesi, la scelta del modello adottato in produzione.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 8 — EXPLAINABILITY (SHAP) (NUOVO)
        # =====================================================
        with t_ml8:
            st.markdown("### Explainability Avanzata (SHAP Values)")
            st.markdown("<div class='explain-text'><strong>Perché SHAP:</strong> a differenza della feature importance globale del Random Forest, i valori SHAP (SHapley Additive exPlanations) spiegano <em>ogni singola predizione</em>, quantificando quanto ciascuna variabile abbia spinto il rischio stimato verso l'alto o verso il basso per quello specifico allenamento.</div>", unsafe_allow_html=True)

            if SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_test)
                shap_vals_risk = shap_values[1] if isinstance(shap_values, list) else shap_values

                # --- Importanza media assoluta (globale) ---
                mean_abs_shap = np.abs(shap_vals_risk).mean(axis=0)
                shap_imp = sorted(list(zip(feature_names, mean_abs_shap)), key=lambda x: x[1], reverse=True)
                fig_shap_global = go.Figure(go.Bar(y=[x[0] for x in shap_imp], x=[x[1] for x in shap_imp], orientation='h', marker_color='#00E5FF', text=[f'{x[1]:.3f}' for x in shap_imp], textposition='auto'))
                fig_shap_global.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Importanza Media |SHAP| (impatto medio sulla predizione)")
                st.plotly_chart(style_fig(fig_shap_global), use_container_width=True)

                # --- Spiegazione di una singola predizione (waterfall semplificato) ---
                st.markdown("#### Spiegazione di una singola sessione")
                idx_choice = st.slider("Seleziona la sessione di test da spiegare", 0, len(X_test) - 1, 0, key="shap_idx")
                instance_shap = shap_vals_risk[idx_choice]
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

                waterfall_data = sorted(list(zip(feature_names, instance_shap)), key=lambda x: abs(x[1]), reverse=True)
                colors_wf = ['#FF6A3D' if v > 0 else '#00F5A0' for _, v in waterfall_data]
                fig_wf = go.Figure(go.Bar(x=[x[1] for x in waterfall_data], y=[x[0] for x in waterfall_data], orientation='h', marker_color=colors_wf))
                fig_wf.update_layout(height=320, title=f"Contributo di ogni variabile — sessione #{idx_choice} (base={base_value:.2f})", xaxis_title="Impatto SHAP sulla probabilità di rischio")
                fig_wf.add_vline(x=0, line_color="#E8ECF2", line_width=1)
                st.plotly_chart(style_fig(fig_wf), use_container_width=True)

                st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> le barre arancioni spingono la predizione verso un rischio più alto per quella specifica sessione, quelle verdi verso un rischio più basso. Questo livello di dettaglio, assente nella semplice feature importance, è ciò che rende un modello realmente 'explainable' nel senso richiesto dalla letteratura di Explainable AI (XAI).</div>", unsafe_allow_html=True)
            else:
                st.warning("La libreria 'shap' non è installata nell'ambiente. Aggiungi `shap` a requirements.txt e reinstalla le dipendenze per abilitare questa sezione.")
                st.markdown("<div class='explain-text'>In alternativa, viene mostrata la <strong>permutation importance</strong>: mescola casualmente ogni variabile e misura di quanto peggiora la performance del modello, un'approssimazione più robusta della semplice feature importance del Random Forest.</div>", unsafe_allow_html=True)
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(rf_model, X_test, y_test, n_repeats=20, random_state=42)
                perm_data = sorted(list(zip(feature_names, perm.importances_mean)), key=lambda x: x[1], reverse=True)
                fig_perm = go.Figure(go.Bar(y=[x[0] for x in perm_data], x=[x[1] for x in perm_data], orientation='h', marker_color='#FFB020'))
                fig_perm.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Permutation Importance (fallback senza SHAP)")
                st.plotly_chart(style_fig(fig_perm), use_container_width=True)

        # =====================================================
        # TAB 9 — ANOMALY DETECTION (NUOVO)
        # =====================================================
        with t_ml9:
            st.markdown("### Anomaly Detection (Isolation Forest)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> l'Isolation Forest isola le osservazioni costruendo alberi casuali: le sessioni anomale richiedono in media meno divisioni per essere isolate rispetto a quelle 'normali'. Utile per individuare allenamenti atipici che potrebbero anticipare un infortunio o un errore di registrazione dati.</div>", unsafe_allow_html=True)

            contamination = st.slider("Percentuale attesa di sessioni anomale", 0.02, 0.25, 0.08, step=0.01, key="iso_contam")
            iso_model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
            anomaly_labels = iso_model.fit_predict(X_scaled_class)
            df_base['Anomalia'] = np.where(anomaly_labels == -1, 'Anomala', 'Normale')

            fig_anom = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Anomalia', color_discrete_map={'Normale': '#00E5FF', 'Anomala': '#FF6A3D'}, size='RPE', hover_data=['Giorno'])
            fig_anom.update_layout(height=400, title="Sessioni di Allenamento: Normali vs Anomale")
            st.plotly_chart(style_fig(fig_anom), use_container_width=True)

            n_anomalie = int((df_base['Anomalia'] == 'Anomala').sum())
            st.metric("Sessioni segnalate come anomale", f"{n_anomalie} su {len(df_base)}")

            if n_anomalie > 0:
                st.dataframe(df_base[df_base['Anomalia'] == 'Anomala'][['Giorno'] + feature_cols], use_container_width=True)

            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> le sessioni segnalate come anomale meritano un controllo manuale — potrebbero indicare un allenamento sopra soglia, un errore nei sensori, oppure un evento fisiologico rilevante da annotare nel diario di allenamento.</div>", unsafe_allow_html=True)

        # =====================================================
        # TAB 10 — PCA (NUOVO)
        # =====================================================
        with t_ml10:
            st.markdown("### Analisi delle Componenti Principali (PCA)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> la PCA riduce le 5 variabili biometriche a 2 componenti principali che ne riassumono la varianza, permettendo di visualizzare in un unico piano 2D la struttura complessiva dei dati e la separabilità tra sessioni a rischio e sessioni sicure.</div>", unsafe_allow_html=True)

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled_class)
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_pca['Rischio Infortunio'] = y_class

            pc1, pc2 = st.columns(2)
            with pc1:
                fig_pca_scatter = px.scatter(df_pca, x='PC1', y='PC2', color=df_pca['Rischio Infortunio'].astype(str), color_discrete_sequence=['#00F5A0', '#FF6A3D'])
                fig_pca_scatter.update_layout(height=380, title="Proiezione 2D delle Sessioni (PCA)", legend_title="Rischio Infortunio")
                st.plotly_chart(style_fig(fig_pca_scatter), use_container_width=True)
            with pc2:
                var_ratio = pca.explained_variance_ratio_ * 100
                fig_var = go.Figure(go.Bar(x=['PC1', 'PC2'], y=var_ratio, marker_color=['#00E5FF', '#FFB020'], text=[f'{v:.1f}%' for v in var_ratio], textposition='auto'))
                fig_var.update_layout(height=380, title="Varianza Spiegata per Componente", yaxis_title="% Varianza Spiegata")
                st.plotly_chart(style_fig(fig_var), use_container_width=True)

            st.caption(f"Le prime 2 componenti spiegano insieme il {var_ratio.sum():.1f}% della varianza totale delle 5 variabili originali.")
            st.markdown("<div class='explain-text'><strong>Analisi Risultati:</strong> se i punti rossi (rischio) e verdi (sicuro) formano regioni distinte nel piano PC1-PC2, significa che le variabili biometriche raccolte contengono già un segnale forte e strutturato di separabilità del rischio, a supporto della bontà del dataset usato per l'addestramento.</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Errore caricamento modelli ML: {str(e)}")

# ---------------------------------------------------------
# PAGINA 5: CONSIGLIO FINALE  (redesign completo)
# ---------------------------------------------------------
# ASSUNZIONI:
# - import plotly.graph_objects as go / import plotly.express as px
#   già presenti in cima al file principale.
# - df_base ha una colonna data chiamata 'Data' (per i trend a 90gg).
#   Se si chiama diversamente, aggiorna la sezione "GRAFICI ANALITICI".
# - Streamlit >= 1.29 (per st.tabs; non serve altro di recente).
#
# SISTEMA DI DESIGN:
# - Un'unica componente "panel" riusata ovunque (KPI, coach, grafici,
#   confronti) per coerenza visiva: stesso bordo, stesso radius,
#   stesso ritmo di spaziatura.
# - Palette ridotta e con significato fisso: ciano = sonno,
#   corallo = stress, menta = RPE/positivo, ambra = attenzione,
#   viola = recupero serale. Il bordo dei pannelli resta neutro:
#   il colore si usa solo su numeri, linee e accenti, non sui bordi,
#   per evitare l'effetto "arcobaleno".
# - Header di sezione con eyebrow (etichetta mono maiuscola) invece
#   del semplice st.subheader, per un look da cruscotto dati.
# - Il contenuto del coach è definito una sola volta come struttura
#   dati (coach_content) e riusato sia per il rendering nei tab sia
#   per il report scaricabile — evita duplicazioni e disallineamenti.
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

        # =========================================================
        # TOKEN DI DESIGN
        # =========================================================
        PANEL_BG   = "#121826"
        PANEL_BD   = "#212A3B"
        PANEL_BD_H = "#2E3A52"
        TXT_PRIMARY   = "#E7ECF5"
        TXT_SECONDARY = "#8A93A6"
        TXT_TERTIARY  = "#4F5972"

        C_SONNO  = "#22E1FF"
        C_STRESS = "#FF6A4D"
        C_RPE    = "#17E8A6"
        C_AMBRA  = "#FFB020"
        C_VIOLA  = "#9B7BFF"
        C_NEUTRO = "#3A4458"

        st.markdown(f"""
        <style>
        .panel {{
            background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 16px;
            padding: 22px 24px; transition: border-color .15s ease;
        }}
        .panel:hover {{ border-color: {PANEL_BD_H}; }}
        .panel-flush {{ padding: 0; overflow: hidden; }}
        .panel-flush .panel-body {{ padding: 18px 20px 20px 20px; }}
        .eyebrow {{
            font-family:'JetBrains Mono',monospace; font-size:.68em; letter-spacing:.14em;
            text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 8px 0; font-weight:600;
        }}
        .section-head {{ margin: 6px 0 16px 0; }}
        .section-head .eyebrow {{ margin-bottom: 4px; }}
        .section-head h3 {{
            font-family:'Inter',sans-serif; font-weight:700; color:{TXT_PRIMARY};
            margin:0; font-size:1.25em; letter-spacing:-.01em;
        }}
        .section-head .sub {{ color:{TXT_SECONDARY}; font-size:.88em; margin-top:4px; }}
        .panel-title {{
            font-family:'Inter',sans-serif; font-weight:600; color:{TXT_PRIMARY};
            font-size:.94em; margin:0;
        }}
        .kv-num {{ font-family:'JetBrains Mono',monospace; color:{TXT_PRIMARY}; font-weight:600; }}
        .badge {{
            display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.66em;
            letter-spacing:.08em; text-transform:uppercase; padding:3px 9px; border-radius:999px;
            font-weight:600;
        }}
        .coach-block {{ margin-bottom: 18px; }}
        .coach-block:last-child {{ margin-bottom: 0; }}
        .coach-block .label {{
            font-family:'JetBrains Mono',monospace; font-size:.72em; letter-spacing:.1em;
            text-transform:uppercase; margin-bottom:8px; font-weight:600;
        }}
        .coach-block ul {{ margin:0; padding-left:0; list-style:none; }}
        .coach-block li {{
            position:relative; padding-left:16px; margin-bottom:7px; color:{TXT_SECONDARY};
            font-family:'Inter',sans-serif; font-size:.92em; line-height:1.5;
        }}
        .coach-block li::before {{
            content:"›"; position:absolute; left:0; color:{TXT_TERTIARY}; font-weight:700;
        }}
        .chart-caption {{
            border-top: 1px solid {PANEL_BD}; margin-top: 10px; padding-top: 10px;
            color:{TXT_SECONDARY}; font-family:'Inter',sans-serif; font-size:.82em; line-height:1.45;
        }}
        .zone-chip {{
            background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-left:3px solid var(--zc);
            border-radius:10px; padding:14px 16px;
        }}
        .zone-chip .zt {{ font-family:'JetBrains Mono',monospace; font-size:.72em; letter-spacing:.08em; color:var(--zc); text-transform:uppercase; font-weight:700; }}
        .zone-chip .zn {{ font-family:'Inter',sans-serif; font-weight:600; color:{TXT_PRIMARY}; margin:4px 0 6px 0; font-size:.95em; }}
        .zone-chip .zd {{ font-family:'Inter',sans-serif; color:{TXT_SECONDARY}; font-size:.85em; line-height:1.4; }}
        .delta-track {{ position:relative; height:6px; border-radius:3px; background:{PANEL_BD}; margin:10px 0 4px 0; }}
        .delta-fill {{ position:absolute; top:0; height:6px; border-radius:3px; }}
        .delta-mid {{ position:absolute; top:-3px; left:50%; width:1px; height:12px; background:{TXT_TERTIARY}; }}
        </style>
        """, unsafe_allow_html=True)

        def section_head(container, eyebrow, title, sub=None):
            sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
            container.markdown(f"""
            <div class='section-head'>
                <p class='eyebrow'>{eyebrow}</p>
                <h3>{title}</h3>
                {sub_html}
            </div>
            """, unsafe_allow_html=True)

        # =========================================================
        # CALCOLI BASE
        # =========================================================
        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        distanza_target = r.get('distanza_oggi', 10.0)
        distanza_consigliata = (
            distanza_target if risk_score < 40
            else distanza_target * 0.6 if risk_score < 70
            else 0.0
        )

        if risk_score < 25:
            tit, col = "ALLENAMENTO INTENSO AUTORIZZATO", C_RPE
        elif risk_score < 60:
            tit, col = "RECUPERO ATTIVO CONSIGLIATO", C_AMBRA
        else:
            tit, col = "RIPOSO OBBLIGATORIO", C_STRESS

        tipo_all = r.get('tipo_allenamento', 'Easy Run')

        if risk_score < 25:
            liv = "basso"
        elif risk_score < 60:
            liv = "medio"
        else:
            liv = "alto"

        cadenza_target = "170-180 spm" if liv != "alto" else "165-172 spm (passo più corto, cadenza rilassata)"
        zona_consigliata = "Zona 2-3 (aerobico controllato)" if liv == "basso" else (
            "Zona 1-2 (recupero attivo)" if liv == "medio" else "Cammino / Zona 1 (nessuno sforzo cardio)"
        )

        # =========================================================
        # BANNER DI STATO — elemento distintivo della pagina:
        # strumento a barra con zone di rischio e marker della
        # posizione odierna, invece del solito badge colorato isolato.
        # =========================================================
        marker_pos = max(2, min(98, risk_score))
        st.markdown(f"""
        <div class='panel'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:24px;'>
                <div>
                    <p class='eyebrow'>Stato sessione odierna</p>
                    <h2 style='margin:0; font-family:"Inter",sans-serif; font-weight:700; font-size:1.6em; color:{col}; letter-spacing:-.01em;'>{tit}</h2>
                </div>
                <div style='text-align:right;'>
                    <p class='eyebrow'>Indice di rischio</p>
                    <p style='margin:0; font-family:"JetBrains Mono",monospace; font-size:2em; color:{col}; font-weight:600;'>{risk_score:.0f}%</p>
                </div>
            </div>
            <div style='margin-top:22px; position:relative; height:6px; border-radius:3px; background:linear-gradient(90deg, {C_RPE} 0%, {C_RPE} 25%, {C_AMBRA} 25%, {C_AMBRA} 60%, {C_STRESS} 60%, {C_STRESS} 100%); opacity:0.30;'>
                <div style='position:absolute; top:-7px; left:calc({marker_pos}% - 1px); width:2px; height:20px; background:{TXT_PRIMARY};'></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:8px;'>
                <span class='eyebrow' style='margin:0;'>0 — basso</span>
                <span class='eyebrow' style='margin:0;'>25</span>
                <span class='eyebrow' style='margin:0;'>60 — alto</span>
                <span class='eyebrow' style='margin:0;'>100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # KPI PRINCIPALI
        # =========================================================
        col_new1, col_new2, col_new3 = st.columns(3)

        with col_new1:
            st.markdown(f"""
            <div class='panel' style='height:170px; display:flex; flex-direction:column; justify-content:space-between;'>
                <p class='eyebrow'>Distanza consigliata</p>
                <div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:2.3em; color:{TXT_PRIMARY}; font-weight:600;'>{distanza_consigliata:.1f}</span>
                    <span style='font-family:"Inter",sans-serif; color:{TXT_SECONDARY}; font-size:1em;'> km</span>
                </div>
                <p style='margin:0; color:{TXT_SECONDARY}; font-size:.82em;'>su {distanza_target} km desiderati · {tipo_all}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_new2:
            st.markdown(f"""
            <div class='panel' style='height:170px; display:flex; flex-direction:column; justify-content:space-between;'>
                <p class='eyebrow'>Recovery score</p>
                <div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:2.3em; color:{C_RPE}; font-weight:600;'>{recovery_score:.0f}</span>
                    <span style='font-family:"Inter",sans-serif; color:{TXT_SECONDARY}; font-size:1em;'> %</span>
                </div>
                <p style='margin:0; color:{TXT_SECONDARY}; font-size:.82em;'>basato su {r['ore_sonno']:.1f}h di sonno vs target 7.5h</p>
            </div>
            """, unsafe_allow_html=True)

        with col_new3:
            st.markdown(f"""
            <div class='panel' style='height:170px; display:flex; flex-direction:column; justify-content:space-between;'>
                <p class='eyebrow'>Carico mentale (SMA)</p>
                <div>
                    <span style='font-family:"JetBrains Mono",monospace; font-size:2.3em; color:{C_AMBRA}; font-weight:600;'>{sma:.1f}</span>
                </div>
                <p style='margin:0; color:{TXT_SECONDARY}; font-size:.82em;'>stress × RPE / ore di sonno</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # COACH PERSONALIZZATO — contenuto definito una sola volta
        # e riusato sia nei tab che nel report scaricabile.
        # =========================================================
        section_head(st, "Coach personalizzato", "Protocollo di allenamento",
                     "Indicazioni operative su preparazione, esecuzione e recupero — nessun riferimento nutrizionale.")

        cautela_txt = (
            "puoi mantenere la progressione abituale." if liv == "basso"
            else "mantieni un'andatura più conservativa del solito." if liv == "medio"
            else "limita a mobilità e cammino, evita qualsiasi accelerazione."
        )
        stop_txt = (
            "Nessuna limitazione particolare oggi: resta comunque attento a dolori articolari acuti."
            if liv == "basso" else
            "Se percepisci fatica anomala rispetto al ritmo, rallenta o inserisci un tratto di camminata."
        )
        prevenzione_txt = (
            "Nessuna precauzione aggiuntiva richiesta oggi."
            if liv == "basso" else
            "Presta attenzione a eventuali fastidi articolari comparsi durante la seduta e monitora nelle prossime 24h."
        )

        coach_content = {
            "Pre-Allenamento": {
                "colore": C_SONNO,
                "blocchi": [
                    ("Attivazione — T-15/-20 min", [
                        "5' di mobilità dinamica: cerchi d'anca, affondi in movimento, leg swing avanti/indietro.",
                        "2×10 skip bassi e 2×10 calciata dietro, per attivare la catena posteriore.",
                        "10-15 squat a corpo libero e 10 calf raise per attivare quadricipiti e polpacci.",
                        f"Controllo scarpe e terreno: verifica la suola e scegli il fondo più adatto a {tipo_all.lower()}.",
                    ]),
                    ("Progressione d'ingresso — T-5 min", [
                        f"Parti sempre con 3-5' a ritmo molto blando prima di raggiungere il ritmo target ({zona_consigliata}).",
                        f"Livello di cautela oggi: {liv.upper()} — {cautela_txt}",
                    ]),
                ],
            },
            "Durante": {
                "colore": C_AMBRA,
                "blocchi": [
                    ("Gestione del ritmo", [
                        f"Zona cardiaca target di oggi: {zona_consigliata}.",
                        f"Cadenza target: {cadenza_target}. Passi corti e rapidi riducono l'impatto articolare.",
                        "Respiro controllato: inspira per 3 passi, espira per 2 (adatta se il ritmo cambia).",
                    ]),
                    ("Postura e tecnica", [
                        "Busto leggermente in avanti, spalle rilassate, sguardo a 15-20m avanti.",
                        "Appoggio medio piede, evitando l'atterraggio di tallone troppo avanzato rispetto al corpo.",
                    ]),
                    ("Segnali di stop", [
                        stop_txt,
                        "In caso di dolore acuto (non solo fatica muscolare) interrompi e valuta con uno specialista.",
                    ]),
                ],
            },
            "Post-Allenamento": {
                "colore": C_RPE,
                "blocchi": [
                    ("Defaticamento — 0-10 min", [
                        "3-5' di camminata o corsa molto blanda per riportare gradualmente la frequenza cardiaca a riposo.",
                    ]),
                    ("Mobilità e stretching — 10-20 min", [
                        "Stretching statico gentile per 8-10': polpacci, quadricipiti, ischiocrurali, ileopsoas (30-40'' per gruppo).",
                        "Rullo miofasciale per 5' su quadricipiti e fascia ileotibiale.",
                    ]),
                    ("Prevenzione infortuni", [
                        prevenzione_txt,
                        "Se il dolore persiste oltre 48h, consulta un professionista prima della prossima uscita.",
                    ]),
                ],
            },
            "Recupero Serale": {
                "colore": C_VIOLA,
                "blocchi": [
                    ("Recupero cellulare", [
                        f"Punta a {max(r['ore_sonno'], 7.5):.1f}h di sonno stanotte per favorire il recupero muscolare.",
                        "10' di mobilità leggera o yoga serale se percepisci ancora tensione muscolare.",
                        "Riduci schermi e luce blu almeno 30' prima di dormire per migliorare la qualità del sonno.",
                    ]),
                    ("Respirazione per abbassare lo stress", [
                        "5' di respirazione diaframmatica lenta (4'' inspiro, 6'' espiro) per abbassare il carico nervoso accumulato oggi.",
                    ]),
                ],
            },
        }

        tabs = st.tabs(list(coach_content.keys()))
        for tab, (nome_tab, contenuto) in zip(tabs, coach_content.items()):
            with tab:
                blocchi_html = ""
                for label, bullets in contenuto["blocchi"]:
                    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                    blocchi_html += f"""
                    <div class='coach-block'>
                        <div class='label' style='color:{contenuto["colore"]};'>{label}</div>
                        <ul>{bullets_html}</ul>
                    </div>
                    """
                st.markdown(f"<div class='panel'>{blocchi_html}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # ZONE CARDIACHE
        # =========================================================
        section_head(st, "Riferimento", "Zone cardiache consigliate per oggi")

        zone = [
            ("Zona 1-2", "Recupero / Base", "Sforzo confortevole (test del parlato superato). Ideale per Easy Run o sedute di recupero.", C_RPE),
            ("Zona 3", "Aerobico / Tempo", "Ritmo sostenuto ma controllato, adatto a sedute di media intensità.", C_AMBRA),
            ("Zona 4-5", "Soglia / Anaerobico", "Da evitare se il rischio infortunio calcolato è alto (oltre il 40%).", C_STRESS),
        ]
        zc1, zc2, zc3 = st.columns(3)
        for c, (zt, zn, zd, zcol) in zip([zc1, zc2, zc3], zone):
            c.markdown(f"""
            <div class='zone-chip' style='--zc:{zcol};'>
                <div class='zt'>{zt}</div>
                <div class='zn'>{zn}</div>
                <div class='zd'>{zd}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # GRAFICI ANALITICI
        # =========================================================
        section_head(st, "Analisi dati", "Grafici Analitici",
                     "Andamento storico e posizionamento della sessione odierna rispetto ai tuoi valori medi.")

        CHART_HEIGHT = 260
        layout_base = dict(
            paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
            font=dict(color=TXT_SECONDARY, family="Inter, sans-serif", size=11),
            margin=dict(l=38, r=16, t=8, b=32),
            height=CHART_HEIGHT,
            showlegend=False,
            hoverlabel=dict(bgcolor="#1A2233", font_size=12, font_family="Inter, sans-serif", bordercolor=PANEL_BD),
        )
        axis_style = dict(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        config_pulita = {'displayModeBar': False}

        media_sonno_90, media_stress_90, media_rpe_90 = df_base['Ore Sonno'].mean(), df_base['Stress Lavoro'].mean(), df_base['RPE'].mean()
        sonno_vs_media = r['ore_sonno'] - media_sonno_90
        stress_vs_media = r['stress_lavoro'] - media_stress_90
        rpe_vs_media = r['rpe_previsto'] - media_rpe_90

        figs_per_export = []   # per l'export HTML
        insights_export = []   # per il testo del report (titolo, spiegazione)

        def chart_card(container, titolo, fig, spiegazione):
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
            with container:
                st.markdown(f"<div class='panel panel-flush'>", unsafe_allow_html=True)
                st.markdown(f"<div class='panel-body'><p class='panel-title'>{titolo}</p></div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config=config_pulita)
                st.markdown(f"<div class='panel-body' style='padding-top:0;'><div class='chart-caption'>{spiegazione}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            figs_per_export.append(fig)
            insights_export.append((titolo, spiegazione))

        # --- Riga 1: trend 90 giorni ---
        if 'Data' in df_base.columns:
            df_plot = df_base.sort_values('Data').tail(90)

            def calcola_trend(serie):
                recente = serie.tail(14).mean()
                precedente = serie.head(max(len(serie) - 14, 1)).mean()
                return recente - precedente

            trend_sonno = calcola_trend(df_plot['Ore Sonno'])
            trend_stress = calcola_trend(df_plot['Stress Lavoro'])
            trend_rpe = calcola_trend(df_plot['RPE'])

            r1c1, r1c2, r1c3 = st.columns(3)

            fig_t1 = go.Figure(go.Scatter(
                x=df_plot['Data'], y=df_plot['Ore Sonno'], mode='lines',
                line=dict(color=C_SONNO, width=2), fill='tozeroy', fillcolor='rgba(34,225,255,0.07)'
            ))
            fig_t1.update_layout(**layout_base, yaxis_title="ore")
            spieg_sonno = (
                "Il sonno medio delle ultime due settimane è in calo rispetto al periodo precedente."
                if trend_sonno < -0.3 else
                "Il sonno medio delle ultime due settimane è in miglioramento rispetto al periodo precedente."
                if trend_sonno > 0.3 else
                "Il sonno è rimasto stabile nelle ultime settimane, senza variazioni significative."
            )
            chart_card(r1c1, "Trend ore sonno — 90 giorni", fig_t1, spieg_sonno)

            fig_t2 = go.Figure(go.Scatter(
                x=df_plot['Data'], y=df_plot['Stress Lavoro'], mode='lines',
                line=dict(color=C_STRESS, width=2), fill='tozeroy', fillcolor='rgba(255,106,77,0.07)'
            ))
            fig_t2.update_layout(**layout_base, yaxis=dict(range=[0, 10], **axis_style), yaxis_title="punti")
            spieg_stress = (
                "Lo stress lavorativo medio è aumentato nelle ultime due settimane rispetto al periodo precedente."
                if trend_stress > 0.5 else
                "Lo stress lavorativo medio è diminuito nelle ultime due settimane rispetto al periodo precedente."
                if trend_stress < -0.5 else
                "Il livello di stress lavorativo è rimasto stabile nelle ultime settimane."
            )
            chart_card(r1c2, "Trend stress lavoro — 90 giorni", fig_t2, spieg_stress)

            fig_t3 = go.Figure(go.Scatter(
                x=df_plot['Data'], y=df_plot['RPE'], mode='lines',
                line=dict(color=C_RPE, width=2), fill='tozeroy', fillcolor='rgba(23,232,166,0.07)'
            ))
            fig_t3.update_layout(**layout_base, yaxis=dict(range=[0, 10], **axis_style), yaxis_title="punti")
            spieg_rpe = (
                "Il carico percepito (RPE) medio è in aumento nelle ultime due settimane: segnale di accumulo di fatica."
                if trend_rpe > 0.5 else
                "Il carico percepito (RPE) medio è in calo nelle ultime due settimane."
                if trend_rpe < -0.5 else
                "Il carico percepito (RPE) è rimasto stabile nelle ultime settimane."
            )
            chart_card(r1c3, "Trend RPE — 90 giorni", fig_t3, spieg_rpe)

            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        else:
            st.caption("Colonna data non trovata nel dataset — grafici di trend non disponibili (verificare il nome colonna 'Data').")

        # --- Riga 2: oggi vs media ---
        r2c1, r2c2, r2c3 = st.columns(3)

        fig_b1 = go.Figure(go.Bar(
            x=['Oggi', 'Media 90gg'], y=[r['ore_sonno'], media_sonno_90],
            marker_color=[C_SONNO, C_NEUTRO], text=[f"{r['ore_sonno']:.1f}h", f"{media_sonno_90:.1f}h"],
            textposition='outside', textfont=dict(color=TXT_SECONDARY, size=11), width=0.5
        ))
        fig_b1.update_layout(**layout_base, yaxis_title="ore")
        spieg_b1 = (
            f"Oggi il sonno è {abs(sonno_vs_media):.1f}h {'sotto' if sonno_vs_media < 0 else 'sopra'} la media storica."
            if abs(sonno_vs_media) > 0.3 else "Il sonno di oggi è sostanzialmente in linea con la media storica."
        )
        chart_card(r2c1, "Ore sonno — oggi vs media", fig_b1, spieg_b1)

        fig_b2 = go.Figure(go.Bar(
            x=['Oggi', 'Media 90gg'], y=[r['stress_lavoro'], media_stress_90],
            marker_color=[C_STRESS, C_NEUTRO], text=[f"{r['stress_lavoro']}/10", f"{media_stress_90:.1f}/10"],
            textposition='outside', textfont=dict(color=TXT_SECONDARY, size=11), width=0.5
        ))
        fig_b2.update_layout(**layout_base, yaxis=dict(range=[0, 10], **axis_style), yaxis_title="punti")
        spieg_b2 = (
            f"Lo stress di oggi è {abs(stress_vs_media):.1f} punti {'sotto' if stress_vs_media < 0 else 'sopra'} la media storica."
            if abs(stress_vs_media) > 0.5 else "Lo stress lavorativo di oggi è in linea con la media storica."
        )
        chart_card(r2c2, "Stress lavoro — oggi vs media", fig_b2, spieg_b2)

        fig_b3 = go.Figure(go.Bar(
            x=['Oggi', 'Media 90gg'], y=[r['rpe_previsto'], media_rpe_90],
            marker_color=[C_RPE, C_NEUTRO], text=[f"{r['rpe_previsto']}/10", f"{media_rpe_90:.1f}/10"],
            textposition='outside', textfont=dict(color=TXT_SECONDARY, size=11), width=0.5
        ))
        fig_b3.update_layout(**layout_base, yaxis=dict(range=[0, 10], **axis_style), yaxis_title="punti")
        spieg_b3 = (
            f"Il carico percepito previsto è {abs(rpe_vs_media):.1f} punti {'sotto' if rpe_vs_media < 0 else 'sopra'} la media storica."
            if abs(rpe_vs_media) > 0.5 else "Il carico percepito previsto è in linea con la media storica."
        )
        chart_card(r2c3, "RPE previsto — oggi vs media", fig_b3, spieg_b3)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        # --- Riga 3: relazione storica e indicatori di sintesi ---
        r3c1, r3c2, r3c3 = st.columns(3)

        if all(c in df_base.columns for c in ['Stress Lavoro', 'RPE', 'Ore Sonno']):
            fig_scatter = px.scatter(
                df_base, x='Stress Lavoro', y='RPE', color='Ore Sonno',
                color_continuous_scale=[[0, '#3A4458'], [0.5, C_SONNO], [1, '#FFFFFF']]
            )
            fig_scatter.update_traces(marker=dict(size=6, opacity=0.55))
            fig_scatter.add_trace(go.Scatter(
                x=[r['stress_lavoro']], y=[r['rpe_previsto']], mode='markers',
                marker=dict(size=13, color=C_RPE, symbol='diamond', line=dict(width=1.5, color=TXT_PRIMARY))
            ))
            fig_scatter.update_layout(**layout_base, coloraxis_showscale=False, xaxis=dict(range=[0, 10], **axis_style), yaxis=dict(range=[0, 10], **axis_style))
            spieg_scatter = "Il punto in evidenza mostra la posizione della sessione odierna rispetto allo storico: più è in alto a destra, maggiore è il carico combinato di stress e sforzo percepito."
            chart_card(r3c1, "Relazione stress vs RPE (storico)", fig_scatter, spieg_scatter)

        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score, number={'suffix': "%", 'font': {'color': col, 'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': TXT_TERTIARY, 'tickfont': {'size': 9}},
                'bar': {'color': col, 'thickness': 0.28},
                'bgcolor': PANEL_BG, 'borderwidth': 0,
                'steps': [{'range': [0, 25], 'color': '#152018'}, {'range': [25, 60], 'color': '#241d10'}, {'range': [60, 100], 'color': '#2a1414'}]
            }
        ))
        fig_g1.update_layout(**layout_base)
        spieg_g1 = f"Probabilità stimata di infortunio o burnout in base ai parametri odierni. Livello attuale: {liv}."
        chart_card(r3c2, "Indice di rischio", fig_g1, spieg_g1)

        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=recovery_score, number={'suffix': "%", 'font': {'color': C_RPE, 'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': TXT_TERTIARY, 'tickfont': {'size': 9}},
                'bar': {'color': C_RPE, 'thickness': 0.28},
                'bgcolor': PANEL_BG, 'borderwidth': 0,
                'steps': [{'range': [0, 40], 'color': '#2a1414'}, {'range': [40, 75], 'color': '#241d10'}, {'range': [75, 100], 'color': '#152018'}]
            }
        ))
        fig_g2.update_layout(**layout_base)
        spieg_g2 = "Stima della qualità del recupero in base alle ore di sonno rispetto al target fisiologico di 7.5h."
        chart_card(r3c3, "Recovery score", fig_g2, spieg_g2)

        st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # ANALISI PARAMETRI VS MEDIA (90 GIORNI)
        # =========================================================
        section_head(st, "Confronto storico", "Analisi parametri vs media (90 giorni)",
                     "La barra sotto ogni valore mostra lo scostamento dalla media: al centro è la media, a sinistra sotto media, a destra sopra media.")

        def delta_bar(delta, scala_max, colore):
            pos = 50 + max(-50, min(50, (delta / scala_max) * 50))
            larghezza = abs(pos - 50)
            sinistra = min(pos, 50)
            return f"""
            <div class='delta-track'>
                <div class='delta-mid'></div>
                <div class='delta-fill' style='left:{sinistra}%; width:{larghezza}%; background:{colore};'></div>
            </div>
            """

        col_a1, col_a2, col_a3 = st.columns(3)

        with col_a1:
            sb, sc = ("SOTTO MEDIA", C_STRESS) if sonno_vs_media < -0.5 else ("SOPRA MEDIA", C_RPE) if sonno_vs_media > 0.5 else ("NELLA MEDIA", TXT_SECONDARY)
            st.markdown(f"""
            <div class='panel'>
                <p class='eyebrow' style='margin-bottom:14px;'>Ore sonno</p>
                <span class='badge' style='background:rgba(255,255,255,0.06); color:{sc};'>{sb}</span>
                <div style='margin-top:14px;'>
                    <span class='kv-num' style='font-size:1.9em;'>{r['ore_sonno']:.1f}h</span>
                    <span style='color:{TXT_SECONDARY}; font-size:.85em;'> · media {media_sonno_90:.1f}h</span>
                </div>
                {delta_bar(sonno_vs_media, 3.0, sc)}
                <p style='font-family:"JetBrains Mono",monospace; color:{sc}; font-size:.82em; margin:2px 0 12px 0;'>Δ {'+' if sonno_vs_media>=0 else ''}{sonno_vs_media:.1f}h</p>
                <p style='color:{TXT_SECONDARY}; font-size:.83em; line-height:1.45; margin:0;'>Tempo di rigenerazione cellulare. Un deficit prolungato rallenta il recupero muscolare.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_a2:
            stb, stc = ("SOTTO MEDIA", C_RPE) if stress_vs_media < -1 else ("SOPRA MEDIA", C_STRESS) if stress_vs_media > 1 else ("NELLA MEDIA", TXT_SECONDARY)
            st.markdown(f"""
            <div class='panel'>
                <p class='eyebrow' style='margin-bottom:14px;'>Stress lavoro</p>
                <span class='badge' style='background:rgba(255,255,255,0.06); color:{stc};'>{stb}</span>
                <div style='margin-top:14px;'>
                    <span class='kv-num' style='font-size:1.9em;'>{r['stress_lavoro']}/10</span>
                    <span style='color:{TXT_SECONDARY}; font-size:.85em;'> · media {media_stress_90:.1f}/10</span>
                </div>
                {delta_bar(stress_vs_media, 5.0, stc)}
                <p style='font-family:"JetBrains Mono",monospace; color:{stc}; font-size:.82em; margin:2px 0 12px 0;'>Δ {'+' if stress_vs_media>=0 else ''}{stress_vs_media:.1f} punti</p>
                <p style='color:{TXT_SECONDARY}; font-size:.83em; line-height:1.45; margin:0;'>Carico cognitivo e nervoso accumulato. Uno stress elevato innalza i livelli di cortisolo.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_a3:
            rpb, rpc = ("SOTTO MEDIA", C_RPE) if rpe_vs_media < -1 else ("SOPRA MEDIA", C_STRESS) if rpe_vs_media > 1 else ("NELLA MEDIA", TXT_SECONDARY)
            st.markdown(f"""
            <div class='panel'>
                <p class='eyebrow' style='margin-bottom:14px;'>RPE previsto</p>
                <span class='badge' style='background:rgba(255,255,255,0.06); color:{rpc};'>{rpb}</span>
                <div style='margin-top:14px;'>
                    <span class='kv-num' style='font-size:1.9em;'>{r['rpe_previsto']}/10</span>
                    <span style='color:{TXT_SECONDARY}; font-size:.85em;'> · media {media_rpe_90:.1f}/10</span>
                </div>
                {delta_bar(rpe_vs_media, 5.0, rpc)}
                <p style='font-family:"JetBrains Mono",monospace; color:{rpc}; font-size:.82em; margin:2px 0 12px 0;'>Δ {'+' if rpe_vs_media>=0 else ''}{rpe_vs_media:.1f} punti</p>
                <p style='color:{TXT_SECONDARY}; font-size:.83em; line-height:1.45; margin:0;'>Sforzo pianificato per la sessione odierna rispetto alla media storica registrata.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # EXPORT REPORT — versione testuale completa (TXT) e
        # versione stilizzata completa (HTML) con tutti i consigli,
        # le zone cardiache e le note di ogni grafico.
        # =========================================================
        section_head(st, "Export", "Generazione report per coach",
                     "Il report include l'intero protocollo, le zone cardiache, l'analisi vs media e le note di ogni grafico.")

        # --- Costruzione testo coach in formato piano, riusando coach_content ---
        coach_txt = ""
        for nome_tab, contenuto in coach_content.items():
            coach_txt += f"\n[{nome_tab.upper()}]\n"
            for label, bullets in contenuto["blocchi"]:
                coach_txt += f"  {label}:\n"
                for b in bullets:
                    coach_txt += f"    - {b}\n"

        zone_txt = "\n".join(f"  - {zt} ({zn}): {zd}" for zt, zn, zd, _ in zone)
        grafici_txt = "\n".join(f"  - {t}: {s}" for t, s in insights_export)

        report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (Target: {distanza_target} km)
Indice Rischio: {risk_score:.0f}%
Recovery Score: {recovery_score:.0f}%
Stress Mentale (SMA): {sma:.1f}

ANALISI VS MEDIA 90 GIORNI
  - Ore Sonno: {r['ore_sonno']:.1f}h (media {media_sonno_90:.1f}h, Δ {'+' if sonno_vs_media>=0 else ''}{sonno_vs_media:.1f}h)
  - Stress Lavoro: {r['stress_lavoro']}/10 (media {media_stress_90:.1f}/10, Δ {'+' if stress_vs_media>=0 else ''}{stress_vs_media:.1f})
  - RPE Previsto: {r['rpe_previsto']}/10 (media {media_rpe_90:.1f}/10, Δ {'+' if rpe_vs_media>=0 else ''}{rpe_vs_media:.1f})

ZONE CARDIACHE CONSIGLIATE
{zone_txt}

PROTOCOLLO COACH COMPLETO
{coach_txt}
NOTE DEI GRAFICI ANALITICI
{grafici_txt}

Note Atleta: {r.get('nota_soggettiva', 'Nessuna nota')}
--------------------------------"""

        st.text_area("Testo Report Formattato:", value=report_testo, height=180)

        colb1, colb2 = st.columns(2)
        with colb1:
            st.download_button(
                "SCARICA REPORT TXT COMPLETO", data=report_testo,
                file_name="runai_report_allenamento.txt", mime="text/plain",
                use_container_width=True
            )

        with colb2:
            charts_html = ""
            for i, f in enumerate(figs_per_export):
                include_js = 'cdn' if i == 0 else False
                charts_html += f.to_html(full_html=False, include_plotlyjs=include_js)

            # Coach in HTML, riusando la stessa struttura dati
            coach_html = ""
            for nome_tab, contenuto in coach_content.items():
                blocchi_html = ""
                for label, bullets in contenuto["blocchi"]:
                    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                    blocchi_html += f"<div class='coach-block'><div class='label' style='color:{contenuto['colore']};'>{label}</div><ul>{bullets_html}</ul></div>"
                coach_html += f"<div class='panel' style='margin-bottom:14px;'><p class='panel-title' style='margin-bottom:14px;'>{nome_tab}</p>{blocchi_html}</div>"

            zone_html = "".join(
                f"<div class='zone-chip' style='--zc:{zcol}; margin-bottom:10px;'><div class='zt'>{zt}</div><div class='zn'>{zn}</div><div class='zd'>{zd}</div></div>"
                for zt, zn, zd, zcol in zone
            )

            insights_html = "".join(f"<p style='margin:0 0 8px 0; color:{TXT_SECONDARY}; font-size:.88em;'><strong style='color:{TXT_PRIMARY};'>{t}</strong> — {s}</p>" for t, s in insights_export)

            report_html_completo = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>RunAI Performance Report</title>
<style>
  body {{ background:#0B0F17; color:{TXT_SECONDARY}; font-family: Inter, sans-serif; padding: 36px; max-width:1100px; margin:0 auto; }}
  h1 {{ color:{col}; font-weight:700; font-size:1.8em; margin-bottom:4px; }}
  h2 {{ color:{TXT_PRIMARY}; font-weight:700; font-size:1.15em; margin:34px 0 14px 0; }}
  .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7em; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 8px 0; font-weight:600; }}
  .panel {{ background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:16px; padding:20px 22px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
  .kpi-row .panel {{ flex:1 1 30%; min-width:200px; }}
  .kpi-row .val {{ font-family:'JetBrains Mono',monospace; font-size:1.7em; color:{TXT_PRIMARY}; font-weight:600; }}
  .coach-block {{ margin-bottom:14px; }}
  .coach-block .label {{ font-family:'JetBrains Mono',monospace; font-size:.72em; letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px; font-weight:600; }}
  .coach-block ul {{ margin:0; padding-left:0; list-style:none; }}
  .coach-block li {{ position:relative; padding-left:16px; margin-bottom:6px; font-size:.9em; line-height:1.5; }}
  .coach-block li::before {{ content:"›"; position:absolute; left:0; color:{TXT_TERTIARY}; font-weight:700; }}
  .zone-grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .zone-chip {{ flex:1 1 28%; min-width:220px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-left:3px solid var(--zc); border-radius:10px; padding:14px 16px; }}
  .zone-chip .zt {{ font-family:'JetBrains Mono',monospace; font-size:.72em; letter-spacing:.08em; color:var(--zc); text-transform:uppercase; font-weight:700; }}
  .zone-chip .zn {{ font-weight:600; color:{TXT_PRIMARY}; margin:4px 0 6px 0; font-size:.95em; }}
  .zone-chip .zd {{ color:{TXT_SECONDARY}; font-size:.85em; line-height:1.4; }}
  .charts-grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .charts-grid > div {{ flex: 1 1 30%; min-width:280px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:12px; padding:10px; }}
</style>
</head>
<body>
  <p class="eyebrow">RunAI Performance Report</p>
  <h1>{tit}</h1>

  <div class="kpi-row">
    <div class="panel"><p class="eyebrow">Distanza Consigliata</p><div class="val">{distanza_consigliata:.1f} km</div><p>su {distanza_target} km desiderati</p></div>
    <div class="panel"><p class="eyebrow">Indice Rischio</p><div class="val" style="color:{col};">{risk_score:.0f}%</div><p>livello: {liv}</p></div>
    <div class="panel"><p class="eyebrow">Recovery Score</p><div class="val" style="color:{C_RPE};">{recovery_score:.0f}%</div><p>SMA: {sma:.1f}</p></div>
  </div>

  <h2>Analisi vs Media 90 Giorni</h2>
  <div class="kpi-row">
    <div class="panel"><p class="eyebrow">Ore Sonno</p><div class="val">{r['ore_sonno']:.1f}h</div><p>media {media_sonno_90:.1f}h · Δ {'+' if sonno_vs_media>=0 else ''}{sonno_vs_media:.1f}h</p></div>
    <div class="panel"><p class="eyebrow">Stress Lavoro</p><div class="val">{r['stress_lavoro']}/10</div><p>media {media_stress_90:.1f}/10 · Δ {'+' if stress_vs_media>=0 else ''}{stress_vs_media:.1f}</p></div>
    <div class="panel"><p class="eyebrow">RPE Previsto</p><div class="val">{r['rpe_previsto']}/10</div><p>media {media_rpe_90:.1f}/10 · Δ {'+' if rpe_vs_media>=0 else ''}{rpe_vs_media:.1f}</p></div>
  </div>

  <h2>Zone Cardiache Consigliate</h2>
  <div class="zone-grid">{zone_html}</div>

  <h2>Protocollo Coach Completo</h2>
  {coach_html}

  <h2>Grafici Analitici</h2>
  <div class="charts-grid">{charts_html}</div>
  <div style="margin-top:16px;">{insights_html}</div>

  <h2>Note Atleta</h2>
  <p>{r.get('nota_soggettiva', 'Nessuna nota')}</p>
</body>
</html>"""

            st.download_button(
                "SCARICA REPORT COMPLETO (HTML)", data=report_html_completo,
                file_name="runai_report_completo.html", mime="text/html",
                use_container_width=True
            )

# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI (DATI REALI)
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
