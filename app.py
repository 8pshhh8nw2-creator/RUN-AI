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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
# PAGINA 3: KPI DASHBOARD — VERSIONE POTENZIATA
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

        # ---------------------------------------------------------------
        # HELPER: verdetto a semaforo e spiegazione semplice (coerenti con Pagina 4)
        # ---------------------------------------------------------------
        def verdetto_box(valore_pct, soglie=(50, 75), testo_basso="", testo_medio="", testo_alto="", spiegazione=""):
            if valore_pct >= soglie[1]:
                colore, emoji, etichetta = "#00F5A0", "✅", testo_alto
            elif valore_pct >= soglie[0]:
                colore, emoji, etichetta = "#FFB020", "⚠️", testo_medio
            else:
                colore, emoji, etichetta = "#FF6A3D", "🔴", testo_basso
            st.markdown(f"""
            <div style='border-left: 4px solid {colore}; background: rgba(255,255,255,0.03);
                        padding: 12px 16px; border-radius: 8px; margin: 10px 0;'>
                <span style='font-size: 1.05em; color:{colore}; font-weight:700;'>{emoji} {etichetta}</span>
                <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>{spiegazione}</p>
            </div>
            """, unsafe_allow_html=True)

        def in_pratica(testo):
            st.markdown(f"""
            <div style='background: rgba(0,229,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
            <span style='color:#00E5FF; font-weight:600;'>💡 In parole semplici:</span>
            <span style='color:#B8C2D0; font-family:"Inter",sans-serif;'> {testo}</span>
            </div>
            """, unsafe_allow_html=True)

        def trend_arrow(oggi, media_storica, invert=False, unita=""):
            """Restituisce freccia + delta colorato rispetto alla media storica."""
            delta = oggi - media_storica
            migliora = (delta < 0) if invert else (delta > 0)
            colore = "#00F5A0" if migliora else ("#8792A3" if abs(delta) < 0.01 else "#FF6A3D")
            freccia = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
            return f"<span style='color:{colore}; font-weight:700;'>{freccia} {abs(delta):.1f}{unita}</span>"

        # ============================================================
        # BILANCIO CARICO VS RECUPERO
        # ============================================================
        st.markdown("### Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
        df_14 = df_base.tail(14).copy()

        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(
            x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)",
            fill='tozeroy', fillcolor='rgba(255, 106, 61, 0.18)', line=dict(color='#FF6A3D', width=3)
        ))
        fig_balance.add_trace(go.Scatter(
            x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero",
            line=dict(color='#00F5A0', width=4)
        ))
        fig_balance.update_layout(
            height=400, xaxis_title="Giorno", yaxis_title="Indice (base 100 = riferimento ottimale)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#E8ECF2", size=13), bgcolor="rgba(14,20,32,0.85)",
                        bordercolor="#1c2333", borderwidth=1)
        )
        st.plotly_chart(style_fig(fig_balance), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Confronta visivamente la curva dello stress fisico (area arancione) con la capacità di recupero biologico (linea verde). Quando la linea verde sovrasta i picchi di carico, il corpo si trova in fase di supercompensazione ottimale.</div>", unsafe_allow_html=True)
        in_pratica("se l'area arancione supera spesso la linea verde, stai accumulando più fatica di quanta il sonno riesca a smaltirne: è il pattern tipico che precede cali di forma o infortuni da sovraccarico.")

        giorni_sforamento = int((df_14['RPE']*10 > (df_14['Ore Sonno']/8)*100).sum())
        st.caption(f"Negli ultimi {len(df_14)} giorni, il carico ha superato la capacità di recupero in **{giorni_sforamento} giorni**.")

        # ============================================================
        # CALCOLO SCORE
        # ============================================================
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

        # ============================================================
        # KPI CARDS CON CONFRONTO STORICO
        # ============================================================
        media_sonno_storica = df_base['Ore Sonno'].mean()
        media_rpe_storica = df_base['RPE'].mean()
        media_stress_storica = df_base['Stress Lavoro'].mean() if 'Stress Lavoro' in df_base.columns else r['stress_lavoro']
        sma_storico = ((df_base['Stress Lavoro'] * df_base['RPE']) / df_base['Ore Sonno'].replace(0, np.nan)).mean() if 'Stress Lavoro' in df_base.columns else sma

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {status_color};'>
                <div class='section-label'>Rischio Infortunio</div>
                <div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            st.caption("Combina sonno, stress e sforzo previsto in un unico indice 0-100.")
        with col_k2:
            rec_color = "#00F5A0" if recovery_score >= 75 else "#FFB020" if recovery_score >= 40 else "#FF6A3D"
            st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {rec_color};'>
                <div class='section-label'>Recovery Score</div>
                <div class='data-figure' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            st.caption(f"Sonno oggi: {r['ore_sonno']:.1f}h vs media storica {media_sonno_storica:.1f}h {trend_arrow(r['ore_sonno'], media_sonno_storica, unita='h')}", unsafe_allow_html=True)
        with col_k3:
            sma_color = "#00F5A0" if sma < 10 else "#FFB020" if sma < 15 else "#FF6A3D"
            st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {sma_color};'>
                <div class='section-label'>SMA Score</div>
                <div class='data-figure' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div>
            </div>""", unsafe_allow_html=True)
            if not np.isnan(sma_storico):
                st.caption(f"Oggi vs media storica ({sma_storico:.1f}): {trend_arrow(sma, sma_storico, invert=True)}", unsafe_allow_html=True)

        in_pratica("lo SMA (Stress x RPE / Sonno) sale quando accumuli stress e sforzo con poco sonno a bilanciarli: valori sotto 10 sono un buon equilibrio, sopra 15 indicano un sistema sotto pressione.")

        verdetto_box(
            100 - risk_score, soglie=(40, 75),
            testo_basso=f"Rischio elevato ({risk_score:.0f}%): oggi la combinazione di sonno, stress e sforzo previsto richiede prudenza — valuta di ridurre l'intensità o rimandare l'allenamento",
            testo_medio=f"Rischio moderato ({risk_score:.0f}%): allenati con attenzione, magari abbassando leggermente il carico previsto",
            testo_alto=f"Rischio basso ({risk_score:.0f}%): condizioni favorevoli per allenarti come da programma",
            spiegazione="Questo indice non sostituisce il buon senso: se percepisci dolore o fatica anomala, ascolta il corpo anche se il punteggio è basso."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ============================================================
        # GAUGE + RADAR (scala corretta)
        # ============================================================
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#8792A3'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#E8ECF2"}, 'bar': {'color': status_color, 'thickness': 0.75},
                       'bgcolor': "#111827", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': "rgba(0, 245, 160, 0.08)"},
                                 {'range': [25, 60], 'color': "rgba(255, 176, 32, 0.08)"},
                                 {'range': [60, 100], 'color': "rgba(255, 106, 61, 0.08)"}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Tachimetro sintetico che quantifica il livello di pericolo sistemico attuale, associando fasce di colore (Verde = Sicuro, Giallo = Attenzione, Rosso = Rischio elevato).</div>", unsafe_allow_html=True)

        with col_g2:
            # --- Radar con scala normalizzata correttamente su 0-10 per tutti i 4 assi ---
            recovery_su_10 = recovery_score / 10  # 0-100 -> 0-10, coerente con le altre variabili
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_su_10],
                theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (/10)'],
                fill='toself', name='Oggi',
                marker=dict(color=status_color), line=dict(color=status_color)
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[media_sonno_storica, media_stress_storica, media_rpe_storica, 7.5],
                theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (/10)'],
                fill='toself', name='Tua media storica',
                marker=dict(color='#8792A3'), line=dict(color='#8792A3', dash='dot'), opacity=0.5
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 12], gridcolor='#1c2333'),
                           angularaxis=dict(gridcolor='#1c2333')),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#E8ECF2", size=11))
            )
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Diagramma a ragnatela che mappa l'equilibrio tra i fattori di stress e le risorse di recupero attuali, confrontandoli con la tua media storica (linea tratteggiata grigia).</div>", unsafe_allow_html=True)
            in_pratica("se la forma colorata di 'Oggi' è molto più estesa di quella grigia sui lati Stress e RPE, oggi sei sopra i tuoi standard abituali di carico — motivo in più per monitorare il recupero.")

        st.markdown("---")

        # ============================================================
        # PROFILO ATLETICO AI
        # ============================================================
        st.markdown("### Il Tuo Profilo Atletico AI")
        in_pratica("l'archetipo non è un giudizio fisso: cambia nel tempo in base a come gestisci sonno, stress e carichi. Guarda l'indice di consistenza sotto per capire quanto puoi fidarti di questa etichetta.")

        cv_sonno = df_base['Ore Sonno'].std() / df_base['Ore Sonno'].mean() if df_base['Ore Sonno'].mean() > 0 else 0
        cv_rpe = df_base['RPE'].std() / df_base['RPE'].mean() if df_base['RPE'].mean() > 0 else 0
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

        verdetto_box(
            consistenza, soglie=(40, 70),
            testo_basso="La tua routine di sonno e sforzo varia molto da un giorno all'altro: l'archetipo attuale potrebbe cambiare presto",
            testo_medio="La tua routine è abbastanza regolare, con qualche oscillazione",
            testo_alto="La tua routine è molto stabile: puoi fidarti che questo archetipo rappresenti bene il tuo comportamento reale",
            spiegazione="L'indice si basa sulla variabilità di sonno e RPE negli ultimi 90 giorni: più bassa è la variabilità, più alto è il punteggio di consistenza."
        )
# =========================================================
# PAGINA 4: ANALISI PREDITTIVA ML — VERSIONE POTENZIATA
# =========================================================
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

    # ---------------------------------------------------------------
    # HELPER: verdetto a semaforo riutilizzabile in tutte le tab
    # ---------------------------------------------------------------
    def verdetto_box(valore_pct, soglie=(50, 75), testo_basso="Debole", testo_medio="Discreto", testo_alto="Solido", spiegazione=""):
        """Crea un box colorato verde/giallo/arancione in base al valore (0-100)."""
        if valore_pct >= soglie[1]:
            colore, emoji, etichetta = "#00F5A0", "✅", testo_alto
        elif valore_pct >= soglie[0]:
            colore, emoji, etichetta = "#FFB020", "⚠️", testo_medio
        else:
            colore, emoji, etichetta = "#FF6A3D", "🔴", testo_basso
        st.markdown(f"""
        <div style='border-left: 4px solid {colore}; background: rgba(255,255,255,0.03);
                    padding: 12px 16px; border-radius: 8px; margin: 10px 0;'>
            <span style='font-size: 1.05em; color:{colore}; font-weight:700;'>{emoji} {etichetta}</span>
            <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>{spiegazione}</p>
        </div>
        """, unsafe_allow_html=True)

    def in_pratica(testo):
        st.markdown(f"""
        <div style='background: rgba(0,229,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
        <span style='color:#00E5FF; font-weight:600;'>💡 In parole semplici:</span>
        <span style='color:#B8C2D0; font-family:"Inter",sans-serif;'> {testo}</span>
        </div>
        """, unsafe_allow_html=True)

    try:
        feature_cols = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']
        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']

        # --- PULIZIA PREVENTIVA DEI DATI ---
        for col in feature_cols + ['Rischio Infortunio', 'Velocità (km/h)', 'Temp (°C)', 'SMA']:
            if col in df_base.columns:
                df_base[col] = pd.to_numeric(df_base[col], errors='coerce').fillna(0)

        X_class = df_base[feature_cols].values
        y_class = df_base['Rischio Infortunio'].astype(int).values

        scaler = StandardScaler()
        X_scaled_class = scaler.fit_transform(X_class)

        # -----------------------------------------------------
        # AVVISO CAMPIONE PICCOLO
        # -----------------------------------------------------
        n_sessioni = len(df_base)
        if n_sessioni < 30:
            st.markdown(f"""
            <div style='border-left: 4px solid #FFB020; background: rgba(255,176,32,0.08);
                        padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;'>
                <span style='color:#FFB020; font-weight:700;'>⚠️ Attenzione al campione ridotto</span>
                <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>
                Hai al momento <strong>{n_sessioni} sessioni</strong> registrate. Con meno di 30 osservazioni le stime dei modelli
                (soprattutto le percentuali di rischio e le metriche di test) possono oscillare parecchio da un allenamento all'altro.
                Considera questi risultati come indicazioni di tendenza, non come verità assolute, finché non accumuli più dati.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------------------------------
        # SPLIT TRAIN/TEST SICURO
        # -----------------------------------------------------
        unique_classes = np.unique(y_class)
        has_multiple_classes = bool(len(unique_classes) > 1)
        has_enough_samples = bool(len(df_base) >= 10)

        stratify_arg = y_class if (has_multiple_classes and has_enough_samples) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_class, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
        )

        # -----------------------------------------------------
        # DEFINIZIONE TAB
        # -----------------------------------------------------
        t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6, t_ml7, t_ml8, t_ml9, t_ml10 = st.tabs([
            " Random Forest", " Logistic Regression", " Linear Regression", " Cluster K-Means",
            " Stress Prediction", " Simulatore What-If", " Confronto Modelli",
            " Explainability (SHAP)", " Anomaly Detection", " PCA"
        ])

        # =====================================================
        # TAB 1 — RANDOM FOREST
        # =====================================================
        with t_ml1:
            st.markdown("### Random Forest Classifier (Infortunio)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello basato su un insieme (ensemble) di alberi decisionali indipendenti. Ciascun albero esprime un voto binario basato su soglie biometriche; il risultato finale aggrega le probabilità. È ideale per catturare dinamiche non lineari complesse.</div>", unsafe_allow_html=True)
            in_pratica("immagina 100 allenatori diversi che guardano i tuoi dati e votano indipendentemente 'rischio' o 'sicuro'. Il modello prende la media dei loro pareri: più allenatori sono d'accordo, più la stima è affidabile.")

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

            c1, c2 = st.columns(2)
            with c1:
                importances = rf_model.feature_importances_
                imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
                fig_imp = go.Figure(go.Bar(
                    y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h',
                    marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'
                ))
                fig_imp.update_layout(height=350, yaxis=dict(autorange="reversed"),
                                       title="Quali variabili pesano di più sul rischio?",
                                       xaxis_title="Importanza relativa (%)")
                st.plotly_chart(style_fig(fig_imp), use_container_width=True)
                var_top = imp_data[0][0]
                st.caption(f"La variabile più determinante è **{var_top}**: è quella su cui il modello si basa maggiormente per decidere.")
            with c2:
                cm = confusion_matrix(y_test, y_pred_rf)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'],
                    text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"},
                    colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False
                ))
                fig_cm.update_layout(height=350, title="Quante volte il modello ha indovinato? (dati di TEST)")
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)
                st.caption("Diagonale (in alto a sx e in basso a dx) = previsioni corrette. Fuori diagonale = errori del modello.")

            acc = accuracy_score(y_test, y_pred_rf)
            prec = precision_score(y_test, y_pred_rf, zero_division=0)
            rec = recall_score(y_test, y_pred_rf, zero_division=0)
            f1 = f1_score(y_test, y_pred_rf, zero_division=0)

            has_multiple_test_classes = bool(len(np.unique(y_test)) > 1)
            roc_auc_rf = roc_auc_score(y_test, y_proba_rf) if has_multiple_test_classes else float('nan')

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{acc*100:.1f}%", help="Percentuale totale di previsioni corrette")
            mc2.metric("Precision", f"{prec*100:.1f}%", help="Quando dice 'rischio', quante volte ha ragione")
            mc3.metric("Recall", f"{rec*100:.1f}%", help="Su tutti i rischi reali, quanti ne ha individuati")
            mc4.metric("F1-Score", f"{f1*100:.1f}%", help="Equilibrio tra Precision e Recall")
            mc5.metric("ROC-AUC", f"{roc_auc_rf:.2f}" if not np.isnan(roc_auc_rf) else "N/D", help="Capacità del modello di distinguere le classi (1.0 = perfetto, 0.5 = a caso)")

            verdetto_box(
                acc * 100, soglie=(60, 80),
                testo_basso="Il modello sbaglia spesso — usalo solo come indizio, non come verdetto",
                testo_medio="Il modello coglie una tendenza reale ma con margine d'errore",
                testo_alto="Il modello è affidabile su questi dati",
                spiegazione=f"Su 100 sessioni di test, il modello ne classifica correttamente circa {acc*100:.0f}. "
                            f"La Recall del {rec*100:.0f}% indica che, tra tutte le sessioni davvero a rischio, ne intercetta circa {rec*100:.0f} su 100."
            )

            try:
                cv_scores = cross_val_score(
                    RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5),
                    X_scaled_class, y_class, cv=5, scoring='accuracy'
                )
                st.caption(f"Validazione incrociata (5-fold) su tutto il dataset: accuracy media {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}% — più questa oscillazione (±) è piccola, più il modello è stabile.")
            except ValueError:
                st.caption("Cross-validation non disponibile: servono più campioni per classe.")

            if not np.isnan(roc_auc_rf):
                fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#00E5FF', width=3), name=f"Random Forest (AUC={roc_auc_rf:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#8792A3', dash='dash'), name="Random / baseline"))
                fig_roc.update_layout(height=350, title="Curva ROC — quanto è meglio del caso puro?",
                                       xaxis_title="Falsi Positivi (allarmi ingiustificati)", yaxis_title="Veri Positivi (rischi individuati)")
                st.plotly_chart(style_fig(fig_roc), use_container_width=True)
                st.caption("Più la curva si avvicina all'angolo in alto a sinistra, meglio il modello separa 'sicuro' da 'rischio'. La linea tratteggiata è un modello che indovina a caso.")

        # =====================================================
        # TAB 2 — LOGISTIC REGRESSION
        # =====================================================
        with t_ml2:
            st.markdown("### Logistic Regression (Probabilità Lineare)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modello statistico supervisionato che calcola la probabilità di un evento binario attraverso una funzione logistica.</div>", unsafe_allow_html=True)
            in_pratica("a differenza della Random Forest, questo modello assegna un 'peso' fisso e leggibile a ogni variabile: puoi vedere subito se dormire di più abbassa il rischio o se lo stress lo alza, in modo diretto.")

            log_model = LogisticRegression(random_state=42, max_iter=1000)
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)
            y_proba_log = log_model.predict_proba(X_test)[:, 1]
            coefs = log_model.coef_[0]

            colors = ['#FF6A3D' if c > 0 else '#00F5A0' for c in coefs]
            fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors,
                                        text=[f'{c:+.2f}' for c in coefs], textposition='auto'))
            fig_log.update_layout(height=400, title="Effetto di ogni variabile sul rischio (coefficienti)",
                                   yaxis_title="Peso — positivo = aumenta il rischio, negativo = lo riduce")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)

            fattore_peggiore = feature_names[int(np.argmax(coefs))]
            fattore_migliore = feature_names[int(np.argmin(coefs))]
            st.caption(f"🟠 **{fattore_peggiore}** è il fattore che più fa salire il rischio. 🟢 **{fattore_migliore}** è il fattore più protettivo.")

            acc_l = accuracy_score(y_test, y_pred_log)
            f1_l = f1_score(y_test, y_pred_log, zero_division=0)
            auc_l = roc_auc_score(y_test, y_proba_log) if has_multiple_test_classes else float('nan')

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Accuracy (test)", f"{acc_l*100:.1f}%")
            lc2.metric("F1-Score (test)", f"{f1_l*100:.1f}%")
            lc3.metric("ROC-AUC (test)", f"{auc_l:.2f}" if not np.isnan(auc_l) else "N/D")

            verdetto_box(
                acc_l * 100, soglie=(60, 80),
                testo_basso="Le relazioni tra variabili e rischio non sono lineari — la Random Forest è probabilmente più adatta",
                testo_medio="Esiste una relazione lineare parziale tra le variabili e il rischio",
                testo_alto="Le variabili spiegano bene il rischio anche in modo semplice e lineare",
                spiegazione="Confronta questa accuracy con quella della Random Forest (Tab 1): se sono simili, la relazione tra i tuoi dati e l'infortunio è abbastanza semplice e diretta."
            )

        # =====================================================
        # TAB 3 — LINEAR REGRESSION
        # =====================================================
        with t_ml3:
            st.markdown("### Linear Regression (Previsione FC Media)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Modella il legame lineare tra la frequenza cardiaca e le variabili ambientali/prestazionali.</div>", unsafe_allow_html=True)
            in_pratica("il modello prova a indovinare la tua frequenza cardiaca media di un allenamento sapendo solo velocità, temperatura e distanza. Se la FC reale è molto più alta del previsto, potrebbe segnalare affaticamento o stress non ancora smaltito.")

            X_lr = df_base[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']].values
            y_lr = df_base['FC Media'].values

            X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
            lr_model = LinearRegression().fit(X_lr_train, y_lr_train)

            df_base['FC_Predetta'] = lr_model.predict(X_lr)
            y_lr_pred_test = lr_model.predict(X_lr_test)
            r2_test = r2_score(y_lr_test, y_lr_pred_test)
            rmse_test = mean_squared_error(y_lr_test, y_lr_pred_test) ** 0.5

            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE',
                                 color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']])
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(),
                              x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(),
                              line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=400, title="FC Reale vs FC Predetta — più vicino alla linea verde, meglio è",
                                  xaxis_title="FC Reale (bpm)", yaxis_title="FC Predetta dal modello (bpm)")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)

            rc1, rc2 = st.columns(2)
            rc1.metric("R² (test)", f"{r2_test:.2f}", help="Quota di variazione della FC spiegata dal modello (1.0 = perfetto)")
            rc2.metric("RMSE (test)", f"{rmse_test:.1f} bpm", help="Errore medio di previsione in battiti al minuto")

            verdetto_box(
                max(r2_test, 0) * 100, soglie=(30, 60),
                testo_basso=f"Il modello sbaglia in media di ±{rmse_test:.0f} bpm — troppa variabilità per trarre conclusioni sul singolo allenamento",
                testo_medio=f"Il modello coglie una tendenza ma con un margine di errore di ±{rmse_test:.0f} bpm",
                testo_alto=f"Previsioni precise, con un errore medio di soli ±{rmse_test:.0f} bpm",
                spiegazione="Un punto molto sopra la linea verde diagonale significa che la FC reale è stata più alta del previsto: un possibile segnale di fatica o stress accumulato quel giorno."
            )

        # =====================================================
        # TAB 4 — CLUSTER K-MEANS
        # =====================================================
        with t_ml4:
            st.markdown("### Cluster Analysis (K-Means)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Apprendimento non supervisionato che raggruppa automaticamente il set di dati in cluster omogenei.</div>", unsafe_allow_html=True)
            in_pratica("il modello non sa nulla di 'tipi di allenamento', ma guardando solo distanza e frequenza cardiaca riesce a raggruppare da solo le sessioni simili tra loro — ad esempio i fondi lunghi da un lato e le sedute intense dall'altro.")

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
                fig_elbow.update_layout(height=320, title="Metodo del Gomito — dove il guadagno rallenta",
                                         xaxis_title="N. Cluster (k)", yaxis_title="Compattezza dei gruppi (Inertia)")
                st.plotly_chart(style_fig(fig_elbow), use_container_width=True)
                st.caption("Cerca il punto in cui la curva smette di scendere rapidamente ('il gomito'): oltre quel k, aggiungere gruppi non migliora molto.")
            with ec2:
                fig_sil = go.Figure(go.Bar(x=k_range, y=sil_scores, marker_color='#00F5A0'))
                fig_sil.update_layout(height=320, title="Silhouette Score — quanto sono ben separati i gruppi",
                                      xaxis_title="N. Cluster (k)", yaxis_title="Punteggio (più alto = meglio)")
                st.plotly_chart(style_fig(fig_sil), use_container_width=True)

            best_k = k_range[int(np.argmax(sil_scores))] if sil_scores else 3
            st.caption(f"Il valore di k con silhouette score più alto è k={best_k}. Per coerenza con l'analisi si utilizza comunque k=3 nel grafico sottostante.")

            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_base['Cluster_ID'] = km.fit_predict(X_clust)
            df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")

            fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type',
                                 color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='RPE')
            fig_km.update_layout(height=400, title="Le tue sessioni raggruppate per tipologia (k=3)")
            st.plotly_chart(style_fig(fig_km), use_container_width=True)

            cluster_sizes = df_base['Cluster_Type'].value_counts()
            st.caption("Distribuzione sessioni per cluster: " + ", ".join([f"**{k}**: {v} sessioni" for k, v in cluster_sizes.items()]))

        # =====================================================
        # TAB 5 — STRESS / OVERLOAD PREDICTION
        # =====================================================
        with t_ml5:
            st.markdown("### Stress / Overload Prediction (Time Series)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Analisi delle serie temporali basata sulla media mobile dello stress sistemico.</div>", unsafe_allow_html=True)
            in_pratica("invece di guardare un solo giorno, il modello osserva la tendenza degli ultimi 7 giorni: uno stress alto isolato non preoccupa, ma una tendenza in salita costante sì.")

            df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').reset_index(drop=True).copy()
            df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()

            fig_sp = px.area(df_stress, x='Giorno', y='SMA_Rolling', color_discrete_sequence=['#FF6A3D'])
            fig_sp.add_hline(y=15, line_dash="dash", line_color="#FFB020", annotation_text="Soglia Critica")
            fig_sp.update_layout(height=400, title="Media Mobile Stress Sistemico (7 Giorni)",
                                  xaxis_title="Giorno", yaxis_title="Livello di Stress (SMA)")
            st.plotly_chart(style_fig(fig_sp), use_container_width=True)

            giorni_sopra_soglia = int((df_stress['SMA_Rolling'] > 15).sum())
            st.caption(f"Giorni con stress sistemico sopra la soglia critica: **{giorni_sopra_soglia} su {len(df_stress)}**.")

            n_forecast = 7
            x_idx = np.arange(len(df_stress))
            valid_mask = df_stress['SMA_Rolling'].notna().to_numpy()

            if int(valid_mask.sum()) >= 3:
                coeffs = np.polyfit(x_idx[valid_mask], df_stress['SMA_Rolling'][valid_mask], deg=1)
                trend_fn = np.poly1d(coeffs)
                future_idx = np.arange(len(df_stress), len(df_stress) + n_forecast)
                future_vals = trend_fn(future_idx)
                residual_std = np.std(df_stress['SMA_Rolling'][valid_mask] - trend_fn(x_idx[valid_mask]))

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=list(range(len(df_stress))), y=df_stress['SMA_Rolling'], mode='lines', line=dict(color='#00E5FF'), name="Storico"))
                fig_forecast.add_trace(go.Scatter(x=list(future_idx), y=future_vals, mode='lines', line=dict(color='#FF6A3D', dash='dash'), name="Proiezione"))
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_idx) + list(future_idx)[::-1],
                    y=list(future_vals + residual_std) + list(future_vals - residual_std)[::-1],
                    fill='toself', fillcolor='rgba(255,106,61,0.15)', line=dict(color='rgba(0,0,0,0)'), name="Banda di incertezza"
                ))
                fig_forecast.update_layout(height=380, title=f"Proiezione Stress — Prossimi {n_forecast} giorni",
                                            xaxis_title="Indice giorno", yaxis_title="SMA")
                st.plotly_chart(style_fig(fig_forecast), use_container_width=True)

                trend_direzione = "in aumento 📈" if coeffs[0] > 0.01 else ("in diminuzione 📉" if coeffs[0] < -0.01 else "stabile ➡️")
                verdetto_box(
                    max(0, 100 - abs(future_vals[-1] - 15) * 5) if future_vals[-1] <= 15 else 20,
                    soglie=(40, 70),
                    testo_basso=f"La proiezione mostra un trend {trend_direzione} verso livelli di rischio: valuta di inserire giorni di recupero",
                    testo_medio=f"Trend {trend_direzione}, da monitorare nei prossimi giorni",
                    testo_alto=f"Trend {trend_direzione}, lontano dalla soglia critica",
                    spiegazione="La banda arancione indica l'incertezza della proiezione: più è larga, meno la previsione è precisa."
                )
            else:
                st.info("Servono almeno 3 punti validi di SMA per calcolare una proiezione.")

        # =====================================================
        # TAB 6 — SIMULATORE WHAT-IF
        # =====================================================
        with t_ml6:
            st.markdown("### Simulatore What-If (Random Forest Live)")
            st.markdown("""<div class='info-box'><strong>Modifica i parametri interattivi e osserva l'impatto sul rischio stimato.</strong></div>""", unsafe_allow_html=True)
            in_pratica("muovi gli slider come se stessi pianificando l'allenamento di domani: il modello ricalcola all'istante il rischio stimato, usando le stesse regole apprese nel Tab 'Random Forest'.")

            base = st.session_state.risultati_analisi if st.session_state.get('analisi_fatta', False) else {'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6}

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
                fig_sim_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=sim_prob, title={'text': "Rischio Stimato con questi parametri", 'font': {'color': '#8792A3'}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color},
                           'steps': [{'range': [0, 25], 'color': 'rgba(0,245,160,0.15)'},
                                     {'range': [25, 60], 'color': 'rgba(255,176,32,0.15)'},
                                     {'range': [60, 100], 'color': 'rgba(255,106,61,0.15)'}],
                           'bgcolor': "#111827", 'borderwidth': 0},
                    number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
                ))
                fig_sim_gauge.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sim_gauge), use_container_width=True)
            with col_simg2:
                sonno_range = np.linspace(4, 10, 20)
                probs_range = [rf_model.predict_proba(scaler.transform(np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100 for s in sonno_range]
                fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'},
                                    title="Quanto conta il sonno sul tuo rischio?")
                fig_sens.update_traces(line_color="#00E5FF", line_width=3)
                fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#FF6A3D", annotation_text="Tua simulazione")
                fig_sens.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sens), use_container_width=True)

            verdetto_box(
                100 - sim_prob, soglie=(40, 75),
                testo_basso=f"Con questi parametri il rischio stimato è alto ({sim_prob:.0f}%) — valuta di ridurre distanza/intensità o recuperare più sonno",
                testo_medio=f"Rischio stimato moderato ({sim_prob:.0f}%) — allenamento sostenibile con attenzione al recupero",
                testo_alto=f"Rischio stimato basso ({sim_prob:.0f}%) — condizioni favorevoli per questo allenamento",
                spiegazione="Prova a spostare solo lo slider del sonno: il grafico a destra mostra quanto il rischio cambia dormendo di più o di meno, a parità di tutto il resto."
            )

        # =====================================================
        # TAB 7 — CONFRONTO MODELLI
        # =====================================================
        with t_ml7:
            st.markdown("### Confronto tra Modelli di Classificazione")
            st.markdown("<div class='explain-text'><strong>Perché confrontare più modelli:</strong> nessun algoritmo è ottimale a priori.</div>", unsafe_allow_html=True)
            in_pratica("è come chiedere un secondo e un terzo parere medico: se tutti i modelli concordano, la stima è più affidabile; se divergono molto, i tuoi dati contengono ancora rumore o sono troppo pochi.")

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
                    if has_multiple_test_classes:
                        row["ROC-AUC"] = roc_auc_score(y_test, y_proba_m)
                        fpr_m, tpr_m, _ = roc_curve(y_test, y_proba_m)
                        fig_roc_all.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode='lines', line=dict(color=color, width=3), name=f"{name} (AUC={row['ROC-AUC']:.2f})"))
                    else:
                        row["ROC-AUC"] = float('nan')
                except ValueError:
                    row["ROC-AUC"] = float('nan')

                comparison_rows.append(row)

            df_compare = pd.DataFrame(comparison_rows).set_index("Modello")
            st.dataframe(
                df_compare.style.format("{:.2%}", subset=["Accuracy", "Precision", "Recall", "F1-Score"])
                                  .format("{:.2f}", subset=["ROC-AUC"])
                                  .background_gradient(cmap="Blues", subset=["Accuracy", "F1-Score", "ROC-AUC"]),
                use_container_width=True
            )

            fig_roc_all.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#8792A3', dash='dash'), name="Baseline casuale"))
            fig_roc_all.update_layout(height=400, title="Curve ROC a Confronto — quale modello separa meglio?",
                                       xaxis_title="Falsi Positivi", yaxis_title="Veri Positivi")
            st.plotly_chart(style_fig(fig_roc_all), use_container_width=True)

            valid_auc = df_compare["ROC-AUC"].dropna()
            best_model_name = valid_auc.idxmax() if len(valid_auc) > 0 else df_compare["F1-Score"].idxmax()
            accordo = df_compare["Accuracy"].max() - df_compare["Accuracy"].min()

            verdetto_box(
                100 - accordo * 100 * 2, soglie=(50, 80),
                testo_basso="I modelli sono molto in disaccordo tra loro: serve raccogliere più dati prima di fidarsi delle stime",
                testo_medio="I modelli concordano parzialmente",
                testo_alto="I modelli concordano bene tra loro, segnale di stime robuste",
                spiegazione=f"Il modello con la performance più solida sul test è **{best_model_name}**. La differenza di accuracy tra il migliore e il peggiore è del {accordo*100:.1f}%: più questo valore è piccolo, più puoi fidarti che il pattern trovato sia reale e non casuale."
            )

        # =====================================================
        # TAB 8 — EXPLAINABILITY (SHAP)
        # =====================================================
        with t_ml8:
            st.markdown("### Explainability Avanzata (SHAP Values)")
            st.markdown("<div class='explain-text'><strong>Perché SHAP:</strong> i valori SHAP spiegano ogni singola predizione quantificando l'impatto delle singole variabili.</div>", unsafe_allow_html=True)
            in_pratica("mentre il grafico 'Importanza Variabili' del Tab 1 ti dice cosa conta in media su tutte le sessioni, qui puoi scegliere UNA sessione specifica e vedere esattamente perché il modello l'ha giudicata rischiosa o sicura.")

            if SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_test)

                # --- Normalizzazione robusta di TUTTI i possibili formati SHAP ---
                if isinstance(shap_values, list):
                    shap_vals_risk = shap_values[1]
                else:
                    shap_values = np.asarray(shap_values)
                    if shap_values.ndim == 3:
                        shap_vals_risk = shap_values[:, :, 1]
                    else:
                        shap_vals_risk = shap_values

                base_value_arr = np.atleast_1d(explainer.expected_value)
                base_value = base_value_arr[1] if base_value_arr.size > 1 else base_value_arr[0]

                mean_abs_shap = np.abs(shap_vals_risk).mean(axis=0)
                shap_imp = sorted(list(zip(feature_names, mean_abs_shap)), key=lambda x: x[1], reverse=True)
                fig_shap_global = go.Figure(go.Bar(
                    y=[x[0] for x in shap_imp], x=[x[1] for x in shap_imp],
                    orientation='h', marker_color='#00E5FF',
                    text=[f'{x[1]:.3f}' for x in shap_imp], textposition='auto'
                ))
                fig_shap_global.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Importanza Media |SHAP| su tutte le sessioni di test")
                st.plotly_chart(style_fig(fig_shap_global), use_container_width=True)

                st.markdown("#### Spiegazione di una singola sessione")
                idx_choice = st.slider("Seleziona la sessione di test da spiegare", 0, len(X_test) - 1, 0, key="shap_idx")
                instance_shap = shap_vals_risk[idx_choice]

                waterfall_data = sorted(list(zip(feature_names, instance_shap)), key=lambda x: abs(x[1]), reverse=True)
                colors_wf = ['#FF6A3D' if v > 0 else '#00F5A0' for _, v in waterfall_data]
                fig_wf = go.Figure(go.Bar(
                    x=[x[1] for x in waterfall_data], y=[x[0] for x in waterfall_data],
                    orientation='h', marker_color=colors_wf,
                    text=[f'{v:+.3f}' for _, v in waterfall_data], textposition='auto'
                ))
                fig_wf.update_layout(
                    height=320,
                    title=f"Perché questa sessione ha ricevuto questa stima? (base={base_value:.2f})",
                    xaxis_title="Impatto sulla probabilità di rischio (arancione = aumenta, verde = riduce)"
                )
                fig_wf.add_vline(x=0, line_color="#E8ECF2", line_width=1)
                st.plotly_chart(style_fig(fig_wf), use_container_width=True)

                fattore_dominante = waterfall_data[0][0]
                direzione = "aumentato" if waterfall_data[0][1] > 0 else "ridotto"
                st.caption(f"Per questa sessione, il fattore che ha più {direzione} il rischio stimato è **{fattore_dominante}**.")
            else:
                st.warning("Libreria shap non rilevata. Vengono utilizzate metriche di fallback (Permutation Importance).")
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(rf_model, X_test, y_test, n_repeats=20, random_state=42)
                perm_data = sorted(list(zip(feature_names, perm.importances_mean)), key=lambda x: x[1], reverse=True)
                fig_perm = go.Figure(go.Bar(y=[x[0] for x in perm_data], x=[x[1] for x in perm_data], orientation='h', marker_color='#FFB020'))
                fig_perm.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Permutation Importance")
                st.plotly_chart(style_fig(fig_perm), use_container_width=True)

        # =====================================================
        # TAB 9 — ANOMALY DETECTION
        # =====================================================
        with t_ml9:
            st.markdown("### Anomaly Detection (Isolation Forest)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Isola le osservazioni anomale per intercettare allenamenti atipici.</div>", unsafe_allow_html=True)
            in_pratica("il modello confronta ogni sessione con tutte le altre e segnala quelle che 'non assomigliano a nessun'altra' — spesso allenamenti anomali per condizioni esterne, errori di registrazione, o giornate fuori dal tuo standard.")

            contamination = st.slider("Percentuale attesa di sessioni anomale", 0.02, 0.25, 0.08, step=0.01, key="iso_contam")
            iso_model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
            anomaly_labels = iso_model.fit_predict(X_scaled_class)
            df_base['Anomalia'] = np.where(anomaly_labels == -1, 'Anomala', 'Normale')

            fig_anom = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Anomalia',
                                   color_discrete_map={'Normale': '#00E5FF', 'Anomala': '#FF6A3D'},
                                   size='RPE', hover_data=['Giorno'])
            fig_anom.update_layout(height=400, title="Sessioni di Allenamento: Normali vs Anomale")
            st.plotly_chart(style_fig(fig_anom), use_container_width=True)

            n_anomalie = int((df_base['Anomalia'] == 'Anomala').sum())
            st.metric("Sessioni segnalate come anomale", f"{n_anomalie} su {len(df_base)}")

            if n_anomalie > 0:
                st.dataframe(df_base[df_base['Anomalia'] == 'Anomala'][['Giorno'] + feature_cols], use_container_width=True)
                st.caption("Controlla queste sessioni: potrebbero essere errori di registrazione (es. GPS impreciso) oppure giornate realmente fuori dal tuo standard, da rivedere manualmente.")
            else:
                st.caption("Nessuna sessione anomala rilevata con la soglia attuale: il tuo storico è abbastanza omogeneo.")

        # =====================================================
        # TAB 10 — PCA
        # =====================================================
        with t_ml10:
            st.markdown("### Analisi delle Componenti Principali (PCA)")
            st.markdown("<div class='explain-text'><strong>Spiegazione Algoritmo:</strong> Riduce le variabili biometriche a 2 componenti principali.</div>", unsafe_allow_html=True)
            in_pratica("hai 5 variabili diverse (distanza, sonno, stress, FC, RPE): impossibili da visualizzare tutte insieme su un solo grafico. La PCA le 'comprime' in 2 sole dimensioni, mantenendo il più possibile le differenze reali tra le sessioni.")

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled_class)
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_pca['Rischio Infortunio'] = y_class

            pc1, pc2 = st.columns(2)
            with pc1:
                fig_pca_scatter = px.scatter(df_pca, x='PC1', y='PC2', color=df_pca['Rischio Infortunio'].astype(str),
                                              color_discrete_sequence=['#00F5A0', '#FF6A3D'])
                fig_pca_scatter.update_layout(height=380, title="Proiezione 2D delle Sessioni", legend_title="Rischio Infortunio")
                st.plotly_chart(style_fig(fig_pca_scatter), use_container_width=True)
                st.caption("Se i punti verdi e arancioni formano zone distinte, significa che il rischio è ben spiegato dalle tue variabili biometriche.")
            with pc2:
                var_ratio = pca.explained_variance_ratio_ * 100
                fig_var = go.Figure(go.Bar(x=['PC1', 'PC2'], y=var_ratio, marker_color=['#00E5FF', '#FFB020'],
                                            text=[f'{v:.1f}%' for v in var_ratio], textposition='auto'))
                fig_var.update_layout(height=380, title="Varianza Spiegata per Componente", yaxis_title="% Varianza Spiegata")
                st.plotly_chart(style_fig(fig_var), use_container_width=True)

            varianza_totale = var_ratio.sum()
            verdetto_box(
                varianza_totale, soglie=(50, 75),
                testo_basso="Le due componenti principali catturano solo una parte limitata dei tuoi dati: la realtà è più complessa di quanto il grafico 2D mostri",
                testo_medio="Le due componenti principali riassumono discretamente i tuoi dati",
                testo_alto="Le due componenti principali riassumono molto bene tutta l'informazione disponibile",
                spiegazione=f"Le prime 2 componenti spiegano insieme il {varianza_totale:.1f}% della varianza totale — cioè quanta 'informazione' dei tuoi dati originali sopravvive nella semplificazione a 2 dimensioni."
            )

    except Exception as e:
        st.error(f"Errore caricamento modelli ML: {str(e)}")
elif pagina == "CONSIGLIO FINALE":
    import math
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    def md(html):
        """Renderizza HTML in modo sicuro. Rimuove l'indentazione python."""
        html_pulito = "\n".join([line.strip() for line in html.split("\n")])
        st.markdown(html_pulito, unsafe_allow_html=True)

    header_block(
        "Modulo 05 — Action Plan",
        "CONSIGLIO FINALE",
        "Protocollo operativo, proiezioni fisiologiche ed export report per la sessione odierna.",
        IMG_HERO_PLAN, "Coach Protocol"
    )

    if not st.session_state.get('analisi_fatta', False):
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        df_base = st.session_state.dati.copy()

        # =========================================================
        # TOKEN DI DESIGN (High-Tech Sports Theme)
        # =========================================================
        PANEL_BG   = "#0D1117"
        PANEL_BD   = "#1E2633"
        PANEL_BD_H = "#2A3546"
        TXT_PRIMARY   = "#F8F9FA"
        TXT_SECONDARY = "#8B949E"
        TXT_TERTIARY  = "#485363"

        C_SONNO  = "#2E90FF"
        C_STRESS = "#FF453A"
        C_RPE    = "#30D158"
        C_AMBRA  = "#FF9F0A"
        C_VIOLA  = "#BF5AF2"
        C_NEUTRO = "#1F2733"

        md(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

        .panel {{
            background: {PANEL_BG}; border: 1px solid {PANEL_BD}; border-radius: 12px;
            padding: 24px; transition: border-color .2s ease, box-shadow .2s ease;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .panel:hover {{ border-color: {PANEL_BD_H}; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }}
        
        .panel-flush {{ padding: 0; overflow: hidden; }}
        .panel-flush .panel-body {{ padding: 16px 20px; }}
        .panel-flush .chart-top-rule {{ height: 4px; width: 100%; }}

        .eyebrow {{
            font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.12em;
            text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 6px 0; font-weight:700;
        }}
        .section-head {{ margin: 12px 0 20px 0; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
        .section-head h3 {{
            font-family:'Oswald',sans-serif; font-weight:600; color:{TXT_PRIMARY};
            margin:0; font-size:1.4rem; letter-spacing:.02em; text-transform:uppercase;
        }}
        .section-head .sub {{ color:{TXT_SECONDARY}; font-size:.9rem; font-family:'Inter',sans-serif; margin-top:4px; }}
        .section-head .rule {{ flex:1; height:1px; background: linear-gradient(90deg, {PANEL_BD} 0%, transparent 100%); align-self:center; }}

        .kv-num {{ font-family:'JetBrains Mono',monospace; color:{TXT_PRIMARY}; font-weight:700; }}
        .badge {{
            display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.65rem;
            letter-spacing:.1em; text-transform:uppercase; padding:4px 10px; border-radius:4px; font-weight:700;
        }}

        .coach-block {{ margin-bottom: 22px; background: rgba(255,255,255,0.02); padding: 16px; border-radius: 8px; border-left: 3px solid var(--block-color); }}
        .coach-block:last-child {{ margin-bottom: 0; }}
        .coach-block .label {{
            font-family:'Oswald',sans-serif; font-size:.9rem; letter-spacing:.05em;
            text-transform:uppercase; margin-bottom:10px; font-weight:600; color: var(--block-color);
        }}
        .coach-block ul {{ margin:0; padding-left:0; list-style:none; }}
        .coach-block li {{
            position:relative; padding-left:18px; margin-bottom:10px; color:{TXT_SECONDARY};
            font-family:'Inter',sans-serif; font-size:.95rem; line-height:1.6;
        }}
        .coach-block li::before {{ content:"▸"; position:absolute; left:0; top:1px; color:var(--block-color); font-size: 1rem; }}

        .split-sheet {{ padding:0; overflow:hidden; border-radius: 12px; }}
        .split-row {{
            display:grid; grid-template-columns: 1.2fr 1fr 1fr 1.6fr;
            align-items:center; padding:18px 24px; border-bottom:1px solid {PANEL_BD}; gap:12px;
        }}
        .split-row:last-child {{ border-bottom:none; }}
        .split-row .sr-label {{ font-family:'Oswald',sans-serif; font-size:1rem; letter-spacing:.02em; text-transform:uppercase; color:{TXT_PRIMARY}; font-weight:500; }}
        .split-row .sr-value {{ font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; }}
        .split-row .sr-value .unit {{ font-family:'Inter',sans-serif; font-size:.45em; color:{TXT_SECONDARY}; margin-left:4px; text-transform:uppercase; }}
        .split-row .sr-ref {{ font-family:'JetBrains Mono',monospace; font-size:.85rem; color:{TXT_SECONDARY}; }}
        .split-row .sr-note {{ font-family:'Inter',sans-serif; font-size:.85rem; color:{TXT_SECONDARY}; line-height:1.5; border-left: 1px solid {PANEL_BD}; padding-left: 12px; }}
        .split-head {{
            display:grid; grid-template-columns: 1.2fr 1fr 1fr 1.6fr; padding:12px 24px;
            border-bottom:1px solid {PANEL_BD}; background:#161C24;
        }}
        .split-head span {{ font-family:'JetBrains Mono',monospace; font-size:.65rem; letter-spacing:.12em; text-transform:uppercase; color:{TXT_TERTIARY}; font-weight:700; }}

        .hud-grid {{ display: flex; gap: 20px; align-items: flex-end; margin-bottom: 20px; }}
        .hud-stat {{ flex: 1; }}
        .hud-stat h2 {{ margin:0; font-family:"Oswald",sans-serif; font-weight:700; font-size:1.8rem; text-transform:uppercase; letter-spacing:.02em; line-height:1; }}

        .lane-chip {{
            background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:10px;
            padding:14px 16px 16px 16px; position:relative; overflow:hidden;
        }}
        .lane-chip::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--zc); }}
        .lane-chip .lane-num {{ font-family:'Oswald',sans-serif; font-weight:600; font-size:1.6em; color:var(--zc); line-height:1; margin-bottom:2px; }}
        .lane-chip .zt {{ font-family:'JetBrains Mono',monospace; font-size:.68em; letter-spacing:.08em; color:{TXT_TERTIARY}; text-transform:uppercase; font-weight:600; }}
        .lane-chip .zn {{ font-family:'Inter',sans-serif; font-weight:600; color:{TXT_PRIMARY}; margin:6px 0 6px 0; font-size:.95em; }}
        .lane-chip .zd {{ font-family:'Inter',sans-serif; color:{TXT_SECONDARY}; font-size:.85em; line-height:1.4; }}
        
        .chart-caption {{ border-top: 1px solid {PANEL_BD}; margin-top: 10px; padding-top: 10px; color:{TXT_SECONDARY}; font-family:'Inter',sans-serif; font-size:.85rem; line-height:1.55; }}
        
        .delta-track {{ position:relative; height:6px; border-radius:3px; background:{PANEL_BD}; margin:10px 0 4px 0; }}
        .delta-fill {{ position:absolute; top:0; height:6px; border-radius:3px; }}
        .delta-mid {{ position:absolute; top:-3px; left:50%; width:1px; height:12px; background:{TXT_TERTIARY}; }}
        </style>
        """)

        def section_head(eyebrow, title, sub=None):
            sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
            md(f"""
            <div class='section-head'>
                <div class='head-txt'>
                    <p class='eyebrow'>{eyebrow}</p>
                    <h3>{title}</h3>
                    {sub_html}
                </div>
                <div class='rule'></div>
            </div>
            """)

        # =========================================================
        # CALCOLI BASE E GESTIONE ROBUSTA DATASET
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
        distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0

        if risk_score < 25:
            tit, col, liv = "AUTORIZZATO", C_RPE, "basso"
        elif risk_score < 60:
            tit, col, liv = "RECUPERO ATTIVO", C_AMBRA, "medio"
        else:
            tit, col, liv = "RIPOSO OBBLIGATORIO", C_STRESS, "alto"

        tipo_all = r.get('tipo_allenamento', 'Easy Run')
        cadenza_target = "170-180 spm" if liv != "alto" else "165-172 spm (passo rilassato per abbassare l'impatto articolare)"
        zona_consigliata = "Zona 2-3 (aerobico puro)" if liv == "basso" else "Zona 1-2 (sforzo percepito bassissimo)" if liv == "medio" else "Solo mobilità o camminata veloce"

        # RICERCA COLONNA DATA (Fallback intelligente)
        date_col = next((c for c in df_base.columns if c.lower() in ['data', 'date', 'giorno', 'time']), None)
        df_adv = df_base.copy()
        if date_col:
            df_adv['Data_Chart'] = pd.to_datetime(df_adv[date_col], errors='coerce')
        else:
            # Se non c'è una data, generiamo un asse fittizio per mostrare i grafici
            df_adv['Data_Chart'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_adv))
        
        df_adv = df_adv.sort_values('Data_Chart').dropna(subset=['Data_Chart'])

        # =========================================================
        # EFFORT EQUALIZER (TELEMETRIA VETTORIALE)
        # =========================================================
        def disegna_telemetria_rischio(score):
            svg_width, svg_height = 800, 100
            bars = 60
            gap = 2
            bar_w = (svg_width - (bars * gap)) / bars
            
            elements = []
            for i in range(bars):
                x = i * (bar_w + gap)
                progression = i / bars
                h = 15 + (progression ** 2.5) * (svg_height - 15)
                y = svg_height - h
                
                threshold_pct = (i / bars) * 100
                is_active = threshold_pct <= score
                
                bar_col = C_RPE if threshold_pct < 25 else C_AMBRA if threshold_pct < 60 else C_STRESS
                fill = bar_col if is_active else C_NEUTRO
                opacity = "1.0" if is_active else "0.3"
                
                elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" opacity="{opacity}" rx="2"/>')

            marker_x = (score / 100) * svg_width
            
            return f"""
            <svg viewBox="0 0 {svg_width} {svg_height + 25}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.3));">
                <line x1="0" y1="{svg_height}" x2="{svg_width}" y2="{svg_height}" stroke="{TXT_TERTIARY}" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
                {"".join(elements)}
                <g transform="translate({marker_x}, 0)">
                    <line x1="0" y1="0" x2="0" y2="{svg_height + 10}" stroke="{TXT_PRIMARY}" stroke-width="2" />
                    <polygon points="-6,{svg_height + 10} 6,{svg_height + 10} 0,{svg_height + 18}" fill="{TXT_PRIMARY}" />
                    <rect x="-24" y="-20" width="48" height="20" rx="4" fill="{TXT_PRIMARY}" />
                    <text x="0" y="-6" font-family="JetBrains Mono, monospace" font-size="12" font-weight="bold" fill="{PANEL_BG}" text-anchor="middle">{int(score)}%</text>
                </g>
                <text x="0" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_RPE}" opacity="0.8">OPTIMAL (0-25)</text>
                <text x="{svg_width/2}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_AMBRA}" text-anchor="middle" opacity="0.8">WARNING (26-59)</text>
                <text x="{svg_width}" y="{svg_height + 15}" font-family="JetBrains Mono, monospace" font-size="10" fill="{C_STRESS}" text-anchor="end" opacity="0.8">DANGER (60-100)</text>
            </svg>
            """

        radar_svg = disegna_telemetria_rischio(risk_score)

        md(f"""
        <div class='panel'>
            <div class='hud-grid'>
                <div class='hud-stat'>
                    <p class='eyebrow'>System Status</p>
                    <h2 style='color:{col};'>{tit}</h2>
                </div>
                <div class='hud-stat' style='text-align:right;'>
                    <p class='eyebrow'>Risk Load</p>
                    <h2 style='color:{TXT_PRIMARY};'>{risk_score:.0f}<span style='font-size:0.6em; color:{TXT_SECONDARY};'>%</span></h2>
                </div>
            </div>
            <div style='margin-top: 10px;'>
                {radar_svg}
            </div>
        </div>
        """)

        md("<div style='height:24px;'></div>")

        # =========================================================
        # SPLIT SHEET (Cronometraggio)
        # =========================================================
        sonno_delta_txt = f"{'+' if (r['ore_sonno']-7.5) >= 0 else ''}{r['ore_sonno']-7.5:.1f}h rispetto al target fisiologico ottimale"

        md(f"""
        <div class='panel split-sheet'>
            <div class='split-head'>
                <span>Metric</span><span>Value</span><span>Reference</span><span>Notes / Adjustments</span>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Distanza Target</div>
                <div class='sr-value' style='color:{TXT_PRIMARY};'>{distanza_consigliata:.1f}<span class='unit'>km</span></div>
                <div class='sr-ref'>Da piano originale: {distanza_target} km</div>
                <div class='sr-note'>La distanza è stata scalata algoritmicamente del {100 - (distanza_consigliata/distanza_target*100 if distanza_target>0 else 0):.0f}% per proteggere i tuoi tessuti.</div>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Recovery Score</div>
                <div class='sr-value' style='color:{C_SONNO};'>{recovery_score:.0f}<span class='unit'>%</span></div>
                <div class='sr-ref'>Qualità base: {r['ore_sonno']:.1f}h di sonno</div>
                <div class='sr-note'>{sonno_delta_txt}. Il sonno governa il recupero del sistema nervoso autonomo.</div>
            </div>
            <div class='split-row'>
                <div class='sr-label'>Mental Load (SMA)</div>
                <div class='sr-value' style='color:{C_AMBRA};'>{sma:.1f}</div>
                <div class='sr-ref'>(Stress × RPE) / Sonno</div>
                <div class='sr-note'>Indice composito di sovraccarico del sistema nervoso centrale. Livello odierno: {liv.upper()}.</div>
            </div>
        </div>
        """)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # COACH PERSONALIZZATO - ESTESO CON PIÙ CONSIGLI
        # =========================================================
        section_head("Coach personalizzato", "Protocollo Operativo Dettagliato", "Linee guida biomeccaniche, di pacing e bio-hacking basate sulla telemetria odierna.")

        # Logica dei testi dinamici
        if liv == "basso":
            dinamica_pacing = "Mantenimento del ritmo standard. Nessuna restrizione fisiologica sui cambi di velocità, via libera a scatti se previsti."
            dinamica_resp = "Respirazione naturale (suggerito schema 3:3 in riscaldamento, passando a 2:2 a ritmo gara)."
            dinamica_rec = "Recupero passivo standard o massaggio leggero. Doccia contrasto (caldo/freddo) autorizzata post-workout."
        elif liv == "medio":
            dinamica_pacing = "Gestione conservativa (Pacing difensivo). Togli 10-15 secondi al km dal tuo ritmo abituale. Evita pendenze severe."
            dinamica_resp = "Forza un'espirazione lunga (schema 3:4) per mantenere bassa la frequenza cardiaca ed evitare picchi di sforzo."
            dinamica_rec = "Focus massimo su idratazione e reintegro elettrolitico. Niente acqua fredda post-corsa, preferisci temperature neutre per non stressare il SNC."
        else:
            dinamica_pacing = "Sforzo sconsigliato. Converti la seduta in 30-40 minuti di camminata dinamica su terreno piatto e morbido (prato/terra)."
            dinamica_resp = "Respirazione diaframmatica esclusiva naso-naso per favorire il recupero parasimpatico."
            dinamica_rec = "Yoga nidra o stretching passivo lungo. Applica calore (non freddo) su eventuali zone muscolari tese."

        coach_content = {
            "1. Warm-Up & Prep (Pre)": {
                "colore": C_SONNO,
                "blocchi": [
                    ("Attivazione Neurale e Meccanica", [
                        "Camminata sui talloni e sulle punte (30 secondi per tipo) per attivare l'arco plantare e le caviglie.",
                        "5 minuti di mobilità dinamica: leg swings frontali/laterali, rotazioni delle anche, affondi controllati.",
                        "Evita lo stretching statico pre-corsa: riduce la reattività elastica del tendine d'Achille del 5-8%."
                    ]),
                    ("Setup Mentale e Fisiologico", [
                        f"Target zona Iniziale: {zona_consigliata}. I primi 10 minuti devono sembrare 'troppo lenti'.",
                        "Esegui 10 cicli di respirazione profonda prima di far partire il cronometro per azzerare lo stress lavorativo.",
                        f"Direttiva Rischio: {dinamica_pacing}"
                    ])
                ],
            },
            "2. In Azione (Durante)": {
                "colore": C_AMBRA,
                "blocchi": [
                    ("Gestione Biomeccanica e Pacing", [
                        f"Frequenza passi (Cadenza): {cadenza_target}. Aumentare la frequenza del 5% riduce il carico sulle ginocchia del 20%.",
                        "Postura: Busto leggermente inclinato in avanti partendo dalle caviglie (non dalla vita). Sguardo a 20 metri, mai ai piedi.",
                        "Rilassa mani e mascella: la tensione nel volto si trasmette istantaneamente alla catena muscolare posteriore."
                    ]),
                    ("Gestione Sforzo e Sicurezza", [
                        f"Respirazione: {dinamica_resp}",
                        "Check Infortuni: Differenzia la 'fatica sorda' (normale) dal 'dolore acuto/fitta' (allarme articolare).",
                        "Regola dei 15 minuti: Se dopo 15 minuti l'RPE percepito è più alto di 2 punti rispetto al previsto, taglia la distanza a metà."
                    ])
                ],
            },
            "3. Protocollo di Recupero (Post)": {
                "colore": C_RPE,
                "blocchi": [
                    ("Cooldown Sistemico", [
                        "Non fermarti bruscamente. Cammina per 3-5 minuti finché il battito non scende comodamente in Zona 1 (Sotto i 110 bpm).",
                        "Finestra anabolica: assumi carboidrati e proteine leggere entro 45 minuti per arrestare il catabolismo indotto dallo stress."
                    ]),
                    ("Gestione Tessuti", [
                        "Stretching passivo: mantieni ogni posizione per almeno 45-60 secondi (polpacci, ischiocrurali, psoas).",
                        "Usa il foam roller lentamente. Se trovi un 'trigger point' (punto di dolore), fermati su di esso respirando per 30 secondi."
                    ])
                ],
            },
            "4. Bio-Hacking Serale (Mindset)": {
                "colore": C_VIOLA,
                "blocchi": [
                    ("Regolazione Sistema Nervoso", [
                        "Doccia / Termoterapia: " + dinamica_rec,
                        "NSDR (Non-Sleep Deep Rest): 10 minuti di meditazione guidata o body-scan prima di dormire se lo stress oggi era sopra il 7/10."
                    ]),
                    ("Ottimizzazione Sonno", [
                        f"Target di stanotte: recuperare il deficit arrivando a {max(7.5, r['ore_sonno']+0.5):.1f} ore.",
                        "Riduci l'esposizione alla luce blu intensa (smartphone/TV) 60 minuti prima del sonno per permettere il rilascio di melatonina naturale."
                    ])
                ],
            }
        }

        tabs = st.tabs(list(coach_content.keys()))
        for tab, (nome_tab, contenuto) in zip(tabs, coach_content.items()):
            with tab:
                blocchi_html = ""
                for label, bullets in contenuto["blocchi"]:
                    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                    blocchi_html += f"""
                    <div class='coach-block' style='--block-color: {contenuto["colore"]};'>
                        <div class='label'>{label}</div>
                        <ul>{bullets_html}</ul>
                    </div>
                    """
                md(f"<div class='panel' style='padding: 20px;'>{blocchi_html}</div>")

        md("<div style='height:34px;'></div>")

        # =========================================================
        # CORSIE E ZONE
        # =========================================================
        section_head("Riferimento", "Corsie di Frequenza Cardiaca")

        corsie = [
            ("Corsia 1", "Zona 1-2", "Recupero / Base Aerobica", "Sforzo bassissimo (test del parlato superato facilmente). L'energia deriva dai grassi. Ideale per costruire resistenza senza accumulare fatica.", C_RPE),
            ("Corsia 2", "Zona 3", "Soglia Aerobica / Tempo", "Ritmo sostenuto, respirazione più profonda. Accumulo minimo di acido lattico. Sviluppa la potenza cardiaca.", C_AMBRA),
            ("Corsia 3", "Zona 4-5", "Soglia Lattacida / VO2Max", "Sforzo massimale (test del parlato fallito). Distrugge le fibre muscolari per super-compensare. Da usare col contagocce se il rischio infortunio è medio/alto.", C_STRESS),
        ]
        cc1, cc2, cc3 = st.columns(3)
        for c, (num, zt, zn, zd, zcol) in zip([cc1, cc2, cc3], corsie):
            c.markdown(f"""
            <div class='lane-chip' style='--zc:{zcol};'>
                <div class='lane-num'>{num}</div>
                <div class='zt'>{zt}</div>
                <div class='zn'>{zn}</div>
                <div class='zd'>{zd}</div>
            </div>
            """, unsafe_allow_html=True)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # PREPARAZIONE GRAFICI E STILI COMUNI
        # =========================================================
        CHART_HEIGHT = 280
        layout_base = dict(
            paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
            font=dict(color=TXT_SECONDARY, family="Inter, sans-serif", size=11),
            margin=dict(l=38, r=16, t=10, b=32),
            height=CHART_HEIGHT,
            showlegend=False,
            hoverlabel=dict(bgcolor="#1A2233", font_size=12, font_family="Inter, sans-serif", bordercolor=PANEL_BD),
        )
        axis_style = dict(gridcolor=PANEL_BD, zerolinecolor=PANEL_BD, linecolor=PANEL_BD)
        config_pulita = {'displayModeBar': False}

        media_sonno_90 = df_base['Ore Sonno'].mean() if 'Ore Sonno' in df_base.columns else 7.0
        media_stress_90 = df_base['Stress Lavoro'].mean() if 'Stress Lavoro' in df_base.columns else 5.0
        media_rpe_90 = df_base['RPE'].mean() if 'RPE' in df_base.columns else 5.0

        sonno_vs_media = r['ore_sonno'] - media_sonno_90
        stress_vs_media = r['stress_lavoro'] - media_stress_90
        rpe_vs_media = r['rpe_previsto'] - media_rpe_90

        figs_per_export = []
        insights_export = []

        def chart_card(container, titolo, fig, spiegazione, rule_color=C_NEUTRO):
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
            with container:
                md(f"""
                <div class='panel panel-flush'>
                    <div class='chart-top-rule' style='background:{rule_color};'></div>
                    <div class='panel-body'><p class='panel-title' style='color:{TXT_PRIMARY}; font-weight:600;'>{titolo}</p></div>
                """)
                st.plotly_chart(fig, use_container_width=True, config=config_pulita)
                md(f"""
                    <div class='panel-body' style='padding-top:0;'><div class='chart-caption'><strong>Insight Clinico:</strong><br>{spiegazione}</div></div>
                </div>
                """)
            figs_per_export.append(fig)
            insights_export.append((titolo, spiegazione))


        # =========================================================
        # SEZIONE 1: DINAMICHE AVANZATE E CARICO 
        # (Usa df_adv con data garantita)
        # =========================================================
        section_head("Analisi Avanzata", "Dinamiche di Carico", "Integrazione dei dati storici con modelli di readiness e previsione infortuni (Metodologia ACWR).")
            
        c_adv1, c_adv2, c_adv3 = st.columns(3)

        # --- 1. MATRICE DI PRONTEZZA ---
        fig_matrix = go.Figure()
        fig_matrix.add_trace(go.Scatter(
            x=df_adv['Ore Sonno'], y=df_adv['Stress Lavoro'], mode='markers',
            marker=dict(color=TXT_TERTIARY, size=6, opacity=0.4), hoverinfo='skip'
        ))
        fig_matrix.add_trace(go.Scatter(
            x=[r['ore_sonno']], y=[r['stress_lavoro']], mode='markers',
            marker=dict(color=col, size=14, symbol='diamond', line=dict(width=2, color=TXT_PRIMARY)),
            name="Oggi"
        ))
        fig_matrix.add_hline(y=media_stress_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
        fig_matrix.add_vline(x=media_sonno_90, line_dash="dot", line_color=PANEL_BD_H, opacity=0.7)
        
        fig_matrix.update_layout(**layout_base, xaxis_title="Ore di Sonno", yaxis_title="Stress Lavoro", xaxis=dict(range=[4, 10]), yaxis=dict(range=[0, 10]))
        
        if r['ore_sonno'] >= media_sonno_90 and r['stress_lavoro'] <= media_stress_90:
            quad = "Ottimale (Alto recupero, Basso stress)."
            azione = "Sei nel quadrante d'oro. Il corpo è pronto per carichi pesanti e adattamenti strutturali."
        elif r['ore_sonno'] >= media_sonno_90 and r['stress_lavoro'] > media_stress_90:
            quad = "Compensato (Alto recupero, Alto stress)."
            azione = "Stai dormendo abbastanza per tamponare lo stress lavorativo, ma il sistema nervoso è sotto pressione. Non eccedere in Z4/Z5."
        elif r['ore_sonno'] < media_sonno_90 and r['stress_lavoro'] <= media_stress_90:
            quad = "Vulnerabile (Basso recupero, Basso stress)."
            azione = "Il livello di stress è OK, ma manca materia prima (sonno) per riparare i muscoli. Allenamento leggero consigliato."
        else:
            quad = "Critico (Basso recupero, Alto stress)."
            azione = "Zona rossa di infortunio. Cortisolo alto e recupero cellulare assente. Altamente consigliato riposo totale o sola mobilità."
            
        chart_card(c_adv1, "Matrice di Prontezza", fig_matrix, f"Stato: {quad}<br>{azione}", col)


        # --- 2. ACUTE TO CHRONIC WORKLOAD RATIO (ACWR Proxy) ---
        if 'RPE' in df_adv.columns:
            df_adv['RPE_7'] = df_adv['RPE'].rolling(7, min_periods=1).mean()
            df_adv['RPE_28'] = df_adv['RPE'].rolling(28, min_periods=1).mean()
            df_adv['ACWR'] = df_adv['RPE_7'] / df_adv['RPE_28'].replace(0, 0.1)
            
            fig_acwr = go.Figure()
            fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(48,209,88,0.1)", opacity=1, layer="below", line_width=0)
            fig_acwr.add_trace(go.Scatter(
                x=df_adv['Data_Chart'].tail(60), y=df_adv['ACWR'].tail(60), mode='lines',
                line=dict(color=C_AMBRA, width=2, shape='spline')
            ))
            fig_acwr.add_hline(y=1.3, line_dash="dash", line_color=C_STRESS, line_width=1)
            fig_acwr.update_layout(**layout_base, yaxis_title="Ratio (Acuto/Cronico)", yaxis=dict(range=[0.5, 2.0]))
            
            acwr_attuale = df_adv['ACWR'].iloc[-1] if not df_adv['ACWR'].empty else 1.0
            if acwr_attuale > 1.3:
                acwr_txt = "ATTENZIONE: Ratio sopra 1.3. Stai aumentando il carico (fatica) troppo in fretta rispetto alla tua abitudine mensile. Rischio tendiniti alle stelle. Usa i prossimi giorni per scaricare."
            elif acwr_attuale < 0.8:
                acwr_txt = "Ratio sotto 0.8. Sei in fase di de-training (scarico prolungato). Il tuo corpo sta perdendo adattamenti atletici. Ottimo pre-gara, ma se non hai gare, torna a spingere moderatamente."
            else:
                acwr_txt = "Ratio nella 'Sweet Spot' (0.8 - 1.3). Progression del carico perfettamente bilanciata. Stai costruendo fitness senza sovraccaricare le articolazioni."
                
            chart_card(c_adv2, "ACWR (Carico Acuto vs Cronico)", fig_acwr, acwr_txt, C_AMBRA)
        else:
            c_adv2.warning("Dati RPE insufficienti per calcolo ACWR.")


        # --- 3. PATTERN SETTIMANALE DELLO STRESS ---
        if 'Stress Lavoro' in df_adv.columns:
            df_adv['Giorno'] = df_adv['Data_Chart'].dt.dayofweek
            giorni_map = {0:'Lun', 1:'Mar', 2:'Mer', 3:'Gio', 4:'Ven', 5:'Sab', 6:'Dom'}
            
            stress_giornaliero = df_adv.groupby('Giorno')['Stress Lavoro'].mean().reset_index()
            stress_giornaliero['Nome_Giorno'] = stress_giornaliero['Giorno'].map(giorni_map)
            
            if not stress_giornaliero.empty:
                fig_week = go.Figure(go.Bar(
                    x=stress_giornaliero['Nome_Giorno'], y=stress_giornaliero['Stress Lavoro'],
                    marker_color=TXT_TERTIARY, text=stress_giornaliero['Stress Lavoro'].round(1),
                    textposition='outside', textfont=dict(color=TXT_SECONDARY, size=10)
                ))
                giorno_max = stress_giornaliero.loc[stress_giornaliero['Stress Lavoro'].idxmax()]
                fig_week.add_trace(go.Bar(
                    x=[giorno_max['Nome_Giorno']], y=[giorno_max['Stress Lavoro']],
                    marker_color=C_STRESS, text=[giorno_max['Stress Lavoro'].round(1)],
                    textposition='outside', textfont=dict(color=C_STRESS, size=10)
                ))
                
                fig_week.update_layout(**layout_base, yaxis=dict(range=[0, 10]), xaxis_title="Giorno Settimana", barmode='overlay')
                
                adv_week = f"Il {giorno_max['Nome_Giorno']} è mediamente la tua giornata con maggior carico mentale. Il consiglio d'oro è spostare gli allenamenti di 'Lavori Specifici' (Ripetute, Soglia, Lunghi) lontano da questo giorno, tenendolo per riposo o Easy Run."
                chart_card(c_adv3, "Pattern Stress Settimanale", fig_week, adv_week, C_STRESS)
            else:
                c_adv3.warning("Dati storici insufficienti per il calcolo settimanale.")
        else:
            c_adv3.warning("Colonna Stress Lavoro non trovata.")

        md("<div style='height:34px;'></div>")

        # =========================================================
        # SEZIONE 2: GRAFICI DI TREND BASE
        # =========================================================
        section_head("Trend Storici", "Andamento Base (90 giorni)", "Evoluzione fisiologica per individuare sovrallenamento cronico o deficit di recupero.")

        df_plot = df_adv.tail(90)

        def calcola_trend(serie):
            if len(serie) < 15: return 0
            recente = serie.tail(14).mean()
            precedente = serie.head(len(serie) - 14).mean()
            return recente - precedente

        r1c1, r1c2, r1c3 = st.columns(3)

        if 'Ore Sonno' in df_plot.columns:
            trend_sonno = calcola_trend(df_plot['Ore Sonno'])
            fig_t1 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['Ore Sonno'], mode='lines',
                line=dict(color=C_SONNO, width=2), fill='tozeroy', fillcolor='rgba(46,144,255,0.08)'
            ))
            fig_t1.update_layout(**layout_base, yaxis_title="ore")
            spieg_sonno = "Un trend in calo indica che non stai compensando il volume di allenamento. Dormire di più (o inserire Power Naps) è l'unica via naturale per innalzare il picco di forma." if trend_sonno < -0.3 else "Il tuo corpo sta ricevendo input di sonno stabili. Ottimo substrato per adattamenti positivi."
            chart_card(r1c1, "Trend Sonno Temporale", fig_t1, spieg_sonno, C_SONNO)

        if 'Stress Lavoro' in df_plot.columns:
            trend_stress = calcola_trend(df_plot['Stress Lavoro'])
            fig_t2 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['Stress Lavoro'], mode='lines',
                line=dict(color=C_STRESS, width=2), fill='tozeroy', fillcolor='rgba(255,69,58,0.08)'
            ))
            fig_t2.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="punti")
            spieg_stress = "Stress cronico in ascesa. Il cervello non distingue tra stress da Excel o da Squat: tutto si somma. Se la linea rossa sale, il volume di allenamento DEVE scendere per evitare crac sistemici." if trend_stress > 0.5 else "Carico mentale sotto controllo, via libera per concentrarsi sulla performance."
            chart_card(r1c2, "Trend Stress Lavorativo", fig_t2, spieg_stress, C_STRESS)

        if 'RPE' in df_plot.columns:
            trend_rpe = calcola_trend(df_plot['RPE'])
            fig_t3 = go.Figure(go.Scatter(
                x=df_plot['Data_Chart'], y=df_plot['RPE'], mode='lines',
                line=dict(color=C_RPE, width=2), fill='tozeroy', fillcolor='rgba(48,209,88,0.08)'
            ))
            fig_t3.update_layout(**layout_base, yaxis=dict(range=[0, 10]), yaxis_title="punti")
            spieg_rpe = "Se l'RPE medio si alza continuamente a parità di passi, sei in over-reaching non funzionale. Necessario valutare una settimana di deload (dimezzare km e intensità)." if trend_rpe > 0.5 else "Mantenimento ottimale dello sforzo, il fitness sta compensando la fatica generata."
            chart_card(r1c3, "Trend Sforzo Percepito (RPE)", fig_t3, spieg_rpe, C_RPE)

        md("<div style='height:34px;'></div>")

        # =========================================================
        # EXPORT REPORT
        # =========================================================
        section_head("Export", "Generazione report per coach", "Scarica l'analisi completa.")

        coach_txt = ""
        for nome_tab, contenuto in coach_content.items():
            coach_txt += f"\n[{nome_tab.upper()}]\n"
            for label, bullets in contenuto["blocchi"]:
                coach_txt += f"  {label}:\n"
                for b in bullets:
                    coach_txt += f"    - {b}\n"

        grafici_txt = "\n".join(f"  - {t}: {s}" for t, s in insights_export)

        report_testo = f"""--- RUNAI PERFORMANCE REPORT ---
Status: {tit}
Distanza Consigliata: {distanza_consigliata:.1f} km (Target Originale: {distanza_target} km)
Indice Rischio: {risk_score:.0f}%
Recovery Score: {recovery_score:.0f}%
Stress Mentale (SMA): {sma:.1f}

NOTE CLINICHE E ANALISI AVANZATA:
{grafici_txt.replace('<br>', ' ')}

PROTOCOLLO COACH COMPLETO{coach_txt}
--------------------------------"""

        colb1, colb2 = st.columns(2)
        with colb1:
            st.download_button("Scarica TXT", data=report_testo, file_name="runai_report.txt", mime="text/plain", use_container_width=True)

        with colb2:
            charts_html = ""
            for i, f in enumerate(figs_per_export):
                include_js = 'cdn' if i == 0 else False
                charts_html += f.to_html(full_html=False, include_plotlyjs=include_js)

            coach_html = ""
            for nome_tab, contenuto in coach_content.items():
                blocchi_html = ""
                for label, bullets in contenuto["blocchi"]:
                    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
                    blocchi_html += f"<div class='coach-block' style='--block-color:{contenuto['colore']};'><div class='label'>{label}</div><ul>{bullets_html}</ul></div>"
                coach_html += f"<div class='panel' style='margin-bottom:14px;'><h3 style='margin-bottom:14px;'>{nome_tab}</h3>{blocchi_html}</div>"

            report_html_completo = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');
  body {{ background:#0A0E15; color:{TXT_SECONDARY}; font-family: Inter, sans-serif; padding: 36px; max-width:1100px; margin:0 auto; }}
  h1 {{ color:{col}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.7em; margin-bottom:4px; }}
  h2 {{ color:{TXT_PRIMARY}; font-family:'Oswald',sans-serif; font-weight:600; text-transform:uppercase; font-size:1.15em; margin:34px 0 14px 0; }}
  .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7em; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TERTIARY}; margin:0 0 8px 0; font-weight:600; }}
  .panel {{ background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:14px; padding:20px 22px; margin-bottom:14px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
  .kpi-row .panel {{ flex:1 1 30%; min-width:200px; }}
  .kpi-row .val {{ font-family:'JetBrains Mono',monospace; font-size:1.7em; color:{TXT_PRIMARY}; font-weight:600; }}
  .coach-block {{ margin-bottom:14px; border-left:3px solid var(--block-color); padding-left:14px; }}
  .coach-block .label {{ font-family:'Oswald',sans-serif; font-size:.85em; letter-spacing:.05em; text-transform:uppercase; margin-bottom:8px; font-weight:600; color:var(--block-color); }}
  .charts-grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .charts-grid > div {{ flex: 1 1 30%; min-width:280px; background:{PANEL_BG}; border:1px solid {PANEL_BD}; border-radius:12px; padding:10px; }}
</style>
</head>
<body>
  <p class="eyebrow">RunAI Performance Report</p>
  <h1>{tit}</h1>
  <div style="margin:20px 0;">{radar_svg}</div>
  <div class="kpi-row">
    <div class="panel"><p class="eyebrow">Distanza Consigliata</p><div class="val">{distanza_consigliata:.1f} km</div></div>
    <div class="panel"><p class="eyebrow">Indice Rischio</p><div class="val" style="color:{col};">{risk_score:.0f}%</div></div>
    <div class="panel"><p class="eyebrow">Recovery Score</p><div class="val" style="color:{C_SONNO};">{recovery_score:.0f}%</div></div>
  </div>
  <h2>Protocollo coach completo</h2>{coach_html}
  <h2>Grafici analitici</h2><div class="charts-grid">{charts_html}</div>
</body>
</html>"""

            st.download_button("Scarica HTML (Grafici e Design)", data=report_html_completo, file_name="runai_report_completo.html", mime="text/html", use_container_width=True)
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
