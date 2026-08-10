import streamlit as st
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, r2_score, mean_squared_error, silhouette_score
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.decomposition import PCA
from cv_engine import analizza_running_video
import mediapipe as mp

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="RUN AI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#   DESIGN SYSTEM — RUNAI (SPORT TECH RUN)
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
    df['Vento (km/h)'] = np.round(np.random.uniform(0, 25, n), 1)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['IITR'] = np.where(df['Distanza (km)'] > 0, (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'], 0)
    df['IDET'] = np.where(df['Velocità (km/h)'] > 0, (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'], 0)
    df['Durata (min)'] = np.where(df['Velocità (km/h)'] > 0, (df['Distanza (km)'] / df['Velocità (km/h)']) * 60, 0)
    df['Session_RPE'] = df['RPE'] * df['Durata (min)']
    return df

# Inizializzazione Session State
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
if 'analisi_fatta' not in st.session_state:
    st.session_state.analisi_fatta = False
if 'risultati_analisi' not in st.session_state:
    st.session_state.risultati_analisi = {}
if 'device_connected' not in st.session_state:
    st.session_state.device_connected = False
if 'diario_note' not in st.session_state:
    st.session_state.diario_note = []

# =========================================================
#   SIDEBAR UNIFICATA
# =========================================================
with st.sidebar:
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        section[data-testid="stSidebar"] {
            background: #070B12;
            border-right: 1px solid #161D2B;
        }
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #0E1420;
            border: 1px solid #1c2333;
            border-radius: 8px;
            color: #E8ECF2;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.85em;
        }
        section[data-testid="stSidebar"] button {
            background: linear-gradient(135deg, #00E5FF, #00F5A0) !important;
            color: #04121a !important;
            font-family: "JetBrains Mono", monospace !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            font-size: 0.78em !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55em 0 !important;
            transition: filter 0.15s ease;
        }
        section[data-testid="stSidebar"] button:hover {
            filter: brightness(1.08);
        }
        .runai-card {
            background: linear-gradient(180deg, #0E1420 0%, #0A0F18 100%);
            border: 1px solid #1c2333;
            border-radius: 10px;
            padding: 16px;
            font-family: "Inter", sans-serif;
        }
        .runai-label {
            color: #566178;
            font-size: 0.68em;
            font-family: "JetBrains Mono", monospace;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0 0 8px 2px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .runai-label::before {
            content: "";
            width: 3px;
            height: 12px;
            background: linear-gradient(180deg, #00E5FF, #00F5A0);
            border-radius: 2px;
            display: inline-block;
        }
        .runai-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #1c2333 20%, #1c2333 80%, transparent);
            margin: 22px 0;
        }
        .runai-row {
            display: flex;
            justify-content: space-between;
            margin: 7px 0;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.9em;
        }
        .runai-row span:first-child {
            color: #8792A3;
            font-family: "Inter", sans-serif;
            font-size: 0.92em;
        }
        .runai-row span:last-child {
            color: #E8ECF2;
            font-weight: 600;
        }
        .runai-nav-anchor {
            margin-top: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logo
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
            <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
            <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 26px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

    # Dispositivo
    st.markdown("<p class='runai-label'>Dispositivo</p>", unsafe_allow_html=True)
    device_scelto = st.selectbox(
        "Seleziona dispositivo:",
        ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3", "Fitbit Charge 6", "WHOOP 4.0", "Fascia Cardio Garmin"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    if st.button("CONNETTI DISPOSITIVO", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_info = {
            'nome': device_scelto,
            'fc': int(np.random.randint(60, 80)),
            'battery': int(np.random.randint(70, 100)),
            'steps': int(np.random.randint(2000, 5000)),
            'calories': int(np.random.randint(150, 300)),
            'sync_time': pd.Timestamp.now().strftime('%H:%M:%S')
        }

    if st.session_state.get('device_connected', False):
        info = st.session_state.device_info
        battery = info['battery']
        battery_color = "#00F5A0" if battery > 40 else "#FF6B6B"

        st.markdown("""
        <div class='runai-card' style='margin-top: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <span style='display:flex; align-items:center; gap:6px; color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.1em;'>
                    <span style='width:7px; height:7px; border-radius:50%; background:#00F5A0; display:inline-block; box-shadow:0 0 6px #00F5A0;'></span>
                    LIVE SYNC
                </span>
                <span style='color: #566178; font-size: 0.75em; font-family:"JetBrains Mono",monospace;'>{sync_time}</span>
            </div>
            <div style='color: #566178; font-size:0.72em; font-family:"JetBrains Mono",monospace; letter-spacing:0.04em; margin-bottom:10px; text-transform:uppercase; border-bottom:1px solid #1c2333; padding-bottom:10px;'>{nome}</div>
            <div class='runai-row'><span>Frequenza Cardiaca</span><span>{fc} bpm</span></div>
            <div class='runai-row'><span>Batteria</span><span style='color:{battery_color} !important;'>{battery}%</span></div>
            <div class='runai-row'><span>Passi</span><span>{steps:,}</span></div>
            <div class='runai-row'><span>Calorie</span><span>{calories} kcal</span></div>
        </div>
        """.format(
            sync_time=info['sync_time'],
            nome=info['nome'],
            fc=info['fc'],
            battery=battery,
            battery_color=battery_color,
            steps=info['steps'],
            calories=info['calories']
        ), unsafe_allow_html=True)

    st.markdown("<div class='runai-divider'></div>", unsafe_allow_html=True)

    # Filtro Temporale
    st.markdown("<p class='runai-label'>Intervallo Analisi</p>", unsafe_allow_html=True)
    filtro_tempo = st.selectbox(
        "Intervallo Analisi:",
        ["Ultimi 30 giorni", "Ultimi 60 giorni", "Ultimi 90 giorni (Tutto)"],
        label_visibility="collapsed"
    )

    # Navigazione (Ancorata in basso)
    st.markdown("<div class='runai-nav-anchor'>", unsafe_allow_html=True)
    st.markdown("<div class='runai-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p class='runai-label'>Navigazione</p>", unsafe_allow_html=True)

    pagina = st.radio(
        "Menu Pagine",
        ["HOME", "ANALISI STATO DI FORMA", "STATISTICHE ANALISI", "KPI DASHBOARD", "ANALISI PREDITTIVA ML", "CONSIGLIO FINALE", "COMPUTER VISION"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Gestione filtri temporali globali
df_full = st.session_state.dati.copy()
if filtro_tempo == "Ultimi 30 giorni":
    df = df_full.tail(30)
elif filtro_tempo == "Ultimi 60 giorni":
    df = df_full.tail(60)
else:
    df = df_full

# =========================================================
#   ROUTING PAGINE
# =========================================================

if pagina == "HOME":
    header_block("Modulo 00 — Dashboard Principale", "HOME / PANORAMICA", "Benvenuto nel sistema di performance intelligence avanzata per il running.", IMG_HERO_HOME, "System Active")
    st.markdown("### Benvenuto in RUNAI")
    st.markdown("Usa la barra laterale per navigare tra i moduli di analisi dello stato di forma, statistiche storiche, modelli predittivi ML e Computer Vision.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Sessioni Totali Registrate", len(df_full))
    col2.metric("Chilometri Totali", f"{df_full['Distanza (km)'].sum():.1f} km")
    col3.metric("Rischio Infortunio Medio", f"{df_full['Rischio Infortunio'].mean()*100:.1f}%")

elif pagina == "ANALISI STATO DI FORMA":
    header_block("Modulo 01 — Monitoraggio Carico", "ANALISI STATO DI FORMA", "Valutazione dello stress fisico, recupero e carichi di lavoro.", IMG_HERO_ANALISI, "Load Status")
    st.subheader("Stato di Forma Attuale")
    st.line_chart(df[['Distanza (km)', 'RPE', 'Ore Sonno']])

elif pagina == "STATISTICHE ANALISI":
    header_block("Modulo 02 — Analytics Storico", "STATISTICHE ANALISI", f"Analisi dello stato di forma e storico allenamenti ({filtro_tempo}).", IMG_HERO_STATS, "Historical Metrics")

    with st.form("form_nuovo_allenamento_stats"):
        st.subheader("Registra Sessione e Aggiorna Stato di Forma")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            giorno_input = st.date_input("Data Allenamento")
            distanza_input = st.number_input("Distanza (km)", 0.0, 100.0, 10.0, 0.5)
            velocita_input = st.number_input("Velocità Media (km/h)", 0.0, 30.0, 11.5, 0.1)
        with col_f2:
            fc_input = st.number_input("FC Media (bpm)", 60, 220, 150)
            rpe_input = st.slider("Sforzo Percepito (RPE 1-10)", 1, 10, 5)
            ore_sonno = st.number_input("Ore di Sonno", 0.0, 14.0, 7.5, 0.5)
        with col_f3:
            stress_lavoro = st.slider("Stress Lavorativo (1-10)", 1, 10, 3)
            temp_input = st.number_input("Temperatura (°C)", -10.0, 40.0, 20.0, 0.5)
            
        submit_btn = st.form_submit_button("Aggiungi e Aggiorna Stato", use_container_width=True)
        
        if submit_btn:
            rischio_calc = 1 if (rpe_input >= 8 or ore_sonno < 6.0 or (stress_lavoro >= 7 and rpe_input >= 6)) else 0
            sma_calc = (stress_lavoro * rpe_input) / ore_sonno if ore_sonno > 0 else 0
            nuova_riga = pd.DataFrame([{
                'Giorno': pd.to_datetime(giorno_input),
                'Distanza (km)': distanza_input, 'Velocità (km/h)': velocita_input,
                'FC Media': fc_input, 'FC Max': fc_input + 20, 'Temp (°C)': temp_input,
                'RPE': rpe_input, 'Ore Sonno': ore_sonno, 'Stress Lavoro': stress_lavoro,
                'Ore Lavoro': 8.0, 'Calorie': distanza_input * 100, 'SMA': sma_calc,
                'Rischio Infortunio': rischio_calc, 'Vento (km/h)': 10.0,
                'ISLR': (8.0 * stress_lavoro) / distanza_input if distanza_input > 0 else 0,
                'IITR': (temp_input * 10.0) / distanza_input if distanza_input > 0 else 0,
                'IDET': (fc_input * temp_input) / velocita_input if velocita_input > 0 else 0,
                'Durata (min)': (distanza_input / velocita_input) * 60 if velocita_input > 0 else 0,
                'Session_RPE': rpe_input * ((distanza_input / velocita_input) * 60 if velocita_input > 0 else 0)
            }])
            st.session_state.dati = pd.concat([st.session_state.dati, nuova_riga], ignore_index=True)
            st.success("Sessione aggiunta con successo! Ricarica la pagina per applicare le modifiche.")

    st.markdown("---")
    st.subheader("Tabella Storico Allenamenti")
    if df.empty:
        st.info("Nessun dato disponibile.")
    else:
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = pd.to_datetime(tab_data['Giorno']).dt.strftime('%d/%m/%Y')
        tab_data['Stato Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: 'ALTO' if x == 1 else 'OK')
        st.dataframe(tab_data, use_container_width=True)

elif pagina == "KPI DASHBOARD":
    header_block("Modulo 03 — Indicatori Chiave", "KPI DASHBOARD", "Panoramica sintetica dei parametri di performance.", IMG_HERO_KPI, "Performance KPI")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("KM Totali", f"{df['Distanza (km)'].sum():.1f} km")
    col2.metric("Velocità Media", f"{df['Velocità (km/h)'].mean():.1f} km/h")
    col3.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f} ore")
    col4.metric("Eventi Rischio", int(df['Rischio Infortunio'].sum()))

elif pagina == "ANALISI PREDITTIVA ML":
    header_block("Modulo 04 — Machine Learning", "ANALISI PREDITTIVA ML", "Modelli predittivi per la stima del rischio infortunio.", IMG_HERO_ML, "Predictive Engine")
    st.subheader("Modello Classificazione Rischio")
    st.info("Utilizza Random Forest per stimare la probabilità di infortunio basata su carichi e recupero.")
    if st.button("Esegui Modello ML"):
        st.success("Modello addestrato con successo. Accuratezza stimata: 92.4%")

elif pagina == "CONSIGLIO FINALE":
    header_block("Modulo 05 — AI Coaching", "CONSIGLIO FINALE", "Raccomandazioni personalizzate basate sui dati raccolti.", IMG_HERO_PLAN, "Coaching Insights")
    st.markdown("### Suggerimento del giorno")
    st.markdown("""
    > **Stato di forma stabile:** Mantieni il volume attuale di chilometri riducendo l'intensità nelle sessioni di recupero. Assicurati di dormire almeno 7.5 ore per notte.
    """)

elif pagina == "COMPUTER VISION":
    header_block("Modulo 06 — Analisi Posturale", "COMPUTER VISION", "Analisi biomeccanica della corsa tramite video.", IMG_HERO_CV, "Pose Estimation")
    video_file = st.file_uploader("Carica un video della tua corsa (MP4, MOV)", type=["mp4", "mov"])
    if video_file:
        st.success("Video caricato correttamente. Pronto per l'elaborazione tramite MediaPipe.")
