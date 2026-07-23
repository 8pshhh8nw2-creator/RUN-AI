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
