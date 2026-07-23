import streamlit as st

def carica_css():
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
