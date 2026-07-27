"""
Advanced Machine Learning Suite - Dashboard interattiva per tesi magistrale.
File unico. Avvio: streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
APP_TITLE = "Advanced Machine Learning Suite"
APP_SUBTITLE = (
    "Framework interattivo per la stima della performance, la classificazione del rischio "
    "di overload, l'analisi dei driver del sovraccarico e la scoperta di profili latenti "
    "di allenamento."
)
APP_ICON = "🧠"
APP_CAPTION = "Core analitico della tesi magistrale · Sport Data Science"

COLORS = {
    "bg": "#060b14", "bg2": "#0a1424", "surface": "#0f1b2d", "surface_2": "#13243b",
    "border": "#1f3252", "border_soft": "rgba(148,163,184,0.16)",
    "text": "#f8fafc", "text_soft": "#cbd5e1", "muted": "#8fa3bd",
    "blue": "#38bdf8", "cyan": "#22d3ee", "green": "#34d399", "amber": "#fbbf24",
    "red": "#f87171", "purple": "#a78bfa", "pink": "#f472b6",
}
QUALITATIVE = [COLORS['blue'], COLORS['purple'], COLORS['cyan'], COLORS['amber'], COLORS['pink'], COLORS['green']]
CLASS_COLORS = {"Nessun rischio": COLORS['blue'], "Rischio overload": COLORS['red']}
SPLIT_COLORS = {"Train": COLORS['blue'], "Test": COLORS['amber']}
SEQUENTIAL = ["#0e3a52", "#0e7490", "#22d3ee", "#a5f3fc"]
CLUSTER_COLORS = {"Rigenerativo": COLORS['green'], "Qualità / Misto": COLORS['amber'], "Elevato Stress": COLORS['red']}
CLUSTER_LABELS = ["Rigenerativo", "Qualità / Misto", "Elevato Stress"]

TARGET = "Rischio Overload"
TIME_TARGET = "Tempo (min)"
RF_FEATURES = ["Distanza (km)", "Ore Sonno", "SMA", "ISLR", "IDET", "IITR"]
CLUSTER_FEATURES = ["FC Media", "ISLR"]
LR_FEATURES = ["Distanza (km)"]
LOG_FEATURES = ["ISLR"]

GLOSSARY = {
    "SMA": "Stress Metabolico Apparente - (Stress lavoro × RPE) / Ore di sonno.",
    "ISLR": "Indice di Stress Lavoro-Relativo - (Ore lavoro × Stress) / Distanza.",
    "IDET": "Indice di Domanda Emodinamico-Termica - (FC media × Temperatura) / Velocità.",
    "IITR": "Indice di Interferenza Termo-Ventosa - (Temperatura × Vento) / Distanza.",
    "RPE": "Rating of Perceived Exertion: percezione soggettiva dello sforzo (1-10).",
    TARGET: "Etichetta binaria: 1 se la sessione ricade in area di sovraccarico.",
}

RISK_BANDS = ((40.0, "Basso", COLORS['green']), (70.0, "Moderato", COLORS['amber']), (101.0, "Alto", COLORS['red']))


def risk_band(prob: float) -> tuple[str, str]:
    for thr, label, color in RISK_BANDS:
        if prob < thr:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


@dataclass(frozen=True)
class Settings:
    n_sessions: int = 320
    seed: int = 42
    test_size: float = 0.25
    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 4
    n_clusters: int = 3
    cv_folds: int = 5
    risk_prevalence: float = 0.30
    label_noise: float = 0.55
    cluster_grid: tuple = field(default=(2, 3, 4, 5, 6))

    @property
    def cache_key(self) -> str:
        return f"{self.n_sessions}|{self.seed}|{self.test_size}|{self.n_estimators}|{self.max_depth}|{self.min_samples_leaf}|{self.n_clusters}|{self.cv_folds}|{self.risk_prevalence}|{self.label_noise}"


# ============================================================================
# THEME
# ============================================================================
PLOTLY_TEMPLATE = "ml_suite"


def register_plotly_template():
    if PLOTLY_TEMPLATE in pio.templates:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        colorway=QUALITATIVE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS['text_soft'], size=13),
        title=dict(font=dict(size=17, color=COLORS['text']), x=0.01, xanchor="left", y=0.96),
        margin=dict(t=64, l=16, r=16, b=16),
        hoverlabel=dict(bgcolor=COLORS['surface_2'], bordercolor=COLORS['border'], font=dict(family="Inter", color=COLORS['text'], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)", font=dict(size=12), title_text=""),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=COLORS['border_soft'], ticks="outside", tickcolor=COLORS['border_soft'], title=dict(font=dict(size=12, color=COLORS['muted']))),
        yaxis=dict(gridcolor="rgba(148,163,184,0.10)", zeroline=False, linecolor="rgba(0,0,0,0)", title=dict(font=dict(size=12, color=COLORS['muted']))),
        colorscale=dict(sequential=[[0, "#0e3a52"], [0.5, "#0e7490"], [1, "#a5f3fc"]]),
    )
    pio.templates[PLOTLY_TEMPLATE] = tpl
    pio.templates.default = PLOTLY_TEMPLATE


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    :root {{
        --bg:{COLORS['bg']}; --bg2:{COLORS['bg2']}; --surface:{COLORS['surface']};
        --surface2:{COLORS['surface_2']}; --border:{COLORS['border']};
        --border-soft:{COLORS['border_soft']}; --text:{COLORS['text']};
        --text-soft:{COLORS['text_soft']}; --muted:{COLORS['muted']};
        --blue:{COLORS['blue']}; --cyan:{COLORS['cyan']}; --amber:{COLORS['amber']};
        --purple:{COLORS['purple']}; --radius-lg:24px; --radius-md:16px; --radius-sm:12px;
        --shadow-lg:0 24px 60px rgba(2,6,23,0.45); --shadow-md:0 10px 28px rgba(2,6,23,0.28);
    }}
    html,body,[class*="css"],button,input,textarea,select {{
        font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
        font-feature-settings:'cv02','cv03','ss01';
    }}
    .stApp {{
        background:
            radial-gradient(900px 520px at 8% -6%,rgba(56,189,248,0.13),transparent 70%),
            radial-gradient(760px 480px at 96% 0%,rgba(167,139,250,0.13),transparent 70%),
            radial-gradient(1100px 700px at 50% 118%,rgba(34,211,238,0.07),transparent 70%),
            linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);
        background-attachment:fixed; color:var(--text);
    }}
    #MainMenu,footer,header [data-testid="stToolbar"] {{ visibility:hidden; }}
    .block-container {{ max-width:1500px; padding-top:2.2rem; padding-bottom:4rem; }}
    h1,h2,h3,h4 {{ letter-spacing:-0.02em; color:var(--text); }}
    a {{ color:var(--cyan); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    .hero {{
        position:relative; overflow:hidden;
        background:linear-gradient(135deg,rgba(15,27,45,0.97) 0%,rgba(19,36,59,0.92) 100%);
        border:1px solid var(--border); border-radius:var(--radius-lg);
        padding:2.3rem 2.4rem; margin-bottom:1.4rem; box-shadow:var(--shadow-lg);
    }}
    .hero::after {{
        content:""; position:absolute; inset:0 0 auto 0; height:2px;
        background:linear-gradient(90deg,transparent,var(--cyan),var(--purple),transparent); opacity:0.75;
    }}
    .hero-eyebrow {{
        display:inline-flex; align-items:center; gap:0.5rem; font-size:0.72rem; font-weight:700;
        letter-spacing:0.18em; text-transform:uppercase; color:var(--cyan);
        background:rgba(34,211,238,0.10); border:1px solid rgba(34,211,238,0.22);
        border-radius:999px; padding:0.35rem 0.8rem; margin-bottom:1.1rem;
    }}
    .hero-title {{
        margin:0; font-size:clamp(2rem,3.4vw,2.9rem); line-height:1.04;
        letter-spacing:-0.045em; font-weight:800;
        background:linear-gradient(120deg,#ffffff 0%,#cfe9ff 55%,#b8c8ff 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }}
    .hero-subtitle {{ margin-top:0.9rem; color:var(--muted); font-size:1.02rem; line-height:1.75; max-width:74ch; }}
    .pill-row {{ display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.9rem; }}
    .pill {{
        font-size:0.78rem; font-weight:600; color:var(--text-soft);
        background:rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.18);
        border-radius:999px; padding:0.3rem 0.75rem;
    }}
    .callout {{ border-radius:var(--radius-md); padding:1rem 1.15rem; margin:0.55rem 0 1rem 0; font-size:0.95rem; line-height:1.7; }}
    .callout b {{ color:#ffffff; }}
    .callout--simple {{ background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.20); border-left:4px solid var(--blue); color:#dbeafe; }}
    .callout--theory {{ background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.20); border-left:4px solid var(--amber); color:#fef3c7; }}
    .callout--insight {{ background:rgba(167,139,250,0.09); border:1px solid rgba(167,139,250,0.20); border-left:4px solid var(--purple); color:#ede9fe; }}
    .callout--neutral {{ background:rgba(148,163,184,0.07); border:1px solid var(--border-soft); border-left:4px solid var(--muted); color:var(--text-soft); }}
    .section-head {{
        display:flex; flex-direction:column; gap:0.3rem; padding:1.35rem 1.5rem;
        margin:0.4rem 0 1.1rem 0;
        background:linear-gradient(180deg,rgba(15,27,45,0.88),rgba(15,27,45,0.55));
        border:1px solid var(--border); border-radius:var(--radius-md); box-shadow:var(--shadow-md);
    }}
    .section-kicker {{ font-size:0.7rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--cyan); }}
    .section-title {{ font-size:1.45rem; font-weight:800; line-hei
