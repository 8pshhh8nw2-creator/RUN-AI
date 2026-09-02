import streamlit as st
import pandas as pd
from datetime import timedelta

# =========================================================
#   CSS CONDIVISO (design system RUNAI)
# =========================================================
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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

    /* --- Bottone connetti --- */
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

    .runai-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%);
        border: 1px solid var(--line); border-radius: 12px; padding: 16px;
        animation: runai-fadein 0.3s var(--ease);
    }
    @keyframes runai-fadein {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .runai-label {
        color: var(--text-faint); font-size: 0.66em; font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.16em; text-transform: uppercase; margin: 0 0 10px 2px;
    }
    .runai-row { display: flex; justify-content: space-between; margin: 8px 0; font-family: "JetBrains Mono", monospace; font-size: 0.88em; }
    .runai-row span:first-child { color: var(--text-dim); font-family: "Inter", sans-serif; }
    .runai-row span:last-child { color: var(--text); font-weight: 600; }

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

     /* =========================================================
       NAV PAGINE — pulita, leggibile, stabile al click
    ========================================================= */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        margin-top: 24px;
        padding: 16px 0 0 0;
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
        font-family: "Inter", sans-serif;
        font-size: 0.95em;
        font-weight: 500;
        padding: 10px 12px;
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

    .runai-footer {
        order: 3;
        margin-top: auto;
        padding: 18px 8px 2px 8px;
        color: var(--text-faint);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.65em;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.7;
    }
