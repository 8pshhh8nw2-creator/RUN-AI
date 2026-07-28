import streamlit as st
import numpy as np
import pandas as pd
from utils.data import genera_dati

def sidebar_comune():
    """
    Disegna la sidebar comune (logo, device, filtro temporale) in alto
    e la navigazione delle pagine in basso.
    """
    if 'dati' not in st.session_state or st.session_state.dati is None:
        st.session_state.dati = genera_dati()
    if 'analisi_fatta' not in st.session_state:
        st.session_state.analisi_fatta = False
    if 'risultati_analisi' not in st.session_state:
        st.session_state.risultati_analisi = {}
    if 'device_connected' not in st.session_state:
        st.session_state.device_connected = False

    with st.sidebar:
        # ---------- STILE GLOBALE SIDEBAR ----------
        st.markdown("""
            <style>
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
            </style>
        """, unsafe_allow_html=True)

        # ---------- HEADER / LOGO (in cima) ----------
        st.markdown("""
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
                <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
                <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 26px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

        # ---------- DISPOSITIVO (subito sotto il logo) ----------
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

        # ---------- FILTRO TEMPORALE (sotto Dispositivo) ----------
        st.markdown("<p class='runai-label'>Intervallo Analisi</p>", unsafe_allow_html=True)
        filtro_tempo = st.selectbox(
            "Intervallo Analisi:",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Ultimi 90 giorni (Tutto)"],
            label_visibility="collapsed"
        )

        # ---------- NAVIGAZIONE PAGINE (IN FONDO) ----------
        # Nota: Usiamo i link diretti corrispondenti ai file reali della cartella
        st.markdown("<div class='runai-divider'></div>", unsafe_allow_html=True)
        st.markdown("<p class='runai-label'>Navigazione</p>", unsafe_allow_html=True)
        
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_Stato_Forma.py", label="Stato Forma")
        st.page_link("pages/2_Statistiche_Analisi.py", label="Statistiche Analisi")
        st.page_link("pages/3_KPI_Dashboard.py", label="KPI Dashboard")
        st.page_link("pages/4_Analisi_Predittiva_ML.py", label="Analisi Predittiva ML")
        st.page_link("pages/5_Consiglio_Finale.py", label="Consiglio Finale")
        st.page_link("pages/6_ComputerVision.py", label="Computer Vision")

    df_full = st.session_state.dati.copy()
    if filtro_tempo == "Ultimi 30 giorni":
        df = df_full.tail(30)
    elif filtro_tempo == "Ultimi 60 giorni":
        df = df_full.tail(60)
    else:
        df = df_full

    return df, df_full, filtro_tempo
