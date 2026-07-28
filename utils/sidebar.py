import streamlit as st
import numpy as np
import pandas as pd
from utils.data import genera_dati

def sidebar_comune():
    """
    Disegna la sidebar comune con il selettore temporale e il dispositivo in alto,
    e la navigazione delle pagine pulita in basso, senza emoji e con stile curato.
    """
    # Inizializzazione sicura di tutte le variabili di stato
    if 'dati' not in st.session_state or st.session_state.dati is None:
        st.session_state.dati = genera_dati()
    if 'analisi_fatta' not in st.session_state:
        st.session_state.analisi_fatta = False
    if 'risultati_analisi' not in st.session_state:
        st.session_state.risultati_analisi = {}
    if 'device_connected' not in st.session_state:
        st.session_state.device_connected = False

    with st.sidebar:
        # LOGO E HEADER PRINCIPALE
        st.markdown("""
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
                <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
                <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
            </div>
            <p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; font-family:"JetBrains Mono",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>
        """, unsafe_allow_html=True)

        # =========================================================
        # 1. SELEZIONE TEMPORALE (SOPRA)
        # =========================================================
        st.markdown("<p style='font-family:\"Oswald\",sans-serif; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:#F8F9FA; margin-bottom:6px; font-weight:600;'>Periodo di Analisi</p>", unsafe_allow_html=True)
        filtro_tempo = st.selectbox(
            "Intervallo Analisi:",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Ultimi 90 giorni (Tutto)"],
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # 2. DISPOSITIVO RUNAI (SOPRA)
        # =========================================================
        st.markdown("<p style='font-family:\"Oswald\",sans-serif; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:#F8F9FA; margin-bottom:6px; font-weight:600;'>Dispositivo RunAI</p>", unsafe_allow_html=True)
        device_scelto = st.selectbox(
            "Seleziona dispositivo:",
            ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3", "Fitbit Charge 6", "WHOOP 4.0", "Fascia Cardio Garmin"],
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

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
            st.markdown(f"""
            <div style='background-color: #0E1420; border: 1px solid #1c2333; border-radius: 10px; padding: 14px; font-family:"Inter",sans-serif; margin-top: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #1c2333; padding-bottom: 6px;'>
                    <span style='color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.75em; letter-spacing:0.1em;'>LIVE SYNC</span>
                    <span style='color: #566178; font-size: 0.7em; font-family:"JetBrains Mono",monospace;'>{info['sync_time']}</span>
                </div>
                <div style='color: #E8ECF2; font-family:"JetBrains Mono",monospace; font-size:0.88em;'>
                    <div style='display:flex; justify-content:space-between; margin: 5px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>FC</span><span style='font-weight:600;'>{info['fc']} bpm</span></div>
                    <div style='display:flex; justify-content:space-between; margin: 5px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Batteria</span><span style='font-weight:600; color:#00F5A0;'>{info['battery']}%</span></div>
                    <div style='display:flex; justify-content:space-between; margin: 5px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Passi</span><span style='font-weight:600;'>{info['steps']:,}</span></div>
                    <div style='display:flex; justify-content:space-between; margin: 5px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Calorie</span><span style='font-weight:600;'>{info['calories']} kcal</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # =========================================================
        # 3. NAVIGAZIONE PAGINE (SOTTO)
        # =========================================================
        st.markdown("<p style='font-family:\"Oswald\",sans-serif; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:#F8F9FA; margin-bottom:10px; font-weight:600;'>Navigazione Pagine</p>", unsafe_allow_html=True)
        
        st.page_link("app.py", label="Dashboard Principale")
        st.page_link("pages/01_analisi_stato_di_forma.py", label="Analisi Stato di Forma")
        st.page_link("pages/05_consiglio_finale.py", label="Consiglio Finale & Report")

    # Gestione filtraggio dataset in base alla scelta temporale
    df_full = st.session_state.dati.copy()
    if filtro_tempo == "Ultimi 30 giorni":
        df = df_full.tail(30)
    elif filtro_tempo == "Ultimi 60 giorni":
        df = df_full.tail(60)
    else:
        df = df_full

    return df, df_full, filtro_tempo
