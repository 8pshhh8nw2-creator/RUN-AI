import streamlit as st
import numpy as np
import pandas as pd
from utils.data import genera_dati

def sidebar_comune():
    """
    Disegna la sidebar comune (logo, device, filtro temporale)
    e restituisce i dati filtrati da usare in ogni pagina.
    """
    if 'dati' not in st.session_state or st.session_state.dati is None:
        st.session_state.dati = genera_dati()

    with st.sidebar:
        st.markdown("""
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
                <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
                <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

        st.subheader("Dispositivo")
        device_scelto = st.selectbox(
            "Seleziona dispositivo:",
            ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3", "Fitbit Charge 6", "WHOOP 4.0", "Fascia Cardio Garmin"],
            label_visibility="collapsed"
        )

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
                st.session_state.device_info['nome'], 
                st.session_state.device_info['fc'], 
                st.session_state.device_info['battery'],
                st.session_state.device_info['steps'], 
                st.session_state.device_info['calories']
            ), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Filtri Temporali Storico")
        filtro_tempo = st.selectbox(
            "Intervallo Analisi:",
            ["Ultimi 30 giorni", "Ultimi 60 giorni", "Ultimi 90 giorni (Tutto)"],
            label_visibility="collapsed"
        )

    df_full = st.session_state.dati.copy()
    if filtro_tempo == "Ultimi 30 giorni":
        df = df_full.tail(30)
    elif filtro_tempo == "Ultimi 60 giorni":
        df = df_full.tail(60)
    else:
        df = df_full

    return df, df_full, filtro_tempo
