from pathlib import Path

import pandas as pd
import streamlit as st

# Metti il CSV nella stessa cartella di questo file (oppure cambia il percorso)
CSV_PATH = Path(__file__).parent / "corse_dataset_completo_kpi.csv"


@st.cache_data
def carica_dati(path=CSV_PATH):
    """Carica i dati reali e li restituisce con gli STESSI nomi di colonna
    che produceva genera_dati(), così il resto dell'app non va toccato."""
    raw = pd.read_csv(path, parse_dates=["data"])

    df = pd.DataFrame({
        "Giorno": raw["data"],
        "Distanza (km)": raw["distanza_km"],
        "Velocità (km/h)": raw["velocita_kmh"],
        "FC Media": raw["fc_media"],
        "FC Max": raw["fc_max"],
        "Temp (°C)": raw["gradi_celsius"],
        "RPE": raw["rpe"],
        "Ore Sonno": raw["ore_sonno"],
        "Stress Lavoro": raw["stress_mentale"],
        "Ore Lavoro": raw["ore_lavorate"],
        "Calorie": raw["calorie"],
        "SMA": raw["SMA"],
        # colonne extra disponibili nei dati reali (opzionali per l'app)
        "Dislivello (m)": raw["dislivello_m"],
        "Cadenza": raw["cad_media"],
        "Vento (km/h)": raw["velocita_vento_kmh"],
        "TRIMP": raw["trimp"],
        "sRPE Load": raw["sRPE_Load"],
    })

    # stessa regola di prima
    df["Rischio Infortunio"] = (
        (df["RPE"] > 7) & (df["Ore Sonno"] < 6.5) & (df["FC Media"] > 155)
    ).astype(int)

    return df.sort_values("Giorno").reset_index(drop=True)


# alias: il resto dell'app può continuare a chiamare genera_dati()
genera_dati = carica_dati
