import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def genera_dati():
    """
    Carica i dati reali da corse_con_note.csv e restituisce
    un DataFrame compatibile con le pagine della dashboard.
    """

    # Legge il CSV presente nella cartella principale del progetto.
    # Esempio struttura:
    # progetto/
    # ├── corse_con_note.csv
    # └── utils/
    #     └── data.py
    df = pd.read_csv("corse_con_note.csv")

    # Tiene solo le attività Running, se nel CSV fossero presenti altri sport.
    if "sport" in df.columns:
        df = df[
            df["sport"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("running")
        ].copy()

    # Rinomina le colonne del tuo CSV nei nomi usati dalla dashboard.
    df = df.rename(columns={
        "data": "Giorno",
        "distanza_km": "Distanza (km)",
        "durata_min": "Durata (min)",
        "passo_min_km": "Passo (min/km)",
        "velocita_kmh": "Velocità (km/h)",
        "fc_media": "FC Media",
        "fc_max": "FC Max",
        "calorie": "Calorie",
        "cad_media": "Cadenza Media",
        "dislivello_m": "Dislivello (m)",
        "rpe": "RPE"
    })

    # Converte correttamente la data.
    df["Giorno"] = pd.to_datetime(df["Giorno"], errors="coerce")

    # Converte tutte le colonne numeriche importanti.
    colonne_numeriche = [
        "Distanza (km)",
        "Durata (min)",
        "Passo (min/km)",
        "Velocità (km/h)",
        "FC Media",
        "FC Max",
        "Calorie",
        "Cadenza Media",
        "Dislivello (m)",
        "RPE"
    ]

    for colonna in colonne_numeriche:
        if colonna in df.columns:
            df[colonna] = pd.to_numeric(df[colonna], errors="coerce")

    # Elimina allenamenti senza data o distanza.
    # Nel tuo CSV esiste almeno una riga finale senza data: viene esclusa.
    df = df.dropna(subset=["Giorno", "Distanza (km)"]).copy()

    # Se la velocità dovesse mancare in qualche allenamento,
    # viene calcolata da distanza e durata.
    if "Velocità (km/h)" not in df.columns:
        df["Velocità (km/h)"] = np.nan

    df["Velocità (km/h)"] = df["Velocità (km/h)"].fillna(
        (df["Distanza (km)"] / df["Durata (min)"]) * 60
    )

    # Se il passo dovesse mancare, lo calcola in minuti per km.
    if "Passo (min/km)" not in df.columns:
        df["Passo (min/km)"] = np.nan

    df["Passo (min/km)"] = df["Passo (min/km)"].fillna(
        df["Durata (min)"] / df["Distanza (km)"]
    )

    # Il CSV non contiene questi dati di recupero.
    # Li creiamo vuoti, senza generare valori falsi.
    df["Ore Sonno"] = np.nan
    df["Stress Lavoro"] = np.nan
    df["Ore Lavoro"] = np.nan
    df["Temp (°C)"] = np.nan

    # SMA: non calcolabile senza sonno e stress reali.
    df["SMA"] = np.nan

    # Rischio Infortunio:
    # 0 = nessuna segnalazione nel dataset.
    # Questo NON significa che non esista rischio clinico:
    # indica soltanto che non hai ancora un dato/criterio registrato.
    df["Rischio Infortunio"] = 0

    # Ordina le corse nel tempo: fondamentale per cumulativa,
    # grafici settimanali e rolling windows.
    df = df.sort_values("Giorno").reset_index(drop=True)

    # Ordine più leggibile per eventuali tabelle.
    colonne_principali = [
        "Giorno",
        "Distanza (km)",
        "Durata (min)",
        "Passo (min/km)",
        "Velocità (km/h)",
        "FC Media",
        "FC Max",
        "RPE",
        "Calorie",
        "Cadenza Media",
        "Dislivello (m)",
        "Ore Sonno",
        "Stress Lavoro",
        "Ore Lavoro",
        "Temp (°C)",
        "SMA",
        "Rischio Infortunio"
    ]

    colonne_esistenti = [
        colonna for colonna in colonne_principali
        if colonna in df.columns
    ]

    colonne_extra = [
        colonna for colonna in df.columns
        if colonna not in colonne_esistenti
    ]

    return df[colonne_esistenti + colonne_extra]
