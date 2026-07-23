"""
utils/kpi_engine.py
--------------------
Motore di calcolo dei KPI proprietari di RunAI (SMA, ISLR, IITR, IDET),
descritti nel Capitolo 2 della tesi, + un risk_score ricostruito sui pesi
reali del Random Forest (Feature Importance) invece di pesi inventati.

NOMI COLONNE: i nomi qui sotto sono la mia migliore stima in base al resto
del codice che mi hai mostrato + alla tesi. Se il tuo dataframe usa nomi
diversi, aggiorna solo la sezione CONFIG qui sotto: il resto del file non
va toccato.
"""

import numpy as np
import pandas as pd

# ============================================================
# CONFIG — nomi colonne attese nel dataframe storico (df_base)
# ============================================================
COL_SONNO = "Ore Sonno"
COL_RPE = "RPE"
COL_STRESS = "Stress Lavoro"
COL_ORE_LAVORO = "Ore Lavoro"
COL_DISTANZA = "Distanza (km)"
COL_TEMPERATURA = "Temperatura"
COL_VENTO = "Vento (km/h)"
COL_FC_MEDIA = "FC Media"
COL_VELOCITA = "Velocita (km/h)"  # se hai solo il passo (min/km): velocita = 60 / passo

# Pesi di Feature Importance dal Random Forest della tesi (sommano a 100)
PESI_FEATURE_IMPORTANCE = {
    "ISLR": 31.5,
    "RMSSD": 22.8,
    "IDET": 14.6,
    "Ore Sonno": 12.8,
    "Volume Settimanale": 10.2,
    "Passo Medio": 8.1,
}

# Unica soglia realmente derivata dal tuo modello (punto di transizione P=50%
# nella regressione logistica ISLR -> Overload). Le altre soglie sotto sono
# stime ragionevoli da tarare quando avrai più dati: non sono nella tesi.
ISLR_SOGLIA_CRITICA = 6.3


# ============================================================
# LE 4 FORMULE (Capitolo 2)
# ============================================================
def calcola_sma(stress_giornata, rpe, ore_sonno):
    """SMA = Stress Giornata x RPE / Ore Sonno"""
    if ore_sonno <= 0:
        return 0.0
    return (stress_giornata * rpe) / ore_sonno


def calcola_islr(ore_lavoro, stress_mentale, distanza_km):
    """ISLR = Ore Lavoro x Stress Mentale / Distanza(km)"""
    if distanza_km <= 0:
        return 0.0
    return (ore_lavoro * stress_mentale) / distanza_km


def calcola_iitr(gradi_c, vento_kmh, distanza_km):
    """IITR = Gradi Celsius x Velocita Vento(km/h) / Distanza(km)"""
    if distanza_km <= 0:
        return 0.0
    return (gradi_c * vento_kmh) / distanza_km


def calcola_idet(fc_media, gradi_c, velocita_kmh):
    """IDET = FC media x Gradi Celsius / Velocita(km/h)"""
    if velocita_kmh <= 0:
        return 0.0
    return (fc_media * gradi_c) / velocita_kmh


def calcola_kpi_giornalieri(row):
    """
    Calcola i 4 KPI per una riga del dataframe storico o per il record di oggi.
    Gestisce in modo sicuro sia dizionari che serie pandas, con fallback intelligenti.
    """
    # Estrazione sicura con fallback sui nomi alternativi
    stress = row.get(COL_STRESS, row.get("stress_lavoro", row.get("Stress Lavoro", 5)))
    rpe = row.get(COL_RPE, row.get("rpe_previsto", row.get("RPE", 5)))
    sonno = row.get(COL_SONNO, row.get("ore_sonno", row.get("Ore Sonno", 7.5)))
    
    ore_lavoro = row.get(COL_ORE_LAVORO, row.get("ore_lavoro", row.get("Ore Lavoro", 8.0)))
    distanza = row.get(COL_DISTANZA, row.get("distanza_oggi", row.get("Distanza (km)", 10.0)))
    
    temp = row.get(COL_TEMPERATURA, row.get("temperatura", 20.0))
    vento = row.get(COL_VENTO, row.get("vento", 5.0))
    
    fc_media = row.get(COL_FC_MEDIA, row.get("fc_riposo", row.get("FC Media", 140.0)))
    velocita = row.get(COL_VELOCITA, row.get("velocita", 10.0))

    sma = calcola_sma(stress, rpe, sonno)
    islr = calcola_islr(ore_lavoro, stress, distanza)
    iitr = calcola_iitr(temp, vento, distanza)
    idet = calcola_idet(fc_media, temp, velocita) if velocita and velocita > 0 else np.nan

    return {"SMA": sma, "ISLR": islr, "IITR": iitr, "IDET": idet}
# ============================================================
# NORMALIZZAZIONE "N-of-1": ogni valore diventa uno score 0-100
# basato sulla sua posizione percentile nel TUO storico personale,
# non su soglie assolute inventate — coerente con l'approccio
# single-subject case study della tesi.
# ============================================================
def normalizza_a_rischio(valore_oggi, serie_storica, invert=False):
    """
    invert=True quando un valore storicamente ALTO significa MENO rischio
    (es. Ore Sonno, RMSSD). Default: valore ALTO = PIU rischio (es. ISLR, IDET).
    Se lo storico ha meno di 5 punti validi, ritorna 50 (neutro) perche'
    non c'e' ancora abbastanza dato per un confronto affidabile.
    """
    serie_storica = pd.Series(serie_storica).dropna()
    if len(serie_storica) < 5:
        return 50.0
    percentile = (serie_storica < valore_oggi).mean() * 100
    return round(100 - percentile, 1) if invert else round(percentile, 1)


def calcola_risk_score_pesato(oggi: dict, storico: pd.DataFrame, rmssd_oggi=None):
    """
    Ricostruisce il risk_score usando i pesi REALI del Random Forest della
    tesi, al posto dei pesi arbitrari (40/35/30/20) della Pagina 3 originale
    -- che tra l'altro non includeva affatto l'ISLR, il KPI piu predittivo
    (31.5% di Feature Importance).

    oggi: dict con i valori di oggi, es:
        {"ISLR": ..., "IDET": ..., "Ore Sonno": ..., "Volume Settimanale": ...,
         "Passo Medio": ...}
    storico: dataframe con le stesse colonne calcolate storicamente
             (serve per il confronto percentile).
    rmssd_oggi: opzionale, se non hai ancora un sensore HRV nell'app il suo
                peso viene redistribuito proporzionalmente sugli altri KPI.
    """
    pesi = PESI_FEATURE_IMPORTANCE.copy()
    if rmssd_oggi is None:
        peso_rmssd = pesi.pop("RMSSD")
        totale_restante = sum(pesi.values())
        pesi = {k: v + (v / totale_restante) * peso_rmssd for k, v in pesi.items()}

    scores = {
        "ISLR": normalizza_a_rischio(oggi["ISLR"], storico.get("ISLR", []), invert=False),
        "IDET": normalizza_a_rischio(oggi["IDET"], storico.get("IDET", []), invert=False),
        "Ore Sonno": normalizza_a_rischio(oggi["Ore Sonno"], storico.get("Ore Sonno", []), invert=True),
        "Volume Settimanale": normalizza_a_rischio(oggi["Volume Settimanale"], storico.get("Volume Settimanale", []), invert=False),
        "Passo Medio": normalizza_a_rischio(oggi["Passo Medio"], storico.get("Passo Medio", []), invert=False),
    }
    if rmssd_oggi is not None:
        scores["RMSSD"] = normalizza_a_rischio(rmssd_oggi, storico.get("RMSSD", []), invert=True)

    score_finale = sum(scores[k] * (pesi[k] / 100) for k in scores)
    return round(min(100, max(0, score_finale)), 1), scores


# ============================================================
# Dati statici della Feature Importance (dal tuo Random Forest,
# Capitolo 2) — da usare per il grafico "Cosa pesa davvero" in app
# ============================================================
FEATURE_IMPORTANCE_CHART_DATA = [
    ("ISLR (Sforzo Lavorativo Residuo)", 31.5),
    ("RMSSD (Recupero Parasimpatico)", 22.8),
    ("IDET (Stress Termico)", 14.6),
    ("Ore di Sonno Notturno", 12.8),
    ("Volume Settimanale (km)", 10.2),
    ("Passo Medio di Corsa", 8.1),
]
