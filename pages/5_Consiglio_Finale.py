import html
import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import carica_css
from utils.data import genera_dati
from utils.sidebar import sidebar_comune
from utils.kpi_engine import calcola_kpi_giornalieri
from utils.ml_engine import (
    analisi_completa_oggi,
    punteggio_rischio_combinato,
    PESI_RISCHIO,
)

st.set_page_config(
    page_title="Consiglio Finale",
    layout="wide",
)

carica_css()
logger = logging.getLogger(__name__)

COLORI = {
    "basso": "#30D158",
    "medio": "#FFB020",
    "alto": "#FF453A",
}

POLITICHE = {
    "basso": {
        "titolo": "ATTENZIONE BASSA",
        "fattore_distanza": 1.0,
        "messaggio": (
            "L'indice non segnala una forte attenzione secondo le "
            "regole dell'app. Non è una garanzia di sicurezza: valuta "
            "anche sensazioni, sintomi e indicazioni del tuo coach."
        ),
        "attivita": (
            "Puoi valutare di mantenere il programma previsto, "
            "senza aggiungere automaticamente intensità o chilometri."
        ),
    },
    "medio": {
        "titolo": "VALUTA UNA SEDUTA PIÙ LEGGERA",
        "fattore_distanza": 0.6,
        "messaggio": (
            "I segnali suggeriscono maggiore prudenza. La riduzione "
            "proposta è una regola dimostrativa dell'app, non una "
            "prescrizione individuale validata."
        ),
        "attivita": (
            "Valuta una corsa facile e più breve, oppure recupero, "
            "in base alle tue sensazioni."
        ),
    },
    "alto": {
        "titolo": "VALUTA IL RECUPERO",
        "fattore_distanza": 0.0,
        "messaggio": (
            "L'indice supera la soglia di attenzione alta dell'app. "
            "La pagina propone di non correre, senza formulare "
            "diagnosi o imporre un riposo clinico."
        ),
        "attivita": (
            "Valuta riposo o attività leggera soltanto se compatibile "
            "con le tue condizioni e con le indicazioni ricevute."
        ),
    },
}

NOMI_COMPONENTI = {
    "euristica": "Regole euristiche",
    "random_forest": "Random Forest",
    "logistic": "Regressione logistica",
    "cluster": "Profilo storico",
    "trend_fisiologico": "Scostamento FC storico",
}


def numero(valore, minimo=None, massimo=None):
    """Converte un valore numerico valido; non inventa dati mancanti."""
    if valore is None or isinstance(valore, (bool, np.bool_)):
        return None

    try:
        risultato = float(valore)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(risultato):
        return None
    if minimo is not None and risultato < minimo:
        return None
    if massimo is not None and risultato > massimo:
        return None

    return risultato


def estrai(record, nomi, minimo=None, massimo=None):
    for nome in nomi:
        if nome in record:
            risultato = numero(record[nome], minimo, massimo)
            if risultato is not None:
                return risultato
    return None


def testo_numero(valore, decimali=1):
    if valore is None:
        return "N/D"
    return f"{valore:.{decimali}f}"


def livello_da_score(score):
    if score < 25:
        return "basso"
    if score < 60:
        return "medio"
    return "alto"


def punteggio_euristico(sonno, stress, rpe):
    penalita_sonno = 40 if sonno < 6 else 25 if sonno < 6.5 else 10
    penalita_stress = 35 if stress >= 8 else 20 if stress >= 6 else 5
    penalita_rpe = 30 if rpe >= 8 else 15 if rpe >= 6 else 5

    interazione = (
        20
        if sonno < 6.5 and stress >= 7 and rpe >= 7
        else 0
    )

    return float(min(
        100,
        penalita_sonno + penalita_stress + penalita_rpe + interazione,
    ))


def prepara_storico(df):
    risultato = df.copy()

    colonna_data = next(
        (
            c for c in risultato.columns
            if str(c).strip().casefold()
            in {"giorno", "data", "date", "timestamp", "time"}
        ),
        None,
    )

    if colonna_data is not None:
        risultato["_data"] = pd.to_datetime(
            risultato[colonna_data],
            errors="coerce",
            utc=True,
        )
        risultato = risultato.sort_values(
            "_data",
            kind="stable",
            na_position="last",
        )

    return risultato


def calcola_indicatori(record):
    sonno = estrai(record, ["ore_sonno", "Ore Sonno"], 0, 24)
    stress = estrai(record, ["stress_lavoro", "Stress Lavoro"], 0, 10)
    rpe = estrai(record, ["rpe_previsto", "RPE"], 0, 10)
    distanza = estrai(record, ["distanza_oggi", "Distanza (km)"], 0)
    lavoro = estrai(record, ["ore_lavoro", "Ore Lavoro"], 0, 24)

    temperatura = estrai(
        record,
        ["temperatura", "Temp (°C)", "Temperatura"],
    )
    vento = estrai(record, ["vento", "Vento (km/h)"], 0)

    # Non usare fc_riposo come sostituto della FC media della corsa.
    fc = estrai(record, ["fc_media", "FC Media"], 1)
    velocita = estrai(
        record,
        ["velocita", "Velocità (km/h)", "Velocita (km/h)"],
        0,
    )

    risultati = {
        "SMA": None,
        "ISLR": None,
        "IITR": None,
        "IDET": None,
    }

    # Fornisce tutti i valori al modulo per evitare i suoi fallback.
    # I KPI con input mancanti vengono poi lasciati esplicitamente N/D.
    dati_kpi = {
        "Ore Sonno": sonno if sonno is not None else 0,
        "Stress Lavoro": stress if stress is not None else 0,
        "RPE": rpe if rpe is not None else 0,
        "Ore Lavoro": lavoro if lavoro is not None else 0,
        "Distanza (km)": distanza if distanza is not None else 0,
        "Temperatura": temperatura if temperatura is not None else 0,
        "Vento (km/h)": vento if vento is not None else 0,
        "FC Media": fc if fc is not None else 0,
        "Velocita (km/h)": velocita if velocita is not None else 0,
    }

    requisiti = {
        "SMA": (
            sonno is not None and sonno > 0
            and stress is not None and rpe is not None
        ),
        "ISLR": (
            distanza is not None and distanza > 0
            and lavoro is not None and stress is not None
        ),
        "IITR": (
            distanza is not None and distanza > 0
            and temperatura is not None and vento is not None
        ),
        "IDET": (
            velocita is not None and velocita > 0
            and fc is not None and temperatura is not None
        ),
    }

    try:
        calcolati = calcola_kpi_giornalieri(dati_kpi)
        for nome, disponibili in requisiti.items():
            if disponibili:
                risultati[nome] = numero(calcolati.get(nome))
    except Exception:
        logger.exception("Errore nel calcolo dei KPI")
        st.warning(
            "Non è stato possibile calcolare i KPI. "
            "Gli indicatori vengono mostrati come N/D."
        )

    return risultati


# ------------------------------------------------------------------
# Stato e dati
# ------------------------------------------------------------------
if st.session_state.get("dati") is None:
    st.session_state.dati = genera_dati()

st.session_state.setdefault("analisi_fatta", False)
st.session_state.setdefault("risultati_analisi", {})

sidebar_result = sidebar_comune()

if (
    isinstance(sidebar_result, tuple)
    and len(sidebar_result) == 3
):
    _, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.get("dati")
    filtro_tempo = st.session_state.get("filtro_tempo", "Tutto")

if not isinstance(df_full, pd.DataFrame):
    st.error("I dati storici non sono un DataFrame valido.")
    st.stop()

df_base = prepara_storico(df_full)

st.title("Consiglio finale")
st.caption("Sintesi giornaliera · Indicatori sperimentali RunAI")

st.info(
    "Questa pagina offre supporto informativo, non diagnosi o "
    "autorizzazioni mediche. Nel dataset attuale, la classe chiamata "
    "'Rischio Infortunio' è generata da una regola su RPE, sonno e "
    "frequenza cardiaca: non rappresenta un infortunio osservato."
)

if not st.session_state.get("analisi_fatta", False):
    st.warning(
        "Completa prima il questionario nella pagina "
        "'Analisi stato di forma'."
    )
    st.stop()

record = st.session_state.get("risultati_analisi")

if not isinstance(record, dict) or not record:
    st.error("Il questionario non contiene risultati validi.")
    st.stop()

sonno = estrai(record, ["ore_sonno", "Ore Sonno"], 0, 24)
stress = estrai(record, ["stress_lavoro", "Stress Lavoro"], 0, 10)
rpe = estrai(record, ["rpe_previsto", "RPE"], 0, 10)
distanza_target = estrai(
    record,
    ["distanza_oggi", "Distanza (km)"],
    0,
)

input_oggi = {
    "Ore di sonno": sonno,
    "Stress": stress,
    "RPE previsto": rpe,
    "Distanza prevista": distanza_target,
}

mancanti = [
    nome for nome, valore in input_oggi.items()
    if valore is None
]

if mancanti:
    st.error("Dati mancanti o non validi: " + ", ".join(mancanti))
    st.stop()

euristica = punteggio_euristico(sonno, stress, rpe)
kpi = calcola_indicatori(record)

# ------------------------------------------------------------------
# ML: mantiene la compatibilità con il motore attuale.
# In caso di errore usa soltanto l'euristica, dichiarandolo.
# ------------------------------------------------------------------
analisi_ml = None
componenti = {"euristica": euristica}
score = euristica
modalita = "Solo regole euristiche"
errore_ml = None

colonne_ml = [
    "Distanza (km)",
    "Ore Sonno",
    "Stress Lavoro",
    "FC Media",
    "RPE",
    "Velocità (km/h)",
    "Temp (°C)",
    "Rischio Infortunio",
]

try:
    assenti = [c for c in colonne_ml if c not in df_base.columns]
    if assenti:
        raise ValueError("Colonne mancanti: " + ", ".join(assenti))

    if len(df_base) < 10:
        raise ValueError("Storico insufficiente per il motore attuale.")

    dati_numerici = df_base[colonne_ml].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.isfinite(dati_numerici.to_numpy(dtype=float)).all():
        raise ValueError("Lo storico contiene valori mancanti o non numerici.")

    target = dati_numerici["Rischio Infortunio"]
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("La classe euristica deve contenere solo 0 e 1.")

    conteggi = target.value_counts()
    if len(conteggi) != 2 or conteggi.min() < 2:
        raise ValueError(
            "Servono entrambe le classi con almeno due osservazioni ciascuna."
        )

    df_ml = df_base.copy()
    df_ml[colonne_ml] = dati_numerici

    analisi_ml = analisi_completa_oggi(
        df_ml,
        distanza=distanza_target,
        ore_sonno=sonno,
        stress=stress,
        rpe=rpe,
    )

    segnali_ml = {}
    for nome in PESI_RISCHIO:
        if nome == "euristica":
            continue

        valore = numero(
            analisi_ml["componenti"].get(nome),
            0,
            100,
        )

        if valore is None:
            raise ValueError(f"Segnale ML non valido: {nome}")

        segnali_ml[nome] = valore

    pesi = [numero(v, 0) for v in PESI_RISCHIO.values()]
    if any(v is None for v in pesi):
        raise ValueError("Pesi del punteggio non validi.")
    if not math.isclose(sum(pesi), 1.0, abs_tol=1e-6):
        raise ValueError("I pesi del punteggio devono sommare a 1.")

    score_combinato = numero(
        punteggio_rischio_combinato(segnali_ml, euristica),
        0,
        100,
    )
    if score_combinato is None:
        raise ValueError("Punteggio combinato non valido.")

    score = score_combinato
    componenti.update(segnali_ml)
    modalita = "Indice combinato sperimentale"

except Exception as exc:
    logger.exception("Analisi ML non disponibile")
    errore_ml = str(exc)
    analisi_ml = None

if errore_ml:
    st.warning(
        "Analisi ML non disponibile: viene mostrato soltanto "
        "il punteggio euristico."
    )
    with st.expander("Dettaglio tecnico"):
        st.code(errore_ml, language="text")

# ------------------------------------------------------------------
# Un'unica politica per titolo, colore, distanza e attività.
# ------------------------------------------------------------------
livello = livello_da_score(score)
politica = POLITICHE[livello]
colore = COLORI[livello]

distanza_proposta = distanza_target * politica["fattore_distanza"]
riduzione = (
    (1 - politica["fattore_distanza"]) * 100
    if distanza_target > 0 else None
)

st.subheader(politica["titolo"])
st.write(politica["messaggio"])
st.caption(f"Modalità di calcolo: {modalita}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Indice di attenzione", f"{score:.0f}/100")
c2.metric("Distanza proposta", f"{distanza_proposta:.1f} km")
c3.metric("Sonno dichiarato", f"{sonno:.1f} h")
c4.metric("RPE previsto", f"{rpe:.1f}/10")

if riduzione is not None:
    st.caption(
        f"Piano inserito: {distanza_target:.1f} km · "
        f"Riduzione proposta: {riduzione:.0f}%."
    )
else:
    st.caption("Il piano inserito non prevede chilometri di corsa.")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    number={"suffix": "/100"},
    title={"text": "Indice di attenzione"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": colore},
        "steps": [
            {"range": [0, 25], "color": "rgba(48,209,88,0.15)"},
            {"range": [25, 60], "color": "rgba(255,176,32,0.15)"},
            {"range": [60, 100], "color": "rgba(255,69,58,0.15)"},
        ],
    },
))
gauge.update_layout(
    height=280,
    margin={"l": 35, "r": 35, "t": 50, "b": 20},
    paper_bgcolor="rgba(0,0,0,0)",
    font={"color": "#E8ECF2"},
)
st.plotly_chart(gauge, use_container_width=True)

st.caption(
    "Soglie convenzionali dell'app: basso <25; medio da 25 a <60; "
    "alto ≥60. Il punteggio non è una probabilità di infortunio."
)

# ------------------------------------------------------------------
# KPI: niente Recovery Score artificiale e niente valori inventati.
# ------------------------------------------------------------------
st.header("Indicatori di oggi")

tabella_kpi = pd.DataFrame([
    {
        "Indicatore": "SMA",
        "Valore": testo_numero(kpi["SMA"]),
        "Formula": "(Stress × RPE) / Ore sonno",
    },
    {
        "Indicatore": "ISLR",
        "Valore": testo_numero(kpi["ISLR"]),
        "Formula": "(Ore lavoro × Stress) / Distanza",
    },
    {
        "Indicatore": "IITR",
        "Valore": testo_numero(kpi["IITR"]),
        "Formula": "(Temperatura × Vento) / Distanza",
    },
    {
        "Indicatore": "IDET",
        "Valore": testo_numero(kpi["IDET"]),
        "Formula": "(FC media corsa × Temperatura) / Velocità",
    },
])
st.dataframe(tabella_kpi, hide_index=True, use_container_width=True)
st.caption(
    "Indicatori sperimentali descrittivi. N/D indica input mancanti "
    "o un denominatore non valido. La FC a riposo non sostituisce "
    "la FC media della corsa. Nessuno di questi valori identifica "
    "da solo la causa di affaticamento o di variazioni del battito."
)

# ------------------------------------------------------------------
# Consigli: niente prescrizioni numeriche non documentate.
# ------------------------------------------------------------------
st.header("Indicazioni operative")

consigli = {
    "Prima": [
        politica["attivita"],
        "Non trattare il punteggio come un'autorizzazione a correre.",
        "Se inizi la seduta, parti gradualmente e valuta le sensazioni.",
    ],
    "Durante": [
        (
            "Se scegli di rallentare, aumenta i minuti o i secondi "
            "per chilometro: diminuirli significa correre più veloce."
        ),
        "Non forzare un obiettivo di cadenza imposto dalla dashboard.",
        (
            "Non ignorare dolore, malessere o sintomi nuovi perché "
            "l'indice è basso."
        ),
    ],
    "Dopo": [
        "Registra durata, distanza e RPE effettivi della seduta.",
        "Annota eventuali sintomi e differenze rispetto al piano.",
        (
            "Usa queste informazioni per discutere gli adattamenti "
            "del programma con il tuo coach o professionista."
        ),
    ],
}

tabs = st.tabs(list(consigli))
for tab, (_, punti) in zip(tabs, consigli.items()):
    with tab:
        for punto in punti:
            st.write(f"- {punto}")

# ------------------------------------------------------------------
# Componenti: nessun voto di maggioranza o affidabilità artificiale.
# ------------------------------------------------------------------
st.header("Da dove nasce l'indice")

righe_componenti = []
for nome, valore in componenti.items():
    peso = PESI_RISCHIO[nome] if analisi_ml is not None else 1.0
    righe_componenti.append({
        "Segnale": NOMI_COMPONENTI.get(nome, nome),
        "Valore": valore,
        "Peso": peso,
        "Contributo": valore * peso,
        "Fascia": livello_da_score(valore),
    })

tabella_componenti = pd.DataFrame(righe_componenti)
st.dataframe(
    tabella_componenti,
    hide_index=True,
    use_container_width=True,
)

fig_componenti = go.Figure(go.Bar(
    x=tabella_componenti["Contributo"],
    y=tabella_componenti["Segnale"],
    orientation="h",
    marker_color=[
        COLORI[fascia] for fascia in tabella_componenti["Fascia"]
    ],
    text=[
        f"{v:.1f} pt" for v in tabella_componenti["Contributo"]
    ],
    textposition="auto",
))
fig_componenti.update_layout(
    title="Contributi all'indice",
    xaxis_title="Punti",
    height=330,
    margin={"l": 160, "r": 30, "t": 50, "b": 40},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#E8ECF2"},
)
st.plotly_chart(fig_componenti, use_container_width=True)

st.caption(
    "I segnali non sono necessariamente indipendenti. "
    "La loro concordanza non dimostra che il risultato sia corretto. "
    "L'arrotondamento può produrre una piccola differenza tra "
    "la somma dei contributi e l'indice visualizzato."
)

if analisi_ml is not None:
    with st.expander("Metriche e limiti del motore attuale"):
        metriche = []
        for etichetta, chiave in [
            ("AUC Random Forest", "auc_rf"),
            ("AUC regressione logistica", "auc_log"),
            ("R² regressione FC", "r2_regressione"),
            ("Silhouette clustering", "silhouette_cluster"),
        ]:
            metriche.append({
                "Metrica": etichetta,
                "Valore": testo_numero(
                    numero(analisi_ml.get(chiave)),
                    3,
                ),
            })

        st.dataframe(pd.DataFrame(metriche), hide_index=True)
        st.write(
            "Queste metriche provengono dal motore attuale. "
            "La classificazione è valutata con split casuali e "
            "scaling eseguito prima della validazione. "
            "Non costituiscono una validazione prospettica "
            "del rischio di infortunio."
        )
        st.write(
            "Per lo scenario odierno il motore usa una FC proxy "
            "pari a 100 + RPE × 10, non una misura reale. "
            "Lo scostamento recente della FC usa attualmente le "
            "ultime 14 osservazioni, non 14 giorni, e residui "
            "calcolati sul dataset di addestramento."
        )

# ------------------------------------------------------------------
# Storico: filtro esplicito riferito all'ultima registrazione.
# Il motore usa invece lo storico completo.
# ------------------------------------------------------------------
st.header("Storico descrittivo")
st.caption(
    "Il filtro temporale si applica ai grafici sottostanti. "
    "Il motore ML usa lo storico completo."
)

grafici_export = [
    ("Indice di attenzione", gauge),
    ("Contributi all'indice", fig_componenti),
]

if "_data" not in df_base.columns:
    st.warning("Colonna data non trovata: grafici temporali non disponibili.")
else:
    df_vis = df_base.dropna(subset=["_data"]).copy()
    date_non_valide = int(df_base["_data"].isna().sum())

    if date_non_valide:
        st.warning(
            f"{date_non_valide} registrazioni senza data valida "
            "sono escluse dai grafici temporali."
        )

    if df_vis.empty:
        st.warning("Non ci sono registrazioni con date valide.")
    else:
        ultima_data = df_vis["_data"].max()

        finestre = {
            "Ultimi 30 giorni": 30,
            "Ultimi 60 giorni": 60,
        }

        giorni = finestre.get(filtro_tempo)
        if giorni is not None:
            soglia = ultima_data - pd.Timedelta(days=giorni)
            df_vis = df_vis[df_vis["_data"] > soglia]

        st.caption(
            f"Intervallo: {filtro_tempo} · riferimento: ultima "
            f"registrazione del {ultima_data.strftime('%d/%m/%Y')} · "
            f"{len(df_vis)} registrazioni. Più sedute nello stesso "
            "giorno restano osservazioni distinte."
        )

        for colonna, titolo, unita in [
            ("Ore Sonno", "Sonno registrato", "Ore"),
            ("Stress Lavoro", "Stress registrato", "Scala 0–10"),
            ("RPE", "Sforzo percepito registrato", "Scala 0–10"),
        ]:
            if colonna not in df_vis.columns:
                continue

            y = pd.to_numeric(df_vis[colonna], errors="coerce")
            y = y.where(np.isfinite(y))
            validi = y.notna()

            if not validi.any():
                continue

            fig = go.Figure(go.Scatter(
                x=df_vis.loc[validi, "_data"],
                y=y.loc[validi],
                mode="lines+markers",
                name=colonna,
            ))
            fig.update_layout(
                title=titolo,
                yaxis_title=unita,
                height=280,
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E8ECF2"},
            )
            if colonna in {"Stress Lavoro", "RPE"}:
                fig.update_yaxes(range=[0, 10])

            st.plotly_chart(fig, use_container_width=True)
            grafici_export.append((titolo, fig))

# ------------------------------------------------------------------
# Export TXT e HTML: valori e grafici della pagina.
# ------------------------------------------------------------------
st.header("Esporta il report")

generato = datetime.now().astimezone().isoformat(timespec="seconds")

limiti_report = (
    "Report informativo, non diagnostico. Indice sperimentale 0–100, "
    "non probabilità di infortunio. Target storico generato da regole "
    "su RPE, sonno e FC. Soglie e riduzione della distanza sono "
    "convenzioni dimostrative dell'app."
)

if analisi_ml is not None:
    limiti_report += (
        " Il motore attuale usa una FC proxy, validazione casuale "
        "con scaling globale e residui storici in-sample. "
        "Richiede correzioni nel modulo ml_engine.py."
    )
else:
    limiti_report += " ML non disponibile: risultato solo euristico."

righe_txt = [
    "RUNAI — CONSIGLIO FINALE",
    f"Generato: {generato}",
    f"Esito: {politica['titolo']}",
    f"Modalità: {modalita}",
    f"Indice di attenzione: {score:.0f}/100",
    f"Distanza prevista: {distanza_target:.1f} km",
    f"Distanza proposta: {distanza_proposta:.1f} km",
    f"Sonno dichiarato: {sonno:.1f} h",
    f"Stress: {stress:.1f}/10",
    f"RPE previsto: {rpe:.1f}/10",
    "",
    politica["messaggio"],
    "",
    "INDICATORI SPERIMENTALI",
]

for nome, valore in kpi.items():
    righe_txt.append(f"{nome}: {testo_numero(valore)}")

righe_txt.extend(["", "COMPONENTI"])
for riga in righe_componenti:
    righe_txt.append(
        f"{riga['Segnale']}: {riga['Valore']:.1f}/100; "
        f"peso {riga['Peso']:.2f}; "
        f"contributo {riga['Contributo']:.1f} punti"
    )

for fase, punti in consigli.items():
    righe_txt.extend(["", fase.upper()])
    righe_txt.extend(f"- {punto}" for punto in punti)

righe_txt.extend([
    "",
    f"Filtro grafici: {filtro_tempo}, riferito all'ultima registrazione.",
    "Calcolo ML: storico completo.",
    "",
    "LIMITI",
    limiti_report,
])

report_txt = "\n".join(righe_txt)

blocchi_grafici = []
for i, (titolo, fig) in enumerate(grafici_export):
    grafico_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True if i == 0 else False,
        config={"displayModeBar": False, "responsive": True},
    )
    blocchi_grafici.append(
        f"<section><h2>{html.escape(titolo)}</h2>"
        f"{grafico_html}</section>"
    )

report_html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RunAI — Consiglio finale</title>
<style>
body {{
    margin: 0 auto;
    padding: 24px;
    max-width: 1100px;
    background: #080B12;
    color: #E8ECF2;
    font-family: Arial, sans-serif;
    line-height: 1.6;
}}
h1 {{ color: {colore}; }}
section {{
    margin: 20px 0;
    padding: 16px;
    background: #0E1420;
    border: 1px solid #263044;
    border-radius: 12px;
}}
pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: inherit;
}}
</style>
</head>
<body>
<h1>{html.escape(politica["titolo"])}</h1>
<section><pre>{html.escape(report_txt)}</pre></section>
{''.join(blocchi_grafici)}
</body>
</html>
"""

b1, b2 = st.columns(2)

with b1:
    st.download_button(
        "Scarica TXT",
        data=report_txt,
        file_name="runai_consiglio_finale.txt",
        mime="text/plain; charset=utf-8",
        use_container_width=True,
    )

with b2:
    st.download_button(
        "Scarica HTML con grafici",
        data=report_html,
        file_name="runai_consiglio_finale.html",
        mime="text/html; charset=utf-8",
        use_container_width=True,
    )
