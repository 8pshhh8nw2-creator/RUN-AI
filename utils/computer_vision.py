"""
utils/computer_vision.py
------------------------
Analisi cinematica 2D della corsa, ripresa laterale, per RUNAI.

Estrazione dello scheletro con MediaPipe Pose legacy e analisi dei cicli
di un singolo piede.

MODIFICHE DI QUESTA VERSIONE
    - Nessun fallback automatico al modello heavy (complexity=2).
    - Soglia minima di risalita per TO ridotta da 2 * tol a tol.
    - Diagnostica per ogni ciclo: IC, TO, soglia richiesta e risalita disponibile.
    - Controllo di coordinate finite sia per x sia per y.
    - Selezione del lato basata sulla qualità dei landmark core.
    - Frame rappresentativo scelto tra i veri contatti rilevati.
    - Rilascio del video anche in caso di errore.

LIMITI
    IC e TO sono stime ottenute da soglie cinematiche, non misure dirette
    del contatto al terreno. La modifica della soglia TO va verificata
    confrontando gli eventi rilevati con i frame del video.

    Gli angoli sono proiezioni sul piano immagine.
    La scala in cm deriva dall'altezza dichiarata e da proporzioni
    antropometriche medie: non è una calibrazione metrica del video.

    Il modulo non stima rischio di infortunio o carichi articolari.

CONVENZIONI
    Le coordinate immagine hanno y crescente verso il basso.
    "stride" indica un ciclo completo dello stesso piede.
    Le chiavi "passi_validi" e "passi_scartati" sono mantenute per
    compatibilità con la pagina esistente, ma contano cicli di un piede.

RIFERIMENTI DA VERIFICARE PRIMA DELL'USO SCIENTIFICO
    - Proporzioni antropometriche coscia/gamba.
    - Soglie di classificazione del foot strike angle.
    - Definizioni e accuratezza degli eventi cinematici IC/TO.
"""

import math
from collections import Counter

import numpy as np


# ----------------------------------------------------------------------------
# Costanti
# ----------------------------------------------------------------------------

LATI = ("sx", "dx")
PARTI = (
    "spalla",
    "anca",
    "ginocchio",
    "caviglia",
    "tallone",
    "punta",
)
CORE = ("anca", "ginocchio", "caviglia", "tallone", "punta")

FRAZ_COSCIA = 0.245
FRAZ_GAMBA = 0.246

VIS_MIN = 0.30
MIN_FRAZ_VALIDA = 0.40
MAX_GAP_S = 0.40
FILTRO_S = 0.10

DIST_MIN_PASSO_S = 0.40
TOL_SUOLO = 0.03
FRAZ_AMPIEZZA_IC = 0.20
FRAZ_AMPIEZZA_TO = 0.35

# Prima era equivalente a 2.0: ora la risalita minima è tol, non 2 * tol.
# Modifica sperimentale da verificare sui frame del video.
MOLTIPLICATORE_TOL_TO = 1.0

SOGLIA_VERSO = 0.05

CONTATTO_MIN_S = 0.10
CONTATTO_MAX_S = 0.60
STRIDE_MIN_S = 0.45
STRIDE_MAX_S = 1.60

FSA_TALLONE = 8.0
FSA_AVAMPIEDE = -1.6

# Stampa la diagnostica nei log del processo Streamlit.
DEBUG_EVENTI = True

ETICHETTE_APPOGGIO = {
    "tallone": "Appoggio di Tallone (Rearfoot Strike)",
    "mesopiede": "Appoggio di Mesopiede (Midfoot Strike)",
    "avampiede": "Appoggio di Avampiede (Forefoot Strike)",
}


# ----------------------------------------------------------------------------
# Utilità numeriche
# ----------------------------------------------------------------------------

def _savgol(y, finestra, ordine=3):
    """Filtro Savitzky-Golay implementato con NumPy."""
    y = np.asarray(y, dtype=float)

    finestra = max(int(finestra), ordine + 2)
    if finestra % 2 == 0:
        finestra += 1

    if len(y) < finestra:
        return y.copy()

    m = finestra // 2
    x = np.arange(-m, m + 1, dtype=float)
    a = np.vander(x, ordine + 1, increasing=True)
    coef = np.linalg.pinv(a)[0]

    ypad = np.pad(y, m, mode="edge")
    return np.convolve(ypad, coef[::-1], mode="valid")


def _massimi(y, dist_min, soglia):
    """Massimi locali sopra soglia, mantenendo prima quelli più alti."""
    cand = [
        i
        for i in range(1, len(y) - 1)
        if (
            y[i] >= y[i - 1]
            and y[i] > y[i + 1]
            and y[i] >= soglia
        )
    ]

    cand.sort(key=lambda i: y[i], reverse=True)
    accettati = []

    for i in cand:
        if all(abs(i - j) >= dist_min for j in accettati):
            accettati.append(i)

    return sorted(accettati)


def _buchi_lunghi(ok, max_gap):
    """Segnala buchi lunghi e code esterne ai dati validi."""
    ok = np.asarray(ok, dtype=bool)
    n = len(ok)
    flag = np.zeros(n, dtype=bool)

    validi = np.flatnonzero(ok)
    if len(validi) == 0:
        return np.ones(n, dtype=bool)

    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue

        j = i
        while j < n and not ok[j]:
            j += 1

        if j - i > max_gap:
            flag[i:j] = True

        i = j

    flag[:validi[0]] = True
    flag[validi[-1] + 1:] = True
    return flag


def _periodo_autocorr(y, fps):
    """Stima del periodo del ciclo dall'autocorrelazione."""
    s = np.asarray(y, dtype=float)
    s = s - np.mean(s)
    n = len(s)

    lo = max(1, int(round(STRIDE_MIN_S * fps)))
    hi = min(int(round(STRIDE_MAX_S * fps)), n // 2)

    if hi <= lo + 1:
        return None

    ac = np.correlate(s, s, mode="full")[n - 1:]

    if ac[0] <= 0 or not np.isfinite(ac[0]):
        return None

    ac = ac / ac[0]
    k = lo + int(np.argmax(ac[lo:hi + 1]))

    return k / fps if ac[k] > 0.1 else None


def _angolo_incluso(ax, ay, bx, by, cx, cy):
    """Angolo in gradi al vertice b tra b->a e b->c."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    den = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    den = np.where(den == 0, np.nan, den)

    cosang = np.clip(
        (v1x * v2x + v1y * v2y) / den,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosang))


def _stat(valori, cifre=1):
    v = np.asarray(valori, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return {"media": None, "sd": None, "n": 0}

    return {
        "media": round(float(v.mean()), cifre),
        "sd": (
            round(float(v.std(ddof=1)), cifre)
            if len(v) > 1
            else 0.0
        ),
        "n": int(len(v)),
    }


def _maschera_valida(d):
    d = np.asarray(d, dtype=float)
    return (
        np.isfinite(d[:, :2]).all(axis=1)
        & np.isfinite(d[:, 2])
        & (d[:, 2] >= VIS_MIN)
    )


def _riepilogo_to(debug_eventi, massimo=5):
    mancanti = [
        r for r in debug_eventi
        if r["ic_frame"] is not None and r["to_frame"] is None
    ]

    parti = []
    for r in mancanti[:massimo]:
        disponibile = r["risalita_disponibile_px"]
        disponibile_txt = (
            f"{disponibile:.1f}"
            if disponibile is not None
            else "non disponibile"
        )

        parti.append(
            f"ciclo {r['ciclo']}: richiesta "
            f"{r['risalita_richiesta_px']:.1f} px, "
            f"disponibile {disponibile_txt} px"
        )

    return "; ".join(parti)


# ----------------------------------------------------------------------------
# Estrazione dei landmark
# ----------------------------------------------------------------------------

def estrai_landmarks(video_path, model_complexity=1):
    """
    Restituisce:
        punti[(parte, lato)] -> array (N, 3): x, y, visibilità
        fps
        avvisi
    """
    import cv2
    import mediapipe as mp

    if model_complexity not in (0, 1, 2):
        raise ValueError("model_complexity deve essere 0, 1 oppure 2.")

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "La versione installata di MediaPipe non espone mp.solutions. "
            "Questo modulo usa l'API Pose legacy: occorre un ambiente "
            "compatibile oppure una migrazione a MediaPipe Tasks."
        )

    mp_pose = mp.solutions.pose
    lm = mp_pose.PoseLandmark

    mappa = {
        ("spalla", "sx"): lm.LEFT_SHOULDER,
        ("spalla", "dx"): lm.RIGHT_SHOULDER,
        ("anca", "sx"): lm.LEFT_HIP,
        ("anca", "dx"): lm.RIGHT_HIP,
        ("ginocchio", "sx"): lm.LEFT_KNEE,
        ("ginocchio", "dx"): lm.RIGHT_KNEE,
        ("caviglia", "sx"): lm.LEFT_ANKLE,
        ("caviglia", "dx"): lm.RIGHT_ANKLE,
        ("tallone", "sx"): lm.LEFT_HEEL,
        ("tallone", "dx"): lm.RIGHT_HEEL,
        ("punta", "sx"): lm.LEFT_FOOT_INDEX,
        ("punta", "dx"): lm.RIGHT_FOOT_INDEX,
    }
    chiavi = list(mappa)

    cap = cv2.VideoCapture(str(video_path))
    avvisi = []
    righe = []
    timestamp = []

    try:
        if not cap.isOpened():
            raise ValueError(
                "Impossibile aprire il file video. "
                "Verifica il percorso e il formato."
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1:
            fps = 30.0
            avvisi.append(
                "FPS non presenti o non validi: assunti 30. "
                "Cadenza e tempi di contatto potrebbero essere errati."
            )

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                letto, frame = cap.read()
                if not letto:
                    break

                timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ris = pose.process(rgb)

                riga = np.full((len(chiavi), 3), np.nan)

                if ris.pose_landmarks is not None:
                    landmarks = ris.pose_landmarks.landmark

                    for k, chiave in enumerate(chiavi):
                        p = landmarks[mappa[chiave]]
                        riga[k] = (
                            p.x * w,
                            p.y * h,
                            p.visibility,
                        )

                righe.append(riga)

    finally:
        cap.release()

    if len(righe) < 10:
        raise ValueError("Video troppo corto o non leggibile.")

    dati = np.stack(righe)
    punti = {
        chiave: dati[:, k, :]
        for k, chiave in enumerate(chiavi)
    }

    ts = np.asarray(timestamp, dtype=float)

    if np.isfinite(ts).all() and np.all(np.diff(ts) > 0):
        fps_ts = 1000.0 * (len(ts) - 1) / (ts[-1] - ts[0])

        if abs(fps_ts - fps) / fps > 0.05:
            avvisi.append(
                f"FPS dichiarati ({fps:.1f}) e FPS dai timestamp "
                f"({fps_ts:.1f}) differiscono. Possibile frame rate "
                "variabile: verifica cadenza e tempi di contatto."
            )

    return punti, fps, avvisi


# ----------------------------------------------------------------------------
# Analisi delle serie
# ----------------------------------------------------------------------------

def analizza_serie(
    punti,
    fps,
    altezza_cm=175.0,
    avvisi_iniziali=None,
    debug=DEBUG_EVENTI,
):
    avvisi = list(avvisi_iniziali or [])

    fps = float(fps)
    altezza_cm = float(altezza_cm)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Gli FPS devono essere finiti e maggiori di zero.")

    if not np.isfinite(altezza_cm) or altezza_cm <= 0:
        raise ValueError("L'altezza deve essere finita e maggiore di zero.")

    # --- Verifica struttura dati ---
    lunghezze = set()

    for lato_candidato in LATI:
        for parte in PARTI:
            chiave = (parte, lato_candidato)

            if chiave not in punti:
                raise ValueError(f"Landmark mancante: {chiave}.")

            d = np.asarray(punti[chiave], dtype=float)

            if d.ndim != 2 or d.shape[1] != 3:
                raise ValueError(
                    f"Il landmark {chiave} deve avere forma (N, 3)."
                )

            lunghezze.add(d.shape[0])

    if len(lunghezze) != 1:
        raise ValueError("Le serie dei landmark hanno lunghezze diverse.")

    n = next(iter(lunghezze))
    if n < 10:
        raise ValueError("Serie troppo corta per l'analisi.")

    # --- Lato con migliore copertura dei landmark core ---
    def qualita_lato(lato_candidato):
        frazioni_lato = [
            float(
                _maschera_valida(
                    punti[(parte, lato_candidato)]
                ).mean()
            )
            for parte in CORE
        ]

        return min(frazioni_lato), float(np.mean(frazioni_lato))

    lato = max(LATI, key=qualita_lato)

    P = {
        parte: np.asarray(punti[(parte, lato)], dtype=float)
        for parte in PARTI
    }

    # --- Validità per landmark ---
    ok = {
        parte: _maschera_valida(P[parte])
        for parte in PARTI
    }

    frazioni = {
        parte: float(ok[parte].mean())
        for parte in PARTI
    }

    peggiore = min(CORE, key=lambda parte: frazioni[parte])

    if (
        frazioni[peggiore] < MIN_FRAZ_VALIDA
        or any(ok[parte].sum() < 2 for parte in PARTI)
    ):
        dett = ", ".join(
            f"{parte} {frazioni[parte] * 100:.0f}%"
            for parte in PARTI
        )

        raise ValueError(
            "Tracking insufficiente sul lato selezionato "
            f"({lato}). Frame validi per punto: {dett}. "
            "Controlla anche eventuali tratti senza persona "
            "all'inizio o alla fine del video."
        )

    fraz_core = float(
        np.mean([frazioni[parte] for parte in CORE])
    )

    if fraz_core < 0.8:
        dett = ", ".join(
            f"{parte} {frazioni[parte] * 100:.0f}%"
            for parte in PARTI
        )
        avvisi.append(
            f"Copertura media dei landmark core: "
            f"{fraz_core * 100:.0f}% ({dett}). "
            "La copertura non misura l'accuratezza delle coordinate."
        )

    # --- Interpolazione e filtro ---
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    idx = np.arange(n)
    finestra = max(5, int(round(FILTRO_S * fps)))

    S = {}
    inaff = np.zeros(n, dtype=bool)

    for parte in PARTI:
        for c, nome in ((0, "x"), (1, "y")):
            interpolata = np.interp(
                idx,
                idx[ok[parte]],
                P[parte][ok[parte], c],
            )
            S[(parte, nome)] = _savgol(interpolata, finestra)

        if parte in CORE:
            inaff |= _buchi_lunghi(ok[parte], max_gap)

    # --- Scala pixel -> cm ---
    coscia = np.hypot(
        S[("anca", "x")] - S[("ginocchio", "x")],
        S[("anca", "y")] - S[("ginocchio", "y")],
    )

    gamba = np.hypot(
        S[("ginocchio", "x")] - S[("caviglia", "x")],
        S[("ginocchio", "y")] - S[("caviglia", "y")],
    )

    m3 = ok["anca"] & ok["ginocchio"] & ok["caviglia"]

    if m3.sum() < 3:
        raise ValueError(
            "Troppi pochi frame con anca, ginocchio e caviglia "
            "simultaneamente validi per stimare la scala."
        )

    lung_arto_px = float(np.median((coscia + gamba)[m3]))

    if not np.isfinite(lung_arto_px) or lung_arto_px <= 0:
        raise ValueError("Impossibile stimare la lunghezza dell'arto.")

    scala = (
        (FRAZ_COSCIA + FRAZ_GAMBA)
        * altezza_cm
        / lung_arto_px
    )

    # --- Punto più basso del piede ---
    y_piede = np.maximum(
        S[("tallone", "y")],
        S[("punta", "y")],
    )

    # --- Picchi di sollevamento del piede ---
    periodo = _periodo_autocorr(y_piede, fps)

    distanza_s = (
        0.6 * periodo
        if periodo is not None
        else DIST_MIN_PASSO_S
    )

    dist_min = max(
        1,
        int(round(distanza_s * fps)),
        int(round(0.3 * fps)),
    )

    s_alto = -y_piede
    pk = _massimi(
        s_alto,
        dist_min,
        float(np.median(s_alto)),
    )

    if len(pk) < 3:
        raise ValueError(
            f"Picchi del piede insufficienti: {len(pk)}. "
            "Non è possibile delimitare abbastanza cicli. "
            "Controlla tracking e andamento verticale del piede."
        )

    # --- IC e TO tra picchi consecutivi ---
    tol = TOL_SUOLO * lung_arto_px
    ic = []
    to = []
    debug_eventi = []
    n_ic_mancanti = 0

    for k in range(len(pk) - 1):
        a, b = pk[k], pk[k + 1]
        seg = y_piede[a:b + 1]

        suolo = float(np.percentile(seg, 90))
        alto = float(np.percentile(seg, 5))
        escursione_px = max(0.0, suolo - alto)
        ampiezza = max(escursione_px, tol)

        tol_ic = max(
            tol,
            FRAZ_AMPIEZZA_IC * ampiezza,
        )

        # Modifica rispetto alla versione precedente:
        # minimo = tol invece di 2 * tol.
        tol_to = max(
            MOLTIPLICATORE_TOL_TO * tol,
            FRAZ_AMPIEZZA_TO * ampiezza,
        )

        riga_debug = {
            "ciclo": int(k + 1),
            "inizio_frame": int(a),
            "fine_frame": int(b),
            "durata_picco_picco_s": round((b - a) / fps, 3),
            "ic_frame": None,
            "to_frame": None,
            "suolo_stimato_y_px": round(suolo, 2),
            "escursione_px": round(escursione_px, 2),
            "ampiezza_usata_px": round(float(ampiezza), 2),
            "risalita_richiesta_px": round(float(tol_to), 2),
            "risalita_disponibile_px": None,
            "inizio_ricerca_to_frame": None,
        }

        c = np.flatnonzero(seg >= suolo - tol_ic)

        if len(c) == 0:
            ic.append(None)
            to.append(None)
            n_ic_mancanti += 1
            debug_eventi.append(riga_debug)
            continue

        ic_k = a + int(c[0])

        minimo_contatto_frame = max(
            1,
            int(math.ceil(CONTATTO_MIN_S * fps)),
        )

        a2 = ic_k + minimo_contatto_frame
        b2 = b
        to_k = None

        riga_debug["ic_frame"] = int(ic_k)
        riga_debug["inizio_ricerca_to_frame"] = int(a2)

        if a2 <= b2:
            finestra_to = y_piede[a2:b2 + 1]

            riga_debug["risalita_disponibile_px"] = round(
                float(suolo - np.min(finestra_to)),
                2,
            )

            c2 = np.flatnonzero(
                finestra_to < suolo - tol_to
            )

            if len(c2):
                to_k = a2 + int(c2[0])
                riga_debug["to_frame"] = int(to_k)

        ic.append(ic_k)
        to.append(to_k)
        debug_eventi.append(riga_debug)

    if debug:
        print(
            "DEBUG ANALISI:",
            {
                "lato": lato,
                "fps": round(fps, 2),
                "lunghezza_arto_px": round(lung_arto_px, 2),
                "tol_suolo_px": round(tol, 2),
                "periodo_autocorr_s": (
                    round(periodo, 3)
                    if periodo is not None
                    else None
                ),
                "picchi_rilevati": len(pk),
                "distanza_minima_picchi_frame": dist_min,
            },
        )

        for riga in debug_eventi:
            if riga["ic_frame"] is None:
                print("DEBUG IC MANCANTE:", riga)
            elif riga["to_frame"] is None:
                print("DEBUG TOE-OFF MANCANTE:", riga)
            else:
                print("DEBUG EVENTI OK:", riga)

    # --- Direzione di corsa ---
    entrambi_validi = ok["punta"] & ok["tallone"]

    if entrambi_validi.any():
        orientamento = float(
            np.median(
                (
                    P["punta"][:, 0]
                    - P["tallone"][:, 0]
                )[entrambi_validi]
            )
        )
        dir_piede = 1.0 if orientamento >= 0 else -1.0
    else:
        dir_piede = 1.0
        avvisi.append(
            "Orientamento tallone-punta non disponibile: "
            "il valore di ripiego è verso destra."
        )

    dxr = S[("caviglia", "x")] - S[("anca", "x")]

    diffs = [
        float(dxr[i] - dxr[t])
        for i, t in zip(ic, to)
        if i is not None and t is not None
    ]

    d_dir = float(np.median(diffs)) if diffs else 0.0

    if abs(d_dir) >= SOGLIA_VERSO * lung_arto_px:
        direzione = 1.0 if d_dir > 0 else -1.0

        if direzione != dir_piede:
            avvisi.append(
                "Il verso ricavato dal movimento non coincide "
                "con l'orientamento tallone-punta. "
                "È stato usato il movimento."
            )
    else:
        direzione = dir_piede
        avvisi.append(
            "Verso di corsa stimato dall'orientamento del piede: "
            "spostamento caviglia-anca IC→TO insufficiente "
            "per una stima dal movimento."
        )

    # --- Serie derivate ---
    xrel = direzione * dxr

    incluso = _angolo_incluso(
        S[("anca", "x")],
        S[("anca", "y")],
        S[("ginocchio", "x")],
        S[("ginocchio", "y")],
        S[("caviglia", "x")],
        S[("caviglia", "y")],
    )
    flex = 180.0 - incluso

    tibia = np.degrees(
        np.arctan2(
            direzione * (
                S[("caviglia", "x")]
                - S[("ginocchio", "x")]
            ),
            S[("caviglia", "y")]
            - S[("ginocchio", "y")],
        )
    )

    busto = np.degrees(
        np.arctan2(
            direzione * (
                S[("spalla", "x")]
                - S[("anca", "x")]
            ),
            S[("anca", "y")]
            - S[("spalla", "y")],
        )
    )

    fsa = np.degrees(
        np.arctan2(
            S[("tallone", "y")]
            - S[("punta", "y")],
            direzione * (
                S[("punta", "x")]
                - S[("tallone", "x")]
            ),
        )
    )

    # --- Metriche per ciclo ---
    chiavi_metriche = (
        "stride",
        "contatto",
        "over_cm",
        "over_pct",
        "tibia",
        "flex_ic",
        "flex_stance",
        "flex_to",
        "flex_swing",
        "fsa",
        "busto",
        "vo",
    )
    m = {chiave: [] for chiave in chiavi_metriche}

    ic_validi = []
    to_validi = []
    scartati = 0
    motivi = Counter()
    dbg_stride = []
    dbg_cont = []
    debug_validazione = []

    n_cicli = max(0, len(ic) - 1)

    for k in range(n_cicli):
        i0 = ic[k]
        i1 = ic[k + 1]
        t0 = to[k]

        dettaglio = {
            "ciclo": int(k + 1),
            "ic_frame": int(i0) if i0 is not None else None,
            "ic_successivo_frame": (
                int(i1) if i1 is not None else None
            ),
            "to_frame": int(t0) if t0 is not None else None,
            "stride_s": None,
            "contatto_s": None,
            "valido": False,
            "motivo_scarto": None,
        }

        motivo = None

        if i0 is None or i1 is None:
            motivo = "contatto iniziale non trovato"

        elif t0 is None:
            motivo = (
                "stacco (toe-off) non trovato: "
                "soglia di risalita non attraversata"
            )

        elif not (i0 < t0 < i1):
            motivo = "sequenza contatto/stacco incoerente"

        else:
            t_stride = (i1 - i0) / fps
            t_cont = (t0 - i0) / fps

            dbg_stride.append(t_stride)
            dbg_cont.append(t_cont)

            dettaglio["stride_s"] = round(t_stride, 3)
            dettaglio["contatto_s"] = round(t_cont, 3)

            if inaff[i0:i1 + 1].any():
                motivo = "tracking interrotto per troppo tempo"

            elif not (
                STRIDE_MIN_S <= t_stride <= STRIDE_MAX_S
            ):
                motivo = (
                    "durata del ciclo fuori range "
                    f"({STRIDE_MIN_S}-{STRIDE_MAX_S} s)"
                )

            elif not (
                CONTATTO_MIN_S <= t_cont <= CONTATTO_MAX_S
            ):
                motivo = (
                    "tempo di contatto fuori range "
                    f"({CONTATTO_MIN_S}-{CONTATTO_MAX_S} s)"
                )

        if motivo is not None:
            scartati += 1
            motivi[motivo] += 1
            dettaglio["motivo_scarto"] = motivo
            debug_validazione.append(dettaglio)
            continue

        dettaglio["valido"] = True
        debug_validazione.append(dettaglio)

        ic_validi.append(i0)
        to_validi.append(t0)

        m["stride"].append(t_stride)
        m["contatto"].append(t_cont * 1000.0)
        m["over_cm"].append(xrel[i0] * scala)
        m["over_pct"].append(
            xrel[i0] / lung_arto_px * 100.0
        )
        m["tibia"].append(tibia[i0])
        m["flex_ic"].append(flex[i0])
        m["flex_stance"].append(
            np.nanmax(flex[i0:t0 + 1])
        )
        m["flex_to"].append(flex[t0])
        m["flex_swing"].append(
            np.nanmax(flex[t0:i1 + 1])
        )
        m["fsa"].append(fsa[i0])
        m["busto"].append(
            np.mean(busto[i0:i1 + 1])
        )

        y_hip = S[("anca", "y")][i0:i1 + 1]
        trend = np.linspace(
            y_hip[0],
            y_hip[-1],
            len(y_hip),
        )
        y_detrend = y_hip - trend

        m["vo"].append(
            (np.max(y_detrend) - np.min(y_detrend))
            * scala
        )

    n_validi = len(ic_validi)

    if debug:
        print(
            "DEBUG VALIDAZIONE:",
            {
                "cicli_candidati": n_cicli,
                "cicli_validi": n_validi,
                "cicli_scartati": scartati,
                "motivi": dict(motivi),
            },
        )

        for riga in debug_validazione:
            if not riga["valido"]:
                print("DEBUG CICLO SCARTATO:", riga)

    # --- Errore diagnostico ---
    if n_validi < 3:
        dettaglio = "; ".join(
            f"{conteggio}× {motivo}"
            for motivo, conteggio in motivi.items()
        ) or "nessun dettaglio"

        misure = ""
        if dbg_stride:
            misure = (
                f" Valori misurati: ciclo "
                f"{min(dbg_stride):.2f}-"
                f"{max(dbg_stride):.2f} s, "
                f"contatto {min(dbg_cont):.2f}-"
                f"{max(dbg_cont):.2f} s "
                f"a {fps:.0f} fps."
            )

        vis_txt = ", ".join(
            f"{parte} {frazioni[parte] * 100:.0f}%"
            for parte in PARTI
        )

        verso_txt = (
            "destra" if direzione > 0 else "sinistra"
        )
        piede_txt = (
            "destra" if dir_piede > 0 else "sinistra"
        )

        diag = (
            f" Diagnostica: verso {verso_txt}, "
            f"spostamento caviglia-anca IC→TO "
            f"{d_dir:.1f} px, "
            f"piede verso {piede_txt}, "
            f"ciclo mediano picco-picco "
            f"{np.median(np.diff(pk)) / fps:.2f} s, "
            f"cicli {n_cicli}, "
            f"IC mancanti {n_ic_mancanti}, "
            f"lato {lato}, "
            f"frame validi per punto: {vis_txt}."
        )

        riepilogo_to = _riepilogo_to(debug_eventi)
        if riepilogo_to:
            diag += (
                " Diagnostica TO: "
                + riepilogo_to
                + "."
            )

        raise ValueError(
            f"Solo {n_validi} cicli validi su {n_cicli}. "
            f"Motivi di scarto: {dettaglio}."
            f"{misure}{diag} "
            "Il fallimento può dipendere da tracking, soglie "
            "o segmentazione dei cicli; non dimostra da solo "
            "che la ripresa sia inutilizzabile."
        )

    # --- Avvisi ---
    if n_validi < 8:
        avvisi.append(
            f"Solo {n_validi} cicli validi dello stesso piede: "
            "le statistiche sono basate su un campione piccolo."
        )

    if fps < 60:
        avvisi.append(
            f"A {fps:.0f} fps l'intervallo tra frame è "
            f"{1000.0 / fps:.0f} ms. "
            "IC e TO restano stime da verificare sul video; "
            "l'errore può superare un intervallo di frame."
        )

    avvisi.append(
        "Soglia minima TO impostata a "
        f"{MOLTIPLICATORE_TOL_TO:g} × tol. "
        "Configurazione sperimentale: verificare gli eventi "
        "sui frame prima di interpretare il tempo di contatto."
    )

    avvisi.append(
        "Analisi 2D: angoli proiettati sul piano immagine. "
        "Scala in cm indicativa, derivata da proporzioni medie "
        "e altezza dichiarata, non da calibrazione del video."
    )

    if scartati:
        avvisi.append(
            f"Scartati {scartati} cicli su {n_cicli}. "
            "Le medie descrivono solo i cicli accettati."
        )

    # --- Classificazione dell'appoggio ---
    classi = [
        (
            "tallone"
            if valore > FSA_TALLONE
            else (
                "avampiede"
                if valore < FSA_AVAMPIEDE
                else "mesopiede"
            )
        )
        for valore in m["fsa"]
    ]

    conteggio = Counter(classi)
    classe_prevalente = conteggio.most_common(1)[0][0]

    distribuzione = {
        classe: round(
            100.0 * conteggio.get(classe, 0) / n_validi,
            1,
        )
        for classe in ("tallone", "mesopiede", "avampiede")
    }

    tempo_stride_medio = float(np.mean(m["stride"]))
    cadenza = 120.0 / tempo_stride_medio

    flex_fasi = [
        _stat(m["flex_ic"]),
        _stat(m["flex_stance"]),
        _stat(m["flex_to"]),
        _stat(m["flex_swing"]),
    ]

    fasi = [
        "Contatto iniziale",
        "Max flessione (appoggio)",
        "Stacco",
        "Max flessione (oscillazione)",
    ]

    # Sceglie un contatto realmente rilevato, non un frame intermedio.
    mediana_ic = float(np.median(ic_validi))
    ic_centrale = min(
        ic_validi,
        key=lambda frame: abs(frame - mediana_ic),
    )

    return {
        # Metadati
        "lato_analizzato": (
            "Sinistro" if lato == "sx" else "Destro"
        ),
        "direzione_corsa": (
            "verso destra"
            if direzione > 0
            else "verso sinistra"
        ),
        "fps_video": round(fps, 1),
        "frame_strike_analizzato": int(ic_centrale),
        "frame_contatto": [int(v) for v in ic_validi],
        "frame_stacco": [int(v) for v in to_validi],

        # Qualità
        "qualita": {
            "frame_totali": int(n),
            "frame_tracking_valido_pct": round(
                fraz_core * 100,
                1,
            ),
            "passi_validi": int(n_validi),
            "passi_scartati": int(scartati),
            "cicli_validi": int(n_validi),
            "cicli_scartati": int(scartati),
            "risoluzione_temporale_ms": round(
                1000.0 / fps,
                1,
            ),
            "scala_cm_per_pixel": round(scala, 4),
        },
        "avvisi": avvisi,

        # Metriche
        "cadenza_spm": round(cadenza, 1),
        "tempo_contatto_ms": _stat(m["contatto"], 0),

        "overstride_cm": round(
            float(np.mean(m["over_cm"])),
            1,
        ),
        "overstride": {
            "cm": _stat(m["over_cm"]),
            "pct_lunghezza_arto": _stat(m["over_pct"]),
        },

        "angolo_tibia_contatto_gradi": _stat(m["tibia"]),
        "flessione_ginocchio_gradi": dict(
            zip(fasi, flex_fasi)
        ),

        "angolo_ginocchio_appoggio": round(
            180.0 - float(np.mean(m["flex_ic"])),
            1,
        ),

        "angolo_inclinazione_busto": round(
            float(np.mean(m["busto"])),
            1,
        ),
        "inclinazione_busto_gradi": _stat(m["busto"]),

        "oscillazione_verticale": round(
            float(np.mean(m["vo"])),
            1,
        ),
        "oscillazione_verticale_cm": _stat(m["vo"]),

        "foot_strike_angle_gradi": _stat(m["fsa"]),
        "tipo_appoggio": ETICHETTE_APPOGGIO[
            classe_prevalente
        ],
        "distribuzione_appoggio_pct": distribuzione,

        # Compatibilità grafici
        "fasi_gait": fasi,
        "angoli_fase": [
            (
                round(180.0 - s["media"], 1)
                if s["media"] is not None
                else None
            )
            for s in flex_fasi
        ],

        # Diagnostica serializzabile
        "diagnostica": {
            "periodo_autocorr_s": (
                round(periodo, 3)
                if periodo is not None
                else None
            ),
            "picchi_piede_frame": [int(v) for v in pk],
            "lunghezza_arto_px": round(lung_arto_px, 2),
            "tolleranza_suolo_px": round(tol, 2),
            "moltiplicatore_tol_to": (
                MOLTIPLICATORE_TOL_TO
            ),
            "frazione_ampiezza_to": FRAZ_AMPIEZZA_TO,
            "copertura_landmark_pct": {
                parte: round(frazioni[parte] * 100, 1)
                for parte in PARTI
            },
            "motivi_scarto": dict(motivi),
            "eventi": debug_eventi,
            "validazione_cicli": debug_validazione,
        },

        "note_metodologiche": [
            (
                "I cicli candidati sono delimitati da picchi "
                "di sollevamento del piede. Un picco non è "
                "garantito per ogni ciclo."
            ),
            (
                "IC = primo avvicinamento del punto più basso "
                "del piede al livello di suolo stimato "
                "nel ciclo."
            ),
            (
                "TO = primo attraversamento della soglia "
                "di risalita dopo il contatto minimo. "
                "Non è una misura diretta del distacco."
            ),
            (
                "Verso di corsa stimato dal movimento "
                "caviglia-anca IC→TO, con ripiego "
                "sull'orientamento tallone-punta."
            ),
            (
                "Overstride = distanza orizzontale "
                "caviglia-anca all'IC; positiva se "
                "il piede è davanti all'anca."
            ),
            (
                "Flessione ginocchio = 180° meno "
                "l'angolo anca-ginocchio-caviglia."
            ),
            (
                "Classificazione FSA con soglie configurate: "
                "> 8° tallone; da -1,6° a 8° mesopiede; "
                "< -1,6° avampiede. Soglie da verificare "
                "rispetto al protocollo di misura."
            ),
            (
                "Cadenza = 120 / periodo medio dello stesso "
                "piede: presuppone due passi per ciclo."
            ),
            (
                "Le chiavi passi_validi e passi_scartati "
                "contano cicli dello stesso piede."
            ),
            (
                "Nessun indice di rischio infortunio "
                "o carico articolare viene calcolato."
            ),
        ],
    }


# ----------------------------------------------------------------------------
# API pubblica
# ----------------------------------------------------------------------------

def analizza_running_video(
    video_path,
    altezza_cm=175,
    model_complexity=1,
    debug=DEBUG_EVENTI,
):
    """
    Analizza il video usando il modello legacy 1.

    Il modello heavy e il fallback automatico sono disattivati
    per evitare il download nella cartella protetta del pacchetto.
    """
    if model_complexity != 1:
        raise ValueError(
            "Questa configurazione usa model_complexity=1. "
            "Il modello heavy e il fallback automatico "
            "sono disattivati per evitare l'errore "
            "di permessi sul file del modello."
        )

    punti, fps, avvisi = estrai_landmarks(
        video_path,
        model_complexity=1,
    )

    return analizza_serie(
        punti,
        fps,
        altezza_cm=altezza_cm,
        avvisi_iniziali=avvisi,
        debug=debug,
    )


def diagnostica_video(
    video_path,
    altezza_cm=175,
    model_complexity=1,
):
    """Stampa tracking, eventi rilevati e risultato dell'analisi."""
    if model_complexity != 1:
        raise ValueError(
            "Usa model_complexity=1 in questo deployment."
        )

    punti, fps, avvisi = estrai_landmarks(
        video_path,
        model_complexity=1,
    )

    n = next(iter(punti.values())).shape[0]

    print(
        f"Frame: {n} | FPS: {fps:.1f} | "
        f"Durata approssimativa: {n / fps:.1f} s"
    )

    for avviso in avvisi:
        print("AVVISO:", avviso)

    print(
        "Tracking per landmark: rilevato / "
        "visibilità media / coordinate valide sopra soglia"
    )

    for lato in LATI:
        for parte in PARTI:
            d = punti[(parte, lato)]
            vis = np.nan_to_num(
                d[:, 2],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            rilevato = np.isfinite(d[:, :2]).all(axis=1)
            validi = _maschera_valida(d)

            print(
                f"  {lato} {parte:10s} "
                f"rilevato {100 * rilevato.mean():5.1f}% | "
                f"vis media {vis.mean():.2f} | "
                f"validi {100 * validi.mean():5.1f}%"
            )

    try:
        risultato = analizza_serie(
            punti,
            fps,
            altezza_cm=altezza_cm,
            avvisi_iniziali=avvisi,
            debug=True,
        )

        print(
            "\nANALISI OK:",
            risultato["qualita"]["cicli_validi"],
            "cicli validi | cadenza",
            risultato["cadenza_spm"],
            "spm | contatto",
            risultato["tempo_contatto_ms"]["media"],
            "ms |",
            risultato["direzione_corsa"],
        )

        for avviso in risultato["avvisi"]:
            print("AVVISO:", avviso)

        return risultato

    except ValueError as exc:
        print("\nANALISI NON RIUSCITA:", exc)
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Uso: python utils/computer_vision.py "
            "video.mp4 [altezza_cm] [model_complexity]"
        )
    else:
        diagnostica_video(
            sys.argv[1],
            altezza_cm=(
                float(sys.argv[2])
                if len(sys.argv) > 2
                else 175.0
            ),
            model_complexity=(
                int(sys.argv[3])
                if len(sys.argv) > 3
                else 1
            ),
        )
