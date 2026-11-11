"""
utils/computer_vision.py
------------------------
Analisi cinematica 2D della corsa (ripresa laterale) con MediaPipe Pose legacy,
piu' stima euristica di carichi e rischio per distretto.

LIMITI
    - IC e TO sono stime cinematiche dal segnale del piede, non misure del
      contatto. A 30 fps l'errore puo' superare un frame (33 ms).
    - Gli angoli sono proiezioni 2D; la scala in cm deriva dall'altezza
      dichiarata e da proporzioni medie.
    - Carichi e rischio sono PUNTEGGI EURISTICI, non un modello clinico.
"""

from collections import Counter

import numpy as np


# ----------------------------------------------------------------------------
# Costanti
# ----------------------------------------------------------------------------

LATI = ("sx", "dx")
PARTI = ("spalla", "anca", "ginocchio", "caviglia", "tallone", "punta")
CORE = ("anca", "ginocchio", "caviglia", "tallone", "punta")

FRAZ_COSCIA = 0.245
FRAZ_GAMBA = 0.246

VIS_MIN = 0.20
MIN_FRAZ_VALIDA = 0.25
MAX_GAP_S = 0.30
FILTRO_S = 0.10
MAX_FRAZ_INTERP = 0.35

DIST_MIN_PASSO_S = 0.35
TOL_SUOLO = 0.03
FRAZ_PICCO = 0.40
FRAZ_STANCE = 0.15
MIN_ESC_FRAZ = 0.08
MIN_TRATTO_S = 0.5

SOGLIA_VERSO = 0.05

CONTATTO_MIN_S = 0.08
CONTATTO_MAX_S = 0.70
STRIDE_MIN_S = 0.40
STRIDE_MAX_S = 1.80

MIN_CICLI_VALIDI = 2

FSA_TALLONE = 8.0
FSA_AVAMPIEDE = -1.6

DEBUG_EVENTI = False

ETICHETTE_APPOGGIO = {
    "tallone": "Appoggio di Tallone (Rearfoot Strike)",
    "mesopiede": "Appoggio di Mesopiede (Midfoot Strike)",
    "avampiede": "Appoggio di Avampiede (Forefoot Strike)",
}


class AnalisiNonRiuscita(ValueError):
    """ValueError con i dati di diagnostica (serie del piede, picchi, cicli)."""

    def __init__(self, messaggio, dati=None):
        super().__init__(messaggio)
        self.dati = dati or {}


# ----------------------------------------------------------------------------
# Utilita' numeriche
# ----------------------------------------------------------------------------

def _savgol(y, finestra, ordine=3):
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
    """Massimi locali >= soglia, con distanza minima (priorita' ai piu' alti)."""
    cand = [
        i for i in range(1, len(y) - 1)
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= soglia
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
    cosang = np.clip((v1x * v2x + v1y * v2y) / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _stat(valori, cifre=1):
    v = np.asarray(valori, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"media": None, "sd": None, "n": 0}
    return {
        "media": round(float(v.mean()), cifre),
        "sd": round(float(v.std(ddof=1)), cifre) if len(v) > 1 else 0.0,
        "n": int(len(v)),
    }


def _maschera_valida(d):
    d = np.asarray(d, dtype=float)
    return (
        np.isfinite(d[:, :2]).all(axis=1)
        & np.isfinite(d[:, 2])
        & (d[:, 2] >= VIS_MIN)
    )


# ----------------------------------------------------------------------------
# Stima euristica di carichi e rischio
# ----------------------------------------------------------------------------

def stima_stress(res):
    """
    Calcola dalle metriche reali carichi per distretto, rischio relativo e
    indice complessivo. PUNTEGGI EURISTICI: pesi scelti a mano, non validati.
    """
    ang = float(res["angolo_ginocchio_appoggio"])
    over = max(0.0, float(res["overstride_cm"]))
    busto = float(res["angolo_inclinazione_busto"])
    ext = max(0.0, ang - 165.0)               # ginocchio troppo esteso al contatto
    distr = res.get("distribuzione_appoggio_pct", {})
    tal = float(distr.get("tallone", 0.0))
    ava = float(distr.get("avampiede", 0.0))
    dev_busto = abs(busto - 6.5)              # distanza da inclinazione ~5-8 gradi

    carichi = {
        "Ginocchio": 25 + 2.0 * over + 1.0 * ext,
        "Tibia": 20 + 1.5 * over + 0.25 * tal,
        "Tendine d'Achille": 20 + 0.30 * ava + 0.5 * over,
        "Anca": 15 + 2.0 * dev_busto,
        "Piede/Fascia": 15 + 0.20 * tal + 0.2 * ava,
    }
    carichi = {k: float(np.clip(v, 5, 95)) for k, v in carichi.items()}
    tot = sum(carichi.values())
    top = max(carichi, key=carichi.get)
    nomi = {
        "Ginocchio": "Sindrome femoro-rotulea (ginocchio del corridore)",
        "Tibia": "Periostite tibiale (shin splints)",
        "Tendine d'Achille": "Tendinopatia achillea",
        "Anca": "Sovraccarico anca / bandelletta ileo-tibiale",
        "Piede/Fascia": "Fascite plantare",
    }
    prob = float(np.clip(
        10 + 2.2 * over + 0.8 * ext + 0.15 * tal + 1.0 * dev_busto, 5, 95))

    return {
        "articolazioni_carico": list(carichi.keys()),
        "carichi_pct": [round(v, 1) for v in carichi.values()],
        "distretti_rischio": list(carichi.keys()),
        "rischi_ml": [round(100.0 * v / tot, 1) for v in carichi.values()],
        "sovraccarico_prevalente": top,
        "infortunio_predetto": nomi[top],
        "probabilita_infortunio_ml": round(prob, 1),
    }


# ----------------------------------------------------------------------------
# Estrazione dei landmark
# ----------------------------------------------------------------------------

def estrai_landmarks(video_path, model_complexity=1):
    """
    Restituisce punti[(parte, lato)] -> array (N, 3) con x, y, visibilita',
    gli fps e una lista di avvisi.
    """
    import cv2
    import mediapipe as mp

    if model_complexity not in (0, 1, 2):
        raise ValueError("model_complexity deve essere 0, 1 oppure 2.")

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "La versione installata di MediaPipe non espone mp.solutions. "
            "Fissa mediapipe a 0.10.x in requirements.txt (es. mediapipe==0.10.14)."
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
    avvisi, righe, timestamp = [], [], []

    try:
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il file video.")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1:
            fps = 30.0
            avvisi.append("FPS non validi nel file: assunti 30.")

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
                ris = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                riga = np.full((len(chiavi), 3), np.nan)
                if ris.pose_landmarks is not None:
                    lms = ris.pose_landmarks.landmark
                    for k, chiave in enumerate(chiavi):
                        p = lms[mappa[chiave]]
                        riga[k] = (p.x * w, p.y * h, p.visibility)
                righe.append(riga)
    finally:
        cap.release()

    if len(righe) < 10:
        raise ValueError("Video troppo corto o non leggibile.")

    dati = np.stack(righe)
    punti = {chiave: dati[:, k, :] for k, chiave in enumerate(chiavi)}

    ts = np.asarray(timestamp, dtype=float)
    if np.isfinite(ts).all() and len(ts) > 1 and np.all(np.diff(ts) > 0):
        fps_ts = 1000.0 * (len(ts) - 1) / (ts[-1] - ts[0])
        if abs(fps_ts - fps) / fps > 0.05:
            avvisi.append(
                f"FPS dichiarati ({fps:.1f}) diversi da quelli dai timestamp "
                f"({fps_ts:.1f}): possibile frame rate variabile."
            )

    return punti, fps, avvisi


# ----------------------------------------------------------------------------
# Analisi di un singolo lato
# ----------------------------------------------------------------------------

def _analizza_lato(punti, fps, altezza_cm, lato, avvisi_iniziali, debug, dbg):
    avvisi = list(avvisi_iniziali)

    P = {p: np.asarray(punti[(p, lato)], dtype=float) for p in PARTI}
    n = len(P["anca"])
    ok = {p: _maschera_valida(P[p]) for p in PARTI}
    frazioni = {p: float(ok[p].mean()) for p in PARTI}
    dett_cop = ", ".join(f"{p} {frazioni[p] * 100:.0f}%" for p in PARTI)

    peggiore = min(CORE, key=lambda p: frazioni[p])
    if frazioni[peggiore] < MIN_FRAZ_VALIDA or any(ok[p].sum() < 2 for p in PARTI):
        raise ValueError(
            f"Tracking insufficiente sul lato {lato}. "
            f"Frame validi per punto: {dett_cop}."
        )

    fraz_core = float(np.mean([frazioni[p] for p in CORE]))
    if fraz_core < 0.8:
        avvisi.append(
            f"Copertura media dei landmark core: {fraz_core * 100:.0f}% "
            f"({dett_cop}). La copertura non misura l'accuratezza."
        )

    # --- Interpolazione (solo buchi brevi) e filtro ---
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    idx = np.arange(n)
    finestra = max(7, int(round(FILTRO_S * fps)))

    S = {}
    inaff = np.zeros(n, dtype=bool)
    mancante = np.zeros(n, dtype=bool)
    for parte in PARTI:
        for c, nome in ((0, "x"), (1, "y")):
            serie = np.interp(idx, idx[ok[parte]], P[parte][ok[parte], c])
            S[(parte, nome)] = _savgol(serie, finestra)
        if parte in CORE:
            inaff |= _buchi_lunghi(ok[parte], max_gap)
            mancante |= ~ok[parte]

    # --- Scala pixel -> cm ---
    coscia = np.hypot(S[("anca", "x")] - S[("ginocchio", "x")],
                      S[("anca", "y")] - S[("ginocchio", "y")])
    gamba = np.hypot(S[("ginocchio", "x")] - S[("caviglia", "x")],
                     S[("ginocchio", "y")] - S[("caviglia", "y")])
    m3 = ok["anca"] & ok["ginocchio"] & ok["caviglia"]
    if m3.sum() < 3:
        raise ValueError(f"Lato {lato}: pochi frame con anca, ginocchio e caviglia validi.")

    lung_arto_px = float(np.median((coscia + gamba)[m3]))
    if not np.isfinite(lung_arto_px) or lung_arto_px <= 0:
        raise ValueError(f"Lato {lato}: impossibile stimare la lunghezza dell'arto.")
    scala = (FRAZ_COSCIA + FRAZ_GAMBA) * altezza_cm / lung_arto_px

    # --- Punto piu' basso del piede ---
    y_piede = np.maximum(S[("tallone", "y")], S[("punta", "y")])
    dbg["lato"] = lato
    dbg["fps"] = fps
    dbg["y_piede_px"] = [round(float(v), 1) for v in y_piede]

    # --- Tratti con tracking continuo ---
    tratti = []
    i = 0
    lung_min_tratto = max(8, int(round(MIN_TRATTO_S * fps)))
    while i < n:
        if inaff[i]:
            i += 1
            continue
        j = i
        while j < n and not inaff[j]:
            j += 1
        if j - i >= lung_min_tratto:
            tratti.append((i, j))
        i = j
    if not tratti:
        raise ValueError(
            f"Lato {lato}: nessun tratto continuo di tracking abbastanza lungo "
            f"(buchi piu' lunghi di {MAX_GAP_S} s)."
        )

    s0, e0 = max(tratti, key=lambda t: t[1] - t[0])
    periodo = _periodo_autocorr(y_piede[s0:e0], fps)
    dist_min_s = float(np.clip(0.6 * periodo, 0.30, 0.50)) if periodo else DIST_MIN_PASSO_S
    dist_min = max(2, int(round(dist_min_s * fps)))

    # --- Picchi e cicli, tratto per tratto ---
    tol = TOL_SUOLO * lung_arto_px
    cicli = []
    picchi_tot = []

    for (s, e) in tratti:
        y = y_piede[s:e]
        suolo_t = float(np.percentile(y, 90))
        alto_t = float(np.percentile(y, 5))
        if suolo_t - alto_t < MIN_ESC_FRAZ * lung_arto_px:
            continue
        soglia = -(suolo_t - FRAZ_PICCO * (suolo_t - alto_t))
        pk = [s + int(p) for p in _massimi(-y, dist_min, soglia)]
        picchi_tot += pk

        for a, b in zip(pk[:-1], pk[1:]):
            seg = y_piede[a:b + 1]
            suolo = float(np.percentile(seg, 90))
            esc = suolo - float(np.percentile(seg, 5))
            rec = {
                "ciclo": len(cicli) + 1,
                "inizio_frame": int(a),
                "fine_frame": int(b),
                "escursione_px": round(esc, 1),
                "ic_frame": None,
                "to_frame": None,
                "motivo": None,
            }

            if mancante[a:b + 1].mean() > MAX_FRAZ_INTERP:
                rec["motivo"] = "troppi frame interpolati nel ciclo"
            elif esc < MIN_ESC_FRAZ * lung_arto_px:
                rec["motivo"] = "escursione verticale del piede insufficiente"
            else:
                tol_st = max(tol, FRAZ_STANCE * esc)
                vicino = seg >= suolo - tol_st
                k0 = int(np.argmax(seg))
                i0 = k0
                while i0 > 0 and vicino[i0 - 1]:
                    i0 -= 1
                i1 = k0
                while i1 < len(seg) - 1 and vicino[i1 + 1]:
                    i1 += 1
                rec["ic_frame"] = int(a + i0)
                rec["to_frame"] = int(a + min(i1 + 1, len(seg) - 1))
            cicli.append(rec)

    dbg["picchi_piede_frame"] = [int(p) for p in picchi_tot]
    dbg["tratti"] = [[int(s), int(e)] for s, e in tratti]
    dbg["cicli"] = cicli

    if not cicli:
        raise ValueError(
            f"Lato {lato}: nessun ciclo candidato ({len(picchi_tot)} picchi, "
            f"{len(tratti)} tratti continui). Il piede sembra non muoversi "
            "abbastanza in verticale: controlla inquadratura e tracking."
        )

    # --- Direzione di corsa ---
    entrambi = ok["punta"] & ok["tallone"]
    if entrambi.any():
        orient = float(np.median((P["punta"][:, 0] - P["tallone"][:, 0])[entrambi]))
        dir_piede = 1.0 if orient >= 0 else -1.0
    else:
        dir_piede = 1.0
        avvisi.append("Orientamento tallone-punta non disponibile: assunto verso destra.")

    dxr = S[("caviglia", "x")] - S[("anca", "x")]
    diffs = [
        float(dxr[c["ic_frame"]] - dxr[c["to_frame"]])
        for c in cicli if c["ic_frame"] is not None and c["to_frame"] is not None
    ]
    d_dir = float(np.median(diffs)) if diffs else 0.0

    if abs(d_dir) >= SOGLIA_VERSO * lung_arto_px:
        direzione = 1.0 if d_dir > 0 else -1.0
        if direzione != dir_piede:
            avvisi.append(
                "Il verso ricavato dal movimento non coincide con "
                "l'orientamento tallone-punta: usato il movimento."
            )
    else:
        direzione = dir_piede
        avvisi.append(
            "Verso di corsa dedotto dall'orientamento del piede "
            "(spostamento caviglia-anca troppo piccolo per usare il movimento)."
        )

    # --- Serie derivate ---
    xrel = direzione * dxr
    incluso = _angolo_incluso(
        S[("anca", "x")], S[("anca", "y")],
        S[("ginocchio", "x")], S[("ginocchio", "y")],
        S[("caviglia", "x")], S[("caviglia", "y")],
    )
    flex = 180.0 - incluso
    tibia = np.degrees(np.arctan2(
        direzione * (S[("caviglia", "x")] - S[("ginocchio", "x")]),
        S[("caviglia", "y")] - S[("ginocchio", "y")],
    ))
    busto = np.degrees(np.arctan2(
        direzione * (S[("spalla", "x")] - S[("anca", "x")]),
        S[("anca", "y")] - S[("spalla", "y")],
    ))
    fsa = np.degrees(np.arctan2(
        S[("tallone", "y")] - S[("punta", "y")],
        direzione * (S[("punta", "x")] - S[("tallone", "x")]),
    ))

    # --- Validazione e metriche per ciclo ---
    chiavi_m = ("stride", "contatto", "over_cm", "over_pct", "tibia",
                "flex_ic", "flex_stance", "flex_to", "flex_swing",
                "fsa", "busto", "vo")
    m = {k: [] for k in chiavi_m}
    motivi = Counter()
    righe = []
    ic_validi, to_validi = [], []
    coda_swing = int(round(0.10 * fps))

    for rec in cicli:
        a, b = rec["inizio_frame"], rec["fine_frame"]
        i0, t0 = rec["ic_frame"], rec["to_frame"]
        stride = (b - a) / fps
        cont = (t0 - i0) / fps if (i0 is not None and t0 is not None) else None
        motivo = rec["motivo"]

        if motivo is None:
            if not (STRIDE_MIN_S <= stride <= STRIDE_MAX_S):
                motivo = f"durata del ciclo fuori range ({STRIDE_MIN_S}-{STRIDE_MAX_S} s)"
            elif not (CONTATTO_MIN_S <= cont <= CONTATTO_MAX_S):
                motivo = f"tempo di contatto fuori range ({CONTATTO_MIN_S}-{CONTATTO_MAX_S} s)"

        righe.append({
            "ciclo": rec["ciclo"], "inizio_frame": a, "fine_frame": b,
            "ic_frame": i0, "to_frame": t0,
            "stride_s": round(stride, 3),
            "contatto_s": round(cont, 3) if cont is not None else None,
            "valido": motivo is None, "motivo_scarto": motivo,
        })

        if motivo is not None:
            motivi[motivo] += 1
            continue

        ic_validi.append(i0)
        to_validi.append(t0)
        fine_sw = min(n - 1, b + coda_swing)

        m["stride"].append(stride)
        m["contatto"].append(cont * 1000.0)
        m["over_cm"].append(xrel[i0] * scala)
        m["over_pct"].append(xrel[i0] / lung_arto_px * 100.0)
        m["tibia"].append(tibia[i0])
        m["flex_ic"].append(flex[i0])
        m["flex_stance"].append(np.nanmax(flex[i0:t0 + 1]))
        m["flex_to"].append(flex[t0])
        m["flex_swing"].append(np.nanmax(flex[t0:fine_sw + 1]))
        m["fsa"].append(fsa[i0])
        m["busto"].append(np.mean(busto[a:b + 1]))

        y_hip = S[("anca", "y")][a:b + 1]
        detrend = y_hip - np.linspace(y_hip[0], y_hip[-1], len(y_hip))
        m["vo"].append((np.max(detrend) - np.min(detrend)) * scala)

    n_validi = len(ic_validi)
    scartati = len(righe) - n_validi

    dbg["validazione_cicli"] = righe
    dbg["frame_contatto"] = [int(r["ic_frame"]) for r in righe if r["ic_frame"] is not None]
    dbg["frame_stacco"] = [int(r["to_frame"]) for r in righe if r["to_frame"] is not None]

    if debug:
        print("DEBUG", lato, {
            "fps": round(fps, 2), "lung_arto_px": round(lung_arto_px, 1),
            "periodo_s": round(periodo, 3) if periodo else None,
            "tratti": tratti, "picchi": len(picchi_tot),
            "cicli": len(righe), "validi": n_validi, "motivi": dict(motivi),
        })

    if n_validi < MIN_CICLI_VALIDI:
        elenco = "; ".join(f"{c}x {mot}" for mot, c in motivi.items()) or "nessun dettaglio"
        raise AnalisiNonRiuscita(
            f"Lato {lato}: solo {n_validi} cicli validi su {len(righe)}. "
            f"Motivi di scarto: {elenco}. "
            f"Frame validi per punto: {dett_cop}. "
            f"Tratti continui: {len(tratti)}, picchi: {len(picchi_tot)}, "
            f"periodo stimato: {round(periodo, 2) if periodo else 'n.d.'} s.",
            dbg,
        )

    # --- Avvisi ---
    if n_validi < 8:
        avvisi.append(f"Solo {n_validi} cicli validi: statistiche su campione piccolo.")
    if fps < 60:
        avvisi.append(
            f"A {fps:.0f} fps l'intervallo tra frame e' {1000.0 / fps:.0f} ms: "
            "contatto, stacco e tempo di contatto hanno un errore di almeno un frame."
        )
    avvisi.append(
        "Analisi 2D: angoli proiettati sul piano immagine; scala in cm indicativa."
    )
    if scartati:
        avvisi.append(
            f"Scartati {scartati} cicli su {len(righe)}: le medie descrivono "
            "solo i cicli accettati."
        )

    # --- Classificazione dell'appoggio ---
    classi = [
        "tallone" if v > FSA_TALLONE else ("avampiede" if v < FSA_AVAMPIEDE else "mesopiede")
        for v in m["fsa"]
    ]
    conteggio = Counter(classi)
    prevalente = conteggio.most_common(1)[0][0]
    distribuzione = {
        c: round(100.0 * conteggio.get(c, 0) / n_validi, 1)
        for c in ("tallone", "mesopiede", "avampiede")
    }

    cadenza = 120.0 / float(np.mean(m["stride"]))

    flex_fasi = [_stat(m["flex_ic"]), _stat(m["flex_stance"]),
                 _stat(m["flex_to"]), _stat(m["flex_swing"])]
    fasi = ["Contatto iniziale", "Max flessione (appoggio)", "Stacco",
            "Max flessione (oscillazione)"]

    mediana_ic = float(np.median(ic_validi))
    ic_centrale = min(ic_validi, key=lambda f: abs(f - mediana_ic))

    return {
        "lato_analizzato": "Sinistro" if lato == "sx" else "Destro",
        "direzione_corsa": "verso destra" if direzione > 0 else "verso sinistra",
        "fps_video": round(fps, 1),
        "frame_strike_analizzato": int(ic_centrale),
        "frame_contatto": [int(v) for v in ic_validi],
        "frame_stacco": [int(v) for v in to_validi],

        "qualita": {
            "frame_totali": int(n),
            "frame_tracking_valido_pct": round(fraz_core * 100, 1),
            "passi_validi": int(n_validi),
            "passi_scartati": int(scartati),
            "cicli_validi": int(n_validi),
            "cicli_scartati": int(scartati),
            "risoluzione_temporale_ms": round(1000.0 / fps, 1),
            "scala_cm_per_pixel": round(scala, 4),
        },
        "avvisi": avvisi,

        "cadenza_spm": round(cadenza, 1),
        "tempo_contatto_ms": _stat(m["contatto"], 0),

        "overstride_cm": round(float(np.mean(m["over_cm"])), 1),
        "overstride": {"cm": _stat(m["over_cm"]),
                       "pct_lunghezza_arto": _stat(m["over_pct"])},

        "angolo_tibia_contatto_gradi": _stat(m["tibia"]),
        "flessione_ginocchio_gradi": dict(zip(fasi, flex_fasi)),
        "angolo_ginocchio_appoggio": round(180.0 - float(np.mean(m["flex_ic"])), 1),

        "angolo_inclinazione_busto": round(float(np.mean(m["busto"])), 1),
        "inclinazione_busto_gradi": _stat(m["busto"]),

        "oscillazione_verticale": round(float(np.mean(m["vo"])), 1),
        "oscillazione_verticale_cm": _stat(m["vo"]),

        "foot_strike_angle_gradi": _stat(m["fsa"]),
        "tipo_appoggio": ETICHETTE_APPOGGIO[prevalente],
        "distribuzione_appoggio_pct": distribuzione,

        "fasi_gait": fasi,
        "angoli_fase": [
            round(180.0 - s["media"], 1) if s["media"] is not None else None
            for s in flex_fasi
        ],

        "diagnostica": {
            "periodo_autocorr_s": round(periodo, 3) if periodo else None,
            "picchi_piede_frame": [int(v) for v in picchi_tot],
            "y_piede_px": dbg["y_piede_px"],
            "tratti": dbg["tratti"],
            "lunghezza_arto_px": round(lung_arto_px, 2),
            "tolleranza_suolo_px": round(tol, 2),
            "copertura_landmark_pct": {p: round(frazioni[p] * 100, 1) for p in PARTI},
            "motivi_scarto": dict(motivi),
            "validazione_cicli": righe,
        },

        "note_metodologiche": [
            "Cicli delimitati da picchi di sollevamento del piede, calcolati "
            "solo nei tratti con tracking continuo.",
            "IC = primo frame in cui il punto piu' basso del piede e' vicino "
            "al suolo stimato nel ciclo; TO = primo frame successivo in cui "
            "se ne stacca. Sono stime, non misure dirette.",
            "Overstride = distanza orizzontale caviglia-anca all'IC (positiva "
            "se il piede e' davanti all'anca).",
            "Flessione ginocchio = 180 gradi meno l'angolo anca-ginocchio-caviglia.",
            f"Foot strike angle: > {FSA_TALLONE} tallone; da {FSA_AVAMPIEDE} a "
            f"{FSA_TALLONE} mesopiede; < {FSA_AVAMPIEDE} avampiede.",
            "Cadenza = 120 / durata media del ciclo di un piede.",
            "Carichi e rischio per distretto sono punteggi euristici calcolati "
            "dalle metriche misurate, non un modello clinico validato.",
        ],
    }


# ----------------------------------------------------------------------------
# Analisi su entrambi i lati
# ----------------------------------------------------------------------------

def analizza_serie(punti, fps, altezza_cm=175.0, avvisi_iniziali=None,
                   debug=DEBUG_EVENTI, lato=None):
    """Prova entrambi i lati e restituisce quello con piu' cicli validi."""
    fps = float(fps)
    altezza_cm = float(altezza_cm)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Gli FPS devono essere finiti e maggiori di zero.")
    if not np.isfinite(altezza_cm) or altezza_cm <= 0:
        raise ValueError("L'altezza deve essere finita e maggiore di zero.")

    lunghezze = set()
    for l in LATI:
        for parte in PARTI:
            if (parte, l) not in punti:
                raise ValueError(f"Landmark mancante: {(parte, l)}.")
            d = np.asarray(punti[(parte, l)], dtype=float)
            if d.ndim != 2 or d.shape[1] != 3:
                raise ValueError(f"Il landmark {(parte, l)} deve avere forma (N, 3).")
            lunghezze.add(d.shape[0])
    if len(lunghezze) != 1:
        raise ValueError("Le serie dei landmark hanno lunghezze diverse.")
    if next(iter(lunghezze)) < 10:
        raise ValueError("Serie troppo corta per l'analisi.")

    avvisi = list(avvisi_iniziali or [])
    risultati, errori, dbgs = {}, {}, {}

    for l in ((lato,) if lato else LATI):
        dbg = {}
        try:
            risultati[l] = _analizza_lato(punti, fps, altezza_cm, l, avvisi, debug, dbg)
        except ValueError as exc:
            errori[l] = exc
            dbgs[l] = dbg

    if risultati:
        def punteggio(l):
            q = risultati[l]["qualita"]
            return (q["cicli_validi"], -q["cicli_scartati"], q["frame_tracking_valido_pct"])

        migliore = max(risultati, key=punteggio)
        res = risultati[migliore]
        if len(risultati) == 2:
            altro = "dx" if migliore == "sx" else "sx"
            res["avvisi"].insert(
                0,
                f"Lato {migliore} scelto per numero di cicli validi "
                f"({risultati[migliore]['qualita']['cicli_validi']} contro "
                f"{risultati[altro]['qualita']['cicli_validi']} del lato {altro}). "
                "Verifica sul video che sia la gamba vicina alla camera.",
            )
        elif errori:
            (l_err, exc), = errori.items()
            res["avvisi"].insert(0, f"Lato {l_err} non analizzabile: {exc}")
        res.update(stima_stress(res))
        return res

    migliore = max(
        errori, key=lambda l: len(dbgs[l].get("cicli", [])) if dbgs[l] else -1
    )
    messaggio = " || ".join(f"[{l}] {e}" for l, e in errori.items())
    raise AnalisiNonRiuscita(messaggio, dbgs.get(migliore) or {})


# ----------------------------------------------------------------------------
# API pubblica
# ----------------------------------------------------------------------------

def analizza_running_video(video_path, altezza_cm=175, model_complexity=1,
                           debug=DEBUG_EVENTI):
    punti, fps, avvisi = estrai_landmarks(video_path, model_complexity=model_complexity)
    return analizza_serie(punti, fps, altezza_cm=altezza_cm,
                          avvisi_iniziali=avvisi, debug=debug)


def diagnostica_video(video_path, altezza_cm=175, model_complexity=1):
    """Stampa copertura dei landmark e risultato dell'analisi."""
    punti, fps, avvisi = estrai_landmarks(video_path, model_complexity=model_complexity)
    n = next(iter(punti.values())).shape[0]
    print(f"Frame: {n} | FPS: {fps:.1f} | Durata: {n / fps:.1f} s")
    for a in avvisi:
        print("AVVISO:", a)
    for l in LATI:
        for parte in PARTI:
            d = punti[(parte, l)]
            vis = np.nan_to_num(d[:, 2], nan=0.0)
            print(f"  {l} {parte:10s} rilevato {100 * np.isfinite(d[:, :2]).all(axis=1).mean():5.1f}% "
                  f"| vis media {vis.mean():.2f} | validi {100 * _maschera_valida(d).mean():5.1f}%")
    try:
        r = analizza_serie(punti, fps, altezza_cm=altezza_cm,
                           avvisi_iniziali=avvisi, debug=True)
        print("\nANALISI OK:", r["qualita"]["cicli_validi"], "cicli | cadenza",
              r["cadenza_spm"], "| contatto", r["tempo_contatto_ms"]["media"], "ms |",
              r["direzione_corsa"], "| lato", r["lato_analizzato"])
        for a in r["avvisi"]:
            print("AVVISO:", a)
        return r
    except ValueError as exc:
        print("\nANALISI NON RIUSCITA:", exc)
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python utils/computer_vision.py video.mp4 [altezza_cm]")
    else:
        diagnostica_video(sys.argv[1],
                          altezza_cm=float(sys.argv[2]) if len(sys.argv) > 2 else 175.0)
