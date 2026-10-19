"""
utils/computer_vision.py
------------------------
Analisi cinematica 2D della corsa (ripresa laterale) per RUNAI.

Il modulo estrae lo scheletro con MediaPipe Pose e misura, PASSO PER PASSO,
parametri cinematici con definizioni esplicite. I risultati sono riportati
come media ± deviazione standard sul numero di passi validi.

COSA FA
    - contatto iniziale (IC) e stacco (TO) di ogni passo, da cinematica
    - overstride (distanza orizzontale caviglia-anca all'IC, con segno)
    - angolo della tibia rispetto alla verticale all'IC (indipendente dalla scala)
    - flessione del ginocchio: IC, massimo in appoggio, TO, massimo in oscillazione
    - foot strike angle (FSA) all'IC e classificazione dell'appoggio
    - cadenza, tempo di contatto, inclinazione del busto, oscillazione verticale

COSA NON FA (volutamente)
    Non produce probabilità di infortunio, punteggi di rischio per distretto
    o percentuali di "carico articolare". Nessuno di questi valori è ricavabile
    da un video 2D senza un modello validato su dati clinici: sarebbero numeri
    inventati. Per il rischio usa i carichi di allenamento (TRIMP, sRPE Load).

RIFERIMENTI (da verificare sulle fonti originali prima di citarli)
    - Winter, Biomechanics and Motor Control of Human Movement (proporzioni
      dei segmenti: coscia 0,245 H, gamba 0,246 H)
    - Altman & Davis 2012, Gait & Posture (soglie foot strike angle:
      > 8° tallone; da -1,6° a 8° mesopiede; < -1,6° avampiede)
    - Fellin et al. 2010, J Sci Med Sport (individuazione cinematica di IC/TO)
    - Lieberman et al. 2015, J Exp Biol (posizione del piede all'atterraggio
      e forza frenante)
    - Heiderscheit et al. 2011, Med Sci Sports Exerc (frequenza di passo e
      carichi articolari)

LIMITI
    Analisi 2D: gli angoli sono proiezioni sul piano immagine e sono corretti
    solo se la camera è perpendicolare alla direzione di corsa, ferma e
    orizzontale. La scala in cm dipende dall'altezza dichiarata e dalle
    proporzioni medie dei segmenti: è indicativa (circa ±10%). La risoluzione
    temporale è 1/fps: a 30 fps l'IC ha un'incertezza di circa ±33 ms.
    I parametri di rilevamento sono costanti in cima al file e vanno validati
    su un campione di video confrontandoli con misure manuali (es. Kinovea).
"""

import math
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# Costanti (da validare)
# ----------------------------------------------------------------------------
LATI = ("sx", "dx")
PARTI = ("spalla", "anca", "ginocchio", "caviglia", "tallone", "punta")

FRAZ_COSCIA = 0.245      # lunghezza coscia / altezza (Winter)
FRAZ_GAMBA = 0.246       # lunghezza gamba (ginocchio-caviglia) / altezza (Winter)

VIS_MIN = 0.5            # visibilità minima MediaPipe per usare un landmark
MAX_GAP_S = 0.2          # buchi di tracking più lunghi rendono inaffidabile il passo
FILTRO_S = 0.10          # ampiezza finestra Savitzky-Golay (secondi)
DIST_MIN_PASSO_S = 0.40  # distanza minima tra due passi dello stesso piede
TOL_SUOLO = 0.03         # tolleranza "piede a terra" (frazione lunghezza arto)
CONTATTO_MIN_S, CONTATTO_MAX_S = 0.10, 0.45
STRIDE_MIN_S, STRIDE_MAX_S = 0.45, 1.60
FSA_TALLONE, FSA_AVAMPIEDE = 8.0, -1.6

ETICHETTE_APPOGGIO = {
    "tallone": "Appoggio di Tallone (Rearfoot Strike)",
    "mesopiede": "Appoggio di Mesopiede (Midfoot Strike)",
    "avampiede": "Appoggio di Avampiede (Forefoot Strike)",
}


# ----------------------------------------------------------------------------
# Utilità numeriche (solo numpy)
# ----------------------------------------------------------------------------
def _savgol(y, finestra, ordine=3):
    """Filtro di Savitzky-Golay (smoothing), bordi replicati."""
    finestra = int(finestra)
    if finestra % 2 == 0:
        finestra += 1
    finestra = max(finestra, ordine + 2 + ((ordine + 2) % 2 == 0))
    if len(y) < finestra:
        return np.asarray(y, float)
    m = finestra // 2
    x = np.arange(-m, m + 1)
    a = np.vander(x, ordine + 1, increasing=True)
    coef = np.linalg.pinv(a)[0]
    ypad = np.pad(np.asarray(y, float), m, mode="edge")
    return np.convolve(ypad, coef[::-1], mode="valid")


def _massimi(y, dist_min, soglia):
    """Massimi locali sopra soglia, a distanza minima (si tengono i più alti)."""
    cand = [i for i in range(1, len(y) - 1)
            if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= soglia]
    cand.sort(key=lambda i: y[i], reverse=True)
    accettati = []
    for i in cand:
        if all(abs(i - j) >= dist_min for j in accettati):
            accettati.append(i)
    return sorted(accettati)


def _angolo_incluso(ax, ay, bx, by, cx, cy):
    """Angolo (gradi) al vertice b tra b->a e b->c, vettorializzato."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    den = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    den = np.where(den == 0, np.nan, den)
    cosang = np.clip((v1x * v2x + v1y * v2y) / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _stat(valori, cifre=1):
    v = np.asarray(valori, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"media": None, "sd": None, "n": 0}
    return {
        "media": round(float(v.mean()), cifre),
        "sd": round(float(v.std(ddof=1)), cifre) if len(v) > 1 else 0.0,
        "n": int(len(v)),
    }


# ----------------------------------------------------------------------------
# Estrazione dei landmark dal video (unica parte che richiede cv2/mediapipe)
# ----------------------------------------------------------------------------
def estrai_landmarks(video_path, model_complexity=1):
    """
    Legge il video e restituisce:
        punti[(parte, lato)] -> array (N, 3) con x, y (pixel) e visibilità;
        i frame senza scheletro restano NaN, così l'indice = numero di frame.
    """
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    lm = mp_pose.PoseLandmark
    mappa = {
        ("spalla", "sx"): lm.LEFT_SHOULDER, ("spalla", "dx"): lm.RIGHT_SHOULDER,
        ("anca", "sx"): lm.LEFT_HIP, ("anca", "dx"): lm.RIGHT_HIP,
        ("ginocchio", "sx"): lm.LEFT_KNEE, ("ginocchio", "dx"): lm.RIGHT_KNEE,
        ("caviglia", "sx"): lm.LEFT_ANKLE, ("caviglia", "dx"): lm.RIGHT_ANKLE,
        ("tallone", "sx"): lm.LEFT_HEEL, ("tallone", "dx"): lm.RIGHT_HEEL,
        ("punta", "sx"): lm.LEFT_FOOT_INDEX, ("punta", "dx"): lm.RIGHT_FOOT_INDEX,
    }
    chiavi = list(mappa.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossibile aprire il file video. Verifica il formato (MP4/MOV/AVI).")

    fps = cap.get(cv2.CAP_PROP_FPS)
    avvisi = []
    if not fps or fps <= 1:
        fps = 30.0
        avvisi.append("FPS non presenti nel file: assunti 30. Cadenza e tempi di contatto potrebbero essere errati.")

    righe, timestamp = [], []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            h, w = frame.shape[:2]
            ris = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            riga = np.full((len(chiavi), 3), np.nan)
            if ris.pose_landmarks is not None:
                pts = ris.pose_landmarks.landmark
                for k, chiave in enumerate(chiavi):
                    p = pts[mappa[chiave]]
                    riga[k] = (p.x * w, p.y * h, p.visibility)
            righe.append(riga)
    cap.release()

    if len(righe) < 10:
        raise ValueError("Video troppo corto o non leggibile.")

    dati = np.stack(righe)  # (N, 12, 3)
    punti = {chiave: dati[:, k, :] for k, chiave in enumerate(chiavi)}

    ts = np.asarray(timestamp, float)
    if np.all(np.diff(ts) > 0):
        fps_ts = 1000.0 * (len(ts) - 1) / (ts[-1] - ts[0])
        if abs(fps_ts - fps) / fps > 0.05:
            avvisi.append(
                f"FPS dichiarati ({fps:.1f}) e FPS dai timestamp ({fps_ts:.1f}) differiscono: "
                "video a frame rate variabile? Verifica i tempi (cadenza, contatto)."
            )
    return punti, float(fps), avvisi


# ----------------------------------------------------------------------------
# Analisi (pura: lavora su array, testabile senza video)
# ----------------------------------------------------------------------------
def analizza_serie(punti, fps, altezza_cm=175.0, avvisi_iniziali=None):
    avvisi = list(avvisi_iniziali or [])

    # --- lato più visibile ---
    def vis_media(lato):
        return float(np.mean([np.nan_to_num(punti[(p, lato)][:, 2], nan=0.0) for p in PARTI]))

    lato = "sx" if vis_media("sx") >= vis_media("dx") else "dx"
    P = {p: punti[(p, lato)] for p in PARTI}
    n = P["anca"].shape[0]

    valido = np.ones(n, bool)
    for p in PARTI:
        valido &= ~np.isnan(P[p][:, 0]) & (np.nan_to_num(P[p][:, 2], nan=0.0) >= VIS_MIN)
    if valido.mean() < 0.5:
        raise ValueError(
            "Il corridore è tracciato in meno del 50% dei frame. Usa un video ripreso "
            "lateralmente, con il corridore intero e ben illuminato."
        )
    if valido.mean() < 0.8:
        avvisi.append(f"Tracking valido solo nel {valido.mean() * 100:.0f}% dei frame: risultati meno affidabili.")

    # --- frame inaffidabili: buchi lunghi e code del video ---
    max_gap = int(round(MAX_GAP_S * fps))
    inaff = np.zeros(n, bool)
    i = 0
    while i < n:
        if valido[i]:
            i += 1
            continue
        j = i
        while j < n and not valido[j]:
            j += 1
        if (j - i) > max_gap:
            inaff[i:j] = True
        i = j
    prima, ultima = np.argmax(valido), n - 1 - np.argmax(valido[::-1])
    inaff[:prima] = True
    inaff[ultima + 1:] = True

    # --- interpolazione buchi corti e filtro ---
    idx = np.arange(n)
    finestra = max(5, int(round(FILTRO_S * fps)))
    S = {}
    for p in PARTI:
        for c, nome in ((0, "x"), (1, "y")):
            S[(p, nome)] = _savgol(np.interp(idx, idx[valido], P[p][valido, c]), finestra)

    # --- direzione di corsa: verso in cui punta il piede ---
    direzione = 1.0 if np.median((P["punta"][:, 0] - P["tallone"][:, 0])[valido]) >= 0 else -1.0

    # --- scala pixel -> cm da lunghezza coscia + gamba ---
    coscia = np.hypot(S[("anca", "x")] - S[("ginocchio", "x")], S[("anca", "y")] - S[("ginocchio", "y")])
    gamba = np.hypot(S[("ginocchio", "x")] - S[("caviglia", "x")], S[("ginocchio", "y")] - S[("caviglia", "y")])
    lung_arto_px = float(np.median((coscia + gamba)[valido]))
    scala = (FRAZ_COSCIA + FRAZ_GAMBA) * altezza_cm / lung_arto_px  # cm per pixel

    # --- serie derivate ---
    xrel = direzione * (S[("caviglia", "x")] - S[("anca", "x")])          # + = piede davanti all'anca
    incluso = _angolo_incluso(S[("anca", "x")], S[("anca", "y")], S[("ginocchio", "x")],
                              S[("ginocchio", "y")], S[("caviglia", "x")], S[("caviglia", "y")])
    flex = 180.0 - incluso                                                # flessione del ginocchio
    tibia = np.degrees(np.arctan2(direzione * (S[("caviglia", "x")] - S[("ginocchio", "x")]),
                                  S[("caviglia", "y")] - S[("ginocchio", "y")]))  # + = piede davanti al ginocchio
    busto = np.degrees(np.arctan2(direzione * (S[("spalla", "x")] - S[("anca", "x")]),
                                  S[("anca", "y")] - S[("spalla", "y")]))          # + = inclinato in avanti
    fsa = np.degrees(np.arctan2(S[("tallone", "y")] - S[("punta", "y")],
                                direzione * (S[("punta", "x")] - S[("tallone", "x")])))  # + = punta sopra il tallone
    y_piede = np.maximum(S[("tallone", "y")], S[("punta", "y")])          # punto più basso del piede

    # --- passi: massimi di posizione in avanti del piede rispetto all'anca ---
    fm = _massimi(xrel, int(round(DIST_MIN_PASSO_S * fps)), float(np.mean(xrel)))
    if len(fm) < 3:
        raise ValueError(
            "Passi rilevati insufficienti. Servono almeno 5-10 secondi di corsa ripresa "
            "lateralmente, con la camera ferma."
        )

    tol = TOL_SUOLO * lung_arto_px
    pre, post = int(round(0.03 * fps)), int(round(0.25 * fps))
    ic, to = [], []
    for k, f in enumerate(fm):
        fine = fm[k + 1] if k + 1 < len(fm) else min(n - 1, f + int(round(1.0 * fps)))
        suolo = np.percentile(y_piede[f:fine + 1], 95)
        a, b = max(0, f - pre), min(fine, f + post)
        c = np.where(y_piede[a:b + 1] >= suolo - tol)[0]
        ic_k = a + int(c[0]) if len(c) else None
        to_k = None
        if ic_k is not None:
            a2, b2 = ic_k + int(round(0.08 * fps)), min(n - 1, ic_k + int(round(0.5 * fps)))
            if a2 <= b2:
                c2 = np.where(y_piede[a2:b2 + 1] < suolo - 2 * tol)[0]
                to_k = a2 + int(c2[0]) if len(c2) else None
        ic.append(ic_k)
        to.append(to_k)

    # --- metriche per passo ---
    m = {k: [] for k in ("stride", "contatto", "over_cm", "over_pct", "tibia", "flex_ic",
                         "flex_stance", "flex_to", "flex_swing", "fsa", "busto", "vo")}
    ic_validi, scartati = [], 0
    for k in range(len(fm) - 1):
        i0, i1, t0 = ic[k], ic[k + 1], to[k]
        if i0 is None or i1 is None or t0 is None or not (i0 < t0 < i1) or i1 >= n:
            scartati += 1
            continue
        t_stride, t_cont = (i1 - i0) / fps, (t0 - i0) / fps
        if (inaff[i0:i1 + 1].any() or not (STRIDE_MIN_S <= t_stride <= STRIDE_MAX_S)
                or not (CONTATTO_MIN_S <= t_cont <= CONTATTO_MAX_S)):
            scartati += 1
            continue
        ic_validi.append(i0)
        m["stride"].append(t_stride)
        m["contatto"].append(t_cont * 1000.0)
        m["over_cm"].append(xrel[i0] * scala)
        m["over_pct"].append(xrel[i0] / lung_arto_px * 100.0)
        m["tibia"].append(tibia[i0])
        m["flex_ic"].append(flex[i0])
        m["flex_stance"].append(np.nanmax(flex[i0:t0 + 1]))
        m["flex_to"].append(flex[t0])
        m["flex_swing"].append(np.nanmax(flex[t0:i1 + 1]))
        m["fsa"].append(fsa[i0])
        m["busto"].append(np.mean(busto[i0:i1 + 1]))
        y_hip = S[("anca", "y")][i0:i1 + 1]
        trend = np.linspace(y_hip[0], y_hip[-1], len(y_hip))   # toglie pendenza suolo / deriva camera
        m["vo"].append((np.max(y_hip - trend) - np.min(y_hip - trend)) * scala)

    n_validi = len(ic_validi)
    if n_validi < 3:
        raise ValueError(
            f"Solo {n_validi} passi validi su {len(fm) - 1}. Migliora la ripresa "
            "(laterale, camera ferma e perpendicolare, corridore intero)."
        )
    if n_validi < 8:
        avvisi.append(f"Solo {n_validi} passi validi: le stime sono poco stabili (consigliati almeno 10).")
    if fps < 60:
        avvisi.append(f"A {fps:.0f} fps la risoluzione temporale è di {1000 / fps:.0f} ms: "
                      "IC e stacco sono poco precisi. Meglio 120-240 fps (slow motion).")
    avvisi.append("Analisi 2D: valida se la camera è perpendicolare alla corsa. "
                  "Scala in cm indicativa (circa ±10%).")

    # --- classificazione appoggio (moda sui passi) ---
    classi = ["tallone" if v > FSA_TALLONE else "avampiede" if v < FSA_AVAMPIEDE else "mesopiede"
              for v in m["fsa"]]
    conteggio = Counter(classi)
    classe_prevalente = conteggio.most_common(1)[0][0]
    distribuzione = {c: round(100.0 * conteggio.get(c, 0) / n_validi, 1)
                     for c in ("tallone", "mesopiede", "avampiede")}

    tempo_stride_medio = float(np.mean(m["stride"]))
    cadenza = 120.0 / tempo_stride_medio  # passi/min (2 passi per ciclo dello stesso piede)

    flex_fasi = [_stat(m["flex_ic"]), _stat(m["flex_stance"]), _stat(m["flex_to"]), _stat(m["flex_swing"])]
    fasi = ["Contatto iniziale", "Max flessione (appoggio)", "Stacco", "Max flessione (oscillazione)"]

    return {
        # ---- metadati e qualità ----
        "lato_analizzato": "Sinistro" if lato == "sx" else "Destro",
        "direzione_corsa": "verso destra" if direzione > 0 else "verso sinistra",
        "fps_video": round(float(fps), 1),
        "frame_strike_analizzato": int(np.median(ic_validi)),
        "frame_contatto": [int(v) for v in ic_validi],
        "qualita": {
            "frame_totali": int(n),
            "frame_tracking_valido_pct": round(float(valido.mean()) * 100, 1),
            "passi_validi": int(n_validi),
            "passi_scartati": int(scartati),
            "risoluzione_temporale_ms": round(1000.0 / fps, 1),
            "scala_cm_per_pixel": round(float(scala), 4),
        },
        "avvisi": avvisi,
        # ---- metriche (media ± SD sui passi) ----
        "cadenza_spm": round(cadenza, 1),
        "tempo_contatto_ms": _stat(m["contatto"], 0),
        "overstride_cm": round(float(np.mean(m["over_cm"])), 1),          # compatibilità pagina
        "overstride": {"cm": _stat(m["over_cm"]), "pct_lunghezza_arto": _stat(m["over_pct"])},
        "angolo_tibia_contatto_gradi": _stat(m["tibia"]),
        "flessione_ginocchio_gradi": dict(zip(fasi, flex_fasi)),
        "angolo_ginocchio_appoggio": round(180.0 - float(np.mean(m["flex_ic"])), 1),  # angolo incluso all'IC
        "angolo_inclinazione_busto": round(float(np.mean(m["busto"])), 1),            # + = in avanti
        "inclinazione_busto_gradi": _stat(m["busto"]),
        "oscillazione_verticale": round(float(np.mean(m["vo"])), 1),                   # cm, compatibilità
        "oscillazione_verticale_cm": _stat(m["vo"]),
        "foot_strike_angle_gradi": _stat(m["fsa"]),
        "tipo_appoggio": ETICHETTE_APPOGGIO[classe_prevalente],
        "distribuzione_appoggio_pct": distribuzione,
        # ---- per i grafici delle fasi (angolo incluso, come nella versione precedente) ----
        "fasi_gait": fasi,
        "angoli_fase": [round(180.0 - s["media"], 1) for s in flex_fasi],
        "note_metodologiche": [
            "IC/TO da cinematica: IC = primo frame con il punto più basso del piede vicino al suolo "
            "dopo il massimo avanzamento della caviglia rispetto all'anca; TO = risalita del piede.",
            "Overstride = distanza orizzontale caviglia-anca all'IC (positiva = piede davanti).",
            "Flessione ginocchio = 180° - angolo anca-ginocchio-caviglia (0° = gamba estesa).",
            "FSA: > 8° tallone, da -1,6° a 8° mesopiede, < -1,6° avampiede (Altman & Davis 2012).",
            "Nessun indice di rischio infortunio: non è stimabile in modo valido da un video 2D.",
        ],
    }


def analizza_running_video(video_path, altezza_cm=175, model_complexity=1):
    """Analizza un video di corsa (profilo laterale) e restituisce il dizionario di metriche."""
    punti, fps, avvisi = estrai_landmarks(video_path, model_complexity=model_complexity)
    return analizza_serie(punti, fps, altezza_cm, avvisi_iniziali=avvisi)
