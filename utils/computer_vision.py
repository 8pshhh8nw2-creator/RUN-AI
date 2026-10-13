"""
Analisi cinematica 2D della corsa per RUNAI.

Restituisce metriche cinematiche, qualità del tracking e diagnostica
degli scarti. Non restituisce diagnosi, probabilità di infortunio,
forze al suolo o percentuali di carico articolare.

Le soglie e la scala antropometrica sono convenzioni del prototipo:
richiedono validazione su video annotati manualmente.
"""

from collections import Counter
import math

import numpy as np


LATI = ("sx", "dx")
PARTI = ("spalla", "anca", "ginocchio", "caviglia", "tallone", "punta")

FRAZ_COSCIA = 0.245
FRAZ_GAMBA = 0.246

VIS_MIN = 0.5
MAX_GAP_S = 0.2
FILTRO_S = 0.10
DIST_MIN_PASSO_S = 0.40
TOL_SUOLO = 0.03

CONTATTO_MIN_S = 0.10
CONTATTO_MAX_S = 0.45
STRIDE_MIN_S = 0.45
STRIDE_MAX_S = 1.60

MIN_CICLI_VALIDI = 3

FSA_TALLONE = 8.0
FSA_AVAMPIEDE = -1.6

ETICHETTE_APPOGGIO = {
    "tallone": "Appoggio di Tallone (Rearfoot Strike)",
    "mesopiede": "Appoggio di Mesopiede (Midfoot Strike)",
    "avampiede": "Appoggio di Avampiede (Forefoot Strike)",
}


class AnalisiVideoError(ValueError):
    """Errore con diagnostica accessibile dalla pagina Streamlit."""

    def __init__(self, messaggio, diagnostica=None):
        super().__init__(messaggio)
        self.diagnostica = diagnostica or {}


def _savgol(y, finestra, ordine=3):
    y = np.asarray(y, dtype=float)

    finestra = max(int(finestra), ordine + 2)
    if finestra % 2 == 0:
        finestra += 1

    if len(y) < finestra:
        return y.copy()

    m = finestra // 2
    x = np.arange(-m, m + 1, dtype=float)
    matrice = np.vander(x, ordine + 1, increasing=True)
    coefficienti = np.linalg.pinv(matrice)[0]

    padded = np.pad(y, m, mode="edge")
    return np.convolve(
        padded,
        coefficienti[::-1],
        mode="valid",
    )


def _massimi(y, dist_min, soglia):
    y = np.asarray(y, dtype=float)
    dist_min = max(1, int(dist_min))

    candidati = [
        i
        for i in range(1, len(y) - 1)
        if np.isfinite(y[i - 1:i + 2]).all()
        and y[i] >= y[i - 1]
        and y[i] > y[i + 1]
        and y[i] >= soglia
    ]

    candidati.sort(key=lambda i: y[i], reverse=True)
    accettati = []

    for i in candidati:
        if all(abs(i - j) >= dist_min for j in accettati):
            accettati.append(i)

    return sorted(accettati)


def _angolo_incluso(ax, ay, bx, by, cx, cy):
    v1x = ax - bx
    v1y = ay - by
    v2x = cx - bx
    v2y = cy - by

    denominatore = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    denominatore = np.where(
        denominatore > 1e-12,
        denominatore,
        np.nan,
    )

    coseno = np.clip(
        (v1x * v2x + v1y * v2y) / denominatore,
        -1.0,
        1.0,
    )

    return np.degrees(np.arccos(coseno))


def _stat(valori, cifre=1):
    valori = np.asarray(valori, dtype=float)
    valori = valori[np.isfinite(valori)]

    if len(valori) == 0:
        return {"media": None, "sd": None, "n": 0}

    return {
        "media": round(float(valori.mean()), cifre),
        "sd": (
            round(float(valori.std(ddof=1)), cifre)
            if len(valori) > 1
            else None
        ),
        "n": int(len(valori)),
    }


def _intervalli_veri(mask):
    """Intervalli inclusivi in cui una maschera è True."""
    mask = np.asarray(mask, dtype=bool)
    cambi = np.diff(np.r_[False, mask, False].astype(int))
    inizi = np.where(cambi == 1)[0]
    fini = np.where(cambi == -1)[0] - 1
    return list(zip(inizi, fini))


def estrai_landmarks(video_path, model_complexity=1):
    import cv2
    import mediapipe as mp

    if model_complexity not in (0, 1, 2):
        raise ValueError("model_complexity deve essere 0, 1 oppure 2.")

    if not hasattr(mp, "solutions"):
        raise AnalisiVideoError(
            "L'installazione di MediaPipe non espone mp.solutions. "
            "Questo motore usa l'API Pose legacy: verifica la versione "
            "installata oppure adatta il motore all'API disponibile."
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
    righe = []
    timestamp = []
    avvisi = []

    try:
        if not cap.isOpened():
            raise AnalisiVideoError(
                "Impossibile aprire il video. Verifica il file e il codec."
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if not math.isfinite(fps) or fps <= 1:
            raise AnalisiVideoError(
                "FPS mancanti o non validi: non è possibile stimare "
                "i tempi senza una frequenza di campionamento attendibile."
            )

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

                timestamp.append(
                    float(cap.get(cv2.CAP_PROP_POS_MSEC))
                )

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                risultato = pose.process(rgb)

                riga = np.full(
                    (len(chiavi), 3),
                    np.nan,
                    dtype=float,
                )

                if risultato.pose_landmarks is not None:
                    landmarks = risultato.pose_landmarks.landmark

                    for k, chiave in enumerate(chiavi):
                        punto = landmarks[mappa[chiave]]
                        riga[k] = (
                            punto.x * w,
                            punto.y * h,
                            punto.visibility,
                        )

                righe.append(riga)

    finally:
        cap.release()

    if len(righe) < 10:
        raise AnalisiVideoError(
            "Video troppo corto oppure pochi frame decodificabili.",
            {"frame_letti": len(righe)},
        )

    dati = np.stack(righe)
    punti = {
        chiave: dati[:, k, :]
        for k, chiave in enumerate(chiavi)
    }

    ts = np.asarray(timestamp, dtype=float)

    if (
        len(ts) > 1
        and np.isfinite(ts).all()
        and np.all(np.diff(ts) > 0)
    ):
        fps_timestamp = (
            1000.0 * (len(ts) - 1) / (ts[-1] - ts[0])
        )

        if abs(fps_timestamp - fps) / fps > 0.05:
            avvisi.append(
                f"FPS dichiarati: {fps:.2f}; FPS medi dai timestamp: "
                f"{fps_timestamp:.2f}. I tempi sono stimati con gli "
                "FPS dichiarati e richiedono verifica."
            )

    return punti, fps, avvisi


def analizza_serie(
    punti,
    fps,
    altezza_cm=175.0,
    avvisi_iniziali=None,
):
    try:
        fps = float(fps)
        altezza_cm = float(altezza_cm)
    except (TypeError, ValueError) as exc:
        raise ValueError("FPS e altezza devono essere numerici.") from exc

    if not math.isfinite(fps) or fps <= 1:
        raise ValueError("FPS non validi.")

    if not math.isfinite(altezza_cm) or altezza_cm <= 0:
        raise ValueError("Altezza non valida.")

    avvisi = list(avvisi_iniziali or [])

    arrays = {}
    n = None

    for lato in LATI:
        for parte in PARTI:
            chiave = (parte, lato)

            if chiave not in punti:
                raise ValueError(f"Landmark mancante: {chiave}")

            array = np.asarray(punti[chiave], dtype=float)

            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(
                    f"Formato non valido per {chiave}: atteso (N, 3)."
                )

            if n is None:
                n = len(array)

            if len(array) != n:
                raise ValueError(
                    "Le serie dei landmark hanno lunghezze differenti."
                )

            arrays[chiave] = array

    if n is None or n < 10:
        raise ValueError("Serie troppo corta.")

    def maschera_lato(lato):
        valido = np.ones(n, dtype=bool)

        for parte in PARTI:
            array = arrays[(parte, lato)]
            valido &= (
                np.isfinite(array[:, :2]).all(axis=1)
                & np.isfinite(array[:, 2])
                & (array[:, 2] >= VIS_MIN)
            )

        return valido

    maschere = {
        lato: maschera_lato(lato)
        for lato in LATI
    }

    def vis_media(lato):
        valori = np.concatenate([
            arrays[(parte, lato)][:, 2]
            for parte in PARTI
        ])
        return float(np.mean(
            np.where(np.isfinite(valori), valori, 0.0)
        ))

    # Preferisce il lato con più frame completamente utilizzabili.
    lato = max(
        LATI,
        key=lambda l: (
            int(maschere[l].sum()),
            vis_media(l),
        ),
    )

    P = {
        parte: arrays[(parte, lato)]
        for parte in PARTI
    }
    valido = maschere[lato]
    percentuale_tracking = float(valido.mean() * 100)

    diagnostica_base = {
        "frame_totali": int(n),
        "fps_video": round(fps, 2),
        "lato": lato,
        "tracking_valido_pct": round(percentuale_tracking, 1),
        "tracking_per_lato_pct": {
            l: round(float(maschere[l].mean() * 100), 1)
            for l in LATI
        },
        "visibilita_minima": VIS_MIN,
    }

    if valido.mean() < 0.5:
        raise AnalisiVideoError(
            f"Tracking completo valido nel {percentuale_tracking:.1f}% "
            "dei frame: inferiore al minimo configurato del 50%.",
            diagnostica_base,
        )

    if valido.mean() < 0.8:
        avvisi.append(
            f"Tracking completo valido nel {percentuale_tracking:.1f}% "
            "dei frame."
        )

    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    inaffidabile = np.zeros(n, dtype=bool)

    for inizio, fine in _intervalli_veri(~valido):
        if fine - inizio + 1 > max_gap:
            inaffidabile[inizio:fine + 1] = True

    indici_validi = np.flatnonzero(valido)
    primo = int(indici_validi[0])
    ultimo = int(indici_validi[-1])

    inaffidabile[:primo] = True
    inaffidabile[ultimo + 1:] = True

    # Espande l'esclusione vicino ai buchi lunghi per evitare
    # che lo smoothing renda utilizzabili i loro bordi.
    finestra = max(5, int(round(FILTRO_S * fps)))
    if finestra % 2 == 0:
        finestra += 1

    margine = finestra // 2
    esclusione = inaffidabile.copy()

    for inizio, fine in _intervalli_veri(inaffidabile):
        esclusione[
            max(0, inizio - margine):
            min(n, fine + margine + 1)
        ] = True

    idx = np.arange(n)
    S = {}

    for parte in PARTI:
        for coordinata, nome in ((0, "x"), (1, "y")):
            interpolata = np.interp(
                idx,
                idx[valido],
                P[parte][valido, coordinata],
            )
            S[(parte, nome)] = _savgol(
                interpolata,
                finestra,
            )

    direzione = (
        1.0
        if np.median(
            (
                P["punta"][:, 0]
                - P["tallone"][:, 0]
            )[valido]
        ) >= 0
        else -1.0
    )

    coscia = np.hypot(
        S[("anca", "x")] - S[("ginocchio", "x")],
        S[("anca", "y")] - S[("ginocchio", "y")],
    )

    gamba = np.hypot(
        S[("ginocchio", "x")] - S[("caviglia", "x")],
        S[("ginocchio", "y")] - S[("caviglia", "y")],
    )

    lunghezza_arto = float(np.median(
        (coscia + gamba)[valido]
    ))

    if not math.isfinite(lunghezza_arto) or lunghezza_arto <= 1e-6:
        raise AnalisiVideoError(
            "Scala non calcolabile: lunghezza dell'arto non valida.",
            diagnostica_base,
        )

    scala = (
        (FRAZ_COSCIA + FRAZ_GAMBA)
        * altezza_cm
        / lunghezza_arto
    )

    xrel = direzione * (
        S[("caviglia", "x")] - S[("anca", "x")]
    )

    angolo_incluso = _angolo_incluso(
        S[("anca", "x")],
        S[("anca", "y")],
        S[("ginocchio", "x")],
        S[("ginocchio", "y")],
        S[("caviglia", "x")],
        S[("caviglia", "y")],
    )

    flessione = 180.0 - angolo_incluso

    tibia = np.degrees(np.arctan2(
        direzione * (
            S[("caviglia", "x")] - S[("ginocchio", "x")]
        ),
        S[("caviglia", "y")] - S[("ginocchio", "y")],
    ))

    busto = np.degrees(np.arctan2(
        direzione * (
            S[("spalla", "x")] - S[("anca", "x")]
        ),
        S[("anca", "y")] - S[("spalla", "y")],
    ))

    fsa = np.degrees(np.arctan2(
        S[("tallone", "y")] - S[("punta", "y")],
        direzione * (
            S[("punta", "x")] - S[("tallone", "x")]
        ),
    ))

    y_piede = np.maximum(
        S[("tallone", "y")],
        S[("punta", "y")],
    )

    # Cerca i massimi separatamente nei tratti non esclusi:
    # non individua eventi dentro buchi lunghi interpolati.
    fm = []
    dist_min = max(1, int(round(DIST_MIN_PASSO_S * fps)))

    for inizio, fine in _intervalli_veri(~esclusione):
        segmento = xrel[inizio:fine + 1]

        if len(segmento) < 3:
            continue

        locali = _massimi(
            segmento,
            dist_min,
            float(np.mean(segmento)),
        )
        fm.extend(inizio + i for i in locali)

    fm = sorted(fm)

    if len(fm) < MIN_CICLI_VALIDI + 1:
        diagnostica_base["massimi_rilevati"] = len(fm)
        raise AnalisiVideoError(
            f"Solo {len(fm)} massimi cinematici rilevati. "
            f"Servono almeno {MIN_CICLI_VALIDI + 1} eventi "
            f"per valutare {MIN_CICLI_VALIDI} cicli.",
            diagnostica_base,
        )

    tolleranza = TOL_SUOLO * lunghezza_arto
    pre = max(1, int(round(0.03 * fps)))
    post = max(1, int(round(0.25 * fps)))

    ic = []
    to = []

    for k, massimo in enumerate(fm):
        fine = (
            fm[k + 1]
            if k + 1 < len(fm)
            else min(
                n - 1,
                massimo + int(round(STRIDE_MAX_S * fps)),
            )
        )

        suolo = float(np.percentile(
            y_piede[massimo:fine + 1],
            95,
        ))

        a = max(0, massimo - pre)
        b = min(fine, massimo + post)

        candidati_ic = np.flatnonzero(
            y_piede[a:b + 1] >= suolo - tolleranza
        )

        ic_k = (
            a + int(candidati_ic[0])
            if len(candidati_ic)
            else None
        )

        to_k = None

        if ic_k is not None:
            a2 = ic_k + max(1, int(round(0.08 * fps)))

            # Non cerca lo stacco oltre il prossimo massimo.
            b2 = min(
                fine,
                n - 1,
                ic_k + int(round(0.5 * fps)),
            )

            if a2 <= b2:
                candidati_to = np.flatnonzero(
                    y_piede[a2:b2 + 1]
                    < suolo - 2 * tolleranza
                )

                if len(candidati_to):
                    to_k = a2 + int(candidati_to[0])

        ic.append(ic_k)
        to.append(to_k)

    metriche = {
        nome: []
        for nome in (
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
    }

    ic_validi = []
    dettagli_cicli = []
    motivi_scarto = Counter()

    for k in range(len(fm) - 1):
        i0 = ic[k]
        i1 = ic[k + 1]
        t0 = to[k]

        dettaglio = {
            "ciclo": k + 1,
            "massimo_frame": int(fm[k]),
            "ic_frame": i0,
            "to_frame": t0,
            "ic_successivo_frame": i1,
            "durata_ciclo_s": None,
            "contatto_ms": None,
            "valido": False,
            "motivi_scarto": [],
        }

        motivi = []

        if i0 is None:
            motivi.append("contatto iniziale non rilevato")
        if i1 is None:
            motivi.append("contatto successivo non rilevato")
        if t0 is None:
            motivi.append("stacco non rilevato")

        if not motivi:
            if not (0 <= i0 < t0 < i1 < n):
                motivi.append("ordine degli eventi non valido")
            else:
                durata = (i1 - i0) / fps
                contatto = (t0 - i0) / fps

                dettaglio["durata_ciclo_s"] = round(durata, 4)
                dettaglio["contatto_ms"] = round(contatto * 1000, 1)

                if esclusione[i0:i1 + 1].any():
                    motivi.append("tracking inaffidabile nel ciclo")

                if not (STRIDE_MIN_S <= durata <= STRIDE_MAX_S):
                    motivi.append("durata del ciclo fuori intervallo")

                if not (
                    CONTATTO_MIN_S
                    <= contatto
                    <= CONTATTO_MAX_S
                ):
                    motivi.append("tempo di contatto fuori intervallo")

                serie_controllate = [
                    flessione[i0:i1 + 1],
                    busto[i0:i1 + 1],
                    tibia[i0:i1 + 1],
                    fsa[i0:i1 + 1],
                ]

                if not all(
                    np.isfinite(s).all()
                    for s in serie_controllate
                ):
                    motivi.append("metriche angolari non finite")

        if motivi:
            dettaglio["motivi_scarto"] = motivi
            motivi_scarto.update(motivi)
            dettagli_cicli.append(dettaglio)
            continue

        dettaglio["valido"] = True
        dettagli_cicli.append(dettaglio)
        ic_validi.append(int(i0))

        metriche["stride"].append(durata)
        metriche["contatto"].append(contatto * 1000)
        metriche["over_cm"].append(xrel[i0] * scala)
        metriche["over_pct"].append(
            xrel[i0] / lunghezza_arto * 100
        )
        metriche["tibia"].append(tibia[i0])
        metriche["flex_ic"].append(flessione[i0])
        metriche["flex_stance"].append(
            float(np.max(flessione[i0:t0 + 1]))
        )
        metriche["flex_to"].append(flessione[t0])
        metriche["flex_swing"].append(
            float(np.max(flessione[t0:i1 + 1]))
        )
        metriche["fsa"].append(fsa[i0])
        metriche["busto"].append(
            float(np.mean(busto[i0:i1 + 1]))
        )

        y_anca = S[("anca", "y")][i0:i1 + 1]
        deriva = np.linspace(
            y_anca[0],
            y_anca[-1],
            len(y_anca),
        )

        metriche["vo"].append(
            float(np.ptp(y_anca - deriva) * scala)
        )

    n_validi = len(ic_validi)
    n_candidati = len(dettagli_cicli)
    n_scartati = n_candidati - n_validi

    diagnostica = {
        **diagnostica_base,
        "massimi_rilevati": len(fm),
        "cicli_candidati": n_candidati,
        "cicli_validi": n_validi,
        "cicli_scartati": n_scartati,
        "motivi_scarto": dict(motivi_scarto),
        "dettagli_cicli": dettagli_cicli,
        "soglie": {
            "ciclo_min_s": STRIDE_MIN_S,
            "ciclo_max_s": STRIDE_MAX_S,
            "contatto_min_s": CONTATTO_MIN_S,
            "contatto_max_s": CONTATTO_MAX_S,
            "min_cicli_validi": MIN_CICLI_VALIDI,
        },
    }

    if n_validi < MIN_CICLI_VALIDI:
        riepilogo = "; ".join(
            f"{motivo}: {conteggio}"
            for motivo, conteggio
            in motivi_scarto.most_common()
        )

        raise AnalisiVideoError(
            f"Solo {n_validi} cicli validi su {n_candidati}. "
            f"Tracking completo: {percentuale_tracking:.1f}%. "
            f"FPS: {fps:.1f}. "
            f"Motivi degli scarti: {riepilogo}. "
            "Un ciclo può avere più motivi di scarto.",
            diagnostica,
        )

    if n_validi < 8:
        avvisi.append(
            f"Risultati basati su {n_validi} cicli dello stesso piede: "
            "campione limitato."
        )

    if fps < 60:
        avvisi.append(
            f"Risoluzione nominale: {1000 / fps:.1f} ms per frame. "
            "I tempi di contatto e gli eventi richiedono verifica."
        )

    avvisi.append(
        "Angoli proiettati sul piano immagine. "
        "La scala in cm è una stima antropometrica, "
        "non una calibrazione metrica del video."
    )

    classi = [
        (
            "tallone"
            if valore > FSA_TALLONE
            else "avampiede"
            if valore < FSA_AVAMPIEDE
            else "mesopiede"
        )
        for valore in metriche["fsa"]
    ]

    conteggio = Counter(classi)
    classe_prevalente = conteggio.most_common(1)[0][0]

    distribuzione = {
        classe: round(
            100 * conteggio.get(classe, 0) / n_validi,
            1,
        )
        for classe in ("tallone", "mesopiede", "avampiede")
    }

    if sum(
        conteggio[c] == conteggio[classe_prevalente]
        for c in conteggio
    ) > 1:
        avvisi.append(
            "Parità nella classificazione dell'appoggio: "
            "consulta la distribuzione, non solo l'etichetta."
        )

    fasi = [
        "Contatto iniziale",
        "Max flessione (appoggio)",
        "Stacco",
        "Max flessione (oscillazione)",
    ]

    statistiche_flessione = [
        _stat(metriche["flex_ic"]),
        _stat(metriche["flex_stance"]),
        _stat(metriche["flex_to"]),
        _stat(metriche["flex_swing"]),
    ]

    # Usa un contatto realmente osservato, non la mediana
    # numerica di due frame differenti.
    frame_rappresentativo = ic_validi[len(ic_validi) // 2]

    scheletro_frame = {
        parte: {
            "x": round(float(S[(parte, "x")][frame_rappresentativo]), 2),
            "y": round(float(S[(parte, "y")][frame_rappresentativo]), 2),
        }
        for parte in PARTI
    }

    return {
        "lato_analizzato": (
            "Sinistro" if lato == "sx" else "Destro"
        ),
        "direzione_corsa": (
            "verso destra" if direzione > 0 else "verso sinistra"
        ),
        "fps_video": round(fps, 1),
        "frame_strike_analizzato": frame_rappresentativo,
        "frame_contatto": ic_validi,
        "scheletro_frame": scheletro_frame,
        "qualita": {
            "frame_totali": int(n),
            "frame_tracking_valido_pct": round(
                percentuale_tracking,
                1,
            ),
            "cicli_validi": n_validi,
            "cicli_scartati": n_scartati,
            # Alias per eventuali pagine già esistenti.
            "passi_validi": n_validi,
            "passi_scartati": n_scartati,
            "risoluzione_temporale_ms": round(1000 / fps, 1),
            "scala_cm_per_pixel": round(scala, 4),
        },
        "diagnostica": diagnostica,
        "avvisi": avvisi,
        "cadenza_spm": round(
            120 / float(np.mean(metriche["stride"])),
            1,
        ),
        "tempo_contatto_ms": _stat(metriche["contatto"], 0),
        "overstride_cm": round(
            float(np.mean(metriche["over_cm"])),
            1,
        ),
        "overstride": {
            "cm": _stat(metriche["over_cm"]),
            "pct_lunghezza_arto": _stat(metriche["over_pct"]),
        },
        "angolo_tibia_contatto_gradi": _stat(metriche["tibia"]),
        "flessione_ginocchio_gradi": dict(
            zip(fasi, statistiche_flessione)
        ),
        "angolo_ginocchio_appoggio": round(
            180 - float(np.mean(metriche["flex_ic"])),
            1,
        ),
        "angolo_inclinazione_busto": round(
            float(np.mean(metriche["busto"])),
            1,
        ),
        "inclinazione_busto_gradi": _stat(metriche["busto"]),
        "oscillazione_verticale": round(
            float(np.mean(metriche["vo"])),
            1,
        ),
        "oscillazione_verticale_cm": _stat(metriche["vo"]),
        "foot_strike_angle_gradi": _stat(metriche["fsa"]),
        "tipo_appoggio": ETICHETTE_APPOGGIO[classe_prevalente],
        "distribuzione_appoggio_pct": distribuzione,
        "fasi_gait": fasi,
        "angoli_fase": [
            round(180 - s["media"], 1)
            for s in statistiche_flessione
        ],
        "metriche_per_ciclo": {
            nome: [float(v) for v in valori]
            for nome, valori in metriche.items()
        },
        "note_metodologiche": [
            "IC e TO sono eventi stimati da soglie cinematiche.",
            "Overstride: distanza orizzontale caviglia-anca all'IC.",
            "Flessione: 180° meno l'angolo anca-ginocchio-caviglia.",
            "Cadenza ricavata dai cicli dello stesso piede, "
            "assumendo due passi per ciclo.",
            "Classificazione dell'appoggio basata sulle soglie "
            "FSA configurate nel prototipo.",
            "Nessuna probabilità di infortunio o forza articolare "
            "viene calcolata.",
        ],
    }


def analizza_running_video(
    video_path,
    altezza_cm=175,
    model_complexity=1,
):
    punti, fps, avvisi = estrai_landmarks(
        video_path,
        model_complexity=model_complexity,
    )

    return analizza_serie(
        punti,
        fps,
        altezza_cm=altezza_cm,
        avvisi_iniziali=avvisi,
    )
