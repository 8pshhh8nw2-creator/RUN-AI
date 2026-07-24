"""
utils/computer_vision.py
-------------------------
Modulo di Computer Vision per RUNAI.

Estrae lo scheletro posturale da un video di corsa (profilo laterale)
tramite MediaPipe Pose, calcola gli angoli articolari, l'overstride,
l'oscillazione verticale e produce una stima euristica del rischio
di infortunio.

NOTA METODOLOGICA:
Il punteggio di rischio ("probabilita_infortunio_ml") è un indice
euristico costruito su feature biomeccaniche reali (angolo ginocchio,
overstride, oscillazione verticale, inclinazione busto), NON un modello
di Machine Learning addestrato su un dataset clinico di infortuni.
Va presentato in tesi come tale.
"""

import math
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Indici dei landmark MediaPipe Pose che ci servono
LM = mp_pose.PoseLandmark


def _angolo_tra_punti(a, b, c):
    """Angolo (in gradi) al vertice b, tra i segmenti b->a e b->c, nel piano immagine (x,y)."""
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cos_ang = np.dot(ba, bc) / denom
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))


def _angolo_rispetto_verticale(p_alto, p_basso):
    """Angolo (in gradi) tra il segmento p_alto->p_basso e la verticale."""
    dx = p_basso[0] - p_alto[0]
    dy = p_basso[1] - p_alto[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _estrai_punto(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h, lm.visibility)


def analizza_running_video(video_path, altezza_cm=175):
    """
    Analizza un video di corsa e restituisce un dizionario con tutte
    le metriche biomeccaniche usate dalla pagina Streamlit "Computer Vision".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossibile aprire il file video. Verifica il formato (MP4/MOV/AVI).")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video <= 1:
        fps_video = 30.0

    frames_dati = []  # lista di dict per ogni frame con landmark validi

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            risultato = pose.process(rgb)
            if risultato.pose_landmarks is None:
                continue
            lms = risultato.pose_landmarks.landmark

            dati_frame = {
                "spalla_sx": _estrai_punto(lms, LM.LEFT_SHOULDER, w, h),
                "spalla_dx": _estrai_punto(lms, LM.RIGHT_SHOULDER, w, h),
                "anca_sx": _estrai_punto(lms, LM.LEFT_HIP, w, h),
                "anca_dx": _estrai_punto(lms, LM.RIGHT_HIP, w, h),
                "ginocchio_sx": _estrai_punto(lms, LM.LEFT_KNEE, w, h),
                "ginocchio_dx": _estrai_punto(lms, LM.RIGHT_KNEE, w, h),
                "caviglia_sx": _estrai_punto(lms, LM.LEFT_ANKLE, w, h),
                "caviglia_dx": _estrai_punto(lms, LM.RIGHT_ANKLE, w, h),
                "tallone_sx": _estrai_punto(lms, LM.LEFT_HEEL, w, h),
                "tallone_dx": _estrai_punto(lms, LM.RIGHT_HEEL, w, h),
                "punta_sx": _estrai_punto(lms, LM.LEFT_FOOT_INDEX, w, h),
                "punta_dx": _estrai_punto(lms, LM.RIGHT_FOOT_INDEX, w, h),
                "naso": _estrai_punto(lms, LM.NOSE, w, h),
            }
            frames_dati.append(dati_frame)

    cap.release()

    if len(frames_dati) < 10:
        raise ValueError(
            "Scheletro rilevato in troppo pochi frame. Usa un video con il corridore "
            "ben visibile, ripreso lateralmente, per almeno 2-3 secondi."
        )

    # --- 1. Determina il lato (sinistro/destro) più visibile in media ---
    vis_sx = np.mean([f["caviglia_sx"][2] + f["ginocchio_sx"][2] + f["anca_sx"][2] for f in frames_dati])
    vis_dx = np.mean([f["caviglia_dx"][2] + f["ginocchio_dx"][2] + f["anca_dx"][2] for f in frames_dati])
    lato = "sx" if vis_sx >= vis_dx else "dx"
    lato_label = "Sinistro" if lato == "sx" else "Destro"

    anca_key = f"anca_{lato}"
    ginocchio_key = f"ginocchio_{lato}"
    caviglia_key = f"caviglia_{lato}"
    tallone_key = f"tallone_{lato}"
    punta_key = f"punta_{lato}"

    # --- 2. Serie temporale della caviglia (asse y) per individuare l'appoggio (strike) ---
    caviglia_y = np.array([f[caviglia_key][1] for f in frames_dati])

    # L'appoggio corrisponde a un massimo locale di y (punto più vicino al suolo nell'immagine)
    picchi = []
    for i in range(2, len(caviglia_y) - 2):
        finestra = caviglia_y[i - 2:i + 3]
        if caviglia_y[i] == finestra.max() and caviglia_y[i] > caviglia_y[i - 2] and caviglia_y[i] > caviglia_y[i + 2]:
            picchi.append(i)

    if not picchi:
        # fallback: usa il frame con la caviglia più in basso in assoluto
        picchi = [int(np.argmax(caviglia_y))]

    frame_strike = picchi[len(picchi) // 2]  # strike "centrale" e rappresentativo

    # --- 3. Calibrazione pixel -> cm usando l'altezza dichiarata dall'utente ---
    altezze_px = [
        abs(f["naso"][1] - f[caviglia_key][1]) for f in frames_dati
        if f["naso"][2] > 0.4 and f[caviglia_key][2] > 0.4
    ]
    riferimento_px = max(altezze_px) if altezze_px else 1.0
    scala_cm_per_px = altezza_cm / riferimento_px if riferimento_px > 0 else 0.2

    # --- 4. Angolo del ginocchio all'appoggio ---
    f_strike = frames_dati[frame_strike]
    angolo_ginocchio = _angolo_tra_punti(f_strike[anca_key], f_strike[ginocchio_key], f_strike[caviglia_key])

    # --- 5. Inclinazione del busto (spalla-anca vs verticale) ---
    spalla_key = f"spalla_{lato}"
    angolo_busto = _angolo_rispetto_verticale(f_strike[spalla_key], f_strike[anca_key])

    # --- 6. Overstride: distanza orizzontale caviglia-anca al momento dello strike ---
    overstride_px = abs(f_strike[caviglia_key][0] - f_strike[anca_key][0])
    overstride_cm = overstride_px * scala_cm_per_px

    # --- 7. Oscillazione verticale dell'anca lungo il ciclo del passo ---
    finestra_ciclo = frames_dati[max(0, frame_strike - 10): frame_strike + 10]
    anca_y_ciclo = [f[anca_key][1] for f in finestra_ciclo]
    oscillazione_verticale = (max(anca_y_ciclo) - min(anca_y_ciclo)) * scala_cm_per_px

    # --- 8. Tipo di appoggio (rearfoot / midfoot / forefoot) ---
    tallone_y = f_strike[tallone_key][1]
    punta_y = f_strike[punta_key][1]
    diff_appoggio = tallone_y - punta_y
    if diff_appoggio > 8:
        tipo_appoggio = "Appoggio di Tallone (Rearfoot Strike)"
    elif diff_appoggio < -8:
        tipo_appoggio = "Appoggio di Avampiede (Forefoot Strike)"
    else:
        tipo_appoggio = "Appoggio di Mesopiede (Midfoot Strike)"

    # --- 9. Distribuzione euristica del carico sui distretti articolari ---
    peso_ginocchio = max(0.0, (170 - angolo_ginocchio)) * 1.3
    peso_tibia = max(0.0, overstride_cm - 5) * 4.0
    peso_anca = angolo_busto * 2.0
    peso_caviglia = max(0.0, 10 - overstride_cm) * 2.5 + (15 if "Avampiede" in tipo_appoggio else 0)
    peso_lombare = max(0.0, angolo_busto - 8) * 3.0

    pesi = np.array([peso_ginocchio, peso_tibia, peso_anca, peso_caviglia, peso_lombare], dtype=float)
    pesi = np.clip(pesi, 0.1, None)
    carichi_pct = (pesi / pesi.sum() * 100).round(1)
    articolazioni_carico = ["Ginocchio", "Tibia/Stinco", "Anca", "Caviglia/Achille", "Zona Lombare"]
    sovraccarico_prevalente = articolazioni_carico[int(np.argmax(carichi_pct))]

    # --- 10. Angoli del ginocchio nelle 4 fasi del gait cycle ---
    n = len(frames_dati)
    idx_midstance = min(n - 1, frame_strike + max(2, (picchi[1] - picchi[0]) // 4) if len(picchi) > 1 else frame_strike + 4)
    idx_toeoff = min(n - 1, frame_strike + max(4, (picchi[1] - picchi[0]) // 2) if len(picchi) > 1 else frame_strike + 8)
    idx_swing = min(n - 1, frame_strike + max(6, 3 * (picchi[1] - picchi[0]) // 4) if len(picchi) > 1 else frame_strike + 12)

    def _angolo_ginocchio_a(idx):
        f = frames_dati[idx]
        return round(_angolo_tra_punti(f[anca_key], f[ginocchio_key], f[caviglia_key]), 1)

    fasi_gait = ["Strike", "Mid-Stance", "Toe-Off", "Swing"]
    angoli_fase = [
        round(angolo_ginocchio, 1),
        _angolo_ginocchio_a(idx_midstance),
        _angolo_ginocchio_a(idx_toeoff),
        _angolo_ginocchio_a(idx_swing),
    ]

    # --- 11. Stima euristica del rischio per distretto/patologia ---
    rischio_ginocchio = np.clip(max(0, (165 - angolo_ginocchio)) * 1.8, 0, 100)
    rischio_achille = np.clip(max(0, overstride_cm - 8) * 6.0 + (10 if "Avampiede" in tipo_appoggio else 0), 0, 100)
    rischio_fascite = np.clip(max(0, overstride_cm - 6) * 5.0, 0, 100)
    rischio_bandelletta = np.clip(oscillazione_verticale * 4.0, 0, 100)
    rischio_lombalgia = np.clip(max(0, angolo_busto - 6) * 5.5, 0, 100)

    distretti_rischio = [
        "Ginocchio (Runner's Knee)",
        "Tendine d'Achille",
        "Fascite Plantare",
        "Bandelletta Ileotibiale",
        "Zona Lombare",
    ]
    rischi_ml = np.round([
        rischio_ginocchio, rischio_achille, rischio_fascite, rischio_bandelletta, rischio_lombalgia
    ], 1).tolist()

    idx_max_rischio = int(np.argmax(rischi_ml))
    infortunio_predetto = distretti_rischio[idx_max_rischio]

    probabilita_infortunio_ml = round(float(np.clip(np.mean(rischi_ml) * 1.1, 0, 100)), 1)

    return {
        "lato_analizzato": lato_label,
        "fps_video": round(fps_video, 1),
        "frame_strike_analizzato": int(frame_strike),
        "angolo_ginocchio_appoggio": round(float(angolo_ginocchio), 1),
        "angolo_inclinazione_busto": round(float(angolo_busto), 1),
        "overstride_cm": round(float(overstride_cm), 1),
        "oscillazione_verticale": round(float(oscillazione_verticale), 1),
        "articolazioni_carico": articolazioni_carico,
        "carichi_pct": carichi_pct.tolist(),
        "fasi_gait": fasi_gait,
        "angoli_fase": angoli_fase,
        "distretti_rischio": distretti_rischio,
        "rischi_ml": rischi_ml,
        "tipo_appoggio": tipo_appoggio,
        "sovraccarico_prevalente": sovraccarico_prevalente,
        "probabilita_infortunio_ml": probabilita_infortunio_ml,
        "infortunio_predetto": infortunio_predetto,
    }
