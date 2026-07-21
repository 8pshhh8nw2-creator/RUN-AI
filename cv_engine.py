"""
cv_engine.py
Motore di Computer Vision per l'analisi biomeccanica della corsa.
Estrae lo scheletro con MediaPipe Pose da un video reale (profilo laterale)
e calcola le metriche biomeccaniche SENZA dati inventati.

Dipendenze:
    pip install mediapipe opencv-python numpy

Uso in Streamlit:
    from cv_engine import analizza_running_video
    dati_cv = analizza_running_video(video_path, altezza_cm=175)
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Indici landmark MediaPipe Pose rilevanti (BlazePose 33 punti)
LM = mp_vision.PoseLandmark

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "pose_landmarker_lite.task")


def _assicura_modello():
    """Scarica il modello PoseLandmarker (lite, ~5MB) al primo utilizzo e lo mette in cache."""
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _crea_pose_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=_assicura_modello())
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _angolo(a, b, c):
    """Angolo in gradi nel vertice b, tra i segmenti b-a e b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _angolo_verticale(a, b):
    """Angolo tra il segmento a->b e la verticale (0 = perfettamente verticale)."""
    a, b = np.array(a), np.array(b)
    v = b - a
    vert = np.array([0, -1])  # verso l'alto in coordinate immagine (y cresce in basso)
    cos_ang = np.dot(v, vert) / (np.linalg.norm(v) + 1e-9)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _estrai_landmark_video(video_path, min_visibilita=0.5, max_frame=600):
    """
    Estrae i landmark per ogni frame processabile del video.
    Ritorna: fps, lista di dict {idx_frame, tempo, landmarks(px)}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossibile aprire il file video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    risultati = []
    landmarker = _crea_pose_landmarker()
    try:
        idx = 0
        while cap.isOpened() and idx < max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((idx / fps) * 1000)
            out = landmarker.detect_for_video(mp_image, timestamp_ms)
            if out.pose_landmarks:
                landmark_list = out.pose_landmarks[0]  # prima persona rilevata
                pts = {}
                for lm_id in LM:
                    p = landmark_list[lm_id.value]
                    visibilita = getattr(p, "visibility", 1.0)
                    if visibilita >= min_visibilita:
                        pts[lm_id.value] = (p.x * frame_w, p.y * frame_h, visibilita)
                if pts:
                    risultati.append({"idx": idx, "t": idx / fps, "pts": pts})
            idx += 1
    finally:
        landmarker.close()
    cap.release()

    if len(risultati) < 10:
        raise ValueError(
            "Scheletro rilevato in troppo pochi frame: video troppo corto, "
            "corridore non ben visibile o inquadratura non laterale."
        )
    return fps, frame_h, risultati


def _scegli_lato(risultati):
    """Determina se usare landmark del lato destro o sinistro (media visibilità più alta)."""
    vis_dx, vis_sx, n_dx, n_sx = 0.0, 0.0, 0, 0
    for r in risultati:
        pts = r["pts"]
        if LM.RIGHT_HIP.value in pts:
            vis_dx += pts[LM.RIGHT_HIP.value][2]
            n_dx += 1
        if LM.LEFT_HIP.value in pts:
            vis_sx += pts[LM.LEFT_HIP.value][2]
            n_sx += 1
    media_dx = vis_dx / n_dx if n_dx else 0
    media_sx = vis_sx / n_sx if n_sx else 0
    return "DX" if media_dx >= media_sx else "SX"


def _serie(risultati, key, lato_suffix):
    """Estrae la serie temporale (idx, x, y) di un landmark, ignorando i frame mancanti."""
    idmap = {
        "shoulder": LM.RIGHT_SHOULDER.value if lato_suffix == "DX" else LM.LEFT_SHOULDER.value,
        "hip": LM.RIGHT_HIP.value if lato_suffix == "DX" else LM.LEFT_HIP.value,
        "knee": LM.RIGHT_KNEE.value if lato_suffix == "DX" else LM.LEFT_KNEE.value,
        "ankle": LM.RIGHT_ANKLE.value if lato_suffix == "DX" else LM.LEFT_ANKLE.value,
        "heel": LM.RIGHT_HEEL.value if lato_suffix == "DX" else LM.LEFT_HEEL.value,
        "toe": LM.RIGHT_FOOT_INDEX.value if lato_suffix == "DX" else LM.LEFT_FOOT_INDEX.value,
    }
    lm_id = idmap[key]
    out = []
    for r in risultati:
        if lm_id in r["pts"]:
            x, y, v = r["pts"][lm_id]
            out.append((r["idx"], x, y))
    return out


def analizza_running_video(video_path, altezza_cm=175.0):
    """
    Analizza un video di corsa (profilo laterale) ed estrae le metriche
    biomeccaniche REALI tramite pose estimation. Nessun dato è inventato:
    ogni valore deriva da coordinate dei keypoint calcolate sul video.

    Ritorna un dizionario compatibile con il report Streamlit esistente,
    incluse le serie per i grafici (angoli di fase, mappa sovraccarico,
    rischio per distretto).
    """
    fps, frame_h, risultati = _estrai_landmark_video(video_path)
    lato = _scegli_lato(risultati)

    hip = _serie(risultati, "hip", lato)
    knee = _serie(risultati, "knee", lato)
    ankle = _serie(risultati, "ankle", lato)
    heel = _serie(risultati, "heel", lato)
    toe = _serie(risultati, "toe", lato)
    shoulder = _serie(risultati, "shoulder", lato)

    if len(heel) < 10 or len(hip) < 10:
        raise ValueError("Dati insufficienti sul piede/anca per calcolare la falcata.")

    # --- Calibrazione pixel -> cm usando l'altezza dichiarata dall'utente ---
    # Altezza scheletro = distanza media spalla-caviglia (proxy di statura in verticale)
    alt_px = []
    idx_to_shoulder = {i: (x, y) for i, x, y in shoulder}
    idx_to_ankle = {i: (x, y) for i, x, y in ankle}
    for i in idx_to_shoulder:
        if i in idx_to_ankle:
            sx, sy = idx_to_shoulder[i]
            ax, ay = idx_to_ankle[i]
            alt_px.append(abs(ay - sy))
    if not alt_px:
        raise ValueError("Impossibile calibrare la scala pixel/cm dal video.")
    px_riferimento = float(np.percentile(alt_px, 90))  # punto di massima estensione verticale
    px_per_cm = px_riferimento / (altezza_cm * 0.87)  # ~87% statura = spalla-caviglia

    # --- Individuazione dei momenti di appoggio (heel strike) ---
    idx_heel = np.array([i for i, x, y in heel])
    y_heel = np.array([y for i, x, y in heel])

    picchi = []
    dist_min = max(int(fps * 0.25), 3)
    for k in range(2, len(y_heel) - 2):
        finestra = y_heel[max(0, k - dist_min): k + dist_min + 1]
        if y_heel[k] == finestra.max() and y_heel[k] > np.mean(y_heel):
            if not picchi or (idx_heel[k] - picchi[-1]) >= dist_min:
                picchi.append(idx_heel[k])

    if len(picchi) < 1:
        raise ValueError("Nessun appoggio del piede rilevato: video troppo breve o corridore non visibile per un ciclo completo.")

    # Scarta il primo/ultimo 10% dei frame (partenza/arresto, spesso rumorosi)
    n_frame_tot = risultati[-1]["idx"]
    picchi_validi = [p for p in picchi if 0.1 * n_frame_tot < p < 0.9 * n_frame_tot] or picchi
    frame_strike = int(picchi_validi[len(picchi_validi) // 2])  # appoggio centrale, il più stabile

    def pos(serie, frame_idx):
        d = {i: (x, y) for i, x, y in serie}
        if frame_idx in d:
            return d[frame_idx]
        idx_arr = np.array(list(d.keys()))
        vicino = idx_arr[np.argmin(np.abs(idx_arr - frame_idx))]
        return d[int(vicino)]

    hip_s = pos(hip, frame_strike)
    knee_s = pos(knee, frame_strike)
    ankle_s = pos(ankle, frame_strike)
    heel_s = pos(heel, frame_strike)
    toe_s = pos(toe, frame_strike)
    shoulder_s = pos(shoulder, frame_strike)

    # --- Angolo ginocchio all'appoggio ---
    angolo_ginocchio = _angolo(hip_s[:2], knee_s[:2], ankle_s[:2])

    # --- Inclinazione busto (spalla-anca vs verticale) ---
    inclinazione_busto = _angolo_verticale(hip_s[:2], shoulder_s[:2])

    # --- Tipo di appoggio: confronto altezza tallone vs punta al momento dello strike ---
    diff_talloe = heel_s[1] - toe_s[1]  # y maggiore = più in basso/vicino al suolo
    if diff_talloe > 0.015 * frame_h:
        tipo_appoggio = "Appoggio di Tallone Marcato (Heel Striking)"
    elif diff_talloe < -0.015 * frame_h:
        tipo_appoggio = "Appoggio di Avampiede (Forefoot Striking)"
    else:
        tipo_appoggio = "Appoggio di Mesopiede (Midfoot Striking)"

    # --- Overstride: distanza orizzontale caviglia-anca al momento dello strike, in cm ---
    overstride_px = abs(ankle_s[0] - hip_s[0])
    overstride_cm = overstride_px / px_per_cm

    # --- Oscillazione verticale: ampiezza del movimento verticale dell'anca su un ciclo ---
    finestra_ciclo = [h for h in hip if abs(h[0] - frame_strike) <= int(fps * 0.6)]
    y_hip_ciclo = [y for i, x, y in finestra_ciclo] or [hip_s[1]]
    oscillazione_px = max(y_hip_ciclo) - min(y_hip_ciclo)
    oscillazione_verticale = oscillazione_px / px_per_cm

    # --- Angoli di ginocchio nelle altre fasi del passo ---
    # Mid-stance: punto di massimo abbassamento dell'anca dopo lo strike (picco di carico)
    post_strike = [h for h in hip if frame_strike <= h[0] <= frame_strike + int(fps * 0.3)]
    if post_strike:
        f_midstance = max(post_strike, key=lambda h: h[2])[0]
    else:
        f_midstance = frame_strike
    angolo_midstance = _angolo(pos(hip, f_midstance)[:2], pos(knee, f_midstance)[:2], pos(ankle, f_midstance)[:2])

    # Toe-off: punta ancora bassa ma prossima al sollevamento, ~0.35-0.5 ciclo dopo lo strike
    finestra_toeoff = [t for t in toe if frame_strike + int(fps * 0.15) <= t[0] <= frame_strike + int(fps * 0.45)]
    if finestra_toeoff:
        f_toeoff = min(finestra_toeoff, key=lambda t: t[2])[0]  # punta più in alto = distacco
    else:
        f_toeoff = frame_strike + int(fps * 0.3)
    angolo_toeoff = _angolo(pos(hip, f_toeoff)[:2], pos(knee, f_toeoff)[:2], pos(ankle, f_toeoff)[:2])

    # Swing: massima flessione del ginocchio (angolo minimo) nella finestra successiva
    finestra_swing = [k for k in knee if frame_strike + int(fps * 0.3) <= k[0] <= frame_strike + int(fps * 0.7)]
    angoli_swing = []
    for i, kx, ky in finestra_swing:
        angoli_swing.append((_angolo(pos(hip, i)[:2], (kx, ky), pos(ankle, i)[:2]), i))
    angolo_swing = min(angoli_swing)[0] if angoli_swing else 90.0

    # --- Punteggi di sovraccarico articolare (euristica su feature REALI estratte) ---
    # Nota: non è un modello ML addestrato su dataset clinici; è uno scoring
    # basato su soglie biomeccaniche derivate dai valori misurati sul video.
    score_ginocchio = max(0, 160 - angolo_ginocchio) * 1.4 + overstride_cm * 1.2
    score_achille = max(0, diff_talloe) / max(frame_h, 1) * 4000 + overstride_cm * 0.8
    score_anca = inclinazione_busto * 2.0
    score_schiena = inclinazione_busto * 1.6
    score_caviglie = abs(diff_talloe) / max(frame_h, 1) * 1500

    grezzi = np.array([score_ginocchio, score_achille, score_anca, score_schiena, score_caviglie])
    grezzi = np.clip(grezzi, 0.5, None)
    carichi_pct = (grezzi / grezzi.sum() * 100).round(1).tolist()

    articolazioni = ["Ginocchia", "Achille", "Anca", "Schiena", "Caviglie"]

    # --- Rischio infortunio per distretto (stessa logica, ripesata su patologie tipiche) ---
    r_ginocchio = score_ginocchio * 1.3
    r_achille = score_achille * 1.2
    r_plantare = score_caviglie * 1.1 + (10 if "Avampiede" in tipo_appoggio else 0)
    r_tibia = overstride_cm * 1.5
    r_lombari = inclinazione_busto * 1.0

    grezzi_r = np.clip(np.array([r_ginocchio, r_achille, r_plantare, r_tibia, r_lombari]), 0.5, None)
    rischi_ml = (grezzi_r / grezzi_r.sum() * 100).round(1).tolist()
    distretti_rischio = ["Ginocchio/Rotula", "Tendine Achille", "Fascia Plantare", "Tibia (Periostite)", "Lombari"]

    idx_max = int(np.argmax(rischi_ml))
    mappa_infortuni = {
        0: "Sindrome Patello-Femorale (Ginocchio del Corridore)",
        1: "Tendinopatia Achillea",
        2: "Fascite Plantare",
        3: "Periostite Tibiale (Sindrome da Stress Tibiale)",
        4: "Sovraccarico Lombare Posturale",
    }
    infortunio_predetto = mappa_infortuni[idx_max]
    probabilita_infortunio_ml = round(float(min(95.0, 40 + rischi_ml[idx_max] * 1.3)), 1)

    return {
        "angolo_ginocchio_appoggio": round(angolo_ginocchio, 1),
        "angolo_inclinazione_busto": round(inclinazione_busto, 1),
        "oscillazione_verticale": round(oscillazione_verticale, 1),
        "overstride_cm": round(overstride_cm, 1),
        "sovraccarico_prevalente": f"Complesso {articolazioni[int(np.argmax(carichi_pct))]}",
        "tipo_appoggio": tipo_appoggio,
        "infortunio_predetto": infortunio_predetto,
        "probabilita_infortunio_ml": probabilita_infortunio_ml,
        "lato_analizzato": "Destro" if lato == "DX" else "Sinistro",
        "fps_video": round(fps, 1),
        "frame_strike_analizzato": frame_strike,
        # dati per i grafici (sostituiscono gli array hardcoded)
        "articolazioni_carico": articolazioni,
        "carichi_pct": carichi_pct,
        "fasi_gait": ["Impatto (Strike)", "Mid-Stance", "Toe-Off", "Swing"],
        "angoli_fase": [round(angolo_ginocchio, 1), round(angolo_midstance, 1), round(angolo_toeoff, 1), round(angolo_swing, 1)],
        "distretti_rischio": distretti_rischio,
        "rischi_ml": rischi_ml,
    }
