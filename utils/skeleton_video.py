"""utils/skeleton_video.py

Genera una copia del video di corsa con lo scheletro (MediaPipe Pose)
disegnato sopra ogni frame, pronta per st.video().
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

# Colori BGR
BLU = (255, 110, 30)      # lato sinistro + tronco
VERDE = (60, 220, 60)     # lato destro
BIANCO = (255, 255, 255)

# Indici landmark MediaPipe Pose
SPALLA_SX, SPALLA_DX = 11, 12
GOMITO_SX, GOMITO_DX = 13, 14
POLSO_SX, POLSO_DX = 15, 16
ANCA_SX, ANCA_DX = 23, 24
GINOCCHIO_SX, GINOCCHIO_DX = 25, 26
CAVIGLIA_SX, CAVIGLIA_DX = 27, 28
TALLONE_SX, TALLONE_DX = 29, 30
PUNTA_SX, PUNTA_DX = 31, 32

# (a, b, colore)
OSSA = [
    (SPALLA_SX, SPALLA_DX, BLU), (ANCA_SX, ANCA_DX, BIANCO),
    (SPALLA_SX, ANCA_SX, BLU), (SPALLA_DX, ANCA_DX, BLU),
    (SPALLA_SX, GOMITO_SX, BLU), (GOMITO_SX, POLSO_SX, BLU),
    (SPALLA_DX, GOMITO_DX, VERDE), (GOMITO_DX, POLSO_DX, VERDE),
    (ANCA_SX, GINOCCHIO_SX, BLU), (GINOCCHIO_SX, CAVIGLIA_SX, BLU),
    (CAVIGLIA_SX, TALLONE_SX, BLU), (TALLONE_SX, PUNTA_SX, BLU),
    (CAVIGLIA_SX, PUNTA_SX, BLU),
    (ANCA_DX, GINOCCHIO_DX, VERDE), (GINOCCHIO_DX, CAVIGLIA_DX, VERDE),
    (CAVIGLIA_DX, TALLONE_DX, VERDE), (TALLONE_DX, PUNTA_DX, VERDE),
    (CAVIGLIA_DX, PUNTA_DX, VERDE),
]
PUNTI = [SPALLA_SX, SPALLA_DX, GOMITO_SX, GOMITO_DX, POLSO_SX, POLSO_DX,
         ANCA_SX, ANCA_DX, GINOCCHIO_SX, GINOCCHIO_DX, CAVIGLIA_SX,
         CAVIGLIA_DX, TALLONE_SX, TALLONE_DX, PUNTA_SX, PUNTA_DX]

SOGLIA_VIS = 0.4
LATO_MAX = 960          # lato lungo massimo del video in uscita (px)


def _angolo(a, b, c):
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))


def _hud(frame, righe):
    """Pannello metriche in alto a sinistra."""
    h, w = frame.shape[:2]
    scala = w / 720.0
    fs = 0.5 * scala
    sp = int(26 * scala)
    pw, ph = int(w * 0.62), int(sp * (len(righe) + 1) + 10 * scala)
    ov = frame.copy()
    cv2.rectangle(ov, (int(10 * scala), int(10 * scala)),
                  (int(10 * scala) + pw, int(10 * scala) + ph), (30, 30, 30), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    y = int(10 * scala) + sp
    cv2.putText(frame, "METRICHE RUNNING", (int(20 * scala), y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, BIANCO, max(1, int(scala)), cv2.LINE_AA)
    for nome, val in righe:
        y += sp
        cv2.putText(frame, nome, (int(20 * scala), y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, (190, 190, 190), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(scala)))
        cv2.putText(frame, val, (int(10 * scala) + pw - tw - int(10 * scala), y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, BIANCO, max(1, int(scala)), cv2.LINE_AA)


def genera_video_scheletro(video_path, metriche=None):
    """Ritorna (bytes_mp4, larghezza, altezza) oppure None se non riesce.

    metriche: dict opzionale con le metriche gia' calcolate (angolo_ginocchio_appoggio,
    overstride_cm, angolo_inclinazione_busto, oscillazione_verticale) per il pannello.
    """
    try:
        import mediapipe as mp
        pose_mod = mp.solutions.pose
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 1 or fps > 240:
        fps = 30.0

    righe = []
    if metriche:
        righe = [
            ("ANGOLO GINOCCHIO", f"{metriche['angolo_ginocchio_appoggio']:.0f} DEG"),
            ("OVERSTRIDE", f"{metriche['overstride_cm']:.1f} CM"),
            ("INCLINAZIONE BUSTO", f"{metriche['angolo_inclinazione_busto']:.1f} DEG"),
            ("OSCILL. VERT.", f"{metriche['oscillazione_verticale']:.1f} CM"),
        ]

    tmp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(tmp_dir, "raw.mp4")
    out_path = os.path.join(tmp_dir, "out.mp4")
    writer = None
    size = None

    try:
        with pose_mod.Pose(static_image_mode=False, model_complexity=1,
                           smooth_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                k = min(1.0, LATO_MAX / max(h, w))
                if k < 1.0:
                    frame = cv2.resize(frame, (int(w * k) // 2 * 2, int(h * k) // 2 * 2),
                                       interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
                if writer is None:
                    size = (w, h)
                    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, size)

                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    pts = {i: (np.array([lm[i].x * w, lm[i].y * h]), lm[i].visibility)
                           for i in PUNTI}
                    sp_ossa = max(2, int(w / 220))
                    for a, b, col in OSSA:
                        if pts[a][1] > SOGLIA_VIS and pts[b][1] > SOGLIA_VIS:
                            cv2.line(frame, tuple(pts[a][0].astype(int)),
                                     tuple(pts[b][0].astype(int)), col, sp_ossa, cv2.LINE_AA)
                    for i in PUNTI:
                        if pts[i][1] > SOGLIA_VIS:
                            c = tuple(pts[i][0].astype(int))
                            cv2.circle(frame, c, sp_ossa + 2, BIANCO, -1, cv2.LINE_AA)
                            cv2.circle(frame, c, sp_ossa + 3, (40, 40, 40), 1, cv2.LINE_AA)
                    # angoli ginocchio
                    for an, gi, ca in ((ANCA_SX, GINOCCHIO_SX, CAVIGLIA_SX),
                                       (ANCA_DX, GINOCCHIO_DX, CAVIGLIA_DX)):
                        if min(pts[an][1], pts[gi][1], pts[ca][1]) > SOGLIA_VIS:
                            ang = _angolo(pts[an][0], pts[gi][0], pts[ca][0])
                            if ang is not None:
                                c = pts[gi][0].astype(int)
                                cv2.putText(frame, f"{ang:.0f} DEG", (c[0] + 8, c[1] - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * w / 720.0,
                                            (0, 220, 255), max(1, int(w / 720.0)), cv2.LINE_AA)
                if righe:
                    _hud(frame, righe)
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    try:
        if size is None or not os.path.exists(raw_path):
            return None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        final = raw_path
        if ffmpeg:  # H.264: riproducibile in tutti i browser
            r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", raw_path,
                                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", out_path],
                               capture_output=True)
            if r.returncode == 0 and os.path.exists(out_path):
                final = out_path
        with open(final, "rb") as f:
            return f.read(), size[0], size[1]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
