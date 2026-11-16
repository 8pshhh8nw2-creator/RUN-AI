"""utils/skeleton_video.py  (v2)

Genera una copia del video di corsa con lo scheletro (MediaPipe Pose) disegnato
sopra ogni frame e, nella stessa passata, estrae le serie temporali per frame
(angoli del ginocchio, inclinazione busto, altezza anca, altezza caviglie).

Ritorna sempre un dict:
  {'ok': True,  'bytes': ..., 'mime': 'video/mp4', 'w': ..., 'h': ..., 'serie': {...}}
  {'ok': False, 'errore': 'motivo leggibile'}
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

BLU = (255, 110, 30)      # BGR: lato sinistro + tronco
VERDE = (60, 220, 60)     # lato destro
BIANCO = (255, 255, 255)

SPALLA_SX, SPALLA_DX = 11, 12
GOMITO_SX, GOMITO_DX = 13, 14
POLSO_SX, POLSO_DX = 15, 16
ANCA_SX, ANCA_DX = 23, 24
GINOCCHIO_SX, GINOCCHIO_DX = 25, 26
CAVIGLIA_SX, CAVIGLIA_DX = 27, 28
TALLONE_SX, TALLONE_DX = 29, 30
PUNTA_SX, PUNTA_DX = 31, 32

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
LATO_MAX = 1280          # lato lungo massimo del video in uscita (px)
NAN = float("nan")


def _angolo(a, b, c):
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-6:
        return NAN
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))


def _hud(frame, righe):
    h, w = frame.shape[:2]
    s = w / 720.0
    fs = 0.5 * s
    sp = int(26 * s)
    sp_t = max(1, int(s))
    pw, ph = int(w * 0.62), int(sp * (len(righe) + 1) + 10 * s)
    ov = frame.copy()
    cv2.rectangle(ov, (int(10 * s), int(10 * s)),
                  (int(10 * s) + pw, int(10 * s) + ph), (30, 30, 30), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    y = int(10 * s) + sp
    cv2.putText(frame, "METRICHE RUNNING", (int(20 * s), y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, BIANCO, sp_t, cv2.LINE_AA)
    for nome, val in righe:
        y += sp
        cv2.putText(frame, nome, (int(20 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, (190, 190, 190), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, fs, sp_t)
        cv2.putText(frame, val, (int(10 * s) + pw - tw - int(10 * s), y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, BIANCO, sp_t, cv2.LINE_AA)


def _ffmpeg_candidati():
    """Prima il ffmpeg statico di imageio-ffmpeg (ha sempre libx264), poi quello di sistema."""
    lista = []
    try:
        import imageio_ffmpeg
        lista.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    w = shutil.which("ffmpeg")
    if w and w not in lista:
        lista.append(w)
    return lista


def _codifica_browser(raw_path, out_dir):
    """Converte il file mp4v in H.264 (yuv420p, faststart): riproducibile in Chrome, Safari e Firefox.
    Ritorna (percorso, mime, None) oppure (None, None, motivo)."""
    exes = _ffmpeg_candidati()
    if not exes:
        return None, None, "ffmpeg non trovato: aggiungi 'imageio-ffmpeg' a requirements.txt"
    ultimo = ""
    for exe in exes:
        for enc in ("libx264", "libopenh264"):
            out = os.path.join(out_dir, "out.mp4")
            cmd = [exe, "-y", "-v", "error", "-i", raw_path, "-an",
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", enc, "-pix_fmt", "yuv420p"]
            if enc == "libx264":
                cmd += ["-profile:v", "main", "-preset", "veryfast", "-crf", "23"]
            cmd += ["-movflags", "+faststart", out]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=300)
            except Exception as e:
                ultimo = str(e)
                continue
            if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000:
                return out, "video/mp4", None
            ultimo = (r.stderr or b"").decode("utf-8", "ignore")[-200:]
    return None, None, f"encoder H.264 non disponibile ({ultimo.strip()})"


def genera_video_scheletro(video_path, metriche=None, altezza_cm=175):
    try:
        import mediapipe as mp
        pose_mod = mp.solutions.pose
    except Exception as e:
        return {"ok": False, "errore": f"MediaPipe Pose (mp.solutions) non disponibile: {e}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "errore": "impossibile aprire il video"}
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
    writer = None
    size = None
    serie = {k: [] for k in ("t", "ginocchio_sx", "ginocchio_dx", "busto",
                             "anca_y", "caviglia_y_sx", "caviglia_y_dx")}
    gambe_px = []
    frames_jpg = []
    totale = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    passo = max(1, totale // 80) if totale else 4
    n = 0

    try:
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
                        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not writer.isOpened():
                            return {"ok": False, "errore": "impossibile creare il video di output"}

                    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    riga = dict(gx=NAN, gd=NAN, busto=NAN, anca=NAN, csx=NAN, cdx=NAN)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        pts = {i: (np.array([lm[i].x * w, lm[i].y * h]), lm[i].visibility)
                               for i in PUNTI}
                        vis = lambda *ids: min(pts[i][1] for i in ids) > SOGLIA_VIS
                        sp_ossa = max(2, int(w / 220))
                        for a, b, col in OSSA:
                            if vis(a, b):
                                cv2.line(frame, tuple(pts[a][0].astype(int)),
                                         tuple(pts[b][0].astype(int)), col, sp_ossa, cv2.LINE_AA)
                        for i in PUNTI:
                            if pts[i][1] > SOGLIA_VIS:
                                c = tuple(pts[i][0].astype(int))
                                cv2.circle(frame, c, sp_ossa + 2, BIANCO, -1, cv2.LINE_AA)
                                cv2.circle(frame, c, sp_ossa + 3, (40, 40, 40), 1, cv2.LINE_AA)
                        for lato, (an, gi, ca) in (("sx", (ANCA_SX, GINOCCHIO_SX, CAVIGLIA_SX)),
                                                   ("dx", (ANCA_DX, GINOCCHIO_DX, CAVIGLIA_DX))):
                            if vis(an, gi, ca):
                                ang = _angolo(pts[an][0], pts[gi][0], pts[ca][0])
                                riga["gx" if lato == "sx" else "gd"] = ang
                                gambe_px.append(np.linalg.norm(pts[an][0] - pts[gi][0]) +
                                                np.linalg.norm(pts[gi][0] - pts[ca][0]))
                                c = pts[gi][0].astype(int)
                                cv2.putText(frame, f"{ang:.0f} DEG", (c[0] + 8, c[1] - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * w / 720.0,
                                            (0, 220, 255), max(1, int(w / 720.0)), cv2.LINE_AA)
                        if vis(ANCA_SX, ANCA_DX):
                            riga["anca"] = float((pts[ANCA_SX][0][1] + pts[ANCA_DX][0][1]) / 2)
                        if vis(ANCA_SX, ANCA_DX, SPALLA_SX, SPALLA_DX):
                            sh = (pts[SPALLA_SX][0] + pts[SPALLA_DX][0]) / 2
                            hp = (pts[ANCA_SX][0] + pts[ANCA_DX][0]) / 2
                            d = sh - hp
                            riga["busto"] = float(np.degrees(np.arctan2(abs(d[0]), max(1e-6, -d[1]))))
                        if vis(CAVIGLIA_SX):
                            riga["csx"] = float(pts[CAVIGLIA_SX][0][1])
                        if vis(CAVIGLIA_DX):
                            riga["cdx"] = float(pts[CAVIGLIA_DX][0][1])
                    if righe:
                        _hud(frame, righe)
                    writer.write(frame)
                    if n % passo == 0:
                        piccolo = cv2.resize(frame, (540, int(540 * h / w)), interpolation=cv2.INTER_AREA)
                        okj, buf = cv2.imencode(".jpg", piccolo, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if okj:
                            frames_jpg.append((n / fps, buf.tobytes()))

                    serie["t"].append(n / fps)
                    serie["ginocchio_sx"].append(riga["gx"])
                    serie["ginocchio_dx"].append(riga["gd"])
                    serie["busto"].append(riga["busto"])
                    serie["anca_y"].append(riga["anca"])
                    serie["caviglia_y_sx"].append(riga["csx"])
                    serie["caviglia_y_dx"].append(riga["cdx"])
                    n += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if size is None or n == 0:
            return {"ok": False, "errore": "nessun frame letto dal video"}

        # scala px -> cm: lunghezza gamba (anca-ginocchio-caviglia) ~ 53% dell'altezza
        scala = NAN
        if gambe_px:
            scala = float(0.53 * altezza_cm / np.median(gambe_px))
        serie["cm_per_px"] = scala
        serie["fps"] = float(fps)

        path, mime, motivo = _codifica_browser(raw_path, tmp_dir)
        if path is None:
            return {"ok": False, "errore": motivo, "serie": serie, "frames": frames_jpg}
        with open(path, "rb") as f:
            data = f.read()
        return {"ok": True, "bytes": data, "mime": mime, "w": size[0], "h": size[1],
                "serie": serie, "frames": frames_jpg}
    except Exception as e:
        return {"ok": False, "errore": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
