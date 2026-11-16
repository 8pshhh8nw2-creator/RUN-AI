import os
import hashlib
import tempfile

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_CV
from utils.computer_vision import analizza_running_video
from utils.skeleton_video import genera_video_scheletro
from utils.sidebar import sidebar_comune

st.set_page_config(page_title="Computer Vision", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_CV = get_svg_url(SVG_CV)

df, df_full, filtro_tempo = sidebar_comune()

CHIAVI_OK = {"qualita", "angolo_ginocchio_appoggio", "overstride_cm",
             "angolo_inclinazione_busto", "oscillazione_verticale",
             "tipo_appoggio", "distribuzione_appoggio_pct",
             "fasi_gait", "angoli_fase", "frame_strike_analizzato",
             "lato_analizzato", "fps_video"}


def stima_rischio(d):
    """Aggiunge a d le stime euristiche (carichi, rischi, probabilita')
    calcolate dalle metriche reali. Non e' un modello clinico."""
    d = dict(d)
    ang = float(d['angolo_ginocchio_appoggio'])
    over = max(0.0, float(d['overstride_cm']))
    busto = float(d['angolo_inclinazione_busto'])
    ext = max(0.0, ang - 165.0)
    distr = d.get('distribuzione_appoggio_pct', {})
    tal = float(distr.get('tallone', 0.0))
    ava = float(distr.get('avampiede', 0.0))
    dev_busto = abs(busto - 6.5)

    carichi = {
        "Ginocchio": 25 + 2.0 * over + 1.0 * ext,
        "Tibia": 20 + 1.5 * over + 0.25 * tal,
        "Tendine d'Achille": 20 + 0.30 * ava + 0.5 * over,
        "Anca": 15 + 2.0 * dev_busto,
        "Piede/Fascia": 15 + 0.20 * tal + 0.2 * ava,
    }
    carichi = {k: float(np.clip(v, 5, 95)) for k, v in carichi.items()}
    tot = sum(carichi.values())

    d['articolazioni_carico'] = list(carichi.keys())
    d['carichi_pct'] = [round(v, 1) for v in carichi.values()]
    d['distretti_rischio'] = list(carichi.keys())
    d['rischi_ml'] = [round(100.0 * v / tot, 1) for v in carichi.values()]
    top = max(carichi, key=carichi.get)
    d['sovraccarico_prevalente'] = top
    nomi = {
        "Ginocchio": "Sindrome femoro-rotulea (ginocchio del corridore)",
        "Tibia": "Periostite tibiale (shin splints)",
        "Tendine d'Achille": "Tendinopatia achillea",
        "Anca": "Sovraccarico anca / bandelletta ileo-tibiale",
        "Piede/Fascia": "Fascite plantare",
    }
    d['infortunio_predetto'] = nomi[top]
    d['probabilita_infortunio_ml'] = round(float(np.clip(
        10 + 2.2 * over + 0.8 * ext + 0.15 * tal + 1.0 * dev_busto, 5, 95)), 1)
    return d


# ---------------------------------------------------------
# Analisi temporale: grafici + spiegazioni ricavati dal video
# ---------------------------------------------------------
def _f(x, nd=1):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "n/d"


def _pre(y, w):
    """Interpola i buchi (landmark non visibili) e leviga. None se i dati sono troppo pochi."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if m.sum() < max(10, 0.3 * len(y)):
        return None
    x = np.arange(len(y))
    y = np.interp(x, x[m], y[m])
    w = max(1, int(w))
    if w >= 2:
        pad = np.pad(y, (w // 2, w - 1 - w // 2), mode='edge')
        y = np.convolve(pad, np.ones(w) / w, mode='valid')
    return y


def _picchi(y, dist, prom):
    try:
        from scipy.signal import find_peaks
        p, _ = find_peaks(y, distance=max(1, int(dist)), prominence=prom)
        return np.asarray(p, dtype=int)
    except Exception:
        p = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] >= y[i + 1] and (not p or i - p[-1] >= dist):
                p.append(i)
        return np.asarray(p, dtype=int)


def costruisci_analisi(serie, dati_cv):
    """Ritorna {chiave: (figura, spiegazione_html)} calcolati sulle serie estratte dal video."""
    out = {}
    fps = float(serie['fps'])
    t = np.array(serie['t'], dtype=float)
    scala = serie.get('cm_per_px')
    unita = 'cm'
    if scala is None or not np.isfinite(scala):
        scala, unita = 1.0, 'px'
    w_sm = max(3, int(0.12 * fps))
    ts = dati_cv['frame_strike_analizzato'] / fps
    C_CY, C_GR, C_AM, C_OR = '#00E5FF', '#00F5A0', '#FFB020', '#FF6A3D'

    # 1) Angolo del ginocchio nel tempo ----------------------------------
    gs = _pre(serie['ginocchio_sx'], w_sm)
    gd = _pre(serie['ginocchio_dx'], w_sm)
    if gs is not None or gd is not None:
        fig = go.Figure()
        parti = []
        for nome, y, col in (("Ginocchio sx", gs, C_CY), ("Ginocchio dx", gd, C_GR)):
            if y is not None:
                fig.add_trace(go.Scatter(x=t, y=y, name=nome, line=dict(color=col, width=2)))
                parti.append(f"{nome.lower()}: da {_f(np.min(y))}° (massima flessione) a {_f(np.max(y))}° (estensione), escursione {_f(np.ptp(y))}°")
        fig.add_vline(x=ts, line=dict(color=C_OR, dash='dash', width=1.5))
        fig.update_layout(height=300, xaxis_title='Tempo (s)', yaxis_title='Angolo anca-ginocchio-caviglia (°)',
                          legend=dict(orientation='h', y=1.12))
        testo = ("<strong>Angolo del ginocchio nel tempo:</strong> " + "; ".join(parti) +
                 f". La linea tratteggiata marca l'appoggio analizzato (frame {dati_cv['frame_strike_analizzato']}, a {ts:.2f} s), "
                 f"dove il ginocchio misura {_f(float(dati_cv['angolo_ginocchio_appoggio']))}°. Un'escursione ampia indica una buona ammortizzazione in appoggio.")
        out['ginocchio'] = (fig, testo)

    # 2) Oscillazione verticale del bacino -------------------------------
    anca = _pre(serie['anca_y'], max(3, int(0.05 * fps)))
    if anca is not None:
        trend = _pre(anca, max(3, int(1.0 * fps)))
        disp = -(anca - trend) * scala
        osc = float(np.percentile(disp, 95) - np.percentile(disp, 5))
        fig = go.Figure(go.Scatter(x=t, y=disp, line=dict(color=C_AM, width=2), fill='tozeroy',
                                   fillcolor='rgba(255,176,32,0.15)'))
        fig.update_layout(height=300, xaxis_title='Tempo (s)', yaxis_title=f'Spostamento verticale anca ({unita})')
        giudizio = ("entro il target di 8 cm" if osc < 8 else "sopra il target di 8 cm: parte dell'energia va in rimbalzo verticale anziché in avanzamento") if unita == 'cm' else "scala in cm non disponibile"
        testo = (f"<strong>Rimbalzo del bacino:</strong> l'altezza dell'anca oscilla di {_f(osc)} {unita} (picco-picco, 5°-95° percentile), {giudizio}. "
                 f"Il valore riportato nel report è {_f(float(dati_cv['oscillazione_verticale']))} cm. La scala px→cm è indicativa (gamba ≈ 53% dell'altezza inserita).")
        out['oscillazione'] = (fig, testo)

    # 3) Inclinazione del busto -------------------------------------------
    bu = _pre(serie['busto'], w_sm)
    if bu is not None:
        fig = go.Figure(go.Scatter(x=t, y=bu, line=dict(color=C_GR, width=2)))
        fig.add_hrect(y0=5, y1=8, fillcolor='rgba(0,229,255,0.15)', line_width=0)
        fig.update_layout(height=300, xaxis_title='Tempo (s)', yaxis_title='Inclinazione busto (°)')
        m = float(np.mean(bu))
        if m < 5:
            g = "busto troppo verticale: tende a frenare l'avanzamento"
        elif m > 8:
            g = "busto molto inclinato in avanti: aumenta il carico su anca e lombare"
        else:
            g = "dentro la fascia ottimale 5-8°"
        testo = (f"<strong>Postura del busto:</strong> inclinazione media {_f(m)}° (min {_f(np.min(bu))}°, max {_f(np.max(bu))}°); {g}. "
                 "La fascia azzurra è l'intervallo ottimale. Misura l'angolo rispetto alla verticale, senza segno.")
        out['busto'] = (fig, testo)

    # 4) Altezza dei piedi e cadenza ---------------------------------------
    fig = go.Figure()
    tempi_picchi = []
    dett = []
    for nome, key, col in (("Piede sx", 'caviglia_y_sx', C_CY), ("Piede dx", 'caviglia_y_dx', C_GR)):
        y = _pre(serie[key], max(3, int(0.06 * fps)))
        if y is None:
            continue
        h = -y
        h = (h - h.min()) * scala
        rng = float(np.ptp(h))
        fig.add_trace(go.Scatter(x=t, y=h, name=nome, line=dict(color=col, width=2)))
        if rng > 0:
            pk = _picchi(h, 0.25 * fps, 0.25 * rng)
            if len(pk):
                fig.add_trace(go.Scatter(x=t[pk], y=h[pk], mode='markers', showlegend=False,
                                         marker=dict(color=C_OR, size=8)))
                tempi_picchi += list(t[pk])
                dett.append(f"{nome.lower()}: {len(pk)} oscillazioni, sollevamento massimo {_f(float(np.max(h[pk])))} {unita}")
    if len(fig.data):
        fig.update_layout(height=300, xaxis_title='Tempo (s)', yaxis_title=f'Altezza caviglia ({unita})',
                          legend=dict(orientation='h', y=1.12))
        tp = np.sort(np.array(tempi_picchi))
        if len(tp) >= 3 and (t[-1] - t[0]) > 0:
            cad = len(tp) / (t[-1] - t[0]) * 60.0
            iv = np.diff(tp)
            cv_iv = float(np.std(iv) / np.mean(iv) * 100) if np.mean(iv) > 0 else float('nan')
            nota = ""
            if not (100 <= cad <= 230):
                nota = " Valore fuori dall'intervallo tipico della corsa: verifica che il corridore sia inquadrato per intero."
            testo = (f"<strong>Passi e cadenza:</strong> " + "; ".join(dett) +
                     f". Cadenza stimata ≈ {cad:.0f} passi/min (obiettivo 170-180), intervallo medio tra passi {np.mean(iv) * 1000:.0f} ms "
                     f"con variabilità {_f(cv_iv)}% (più è bassa, più il ritmo è regolare).{nota}")
        else:
            testo = "<strong>Passi e cadenza:</strong> troppe poche oscillazioni del piede rilevate nel video per stimare la cadenza."
        out['piedi'] = (fig, testo)

    # 5) Distribuzione dell'appoggio ---------------------------------------
    distr = dati_cv.get('distribuzione_appoggio_pct') or {}
    if distr:
        fig = go.Figure(go.Pie(labels=[k.capitalize() for k in distr.keys()], values=list(distr.values()), hole=0.55,
                               marker=dict(colors=[C_OR, C_GR, C_CY, C_AM][:len(distr)]), textinfo='label+percent'))
        fig.update_layout(height=300, showlegend=False)
        dom = max(distr, key=distr.get)
        testo = (f"<strong>Tipo di appoggio:</strong> {dati_cv['tipo_appoggio']}. La zona di contatto prevalente è {dom} "
                 f"({_f(float(distr[dom]), 0)}%). Un appoggio di tallone molto avanti al baricentro aumenta l'impatto su ginocchio e tibia.")
        out['appoggio'] = (fig, testo)

    # 6) Simmetria destra/sinistra -------------------------------------------
    if gs is not None and gd is not None:
        rom_s, rom_d = float(np.ptp(gs)), float(np.ptp(gd))
        med_s, med_d = float(np.mean(gs)), float(np.mean(gd))
        sim = 100.0 * min(rom_s, rom_d) / max(rom_s, rom_d) if max(rom_s, rom_d) > 0 else float('nan')
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Escursione (°)', x=['Sx', 'Dx'], y=[rom_s, rom_d], marker_color=C_CY))
        fig.add_trace(go.Bar(name='Angolo medio (°)', x=['Sx', 'Dx'], y=[med_s, med_d], marker_color=C_GR))
        fig.update_layout(height=300, barmode='group', legend=dict(orientation='h', y=1.12))
        g = "buona simmetria" if sim >= 90 else "asimmetria da tenere sotto controllo"
        testo = (f"<strong>Simmetria sx/dx:</strong> escursione del ginocchio {_f(rom_s)}° a sinistra contro {_f(rom_d)}° a destra, "
                 f"indice di simmetria {_f(sim, 0)}% ({g}). Una gamba che si flette molto meno dell'altra può indicare compenso o dolore.")
        out['simmetria'] = (fig, testo)

    return out



# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI (DATI REALI)
# ---------------------------------------------------------
header_block(
    "Modulo 06 — Computer Vision",
    "AI RUNNING FORM ANALYSIS & INJURY PREDICTION",
    "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro biometrico, calcola angoli/sovraccarichi e stima il rischio d'infortunio.",
    IMG_HERO_CV, "Pose Estimation & ML"
)
st.markdown("""
<div class='info-box'>
<strong>Analisi Biometrica Avanzata:</strong> Estrazione dello scheletro posturale, mappatura dei sovraccarichi articolari, analisi angolare della falcata e stima del distretto anatomico a rischio, calcolate direttamente dai keypoint del tuo video.
</div>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    video_file = st.file_uploader("Carica video della corsa (Profilo laterale consigliato, MP4/MOV)", type=["mp4", "mov", "avi"])
with col_h2:
    altezza_cm = st.number_input("La tua altezza (cm) — serve a calibrare le misure", min_value=120, max_value=220, value=175, step=1)

if video_file is not None:
    contenuto = video_file.getvalue()
    firma = hashlib.md5(contenuto).hexdigest() + f"|{altezza_cm}"
    if st.session_state.get('cv_firma') != firma:
        st.session_state['cv_firma'] = firma
        st.session_state['cv_analizzato'] = False
        st.session_state.pop('cv_dati', None)
        st.session_state.pop('cv_media', None)

    # Dati salvati da una versione vecchia del modulo: scartali
    if st.session_state.get('cv_analizzato', False):
        if CHIAVI_OK - set(st.session_state.get('cv_dati', {}).keys()) or 'frames' not in (st.session_state.get('cv_media') or {}):
            st.session_state['cv_analizzato'] = False
            st.session_state.pop('cv_dati', None)
            st.session_state.pop('cv_media', None)

    if not st.session_state.get('cv_analizzato', False):
        if st.button("ELABORA SCHELETRO E STIMA INFORTUNIO", use_container_width=True):
            video_path = None
            ok = False
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(contenuto)
                    video_path = tfile.name
                with st.spinner("Estrazione scheletro (MediaPipe Pose) e calcolo biomeccanico in corso..."):
                    dati_estratti = analizza_running_video(video_path, altezza_cm=altezza_cm)
                    st.session_state.cv_dati = stima_rischio(dati_estratti)
                    # Video con scheletro disegnato (None se non riesce: si mostra l'originale)
                    try:
                        st.session_state.cv_media = genera_video_scheletro(
                            video_path, st.session_state.cv_dati, altezza_cm=altezza_cm)
                    except Exception as e_media:
                        st.session_state.cv_media = {"ok": False, "errore": str(e_media)}
                    st.session_state.cv_analizzato = True
                    ok = True
            except ValueError as e:
                st.error(f"Analisi non riuscita: {e}")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione del video: {e}")
            finally:
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
            if ok:
                st.rerun()

    if st.session_state.get('cv_analizzato', False):
        dati_cv = st.session_state.cv_dati
        st.success(
            f"Analisi video completata. Scheletro tracciato lato {dati_cv['lato_analizzato']}, "
            f"{dati_cv['fps_video']} FPS, appoggio analizzato al frame {dati_cv['frame_strike_analizzato']}."
        )
        st.markdown("---")

        st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:4px; letter-spacing:0.1em;'>KINEMATIC WIREFRAME // GAIT ANALYSIS FRAME</p>", unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Frame Rate", f"{dati_cv['fps_video']} FPS", "Sorgente video")
        mc2.metric("Lato Analizzato", dati_cv['lato_analizzato'], "MediaPipe Pose")
        mc3.metric("Fase", "Strike", f"Frame {dati_cv['frame_strike_analizzato']}")
        st.markdown("<p style='font-size:0.85em; color:#8792A3; margin-top:8px; margin-bottom:16px;'>Tracciamento articolare e analisi vettoriale basata sui dati reali del video:</p>", unsafe_allow_html=True)

        # Video con scheletro (o originale se la generazione non e' riuscita)
        media = st.session_state.get('cv_media') or {}
        verticale = media.get('ok') and media['h'] > media['w']
        ar = f"aspect-ratio:{media['w']} / {media['h']};" if media.get('ok') else "aspect-ratio:9 / 16;"
        st.markdown(
            "<style>div[data-testid='stVideo'] video{width:100%; height:auto; max-height:960px; "
            "object-fit:contain; background:#000; border-radius:12px; " + ar + "}</style>",
            unsafe_allow_html=True
        )
        h_twin, h_grf = (270, 175) if verticale else (380, 235)
        analisi = costruisci_analisi(media['serie'], dati_cv) if media.get('serie') else {}
        col_out1, col_out2 = st.columns([1.05, 0.95] if verticale else [1, 1.1])

        with col_out1:
            if media.get('ok'):
                st.video(media['bytes'], format=media['mime'])
            else:
                if media.get('errore'):
                    st.error(f"Video con scheletro non generato: {media['errore']}")
                if st.button("Rigenera video con scheletro", key="cv_retry", use_container_width=True):
                    st.session_state['cv_analizzato'] = False
                    st.session_state.pop('cv_dati', None)
                    st.session_state.pop('cv_media', None)
                    st.rerun()
                st.video(contenuto)
            frames = media.get('frames') or []
            if frames:
                with st.expander("Il video non parte? Scorri i fotogrammi con lo scheletro", expanded=not media.get('ok')):
                    idx = st.slider("Fotogramma", 0, len(frames) - 1, 0, key="cv_frame_idx",
                                    format="%d", label_visibility="collapsed")
                    st.image(frames[idx][1], caption=f"t = {frames[idx][0]:.2f} s", use_container_width=True)
            st.markdown("<p style='font-size:0.75em; color:#00F5A0; text-align:center; font-family:\"JetBrains Mono\",monospace; margin-top:10px;'>OUTPUT: AI TRACKING COMPLETATO</p>", unsafe_allow_html=True)
        with col_out2:
            dati_REALI = dati_cv
            st.markdown("<p style='font-size:0.82em; color:#00E5FF; font-family:\"JetBrains Mono\",monospace; margin-bottom:6px; letter-spacing:0.1em;'>DIGITAL TWIN // KINEMATIC STRESS MAP (REALE)</p>", unsafe_allow_html=True)
            digital_twin_real_svg = f"""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 320" style="background: radial-gradient(circle at center, #0B111A 0%, #04070B 100%); border-radius: 12px; border: 1px solid #1c2333; width: 100%; height: 100%; box-shadow: 0 8px 30px rgba(0,229,255,0.08);">
              <defs>
                <filter id="glow-red-real" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <pattern id="grid-real" width="25" height="25" patternUnits="userSpaceOnUse">
                  <path d="M 25 0 L 0 0 0 25" fill="none" stroke="#1c2333" stroke-width="0.5" opacity="0.3"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid-real)" />
              <g transform="translate(40, -15)">
                  <path d="M 180 60 C 190 100, 205 140, 215 180 C 200 190, 185 160, 175 110 Z" fill="#00E5FF" opacity="0.25"/>
                  <path d="M 180 60 C 190 100, 205 140, 215 180" fill="none" stroke="#00E5FF" stroke-width="3"/>
                  <path d="M 215 195 Q 205 260, 220 310" fill="none" stroke="#00E5FF" stroke-width="10" opacity="0.2" stroke-linecap="round"/>
                  <circle cx="215" cy="188" r="14" fill="#0E1420" stroke="#00E5FF" stroke-width="2"/>
                  <path d="M 200 240 Q 190 280, 205 310" fill="none" stroke="#FFB020" stroke-width="6" opacity="0.7"/>
                  <polygon points="210,310 230,325 260,325 250,305" fill="#0E1420" stroke="#00E5FF" stroke-width="1.5" opacity="0.8"/>
              </g>
              <circle cx="255" cy="173" r="16" fill="#FF6A3D" opacity="0.5" filter="url(#glow-red-real)"/>
              <circle cx="255" cy="173" r="5" fill="#FFFFFF"/>
              <polyline points="255,173 320,115 560,115" fill="none" stroke="#FF6A3D" stroke-width="1.5"/>
              <rect x="330" y="93" width="235" height="42" fill="#0A0F17" stroke="#FF6A3D" stroke-width="1" rx="4"/>
              <text x="342" y="109" fill="#FF6A3D" font-family="monospace" font-size="10" font-weight="bold">GINOCCHIO: {dati_REALI['angolo_ginocchio_appoggio']}°</text>
              <text x="342" y="123" fill="#8792A3" font-family="monospace" font-size="8">Angolo critico estratto dal video</text>
              <circle cx="240" cy="285" r="12" fill="#FFB020" opacity="0.6" filter="url(#glow-red-real)"/>
              <circle cx="240" cy="285" r="4" fill="#FFFFFF"/>
              <polyline points="240,285 320,225 560,225" fill="none" stroke="#FFB020" stroke-width="1.5"/>
              <rect x="330" y="203" width="235" height="42" fill="#0A0F17" stroke="#FFB020" stroke-width="1" rx="4"/>
              <text x="342" y="219" fill="#FFB020" font-family="monospace" font-size="10" font-weight="bold">OVERSTRIDE: {dati_REALI['overstride_cm']} CM</text>
              <text x="342" y="233" fill="#8792A3" font-family="monospace" font-size="8">Anticipo falcata rilevato dal tracking</text>
              <rect x="20" y="240" width="150" height="70" fill="#0E1420" stroke="#1c2333" stroke-width="1" rx="6"/>
              <text x="28" y="256" fill="#8792A3" font-family="monospace" font-size="7">RISCHIO STIMATO VIDEO</text>
              <text x="28" y="282" fill="#FF6A3D" font-family="monospace" font-size="20" font-weight="bold">{dati_REALI['probabilita_infortunio_ml']}%</text>
              <text x="28" y="298" fill="#FF6A3D" font-family="monospace" font-size="7">STATUS: {"CRITICO" if dati_REALI['probabilita_infortunio_ml'] > 60 else "MODERATO" if dati_REALI['probabilita_infortunio_ml'] > 35 else "BASSO"}</text>
            </svg>
            """
            st.components.v1.html(digital_twin_real_svg, height=h_twin, scrolling=False)
            st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px; margin-bottom:12px;'><strong>Spiegazione:</strong> Mappa anatomica vettoriale guidata dai dati biometrici reali del video, che evidenzia i distretti sottoposti a picchi di stress strutturale.</p>", unsafe_allow_html=True)
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            grf_real_svg = f"""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200" style="background: #080B12; border-radius: 12px; border: 1px solid #1c2333; width: 100%; height: 100%;">
                <defs>
                    <linearGradient id="grfGradReal" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#FF6A3D" stop-opacity="0.5"/>
                        <stop offset="100%" stop-color="#FF6A3D" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <line x1="50" y1="150" x2="570" y2="150" stroke="#566178" stroke-width="1.5"/>
                <line x1="50" y1="40" x2="570" y2="40" stroke="#1c2333" stroke-width="1" stroke-dasharray="3,3"/>
                <text x="15" y="44" fill="#8792A3" font-family="monospace" font-size="8">3.0 BW</text>
                <text x="15" y="154" fill="#8792A3" font-family="monospace" font-size="8">0.0 BW</text>
                <path d="M 50 150 C 150 150, 200 65, 310 65 C 420 65, 470 150, 550 150" fill="none" stroke="#00E5FF" stroke-width="2" stroke-dasharray="4,4" opacity="0.6"/>
                <text x="430" y="55" fill="#00E5FF" font-family="monospace" font-size="8">Standard Ideale</text>
                <path d="M 50 150 L 90 150 L 130 20 L 170 90 C 240 90, 360 55, 470 150 L 550 150" fill="url(#grfGradReal)" />
                <path d="M 50 150 L 90 150 L 130 20 L 170 90 C 240 90, 360 55, 470 150 L 550 150" fill="none" stroke="#FF6A3D" stroke-width="2.5" stroke-linejoin="round"/>
                <circle cx="130" cy="20" r="5" fill="#FFFFFF" stroke="#FF6A3D" stroke-width="2"/>
                <line x1="130" y1="20" x2="190" y2="20" stroke="#FF6A3D" stroke-width="1"/>
                <rect x="195" y="10" width="195" height="20" fill="#0A0F17" stroke="#FF6A3D" stroke-width="1" rx="3"/>
                <text x="202" y="24" fill="#FF6A3D" font-family="monospace" font-size="9" font-weight="bold">OVERSTRIDE: {dati_REALI['overstride_cm']} CM</text>
                <text x="270" y="185" fill="#566178" font-family="monospace" font-size="8">TEMPO DI CONTATTO (ms)</text>
            </svg>
            """
            st.components.v1.html(grf_real_svg, height=h_grf, scrolling=False)
            st.markdown("<p style='font-size:0.75em; color:#8792A3; margin-top:2px;'><strong>Spiegazione:</strong> Grafico illustrativo delle forze d'impatto al suolo (GRF), stimato a partire dall'overstride e dall'angolo di ginocchio rilevati nel video.</p>", unsafe_allow_html=True)
            if 'ginocchio' in analisi:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("### Angolo del ginocchio nel tempo")
                st.plotly_chart(style_fig(analisi['ginocchio'][0]), use_container_width=True)
                st.markdown(f"<div class='explain-text'>{analisi['ginocchio'][1]}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>Report Biomeccanico e Scheletrico Dettagliato</h2>", unsafe_allow_html=True)
        c_met1, c_met2, c_met3, c_met4 = st.columns(4)
        c_met1.metric("Angolo Ginocchio", f"{dati_cv['angolo_ginocchio_appoggio']:.1f}°", "Target > 150°")
        c_met2.metric("Inclinazione Busto", f"{dati_cv['angolo_inclinazione_busto']:.1f}°", "Ottimale 5-8°")
        c_met3.metric("Overstride (Anticipo)", f"{dati_cv['overstride_cm']:.1f} cm", "Target < 10cm")
        c_met4.metric("Oscillazione Vert.", f"{dati_cv['oscillazione_verticale']:.1f} cm", "Target < 8cm")
        st.markdown("<br>", unsafe_allow_html=True)
        cg1, cg2, cg3 = st.columns(3)

        with cg1:
            st.markdown("### 1. Mappatura Sovraccarico (%)")
            fig_bar_load = px.bar(
                x=dati_cv['articolazioni_carico'], y=dati_cv['carichi_pct'],
                labels={'x': 'Distretto', 'y': '% Impatto'},
                color=dati_cv['carichi_pct'], color_continuous_scale=[[0, '#00E5FF'], [0.5, '#FFB020'], [1, '#FF6A3D']]
            )
            fig_bar_load.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_bar_load), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Analisi Carichi:</strong> Stima euristica dello stress relativo sui distretti articolari, calcolata da overstride, angolo di ginocchio, tipo di appoggio e inclinazione del busto misurati nel video.</div>", unsafe_allow_html=True)
        with cg2:
            st.markdown("### 2. Angoli Articolari (Falcata)")
            coppie = [(f, a) for f, a in zip(dati_cv['fasi_gait'], dati_cv['angoli_fase']) if a is not None]
            fig_radar_angles = go.Figure(go.Scatterpolar(
                r=[a for _, a in coppie] + ([coppie[0][1]] if coppie else []),
                theta=[f for f, _ in coppie] + ([coppie[0][0]] if coppie else []),
                fill='toself',
                marker=dict(color='#00F5A0'), line=dict(color='#00F5A0')
            ))
            fig_radar_angles.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[60, 190], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')),
                height=320
            )
            st.plotly_chart(style_fig(fig_radar_angles), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Analisi Angolare:</strong> Angolo anca-ginocchio-caviglia nelle quattro fasi del ciclo del passo (contatto, massima flessione in appoggio, stacco, oscillazione), individuate dal tracking.</div>", unsafe_allow_html=True)
        with cg3:
            st.markdown("### 3. Rischio per Distretto (%)")
            fig_ml_risk = px.bar(
                x=dati_cv['distretti_rischio'], y=dati_cv['rischi_ml'],
                labels={'x': 'Patologia/Distretto', 'y': 'Peso relativo (%)'},
                color=dati_cv['rischi_ml'], color_continuous_scale=[[0, '#00F5A0'], [0.5, '#FFB020'], [1, '#FF6A3D']]
            )
            fig_ml_risk.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_ml_risk), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Stima del Rischio:</strong> Punteggio euristico calcolato dalle feature biomeccaniche reali. Non è un modello clinico addestrato su dataset di infortuni reali.</div>", unsafe_allow_html=True)

        if analisi:
            st.markdown("---")
            st.markdown("<h2>Analisi Temporale Estratta dal Video</h2>", unsafe_allow_html=True)
            righe_graf = [
                [("oscillazione", "Rimbalzo verticale del bacino"), ("busto", "Inclinazione del busto"), ("piedi", "Altezza dei piedi e cadenza")],
                [("appoggio", "Distribuzione dell'appoggio"), ("simmetria", "Simmetria destra/sinistra")],
            ]
            for riga in righe_graf:
                cols = st.columns(len(riga))
                for col, (chiave, titolo) in zip(cols, riga):
                    if chiave in analisi:
                        with col:
                            st.markdown(f"### {titolo}")
                            st.plotly_chart(style_fig(analisi[chiave][0]), use_container_width=True)
                            st.markdown(f"<div class='explain-text'>{analisi[chiave][1]}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h3>Diagnosi Posturale, Errori e Stima del Rischio</h3>", unsafe_allow_html=True)

        st.error(f"PATTERN RILEVATO — {dati_cv['tipo_appoggio']}: angolo del ginocchio al contatto di {dati_cv['angolo_ginocchio_appoggio']}° e overstride di {dati_cv['overstride_cm']} cm, misurati sul frame di appoggio del video.")
        st.warning(f"ZONA DI SOVRACCARICO PRINCIPALE: {dati_cv['sovraccarico_prevalente']}, in base alla distribuzione di carico calcolata sopra.")
        st.markdown(f"""
        <div class='danger-box' style='border-left-color: #FF6A3D;'>
            <h3 style='color: #FF6A3D; margin-top:0;'>STIMA DI RISCHIO (Indice: {dati_cv['probabilita_infortunio_ml']}%)</h3>
            <p style='color: #E8ECF2; font-size: 1.05em;'>In base al pattern di overstride e al tipo di appoggio rilevati nel video, il distretto con il punteggio di rischio relativo più alto è: <strong style='color: #FF6A3D;'>{dati_cv['infortunio_predetto']}</strong>. Questa è una stima euristica orientativa, non una diagnosi medica.</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("PROTOCOLLO DI CORREZIONE BIOMECCANICA CONSIGLIATO:\n1. Riduzione dell'ampiezza della falcata per eliminare l'over-stride anteriore al baricentro.\n2. Incremento della frequenza di passo a 176-180 falcate al minuto (SPM) per facilitare l'atterraggio sul mesopiede.\n3. Integrazione di esercizi di forza eccentrica per il quadricipite e protocollo di rinforzo progressivo per il tendine d'Achille.\n4. Consulta un fisioterapista o un tecnico di corsa per una valutazione clinica completa prima di modificare la tecnica.")
else:
    st.info("Suggerimento: Carica un video registrato lateralmente (profilo laterale, corridore ben visibile per almeno 5-10 secondi) per attivare l'estrazione dello scheletro e l'analisi biomeccanica reale.")
