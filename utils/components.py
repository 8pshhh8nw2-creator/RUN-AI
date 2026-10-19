import base64
import math
import random
import streamlit as st
import plotly.io as pio

pio.templates.default = "plotly_dark"
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#B8C2D0")


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT, title_font=dict(family="Space Grotesk, sans-serif", color="#E8ECF2", size=16),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def get_svg_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def header_block(kicker, title, subtitle, image_url=None, image_tag=None):
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    if image_url:
        col_txt, col_img = st.columns([1.4, 1])
        with col_txt:
            st.markdown(f"""
            <div class="app-header">
                <div class="app-kicker"><span class="dot"></span>{kicker}</div>
                <h1 class="hero-title">{title}</h1>
                <p class="hero-sub">{subtitle}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_img:
            st.markdown(f"""
            <div class="hero-media">
                <img src="{image_url}" />
                <div class="tag">{image_tag or ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ILLUSTRAZIONI ORIGINALI (SVG vettoriali, nessuna foto, nessuna
# dipendenza esterna). Sette scene con lo stesso linguaggio visivo:
# pannelli "vetro" blu, linee con gradiente blu->ciano, glow morbido.
#
# Novita' di questa versione:
#  - PROFONDITA': pannelli in vetro con ombra, riflesso e bordo a
#    gradiente; sfondi a livelli; scanline sottile.
#  - ANIMAZIONI: tracciati che si disegnano, barre che crescono,
#    segnali che scorrono, marker che corrono sul percorso, ecc.
#    (SMIL: funziona nei tag <img> di Streamlit; se un renderer non
#    lo supporta, la scena resta comunque completa e statica).
#  - DATI CREDIBILI: split coerenti col passo medio, carico a blocchi
#    3 settimane + 1 di scarico, distribuzione zone ~polarizzata,
#    serie con rumore deterministico (seed fisso -> sempre uguali).
# =========================================================

C_DEEP, C_NAVY, C_BLUE = "#0B1F3F", "#12386B", "#1B5FA8"
C_MAIN, C_SKY, C_CYAN, C_TEXT = "#2F8FE0", "#7EC8FF", "#00E5FF", "#E8F6FF"
SANS = "Inter, 'Segoe UI', Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, Menlo, monospace"

# Dati "di scena" modificabili al volo (restano coerenti tra loro)
N_SESSIONI = 50


def _defs():
    return """
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.38"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="lineG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0"/><stop offset="100%" stop-color="#00E5FF"/>
    </linearGradient>
    <linearGradient id="panelG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#16427A" stop-opacity="0.88"/>
        <stop offset="100%" stop-color="#0B1F3F" stop-opacity="0.92"/>
    </linearGradient>
    <linearGradient id="glassHi" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.16"/>
        <stop offset="45%" stop-color="#FFFFFF" stop-opacity="0.03"/>
        <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="edgeG" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#7EC8FF" stop-opacity="0.75"/>
        <stop offset="50%" stop-color="#2F8FE0" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="barG" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#5BB4F5"/>
    </linearGradient>
    <linearGradient id="barHiG" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#1B5FA8"/><stop offset="100%" stop-color="#00E5FF"/>
    </linearGradient>
    <linearGradient id="areaG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.45"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="peakG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#3FA0F0"/><stop offset="100%" stop-color="#12386B"/>
    </linearGradient>
    <linearGradient id="sweepG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="50%" stop-color="#00E5FF" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <radialGradient id="sphere" cx="35%" cy="30%" r="75%">
        <stop offset="0%" stop-color="#BFE6FF"/>
        <stop offset="45%" stop-color="#2F8FE0"/>
        <stop offset="100%" stop-color="#0B1F3F"/>
    </radialGradient>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="shadow" x="-15%" y="-15%" width="130%" height="145%">
        <feDropShadow dx="0" dy="10" stdDeviation="11" flood-color="#00142E" flood-opacity="0.55"/>
    </filter>
    <filter id="blur8" x="-30%" y="-100%" width="160%" height="300%">
        <feGaussianBlur stdDeviation="8"/>
    </filter>
    <pattern id="scan" width="6" height="6" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0.5" x2="6" y2="0.5" stroke="#7EC8FF" stroke-opacity="0.045"/>
    </pattern>
    """


def _svg(body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">'
            f'<defs>{_defs()}</defs>'
            f'<ellipse cx="600" cy="250" rx="600" ry="250" fill="url(#glow)" opacity="0.45"/>'
            f'<rect x="0" y="0" width="1200" height="500" fill="url(#scan)"/>'
            f'{body}</svg>')


# ---------------------------------------------------------
# Helper di base
# ---------------------------------------------------------
def _t(x, y, s, size=26, fill=C_TEXT, weight=500, anchor="start", mono=False, op=1.0):
    fam = MONO if mono else SANS
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" fill-opacity="{op}" text-anchor="{anchor}">{s}</text>')


def _panel(x, y, w, h, rx=24):
    """Pannello in vetro: ombra, riflesso in alto, bordo a gradiente."""
    return (f'<g filter="url(#shadow)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="url(#panelG)"/></g>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="url(#glassHi)"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="url(#edgeG)" stroke-width="2"/>'
            f'<path d="M{x + rx},{y + 1.5} H{x + w - rx}" stroke="#FFFFFF" stroke-opacity="0.22" '
            f'stroke-width="1.5" stroke-linecap="round"/>')


def _chip_w(text, size=16):
    return round(size * 0.6 * len(text) + 24)


def _chip(x, y, text, color=C_CYAN, size=16, anchor="start"):
    w, h = _chip_w(text, size), size + 12
    x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
    return (f'<rect x="{x0:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="{h / 2:.1f}" fill="{C_DEEP}" '
            f'fill-opacity="0.9" stroke="{color}" stroke-opacity="0.7" stroke-width="1.6"/>'
            + _t(f"{x0 + w / 2:.1f}", f"{y + h / 2 + size * 0.35:.1f}", text, size, color, 700, "middle", mono=True))


def _dot(x, y, r=6, color=C_CYAN, anim=True):
    a = ('<animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/>'
         if anim else '')
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" filter="url(#softGlow)">{a}</circle>'


def _ring_pulse(x, y, r0=6, r1=26, color=C_CYAN, dur=2.2, begin=0.0):
    return (f'<circle cx="{x}" cy="{y}" r="{r0}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="0">'
            f'<animate attributeName="r" values="{r0};{r1}" dur="{dur}s" begin="{begin}s" repeatCount="indefinite"/>'
            f'<animate attributeName="stroke-opacity" values="0.8;0" dur="{dur}s" begin="{begin}s" repeatCount="indefinite"/>'
            f'</circle>')


def _grid(x0, y0, x1, y1, step, op=0.08):
    s = []
    for x in range(x0, x1 + 1, step):
        s.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}"/>')
    for y in range(y0, y1 + 1, step):
        s.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>')
    return f'<g stroke="{C_SKY}" stroke-opacity="{op}" stroke-width="1">{"".join(s)}</g>'


def _smooth(pts, move="M"):
    """Curva morbida (Catmull-Rom -> Bezier) che passa per i punti."""
    d = f"{move}{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d


def _contours(cx, cy, r0, n, gap, seed=0.0):
    """Curve di livello organiche, come su una mappa topografica."""
    out = []
    for i in range(n):
        r = r0 + i * gap
        pts = []
        for k in range(72):
            t = 2 * math.pi * k / 72
            wob = 1 + 0.07 * math.sin(3 * t + seed + i * 0.6) + 0.04 * math.sin(5 * t - seed * 1.3 + i)
            pts.append(f"{cx + r * wob * 1.15 * math.cos(t):.1f},{cy + r * wob * math.sin(t):.1f}")
        out.append(f'<path d="M{" L".join(pts)} Z"/>')
    return "".join(out)


# ---------------------------------------------------------
# Animazione: valore iniziale tenuto per `delay`, poi easing fino a v1.
# Il valore STATICO dell'attributo resta quello finale, quindi senza
# SMIL la scena e' comunque completa.
# ---------------------------------------------------------
def _anim(attr, v0, v1, dur, delay=0.0, ease=True):
    tot = dur + delay
    d = delay / tot
    spl = ' calcMode="spline" keySplines="0 0 1 1;0.25 0.8 0.25 1"' if ease else ''
    return (f'<animate attributeName="{attr}" values="{v0};{v0};{v1}" keyTimes="0;{d:.4f};1" '
            f'dur="{tot:.2f}s"{spl} fill="freeze"/>')


def _stroke_path(d, stroke, width, dur=1.6, delay=0.0, glow=True, extra=""):
    """Tracciato che si 'disegna' da solo (pathLength normalizzato)."""
    g = (f'<path d="{d}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.28" stroke-width="{width * 2.2:.1f}" '
         f'stroke-linecap="round" stroke-linejoin="round" pathLength="1" stroke-dasharray="1 1" stroke-dashoffset="0">'
         f'{_anim("stroke-dashoffset", 1, 0, dur, delay)}</path>') if glow else ''
    return (g + f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" '
                f'stroke-linejoin="round" pathLength="1" stroke-dasharray="1 1" stroke-dashoffset="0" {extra}>'
                f'{_anim("stroke-dashoffset", 1, 0, dur, delay)}</path>')


def _grow_bar(x, base, w, h, fill, delay=0.0, rx=8, flt=""):
    return (f'<rect x="{x}" y="{base - h:.1f}" width="{w}" height="{h:.1f}" rx="{rx}" fill="{fill}"{flt}>'
            f'{_anim("height", 0, round(h, 1), 0.9, delay)}'
            f'{_anim("y", base, round(base - h, 1), 0.9, delay)}</rect>')


def _series(seed, n, start, drift, noise):
    """Serie deterministica con rumore (stesso seed -> stessa curva)."""
    r = random.Random(seed)
    v, out = start, []
    for _ in range(n):
        v += drift + r.uniform(-noise, noise)
        out.append(v)
    return out


def _sparkline(x, y, w, h, vals, color="url(#lineG)", delay=0.0, labels=None):
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = [(x + w * i / (len(vals) - 1), y + h - (v - lo) / rng * h) for i, v in enumerate(vals)]
    line = _smooth(pts)
    area = line + f" L{x + w},{y + h} L{x},{y + h} Z"
    ex, ey = pts[-1]
    out = (f'<path d="{area}" fill="url(#areaG)"/>'
           + _stroke_path(line, color, 4, 1.4, delay)
           + f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="{C_CYAN}" filter="url(#softGlow)"/>'
           + _ring_pulse(f"{ex:.1f}", f"{ey:.1f}", 6, 20, begin=delay))
    if labels:
        for (px, _), lab in zip(pts, labels):
            out += _t(f"{px:.1f}", y + h + 24, lab, 13, C_SKY, 600, "middle", mono=True, op=0.55)
    return out


# ---------------------------------------------------------
# HOME — mappa GPS del percorso con le metriche di corsa
# ---------------------------------------------------------
def _route_points(cx, cy, n=30):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        a = -math.pi / 2 + 2 * math.pi * 0.965 * t
        r = 1 + 0.17 * math.sin(3 * a + 0.6) + 0.09 * math.sin(5 * a + 1.4)
        pts.append((cx + 142 * r * math.cos(a), cy + 100 * r * math.sin(a)))
    return pts


def _scene_home():
    cx, cy, R = 380, 255, 205
    pts = _route_points(cx, cy)
    sx, sy = pts[0]
    ex, ey = pts[-1]
    d = _smooth(pts)
    rel = _smooth([(x - sx, y - sy) for x, y in pts])

    # bussola: tacche attorno al disco
    ticks = []
    for i in range(48):
        a = 2 * math.pi * i / 48 - math.pi / 2
        major = i % 12 == 0
        r1, r2 = R + 7, R + (21 if major else 13)
        ticks.append(f'<line x1="{cx + r1 * math.cos(a):.1f}" y1="{cy + r1 * math.sin(a):.1f}" '
                     f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy + r2 * math.sin(a):.1f}" '
                     f'stroke="{C_SKY}" stroke-opacity="{0.8 if major else 0.35}" stroke-width="{2.5 if major else 1.6}" '
                     f'stroke-linecap="round"/>')

    roads = [
        "M170,205 L300,195 L420,235 L600,215", "M200,335 L330,305 L450,345 L590,315",
        "M335,40 L355,150 L335,265 L375,470", "M470,40 L450,185 L505,305 L485,470",
    ]
    road_svg = "".join(
        f'<path d="{r}" fill="none" stroke="{C_SKY}" stroke-opacity="0.10" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{r}" fill="none" stroke="{C_SKY}" stroke-opacity="0.18" stroke-width="1" stroke-dasharray="6 8"/>'
        for r in roads)
    river = _smooth([(175, 335), (260, 305), (330, 350), (420, 335), (520, 395), (595, 385)])

    pins = ""
    for idx, lab in ((10, "5"), (20, "10")):
        px, py = pts[idx]
        pins += (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="13" fill="{C_DEEP}" stroke="{C_SKY}" stroke-width="3"/>'
                 + _t(f"{px:.1f}", f"{py + 5:.1f}", lab, 14, C_TEXT, 700, "middle", mono=True))

    # split al km: coerenti col passo medio (4:32)
    splits = [280, 278, 276, 274, 273, 272, 272, 271, 271, 269, 268, 260]
    avg = sum(splits) / len(splits)
    pace = f"{int(avg // 60)}:{int(avg % 60):02d}"
    bars = ""
    for i, p in enumerate(splits):
        h = 14 + (282 - p) * 1.9
        hi = i == len(splits) - 1
        bars += _grow_bar(930 + i * 13, 152, 9, h, "url(#barHiG)" if hi else "url(#barG)", 0.5 + i * 0.07, rx=3)

    ecg = "M910,262"
    for p in range(8):
        x0 = 910 + p * 80
        ecg += (f" L{x0 + 22},262 L{x0 + 30},250 L{x0 + 38},288 L{x0 + 46},224 "
                f"L{x0 + 54},268 L{x0 + 62},262 L{x0 + 80},262")
    elev = _series(7, 14, 40, 1.2, 9)

    body = f"""
    <circle cx="{cx}" cy="{cy}" r="{R + 60}" fill="url(#glow)"/>
    {"".join(ticks)}
    {_t(cx, cy - R - 26, "N", 16, C_CYAN, 700, "middle", mono=True)}
    <g filter="url(#shadow)"><circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#panelG)"/></g>
    <circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="url(#edgeG)" stroke-width="2.5"/>
    <clipPath id="disc"><circle cx="{cx}" cy="{cy}" r="{R - 2}"/></clipPath>
    <g clip-path="url(#disc)">
        {_grid(175, 50, 590, 460, 40, 0.06)}
        <ellipse cx="520" cy="155" rx="70" ry="42" fill="{C_MAIN}" fill-opacity="0.10"/>
        <g fill="none" stroke="{C_MAIN}" stroke-opacity="0.28" stroke-width="1.6">{_contours(430, 195, 40, 7, 34, 1.0)}</g>
        <path d="{river}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.18" stroke-width="16" stroke-linecap="round"/>
        {road_svg}
        {_stroke_path(d, "url(#lineG)", 7, 2.4, 0.2)}
        <path d="{d}" fill="none" stroke="{C_TEXT}" stroke-opacity="0.85" stroke-width="2" stroke-dasharray="2 16" stroke-linecap="round">
            <animate attributeName="stroke-dashoffset" values="0;-36" dur="1.6s" repeatCount="indefinite"/>
        </path>
        {pins}
        <circle cx="{sx:.1f}" cy="{sy:.1f}" r="12" fill="{C_DEEP}" stroke="{C_SKY}" stroke-width="4"/>
        <circle cx="{sx:.1f}" cy="{sy:.1f}" r="10" fill="{C_CYAN}" filter="url(#softGlow)">
            <animateMotion dur="14s" begin="2.2s" repeatCount="indefinite" path="{rel}"/>
        </circle>
    </g>
    <g transform="translate({ex:.1f},{ey:.1f})">
        <path d="M0,0 C-14,-18 -26,-30 -26,-46 A26,26 0 1 1 26,-46 C26,-30 14,-18 0,0 Z" fill="{C_CYAN}" filter="url(#softGlow)"/>
        <circle cx="0" cy="-46" r="10" fill="{C_DEEP}"/>
    </g>
    <g stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7" fill="none">
        <path d="M575,150 C640,140 660,110 700,108"/>
        <path d="M585,255 C650,255 690,258 740,258"/>
        <path d="M570,355 C625,385 655,395 700,395"/>
    </g>
    {_panel(700, 40, 400, 135)}
    {_t(730, 78, "PASSO MEDIO", 22, C_SKY, 600, op=0.85)}
    {_t(730, 142, pace, 58, C_TEXT, 700, mono=True)}
    {_t(872, 142, "/km", 26, C_SKY, 500, mono=True)}
    {_t(930, 78, "SPLIT /KM", 15, C_SKY, 600, mono=True, op=0.6)}
    {bars}
    <line x1="926" y1="153" x2="1090" y2="153" stroke="{C_SKY}" stroke-opacity="0.3" stroke-width="1.5"/>
    {_panel(740, 190, 400, 125)}
    {_t(770, 225, "FREQUENZA CARDIACA", 22, C_SKY, 600, op=0.85)}
    {_t(770, 288, "158", 56, C_TEXT, 700, mono=True)}
    {_t(878, 288, "bpm", 24, C_SKY, 500, mono=True)}
    <clipPath id="ecgwin"><rect x="935" y="228" width="185" height="70"/></clipPath>
    <g clip-path="url(#ecgwin)">
        <g>
            <path d="{ecg}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.35" stroke-width="9" stroke-linejoin="round"/>
            <path d="{ecg}" fill="none" stroke="url(#lineG)" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
            <animateTransform attributeName="transform" type="translate" from="0 0" to="-80 0" dur="2.4s" repeatCount="indefinite"/>
        </g>
    </g>
    {_panel(700, 330, 400, 130)}
    {_t(730, 368, "DISTANZA", 22, C_SKY, 600, op=0.85)}
    {_t(730, 428, "12.4", 52, C_TEXT, 700, mono=True)}
    {_t(862, 428, "km", 26, C_SKY, 500, mono=True)}
    {_t(930, 368, "DISLIVELLO  +118 m", 15, C_SKY, 600, mono=True, op=0.6)}
    {_sparkline(930, 380, 155, 48, elev, delay=0.8)}
    """
    return _svg(body)


# ---------------------------------------------------------
# ANALISI STATO DI FORMA — anello di prontezza + biometria
# ---------------------------------------------------------
def _scene_analisi():
    cx, cy = 360, 250
    heart = "M0,22 C-46,-8 -40,-44 -18,-44 C-8,-44 -2,-38 0,-30 C2,-38 8,-44 18,-44 C40,-44 46,-8 0,22 Z"
    score = 76

    ticks = []
    for i in range(60):
        a = math.radians(-90 + 6 * i)
        lit = i < round(score * 0.6)
        ticks.append(f'<line x1="{cx + 222 * math.cos(a):.1f}" y1="{cy + 222 * math.sin(a):.1f}" '
                     f'x2="{cx + 234 * math.cos(a):.1f}" y2="{cy + 234 * math.sin(a):.1f}" '
                     f'stroke="{C_CYAN if lit else C_SKY}" stroke-opacity="{0.85 if lit else 0.2}" '
                     f'stroke-width="{3 if lit else 2}" stroke-linecap="round"/>')
    ea = math.radians(-90 + 3.6 * score)
    endx, endy = cx + 190 * math.cos(ea), cy + 190 * math.sin(ea)

    rows = [
        (60, "HRV · RMSSD", "62", "ms", [54, 58, 51, 60, 63, 59, 62], "▲ 4%", -35),
        (190, "SMA", "0.41", "", [0.30, 0.35, 0.33, 0.38, 0.36, 0.40, 0.41], "▲ .05", 0),
        (320, "SONNO", "7h20", "", [6.4, 7.1, 6.8, 7.6, 7.0, 7.2, 7.33], "+12 min", 35),
    ]
    parts = []
    for n, (y, lab, val, unit, series, delta, ang) in enumerate(rows):
        a = math.radians(ang)
        sx, sy = cx + 190 * math.cos(a), cy + 190 * math.sin(a)
        ty = y + 55
        parts.append(f'<path d="M{sx:.1f},{sy:.1f} C{sx + 80:.1f},{sy:.1f} 640,{ty} 700,{ty}" '
                     f'fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>')
        parts.append(_dot(round(sx, 1), round(sy, 1), 6))
        parts.append(_panel(700, y, 420, 110))
        parts.append(_t(730, y + 36, lab, 22, C_SKY, 600, op=0.85))
        parts.append(_t(730, y + 90, val, 46, C_TEXT, 700, mono=True))
        vend = 730 + 27.6 * len(val)
        if unit:
            parts.append(_t(f"{vend + 10:.1f}", y + 90, unit, 26, C_SKY, 500, mono=True))
            vend += 10 + 15.6 * len(unit)
        parts.append(_chip(f"{vend + 18:.1f}", y + 58, delta, C_CYAN, 16))
        parts.append(_sparkline(955, y + 26, 145, 44, series, delay=0.5 + n * 0.25, labels="LMMGVSD"))

    body = f"""
    <circle cx="{cx}" cy="{cy}" r="270" fill="url(#glow)"/>
    {"".join(ticks)}
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="{C_NAVY}" stroke-width="28"/>
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="{C_MAIN}" stroke-opacity="0.25" stroke-width="40" stroke-linecap="round"
            pathLength="100" stroke-dasharray="{score} 100" transform="rotate(-90 {cx} {cy})"/>
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="url(#lineG)" stroke-width="28" stroke-linecap="round"
            pathLength="1" stroke-dasharray="{score / 100:.2f} 1" transform="rotate(-90 {cx} {cy})">
        <animate attributeName="stroke-dasharray" values="0 1;0 1;{score / 100:.2f} 1" keyTimes="0;0.001;1" dur="1.8s" fill="freeze"
                 calcMode="spline" keySplines="0 0 1 1;0.25 0.8 0.25 1"/>
    </circle>
    {_dot(f"{endx:.1f}", f"{endy:.1f}", 9)}
    {_ring_pulse(f"{endx:.1f}", f"{endy:.1f}", 9, 30)}
    <circle cx="{cx}" cy="{cy}" r="160" fill="none" stroke="{C_SKY}" stroke-opacity="0.35" stroke-width="2" stroke-dasharray="2 11">
        <animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="360 {cx} {cy}" dur="40s" repeatCount="indefinite"/>
    </circle>
    <circle cx="{cx}" cy="{cy}" r="148" fill="none" stroke="{C_CYAN}" stroke-opacity="0.18" stroke-width="1.5" stroke-dasharray="40 24">
        <animateTransform attributeName="transform" type="rotate" from="360 {cx} {cy}" to="0 {cx} {cy}" dur="55s" repeatCount="indefinite"/>
    </circle>
    <g filter="url(#shadow)"><circle cx="{cx}" cy="{cy}" r="128" fill="url(#panelG)"/></g>
    <circle cx="{cx}" cy="{cy}" r="128" fill="url(#glassHi)"/>
    <circle cx="{cx}" cy="{cy}" r="128" fill="none" stroke="url(#edgeG)" stroke-width="2"/>
    <g transform="translate({cx},{cy - 50})">
        <path d="{heart}" fill="url(#lineG)" filter="url(#softGlow)">
            <animateTransform attributeName="transform" type="scale" values="1;1.1;1" dur="1.2s" repeatCount="indefinite"/>
        </path>
    </g>
    {_t(cx - 8, cy + 50, score, 74, C_TEXT, 700, "middle", mono=True)}
    {_t(cx + 50, cy + 50, "/100", 22, C_SKY, 500, "start", mono=True, op=0.8)}
    {_t(cx, cy + 90, "FORMA", 24, C_SKY, 600, "middle", op=0.9)}
    {"".join(parts)}
    """
    return _svg(body)


# ---------------------------------------------------------
# STATISTICHE — carico settimanale + zone di intensita'
# ---------------------------------------------------------
def _scene_stats():
    # km/settimana: blocchi da 3 settimane di carico + 1 di scarico
    vals = [38, 42, 47, 36, 44, 49, 54, 41, 50, 55, 60, 46]
    avg = [sum(vals[max(0, i - 3):i + 1]) / len(vals[max(0, i - 3):i + 1]) for i in range(len(vals))]
    base, k = 410, 4.6
    bars, pts, xl = [], [], []
    for i, v in enumerate(vals):
        x = 112 + i * 52
        hi = i == len(vals) - 1
        fill = "url(#barHiG)" if hi else "url(#barG)"
        flt = ' filter="url(#softGlow)"' if hi else ""
        bars.append(_grow_bar(x, base, 34, v * k, fill, 0.1 + i * 0.07, 8, flt))
        pts.append((x + 17, base - avg[i] * k))
        xl.append(_t(x + 17, 432, f"S{i + 1}", 13, C_SKY, 600, "middle", mono=True, op=0.55))
    ylab = "".join(
        _t(96, base - v * k + 5, str(v), 14, C_SKY, 600, "end", mono=True, op=0.55)
        + f'<line x1="104" y1="{base - v * k:.1f}" x2="732" y2="{base - v * k:.1f}" stroke="{C_SKY}" stroke-opacity="0.09" stroke-width="1"/>'
        for v in (0, 20, 40, 60))

    segs = [14, 52, 17, 11, 6]
    cols = [C_NAVY, C_BLUE, C_MAIN, C_SKY, C_CYAN]
    donut, off = [], 0.0
    for n, (v, c) in enumerate(zip(segs, cols)):
        a0 = math.radians(-90 + (off + 0.6) * 3.6)
        a1 = math.radians(-90 + (off + v - 0.6) * 3.6)
        x0, y0 = 970 + 105 * math.cos(a0), 235 + 105 * math.sin(a0)
        x1, y1 = 970 + 105 * math.cos(a1), 235 + 105 * math.sin(a1)
        large = 1 if (a1 - a0) > math.pi else 0
        dseg = f"M{x0:.1f},{y0:.1f} A105,105 0 {large} 1 {x1:.1f},{y1:.1f}"
        donut.append(_stroke_path(dseg, c, 36, 0.7, 0.3 + n * 0.25, glow=False))
        off += v
    legend = "".join(
        f'<circle cx="{855 + i * 58}" cy="372" r="9" fill="{c}"/>'
        + _t(855 + i * 58, 406, f"Z{i + 1}", 18, C_SKY, 600, "middle", mono=True, op=0.9)
        + _t(855 + i * 58, 430, f"{segs[i]}%", 15, C_TEXT, 600, "middle", mono=True, op=0.7)
        for i, c in enumerate(cols))

    mpts = _smooth(pts)
    body = f"""
    {_panel(60, 50, 700, 400)}
    {_t(90, 88, "CARICO SETTIMANALE · KM", 18, C_SKY, 600, mono=True, op=0.75)}
    <line x1="560" y1="82" x2="590" y2="82" stroke="{C_TEXT}" stroke-width="3" stroke-linecap="round" stroke-dasharray="1 7"/>
    {_t(600, 88, "media mobile 4 sett.", 14, C_SKY, 500, op=0.7)}
    {ylab}
    <line x1="104" y1="{base}" x2="735" y2="{base}" stroke="{C_SKY}" stroke-opacity="0.4" stroke-width="2"/>
    {"".join(bars)}
    {"".join(xl)}
    <path d="{mpts}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="12" stroke-linecap="round"/>
    <path d="{mpts}" fill="none" stroke="{C_TEXT}" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="1 9">
        <animate attributeName="stroke-dashoffset" values="0;-20" dur="1.4s" repeatCount="indefinite"/>
    </path>
    {_chip(pts[-1][0], base - vals[-1] * k - 44, f"{vals[-1]} km", C_CYAN, 18, "middle")}
    {_panel(800, 50, 340, 400)}
    {_t(970, 92, "ZONE DI INTENSITÀ", 17, C_SKY, 600, "middle", mono=True, op=0.75)}
    <circle cx="970" cy="235" r="140" fill="url(#glow)"/>
    <circle cx="970" cy="235" r="105" fill="none" stroke="{C_DEEP}" stroke-width="40"/>
    <circle cx="970" cy="235" r="130" fill="none" stroke="{C_SKY}" stroke-opacity="0.25" stroke-width="1.5" stroke-dasharray="2 10">
        <animateTransform attributeName="transform" type="rotate" from="0 970 235" to="360 970 235" dur="50s" repeatCount="indefinite"/>
    </circle>
    {"".join(donut)}
    {_t(970, 244, N_SESSIONI, 56, C_TEXT, 700, "middle", mono=True)}
    {_t(970, 278, "sessioni", 24, C_SKY, 500, "middle", op=0.9)}
    {legend}
    """
    return _svg(body)


# ---------------------------------------------------------
# KPI DASHBOARD — contagiri + smartwatch + indici
# ---------------------------------------------------------
def _scene_kpi():
    gx, gy, gr = 380, 330, 230
    val = 0.824
    ticks, nums = [], []
    for i in range(41):
        a = math.pi + math.pi * i / 40
        major = i % 10 == 0
        mid = i % 5 == 0
        r1, r2 = 258, (284 if major else (276 if mid else 268))
        ticks.append(f'<line x1="{gx + r1 * math.cos(a):.1f}" y1="{gy + r1 * math.sin(a):.1f}" '
                     f'x2="{gx + r2 * math.cos(a):.1f}" y2="{gy + r2 * math.sin(a):.1f}" '
                     f'stroke="{C_SKY}" stroke-opacity="{0.85 if major else (0.55 if mid else 0.3)}" '
                     f'stroke-width="{3 if major else 2}" stroke-linecap="round"/>')
        if major:
            nums.append(_t(f"{gx + 308 * math.cos(a):.1f}", f"{gy + 308 * math.sin(a) + 6:.1f}",
                           str(i * 5 // 2), 16, C_SKY, 600, "middle", mono=True, op=0.6))
    na = math.pi + math.pi * val
    tipx, tipy = gx + gr * math.cos(na), gy + gr * math.sin(na)
    arc = f"M{gx - gr},{gy} A{gr},{gr} 0 0 1 {gx + gr},{gy}"
    inner = f"M{gx - 190},{gy} A190,190 0 0 1 {gx + 190},{gy}"
    ang = val * 180
    wx, wy = 860, 250
    rings = ""
    for n, (r, p, c) in enumerate(((54, 82, C_CYAN), (39, 64, C_SKY), (24, 45, C_MAIN))):
        rings += (f'<circle cx="{wx}" cy="{wy}" r="{r}" fill="none" stroke="{c}" stroke-opacity="0.2" stroke-width="11"/>'
                  + _stroke_path_circle(wx, wy, r, p / 100, c, 11, 1.4, 0.4 + n * 0.25))
    kpis = [("SMA", 0.41), ("ISLR", 0.58), ("IITR", 0.33), ("IDET", 0.27)]
    kp = ""
    for n, (name, v) in enumerate(kpis):
        y = 140 + n * 72
        kp += (_t(1012, y, name, 18, C_SKY, 600, mono=True, op=0.85)
               + _t(1162, y, f"{v:.2f}", 22, C_TEXT, 700, "end", mono=True)
               + f'<rect x="1012" y="{y + 12}" width="150" height="8" rx="4" fill="{C_DEEP}"/>'
               + _grow_bar(1012, y + 20, 150, 8, "url(#lineG)", 0.4 + n * 0.2, 4).replace(
                   f'height="8.0"', f'height="8"').replace('<rect ', '<rect ', 1)
               )
    # barre KPI orizzontali: la crescita e' sulla larghezza (non sull'altezza)
    kp = ""
    for n, (name, v) in enumerate(kpis):
        y = 140 + n * 72
        w = 150 * v
        kp += (_t(1012, y, name, 18, C_SKY, 600, mono=True, op=0.85)
               + _t(1162, y, f"{v:.2f}", 22, C_TEXT, 700, "end", mono=True)
               + f'<rect x="1012" y="{y + 12}" width="150" height="8" rx="4" fill="{C_DEEP}"/>'
               + f'<rect x="1012" y="{y + 12}" width="{w:.1f}" height="8" rx="4" fill="url(#lineG)">'
               + _anim("width", 0, f"{w:.1f}", 0.9, 0.4 + n * 0.2) + '</rect>')

    body = f"""
    <circle cx="{gx}" cy="{gy}" r="310" fill="url(#glow)"/>
    {"".join(ticks)}
    {"".join(nums)}
    <path d="{arc}" fill="none" stroke="{C_DEEP}" stroke-width="34" stroke-linecap="round"/>
    <path d="{arc}" fill="none" stroke="{C_NAVY}" stroke-width="30" stroke-linecap="round"/>
    <path d="{inner}" fill="none" stroke="{C_SKY}" stroke-opacity="0.3" stroke-width="2" stroke-dasharray="2 9"/>
    <path d="{arc}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.25" stroke-width="46" stroke-linecap="round"
          pathLength="1" stroke-dasharray="{val} 1"/>
    <path d="{arc}" fill="none" stroke="url(#lineG)" stroke-width="30" stroke-linecap="round" pathLength="1" stroke-dasharray="{val} 1">
        <animate attributeName="stroke-dasharray" values="0 1;0 1;{val} 1" keyTimes="0;0.001;1" dur="1.8s" fill="freeze"
                 calcMode="spline" keySplines="0 0 1 1;0.25 0.8 0.25 1"/>
    </path>
    {_dot(f"{tipx:.1f}", f"{tipy:.1f}", 9)}
    {_ring_pulse(f"{tipx:.1f}", f"{tipy:.1f}", 9, 32)}
    <g transform="rotate({ang:.1f} {gx} {gy})">
        <line x1="{gx + 28}" y1="{gy}" x2="{gx - 186}" y2="{gy}" stroke="{C_TEXT}" stroke-width="5" stroke-linecap="round" filter="url(#softGlow)"/>
        <animateTransform attributeName="transform" type="rotate" values="0 {gx} {gy};0 {gx} {gy};{ang:.1f} {gx} {gy}"
                          keyTimes="0;0.001;1" dur="1.8s" fill="freeze" calcMode="spline" keySplines="0 0 1 1;0.25 0.8 0.25 1"/>
    </g>
    <circle cx="{gx}" cy="{gy}" r="22" fill="url(#sphere)" stroke="{C_CYAN}" stroke-width="3"/>
    {_t(gx, gy + 92, "82.4%", 74, C_TEXT, 700, "middle", mono=True)}
    {_t(gx, gy + 132, "INDICE DI FORMA", 24, C_SKY, 600, "middle", op=0.9)}
    {_chip(gx, gy + 148, "▲ +2.1 vs sett. scorsa", C_CYAN, 15, "middle")}
    <path d="M630,300 C690,300 720,262 768,252" fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>
    {_dot(630, 300, 6)}
    <circle cx="{wx}" cy="{wy}" r="170" fill="url(#glow)"/>
    <rect x="807" y="40" width="96" height="120" rx="20" fill="{C_NAVY}"/>
    <rect x="807" y="340" width="96" height="120" rx="20" fill="{C_NAVY}"/>
    <rect x="940" y="218" width="14" height="40" rx="6" fill="{C_MAIN}"/>
    <g filter="url(#shadow)"><rect x="770" y="150" width="170" height="200" rx="50" fill="{C_DEEP}" stroke="{C_MAIN}" stroke-width="5"/></g>
    <rect x="784" y="164" width="142" height="172" rx="38" fill="#08152B"/>
    <path d="M790,196 Q855,158 920,196 L920,224 Q855,184 790,224 Z" fill="url(#glassHi)"/>
    {_t(wx, 186, "07:42", 15, C_SKY, 600, "middle", mono=True, op=0.8)}
    {rings}
    {_t(wx, wy + 8, "82", 22, C_TEXT, 700, "middle", mono=True)}
    {_t(wx, 330, "FC 152", 13, C_CYAN, 700, "middle", mono=True, op=0.85)}
    {_panel(990, 70, 190, 360, 22)}
    {_t(1012, 108, "INDICI RUNAI", 15, C_SKY, 600, mono=True, op=0.6)}
    {kp}
    """
    return _svg(body)


def _stroke_path_circle(cx, cy, r, frac, color, width, dur, delay):
    """Anello di progresso animato che parte dall'alto, in senso orario."""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" pathLength="1" stroke-dasharray="{frac:.3f} 1" transform="rotate(-90 {cx} {cy})">'
            f'<animate attributeName="stroke-dasharray" values="0 1;0 1;{frac:.3f} 1" keyTimes="0;{delay / (dur + delay):.4f};1" '
            f'dur="{dur + delay:.2f}s" fill="freeze" calcMode="spline" keySplines="0 0 1 1;0.25 0.8 0.25 1"/></circle>')


# ---------------------------------------------------------
# ML / PREVISIONE — rete neurale che alimenta la proiezione
# ---------------------------------------------------------
def _scene_ml():
    layers = [(150, [90, 170, 250, 330, 410]), (255, [125, 205, 295, 375]),
              (355, [125, 205, 295, 375]), (450, [250])]
    feats = ["km 7gg", "sonno", "stress", "temp", "HRV"]
    lines, nodes = [], []
    for li in range(len(layers) - 1):
        for j, ya in enumerate(layers[li][1]):
            for k, yb in enumerate(layers[li + 1][1]):
                hi = (j + k + li) % 4 == 0
                lines.append(f'<line x1="{layers[li][0]}" y1="{ya}" x2="{layers[li + 1][0]}" y2="{yb}" '
                             f'stroke="{C_SKY}" stroke-opacity="{0.6 if hi else 0.14}" stroke-width="{2 if hi else 1.3}"/>')
    # segnali che scorrono lungo alcune connessioni
    rnd = random.Random(11)
    pulses = []
    for n in range(12):
        li = rnd.randrange(0, 3)
        ya = rnd.choice(layers[li][1])
        yb = rnd.choice(layers[li + 1][1])
        xa, xb = layers[li][0], layers[li + 1][0]
        pulses.append(f'<circle cx="{xa}" cy="{ya}" r="3.6" fill="{C_CYAN}" filter="url(#softGlow)">'
                      f'<animateMotion dur="{rnd.uniform(1.0, 1.8):.2f}s" begin="{rnd.uniform(0, 2):.2f}s" '
                      f'repeatCount="indefinite" path="M0,0 L{xb - xa},{yb - ya}"/></circle>')
    for li, (x, ys) in enumerate(layers):
        for j, y in enumerate(ys):
            if li == len(layers) - 1:
                nodes.append(f'<circle cx="{x}" cy="{y}" r="36" fill="url(#glow)"/>'
                             f'<circle cx="{x}" cy="{y}" r="20" fill="{C_CYAN}" filter="url(#softGlow)"/>'
                             + _ring_pulse(x, y, 20, 40))
            else:
                on = (j + li) % 3 == 0
                nodes.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{C_CYAN if on else "url(#sphere)"}" '
                             f'stroke="{C_MAIN}" stroke-width="3"/>')
    flab = "".join(_t(124, y + 6, f, 16, C_SKY, 600, "end", mono=True, op=0.7)
                   for f, y in zip(feats, layers[0][1]))
    tlab = (_t(150, 456, "INPUT", 13, C_SKY, 600, "middle", mono=True, op=0.5)
            + _t(305, 456, "HIDDEN", 13, C_SKY, 600, "middle", mono=True, op=0.5)
            + _t(450, 456, "OUTPUT", 13, C_SKY, 600, "middle", mono=True, op=0.5))

    # storico (indice rischio 0-100) e previsione con cono di incertezza
    hist_v = [38, 41, 40, 47, 52, 55, 61, 66, 72, 68, 64, 60, 57]
    yv = lambda v: 410 - 3 * v
    hist = [(560 + i * 21.67, yv(v)) for i, v in enumerate(hist_v)]
    hline = _smooth(hist)
    harea = hline + " L820,410 L560,410 Z"
    fc_v = [57, 53, 49, 46, 44, 43]
    fx = [820, 878, 936, 994, 1052, 1110]
    sp = [0, 4, 7, 10, 12, 14]
    fc = [(x, yv(v)) for x, v in zip(fx, fc_v)]
    up = [(x, yv(v + s)) for x, v, s in zip(fx, fc_v, sp)]
    lo = [(x, yv(v - s)) for x, v, s in zip(fx, fc_v, sp)][::-1]
    cone = _smooth(up) + _smooth(lo, move="L") + " Z"
    xt = "".join(_t(x, 432, lab, 15, C_SKY, 600, "middle", mono=True, op=0.55)
                 for x, lab in ((560, "-12g"), (690, "-6g"), (820, "oggi"), (965, "+3g"), (1110, "+6g")))

    body = f"""
    {"".join(lines)}
    {"".join(pulses)}
    {"".join(nodes)}
    {flab}
    {tlab}
    <path d="M470,250 L520,250" stroke="url(#lineG)" stroke-width="3" stroke-dasharray="2 8" stroke-linecap="round">
        <animate attributeName="stroke-dashoffset" values="0;-20" dur="1.2s" repeatCount="indefinite"/>
    </path>
    {_panel(520, 50, 640, 400)}
    {_t(548, 88, "RISCHIO OVERLOAD", 18, C_SKY, 600, mono=True, op=0.75)}
    {_chip(1135, 68, "▼ IN CALO", C_CYAN, 16, "end")}
    {_grid(560, 110, 1120, 410, 50, 0.07)}
    <line x1="560" y1="{yv(65)}" x2="1120" y2="{yv(65)}" stroke="{C_SKY}" stroke-opacity="0.5" stroke-width="2" stroke-dasharray="8 8"/>
    {_t(1118, yv(65) - 8, "SOGLIA", 15, C_SKY, 600, "end", mono=True, op=0.8)}
    <path d="{harea}" fill="url(#areaG)"/>
    {_stroke_path(hline, "url(#lineG)", 4.5, 1.6, 0.2)}
    <line x1="820" y1="108" x2="820" y2="410" stroke="{C_TEXT}" stroke-opacity="0.4" stroke-width="2" stroke-dasharray="3 8"/>
    {_chip(820, 100, "OGGI", C_TEXT, 14, "middle")}
    <path d="{cone}" fill="{C_MAIN}" fill-opacity="0.22">
        <animate attributeName="fill-opacity" values="0.15;0.3;0.15" dur="3s" repeatCount="indefinite"/>
    </path>
    <path d="{_smooth(fc)}" fill="none" stroke="url(#lineG)" stroke-width="4.5" stroke-linecap="round" stroke-dasharray="2 12">
        <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.6s" repeatCount="indefinite"/>
    </path>
    {_dot(820, yv(57), 8)}
    {_ring_pulse(820, yv(57), 8, 28)}
    {xt}
    """
    return _svg(body)


# ---------------------------------------------------------
# PIANO ALLENAMENTO — la montagna delle fasi, dalla base al picco
# ---------------------------------------------------------
def _ridge(pts, fill, op, bottom=500):
    d = _smooth(pts) + f" L{pts[-1][0]},{bottom} L{pts[0][0]},{bottom} Z"
    return f'<path d="{d}" fill="{fill}" fill-opacity="{op}"/>'


def _pines(xs, base, h=46):
    out = []
    for i, x in enumerate(xs):
        s = h * (0.8 + 0.4 * ((i * 37) % 10) / 10)
        out.append(f'<polygon points="{x},{base - s:.0f} {x - s * 0.30:.0f},{base - s * 0.45:.0f} {x + s * 0.30:.0f},{base - s * 0.45:.0f}"/>'
                   f'<polygon points="{x},{base - s * 0.72:.0f} {x - s * 0.38:.0f},{base - s * 0.12:.0f} {x + s * 0.38:.0f},{base - s * 0.12:.0f}"/>')
    return f'<g fill="{C_DEEP}" fill-opacity="0.92" stroke="{C_MAIN}" stroke-opacity="0.25" stroke-width="1">{"".join(out)}</g>'


def _scene_plan():
    far = [(0, 330), (120, 270), (210, 310), (330, 220), (450, 300), (560, 250), (700, 330),
           (820, 280), (940, 200), (1060, 290), (1200, 240)]
    mid = [(0, 400), (140, 340), (260, 380), (400, 290), (520, 370), (640, 330), (780, 400),
           (900, 330), (1020, 370), (1200, 310)]
    trail = [(600, 495), (540, 440), (640, 400), (570, 350), (670, 310), (620, 260),
             (700, 230), (668, 190), (700, 125)]
    td = _smooth(trail)
    rel = _smooth([(x - trail[0][0], y - trail[0][1]) for x, y in trail])
    rnd = random.Random(5)
    stars = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C_TEXT}" fill-opacity="{o}">'
        f'<animate attributeName="fill-opacity" values="{o};{o * 0.3:.2f};{o}" dur="{rnd.uniform(2, 5):.1f}s" '
        f'begin="{rnd.uniform(0, 3):.1f}s" repeatCount="indefinite"/></circle>'
        for x, y, r, o in [(120, 70, 2, .7), (230, 120, 1.6, .5), (330, 50, 2.2, .8), (450, 95, 1.5, .5),
                           (560, 40, 2, .6), (880, 60, 1.8, .6), (1100, 170, 2, .5), (60, 170, 1.5, .5),
                           (800, 130, 1.6, .5), (1150, 60, 2.2, .7), (180, 30, 1.4, .5), (980, 28, 1.6, .6)])
    phases = [(400, "BASE", "sett. 1-6", trail[2]), (310, "BUILD", "sett. 7-12", trail[4]),
              (230, "PICCO", "sett. 13-16", trail[6])]
    pills = ""
    for y, name, rng_, (px, py) in phases:
        pills += (f'<g filter="url(#shadow)"><rect x="60" y="{y - 28}" width="156" height="56" rx="24" fill="{C_DEEP}" fill-opacity="0.9"/></g>'
                  f'<rect x="60" y="{y - 28}" width="156" height="56" rx="24" fill="url(#glassHi)"/>'
                  f'<rect x="60" y="{y - 28}" width="156" height="56" rx="24" fill="none" stroke="url(#edgeG)" stroke-width="2"/>'
                  + _t(138, y + 2, name, 22, C_TEXT, 700, "middle")
                  + _t(138, y + 20, rng_, 13, C_SKY, 600, "middle", mono=True, op=0.75)
                  + f'<path d="M218,{y} L{px - 14},{py}" stroke="{C_SKY}" stroke-opacity="0.5" stroke-width="1.5" stroke-dasharray="2 7"/>'
                  + f'<circle cx="{px}" cy="{py}" r="9" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="4"/>')
    body = f"""
    <ellipse cx="640" cy="120" rx="520" ry="110" fill="{C_MAIN}" fill-opacity="0.07" filter="url(#blur8)"/>
    {stars}
    <circle cx="1000" cy="105" r="95" fill="url(#glow)"/>
    <circle cx="1000" cy="105" r="30" fill="{C_TEXT}" fill-opacity="0.92"/>
    <circle cx="990" cy="98" r="6" fill="{C_SKY}" fill-opacity="0.35"/><circle cx="1010" cy="116" r="4" fill="{C_SKY}" fill-opacity="0.3"/>
    {_ridge(far, C_NAVY, 0.6)}
    <path d="{_smooth(far)}" fill="none" stroke="{C_SKY}" stroke-opacity="0.18" stroke-width="2"/>
    {_ridge(mid, C_BLUE, 0.7)}
    <path d="{_smooth(mid)}" fill="none" stroke="{C_SKY}" stroke-opacity="0.25" stroke-width="2"/>
    <g filter="url(#shadow)"><polygon points="330,500 560,260 640,200 700,110 760,190 830,250 1000,500" fill="url(#peakG)"/></g>
    <polygon points="330,500 560,260 640,200 700,110 760,190 830,250 1000,500" fill="none" stroke="{C_SKY}" stroke-opacity="0.35" stroke-width="2" stroke-linejoin="round"/>
    <polygon points="700,110 760,190 830,250 1000,500 740,500 720,300" fill="{C_DEEP}" fill-opacity="0.4"/>
    <g fill="none" stroke="{C_SKY}" stroke-opacity="0.2" stroke-width="1.5" stroke-linejoin="round">
        <polyline points="700,110 668,190 640,300 600,430"/><polyline points="700,110 735,220 770,330 800,430"/>
        <polyline points="640,200 668,190 700,230 735,220 760,190"/>
    </g>
    <polygon points="700,110 668,158 690,150 710,164 728,148" fill="{C_TEXT}" fill-opacity="0.85" stroke-linejoin="round"/>
    {_stroke_path(td, "url(#lineG)", 5, 2.2, 0.2)}
    <path d="{td}" fill="none" stroke="{C_TEXT}" stroke-width="2" stroke-dasharray="2 14" stroke-linecap="round">
        <animate attributeName="stroke-dashoffset" values="0;-32" dur="1.8s" repeatCount="indefinite"/>
    </path>
    <circle cx="{trail[0][0]}" cy="{trail[0][1]}" r="8" fill="{C_CYAN}" filter="url(#softGlow)">
        <animateMotion dur="9s" begin="2.4s" repeatCount="indefinite" path="{rel}"/>
    </circle>
    <g filter="url(#blur8)" fill="{C_SKY}" fill-opacity="0.13">
        <ellipse cx="300" cy="430" rx="280" ry="22"><animateTransform attributeName="transform" type="translate" values="0 0;40 0;0 0" dur="14s" repeatCount="indefinite"/></ellipse>
        <ellipse cx="900" cy="445" rx="300" ry="20"><animateTransform attributeName="transform" type="translate" values="0 0;-40 0;0 0" dur="17s" repeatCount="indefinite"/></ellipse>
    </g>
    {_pines(list(range(18, 330, 27)), 506, 52)}
    {_pines(list(range(930, 1200, 27)), 506, 52)}
    {pills}
    <line x1="700" y1="125" x2="700" y2="52" stroke="{C_SKY}" stroke-width="4" stroke-linecap="round"/>
    <path d="M700,52 L700,84 L758,68 Z" fill="{C_CYAN}" filter="url(#softGlow)">
        <animate attributeName="d" values="M700,52 L700,84 L758,68 Z;M700,52 L700,84 L752,72 Z;M700,52 L700,84 L758,68 Z" dur="1.6s" repeatCount="indefinite"/>
    </path>
    {_t(772, 74, "GARA", 20, C_CYAN, 700, mono=True)}
    {_t(772, 94, "sett. 16", 13, C_SKY, 600, mono=True, op=0.75)}
    """
    return _svg(body)


# ---------------------------------------------------------
# COMPUTER VISION — scheletro biomeccanico con scia di movimento
# ---------------------------------------------------------
def _scene_cv():
    J = dict(head=(448, 92), neck=(436, 132), sh=(432, 150), hip=(398, 268),
             elF=(478, 204), wrF=(524, 176), elB=(382, 200), wrB=(344, 232),
             knF=(470, 322), anF=(440, 398), toF=(474, 404),
             knB=(350, 338), anB=(290, 362), toB=(314, 388))
    chains = [("neck", "sh", "hip"), ("sh", "elF", "wrF"), ("sh", "elB", "wrB"),
              ("hip", "knF", "anF", "toF"), ("hip", "knB", "anB", "toB")]

    def skeleton():
        s = f'<line x1="{J["head"][0]}" y1="{J["head"][1]}" x2="{J["neck"][0]}" y2="{J["neck"][1]}"/>'
        for ch in chains:
            pts = " ".join(f"{J[k][0]},{J[k][1]}" for k in ch)
            s += f'<polyline points="{pts}" fill="none"/>'
        return s

    def joint_dots():
        out = (f'<circle cx="{J["head"][0]}" cy="{J["head"][1]}" r="22" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="4"/>')
        for k, (x, y) in J.items():
            if k != "head":
                out += f'<circle cx="{x}" cy="{y}" r="8" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="3.5"/>'
        for n, k in enumerate(("hip", "knF", "anF", "sh")):
            out += _ring_pulse(J[k][0], J[k][1], 8, 24, C_CYAN, 2.4, n * 0.5)
        return out

    k, h, a = J["knF"], J["hip"], J["anF"]
    v1, v2 = (h[0] - k[0], h[1] - k[1]), (a[0] - k[0], a[1] - k[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    p1 = (k[0] + 42 * v1[0] / n1, k[1] + 42 * v1[1] / n1)
    p2 = (k[0] + 42 * v2[0] / n2, k[1] + 42 * v2[1] / n2)
    sector = f"M{k[0]},{k[1]} L{p1[0]:.1f},{p1[1]:.1f} A42,42 0 0 0 {p2[0]:.1f},{p2[1]:.1f} Z"

    rows = [(115, "ANGOLO GINOCCHIO", f"{ang:.0f}°"), (200, "TEMPO DI CONTATTO", "218 ms"), (285, "CADENZA", "176 spm")]
    rp = ""
    for i, (y, lab, val) in enumerate(rows):
        rp += _t(790, y, lab, 20, C_SKY, 600, op=0.85)
        rp += _t(790, y + 46, val, 40, C_TEXT, 700, mono=True)
        rp += f'<line x1="790" y1="{y + 66}" x2="1110" y2="{y + 66}" stroke="{C_SKY}" stroke-opacity="0.2" stroke-width="1.5"/>'

    # onda dell'oscillazione verticale che scorre
    wave_pts = [(790 + i * 10, 398 + 15 * math.sin(2 * math.pi * i * 10 / 80)
                 + 5 * math.sin(4 * math.pi * i * 10 / 80 + 1.0)) for i in range(50)]
    wd = _smooth(wave_pts)

    br = ""
    x0, x1, y0, y1, L = 262, 560, 62, 432, 34
    for (cx_, cy_, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        br += f'<path d="M{cx_},{cy_ + dy * L} L{cx_},{cy_} L{cx_ + dx * L},{cy_}" fill="none"/>'

    speed = "".join(
        f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="{C_SKY}" stroke-opacity="{o}" stroke-width="2" stroke-linecap="round" stroke-dasharray="40 30">'
        f'<animate attributeName="stroke-dashoffset" values="0;70" dur="{d}s" repeatCount="indefinite"/></line>'
        for xa, xb, y, o, d in [(60, 300, 150, 0.14, 1.1), (90, 290, 212, 0.2, 0.9), (50, 300, 272, 0.12, 1.3),
                                (100, 270, 332, 0.18, 1.0), (70, 280, 382, 0.14, 1.2)])

    body = f"""
    <ellipse cx="400" cy="440" rx="330" ry="22" fill="{C_MAIN}" fill-opacity="0.18"/>
    {speed}
    <line x1="60" y1="424" x2="740" y2="424" stroke="{C_SKY}" stroke-opacity="0.3" stroke-width="2"/>
    <line x1="60" y1="436" x2="740" y2="436" stroke="{C_SKY}" stroke-opacity="0.25" stroke-width="3" stroke-dasharray="26 18">
        <animate attributeName="stroke-dashoffset" values="0;44" dur="0.8s" repeatCount="indefinite"/>
    </line>
    <g transform="translate(110,0)">
        <g transform="translate(-140,0)" stroke="{C_MAIN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.12">{skeleton()}</g>
        <g transform="translate(-70,0)" stroke="{C_MAIN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3">{skeleton()}</g>
        <g stroke="{C_MAIN}" stroke-opacity="0.35" stroke-width="16" stroke-linecap="round" stroke-linejoin="round">{skeleton()}</g>
        <g stroke="url(#lineG)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round">{skeleton()}</g>
        <path d="{sector}" fill="{C_CYAN}" fill-opacity="0.28" stroke="{C_CYAN}" stroke-width="2"/>
        <ellipse cx="{J["toF"][0]}" cy="426" rx="8" ry="3" fill="none" stroke="{C_CYAN}" stroke-width="2">
            <animate attributeName="rx" values="8;46" dur="1.4s" repeatCount="indefinite"/>
            <animate attributeName="ry" values="3;10" dur="1.4s" repeatCount="indefinite"/>
            <animate attributeName="stroke-opacity" values="0.8;0" dur="1.4s" repeatCount="indefinite"/>
        </ellipse>
        {joint_dots()}
        <rect x="{x0}" y="{y0}" width="{x1 - x0}" height="3" fill="url(#sweepG)">
            <animate attributeName="y" values="{y0};{y1};{y0}" dur="5s" repeatCount="indefinite"/>
        </rect>
        <g stroke="{C_CYAN}" stroke-width="4" stroke-linecap="round">{br}</g>
        <rect x="{x0}" y="22" width="168" height="30" rx="15" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-opacity="0.7" stroke-width="2"/>
        {_t(x0 + 84, 44, "ATLETA 98%", 18, C_CYAN, 700, "middle", mono=True)}
    </g>
    <path d="M{k[0] + 110 + 8},{k[1]} C650,{k[1]} 700,150 760,150" fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>
    {_panel(760, 60, 380, 380)}
    {rp}
    {_t(790, 352, "OSCILLAZIONE VERTICALE", 15, C_SKY, 600, mono=True, op=0.6)}
    {_chip(1110, 336, "8.4 cm", C_CYAN, 14, "end")}
    <clipPath id="wavewin"><rect x="790" y="364" width="320" height="60"/></clipPath>
    <g clip-path="url(#wavewin)">
        <g>
            <path d="{wd}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="9" stroke-linecap="round"/>
            <path d="{wd}" fill="none" stroke="url(#lineG)" stroke-width="3.5" stroke-linecap="round"/>
            <animateTransform attributeName="transform" type="translate" from="0 0" to="-80 0" dur="2s" repeatCount="indefinite"/>
        </g>
    </g>
    """
    return _svg(body)


SVG_HOME = _scene_home()
SVG_ANALISI = _scene_analisi()
SVG_STATS = _scene_stats()
SVG_KPI = _scene_kpi()
SVG_ML = _scene_ml()
SVG_PLAN = _scene_plan()
SVG_CV = _scene_cv()
