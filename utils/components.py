import base64
import math
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
# pannelli "vetro" blu, linee con gradiente blu->ciano, glow morbido,
# micro-animazioni. Ogni scena racconta una sezione dell'app.
# =========================================================

C_DEEP, C_NAVY, C_BLUE = "#0B1F3F", "#12386B", "#1B5FA8"
C_MAIN, C_SKY, C_CYAN, C_TEXT = "#2F8FE0", "#7EC8FF", "#00E5FF", "#E8F6FF"
SANS = "Inter, 'Segoe UI', Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, Menlo, monospace"


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
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    """


def _svg(body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">'
            f'<defs>{_defs()}</defs>'
            f'<ellipse cx="600" cy="250" rx="600" ry="250" fill="url(#glow)" opacity="0.45"/>'
            f'{body}</svg>')


def _t(x, y, s, size=26, fill=C_TEXT, weight=500, anchor="start", mono=False, op=1.0):
    fam = MONO if mono else SANS
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" fill-opacity="{op}" text-anchor="{anchor}">{s}</text>')


def _panel(x, y, w, h, rx=24):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="url(#panelG)" '
            f'stroke="{C_MAIN}" stroke-opacity="0.45" stroke-width="2"/>')


def _dot(x, y, r=6, color=C_CYAN, anim=True):
    a = ('<animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/>'
         if anim else '')
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" filter="url(#softGlow)">{a}</circle>'


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


def _sparkline(x, y, w, h, vals, color="url(#lineG)"):
    lo, hi = min(vals), max(vals)
    pts = [(x + w * i / (len(vals) - 1), y + h - (v - lo) / (hi - lo) * h) for i, v in enumerate(vals)]
    line = _smooth(pts)
    area = line + f" L{x + w},{y + h} L{x},{y + h} Z"
    return (f'<path d="{area}" fill="url(#areaG)"/>'
            f'<path d="{line}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="6" fill="{C_CYAN}" filter="url(#softGlow)"/>')


# ---------------------------------------------------------
# HOME — mappa GPS del percorso con le metriche di corsa
# ---------------------------------------------------------
def _scene_home():
    cx, cy, R = 380, 250, 215
    route = [(215, 330), (262, 296), (300, 318), (345, 285), (372, 236), (335, 200),
             (352, 160), (410, 140), (470, 168), (492, 225), (465, 275), (500, 320)]
    d = _smooth(route)
    ecg = "M790,300"
    for p in range(4):
        x0 = 790 + p * 80
        ecg += (f" L{x0 + 22},300 L{x0 + 30},288 L{x0 + 38},326 L{x0 + 46},262 "
                f"L{x0 + 54},306 L{x0 + 62},300 L{x0 + 80},300")
    km = "".join(_dot(*route[i], r=9, color=C_SKY, anim=False) for i in (3, 6, 9))
    body = f"""
    <circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#glow)"/>
    <circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#panelG)" stroke="{C_MAIN}" stroke-opacity="0.5" stroke-width="2.5"/>
    <clipPath id="disc"><circle cx="{cx}" cy="{cy}" r="{R - 2}"/></clipPath>
    <g clip-path="url(#disc)">
        {_grid(160, 30, 600, 470, 40, 0.07)}
        <g fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="1.6">{_contours(430, 190, 40, 7, 34, 1.0)}</g>
        <path d="{d}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="{d}" fill="none" stroke="url(#lineG)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="{d}" fill="none" stroke="{C_TEXT}" stroke-opacity="0.9" stroke-width="2" stroke-dasharray="2 16" stroke-linecap="round">
            <animate attributeName="stroke-dashoffset" values="0;-36" dur="1.6s" repeatCount="indefinite"/>
        </path>
        {km}
        <circle cx="215" cy="330" r="12" fill="{C_DEEP}" stroke="{C_SKY}" stroke-width="4"/>
    </g>
    <g transform="translate(500,320)">
        <path d="M0,0 C-14,-18 -26,-30 -26,-46 A26,26 0 1 1 26,-46 C26,-30 14,-18 0,0 Z" fill="{C_CYAN}" filter="url(#softGlow)"/>
        <circle cx="0" cy="-46" r="10" fill="{C_DEEP}"/>
    </g>
    <g stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7" fill="none">
        <path d="M585,205 C640,190 660,140 700,130"/>
        <path d="M590,265 C650,265 700,275 760,275"/>
        <path d="M570,330 C620,380 650,410 700,415"/>
    </g>
    {_panel(700, 70, 380, 120)}
    {_t(730, 112, "PASSO MEDIO", 22, C_SKY, 600, op=0.85)}
    {_t(730, 170, "4:32", 58, C_TEXT, 700, mono=True)}
    {_t(900, 170, "/km", 28, C_SKY, 500, mono=True)}
    {_panel(760, 215, 380, 120)}
    {_t(790, 250, "FREQUENZA CARDIACA", 22, C_SKY, 600, op=0.85)}
    <path d="{ecg}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.35" stroke-width="9" stroke-linejoin="round"/>
    <path d="{ecg}" fill="none" stroke="url(#lineG)" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
    {_panel(700, 360, 380, 110)}
    {_t(730, 400, "DISTANZA", 22, C_SKY, 600, op=0.85)}
    {_t(730, 452, "12.4", 52, C_TEXT, 700, mono=True)}
    {_t(865, 452, "km", 28, C_SKY, 500, mono=True)}
    """
    return _svg(body)


# ---------------------------------------------------------
# ANALISI STATO DI FORMA — anello di prontezza + biometria
# ---------------------------------------------------------
def _scene_analisi():
    cx, cy = 360, 250
    heart = "M0,22 C-46,-8 -40,-44 -18,-44 C-8,-44 -2,-38 0,-30 C2,-38 8,-44 18,-44 C40,-44 46,-8 0,22 Z"
    rows = [
        (60, "HRV", "62", "ms", [48, 55, 52, 60, 57, 66, 62], -35),
        (190, "SMA", "0.41", "", [0.30, 0.35, 0.33, 0.38, 0.36, 0.40, 0.41], 0),
        (320, "SONNO", "7h20", "", [6.2, 7.0, 6.6, 7.4, 6.9, 7.1, 7.33], 35),
    ]
    parts = []
    for (y, lab, val, unit, series, ang) in rows:
        a = math.radians(ang)
        sx, sy = cx + 190 * math.cos(a), cy + 190 * math.sin(a)
        ty = y + 55
        parts.append(f'<path d="M{sx:.1f},{sy:.1f} C{sx + 80:.1f},{sy:.1f} {640:.1f},{ty} 700,{ty}" '
                     f'fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>')
        parts.append(_dot(round(sx, 1), round(sy, 1), 6))
        parts.append(_panel(700, y, 420, 110))
        parts.append(_t(730, y + 40, lab, 22, C_SKY, 600, op=0.85))
        parts.append(_t(730, y + 92, val, 46, C_TEXT, 700, mono=True))
        if unit:
            parts.append(_t(730 + 28 * len(val) + 12, y + 92, unit, 26, C_SKY, 500, mono=True))
        parts.append(_sparkline(920, y + 22, 170, 62, series))
    body = f"""
    <circle cx="{cx}" cy="{cy}" r="260" fill="url(#glow)"/>
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="{C_NAVY}" stroke-width="28"/>
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="{C_MAIN}" stroke-opacity="0.25" stroke-width="40" stroke-linecap="round"
            pathLength="100" stroke-dasharray="76 100" transform="rotate(-90 {cx} {cy})"/>
    <circle cx="{cx}" cy="{cy}" r="190" fill="none" stroke="url(#lineG)" stroke-width="28" stroke-linecap="round"
            pathLength="100" stroke-dasharray="76 100" transform="rotate(-90 {cx} {cy})"/>
    <circle cx="{cx}" cy="{cy}" r="150" fill="none" stroke="{C_SKY}" stroke-opacity="0.4" stroke-width="2" stroke-dasharray="2 11">
        <animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="360 {cx} {cy}" dur="40s" repeatCount="indefinite"/>
    </circle>
    <circle cx="{cx}" cy="{cy}" r="128" fill="url(#panelG)" stroke="{C_MAIN}" stroke-opacity="0.4" stroke-width="2"/>
    <g transform="translate({cx},{cy - 50})">
        <path d="{heart}" fill="url(#lineG)" filter="url(#softGlow)">
            <animateTransform attributeName="transform" type="scale" values="1;1.1;1" dur="1.2s" repeatCount="indefinite"/>
        </path>
    </g>
    {_t(cx, cy + 50, "76", 74, C_TEXT, 700, "middle", mono=True)}
    {_t(cx, cy + 90, "FORMA", 24, C_SKY, 600, "middle", op=0.9)}
    {"".join(parts)}
    """
    return _svg(body)


# ---------------------------------------------------------
# STATISTICHE — carico settimanale + zone di intensità
# ---------------------------------------------------------
def _scene_stats():
    vals = [120, 160, 140, 200, 180, 230, 170, 250, 210, 280, 240, 300]
    avg = [150, 155, 165, 185, 200, 205, 212, 225, 235, 250, 262, 275]
    base = 410
    bars, pts = [], []
    for i, v in enumerate(vals):
        x = 100 + i * 53
        hi = i == len(vals) - 1
        fill = "url(#barHiG)" if hi else "url(#barG)"
        flt = ' filter="url(#softGlow)"' if hi else ""
        bars.append(f'<rect x="{x}" y="{base - v}" width="36" height="{v}" rx="8" fill="{fill}"{flt}/>')
        pts.append((x + 18, base - avg[i] - 22))
    segs = [18, 32, 28, 15, 7]
    cols = [C_NAVY, C_BLUE, C_MAIN, C_SKY, C_CYAN]
    donut, off = [], 0.0
    for v, c in zip(segs, cols):
        a0 = math.radians(-90 + (off + 0.6) * 3.6)
        a1 = math.radians(-90 + (off + v - 0.6) * 3.6)
        x0, y0 = 970 + 105 * math.cos(a0), 235 + 105 * math.sin(a0)
        x1, y1 = 970 + 105 * math.cos(a1), 235 + 105 * math.sin(a1)
        large = 1 if (a1 - a0) > math.pi else 0
        donut.append(f'<path d="M{x0:.1f},{y0:.1f} A105,105 0 {large} 1 {x1:.1f},{y1:.1f}" '
                     f'fill="none" stroke="{c}" stroke-width="36"/>')
        off += v
    legend = "".join(
        f'<circle cx="{885 + i * 42}" cy="392" r="9" fill="{c}"/>'
        + _t(885 + i * 42, 436, f"Z{i + 1}", 20, C_SKY, 600, "middle", op=0.85)
        for i, c in enumerate(cols))
    body = f"""
    {_panel(60, 50, 700, 400)}
    {_grid(90, 100, 730, 410, 50, 0.07)}
    <line x1="85" y1="{base}" x2="735" y2="{base}" stroke="{C_SKY}" stroke-opacity="0.4" stroke-width="2"/>
    {"".join(bars)}
    <path d="{_smooth(pts)}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="12" stroke-linecap="round"/>
    <path d="{_smooth(pts)}" fill="none" stroke="{C_TEXT}" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="1 9">
        <animate attributeName="stroke-dashoffset" values="0;-20" dur="1.4s" repeatCount="indefinite"/>
    </path>
    <rect x="590" y="62" width="126" height="40" rx="20" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-opacity="0.7" stroke-width="2"/>
    {_t(653, 90, "12.4 km", 22, C_CYAN, 700, "middle", mono=True)}
    {_panel(800, 50, 340, 400)}
    <circle cx="970" cy="235" r="140" fill="url(#glow)"/>
    <circle cx="970" cy="235" r="105" fill="none" stroke="{C_DEEP}" stroke-width="40"/>
    {"".join(donut)}
    {_t(970, 242, "52", 56, C_TEXT, 700, "middle", mono=True)}
    {_t(970, 276, "sessioni", 24, C_SKY, 500, "middle", op=0.9)}
    {legend}
    """
    return _svg(body)


# ---------------------------------------------------------
# KPI DASHBOARD — contagiri + smartwatch
# ---------------------------------------------------------
def _scene_kpi():
    gx, gy, gr = 400, 330, 230
    ticks = []
    for i in range(21):
        a = math.pi + math.pi * i / 20
        major = i % 5 == 0
        r1, r2 = 258, 282 if major else 272
        ticks.append(f'<line x1="{gx + r1 * math.cos(a):.1f}" y1="{gy + r1 * math.sin(a):.1f}" '
                     f'x2="{gx + r2 * math.cos(a):.1f}" y2="{gy + r2 * math.sin(a):.1f}" '
                     f'stroke="{C_SKY}" stroke-opacity="{0.8 if major else 0.4}" stroke-width="{3 if major else 2}" stroke-linecap="round"/>')
    na = math.pi + math.pi * 0.824
    nx, ny = gx + 185 * math.cos(na), gy + 185 * math.sin(na)
    arc = f"M{gx - gr},{gy} A{gr},{gr} 0 0 1 {gx + gr},{gy}"
    wx, wy = 900, 250
    rings = ""
    for r, p, c in ((54, 82, C_CYAN), (39, 64, C_SKY), (24, 45, C_MAIN)):
        rings += (f'<circle cx="{wx}" cy="{wy}" r="{r}" fill="none" stroke="{c}" stroke-opacity="0.2" stroke-width="11"/>'
                  f'<circle cx="{wx}" cy="{wy}" r="{r}" fill="none" stroke="{c}" stroke-width="11" stroke-linecap="round" '
                  f'pathLength="100" stroke-dasharray="{p} 100" transform="rotate(-90 {wx} {wy})"/>')
    body = f"""
    <circle cx="{gx}" cy="{gy}" r="300" fill="url(#glow)"/>
    {"".join(ticks)}
    <path d="{arc}" fill="none" stroke="{C_NAVY}" stroke-width="30" stroke-linecap="round"/>
    <path d="{arc}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.25" stroke-width="44" stroke-linecap="round" pathLength="100" stroke-dasharray="82.4 100"/>
    <path d="{arc}" fill="none" stroke="url(#lineG)" stroke-width="30" stroke-linecap="round" pathLength="100" stroke-dasharray="82.4 100"/>
    <line x1="{gx}" y1="{gy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{C_TEXT}" stroke-width="5" stroke-linecap="round" filter="url(#softGlow)"/>
    <circle cx="{gx}" cy="{gy}" r="18" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="4"/>
    {_t(gx, gy + 92, "82.4%", 74, C_TEXT, 700, "middle", mono=True)}
    {_t(gx, gy + 132, "INDICE DI FORMA", 24, C_SKY, 600, "middle", op=0.9)}
    <path d="M650,300 C730,300 760,262 812,252" fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>
    {_dot(650, 300, 6)}
    <circle cx="{wx}" cy="{wy}" r="170" fill="url(#glow)"/>
    <rect x="852" y="40" width="96" height="120" rx="20" fill="{C_NAVY}"/>
    <rect x="852" y="340" width="96" height="120" rx="20" fill="{C_NAVY}"/>
    <rect x="985" y="218" width="14" height="40" rx="6" fill="{C_MAIN}"/>
    <rect x="815" y="150" width="170" height="200" rx="50" fill="{C_DEEP}" stroke="{C_MAIN}" stroke-width="5"/>
    <rect x="829" y="164" width="142" height="172" rx="38" fill="#08152B"/>
    {rings}
    {_t(wx, wy + 8, "82", 22, C_TEXT, 700, "middle", mono=True)}
    """
    return _svg(body)


# ---------------------------------------------------------
# ML / PREVISIONE — rete neurale che alimenta la proiezione
# ---------------------------------------------------------
def _scene_ml():
    layers = [(90, [90, 170, 250, 330, 410]), (210, [125, 205, 295, 375]),
              (330, [125, 205, 295, 375]), (440, [250])]
    lines, nodes = [], []
    for li in range(len(layers) - 1):
        for j, ya in enumerate(layers[li][1]):
            for k, yb in enumerate(layers[li + 1][1]):
                hi = (j + k + li) % 4 == 0
                lines.append(f'<line x1="{layers[li][0]}" y1="{ya}" x2="{layers[li + 1][0]}" y2="{yb}" '
                             f'stroke="{C_SKY}" stroke-opacity="{0.6 if hi else 0.16}" stroke-width="{2 if hi else 1.4}"/>')
    for li, (x, ys) in enumerate(layers):
        for j, y in enumerate(ys):
            if li == len(layers) - 1:
                nodes.append(f'<circle cx="{x}" cy="{y}" r="34" fill="url(#glow)"/>'
                             f'<circle cx="{x}" cy="{y}" r="20" fill="{C_CYAN}" filter="url(#softGlow)"/>')
            else:
                on = (j + li) % 3 == 0
                nodes.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{C_CYAN if on else C_DEEP}" stroke="{C_MAIN}" stroke-width="3"/>')
    hist_y = [330, 322, 335, 300, 296, 268, 250, 226, 205, 190, 200, 215, 232]
    hist = [(560 + i * 21.7, y) for i, y in enumerate(hist_y)]
    hline = _smooth(hist)
    harea = hline + " L820,410 L560,410 Z"
    fc = [(820, 232), (880, 255), (940, 275), (1000, 292), (1060, 312), (1110, 322)]
    sp = [0, 10, 20, 30, 40, 48]
    up = [(x, y - s) for (x, y), s in zip(fc, sp)]
    lo = [(x, y + s) for (x, y), s in zip(fc, sp)][::-1]
    cone = _smooth(up) + _smooth(lo, move="L") + " Z"
    body = f"""
    {"".join(lines)}
    {"".join(nodes)}
    <path d="M470,250 L520,250" stroke="url(#lineG)" stroke-width="3" stroke-dasharray="2 8" stroke-linecap="round">
        <animate attributeName="stroke-dashoffset" values="0;-20" dur="1.2s" repeatCount="indefinite"/>
    </path>
    {_panel(520, 50, 640, 400)}
    {_grid(560, 100, 1120, 410, 50, 0.07)}
    <line x1="560" y1="210" x2="1120" y2="210" stroke="{C_SKY}" stroke-opacity="0.5" stroke-width="2" stroke-dasharray="8 8"/>
    {_t(572, 200, "SOGLIA", 20, C_SKY, 600, op=0.85)}
    <path d="{harea}" fill="url(#areaG)"/>
    <path d="{hline}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.3" stroke-width="12" stroke-linecap="round"/>
    <path d="{hline}" fill="none" stroke="url(#lineG)" stroke-width="4.5" stroke-linecap="round"/>
    <line x1="820" y1="95" x2="820" y2="410" stroke="{C_TEXT}" stroke-opacity="0.4" stroke-width="2" stroke-dasharray="3 8"/>
    {_t(820, 86, "OGGI", 20, C_TEXT, 600, "middle", op=0.8)}
    <path d="{cone}" fill="{C_MAIN}" fill-opacity="0.22">
        <animate attributeName="fill-opacity" values="0.15;0.3;0.15" dur="3s" repeatCount="indefinite"/>
    </path>
    <path d="{_smooth(fc)}" fill="none" stroke="url(#lineG)" stroke-width="4.5" stroke-linecap="round" stroke-dasharray="2 12">
        <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.6s" repeatCount="indefinite"/>
    </path>
    {_dot(820, 232, 8)}
    {_t(840, 436, "Rischio overload: in calo", 24, C_CYAN, 600, "middle")}
    """
    return _svg(body)


# ---------------------------------------------------------
# PIANO ALLENAMENTO — la montagna delle fasi, dalla base al picco
# ---------------------------------------------------------
def _ridge(pts, fill, op, bottom=500):
    d = _smooth(pts) + f" L{pts[-1][0]},{bottom} L{pts[0][0]},{bottom} Z"
    return f'<path d="{d}" fill="{fill}" fill-opacity="{op}"/>'


def _scene_plan():
    far = [(0, 330), (120, 270), (210, 310), (330, 220), (450, 300), (560, 250), (700, 330),
           (820, 280), (940, 200), (1060, 290), (1200, 240)]
    mid = [(0, 400), (140, 340), (260, 380), (400, 290), (520, 370), (640, 330), (780, 400),
           (900, 330), (1020, 370), (1200, 310)]
    trail = [(600, 495), (540, 440), (640, 400), (570, 350), (670, 310), (620, 260),
             (700, 230), (668, 190), (700, 125)]
    td = _smooth(trail)
    stars = "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C_TEXT}" fill-opacity="{o}"/>'
                    for x, y, r, o in [(120, 70, 2, .7), (230, 120, 1.6, .5), (330, 50, 2.2, .8), (450, 95, 1.5, .5),
                                       (560, 40, 2, .6), (880, 60, 1.8, .6), (1100, 170, 2, .5), (60, 170, 1.5, .5),
                                       (800, 130, 1.6, .5), (1150, 60, 2.2, .7)])
    phases = [(400, "BASE", trail[2]), (310, "BUILD", trail[4]), (230, "PICCO", trail[6])]
    pills = ""
    for y, name, (px, py) in phases:
        pills += (f'<rect x="60" y="{y - 22}" width="150" height="44" rx="22" fill="{C_DEEP}" fill-opacity="0.85" '
                  f'stroke="{C_MAIN}" stroke-opacity="0.6" stroke-width="2"/>'
                  + _t(135, y + 8, name, 24, C_TEXT, 700, "middle")
                  + f'<path d="M212,{y} L{px - 14},{py}" stroke="{C_SKY}" stroke-opacity="0.5" stroke-width="1.5" stroke-dasharray="2 7"/>'
                  + f'<circle cx="{px}" cy="{py}" r="9" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="4"/>')
    body = f"""
    {stars}
    <circle cx="1000" cy="105" r="90" fill="url(#glow)"/>
    <circle cx="1000" cy="105" r="30" fill="{C_TEXT}" fill-opacity="0.92"/>
    {_ridge(far, C_NAVY, 0.6)}
    {_ridge(mid, C_BLUE, 0.7)}
    <polygon points="330,500 560,260 640,200 700,110 760,190 830,250 1000,500" fill="url(#peakG)" stroke="{C_SKY}" stroke-opacity="0.35" stroke-width="2" stroke-linejoin="round"/>
    <polygon points="700,110 760,190 830,250 1000,500 740,500 720,300" fill="{C_DEEP}" fill-opacity="0.4"/>
    <polygon points="700,110 668,158 690,150 710,164 728,148" fill="{C_TEXT}" fill-opacity="0.85" stroke-linejoin="round"/>
    <path d="{td}" fill="none" stroke="{C_MAIN}" stroke-opacity="0.35" stroke-width="14" stroke-linecap="round"/>
    <path d="{td}" fill="none" stroke="url(#lineG)" stroke-width="5" stroke-linecap="round"/>
    <path d="{td}" fill="none" stroke="{C_TEXT}" stroke-width="2" stroke-dasharray="2 14" stroke-linecap="round">
        <animate attributeName="stroke-dashoffset" values="0;-32" dur="1.8s" repeatCount="indefinite"/>
    </path>
    {pills}
    <line x1="700" y1="125" x2="700" y2="52" stroke="{C_SKY}" stroke-width="4" stroke-linecap="round"/>
    <path d="M700,52 L700,84 L758,68 Z" fill="{C_CYAN}" filter="url(#softGlow)">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
    </path>
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

    def skeleton(joints=True):
        s = f'<line x1="{J["head"][0]}" y1="{J["head"][1]}" x2="{J["neck"][0]}" y2="{J["neck"][1]}"/>'
        for ch in chains:
            pts = " ".join(f"{J[k][0]},{J[k][1]}" for k in ch)
            s += f'<polyline points="{pts}" fill="none"/>'
        return s

    def joint_dots():
        out = f'<circle cx="{J["head"][0]}" cy="{J["head"][1]}" r="22" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="4"/>'
        for k, (x, y) in J.items():
            if k != "head":
                out += f'<circle cx="{x}" cy="{y}" r="8" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-width="3.5"/>'
        return out

    k, h, a = J["knF"], J["hip"], J["anF"]
    v1, v2 = (h[0] - k[0], h[1] - k[1]), (a[0] - k[0], a[1] - k[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    p1 = (k[0] + 42 * v1[0] / n1, k[1] + 42 * v1[1] / n1)
    p2 = (k[0] + 42 * v2[0] / n2, k[1] + 42 * v2[1] / n2)
    sector = (f"M{k[0]},{k[1]} L{p1[0]:.1f},{p1[1]:.1f} A42,42 0 0 0 {p2[0]:.1f},{p2[1]:.1f} Z")

    rows = [(115, "ANGOLO GINOCCHIO", f"{ang:.0f}°"), (215, "CARICO TIBIA", "nominale"), (315, "CADENZA", "176 spm")]
    rp = ""
    for i, (y, lab, val) in enumerate(rows):
        rp += _t(790, y, lab, 20, C_SKY, 600, op=0.85)
        rp += _t(790, y + 50, val, 42 if i != 1 else 36, C_TEXT, 700, mono=True)
        if i < 2:
            rp += f'<line x1="790" y1="{y + 78}" x2="1110" y2="{y + 78}" stroke="{C_SKY}" stroke-opacity="0.2" stroke-width="1.5"/>'
    br = ""
    x0, x1, y0, y1, L = 262, 560, 62, 432, 34
    for (cx_, cy_, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        br += f'<path d="M{cx_},{cy_ + dy * L} L{cx_},{cy_} L{cx_ + dx * L},{cy_}" fill="none"/>'
    body = f"""
    <ellipse cx="400" cy="440" rx="330" ry="22" fill="{C_MAIN}" fill-opacity="0.18"/>
    <line x1="60" y1="424" x2="740" y2="424" stroke="{C_SKY}" stroke-opacity="0.3" stroke-width="2"/>
    <g transform="translate(110,0)">
        <g transform="translate(-140,0)" stroke="{C_MAIN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.12">{skeleton()}</g>
        <g transform="translate(-70,0)" stroke="{C_MAIN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3">{skeleton()}</g>
        <g stroke="{C_MAIN}" stroke-opacity="0.35" stroke-width="16" stroke-linecap="round" stroke-linejoin="round">{skeleton()}</g>
        <g stroke="url(#lineG)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round">{skeleton()}</g>
        <path d="{sector}" fill="{C_CYAN}" fill-opacity="0.28" stroke="{C_CYAN}" stroke-width="2"/>
        {joint_dots()}
        <g stroke="{C_CYAN}" stroke-width="4" stroke-linecap="round">{br}</g>
        <rect x="{x0}" y="22" width="168" height="30" rx="15" fill="{C_DEEP}" stroke="{C_CYAN}" stroke-opacity="0.7" stroke-width="2"/>
        {_t(x0 + 84, 44, "ATLETA 98%", 18, C_CYAN, 700, "middle", mono=True)}
    </g>
    <path d="M{k[0] + 110 + 8},{k[1]} C650,{k[1]} 700,150 760,150" fill="none" stroke="{C_SKY}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="2 7"/>
    {_panel(760, 60, 380, 380)}
    {rp}
    """
    return _svg(body)


SVG_HOME = _scene_home()
SVG_ANALISI = _scene_analisi()
SVG_STATS = _scene_stats()
SVG_KPI = _scene_kpi()
SVG_ML = _scene_ml()
SVG_PLAN = _scene_plan()
SVG_CV = _scene_cv()
