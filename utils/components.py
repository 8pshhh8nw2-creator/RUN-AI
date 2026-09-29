import base64
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


# ... qui sotto incolla il blocco "LIBRERIA CONDIVISA" con RUNNER_GLOW_DEFS,
# _backdrop, _runner, _ground e tutte le SVG_* ...



# =========================================================
# LIBRERIA CONDIVISA — illustrazioni vettoriali originali, una scena
# elaborata per ogni pagina. Solo SVG puro: nessuna foto e nessuna
# risorsa esterna. Gli helper (_runner, _panel, _stars, _skyline...)
# sono condivisi così lo stile resta coerente in tutta la dashboard.
# =========================================================

import math
import random

RUNNER_GLOW_DEFS = """
    <radialGradient id="runnerGlow" cx="50%" cy="50%" r="60%">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="sunG" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#CFEBFF" stop-opacity="0.9"/>
        <stop offset="45%" stop-color="#2F8FE0" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="skyG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#040A18"/><stop offset="45%" stop-color="#0B1F3F"/>
        <stop offset="75%" stop-color="#12386B"/><stop offset="100%" stop-color="#2F8FE0"/>
    </linearGradient>
    <linearGradient id="groundG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0B1F3F"/><stop offset="100%" stop-color="#040A18"/>
    </linearGradient>
    <linearGradient id="shirtG" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#9BD6FF"/><stop offset="100%" stop-color="#2F8FE0"/>
    </linearGradient>
    <linearGradient id="skinG" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#EBC3A0"/><stop offset="100%" stop-color="#C99672"/>
    </linearGradient>
    <linearGradient id="speedG" x1="1" y1="0" x2="0" y2="0">
        <stop offset="0%" stop-color="#7EC8FF" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#7EC8FF" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="panelBar" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0"/><stop offset="100%" stop-color="#00E5FF"/>
    </linearGradient>
    <linearGradient id="statBarGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
    <linearGradient id="areaG" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.45"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </linearGradient>
"""


# ---------- helper grafici ----------

def _t(x, y, txt, size=14, fill="#E8F6FF", anchor="start", mono=False, weight=400, op=0.9):
    fam = "'JetBrains Mono', monospace" if mono else "Inter, sans-serif"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" opacity="{op}">{txt}</text>')


def _stars(n, seed, x0, x1, y0, y1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(f'<circle cx="{rnd.uniform(x0, x1):.0f}" cy="{rnd.uniform(y0, y1):.0f}" '
                   f'r="{rnd.choice([0.8, 1.1, 1.5])}" fill="#CFEBFF" opacity="{rnd.uniform(0.3, 0.9):.2f}"/>')
    return "".join(out)


def _skyline(x0, x1, base, hmin, hmax, seed, fill="#0A1E3D", lit=True):
    rnd = random.Random(seed)
    out, x = [], x0
    while x < x1:
        w, h = rnd.randint(28, 64), rnd.randint(hmin, hmax)
        out.append(f'<rect x="{x}" y="{base - h}" width="{w}" height="{h}" fill="{fill}"/>')
        if lit:
            for wy in range(base - h + 10, base - 8, 16):
                for wx in range(x + 6, x + w - 8, 12):
                    if rnd.random() < 0.18:
                        out.append(f'<rect x="{wx}" y="{wy}" width="4" height="6" fill="#7EC8FF" opacity="0.7"/>')
        x += w + rnd.randint(0, 6)
    return "".join(out)


def _pines(seed, x0, x1, base, n, fill="#071427"):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        x, h = rnd.uniform(x0, x1), rnd.uniform(28, 62)
        w, y = h * 0.42, base + rnd.uniform(-6, 6)
        out.append(f'<polygon points="{x - w:.0f},{y:.0f} {x:.0f},{y - h:.0f} {x + w:.0f},{y:.0f}" fill="{fill}"/>'
                   f'<polygon points="{x - w * 0.75:.0f},{y - h * 0.4:.0f} {x:.0f},{y - h * 1.2:.0f} '
                   f'{x + w * 0.75:.0f},{y - h * 0.4:.0f}" fill="{fill}"/>')
    return "".join(out)


def _grid_floor(vx, vy, ybot=500, xa=-600, xb=1800, step=150, rows=(14, 34, 62, 104, 152), opacity=0.2):
    out = []
    for x in range(xa, xb, step):
        out.append(f'<line x1="{vx}" y1="{vy}" x2="{x}" y2="{ybot}" stroke="#2F8FE0" stroke-opacity="{opacity}"/>')
    for r in rows:
        out.append(f'<line x1="0" y1="{vy + r}" x2="1200" y2="{vy + r}" stroke="#2F8FE0" stroke-opacity="{opacity}"/>')
    return "".join(out)


def _speedlines(x, y, n=5, length=160, gap=22):
    return "".join(
        f'<rect x="{x - (length - i * 22)}" y="{y + i * gap}" width="{length - i * 22}" height="3.5" rx="1.75" fill="url(#speedG)"/>'
        for i in range(n))


def _panel(x, y, w, h, title=""):
    return f"""<g transform="translate({x},{y})">
        <rect width="{w}" height="{h}" rx="14" fill="#0B1F3F" fill-opacity="0.75" stroke="#2F8FE0" stroke-opacity="0.55"/>
        <rect width="{w}" height="4" rx="2" fill="url(#panelBar)"/>
        <text x="20" y="28" font-family="Inter, sans-serif" font-size="13" letter-spacing="1.6" fill="#7EC8FF" opacity="0.9">{title}</text>
    </g>"""


def _corners(x, y, w, h, l=28, color="#7EC8FF"):
    return (f'<path d="M{x},{y + l} V{y} H{x + l} M{x + w - l},{y} H{x + w} V{y + l} '
            f'M{x + w},{y + h - l} V{y + h} H{x + w - l} M{x + l},{y + h} H{x} V{y + h - l}" '
            f'fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/>')


def _wave(x0, x1, y, amp, cycles, n=80, phase=0.0):
    return " ".join(
        f"{x0 + (x1 - x0) * i / n:.1f},{y + amp * math.sin(2 * math.pi * cycles * i / n + phase):.1f}"
        for i in range(n + 1))


def _ticks(cx, cy, r1, r2, a0, span, n, major=5):
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + span * i / n)
        big = i % major == 0
        ri = r1 - (7 if big else 0)
        out.append(f'<line x1="{cx + ri * math.cos(a):.1f}" y1="{cy + ri * math.sin(a):.1f}" '
                   f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy + r2 * math.sin(a):.1f}" stroke="#7EC8FF" '
                   f'stroke-width="{2 if big else 1}" opacity="{0.85 if big else 0.4}"/>')
    return "".join(out)


def _arc(c, a, b, r):
    a1, a2 = math.atan2(a[1] - c[1], a[0] - c[0]), math.atan2(b[1] - c[1], b[0] - c[0])
    d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    return (f"M{c[0] + r * math.cos(a1):.0f},{c[1] + r * math.sin(a1):.0f} "
            f"A{r},{r} 0 0 {1 if d > 0 else 0} {c[0] + r * math.cos(a2):.0f},{c[1] + r * math.sin(a2):.0f}")


def _bar_chart(x0, base, vals, labels, w=34, gap=20):
    bars, pts, lab = [], [], []
    for i, v in enumerate(vals):
        x = x0 + i * (w + gap)
        bars.append(f'<rect x="{x}" y="{base - v}" width="{w}" height="{v}" rx="5" fill="url(#statBarGrad)"/>')
        pts.append((x + w / 2, base - v - 8))
        lab.append(_t(x + w / 2, base + 18, labels[i], 13, "#7EC8FF", "middle", op=0.8))
    line = " ".join(f"{px:.0f},{py:.0f}" for px, py in pts)
    area = f"{pts[0][0]:.0f},{base} {line} {pts[-1][0]:.0f},{base}"
    dots = "".join(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="4.5" fill="#00E5FF"/>' for px, py in pts)
    return (f'<polygon points="{area}" fill="url(#areaG)"/>' + "".join(bars) +
            f'<polyline points="{line}" fill="none" stroke="#00E5FF" stroke-width="3" '
            f'stroke-linejoin="round" stroke-linecap="round"/>' + dots + "".join(lab))


def _ground(x1, x2, y):
    return f'<ellipse cx="{(x1 + x2) / 2}" cy="{y}" rx="{(x2 - x1) / 2}" ry="12" fill="#000" opacity="0.4"/>'


def _backdrop(cx, cy, rx, ry):
    return f"""
    <ellipse cx="{cx}" cy="{cy}" rx="{rx + 60}" ry="{ry + 60}" fill="url(#runnerGlow)"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#0A1E3D" opacity="0.55"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#2F8FE0"
             stroke-opacity="0.4" stroke-width="1.5" stroke-dasharray="3,9"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx * 0.78}" ry="{ry * 0.78}" fill="none" stroke="#7EC8FF"
             stroke-opacity="0.15"/>
    """


# ---------- il runner ----------
# Figura dettagliata: maglia, pantaloncini, scarpe, fascia, occhiali e
# orologio. Ingombro locale circa 150x235. Scala tipica 1.2 - 1.75.

def _runner(x, y, s=1.0):
    return f"""
    <g transform="translate({x},{y}) scale({s})" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <polyline points="100,74 66,96" stroke="#1B5FA8" stroke-width="16"/>
        <polyline points="66,96 50,122" stroke="#A67B5B" stroke-width="12"/>
        <circle cx="49" cy="124" r="7" fill="#A67B5B" stroke="none"/>
        <polyline points="84,150 56,192" stroke="#A67B5B" stroke-width="22"/>
        <polyline points="56,192 22,174" stroke="#A67B5B" stroke-width="16"/>
        <polyline points="84,150 68,172" stroke="#0B1F3F" stroke-width="25"/>
        <path d="M10,172 L30,160 L40,171 L20,184 Z" fill="#CFEBFF" stroke="#2F8FE0" stroke-width="2"/>
        <path d="M92,62 C104,54 121,60 123,75 C119,102 109,130 101,153 L72,149 C76,118 82,88 92,62 Z"
              fill="url(#shirtG)" stroke="none"/>
        <path d="M97,74 C93,98 89,122 85,146" stroke="#E8F6FF" stroke-width="3" opacity="0.55"/>
        <polyline points="78,149 98,153" stroke="#0B1F3F" stroke-width="22"/>
        <polyline points="88,148 134,172" stroke="#E2B48F" stroke-width="26"/>
        <polyline points="88,150 114,162" stroke="#0B1F3F" stroke-width="30"/>
        <polyline points="134,172 122,224" stroke="#E2B48F" stroke-width="19"/>
        <polyline points="123,212 122,224" stroke="#CFEBFF" stroke-width="21"/>
        <path d="M110,222 L132,222 L154,234 Q160,244 150,245 L112,245 Q104,241 108,231 Z"
              fill="#F2FAFF" stroke="none"/>
        <path d="M108,243 L156,243" stroke="#00E5FF" stroke-width="4"/>
        <path d="M116,228 L136,232" stroke="#2F8FE0" stroke-width="3"/>
        <line x1="108" y1="52" x2="106" y2="64" stroke="#E2B48F" stroke-width="11"/>
        <polyline points="104,74 138,100" stroke="#3D9BEA" stroke-width="17"/>
        <polyline points="138,100 156,72" stroke="#E2B48F" stroke-width="13"/>
        <line x1="148" y1="86" x2="151" y2="81" stroke="#0B1F3F" stroke-width="15"/>
        <line x1="149.5" y1="83.5" x2="150.2" y2="82.4" stroke="#00E5FF" stroke-width="9"/>
        <circle cx="157" cy="70" r="8" fill="#E2B48F" stroke="none"/>
        <circle cx="112" cy="32" r="17" fill="url(#skinG)" stroke="none"/>
        <path d="M95,30 C93,12 120,6 130,24 C121,19 109,20 100,36 Z" fill="#0B1F3F" stroke="none"/>
        <path d="M96,32 C105,25 121,25 129,30" stroke="#00E5FF" stroke-width="4"/>
        <path d="M116,33 L131,31" stroke="#0B1F3F" stroke-width="5"/>
    </g>
    """


# =========================================================
# HOME — alba sulla città: skyline, sole a fasce, strada in prospettiva
# con traccia GPS, pin di destinazione pulsante e pannelli dati.
# =========================================================
_SUN_BANDS = "".join(f'<rect x="820" y="{y}" width="160" height="{h}" fill="#0C2244"/>'
                     for y, h in ((188, 3), (200, 5), (214, 7), (230, 9)))

SVG_HOME = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <clipPath id="sunClip"><circle cx="900" cy="170" r="70"/></clipPath>
    <linearGradient id="homeTrail" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0.95"/>
    </linearGradient>
</defs>
<rect width="1200" height="500" fill="url(#skyG)"/>
{_stars(90, 7, 0, 1200, 0, 240)}
<circle cx="900" cy="170" r="170" fill="url(#sunG)" opacity="0.6"/>
<circle cx="900" cy="170" r="70" fill="#DDF1FF"/>
<g clip-path="url(#sunClip)">{_SUN_BANDS}</g>
{_skyline(0, 1200, 346, 40, 120, 3, "#0E2A54", lit=False)}
{_skyline(0, 1200, 346, 60, 170, 11, "#08152C")}
<rect y="346" width="1200" height="154" fill="url(#groundG)"/>
{_grid_floor(700, 346)}
<polygon points="694,346 706,346 830,500 130,500" fill="#123468" opacity="0.9"/>
<g stroke="#2F8FE0" stroke-width="3" opacity="0.85" filter="url(#softGlow)">
    <line x1="694" y1="346" x2="130" y2="500"/><line x1="706" y1="346" x2="830" y2="500"/>
</g>
<line x1="700" y1="350" x2="480" y2="500" stroke="#CFEBFF" stroke-width="4" stroke-dasharray="16,16" opacity="0.8">
    <animate attributeName="stroke-dashoffset" values="0;-64" dur="1.4s" repeatCount="indefinite"/>
</line>
<path d="M330,500 C470,470 620,440 690,352" fill="none" stroke="url(#homeTrail)" stroke-width="5"
      stroke-dasharray="2,12" stroke-linecap="round"/>
<g transform="translate(700,244) scale(0.8)">
    <path d="M0,0 C-26,0 -46,20 -46,46 C-46,80 0,120 0,120 C0,120 46,80 46,46 C46,20 26,0 0,0 Z"
          fill="#1B5FA8" filter="url(#softGlow)"/>
    <circle cx="0" cy="44" r="16" fill="#0B1F3F"/>
    <circle cx="0" cy="44" r="6" fill="#00E5FF"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/></circle>
</g>
<ellipse cx="700" cy="342" rx="10" ry="4" fill="none" stroke="#00E5FF" stroke-width="2">
    <animate attributeName="rx" values="10;60" dur="2s" repeatCount="indefinite"/>
    <animate attributeName="ry" values="4;14" dur="2s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.9;0" dur="2s" repeatCount="indefinite"/>
</ellipse>
{_ground(300, 560, 470)}
{_speedlines(325, 190)}
{_runner(300, 45, 1.75)}
{_panel(880, 290, 280, 150, "PASSO MEDIO")}
{_t(900, 366, "4:32", 46, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(1010, 366, "/km", 18, "#E8F6FF", op=0.8)}
<polyline points="{_wave(900, 1140, 412, 10, 2.2, phase=0.6)}" fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linejoin="round"/>
{_panel(40, 40, 250, 92, "DISTANZA")}
{_t(60, 106, "12.4 km", 34, "#7EC8FF", mono=True, weight=700, op=1)}
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# ANALISI STATO DI FORMA — scansione biometrica: anelli radar con
# sweep animato, linea di scansione, sensori sul corpo collegati ai
# pannelli HRV (ECG), SMA (movimento) e sonno (ipnogramma).
# =========================================================
_SMA_BARS = "".join(
    f'<rect x="{930 + i * 17}" y="{302 - h}" width="11" height="{h}" rx="3" fill="url(#statBarGrad)"/>'
    for i, h in enumerate([18, 30, 24, 40, 34, 50, 42, 56, 38, 46, 30, 40]))

SVG_ANALISI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<rect width="1200" height="500" fill="#050B1A"/>
{_stars(50, 21, 0, 1200, 0, 500)}
<ellipse cx="457" cy="262" rx="330" ry="270" fill="url(#runnerGlow)"/>
<g fill="none" stroke="#2F8FE0" stroke-opacity="0.4">
    <circle cx="457" cy="262" r="120" stroke-dasharray="3,7"/>
    <circle cx="457" cy="262" r="175"/>
    <circle cx="457" cy="262" r="225" stroke-dasharray="10,8"/>
</g>
<g>
    <path d="M457,262 L457,37 A225,225 0 0 1 616,103 Z" fill="#2F8FE0" opacity="0.16"/>
    <animateTransform attributeName="transform" type="rotate" from="0 457 262" to="360 457 262" dur="8s" repeatCount="indefinite"/>
</g>
{_runner(310, 45, 1.75)}
<rect x="330" y="80" width="290" height="3" rx="1.5" fill="#00E5FF" filter="url(#softGlow)" opacity="0.9">
    <animate attributeName="y" values="80;465;80" dur="4s" repeatCount="indefinite"/>
</rect>
<g fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7">
    <path d="M485,199 C620,199 660,105 780,105"/>
    <path d="M580,175 C680,175 700,255 780,255"/>
    <path d="M524,441 C650,441 680,405 780,405"/>
</g>
<g fill="#00E5FF">
    <circle cx="485" cy="199" r="7" filter="url(#softGlow)"><animate attributeName="r" values="6;11;6" dur="1.1s" repeatCount="indefinite"/></circle>
    <circle cx="580" cy="175" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/></circle>
    <circle cx="524" cy="441" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.6s" begin="0.5s" repeatCount="indefinite"/></circle>
</g>
{_panel(780, 40, 380, 130, "HRV · VARIABILITÀ CARDIACA")}
{_t(800, 125, "62 ms", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(800, 152, "media 7 giorni", 12, op=0.6)}
<polyline points="930,118 958,118 968,104 978,134 990,80 1002,140 1012,118 1050,118 1062,108 1074,124 1086,118 1140,118"
          fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linejoin="round"/>
<circle r="4" fill="#fff" filter="url(#softGlow)"><animateMotion dur="2.2s" repeatCount="indefinite" path="M930,118 L1140,118"/></circle>
{_panel(780, 190, 380, 130, "SMA · INTENSITÀ DI MOVIMENTO")}
{_t(800, 275, "0.41", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_SMA_BARS}
{_panel(780, 340, 380, 130, "SONNO · IPNOGRAMMA")}
{_t(800, 425, "7h20", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(800, 452, "profondo 1h35 · REM 1h50", 12, op=0.6)}
<path d="M900,404 H915 V420 H945 V404 H965 V388 H990 V404 H1005 V420 H1040 V388 H1055 V404 H1085 V388 H1105 V404 H1140"
      fill="none" stroke="#7EC8FF" stroke-width="3" stroke-linejoin="round"/>
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# STATISTICHE — runner e mini-dashboard: istogramma settimanale con
# trend, ciambella delle zone di frequenza, calendario delle sessioni.
# =========================================================
_rnd_hm = random.Random(5)
_HEAT = "".join(
    f'<rect x="{396 + c * 23}" y="{342 + r * 23}" width="18" height="18" rx="4" '
    f'fill="{["#12386B", "#1B5FA8", "#2F8FE0", "#7EC8FF", "#00E5FF"][lv]}" opacity="{0.55 if lv == 0 else 0.95}"/>'
    for c in range(30) for r in range(5)
    for lv in [_rnd_hm.choices(range(5), weights=[34, 18, 22, 16, 10])[0]])

_ZONES = [("Z1", 0.16, "#1B5FA8"), ("Z2", 0.29, "#2F8FE0"), ("Z3", 0.35, "#7EC8FF"), ("Z4+", 0.20, "#00E5FF")]
_C62 = 2 * math.pi * 62
_off, _DONUT, _LEG = 0.0, "", ""
for _i, (_n, _p, _c) in enumerate(_ZONES):
    _DONUT += (f'<circle cx="930" cy="160" r="62" fill="none" stroke="{_c}" stroke-width="22" '
               f'stroke-dasharray="{_C62 * _p:.1f} {_C62:.1f}" stroke-dashoffset="{-_off:.1f}" '
               f'transform="rotate(-90 930 160)"/>')
    _LEG += (f'<rect x="1040" y="{104 + _i * 30}" width="12" height="12" rx="3" fill="{_c}"/>'
             + _t(1060, 115 + _i * 30, f"{_n} {int(_p * 100)}%", 14))
    _off += _C62 * _p

SVG_STATS = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<rect width="1200" height="500" fill="#050B1A"/>
{_stars(40, 4, 0, 1200, 0, 500)}
{_backdrop(195, 265, 165, 205)}
{_ground(90, 300, 442)}
{_runner(60, 70, 1.5)}
{_panel(380, 30, 420, 250, "VOLUME SETTIMANALE")}
<g stroke="#7EC8FF" stroke-opacity="0.15">
    <line x1="396" y1="210" x2="784" y2="210"/><line x1="396" y1="170" x2="784" y2="170"/>
    <line x1="396" y1="130" x2="784" y2="130"/><line x1="396" y1="250" x2="784" y2="250"/>
</g>
{_bar_chart(402, 250, [60, 95, 75, 120, 100, 140, 110], list("LMMGVSD"))}
{_panel(830, 30, 330, 250, "ZONE DI FREQUENZA")}
<circle cx="930" cy="160" r="62" fill="none" stroke="#12386B" stroke-width="22" opacity="0.5"/>
{_DONUT}
{_t(930, 168, "52", 34, "#E8F6FF", "middle", mono=True, weight=700, op=1)}
{_t(930, 190, "sessioni", 13, "#7EC8FF", "middle")}
{_LEG}
{_panel(380, 300, 780, 170, "CALENDARIO SESSIONI")}
{_HEAT}
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# KPI DASHBOARD — indicatore principale a 270° con tacche e lancetta
# animata, tre anelli secondari e trend a 30 giorni.
# =========================================================
_KPI_TREND = [(870 + i * 9.4, 405 - i * 1.6 + 16 * math.sin(i / 2.2)) for i in range(30)]
_KPI_LINE = " ".join(f"{x:.0f},{y:.0f}" for x, y in _KPI_TREND)
_C42 = 2 * math.pi * 42
_RINGS = "".join(
    f'<circle cx="{cx}" cy="130" r="42" fill="none" stroke="#12386B" stroke-width="10"/>'
    f'<circle cx="{cx}" cy="130" r="42" fill="none" stroke="url(#panelBar)" stroke-width="10" stroke-linecap="round" '
    f'stroke-dasharray="{_C42 * v:.1f} {_C42:.1f}" transform="rotate(-90 {cx} 130)"/>'
    + _t(cx, 137, f"{int(v * 100)}%", 20, "#E8F6FF", "middle", mono=True, weight=700, op=1)
    + _t(cx, 200, lab, 13, "#7EC8FF", "middle")
    for cx, v, lab in ((905, 0.74, "Recupero"), (1010, 0.61, "Carico"), (1115, 0.90, "Costanza")))

SVG_KPI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<rect width="1200" height="500" fill="#050B1A"/>
{_stars(45, 9, 0, 1200, 0, 500)}
{_backdrop(190, 265, 150, 205)}
{_ground(80, 300, 447)}
{_runner(50, 60, 1.7)}
<circle cx="640" cy="255" r="200" fill="url(#runnerGlow)"/>
{_ticks(640, 255, 172, 184, 135, 270, 54, major=9)}
<circle cx="640" cy="255" r="150" fill="none" stroke="#12386B" stroke-width="24" stroke-linecap="round"
        stroke-dasharray="707 943" transform="rotate(135 640 255)"/>
<circle cx="640" cy="255" r="150" fill="none" stroke="url(#panelBar)" stroke-width="24" stroke-linecap="round"
        stroke-dasharray="583 943" transform="rotate(135 640 255)" filter="url(#softGlow)"/>
<g transform="rotate(357.5 640 255)">
    <polygon points="640,250 640,260 770,255" fill="#E8F6FF"/>
    <animateTransform attributeName="transform" type="rotate" from="135 640 255" to="357.5 640 255" dur="1.8s" fill="freeze"/>
</g>
<circle cx="640" cy="255" r="11" fill="#0B1F3F" stroke="#00E5FF" stroke-width="3"/>
{_t(640, 322, "82.4%", 50, "#7EC8FF", "middle", mono=True, weight=700, op=1)}
{_t(640, 348, "INDICE DI FORMA", 14, "#E8F6FF", "middle")}
<rect x="590" y="366" width="100" height="26" rx="13" fill="#00E5FF" opacity="0.18" stroke="#00E5FF"/>
{_t(640, 384, "OTTIMO", 13, "#00E5FF", "middle", weight=700, op=1)}
{_RINGS}
{_panel(850, 230, 310, 230, "TREND 30 GIORNI")}
<polygon points="870,455 {_KPI_LINE} 1142,455" fill="url(#areaG)"/>
<polyline points="{_KPI_LINE}" fill="none" stroke="#00E5FF" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{_KPI_TREND[-1][0]:.0f}" cy="{_KPI_TREND[-1][1]:.0f}" r="6" fill="#fff" filter="url(#softGlow)"/>
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# ML / PREVISIONE — dal runner i dati fluiscono in una rete neurale
# (con impulsi animati) e diventano una previsione con cono di
# incertezza, soglia di rischio e importanza delle variabili.
# =========================================================
_XS, _CNT = [400, 500, 600, 700], [5, 6, 6, 2]
_NODES = [[(x, 250 + (i - (n - 1) / 2) * 52) for i in range(n)] for x, n in zip(_XS, _CNT)]
_LINKS = "".join(f'<line x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}"/>'
                 for a, b in zip(_NODES, _NODES[1:]) for x1, y1 in a for x2, y2 in b)
_CIRC = "".join(f'<circle cx="{x}" cy="{y:.0f}" r="12" fill="{"#0B1F3F" if 0 < li < 3 else "#12386B"}" '
                f'stroke="{"#00E5FF" if li == 3 else "#2F8FE0"}" stroke-width="2"/>'
                for li, layer in enumerate(_NODES) for x, y in layer)


def _pulse(idx, dur, begin):
    p = "M" + " L".join(f"{_NODES[l][k][0]},{_NODES[l][k][1]:.0f}" for l, k in enumerate(idx))
    return (f'<circle r="4.5" fill="#00E5FF" filter="url(#softGlow)"><animateMotion dur="{dur}s" '
            f'begin="{begin}s" repeatCount="indefinite" path="{p}"/></circle>')


_PULSES = _pulse((1, 2, 3, 0), 2.4, 0) + _pulse((3, 4, 1, 1), 2.8, 0.7) + _pulse((0, 5, 2, 0), 3.1, 1.2)
_HIST = [(790 + i * 10, 250 - i * 1.5 + 18 * math.sin(i * 0.9)) for i in range(15)]
_FC = [(930 + j * 22, _HIST[-1][1] + j * 4.5 + 4 * math.sin(j)) for j in range(11)]
_H_LINE = " ".join(f"{x:.0f},{y:.0f}" for x, y in _HIST)
_F_LINE = " ".join(f"{x:.0f},{y:.0f}" for x, y in _FC)
_CONE = " ".join(f"{x:.0f},{y - j * 3.4:.0f}" for j, (x, y) in enumerate(_FC)) + " " + \
        " ".join(f"{x:.0f},{y + j * 3.4:.0f}" for j, (x, y) in reversed(list(enumerate(_FC))))
_FEAT = "".join(
    _t(780, 420 + i * 22, lab, 13) +
    f'<rect x="870" y="{411 + i * 22}" width="{v * 2.6:.0f}" height="9" rx="4.5" fill="url(#panelBar)"/>' +
    _t(870 + v * 2.6 + 8, 420 + i * 22, f"{v}%", 12, "#7EC8FF")
    for i, (lab, v) in enumerate((("Carico 7 gg", 80), ("HRV", 62), ("Sonno", 45))))

SVG_ML = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="mlConeG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.35"/><stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.05"/>
    </linearGradient>
</defs>
<rect width="1200" height="500" fill="#050B1A"/>
{_stars(45, 13, 0, 1200, 0, 500)}
{_backdrop(125, 300, 115, 170)}
{_ground(40, 200, 446)}
{_runner(20, 150, 1.2)}
<path d="M215,290 C290,290 330,250 400,250" fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7"/>
<circle r="4" fill="#fff"><animateMotion dur="1.6s" repeatCount="indefinite" path="M215,290 C290,290 330,250 400,250"/></circle>
<circle r="4" fill="#00E5FF"><animateMotion dur="1.6s" begin="0.8s" repeatCount="indefinite" path="M215,290 C290,290 330,250 400,250"/></circle>
<g stroke="#7EC8FF" stroke-opacity="0.16">{_LINKS}</g>
{_CIRC}
{_PULSES}
{_t(400, 452, "INPUT", 12, "#7EC8FF", "middle")}{_t(550, 452, "STRATI NASCOSTI", 12, "#7EC8FF", "middle")}{_t(700, 452, "OUTPUT", 12, "#7EC8FF", "middle")}
{_panel(760, 50, 420, 300, "RISCHIO OVERLOAD · 14 GIORNI")}
<g stroke="#7EC8FF" stroke-opacity="0.15">
    <line x1="790" y1="130" x2="1150" y2="130"/><line x1="790" y1="200" x2="1150" y2="200"/>
    <line x1="790" y1="270" x2="1150" y2="270"/><line x1="790" y1="320" x2="1150" y2="320"/>
</g>
<line x1="790" y1="170" x2="1150" y2="170" stroke="#FFB84D" stroke-width="1.5" stroke-dasharray="6,5"/>
{_t(1150, 163, "soglia rischio", 12, "#FFB84D", "end")}
<line x1="930" y1="96" x2="930" y2="322" stroke="#7EC8FF" stroke-dasharray="4,5" opacity="0.6"/>
{_t(930, 92, "OGGI", 12, "#7EC8FF", "middle", weight=700)}
<polygon points="{_CONE}" fill="url(#mlConeG)"><animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/></polygon>
<polyline points="{_H_LINE}" fill="none" stroke="#E8F6FF" stroke-width="2.5" stroke-linejoin="round"/>
<polyline points="{_F_LINE}" fill="none" stroke="#00E5FF" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="2,10">
    <animate attributeName="stroke-dashoffset" values="0;-24" dur="1.4s" repeatCount="indefinite"/>
</polyline>
<circle cx="930" cy="{_HIST[-1][1]:.0f}" r="6" fill="#fff" filter="url(#softGlow)"/>
{_panel(760, 370, 420, 115, "IMPORTANZA VARIABILI")}
{_FEAT}
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# PIANO ALLENAMENTO — notte di montagna: creste a strati, pini,
# sentiero a tornanti con le fasi (Base, Build, Specifico, Gara),
# bandiera sulla vetta e settimana tipo con i tipi di seduta.
# =========================================================
_DAYS = [("L", "#12386B"), ("M", "#00E5FF"), ("M", "#2F8FE0"), ("G", "#00E5FF"),
         ("V", "#12386B"), ("S", "#7EC8FF"), ("D", "#2F8FE0")]
_WEEK = "".join(
    f'<rect x="{56 + i * 56}" y="72" width="46" height="46" rx="10" fill="{c}" opacity="0.9"/>'
    + _t(79 + i * 56, 101, d, 16, "#040A18" if c in ("#00E5FF", "#7EC8FF") else "#E8F6FF", "middle", weight=700, op=1)
    for i, (d, c) in enumerate(_DAYS))
_LEGEND = "".join(
    f'<circle cx="{x}" cy="150" r="5" fill="{c}"/>' + _t(x + 10, 154, lab, 12, op=0.8)
    for x, c, lab in ((60, "#12386B", "Riposo"), (140, "#2F8FE0", "Facile"), (222, "#00E5FF", "Intervalli"), (320, "#7EC8FF", "Lungo")))

SVG_PLAN = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<rect width="1200" height="500" fill="url(#skyG)"/>
{_stars(110, 17, 0, 1200, 0, 230)}
<circle cx="1040" cy="90" r="90" fill="url(#sunG)" opacity="0.5"/>
<circle cx="1040" cy="90" r="32" fill="#DDF1FF"/>
<circle cx="1052" cy="82" r="28" fill="#0B1F3F" opacity="0.18"/>
<polygon points="0,400 150,250 260,330 400,200 560,340 700,230 860,350 1010,240 1200,360 1200,500 0,500" fill="#0E2A54"/>
<polygon points="0,450 200,330 330,400 520,270 700,400 880,300 1060,410 1200,340 1200,500 0,500" fill="#12386B"/>
<polygon points="380,500 700,140 1020,500" fill="#1B5FA8"/>
<polygon points="700,140 1020,500 800,500" fill="#12386B" opacity="0.5"/>
<polygon points="700,140 656,206 680,196 700,222 724,194 744,208" fill="#E8F6FF" opacity="0.92"/>
<polygon points="0,500 0,432 180,410 360,450 520,430 700,470 900,440 1200,470 1200,500" fill="#0A1E3D"/>
{_pines(2, 0, 1200, 470, 40)}
<polyline points="110,478 340,455 250,425 480,396 390,360 590,332 530,292 660,262 620,226 700,196 700,148"
          fill="none" stroke="#7EC8FF" stroke-width="4" stroke-dasharray="3,10" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
<g fill="#00E5FF" filter="url(#softGlow)">
    <circle cx="340" cy="455" r="7"/><circle cx="590" cy="332" r="7"/><circle cx="660" cy="262" r="7"/>
</g>
<line x1="700" y1="148" x2="700" y2="86" stroke="#E8F6FF" stroke-width="3"/>
<path d="M700,86 L700,114 L748,100 Z" fill="#00E5FF" filter="url(#softGlow)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
</path>
{_t(360, 480, "BASE", 14, "#CFEBFF", weight=700)}
{_t(610, 352, "BUILD", 14, "#CFEBFF", weight=700)}
{_t(680, 278, "SPECIFICO", 14, "#CFEBFF", weight=700)}
{_t(720, 84, "GARA", 15, "#00E5FF", weight=700, op=1)}
{_ground(140, 270, 432)}
{_runner(133, 215, 0.9)}
{_panel(40, 30, 420, 140, "SETTIMANA TIPO")}
{_WEEK}
{_LEGEND}
{_corners(14, 14, 1172, 472)}
</svg>"""


# =========================================================
# COMPUTER VISION — motion tracking: scia di fotogrammi, scheletro con
# giunti, riquadro di rilevamento, angoli articolari e pannelli con
# ginocchio, cadenza e carico sulla tibia.
# =========================================================
_J = dict(head=(112, 32), neck=(106, 58), sh=(102, 72), elN=(138, 100), haN=(156, 72), elF=(66, 96),
          haF=(50, 122), hip=(86, 148), knN=(134, 172), anN=(122, 226), knF=(56, 192), anF=(20, 172))
_CV = {k: (310 + 1.75 * v[0], 45 + 1.75 * v[1]) for k, v in _J.items()}
_BONES = [("head", "neck"), ("neck", "sh"), ("sh", "elN"), ("elN", "haN"), ("sh", "elF"), ("elF", "haF"),
          ("neck", "hip"), ("hip", "knN"), ("knN", "anN"), ("hip", "knF"), ("knF", "anF")]
_CV_BONES = "".join(f'<line x1="{_CV[a][0]:.0f}" y1="{_CV[a][1]:.0f}" x2="{_CV[b][0]:.0f}" y2="{_CV[b][1]:.0f}"/>'
                    for a, b in _BONES)
_CV_JOINTS = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6"/>' for x, y in _CV.values())
_CV_ARCS = "".join(f'<path d="{_arc(_CV[c], _CV[a], _CV[b], r)}"/>' for c, a, b, r in
                   (("knN", "hip", "anN", 34), ("elN", "sh", "haN", 26), ("hip", "neck", "knN", 30)))
_CAD = "".join(
    f'<rect x="{760 + i * 15}" y="{300 - h:.0f}" width="8" height="{h:.0f}" rx="3" fill="url(#statBarGrad)"/>'
    for i, h in ((i, 30 + 22 * (1 + math.sin(i * 0.6))) for i in range(24)))

SVG_CV = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse">
        <rect width="4" height="1" fill="#7EC8FF" opacity="0.07"/>
    </pattern>
</defs>
<rect width="1200" height="500" fill="#050B1A"/>
<rect y="330" width="1200" height="170" fill="url(#groundG)"/>
{_grid_floor(600, 330, rows=(10, 26, 48, 80, 120, 160), opacity=0.22)}
<ellipse cx="457" cy="270" rx="330" ry="260" fill="url(#runnerGlow)"/>
<g opacity="0.10">{_runner(150, 45, 1.75)}</g>
<g opacity="0.20">{_runner(230, 45, 1.75)}</g>
{_ground(330, 600, 474)}
<g opacity="0.55">{_runner(310, 45, 1.75)}</g>
<g stroke="#00E5FF" stroke-width="3.5" stroke-linecap="round" filter="url(#softGlow)">{_CV_BONES}</g>
<g fill="#0B1F3F" stroke="#00E5FF" stroke-width="2.5">{_CV_JOINTS}</g>
<g fill="none" stroke="#FFB84D" stroke-width="2">{_CV_ARCS}</g>
{_corners(318, 66, 282, 412, 30, "#00E5FF")}
<rect x="318" y="38" width="128" height="24" rx="5" fill="#00E5FF"/>
{_t(330, 55, "ATLETA · 98%", 13, "#040A18", mono=True, weight=700, op=1)}
<g fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7">
    <path d="M{_CV['knN'][0]:.0f},{_CV['knN'][1]:.0f} C660,330 680,120 740,105"/>
    <path d="M{_CV['hip'][0]:.0f},{_CV['hip'][1]:.0f} C640,290 690,255 740,250"/>
    <path d="M{_CV['anN'][0]:.0f},{_CV['anN'][1]:.0f} C640,450 690,405 740,400"/>
</g>
{_panel(740, 40, 420, 130, "GINOCCHIO · 128°")}
<line x1="760" y1="120" x2="1140" y2="120" stroke="#7EC8FF" stroke-dasharray="4,5" opacity="0.35"/>
<polyline points="{_wave(760, 1140, 120, 24, 2.5)}" fill="none" stroke="#00E5FF" stroke-width="2.8" stroke-linejoin="round"/>
{_panel(740, 185, 420, 130, "CADENZA · 176 spm")}
{_CAD}
{_panel(740, 330, 420, 140, "CARICO TIBIA · NOMINALE")}
<line x1="760" y1="374" x2="1140" y2="374" stroke="#FFB84D" stroke-width="1.5" stroke-dasharray="6,5"/>
{_t(1140, 366, "limite", 12, "#FFB84D", "end")}
<polygon points="760,455 {_wave(760, 1140, 420, 10, 3)} 1140,455" fill="url(#areaG)"/>
<polyline points="{_wave(760, 1140, 420, 10, 3)}" fill="none" stroke="#7EC8FF" stroke-width="2.8" stroke-linejoin="round"/>
<circle cx="40" cy="42" r="6" fill="#FF5A5F"><animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/></circle>
{_t(56, 47, "REC · 00:12:48 · F3841", 14, "#E8F6FF", mono=True)}
<rect width="1200" height="500" fill="url(#scan)"/>
</svg>"""
