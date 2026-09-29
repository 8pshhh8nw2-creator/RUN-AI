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


# Rete di sicurezza: azzera qualsiasi cornice/sfondo/bordo che il CSS globale
# potesse applicare al contenitore dell'immagine (è la causa tipica del "rettangolo").
_HERO_RESET_CSS = """
<style>
.hero-media, .hero-media > div, .hero-media img {
    background: none !important; background-color: transparent !important;
    border: none !important; outline: none !important; box-shadow: none !important;
    border-radius: 0 !important; backdrop-filter: none !important; padding: 0 !important;
}
.hero-media::before, .hero-media::after { display: none !important; content: none !important; }
</style>
"""


def header_block(kicker, title, subtitle, image_url=None, image_tag=None):
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    if image_url:
        st.markdown(_HERO_RESET_CSS, unsafe_allow_html=True)
        col_txt, col_img = st.columns([1.1, 1.5])
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
            <div class="hero-media" style="background:transparent;border:none;box-shadow:none;padding:0;">
                <img src="{image_url}" style="display:block;width:100%;background:transparent;border:none;border-radius:0;mix-blend-mode:screen;" />
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
# LIBRERIA CONDIVISA
# ---------------------------------------------------------
# * Il runner è un vero modello 3D: arti e busto sono tubi rastremati a sezione
#   ellittica (con i rigonfiamenti dei muscoli), testa a ellissoide, scarpe.
#   Ogni faccia è un triangolo ombreggiato con luce direzionale (diffusa,
#   speculare, bagliore sui bordi) e ordinato per profondità. Vista a 3/4.
# * Ogni scena è divisa in SCENARIO (passa da una maschera ellittica sfumata,
#   quindi non ha mai bordi o angoli) e FG (runner, pannelli, HUD).
# =========================================================

# ---------- geometria 3D ----------
_K = 1.2                      # scala globale del modello
_YAW = 0.30                   # rotazione a 3/4 (rad)
_CYAW, _SYAW = math.cos(_YAW), math.sin(_YAW)
_PIV = 100.0


def _rot(p):
    dx = p[0] - _PIV
    return (_PIV + dx * _CYAW - p[2] * _SYAW, p[1], dx * _SYAW + p[2] * _CYAW)


def _rotv(v):
    return (v[0] * _CYAW - v[2] * _SYAW, v[1], v[0] * _SYAW + v[2] * _CYAW)


def _v3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    l = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / l, a[1] / l, a[2] / l)


_LIGHT = _norm((0.50, -0.62, 0.60))                      # alto-destra, davanti
_HALF = _norm((_LIGHT[0], _LIGHT[1], _LIGHT[2] + 1.0))

_RAMP = [(0.0, (6, 30, 95)), (0.3, (14, 72, 190)), (0.55, (40, 125, 235)),
         (0.8, (120, 195, 255)), (1.0, (230, 247, 255))]


def _ramp(v):
    v = max(0.0, min(1.0, v))
    for (a, ca), (b, cb) in zip(_RAMP, _RAMP[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return "#%02X%02X%02X" % tuple(int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
    return "#E6F7FF"


def _cr(p0, p1, p2, p3, t):
    """Catmull-Rom su tuple (x, y, z, r_piano, r_profondità)."""
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
                 for a, b, c, d in zip(p0, p1, p2, p3))


def _sample(nodes, per):
    pts, n = [], len(nodes)
    for i in range(n - 1):
        p0, p1, p2, p3 = nodes[max(i - 1, 0)], nodes[i], nodes[i + 1], nodes[min(i + 2, n - 1)]
        for k in range(per):
            pts.append(_cr(p0, p1, p2, p3, k / per))
    pts.append(nodes[-1])
    return pts


def _rings(nodes, sides, per):
    pts = _sample(nodes, per)
    rings = []
    for i, p in enumerate(pts):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        d = _norm((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
        ux, uy = d[1], -d[0]
        m = math.hypot(ux, uy)
        ux, uy = (1.0, 0.0) if m < 1e-6 else (ux / m, uy / m)
        rin, rz = max(0.6, p[3]), max(0.6, p[4])
        ring = []
        for s in range(sides):
            ang = 2 * math.pi * s / sides
            ca, sa = math.cos(ang), math.sin(ang)
            ring.append(_rot((p[0] + rin * ca * ux, p[1] + rin * ca * uy, p[2] + rz * sa)))
        rings.append(dict(c=_rot(p[:3]), p=ring, d=_rotv(d), r=(rin + rz) / 2))
    return rings


# --- modificatori di materiale (t = posizione lungo il tubo, n = normale in vista)
def _m_none(t, n):
    return 0.0


def _m_torso(t, n):
    if t > 0.80:
        return -0.16                                  # pantaloncini (fascia in vita)
    return 0.06 if (t < 0.5 and n[0] > 0.25) else 0.0  # petto illuminato


def _m_leg(t, n):
    if t < 0.36:
        return -0.15                                  # pantaloncini
    if t < 0.41:
        return 0.08                                   # bordo dei pantaloncini
    return 0.03 * math.sin(t * 18)                    # accenno di muscolatura


def _m_head(t, n):
    v = 0.0
    if n[1] < -0.2 and n[0] < 0.55:
        v -= 0.40                                     # capelli
    if 0.36 < t < 0.52 and n[0] > 0.3:
        v -= 0.28                                     # visiera / occhiali
    return v


def _m_foot(t, n):
    return -0.30 if n[1] > 0.45 else 0.0              # suola


def _m_arm(t, n):
    return 0.05 if t > 0.86 else 0.0


# nodi: (x, y, z, raggio_nel_piano, raggio_in_profondità)
_TORSO = [(119, 52, 0, 8.5, 10), (114, 66, 0, 14, 18), (105, 84, 0, 14.5, 16.5),
          (96, 102, 0, 12, 13), (89, 120, 0, 13.5, 14.5), (86, 136, 0, 10.5, 12.5)]
_NECK = [(125, 38, 0, 5.6, 5.8), (121, 48, 0, 5.8, 6.0), (117, 60, 0, 7.0, 7.5)]
_HEAD = [(127, 15.2, 0, 6.0, 5.5), (127, 18, 0, 9.5, 8.5), (128, 25, 0, 14.5, 12),
         (129, 33, 0, 14, 11.5), (129, 40, 0, 10.5, 9), (128, 46, 0, 4.5, 5)]
_ARM_N = [(115, 62, 17, 8.6, 8.6), (121, 78, 17, 6.9, 7.1), (129, 96, 17, 5.5, 5.6),
          (143, 88, 17, 5.6, 5.8), (155, 80, 17, 3.8, 4.0), (160, 77, 17, 5.0, 5.0)]
_ARM_F = [(111, 62, -17, 8.2, 8.2), (97, 71, -17, 6.5, 6.7), (83, 80, -17, 5.3, 5.3),
          (72, 92, -17, 5.1, 5.3), (63, 103, -17, 3.7, 3.9), (59, 107, -17, 4.8, 4.8)]
_LEG_N = [(88, 126, 10, 14.5, 13), (101, 131, 10, 14.5, 12.5), (116, 135, 10, 12, 11.5),
          (130, 138, 10, 8, 8.6), (125, 150, 10, 8.8, 9), (118, 166, 10, 6.4, 6.6), (112, 181, 10, 4.2, 4.6)]
_LEG_F = [(86, 128, -10, 14, 12.5), (76, 141, -10, 13, 12), (65, 154, -10, 10.5, 10.4),
          (56, 166, -10, 7.4, 7.8), (46, 175, -10, 8, 8.4), (33, 187, -10, 5.6, 5.8), (21, 196, -10, 4.0, 4.4)]
_FOOT_N = [(112, 183, 10, 5.2, 5.0), (121, 189, 10, 4.6, 5.2), (134, 191, 10, 3.4, 4.4)]
_FOOT_F = [(21, 197, -10, 4.6, 4.6), (11, 201, -10, 4.2, 4.6), (2, 204, -10, 3.0, 4.2)]

# nome, nodi, lati, suddivisioni, luminosità, modificatore, lato lontano, tappi
_CHAINS = [
    ("torso", _TORSO, 12, 3, 0.62, _m_torso, False, ""),
    ("neck", _NECK, 8, 2, 0.48, _m_none, False, ""),
    ("head", _HEAD, 12, 2, 0.56, _m_head, False, "both"),
    ("armN", _ARM_N, 8, 3, 0.58, _m_arm, False, "end"),
    ("armF", _ARM_F, 8, 3, 0.58, _m_arm, True, "end"),
    ("legN", _LEG_N, 10, 3, 0.54, _m_leg, False, ""),
    ("legF", _LEG_F, 10, 3, 0.54, _m_leg, True, ""),
    ("ftN", _FOOT_N, 8, 2, 0.92, _m_foot, False, "end"),
    ("ftF", _FOOT_F, 8, 2, 0.92, _m_foot, True, "end"),
]

# giunti (per sensori, scheletro, linee dati) — proiettati dal modello 3D
_J3 = dict(head=(128, 30, 0), neck=(121, 50, 0), sh=(114, 64, 17), shF=(112, 64, -17), chest=(107, 80, 0),
           elN=(129, 96, 17), haN=(156, 79, 17), elF=(83, 80, -17), haF=(62, 104, -17), hip=(86, 129, 0),
           knN=(130, 138, 10), anN=(112, 182, 10), knF=(56, 166, -10), anF=(20, 196, -10))
_JT = {k: (_rot(v)[0] * _K, _rot(v)[1] * _K) for k, v in _J3.items()}


def _at(x, y, s, name):
    """Posizione assoluta di un giunto per un runner disegnato in (x, y) con scala s."""
    return x + _JT[name][0] * s, y + _JT[name][1] * s


def _place(cx, s, top):
    """(x, y) con cui disegnare il runner: centrato su cx, cima della testa a 'top'."""
    return cx - 100 * s, top - 17 * s


def _emit(ch, out, rnd):
    name, nodes, sides, per, base, mod, far, caps = ch
    rings = _rings(nodes, sides, per)
    nr = len(rings)

    def add(a, b, c, cen, t):
        n = _cross(_v3(b, a), _v3(c, a))
        m = ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3)
        if _dot(n, _v3(m, cen)) < 0:
            n = (-n[0], -n[1], -n[2])
        n = _norm(n)
        if n[2] < -0.10:                              # facce di spalle: scartate
            return
        diff = max(0.0, _dot(n, _LIGHT))
        spec = 0.24 * max(0.0, _dot(n, _HALF)) ** 14
        rim = 0.20 * (1 - max(0.0, n[2])) ** 2
        v = base * (0.42 + 0.95 * diff) + spec + rim + mod(t, n) + rnd.uniform(-0.035, 0.035)
        if far:
            v *= 0.66
        out.append(((a[2] + b[2] + c[2]) / 3, ((a[0], a[1]), (b[0], b[1]), (c[0], c[1])), _ramp(v)))

    for i in range(nr - 1):
        r0, r1 = rings[i], rings[i + 1]
        cen = tuple((p + q) / 2 for p, q in zip(r0["c"], r1["c"]))
        t = (i + 0.5) / (nr - 1)
        for s in range(sides):
            s2 = (s + 1) % sides
            a, b, c, d = r0["p"][s], r0["p"][s2], r1["p"][s2], r1["p"][s]
            if (i + s) % 2:
                add(a, b, c, cen, t)
                add(a, c, d, cen, t)
            else:
                add(a, b, d, cen, t)
                add(b, c, d, cen, t)

    def cap(ring, sign, t):
        c, d, r = ring["c"], ring["d"], ring["r"]
        tip = (c[0] + sign * d[0] * r * 0.4, c[1] + sign * d[1] * r * 0.4, c[2] + sign * d[2] * r * 0.4)
        for s in range(sides):
            add(tip, ring["p"][s], ring["p"][(s + 1) % sides], c, t)

    if caps in ("end", "both"):
        cap(rings[-1], 1, 0.99)
    if caps == "both":
        cap(rings[0], -1, 0.01)


def _build_mesh():
    rnd = random.Random(7)
    tris = []
    for ch in _CHAINS:
        _emit(ch, tris, rnd)
    tris.sort(key=lambda t: t[0])                     # dal fondo al davanti
    polys, verts = [], []
    for _, pts, col in tris:
        polys.append('<polygon points="' + " ".join(f"{x * _K:.1f},{y * _K:.1f}" for x, y in pts)
                     + f'" fill="{col}"/>')
        verts.extend(pts)
    spark = "".join(f'<circle cx="{x * _K:.1f}" cy="{y * _K:.1f}" r="1.3"/>' for x, y in rnd.sample(verts, 48))
    return "".join(polys), spark


_MESH, _SPARK = _build_mesh()

_NODE_NAMES = ("sh", "elN", "haN", "hip", "knN", "anN", "elF", "knF", "anF")
_RUNNER_NODES = "".join(
    f'<circle cx="{_JT[n][0]:.0f}" cy="{_JT[n][1]:.0f}" r="6.5" fill="none" stroke="#fff" stroke-opacity="0.45"/>'
    f'<circle cx="{_JT[n][0]:.0f}" cy="{_JT[n][1]:.0f}" r="2.8" fill="#fff"/>' for n in _NODE_NAMES)

RUNNER_GLOW_DEFS = """
    <radialGradient id="runnerGlow" cx="50%" cy="50%" r="60%">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.28"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="sunG" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#CFEBFF" stop-opacity="0.85"/>
        <stop offset="45%" stop-color="#2F8FE0" stop-opacity="0.3"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="meshGlow" x="-40%" y="-40%" width="180%" height="180%">
        <feGaussianBlur stdDeviation="8"/>
    </filter>
    <filter id="rim" x="-10%" y="-10%" width="120%" height="120%">
        <feMorphology in="SourceAlpha" operator="dilate" radius="1.3" result="d"/>
        <feComposite in="d" in2="SourceAlpha" operator="out" result="edge"/>
        <feFlood flood-color="#8FD3FF" flood-opacity="0.9"/>
        <feComposite in2="edge" operator="in"/>
    </filter>
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
""" + (f'<g id="meshBody" stroke="#BFE6FF" stroke-width="0.45" stroke-opacity="0.40" stroke-linejoin="round">'
       f'{_MESH}<g stroke="none" fill="#fff" opacity="0.9">{_SPARK}</g></g>')

# Maschera ellittica: lo scenario sfuma a zero prima di toccare i bordi dell'immagine
# (in alto, in basso, a destra e a sinistra), quindi non compare nessun rettangolo.
_FADE_DEFS = (
    '<radialGradient id="fadeG" gradientUnits="userSpaceOnUse" cx="600" cy="250" r="600" '
    'gradientTransform="translate(0 250) scale(1 0.40) translate(0 -250)">'
    '<stop offset="0" stop-color="#fff"/><stop offset="0.55" stop-color="#fff"/>'
    '<stop offset="0.78" stop-color="#fff" stop-opacity="0.6"/>'
    '<stop offset="0.92" stop-color="#fff" stop-opacity="0.15"/>'
    '<stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
    '<mask id="fadeMask" maskUnits="userSpaceOnUse" x="0" y="0" width="1200" height="500">'
    '<rect width="1200" height="500" fill="url(#fadeG)"/></mask>')


def _svg(scene, fg="", extra_defs=""):
    """scene = scenario (sfumato ai bordi); fg = runner, pannelli, HUD (non mascherati)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="20 20 1160 460">'
            f'<defs>{RUNNER_GLOW_DEFS}{extra_defs}{_FADE_DEFS}</defs>'
            f'<g mask="url(#fadeMask)">{scene}</g>{fg}</svg>')


def _runner(x, y, s=1.0, nodes=True):
    return (f'<g transform="translate({x:.1f},{y:.1f}) scale({s})">'
            f'<use href="#meshBody" filter="url(#meshGlow)" opacity="0.6"/>'
            f'<use href="#meshBody" filter="url(#rim)"/>'
            f'<use href="#meshBody"/>{_RUNNER_NODES if nodes else ""}</g>')


def _runner_at(x, y, s=1.0, nodes=True):
    """Runner con riflesso blu sotto i piedi."""
    gx, gy = x + 100 * s, y + 246 * s
    ground = (f'<ellipse cx="{gx:.0f}" cy="{gy:.0f}" rx="{115 * s:.0f}" ry="{16 * s:.0f}" fill="url(#runnerGlow)"/>'
              f'<ellipse cx="{gx:.0f}" cy="{gy:.0f}" rx="{70 * s:.0f}" ry="{4 * s:.0f}" fill="#2F8FE0" opacity="0.35"/>')
    return ground + _runner(x, y, s, nodes)


# ---------- helper grafici ----------

def _t(x, y, txt, size=14, fill="#E8F6FF", anchor="start", mono=False, weight=400, op=0.9):
    fam = "'JetBrains Mono', monospace" if mono else "Inter, sans-serif"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" opacity="{op}">{txt}</text>')


def _stars(n, seed, x0, x1, y0, y1):
    rnd = random.Random(seed)
    return "".join(
        f'<circle cx="{rnd.uniform(x0, x1):.0f}" cy="{rnd.uniform(y0, y1):.0f}" '
        f'r="{rnd.choice([0.8, 1.1, 1.5])}" fill="#CFEBFF" opacity="{rnd.uniform(0.3, 0.9):.2f}"/>'
        for _ in range(n))


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


def _pines(seed, x0, x1, base, n, fill="#010306"):
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
    out = [f'<line x1="{vx}" y1="{vy}" x2="{x}" y2="{ybot}" stroke="#2F8FE0" stroke-opacity="{opacity}"/>'
           for x in range(xa, xb, step)]
    out += [f'<line x1="0" y1="{vy + r}" x2="1200" y2="{vy + r}" stroke="#2F8FE0" stroke-opacity="{opacity}"/>'
            for r in rows]
    return "".join(out)


def _speedlines(x, y, n=5, length=160, gap=22):
    return "".join(
        f'<rect x="{x - (length - i * 22)}" y="{y + i * gap}" width="{length - i * 22}" height="3.5" '
        f'rx="1.75" fill="url(#speedG)"/>' for i in range(n))


def _panel(x, y, w, h, title=""):
    return f"""<g transform="translate({x},{y})">
        <rect width="{w}" height="{h}" rx="14" fill="#04070D" fill-opacity="0.88" stroke="#2F8FE0" stroke-opacity="0.55"/>
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


def _ecg(x0, x1, y, beats=2):
    beat = [(0, 0), (.10, 0), (.14, -14), (.18, 16), (.22, -40), (.26, 24), (.30, 0),
            (.42, 0), (.47, -10), (.53, 6), (.58, 0), (1, 0)]
    w = (x1 - x0) / beats
    return " ".join(f"{x0 + w * (b + fx):.0f},{y + dy}" for b in range(beats) for fx, dy in beat)


def _link(p, tx, ty):
    """Curva a S dal sensore sul corpo al pannello."""
    mx = (p[0] + tx) / 2
    return f'<path d="M{p[0]:.0f},{p[1]:.0f} C{mx:.0f},{p[1]:.0f} {mx:.0f},{ty} {tx},{ty}"/>'


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


def _backdrop(cx, cy, rx, ry):
    return f"""
    <ellipse cx="{cx}" cy="{cy}" rx="{rx + 60}" ry="{ry + 60}" fill="url(#runnerGlow)"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#2F8FE0"
             stroke-opacity="0.4" stroke-width="1.5" stroke-dasharray="3,9"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx * 0.78}" ry="{ry * 0.78}" fill="none" stroke="#7EC8FF"
             stroke-opacity="0.15"/>
    """


# =========================================================
# HOME — notte sulla città: skyline, luna a fasce, strada in prospettiva
# con traccia GPS, pin di destinazione pulsante e pannelli dati.
# Runner grande al centro-sinistra, pannelli negli spazi liberi.
# =========================================================
_SUN_BANDS = "".join(f'<rect x="820" y="{y}" width="160" height="{h}" fill="#01040A"/>'
                     for y, h in ((188, 3), (200, 5), (214, 7), (230, 9)))
_HX, _HY = _place(470, 1.8, 36)

_HOME_SCENE = f"""
{_stars(90, 7, 0, 1200, 0, 260)}
<circle cx="900" cy="170" r="170" fill="url(#sunG)" opacity="0.55"/>
<circle cx="900" cy="170" r="70" fill="#DDF1FF"/>
<g clip-path="url(#sunClip)">{_SUN_BANDS}</g>
{_skyline(0, 1200, 346, 40, 120, 3, "#061226", lit=False)}
{_skyline(0, 1200, 346, 60, 170, 11, "#030912")}
{_grid_floor(700, 346)}
<polygon points="694,346 706,346 830,500 130,500" fill="#0A1A33" opacity="0.95"/>
<g stroke="#2F8FE0" stroke-width="3" opacity="0.85" filter="url(#softGlow)">
    <line x1="694" y1="346" x2="130" y2="500"/><line x1="706" y1="346" x2="830" y2="500"/>
</g>
<line x1="700" y1="350" x2="480" y2="500" stroke="#CFEBFF" stroke-width="4" stroke-dasharray="16,16" opacity="0.8">
    <animate attributeName="stroke-dashoffset" values="0;-64" dur="1.4s" repeatCount="indefinite"/>
</line>
<path d="M330,500 C470,470 620,440 690,352" fill="none" stroke="url(#homeTrail)" stroke-width="5"
      stroke-dasharray="2,12" stroke-linecap="round"/>
"""
_HOME_FG = f"""
<g transform="translate(700,244) scale(0.8)">
    <path d="M0,0 C-26,0 -46,20 -46,46 C-46,80 0,120 0,120 C0,120 46,80 46,46 C46,20 26,0 0,0 Z"
          fill="#1B5FA8" filter="url(#softGlow)"/>
    <circle cx="0" cy="44" r="16" fill="#000"/>
    <circle cx="0" cy="44" r="6" fill="#00E5FF"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/></circle>
</g>
<ellipse cx="700" cy="342" rx="10" ry="4" fill="none" stroke="#00E5FF" stroke-width="2">
    <animate attributeName="rx" values="10;60" dur="2s" repeatCount="indefinite"/>
    <animate attributeName="ry" values="4;14" dur="2s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.9;0" dur="2s" repeatCount="indefinite"/>
</ellipse>
{_speedlines(330, 190)}
{_runner_at(_HX, _HY, 1.8)}
{_panel(40, 40, 250, 92, "DISTANZA")}
{_t(60, 106, "12.4 km", 34, "#7EC8FF", mono=True, weight=700, op=1)}
{_panel(40, 330, 250, 110, "FREQ. CARDIACA")}
{_t(60, 400, "148", 40, "#7EC8FF", mono=True, weight=700, op=1)}{_t(138, 400, "bpm", 16, "#E8F6FF", op=0.8)}
<polyline points="{_wave(160, 270, 384, 7, 3)}" fill="none" stroke="#00E5FF" stroke-width="2.4" stroke-linejoin="round"/>
{_panel(880, 290, 280, 150, "PASSO MEDIO")}
{_t(900, 366, "4:32", 46, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(1010, 366, "/km", 18, "#E8F6FF", op=0.8)}
<polyline points="{_wave(900, 1140, 412, 10, 2.2, phase=0.6)}" fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linejoin="round"/>
"""
SVG_HOME = _svg(_HOME_SCENE, _HOME_FG, extra_defs="""
    <clipPath id="sunClip"><circle cx="900" cy="170" r="70"/></clipPath>
    <linearGradient id="homeTrail" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0.95"/>
    </linearGradient>""")


# =========================================================
# ANALISI STATO DI FORMA — scansione biometrica: anelli radar con
# sweep animato, linea di scansione, sensori sul corpo collegati ai
# pannelli HRV (ECG), SMA (movimento) e sonno (ipnogramma).
# =========================================================
_AX, _AY = _place(400, 1.8, 34)
_SMA_BARS = "".join(
    f'<rect x="{860 + i * 23}" y="{302 - h}" width="15" height="{h}" rx="3" fill="url(#statBarGrad)"/>'
    for i, h in enumerate([18, 30, 24, 40, 34, 50, 42, 56, 38, 46, 30, 40]))
_HYP = [(0, 404), (15, 404), (15, 420), (45, 420), (45, 404), (65, 404), (65, 388), (90, 388), (90, 404),
        (105, 404), (105, 420), (140, 420), (140, 388), (155, 388), (155, 404), (185, 404), (185, 388),
        (205, 388), (205, 404), (240, 404)]
_HYP_PTS = " ".join(f"{860 + x * 1.15:.0f},{y}" for x, y in _HYP)
_HR, _WR, _AK = (_at(_AX, _AY, 1.8, n) for n in ("chest", "haN", "anN"))

_ANALISI_SCENE = f"""
{_stars(60, 21, 0, 1200, 0, 500)}
<ellipse cx="400" cy="250" rx="360" ry="290" fill="url(#runnerGlow)"/>
"""
_ANALISI_FG = f"""
<g fill="none" stroke="#2F8FE0" stroke-opacity="0.4">
    <circle cx="400" cy="250" r="120" stroke-dasharray="3,7"/>
    <circle cx="400" cy="250" r="175"/>
    <circle cx="400" cy="250" r="225" stroke-dasharray="10,8"/>
</g>
<g>
    <path d="M400,250 L400,25 A225,225 0 0 1 559,91 Z" fill="#2F8FE0" opacity="0.16"/>
    <animateTransform attributeName="transform" type="rotate" from="0 400 250" to="360 400 250" dur="8s" repeatCount="indefinite"/>
</g>
{_runner_at(_AX, _AY, 1.8, nodes=False)}
<rect x="240" y="70" width="320" height="3" rx="1.5" fill="#00E5FF" filter="url(#softGlow)" opacity="0.9">
    <animate attributeName="y" values="70;440;70" dur="4s" repeatCount="indefinite"/>
</rect>
<g fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7">
    {_link(_HR, 690, 105)}{_link(_WR, 690, 255)}{_link(_AK, 690, 405)}
</g>
<g fill="#00E5FF">
    <circle cx="{_HR[0]:.0f}" cy="{_HR[1]:.0f}" r="7" filter="url(#softGlow)"><animate attributeName="r" values="6;11;6" dur="1.1s" repeatCount="indefinite"/></circle>
    <circle cx="{_WR[0]:.0f}" cy="{_WR[1]:.0f}" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/></circle>
    <circle cx="{_AK[0]:.0f}" cy="{_AK[1]:.0f}" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.6s" begin="0.5s" repeatCount="indefinite"/></circle>
</g>
{_panel(690, 40, 470, 130, "HRV · VARIABILITÀ CARDIACA")}
{_t(710, 125, "62 ms", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(710, 152, "media 7 giorni", 12, op=0.6)}
<polyline points="{_ecg(860, 1140, 108)}" fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linejoin="round"/>
<circle r="4" fill="#fff" filter="url(#softGlow)"><animateMotion dur="2.2s" repeatCount="indefinite" path="M860,108 L1140,108"/></circle>
{_panel(690, 190, 470, 130, "SMA · INTENSITÀ DI MOVIMENTO")}
{_t(710, 275, "0.41", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_SMA_BARS}
{_panel(690, 340, 470, 130, "SONNO · IPNOGRAMMA")}
{_t(710, 425, "7h20", 40, "#7EC8FF", mono=True, weight=700, op=1)}
{_t(710, 452, "profondo 1h35 · REM 1h50", 12, op=0.6)}
<polyline points="{_HYP_PTS}" fill="none" stroke="#7EC8FF" stroke-width="3" stroke-linejoin="round"/>
"""
SVG_ANALISI = _svg(_ANALISI_SCENE, _ANALISI_FG)


# =========================================================
# STATISTICHE — runner e mini-dashboard: istogramma settimanale con
# trend, ciambella delle zone di frequenza, calendario delle sessioni.
# =========================================================
_SX, _SY = _place(205, 1.8, 34)
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

_STATS_SCENE = f"""
{_stars(40, 4, 0, 1200, 0, 500)}
{_backdrop(205, 250, 170, 205)}
"""
_STATS_FG = f"""
{_runner_at(_SX, _SY, 1.8)}
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
"""
SVG_STATS = _svg(_STATS_SCENE, _STATS_FG)


# =========================================================
# KPI DASHBOARD — indicatore principale a 270° con tacche e lancetta
# animata, tre anelli secondari e trend a 30 giorni.
# =========================================================
_KX, _KY = _place(215, 1.8, 34)
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

_KPI_SCENE = f"""
{_stars(45, 9, 0, 1200, 0, 500)}
{_backdrop(215, 255, 160, 205)}
<circle cx="640" cy="255" r="210" fill="url(#runnerGlow)"/>
"""
_KPI_FG = f"""
{_runner_at(_KX, _KY, 1.8)}
{_ticks(640, 255, 172, 184, 135, 270, 54, major=9)}
<circle cx="640" cy="255" r="150" fill="none" stroke="#12386B" stroke-width="24" stroke-linecap="round"
        stroke-dasharray="707 943" transform="rotate(135 640 255)"/>
<circle cx="640" cy="255" r="150" fill="none" stroke="url(#panelBar)" stroke-width="24" stroke-linecap="round"
        stroke-dasharray="583 943" transform="rotate(135 640 255)" filter="url(#softGlow)"/>
<g transform="rotate(357.5 640 255)">
    <polygon points="640,250 640,260 770,255" fill="#E8F6FF"/>
    <animateTransform attributeName="transform" type="rotate" from="135 640 255" to="357.5 640 255" dur="1.8s" fill="freeze"/>
</g>
<circle cx="640" cy="255" r="11" fill="#000" stroke="#00E5FF" stroke-width="3"/>
{_t(640, 322, "82.4%", 50, "#7EC8FF", "middle", mono=True, weight=700, op=1)}
{_t(640, 348, "INDICE DI FORMA", 14, "#E8F6FF", "middle")}
<rect x="590" y="366" width="100" height="26" rx="13" fill="#00E5FF" opacity="0.18" stroke="#00E5FF"/>
{_t(640, 384, "OTTIMO", 13, "#00E5FF", "middle", weight=700, op=1)}
{_RINGS}
{_panel(850, 230, 310, 230, "TREND 30 GIORNI")}
<polygon points="870,455 {_KPI_LINE} 1142,455" fill="url(#areaG)"/>
<polyline points="{_KPI_LINE}" fill="none" stroke="#00E5FF" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{_KPI_TREND[-1][0]:.0f}" cy="{_KPI_TREND[-1][1]:.0f}" r="6" fill="#fff" filter="url(#softGlow)"/>
"""
SVG_KPI = _svg(_KPI_SCENE, _KPI_FG)


# =========================================================
# ML / PREVISIONE — dal runner i dati fluiscono in una rete neurale
# (con impulsi animati) e diventano una previsione con cono di
# incertezza, soglia di rischio e importanza delle variabili.
# =========================================================
_MX, _MY = _place(175, 1.55, 80)
_MC = _at(_MX, _MY, 1.55, "chest")
_MPATH = f"M{_MC[0]:.0f},{_MC[1]:.0f} C{_MC[0] + 90:.0f},{_MC[1]:.0f} 330,250 400,250"
_XS, _CNT = [400, 500, 600, 700], [5, 6, 6, 2]
_NODES = [[(x, 250 + (i - (n - 1) / 2) * 52) for i in range(n)] for x, n in zip(_XS, _CNT)]
_LINKS = "".join(f'<line x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}"/>'
                 for a, b in zip(_NODES, _NODES[1:]) for x1, y1 in a for x2, y2 in b)
_CIRC = "".join(f'<circle cx="{x}" cy="{y:.0f}" r="12" fill="#000" '
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

_ML_SCENE = f"""
{_stars(45, 13, 0, 1200, 0, 500)}
{_backdrop(175, 262, 135, 190)}
"""
_ML_FG = f"""
{_runner_at(_MX, _MY, 1.55)}
<path d="{_MPATH}" fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7"/>
<circle r="4" fill="#fff"><animateMotion dur="1.6s" repeatCount="indefinite" path="{_MPATH}"/></circle>
<circle r="4" fill="#00E5FF"><animateMotion dur="1.6s" begin="0.8s" repeatCount="indefinite" path="{_MPATH}"/></circle>
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
"""
SVG_ML = _svg(_ML_SCENE, _ML_FG, extra_defs="""
    <linearGradient id="mlConeG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.35"/><stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.05"/>
    </linearGradient>""")


# =========================================================
# PIANO ALLENAMENTO — notte di montagna: creste a strati, pini,
# sentiero a tornanti con le fasi (Base, Build, Specifico, Gara),
# bandiera sulla vetta e settimana tipo con i tipi di seduta.
# Vetta spostata a destra: il runner grande occupa il vuoto a sinistra.
# =========================================================
_PX, _PY = _place(235, 1.3, 178)
_DAYS = [("L", "#12386B"), ("M", "#00E5FF"), ("M", "#2F8FE0"), ("G", "#00E5FF"),
         ("V", "#12386B"), ("S", "#7EC8FF"), ("D", "#2F8FE0")]
_WEEK = "".join(
    f'<rect x="{56 + i * 56}" y="72" width="46" height="46" rx="10" fill="{c}" opacity="0.9"/>'
    + _t(79 + i * 56, 101, d, 16, "#040A18" if c in ("#00E5FF", "#7EC8FF") else "#E8F6FF", "middle", weight=700, op=1)
    for i, (d, c) in enumerate(_DAYS))
_LEGEND = "".join(
    f'<circle cx="{x}" cy="150" r="5" fill="{c}"/>' + _t(x + 10, 154, lab, 12, op=0.8)
    for x, c, lab in ((60, "#12386B", "Riposo"), (140, "#2F8FE0", "Facile"),
                      (222, "#00E5FF", "Intervalli"), (320, "#7EC8FF", "Lungo")))

_PLAN_SCENE = f"""
{_stars(110, 17, 0, 1200, 0, 230)}
<polygon points="0,400 150,250 260,330 400,200 560,340 700,230 860,350 1010,240 1200,360 1200,500 0,500" fill="#050D1C"/>
<polygon points="0,450 200,330 330,400 520,270 700,400 880,300 1060,410 1200,340 1200,500 0,500" fill="#08152C"/>
<polygon points="500,500 820,140 1140,500" fill="#0E2A54"/>
<polygon points="820,140 1140,500 920,500" fill="#061631" opacity="0.7"/>
<polygon points="820,140 776,206 800,196 820,222 844,194 864,208" fill="#E8F6FF" opacity="0.92"/>
<polygon points="0,500 0,432 180,410 360,450 520,430 700,470 900,440 1200,470 1200,500" fill="#02050B"/>
{_pines(2, 0, 1200, 470, 40)}
<polyline points="330,474 460,455 370,425 600,396 510,360 710,332 650,292 780,262 740,226 820,196 820,148"
          fill="none" stroke="#7EC8FF" stroke-width="4" stroke-dasharray="3,10" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
"""
_PLAN_FG = f"""
<circle cx="1050" cy="95" r="90" fill="url(#sunG)" opacity="0.45"/>
<circle cx="1050" cy="95" r="32" fill="#DDF1FF"/>
<circle cx="1062" cy="87" r="28" fill="#000" opacity="0.18"/>
<g fill="#00E5FF" filter="url(#softGlow)">
    <circle cx="460" cy="455" r="7"/><circle cx="710" cy="332" r="7"/><circle cx="780" cy="262" r="7"/>
</g>
<line x1="820" y1="148" x2="820" y2="86" stroke="#E8F6FF" stroke-width="3"/>
<path d="M820,86 L820,114 L868,100 Z" fill="#00E5FF" filter="url(#softGlow)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
</path>
{_t(480, 478, "BASE", 14, "#CFEBFF", weight=700)}
{_t(730, 352, "BUILD", 14, "#CFEBFF", weight=700)}
{_t(800, 278, "SPECIFICO", 14, "#CFEBFF", weight=700)}
{_t(840, 84, "GARA", 15, "#00E5FF", weight=700, op=1)}
{_runner_at(_PX, _PY, 1.3)}
{_panel(40, 30, 420, 140, "SETTIMANA TIPO")}
{_WEEK}
{_LEGEND}
"""
SVG_PLAN = _svg(_PLAN_SCENE, _PLAN_FG)


# =========================================================
# COMPUTER VISION — motion tracking: scia di fotogrammi, scheletro con
# giunti, riquadro di rilevamento, angoli articolari e pannelli con
# ginocchio, cadenza e carico sulla tibia.
# =========================================================
_CS = 1.65
_CVX, _CVY = _place(470, _CS, 66)
_CV = {k: _at(_CVX, _CVY, _CS, k) for k in _JT}
_BONES = [("head", "neck"), ("neck", "sh"), ("neck", "shF"), ("sh", "elN"), ("elN", "haN"),
          ("shF", "elF"), ("elF", "haF"), ("neck", "hip"), ("hip", "knN"), ("knN", "anN"),
          ("hip", "knF"), ("knF", "anF")]
_CV_BONES = "".join(f'<line x1="{_CV[a][0]:.0f}" y1="{_CV[a][1]:.0f}" x2="{_CV[b][0]:.0f}" y2="{_CV[b][1]:.0f}"/>'
                    for a, b in _BONES)
_CV_JOINTS = "".join(f'<circle cx="{_CV[k][0]:.0f}" cy="{_CV[k][1]:.0f}" r="6"/>'
                     for k in ("head", "neck", "sh", "shF", "elN", "haN", "elF", "haF", "hip", "knN", "anN", "knF", "anF"))
_CV_ARCS = "".join(f'<path d="{_arc(_CV[c], _CV[a], _CV[b], r)}"/>' for c, a, b, r in
                   (("knN", "hip", "anN", 34), ("elN", "sh", "haN", 26), ("hip", "neck", "knN", 30)))
_CAD = "".join(
    f'<rect x="{760 + i * 15}" y="{300 - h:.0f}" width="8" height="{h:.0f}" rx="3" fill="url(#statBarGrad)"/>'
    for i, h in ((i, 30 + 22 * (1 + math.sin(i * 0.6))) for i in range(24)))

_CV_SCENE = f"""
{_grid_floor(600, 330, rows=(10, 26, 48, 80, 120, 160), opacity=0.22)}
<ellipse cx="470" cy="270" rx="350" ry="270" fill="url(#runnerGlow)"/>
"""
_CV_FG = f"""
<g opacity="0.10">{_runner(_CVX - 150, _CVY, _CS, nodes=False)}</g>
<g opacity="0.22">{_runner(_CVX - 75, _CVY, _CS, nodes=False)}</g>
<g opacity="0.6">{_runner_at(_CVX, _CVY, _CS, nodes=False)}</g>
<g stroke="#00E5FF" stroke-width="3.5" stroke-linecap="round" filter="url(#softGlow)">{_CV_BONES}</g>
<g fill="#000" stroke="#00E5FF" stroke-width="2.5">{_CV_JOINTS}</g>
<g fill="none" stroke="#FFB84D" stroke-width="2">{_CV_ARCS}</g>
{_corners(290, 52, 360, 424, 30, "#00E5FF")}
<rect x="290" y="26" width="128" height="24" rx="5" fill="#00E5FF"/>
{_t(302, 43, "ATLETA · 98%", 13, "#040A18", mono=True, weight=700, op=1)}
<g fill="none" stroke="#7EC8FF" stroke-width="1.2" stroke-dasharray="3,6" opacity="0.7">
    {_link(_CV["knN"], 740, 105)}{_link(_CV["hip"], 740, 250)}{_link(_CV["anN"], 740, 400)}
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
<circle cx="52" cy="46" r="6" fill="#FF5A5F"><animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/></circle>
{_t(68, 51, "REC · 00:12:48 · F3841", 14, "#E8F6FF", mono=True)}
"""
SVG_CV = _svg(_CV_SCENE, _CV_FG)
