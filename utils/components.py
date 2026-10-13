import base64
import html
import re
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


# Registro: data-URI dell'SVG -> parametri della fotografia da mostrare sopra.
# Motivo: un SVG caricato come <img src="data:..."> NON può scaricare immagini
# esterne (il browser lo blocca per sicurezza), quindi le foto di Unsplash
# dentro l'SVG non comparivano mai. Le foto ora sono livelli HTML veri.
_PHOTO_BY_URI = {}


def get_svg_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    uri = f"data:image/svg+xml;base64,{b64}"
    m = re.search(r"<!--photo:([^>]*?)-->", svg_string)
    if m:
        _PHOTO_BY_URI[uri] = m.group(1).split("|")
    return uri


def _hero_media_html(image_url, image_tag):
    tag = f'<div class="tag">{image_tag or ""}</div>'
    p = _PHOTO_BY_URI.get(image_url)
    if not p or len(p) < 7:
        return f'<div class="hero-media"><img src="{image_url}" />{tag}</div>'

    url = html.escape(p[0], quote=True)
    cx, cy, rx, ry = (float(v) for v in p[1:5])
    tint, tint_op = p[5], p[6]
    # coordinate dell'SVG (viewBox 1200x500) convertite in percentuali
    left, top = (cx - rx) / 12, (cy - ry) / 5
    w, h = 2 * rx / 12, 2 * ry / 5
    mask = "radial-gradient(ellipse closest-side at center, #000 70%, rgba(0,0,0,0) 100%)"

    # Nessuna riga vuota: una riga vuota farebbe uscire Markdown dal blocco HTML.
    return (
        '<div class="hero-media">'
        '<div style="position:relative; width:100%; aspect-ratio:1200/500;">'
        f'<img src="{image_url}" style="position:absolute; inset:0; width:100%; height:100%;" />'
        f'<div style="position:absolute; left:{left:.2f}%; top:{top:.2f}%; width:{w:.2f}%; height:{h:.2f}%; '
        f'border-radius:50%; background-image:url(\'{url}\'), radial-gradient(circle, #12386B 0%, #0A1E3D 100%); '
        f'background-size:cover; background-position:center; -webkit-mask-image:{mask}; mask-image:{mask}; pointer-events:none;">'
        f'<div style="position:absolute; inset:0; border-radius:50%; background:{tint}; opacity:{tint_op};"></div>'
        '</div>'
        '</div>'
        f'{tag}'
        '</div>'
    )


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
            st.markdown(_hero_media_html(image_url, image_tag), unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# LIBRERIA CONDIVISA — fotografie reali (Unsplash) in una "medaglia"
# ellittica con bordo sfumato. La fotografia vera NON sta più nell'SVG:
# _photo_orb disegna solo alone e anello, e lascia nel file un
# commento <!--photo:...--> con i parametri; header_block li legge e
# mette la foto come livello HTML sopra l'SVG.
# =========================================================

RUNNER_GLOW_DEFS = """
    <radialGradient id="runnerGlow" cx="50%" cy="50%" r="60%">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0"/>
    </radialGradient>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
"""

# URL delle fotografie (Unsplash, licenza libera per uso commerciale).
IMG_HOME = "https://images.unsplash.com/photo-1502224562085-639556652f33?fm=jpg&q=70&w=1400&auto=format&fit=crop"
IMG_ANALISI = "https://images.unsplash.com/photo-1523394894855-2feb062d437d?fm=jpg&q=70&w=1400&auto=format&fit=crop"
IMG_STATS = "https://images.unsplash.com/photo-1686061592689-312bbfb5c055?fm=jpg&q=70&w=1400&auto=format&fit=crop"
IMG_KPI = "https://images.unsplash.com/photo-1523394894855-2feb062d437d?fm=jpg&q=70&w=1400&auto=format&fit=crop&crop=focalpoint&fp-x=0.35"
IMG_ML = "https://images.unsplash.com/photo-1586448317606-cb1ec00298fc?fm=jpg&q=70&w=1400&auto=format&fit=crop"
IMG_PLAN = "https://images.unsplash.com/photo-1621650784637-eb439621916c?fm=jpg&q=70&w=1400&auto=format&fit=crop"
IMG_CV = "https://images.unsplash.com/photo-1758506971649-7063f5518cf3?fm=jpg&q=70&w=1400&auto=format&fit=crop"


def _photo_orb(url, id_suffix, cx, cy, rx, ry, tint="#0A1E3D", tint_opacity=0.22):
    """Alone di luce + anello sottile della medaglia. La foto vera viene
    aggiunta da header_block come livello HTML (vedi commento qui sotto)."""
    return f"""
    <ellipse cx="{cx}" cy="{cy}" rx="{rx + 55}" ry="{ry + 55}" fill="url(#runnerGlow)"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#0A1E3D" opacity="0.5"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#2F8FE0" stroke-opacity="0.35" stroke-width="1.5"/>
    <!--photo:{url}|{cx}|{cy}|{rx}|{ry}|{tint}|{tint_opacity}-->
    """


# HOME — foto del corridore in scatto con la traccia GPS che si dipana
# dietro, un pin di destinazione e la rete di sensori che collega il
# runner al percorso.
SVG_HOME = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="homeTrail" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.8"/>
    </linearGradient>
</defs>
<path d="M40,340 C160,335 220,300 300,280 C360,265 380,300 420,280"
      fill="none" stroke="url(#homeTrail)" stroke-width="3" stroke-dasharray="1,10" stroke-linecap="round"/>
{_photo_orb(IMG_HOME, "Home", cx=430, cy=250, rx=280, ry=230)}
<g transform="translate(830,150)">
    <path d="M0,0 C-26,0 -46,20 -46,46 C-46,80 0,120 0,120 C0,120 46,80 46,46 C46,20 26,0 0,0 Z" fill="#1B5FA8" filter="url(#softGlow)"/>
    <circle cx="0" cy="44" r="16" fill="#0B1F3F"/>
</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.6" stroke-dasharray="2,6">
    <path d="M660,220 C720,190 770,180 800,175"/>
</g>
<circle cx="800" cy="175" r="5" fill="#00E5FF" filter="url(#softGlow)"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/></circle>
<g font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.85">
    <text x="880" y="290">Passo medio</text>
    <text x="880" y="322" font-family="'JetBrains Mono', monospace" font-size="30" font-weight="700" fill="#7EC8FF">4:32/km</text>
</g>
</svg>"""

# ANALISI STATO DI FORMA
SVG_ANALISI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
{_photo_orb(IMG_ANALISI, "Analisi", cx=460, cy=250, rx=300, ry=240)}
<g stroke="#7EC8FF" stroke-width="1" opacity="0.55">
    <path d="M700,150 C800,110 900,95 960,80"/>
    <path d="M720,250 C820,235 900,240 990,230"/>
    <path d="M700,330 C800,340 860,345 950,360"/>
</g>
<g fill="#00E5FF">
    <circle cx="700" cy="150" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/></circle>
    <circle cx="720" cy="250" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="0.5;1;0.5" dur="1.6s" begin="0.3s" repeatCount="indefinite"/></circle>
    <circle cx="700" cy="330" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.6s" begin="0.6s" repeatCount="indefinite"/></circle>
</g>
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">
    <text x="960" y="75">HRV — 62 ms</text>
    <text x="990" y="225">SMA — 0.41</text>
    <text x="950" y="355">Sonno — 7h20</text>
</g>
</svg>"""

# STATISTICHE
SVG_STATS = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="statBarGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
</defs>
{_photo_orb(IMG_STATS, "Stats", cx=420, cy=250, rx=280, ry=230)}
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" stroke-dasharray="2,6"><path d="M660,230 C740,215 800,210 840,220"/></g>
<g transform="translate(870,190)" fill="url(#statBarGrad)">
    <rect x="0"   y="90" width="26" height="60" rx="3"/>
    <rect x="36"  y="60" width="26" height="90" rx="3"/>
    <rect x="72"  y="20" width="26" height="130" rx="3">
        <animate attributeName="height" values="130;150;130" dur="2.2s" repeatCount="indefinite"/>
        <animate attributeName="y" values="20;0;20" dur="2.2s" repeatCount="indefinite"/>
    </rect>
    <rect x="108" y="45" width="26" height="105" rx="3"/>
    <rect x="144" y="70" width="26" height="80" rx="3"/>
</g>
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">
    <text x="870" y="365">52 sessioni raccolte</text>
</g>
</svg>"""

# KPI DASHBOARD
SVG_KPI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
{_photo_orb(IMG_KPI, "Kpi", cx=440, cy=250, rx=290, ry=230)}
<g transform="translate(830,190)">
    <rect x="-40" y="-10" width="80" height="100" rx="20" fill="#0B1F3F" stroke="#2F8FE0" stroke-width="3"/>
    <rect x="-28" y="2" width="56" height="76" rx="10" fill="#12386B"/>
    <rect x="-6" y="-24" width="12" height="16" rx="4" fill="#2F8FE0"/>
    <rect x="-6" y="88" width="12" height="16" rx="4" fill="#2F8FE0"/>
</g>
<g stroke="#7EC8FF" stroke-width="1.4" opacity="0.6">
    <circle cx="660" cy="200" r="4" fill="#00E5FF"/><circle cx="710" cy="170" r="4" fill="#00E5FF"/>
    <circle cx="680" cy="250" r="4" fill="#00E5FF"/><circle cx="760" cy="210" r="4" fill="#00E5FF"/>
    <path d="M660,200 L710,170 M710,170 L760,210 M660,200 L680,250 M680,250 L760,210 M760,210 L790,195"/>
</g>
<g font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.9">
    <text x="940" y="215">Indice di forma</text>
</g>
<text x="940" y="255" font-family="'JetBrains Mono', monospace" font-size="42" font-weight="700" fill="#7EC8FF">82.4%</text>
</svg>"""

# ML / PREVISIONE
SVG_ML = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="mlForecastG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
    <linearGradient id="mlConeG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.25"/><stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.03"/>
    </linearGradient>
</defs>
{_photo_orb(IMG_ML, "Ml", cx=400, cy=250, rx=260, ry=220)}
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" fill="none">
    <path d="M620,170 C700,155 750,135 800,120"/>
    <path d="M640,250 C710,240 760,235 800,230"/>
    <path d="M620,330 C700,335 750,338 800,340"/>
</g>
<g fill="#00E5FF"><circle cx="800" cy="120" r="5"/><circle cx="800" cy="230" r="6"/><circle cx="800" cy="340" r="5"/></g>
<path d="M800,230 C880,205 940,175 1000,150 C1040,133 1080,125 1160,105 L1160,175 C1080,195 1040,203 1000,220 C940,245 880,255 800,275 Z" fill="url(#mlConeG)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/>
</path>
<path d="M800,230 C880,208 940,182 1000,162 C1040,148 1080,132 1160,112"
      fill="none" stroke="url(#mlForecastG)" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="2,12">
    <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.6s" repeatCount="indefinite"/>
</path>
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9"><text x="990" y="380">Rischio overload: previsto in calo</text></g>
</svg>"""

# PIANO ALLENAMENTO
SVG_PLAN = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<g fill="#0A1E3D" opacity="0.9">
    <polygon points="0,430 220,430 140,300"/>
    <polygon points="140,300 220,430 320,340"/>
    <polygon points="220,430 460,430 320,340"/>
</g>
<g fill="#12386B" opacity="0.9">
    <polygon points="320,340 460,430 520,260"/>
    <polygon points="460,430 800,430 520,260"/>
</g>
<g fill="#1B5FA8" opacity="0.95">
    <polygon points="520,260 800,430 760,180"/>
    <polygon points="800,430 1120,430 760,180"/>
</g>
<polygon points="760,180 830,150 900,180" fill="#2F8FE0"/>
<line x1="760" y1="180" x2="760" y2="120" stroke="#7EC8FF" stroke-width="3"/>
<path d="M760,120 L760,145 L800,132 Z" fill="#00E5FF" filter="url(#softGlow)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
</path>
{_photo_orb(IMG_PLAN, "Plan", cx=290, cy=250, rx=240, ry=210)}
</svg>"""

# COMPUTER VISION
SVG_CV = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
{_photo_orb(IMG_CV, "Cv", cx=460, cy=250, rx=300, ry=240, tint="#0B2A55", tint_opacity=0.35)}
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" stroke-dasharray="2,6">
    <path d="M700,180 C780,160 840,150 880,155"/>
    <path d="M700,300 C780,320 840,330 880,335"/>
</g>
<g font-family="'JetBrains Mono', monospace" font-size="16" fill="#7EC8FF" opacity="0.85">
    <text x="900" y="160">GINOCCHIO — 128°</text>
    <text x="900" y="190">CARICO TIBIA — nominale</text>
    <text x="900" y="220">CADENZA — 176 spm</text>
</g>
</svg>"""
