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


# =========================================================
# LIBRERIA CONDIVISA — un corridore "low-poly" (mid-stride, profilo
# destro) costruito a facce triangolari blu, piu' una versione a
# scheletro/giunti per i trattamenti a raggi-X. Stessa identita'
# visiva del portfolio: nessuna griglia, nessun pannello da cruscotto,
# solo il soggetto reso in stile dati.
# Coordinate condivise (spazio locale ~330-600 x, 70-330 y) cosi' le
# due versioni si sovrappongono perfettamente quando servono insieme.
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

RUNNER_MESH = """
<g>
    <polygon points="448,202 408,247 392,263" fill="#0A1E3D"/>
    <polygon points="448,202 392,263 432,218" fill="#12386B"/>
    <polygon points="408,247 353,312 337,328" fill="#0A1E3D"/>
    <polygon points="408,247 337,328 392,263" fill="#12386B"/>
    <polygon points="436,144 401,119 389,101" fill="#0A1E3D"/>
    <polygon points="436,144 389,101 424,126" fill="#12386B"/>
    <polygon points="401,119 371,94 359,76" fill="#0A1E3D"/>
    <polygon points="401,119 359,76 389,101" fill="#12386B"/>
    <polygon points="495,120 430,135 472,146" fill="#1B5FA8"/>
    <polygon points="430,135 440,210 472,146" fill="#12386B"/>
    <polygon points="440,210 470,200 472,146" fill="#1B5FA8"/>
    <polygon points="470,200 495,120 472,146" fill="#2F8FE0"/>
    <polygon points="474,190 536,220 524,240" fill="#2F8FE0"/>
    <polygon points="474,190 524,240 466,210" fill="#1B5FA8"/>
    <polygon points="536,220 503,252 487,268" fill="#2F8FE0"/>
    <polygon points="536,220 487,268 524,240" fill="#1B5FA8"/>
    <polygon points="500,112 533,148 517,163" fill="#2F8FE0"/>
    <polygon points="500,112 517,163 490,128" fill="#1B5FA8"/>
    <polygon points="533,148 562,183 548,198" fill="#2F8FE0"/>
    <polygon points="533,148 548,198 517,163" fill="#7EC8FF"/>
    <circle cx="485" cy="95" r="17" fill="#12386B"/>
    <ellipse cx="479" cy="89" rx="6" ry="4" fill="#7EC8FF" opacity="0.5"/>
</g>
"""

RUNNER_BONES = """
<g stroke="#BFE3FF" stroke-width="3" stroke-linecap="round" fill="none" opacity="0.95">
    <path d="M465,125 L430,135"/><path d="M465,125 L495,120"/>
    <path d="M430,135 L395,110"/><path d="M495,120 L525,155"/>
    <path d="M465,125 L455,205"/><path d="M455,205 L470,200"/><path d="M455,205 L440,210"/>
    <path d="M440,210 L400,255"/><path d="M470,200 L530,230"/>
</g>
<path d="M395,110 L365,85" stroke="#00E5FF" stroke-width="3" stroke-linecap="round"/>
<path d="M525,155 L555,190" stroke="#00E5FF" stroke-width="3" stroke-linecap="round"/>
<path d="M400,255 L345,320" stroke="#7EC8FF" stroke-width="3" stroke-linecap="round"/>
<path d="M530,230 L495,260" stroke="#7EC8FF" stroke-width="3" stroke-linecap="round"/>
<g fill="#E8F6FF">
    <circle cx="485" cy="95" r="12"/>
    <circle cx="465" cy="125" r="5"/><circle cx="430" cy="135" r="5"/><circle cx="495" cy="120" r="5"/>
    <circle cx="395" cy="110" r="5"/><circle cx="365" cy="85" r="5"/><circle cx="525" cy="155" r="5"/><circle cx="555" cy="190" r="5"/>
    <circle cx="455" cy="205" r="5"/><circle cx="470" cy="200" r="5"/><circle cx="440" cy="210" r="5"/>
    <circle cx="400" cy="255" r="5"/><circle cx="345" cy="320" r="5"/><circle cx="530" cy="230" r="5"/><circle cx="495" cy="260" r="5"/>
</g>
"""

_RUNNER_BOX = 'x="220" y="20" width="500" height="440"'  # area occupata dal corridore per centrarlo


# HOME — corridore con la traccia GPS che si dipana dietro, un pin di
# destinazione e la rete di sensori che collega corpo e percorso.
SVG_HOME = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="homeTrail" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0"/>
        <stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.8"/>
    </linearGradient>
</defs>
<ellipse cx="520" cy="230" rx="260" ry="200" fill="url(#runnerGlow)"/>
<path d="M40,340 C160,335 220,300 300,280 C360,265 380,300 420,280"
      fill="none" stroke="url(#homeTrail)" stroke-width="3" stroke-dasharray="1,10" stroke-linecap="round"/>
<g transform="translate(60,60) scale(1.05)">{RUNNER_MESH}</g>
<g transform="translate(830,150)">
    <path d="M0,0 C-26,0 -46,20 -46,46 C-46,80 0,120 0,120 C0,120 46,80 46,46 C46,20 26,0 0,0 Z" fill="#1B5FA8" filter="url(#softGlow)"/>
    <circle cx="0" cy="44" r="16" fill="#0B1F3F"/>
</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.6" stroke-dasharray="2,6">
    <path d="M595,190 C670,170 750,175 800,175"/>
</g>
<circle cx="800" cy="175" r="5" fill="#00E5FF" filter="url(#softGlow)"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/></circle>
<g font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.85">
    <text x="880" y="290">Passo medio</text>
    <text x="880" y="322" font-family="'JetBrains Mono', monospace" font-size="30" font-weight="700" fill="#7EC8FF">4:32/km</text>
</g>
</svg>"""

# ANALISI STATO DI FORMA — la rete di sensori appoggiata direttamente
# sul corpo, con le metriche che ne escono, come una vera lettura
# biometrica live.
SVG_ANALISI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<ellipse cx="600" cy="230" rx="280" ry="210" fill="url(#runnerGlow)"/>
<g transform="translate(280,50) scale(1.15)">{RUNNER_MESH}</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.55">
    <path d="M760,145 C830,110 900,95 960,80"/>
    <path d="M700,235 C800,225 900,235 990,230"/>
    <path d="M655,330 C760,340 860,345 950,360"/>
</g>
<g fill="#00E5FF">
    <circle cx="760" cy="145" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/></circle>
    <circle cx="700" cy="235" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="0.5;1;0.5" dur="1.6s" begin="0.3s" repeatCount="indefinite"/></circle>
    <circle cx="655" cy="330" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.6s" begin="0.6s" repeatCount="indefinite"/></circle>
</g>
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">
    <text x="960" y="75">HRV — 62 ms</text>
    <text x="990" y="225">SMA — 0.41</text>
    <text x="950" y="355">Sonno — 7h20</text>
</g>
</svg>"""

# STATISTICHE — il corridore accanto a uno sparkline di sessioni, con
# la rete che collega il corpo ai dati come nel resto della serie.
SVG_STATS = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="statBarGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
</defs>
<ellipse cx="500" cy="230" rx="260" ry="200" fill="url(#runnerGlow)"/>
<g transform="translate(60,60) scale(1.05)">{RUNNER_MESH}</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" stroke-dasharray="2,6"><path d="M600,200 C700,190 780,200 840,220"/></g>
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

# KPI DASHBOARD — il corridore collegato via rete a uno smartwatch,
# con la lettura percentuale del KPI in evidenza, come nel portfolio.
SVG_KPI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<ellipse cx="520" cy="230" rx="270" ry="200" fill="url(#runnerGlow)"/>
<g transform="translate(80,60) scale(1.05)">{RUNNER_MESH}</g>
<g transform="translate(830,190)">
    <rect x="-40" y="-10" width="80" height="100" rx="20" fill="#0B1F3F" stroke="#2F8FE0" stroke-width="3"/>
    <rect x="-28" y="2" width="56" height="76" rx="10" fill="#12386B"/>
    <rect x="-6" y="-24" width="12" height="16" rx="4" fill="#2F8FE0"/>
    <rect x="-6" y="88" width="12" height="16" rx="4" fill="#2F8FE0"/>
</g>
<g stroke="#7EC8FF" stroke-width="1.4" opacity="0.6">
    <circle cx="640" cy="180" r="4" fill="#00E5FF"/><circle cx="700" cy="150" r="4" fill="#00E5FF"/>
    <circle cx="660" cy="230" r="4" fill="#00E5FF"/><circle cx="760" cy="200" r="4" fill="#00E5FF"/>
    <path d="M640,180 L700,150 M700,150 L760,200 M640,180 L660,230 M660,230 L760,200 M760,200 L790,190"/>
</g>
<g font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.9">
    <text x="940" y="215">Indice di forma</text>
</g>
<text x="940" y="255" font-family="'JetBrains Mono', monospace" font-size="42" font-weight="700" fill="#7EC8FF">82.4%</text>
</svg>"""

# ML / PREVISIONE — dal corridore parte un grafo a tre livelli che
# confluisce in una proiezione futura con banda di incertezza.
SVG_ML = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="mlForecastG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
    <linearGradient id="mlConeG" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0.25"/><stop offset="100%" stop-color="#2F8FE0" stop-opacity="0.03"/>
    </linearGradient>
</defs>
<ellipse cx="460" cy="230" rx="250" ry="200" fill="url(#runnerGlow)"/>
<g transform="translate(40,60) scale(1.05)">{RUNNER_MESH}</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" fill="none">
    <path d="M600,150 C680,140 740,130 800,120"/>
    <path d="M600,230 C680,230 740,230 800,230"/>
    <path d="M600,310 C680,320 740,330 800,340"/>
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

# PIANO ALLENAMENTO — il corridore risale una dorsale montuosa in
# low-poly, stessa palette del corpo, fino alla bandiera del picco.
SVG_PLAN = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<ellipse cx="480" cy="260" rx="280" ry="190" fill="url(#runnerGlow)"/>
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
<g transform="translate(150,175) scale(0.85)">{RUNNER_MESH}</g>
</svg>"""

# COMPUTER VISION — lo stesso corridore reso semitrasparente con lo
# scheletro luminoso sovrapposto, come una lettura a raggi-X.
SVG_CV = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<ellipse cx="530" cy="230" rx="280" ry="210" fill="url(#runnerGlow)"/>
<g transform="translate(80,60) scale(1.15)">
    <g opacity="0.35">{RUNNER_MESH}</g>
    <g filter="url(#softGlow)">{RUNNER_BONES}</g>
</g>
<g font-family="'JetBrains Mono', monospace" font-size="16" fill="#7EC8FF" opacity="0.85">
    <text x="900" y="160">GINOCCHIO — 128°</text>
    <text x="900" y="190">CARICO TIBIA — nominale</text>
    <text x="900" y="220">CADENZA — 176 spm</text>
</g>
</svg>"""
