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
# COSTANTI SVG — icone hero per ogni pagina
#
# Ogni icona ha ora tre strati comuni:
#   1. una maschera radiale (*Fade) che dissolve i bordi verso il
#      trasparente, cosi' l'icona si fonde nello sfondo della pagina
#      invece di leggersi come un riquadro incollato sopra;
#   2. una texture di profondita' molto tenue (puntini o linee guida)
#      dietro al soggetto principale;
#   3. micro-animazioni SVG native (nessun JS): glow che pulsa,
#      tratteggio che scorre, punti che "respirano" — lente e
#      continue, mai invadenti.
# Il soggetto di ogni icona resta legato al mondo della corsa
# data-driven, non decorazione generica da dashboard.
# =========================================================

# HOME — traccia GPS di un percorso corso, con partenza, un runner che
# scorre lungo il tracciato in loop e un waypoint successivo.
SVG_HOME = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="homeFade" cx="50%" cy="50%" r="65%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="homeMask"><rect width="900" height="400" fill="url(#homeFade)"/></mask>
    <linearGradient id="homeRoute" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="8%" stop-color="#00E5FF" stop-opacity="0.9"/>
        <stop offset="45%" stop-color="#00F5A0" stop-opacity="0.95"/>
        <stop offset="75%" stop-color="#FFB020" stop-opacity="0.9"/>
        <stop offset="92%" stop-color="#00E5FF" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <filter id="homeGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="7" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <path id="homeRoutePath" d="M0,260 C90,260 120,190 190,175 C250,162 260,225 315,245 C375,267 400,315 465,295 C525,277 525,200 590,188 C655,176 685,235 755,215 C805,200 855,188 900,192"/>
</defs>
<g mask="url(#homeMask)">
    <g opacity="0.08">
        <circle cx="60" cy="60" r="2" fill="#8792A3"/><circle cx="140" cy="340" r="2" fill="#8792A3"/>
        <circle cx="260" cy="60" r="2" fill="#8792A3"/><circle cx="380" cy="340" r="2" fill="#8792A3"/>
        <circle cx="500" cy="60" r="2" fill="#8792A3"/><circle cx="620" cy="340" r="2" fill="#8792A3"/>
        <circle cx="740" cy="60" r="2" fill="#8792A3"/><circle cx="840" cy="340" r="2" fill="#8792A3"/>
        <path d="M0,90 C220,60 380,120 560,80 C700,50 820,90 900,70" stroke="#8792A3" fill="none" stroke-width="1"/>
        <path d="M0,330 C200,360 420,300 620,340 C740,362 830,320 900,340" stroke="#8792A3" fill="none" stroke-width="1"/>
    </g>
    <use href="#homeRoutePath" fill="none" stroke="url(#homeRoute)" stroke-width="3.5" stroke-linecap="round" filter="url(#homeGlow)"/>
    <circle cx="190" cy="175" r="10" fill="none" stroke="#00E5FF" stroke-width="1.5" opacity="0.5"/>
    <circle cx="465" cy="295" r="9" fill="#00F5A0" filter="url(#homeGlow)">
        <animate attributeName="r" values="7;11;7" dur="2.4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="465" cy="295" r="20" fill="none" stroke="#00F5A0" stroke-width="1.5" opacity="0.35"/>
    <circle cx="465" cy="295" r="32" fill="none" stroke="#00F5A0" stroke-width="1" opacity="0.18"/>
    <circle cx="755" cy="215" r="7" fill="#FFB020" filter="url(#homeGlow)"/>
    <circle r="7" fill="#fff" filter="url(#homeGlow)">
        <animateMotion dur="6s" repeatCount="indefinite" rotate="auto">
            <mpath href="#homeRoutePath"/>
        </animateMotion>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.9;1" dur="6s" repeatCount="indefinite"/>
    </circle>
</g>
</svg>"""

# ANALISI STATO DI FORMA — tracciato a battito con highlight che scorre
# lungo la linea, come un monitor vitale che legge lo stato dell'atleta.
SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="analisiFade" cx="50%" cy="50%" r="65%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="analisiMask"><rect width="900" height="400" fill="url(#analisiFade)"/></mask>
    <linearGradient id="analisiLineGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="10%" stop-color="#00E5FF" stop-opacity="0.9"/>
        <stop offset="40%" stop-color="#00F5A0" stop-opacity="0.95"/>
        <stop offset="68%" stop-color="#FFB020" stop-opacity="0.95"/>
        <stop offset="92%" stop-color="#00E5FF" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <filter id="analisiGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="7" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <path id="analisiPath" d="M0,220 C90,220 140,205 190,175 C230,150 250,95 290,90 C325,86 340,175 370,235 C395,285 420,245 460,195 C495,150 520,150 560,175 C610,207 640,230 700,205 C760,180 800,165 900,178"/>
</defs>
<g mask="url(#analisiMask)">
    <g opacity="0.10">
        <line x1="0" y1="130" x2="900" y2="130" stroke="#8792A3" stroke-width="1"/>
        <line x1="0" y1="270" x2="900" y2="270" stroke="#8792A3" stroke-width="1"/>
    </g>
    <use href="#analisiPath" fill="none" stroke="url(#analisiLineGrad)" stroke-width="3.5" stroke-linecap="round" filter="url(#analisiGlow)"/>
    <circle cx="290" cy="90" r="6" fill="#00F5A0" filter="url(#analisiGlow)">
        <animate attributeName="opacity" values="0.5;1;0.5" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="370" cy="235" r="6" fill="#FF6A3D" filter="url(#analisiGlow)">
        <animate attributeName="opacity" values="1;0.4;1" dur="1.8s" begin="0.4s" repeatCount="indefinite"/>
    </circle>
    <circle r="6" fill="#fff" filter="url(#analisiGlow)">
        <animateMotion dur="5s" repeatCount="indefinite">
            <mpath href="#analisiPath"/>
        </animateMotion>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.9;1" dur="5s" repeatCount="indefinite"/>
    </circle>
</g>
</svg>"""

# STATISTICHE — equalizzatore/photo-finish: barre che "respirano" in modo
# scaglionato, linea di tendenza sopra, tutto racchiuso in una dissolvenza
# morbida invece del riquadro netto di prima.
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="statsFadeMask" cx="50%" cy="55%" r="68%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="statsMask"><rect width="900" height="400" fill="url(#statsFadeMask)"/></mask>
    <linearGradient id="statsFade" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="10%" stop-color="#00E5FF" stop-opacity="1"/>
        <stop offset="90%" stop-color="#FFB020" stop-opacity="1"/>
        <stop offset="100%" stop-color="#FFB020" stop-opacity="0"/>
    </linearGradient>
    <filter id="statsGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="4" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#statsMask)">
    <g opacity="0.10"><line x1="0" y1="340" x2="900" y2="340" stroke="#8792A3" stroke-width="1"/></g>
    <g fill="url(#statsFade)">
        <rect x="30"  y="322" width="8" height="18"  rx="3" opacity="0.12"/>
        <rect x="65"  y="310" width="8" height="30"  rx="3" opacity="0.30"/>
        <rect x="100" y="285" width="8" height="55"  rx="3" opacity="0.55"/>
        <rect x="135" y="255" width="8" height="85"  rx="3" opacity="0.78"/>
        <rect x="170" y="225" width="8" height="115" rx="3" opacity="0.9"/>
        <rect x="205" y="190" width="8" height="150" rx="3" opacity="0.92">
            <animate attributeName="height" values="150;168;150" dur="2.2s" begin="0.0s" repeatCount="indefinite"/>
            <animate attributeName="y" values="190;172;190" dur="2.2s" begin="0.0s" repeatCount="indefinite"/>
        </rect>
        <rect x="240" y="160" width="8" height="180" rx="3" opacity="0.92"/>
        <rect x="275" y="135" width="8" height="205" rx="3" opacity="0.92">
            <animate attributeName="height" values="205;222;205" dur="2.6s" begin="0.3s" repeatCount="indefinite"/>
            <animate attributeName="y" values="135;118;135" dur="2.6s" begin="0.3s" repeatCount="indefinite"/>
        </rect>
        <rect x="310" y="115" width="8" height="225" rx="3" opacity="0.92"/>
        <rect x="345" y="135" width="8" height="205" rx="3" opacity="0.9"/>
        <rect x="380" y="165" width="8" height="175" rx="3" opacity="0.9">
            <animate attributeName="height" values="175;192;175" dur="2.4s" begin="0.6s" repeatCount="indefinite"/>
            <animate attributeName="y" values="165;148;165" dur="2.4s" begin="0.6s" repeatCount="indefinite"/>
        </rect>
        <rect x="415" y="190" width="8" height="150" rx="3" opacity="0.85"/>
        <rect x="450" y="170" width="8" height="170" rx="3" opacity="0.85"/>
        <rect x="485" y="140" width="8" height="200" rx="3" opacity="0.9"/>
        <rect x="520" y="105" width="8" height="235" rx="3" opacity="0.92">
            <animate attributeName="height" values="235;252;235" dur="2.1s" begin="0.9s" repeatCount="indefinite"/>
            <animate attributeName="y" values="105;88;105" dur="2.1s" begin="0.9s" repeatCount="indefinite"/>
        </rect>
        <rect x="555" y="85"  width="8" height="255" rx="3" opacity="0.94"/>
        <rect x="590" y="110" width="8" height="230" rx="3" opacity="0.9"/>
        <rect x="625" y="145" width="8" height="195" rx="3" opacity="0.85"/>
        <rect x="660" y="180" width="8" height="160" rx="3" opacity="0.78"/>
        <rect x="695" y="215" width="8" height="125" rx="3" opacity="0.65"/>
        <rect x="730" y="245" width="8" height="95"  rx="3" opacity="0.5"/>
        <rect x="765" y="270" width="8" height="70"  rx="3" opacity="0.35"/>
        <rect x="800" y="298" width="8" height="42"  rx="3" opacity="0.2"/>
        <rect x="835" y="318" width="8" height="22"  rx="3" opacity="0.1"/>
    </g>
    <path d="M30,300 C100,290 140,225 205,180 C260,142 300,105 345,120 C390,135 415,205 450,220 C490,237 530,150 590,105 C640,68 690,190 730,235 C765,270 800,285 835,305"
          fill="none" stroke="#ffffff" stroke-width="2" opacity="0.5" filter="url(#statsGlow)"/>
    <circle cx="555" cy="85" r="6" fill="#FF6A3D" filter="url(#statsGlow)">
        <animate attributeName="r" values="5;8;5" dur="1.6s" repeatCount="indefinite"/>
    </circle>
</g>
</svg>"""

# KPI DASHBOARD — cronometro sportivo: ghiera a tacche, arco di
# completamento che si "carica" in loop, lancetta e lettura digitale.
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="kpiFade" cx="50%" cy="50%" r="62%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="kpiMask"><rect width="900" height="400" fill="url(#kpiFade)"/></mask>
    <linearGradient id="kpiArc" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF"/>
        <stop offset="55%" stop-color="#00F5A0"/>
        <stop offset="100%" stop-color="#FFB020"/>
    </linearGradient>
    <filter id="kpiGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#kpiMask)">
    <circle cx="450" cy="200" r="155" fill="none" stroke="#8792A3" stroke-width="1" opacity="0.15"/>
    <g stroke="#8792A3" stroke-width="2" opacity="0.35">
        <g transform="rotate(0 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(30 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(60 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(90 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(120 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(150 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(180 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(210 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(240 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(270 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(300 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
        <g transform="rotate(330 450 200)"><line x1="450" y1="48" x2="450" y2="60"/></g>
    </g>
    <circle cx="450" cy="200" r="130" fill="none" stroke="#1c2333" stroke-width="14" opacity="0.6"/>
    <circle cx="450" cy="200" r="130" fill="none" stroke="url(#kpiArc)" stroke-width="14" stroke-linecap="round"
            stroke-dasharray="673 817" transform="rotate(-90 450 200)" filter="url(#kpiGlow)">
        <animate attributeName="stroke-dasharray" values="0 817;673 817;673 817" dur="2.6s" keyTimes="0;0.7;1" repeatCount="indefinite"/>
    </circle>
    <line x1="450" y1="200" x2="450" y2="82" stroke="#00E5FF" stroke-width="2" opacity="0.5" transform="rotate(216 450 200)"/>
    <circle cx="450" cy="200" r="7" fill="#00E5FF" filter="url(#kpiGlow)">
        <animate attributeName="r" values="6;9;6" dur="2s" repeatCount="indefinite"/>
    </circle>
    <text x="450" y="212" fill="#E8ECF2" font-family="'JetBrains Mono', monospace" font-size="30" font-weight="700" text-anchor="middle">82.4%</text>
</g>
</svg>"""

# ML / PREVISIONE — trend osservato che si prolunga in una previsione,
# con banda di confidenza che si allarga e "respira" nel tempo, a
# suggerire l'incertezza che cresce guardando piu' avanti.
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="mlFade" cx="50%" cy="50%" r="65%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="mlMask"><rect width="900" height="400" fill="url(#mlFade)"/></mask>
    <linearGradient id="mlHist" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="12%" stop-color="#00E5FF" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#00F5A0" stop-opacity="0.95"/>
    </linearGradient>
    <linearGradient id="mlForecast" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00F5A0"/>
        <stop offset="100%" stop-color="#FFB020"/>
    </linearGradient>
    <linearGradient id="mlCone" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#FFB020" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="#FFB020" stop-opacity="0.05"/>
    </linearGradient>
    <filter id="mlGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#mlMask)">
    <line x1="480" y1="70" x2="480" y2="320" stroke="#8792A3" stroke-width="1" stroke-dasharray="3,5" opacity="0.3"/>
    <path d="M480,205 C560,180 620,150 700,120 C760,100 820,85 900,50 L900,130 C820,155 760,160 700,175 C620,195 560,205 480,225 Z"
          fill="url(#mlCone)">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/>
    </path>
    <path d="M0,260 C80,255 130,230 190,205 C250,180 290,150 350,170 C400,186 420,230 480,215"
          fill="none" stroke="url(#mlHist)" stroke-width="3.5" stroke-linecap="round" filter="url(#mlGlow)"/>
    <path d="M480,215 C560,192 620,165 700,148 C760,135 820,112 900,90"
          fill="none" stroke="url(#mlForecast)" stroke-width="3" stroke-linecap="round" stroke-dasharray="2,10" filter="url(#mlGlow)">
        <animate attributeName="stroke-dashoffset" values="0;-24" dur="1.6s" repeatCount="indefinite"/>
    </path>
    <circle cx="480" cy="215" r="6" fill="#00F5A0" filter="url(#mlGlow)">
        <animate attributeName="r" values="5;8;5" dur="1.8s" repeatCount="indefinite"/>
    </circle>
</g>
</svg>"""

# PIANO ALLENAMENTO — profilo altimetrico dei blocchi di carico verso il
# giorno gara: il colore segue la fase (base -> build -> picco), e la
# bandierina sul picco pulsa per segnare il traguardo.
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="planFadeMask" cx="50%" cy="55%" r="68%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="planMask"><rect width="900" height="400" fill="url(#planFadeMask)"/></mask>
    <linearGradient id="planLine" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0.3"/>
        <stop offset="15%" stop-color="#00E5FF" stop-opacity="0.95"/>
        <stop offset="45%" stop-color="#00F5A0" stop-opacity="0.95"/>
        <stop offset="72%" stop-color="#FFB020" stop-opacity="0.95"/>
        <stop offset="88%" stop-color="#FF6A3D" stop-opacity="0.95"/>
        <stop offset="100%" stop-color="#FF6A3D" stop-opacity="0.5"/>
    </linearGradient>
    <linearGradient id="planFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#00F5A0" stop-opacity="0.20"/>
        <stop offset="100%" stop-color="#00F5A0" stop-opacity="0"/>
    </linearGradient>
    <filter id="planGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#planMask)">
    <path d="M0,340 L60,335 C100,330 120,290 150,260 C180,232 200,290 230,300 C270,313 300,250 360,220 C400,200 415,255 430,270 C460,297 500,205 560,170 C600,148 610,215 630,230 C670,258 700,140 760,110 C800,90 850,150 900,190 L900,340 Z"
          fill="url(#planFill)"/>
    <path d="M0,340 L60,335 C100,330 120,290 150,260 C180,232 200,290 230,300 C270,313 300,250 360,220 C400,200 415,255 430,270 C460,297 500,205 560,170 C600,148 610,215 630,230 C670,258 700,140 760,110 C800,90 850,150 900,190"
          fill="none" stroke="url(#planLine)" stroke-width="3.5" stroke-linecap="round" filter="url(#planGlow)"/>
    <line x1="760" y1="110" x2="760" y2="65" stroke="#FF6A3D" stroke-width="2.5" filter="url(#planGlow)"/>
    <path d="M760,65 L760,90 L792,78 Z" fill="#FF6A3D" filter="url(#planGlow)">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
    </path>
</g>
</svg>"""

# COMPUTER VISION — scheletro di pose-estimation su un atleta a meta'
# falcata, con i giunti che pulsano come marker di tracking live.
SVG_CV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
<defs>
    <radialGradient id="cvFade" cx="48%" cy="52%" r="62%">
        <stop offset="55%" stop-color="#fff"/>
        <stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="cvMask"><rect width="900" height="400" fill="url(#cvFade)"/></mask>
    <filter id="cvGlow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="4" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#cvMask)">
    <g opacity="0.25" stroke="#00E5FF" stroke-width="1.5" fill="none">
        <path d="M330,325 C300,300 285,275 275,245" stroke-dasharray="2,6"/>
        <path d="M545,195 C570,185 590,178 610,168" stroke-dasharray="2,6"/>
    </g>
    <g stroke="#8792A3" stroke-width="3.5" stroke-linecap="round" fill="none" opacity="0.9">
        <path d="M465,125 L430,135"/>
        <path d="M465,125 L495,120"/>
        <path d="M430,135 L395,110"/>
        <path d="M495,120 L525,155"/>
        <path d="M465,125 L455,205"/>
        <path d="M455,205 L470,200"/>
        <path d="M455,205 L440,210"/>
        <path d="M440,210 L400,255"/>
        <path d="M470,200 L530,230"/>
    </g>
    <path d="M395,110 L365,85" stroke="#FF6A3D" stroke-width="3.5" stroke-linecap="round" filter="url(#cvGlow)"/>
    <path d="M525,155 L555,190" stroke="#FF6A3D" stroke-width="3.5" stroke-linecap="round" filter="url(#cvGlow)"/>
    <path d="M400,255 L345,320" stroke="#FFB020" stroke-width="3.5" stroke-linecap="round" filter="url(#cvGlow)"/>
    <path d="M530,230 L495,260" stroke="#00F5A0" stroke-width="3.5" stroke-linecap="round" filter="url(#cvGlow)"/>
    <circle cx="485" cy="95" r="15" fill="#00E5FF" filter="url(#cvGlow)"/>
    <circle cx="465" cy="125" r="5" fill="#00E5FF" filter="url(#cvGlow)"><animate attributeName="opacity" values="0.6;1;0.6" dur="1.7s" repeatCount="indefinite"/></circle>
    <circle cx="430" cy="135" r="5" fill="#00F5A0" filter="url(#cvGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.7s" begin="0.2s" repeatCount="indefinite"/></circle>
    <circle cx="495" cy="120" r="5" fill="#00F5A0" filter="url(#cvGlow)"><animate attributeName="opacity" values="0.5;1;0.5" dur="1.7s" begin="0.4s" repeatCount="indefinite"/></circle>
    <circle cx="365" cy="85" r="5" fill="#FF6A3D" filter="url(#cvGlow)"/>
    <circle cx="555" cy="190" r="5" fill="#FF6A3D" filter="url(#cvGlow)"/>
    <circle cx="455" cy="205" r="5" fill="#E8ECF2" filter="url(#cvGlow)"><animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite"/></circle>
    <circle cx="400" cy="255" r="5" fill="#FFB020" filter="url(#cvGlow)"/>
    <circle cx="530" cy="230" r="5" fill="#00F5A0" filter="url(#cvGlow)"/>
    <circle cx="345" cy="320" r="5" fill="#FFB020" filter="url(#cvGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.7s" begin="0.6s" repeatCount="indefinite"/></circle>
    <circle cx="495" cy="260" r="5" fill="#00F5A0" filter="url(#cvGlow)"/>
</g>
</svg>"""
