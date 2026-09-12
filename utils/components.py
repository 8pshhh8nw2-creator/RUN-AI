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
# COSTANTI SVG — icone hero per ogni pagina, versione HUD/tech.
#
# Canvas piu' grande (1200x500) e linguaggio visivo condiviso da
# vero pannello di controllo:
#   - griglia tecnica di sfondo, molto tenue
#   - quattro bracket ad angolo (stile mirino di uno strumento)
#   - una scanline orizzontale che attraversa lentamente il pannello
#   - dissolvenza radiale ai bordi, cosi' il pannello si fonde nella
#     pagina invece di avere un bordo netto
# Ogni icona traduce poi il proprio soggetto (percorso, biometria,
# statistiche, KPI, previsione ML, piano, pose) in forma wireframe/
# dati invece che una singola linea illustrativa.
# =========================================================

# HOME — pannello di navigazione: terreno a curve di livello in
# wireframe (profondita' simulata con layer sfalsati), percorso GPS
# che lo attraversa e un runner luminoso che lo percorre in loop.
SVG_HOME = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="homeFade" cx="50%" cy="50%" r="70%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="homeMask"><rect width="1200" height="500" fill="url(#homeFade)"/></mask>
    <pattern id="homeGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="homeRoute" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="10%" stop-color="#00E5FF" stop-opacity="0.95"/>
        <stop offset="45%" stop-color="#00F5A0" stop-opacity="1"/>
        <stop offset="75%" stop-color="#FFB020" stop-opacity="0.95"/>
        <stop offset="92%" stop-color="#00E5FF" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="homeScan" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="50%" stop-color="#00E5FF" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <filter id="homeGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="9" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <path id="homeRoutePath" d="M20,340 C130,340 165,255 260,235 C335,218 350,300 420,325 C500,355 535,420 615,395 C695,372 695,275 775,260 C860,245 900,310 995,285 C1070,265 1130,250 1180,255"/>
</defs>
<g mask="url(#homeMask)">
    <rect width="1200" height="500" fill="url(#homeGrid)" opacity="0.5"/>
    <g opacity="0.16" stroke="#00E5FF" fill="none" stroke-width="1.1">
        <path d="M0,120 C260,80 480,155 720,105 C920,65 1080,120 1200,95"/>
        <path d="M0,175 C260,140 480,205 720,165 C920,130 1080,175 1200,155" opacity="0.7"/>
        <path d="M0,410 C260,450 480,385 720,430 C920,460 1080,415 1200,435"/>
        <path d="M0,455 C260,485 480,435 720,470" opacity="0.7"/>
    </g>
    <use href="#homeRoutePath" fill="none" stroke="url(#homeRoute)" stroke-width="4.5" stroke-linecap="round" filter="url(#homeGlow)"/>
    <g transform="translate(260,235)">
        <circle r="14" fill="none" stroke="#00E5FF" stroke-width="1.5" opacity="0.6"/>
        <circle r="22" fill="none" stroke="#00E5FF" stroke-width="1" stroke-dasharray="3,7" opacity="0.5">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="9s" repeatCount="indefinite"/>
        </circle>
    </g>
    <g transform="translate(775,260)">
        <circle r="10" fill="#FFB020" filter="url(#homeGlow)"/>
        <circle r="24" fill="none" stroke="#FFB020" stroke-width="1" stroke-dasharray="2,8" opacity="0.55">
            <animateTransform attributeName="transform" type="rotate" from="360" to="0" dur="7s" repeatCount="indefinite"/>
        </circle>
    </g>
    <circle r="10" fill="#fff" filter="url(#homeGlow)">
        <animateMotion dur="6.5s" repeatCount="indefinite" rotate="auto"><mpath href="#homeRoutePath"/></animateMotion>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.04;0.92;1" dur="6.5s" repeatCount="indefinite"/>
    </circle>
    <g stroke="#00E5FF" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
    <rect x="0" width="1200" height="6" fill="url(#homeScan)">
        <animate attributeName="y" values="-20;520" dur="5.5s" repeatCount="indefinite"/>
    </rect>
</g>
</svg>"""

# ANALISI STATO DI FORMA — pannello biometrico: anello di readiness a
# segmenti dietro un tracciato a battito che lo attraversa da parte a
# parte, come una vera schermata di monitoraggio.
SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="analisiFade" cx="50%" cy="50%" r="70%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="analisiMask"><rect width="1200" height="500" fill="url(#analisiFade)"/></mask>
    <pattern id="analisiGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="analisiLineGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="8%" stop-color="#00E5FF" stop-opacity="0.95"/>
        <stop offset="40%" stop-color="#00F5A0" stop-opacity="1"/>
        <stop offset="68%" stop-color="#FFB020" stop-opacity="1"/>
        <stop offset="92%" stop-color="#00E5FF" stop-opacity="0.95"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="analisiArc" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF"/><stop offset="50%" stop-color="#00F5A0"/><stop offset="100%" stop-color="#FFB020"/>
    </linearGradient>
    <filter id="analisiGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="8" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <path id="analisiPath" d="M0,300 C130,300 200,270 260,225 C310,188 335,110 390,100 C435,92 460,235 505,320 C540,385 580,335 635,255 C680,190 715,190 770,225 C835,265 880,300 960,270 C1030,244 1080,220 1200,240"/>
</defs>
<g mask="url(#analisiMask)">
    <rect width="1200" height="500" fill="url(#analisiGrid)" opacity="0.5"/>
    <g transform="translate(600,250)" opacity="0.55">
        <circle r="185" fill="none" stroke="#1c2333" stroke-width="16"/>
        <circle r="185" fill="none" stroke="url(#analisiArc)" stroke-width="16" stroke-linecap="round"
                stroke-dasharray="760 1163" transform="rotate(-90)" filter="url(#analisiGlow)">
            <animate attributeName="stroke-dasharray" values="0 1163;760 1163;760 1163" dur="2.8s" keyTimes="0;0.7;1" repeatCount="indefinite"/>
        </circle>
        <circle r="150" fill="none" stroke="#8792A3" stroke-width="1" opacity="0.3"/>
    </g>
    <use href="#analisiPath" fill="none" stroke="url(#analisiLineGrad)" stroke-width="4.5" stroke-linecap="round" filter="url(#analisiGlow)"/>
    <circle cx="390" cy="100" r="8" fill="#00F5A0" filter="url(#analisiGlow)"><animate attributeName="opacity" values="0.5;1;0.5" dur="1.8s" repeatCount="indefinite"/></circle>
    <circle cx="505" cy="320" r="8" fill="#FF6A3D" filter="url(#analisiGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.8s" begin="0.4s" repeatCount="indefinite"/></circle>
    <circle r="7" fill="#fff" filter="url(#analisiGlow)">
        <animateMotion dur="5s" repeatCount="indefinite"><mpath href="#analisiPath"/></animateMotion>
        <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.9;1" dur="5s" repeatCount="indefinite"/>
    </circle>
    <text x="600" y="262" fill="#E8ECF2" font-family="'JetBrains Mono', monospace" font-size="34" font-weight="700" text-anchor="middle" opacity="0.9">READY</text>
    <g stroke="#00F5A0" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""

# STATISTICHE — pannello dati isometrico: barre con profondita' 3D
# finta e riflesso sotto, come una vera scheda analitica da HUD.
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="statsFade" cx="50%" cy="52%" r="72%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="statsMask"><rect width="1200" height="500" fill="url(#statsFade)"/></mask>
    <pattern id="statsGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="barGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0.15"/>
        <stop offset="60%" stop-color="#00F5A0" stop-opacity="0.85"/>
        <stop offset="100%" stop-color="#FFB020" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="reflectGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#00F5A0" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#00F5A0" stop-opacity="0"/>
    </linearGradient>
    <filter id="statsGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#statsMask)">
    <rect width="1200" height="500" fill="url(#statsGrid)" opacity="0.5"/>
    <g opacity="0.18" stroke="#8792A3" stroke-width="1"><line x1="60" y1="420" x2="1140" y2="420"/></g>
    <g transform="skewX(-16)">
        <g fill="url(#barGrad)">
            <rect x="130" y="330" width="34" height="90" opacity="0.5"/>
            <rect x="190" y="280" width="34" height="140" opacity="0.65"/>
            <rect x="250" y="220" width="34" height="200" opacity="0.8"/>
            <rect x="310" y="150" width="34" height="270"><animate attributeName="height" values="270;300;270" dur="2.4s" repeatCount="indefinite"/><animate attributeName="y" values="150;120;150" dur="2.4s" repeatCount="indefinite"/></rect>
            <rect x="370" y="190" width="34" height="230" opacity="0.85"/>
            <rect x="430" y="250" width="34" height="170" opacity="0.7"/>
            <rect x="490" y="120" width="34" height="300"><animate attributeName="height" values="300;330;300" dur="2.7s" begin="0.3s" repeatCount="indefinite"/><animate attributeName="y" values="120;90;120" dur="2.7s" begin="0.3s" repeatCount="indefinite"/></rect>
            <rect x="550" y="170" width="34" height="250" opacity="0.85"/>
            <rect x="610" y="230" width="34" height="190" opacity="0.75"/>
            <rect x="670" y="95"  width="34" height="325"><animate attributeName="height" values="325;350;325" dur="2.1s" begin="0.6s" repeatCount="indefinite"/><animate attributeName="y" values="95;70;95" dur="2.1s" begin="0.6s" repeatCount="indefinite"/></rect>
            <rect x="730" y="160" width="34" height="260" opacity="0.85"/>
            <rect x="790" y="240" width="34" height="180" opacity="0.7"/>
            <rect x="850" y="290" width="34" height="130" opacity="0.55"/>
            <rect x="910" y="200" width="34" height="220" opacity="0.8"/>
            <rect x="970" y="340" width="34" height="80"  opacity="0.4"/>
        </g>
        <g fill="url(#reflectGrad)" transform="translate(0,420) scale(1,-0.25)">
            <rect x="310" y="0" width="34" height="270"/><rect x="490" y="0" width="34" height="300"/><rect x="670" y="0" width="34" height="325"/>
        </g>
    </g>
    <path d="M147,370 C207,340 267,270 327,190 C387,150 447,250 507,270 C567,140 627,215 687,95 C747,190 807,255 867,315 C927,255 987,375 1057,320"
          fill="none" stroke="#ffffff" stroke-width="2" opacity="0.55" filter="url(#statsGlow)"/>
    <circle cx="687" cy="95" r="7" fill="#FF6A3D" filter="url(#statsGlow)"><animate attributeName="r" values="6;10;6" dur="1.6s" repeatCount="indefinite"/></circle>
    <g stroke="#FFB020" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""

# KPI DASHBOARD — torus HUD con satelliti orbitanti e lettura digitale
# centrale, come il quadrante principale di un vero cruscotto dati.
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="kpiFade" cx="50%" cy="50%" r="68%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="kpiMask"><rect width="1200" height="500" fill="url(#kpiFade)"/></mask>
    <pattern id="kpiGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="kpiArc" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF"/><stop offset="55%" stop-color="#00F5A0"/><stop offset="100%" stop-color="#FFB020"/>
    </linearGradient>
    <filter id="kpiGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="7" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <path id="kpiOrbit" d="M600,60 A190,190 0 1,1 599,60"/>
</defs>
<g mask="url(#kpiMask)">
    <rect width="1200" height="500" fill="url(#kpiGrid)" opacity="0.5"/>
    <circle cx="600" cy="250" r="215" fill="none" stroke="#8792A3" stroke-width="1" opacity="0.15"/>
    <g stroke="#8792A3" stroke-width="2" opacity="0.35">
        <g transform="rotate(0 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(24 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(48 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(72 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(96 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(120 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(144 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(168 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(192 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(216 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(240 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(264 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(288 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(312 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
        <g transform="rotate(336 600 250)"><line x1="600" y1="38" x2="600" y2="52"/></g>
    </g>
    <circle cx="600" cy="250" r="175" fill="none" stroke="#1c2333" stroke-width="18" opacity="0.6"/>
    <circle cx="600" cy="250" r="175" fill="none" stroke="url(#kpiArc)" stroke-width="18" stroke-linecap="round"
            stroke-dasharray="905 1100" transform="rotate(-90 600 250)" filter="url(#kpiGlow)">
        <animate attributeName="stroke-dasharray" values="0 1100;905 1100;905 1100" dur="2.6s" keyTimes="0;0.7;1" repeatCount="indefinite"/>
    </circle>
    <circle cx="600" cy="250" r="130" fill="none" stroke="#8792A3" stroke-width="1" stroke-dasharray="1,9" opacity="0.4">
        <animateTransform attributeName="transform" type="rotate" from="0 600 250" to="360 600 250" dur="14s" repeatCount="indefinite"/>
    </circle>
    <circle cx="600" cy="250" r="9" fill="#00E5FF" filter="url(#kpiGlow)"><animate attributeName="r" values="7;11;7" dur="2s" repeatCount="indefinite"/></circle>
    <text x="600" y="264" fill="#E8ECF2" font-family="'JetBrains Mono', monospace" font-size="38" font-weight="700" text-anchor="middle">82.4%</text>
    <circle r="7" fill="#00F5A0" filter="url(#kpiGlow)"><animateMotion dur="8s" repeatCount="indefinite"><mpath href="#kpiOrbit"/></animateMotion></circle>
    <circle r="5" fill="#FFB020" filter="url(#kpiGlow)"><animateMotion dur="8s" begin="-2.6s" repeatCount="indefinite"><mpath href="#kpiOrbit"/></animateMotion></circle>
    <g stroke="#00E5FF" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""

# ML / PREVISIONE — grafo neurale a tre layer con impulsi di dati che
# scorrono lungo le connessioni fino al ramo di previsione, con banda
# di incertezza che si allarga verso il futuro.
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="mlFade" cx="50%" cy="50%" r="70%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="mlMask"><rect width="1200" height="500" fill="url(#mlFade)"/></mask>
    <pattern id="mlGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="mlForecast" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00F5A0"/><stop offset="100%" stop-color="#FFB020"/>
    </linearGradient>
    <linearGradient id="mlCone" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#FFB020" stop-opacity="0.25"/><stop offset="100%" stop-color="#FFB020" stop-opacity="0.04"/>
    </linearGradient>
    <filter id="mlGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="7" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#mlMask)">
    <rect width="1200" height="500" fill="url(#mlGrid)" opacity="0.5"/>
    <g stroke="#00E5FF" stroke-width="1" opacity="0.28" fill="none">
        <path id="mle1" d="M120,150 C220,180 260,230 340,250"/>
        <path id="mle2" d="M120,250 C220,240 260,235 340,250"/>
        <path id="mle3" d="M120,350 C220,320 260,270 340,250"/>
        <path id="mle4" d="M340,250 C420,220 460,180 540,170"/>
        <path id="mle5" d="M340,250 C420,250 460,250 540,250"/>
        <path id="mle6" d="M340,250 C420,290 460,330 540,330"/>
    </g>
    <g fill="#00E5FF" opacity="0.6">
        <circle cx="120" cy="150" r="6"/><circle cx="120" cy="250" r="6"/><circle cx="120" cy="350" r="6"/>
        <circle cx="340" cy="250" r="9"/>
        <circle cx="540" cy="170" r="6"/><circle cx="540" cy="250" r="6"/><circle cx="540" cy="330" r="6"/>
    </g>
    <circle r="4" fill="#fff"><animateMotion dur="2.2s" repeatCount="indefinite"><mpath href="#mle1"/></animateMotion></circle>
    <circle r="4" fill="#fff"><animateMotion dur="2.2s" begin="0.5s" repeatCount="indefinite"><mpath href="#mle3"/></animateMotion></circle>
    <circle r="4" fill="#fff"><animateMotion dur="2s" begin="0.3s" repeatCount="indefinite"><mpath href="#mle4"/></animateMotion></circle>
    <circle r="4" fill="#fff"><animateMotion dur="2s" begin="0.9s" repeatCount="indefinite"><mpath href="#mle6"/></animateMotion></circle>
    <line x1="620" y1="70" x2="620" y2="420" stroke="#8792A3" stroke-width="1" stroke-dasharray="3,6" opacity="0.3"/>
    <path d="M620,250 C700,215 760,175 840,140 C920,105 1000,90 1160,55 L1160,150 C1000,180 920,205 840,235 C760,265 700,255 620,275 Z" fill="url(#mlCone)">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/>
    </path>
    <path d="M620,250 C700,222 760,185 840,160 C920,135 1000,110 1160,80"
          fill="none" stroke="url(#mlForecast)" stroke-width="4" stroke-linecap="round" stroke-dasharray="2,12" filter="url(#mlGlow)">
        <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.6s" repeatCount="indefinite"/>
    </path>
    <circle cx="620" cy="250" r="8" fill="#00F5A0" filter="url(#mlGlow)"><animate attributeName="r" values="7;11;7" dur="1.8s" repeatCount="indefinite"/></circle>
    <g stroke="#00F5A0" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""

# PIANO ALLENAMENTO — dorsale montuosa isometrica a strati verso il
# giorno gara, con checkpoint luminosi e bandiera pulsante sul picco.
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="planFade" cx="50%" cy="55%" r="72%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="planMask"><rect width="1200" height="500" fill="url(#planFade)"/></mask>
    <pattern id="planGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="planLine" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0.3"/>
        <stop offset="15%" stop-color="#00E5FF" stop-opacity="0.95"/>
        <stop offset="45%" stop-color="#00F5A0" stop-opacity="0.95"/>
        <stop offset="72%" stop-color="#FFB020" stop-opacity="0.95"/>
        <stop offset="88%" stop-color="#FF6A3D" stop-opacity="0.95"/>
        <stop offset="100%" stop-color="#FF6A3D" stop-opacity="0.5"/>
    </linearGradient>
    <filter id="planGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="7" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#planMask)">
    <rect width="1200" height="500" fill="url(#planGrid)" opacity="0.5"/>
    <g opacity="0.14" fill="#8792A3">
        <path d="M0,430 L150,330 L300,400 L470,270 L620,360 L800,220 L960,320 L1200,240 L1200,500 L0,500 Z"/>
    </g>
    <g opacity="0.22" fill="#00F5A0">
        <path d="M0,460 L150,380 L300,430 L470,330 L620,400 L800,290 L960,370 L1200,310 L1200,500 L0,500 Z"/>
    </g>
    <path d="M0,450 L80,445 C130,440 160,395 200,360 C240,325 265,395 300,405 C355,420 400,335 480,295 C535,268 550,335 570,355 C610,395 665,270 745,225 C795,197 810,275 835,295 C890,335 930,190 1010,150 C1065,124 1130,205 1200,255"
          fill="none" stroke="url(#planLine)" stroke-width="4.5" stroke-linecap="round" filter="url(#planGlow)"/>
    <circle cx="300" cy="405" r="6" fill="#00E5FF" filter="url(#planGlow)"/>
    <circle cx="570" cy="355" r="6" fill="#00F5A0" filter="url(#planGlow)"/>
    <circle cx="835" cy="295" r="6" fill="#FFB020" filter="url(#planGlow)"/>
    <line x1="1010" y1="150" x2="1010" y2="88" stroke="#FF6A3D" stroke-width="3" filter="url(#planGlow)"/>
    <path d="M1010,88 L1010,120 L1052,104 Z" fill="#FF6A3D" filter="url(#planGlow)">
        <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
    </path>
    <g stroke="#FF6A3D" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""

# COMPUTER VISION — scheletro di pose-estimation dentro una capsula di
# scansione: griglia, scanline verticale che lo attraversa, giunti come
# marker luminosi con readout laterale.
SVG_CV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>
    <radialGradient id="cvFade" cx="46%" cy="52%" r="66%">
        <stop offset="55%" stop-color="#fff"/><stop offset="100%" stop-color="#000"/>
    </radialGradient>
    <mask id="cvMask"><rect width="1200" height="500" fill="url(#cvFade)"/></mask>
    <pattern id="cvGrid" width="46" height="46" patternUnits="userSpaceOnUse">
        <path d="M46,0 L0,0 0,46" fill="none" stroke="#8792A3" stroke-width="0.6"/>
    </pattern>
    <linearGradient id="cvScan" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#00E5FF" stop-opacity="0"/>
        <stop offset="50%" stop-color="#00E5FF" stop-opacity="0.5"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0"/>
    </linearGradient>
    <filter id="cvGlow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
</defs>
<g mask="url(#cvMask)">
    <rect width="1200" height="500" fill="url(#cvGrid)" opacity="0.5"/>
    <rect x="430" y="55" width="340" height="410" rx="14" fill="none" stroke="#00E5FF" stroke-width="1" opacity="0.25"/>
    <g transform="translate(90,60) scale(1.5)" opacity="0.28" stroke="#00E5FF" stroke-width="1.2" fill="none">
        <path d="M330,325 C300,300 285,275 275,245" stroke-dasharray="2,6"/>
        <path d="M545,195 C570,185 590,178 610,168" stroke-dasharray="2,6"/>
    </g>
    <g transform="translate(90,60) scale(1.5)" stroke="#8792A3" stroke-width="3" stroke-linecap="round" fill="none" opacity="0.9">
        <path d="M465,125 L430,135"/><path d="M465,125 L495,120"/>
        <path d="M430,135 L395,110"/><path d="M495,120 L525,155"/>
        <path d="M465,125 L455,205"/><path d="M455,205 L470,200"/><path d="M455,205 L440,210"/>
        <path d="M440,210 L400,255"/><path d="M470,200 L530,230"/>
    </g>
    <g transform="translate(90,60) scale(1.5)">
        <path d="M395,110 L365,85" stroke="#FF6A3D" stroke-width="3" stroke-linecap="round" filter="url(#cvGlow)"/>
        <path d="M525,155 L555,190" stroke="#FF6A3D" stroke-width="3" stroke-linecap="round" filter="url(#cvGlow)"/>
        <path d="M400,255 L345,320" stroke="#FFB020" stroke-width="3" stroke-linecap="round" filter="url(#cvGlow)"/>
        <path d="M530,230 L495,260" stroke="#00F5A0" stroke-width="3" stroke-linecap="round" filter="url(#cvGlow)"/>
        <circle cx="485" cy="95" r="13" fill="#00E5FF" filter="url(#cvGlow)"/>
        <circle cx="465" cy="125" r="4.5" fill="#00E5FF" filter="url(#cvGlow)"><animate attributeName="opacity" values="0.6;1;0.6" dur="1.7s" repeatCount="indefinite"/></circle>
        <circle cx="430" cy="135" r="4.5" fill="#00F5A0" filter="url(#cvGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.7s" begin="0.2s" repeatCount="indefinite"/></circle>
        <circle cx="495" cy="120" r="4.5" fill="#00F5A0" filter="url(#cvGlow)"><animate attributeName="opacity" values="0.5;1;0.5" dur="1.7s" begin="0.4s" repeatCount="indefinite"/></circle>
        <circle cx="365" cy="85" r="4.5" fill="#FF6A3D" filter="url(#cvGlow)"/>
        <circle cx="555" cy="190" r="4.5" fill="#FF6A3D" filter="url(#cvGlow)"/>
        <circle cx="455" cy="205" r="4.5" fill="#E8ECF2" filter="url(#cvGlow)"><animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite"/></circle>
        <circle cx="400" cy="255" r="4.5" fill="#FFB020" filter="url(#cvGlow)"/>
        <circle cx="530" cy="230" r="4.5" fill="#00F5A0" filter="url(#cvGlow)"/>
        <circle cx="345" cy="320" r="4.5" fill="#FFB020" filter="url(#cvGlow)"><animate attributeName="opacity" values="1;0.5;1" dur="1.7s" begin="0.6s" repeatCount="indefinite"/></circle>
        <circle cx="495" cy="260" r="4.5" fill="#00F5A0" filter="url(#cvGlow)"/>
    </g>
    <rect x="430" y="0" width="340" height="10" fill="url(#cvScan)">
        <animate attributeName="y" values="55;445;55" dur="4s" repeatCount="indefinite"/>
    </rect>
    <g font-family="'JetBrains Mono', monospace" font-size="13" fill="#00E5FF" opacity="0.55">
        <text x="800" y="150">HIP  Y 172cm</text>
        <text x="800" y="175">KNEE FLEX 128°</text>
        <text x="800" y="200">CADENCE 176spm</text>
    </g>
    <g stroke="#00F5A0" stroke-width="2.5" fill="none" opacity="0.5" stroke-linecap="square">
        <path d="M36,86 L36,40 L82,40"/><path d="M1118,40 L1164,40 L1164,86"/>
        <path d="M36,414 L36,460 L82,460"/><path d="M1164,414 L1164,460 L1118,460"/>
    </g>
</g>
</svg>"""
