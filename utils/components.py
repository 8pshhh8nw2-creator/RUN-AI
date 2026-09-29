# =========================================================
# LIBRERIA CONDIVISA — illustrazioni vettoriali originali, una per pagina.
# Nessuna foto e nessuna risorsa esterna: tutto SVG puro, coerente con
# la palette blu del portfolio. Il runner è una figura stilizzata
# (_runner) riusata in ogni scena e ambientata in contesti diversi.
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
    <linearGradient id="runnerBody" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="240">
        <stop offset="0%" stop-color="#CFEBFF"/><stop offset="100%" stop-color="#2F8FE0"/>
    </linearGradient>
"""


def _backdrop(cx, cy, rx, ry):
    """Alone luminoso + anelli concentrici tratteggiati dietro al soggetto."""
    return f"""
    <ellipse cx="{cx}" cy="{cy}" rx="{rx + 60}" ry="{ry + 60}" fill="url(#runnerGlow)"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#0A1E3D" opacity="0.55"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#2F8FE0"
             stroke-opacity="0.4" stroke-width="1.5" stroke-dasharray="3,9"/>
    <ellipse cx="{cx}" cy="{cy}" rx="{rx * 0.78}" ry="{ry * 0.78}" fill="none" stroke="#7EC8FF"
             stroke-opacity="0.15" stroke-width="1"/>
    """


def _runner(x, y, s=1.0):
    """Figura stilizzata di runner in corsa. Ingombro locale circa 180x240."""
    return f"""
    <g transform="translate({x},{y}) scale({s})" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <g stroke="#1B5FA8" stroke-width="15">
            <polyline points="98,68 66,96 50,124"/>
            <polyline points="88,145 60,188 24,168"/>
        </g>
        <g stroke="url(#runnerBody)" stroke-width="17">
            <line x1="104" y1="56" x2="88" y2="145"/>
            <polyline points="100,68 134,94 152,68"/>
            <polyline points="88,145 130,170 126,226"/>
        </g>
        <circle cx="108" cy="30" r="19" fill="url(#runnerBody)" stroke="none"/>
    </g>
    """


def _ground(x1, x2, y):
    return f'<ellipse cx="{(x1 + x2) / 2}" cy="{y}" rx="{(x2 - x1) / 2}" ry="12" fill="#000" opacity="0.35"/>'


# HOME — il runner su una strada in prospettiva che si perde
# all'orizzonte, con la traccia GPS tratteggiata e il pin d'arrivo.
SVG_HOME = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="road" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#0A1E3D"/>
    </linearGradient>
    <linearGradient id="homeTrail" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2F8FE0" stop-opacity="0"/>
        <stop offset="100%" stop-color="#00E5FF" stop-opacity="0.9"/>
    </linearGradient>
</defs>
{_backdrop(430, 250, 300, 230)}
<polygon points="0,470 520,470 700,300 560,300" fill="url(#road)" opacity="0.9"/>
<polygon points="520,470 1200,470 1200,440 700,300" fill="#0F2B54" opacity="0.8"/>
<path d="M300,480 L620,300" stroke="#7EC8FF" stroke-width="3" stroke-dasharray="18,16" opacity="0.5"/>
<path d="M60,420 C240,400 300,330 470,340 C600,348 700,300 800,200"
      fill="none" stroke="url(#homeTrail)" stroke-width="4" stroke-dasharray="2,12" stroke-linecap="round"/>
{_ground(250, 470, 440)}
{_runner(250, 130, 1.6)}
<g transform="translate(830,110)">
    <path d="M0,0 C-26,0 -46,20 -46,46 C-46,80 0,120 0,120 C0,120 46,80 46,46 C46,20 26,0 0,0 Z" fill="#1B5FA8" filter="url(#softGlow)"/>
    <circle cx="0" cy="44" r="16" fill="#0B1F3F"/>
    <circle cx="0" cy="44" r="6" fill="#00E5FF"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" repeatCount="indefinite"/></circle>
</g>
<g font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.85">
    <text x="900" y="290">Passo medio</text>
    <text x="900" y="326" font-family="'JetBrains Mono', monospace" font-size="30" font-weight="700" fill="#7EC8FF">4:32/km</text>
</g>
</svg>"""

# ANALISI STATO DI FORMA — sensori sul corpo del runner (cuore, polso,
# caviglia) collegati alle metriche biometriche, con traccia ECG.
SVG_ANALISI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
{_backdrop(450, 250, 290, 235)}
{_runner(300, 60, 1.7)}
<g stroke="#7EC8FF" stroke-width="1" opacity="0.6" fill="none">
    <path d="M467,205 C640,150 820,110 930,90"/>
    <path d="M559,176 C700,200 860,225 950,235"/>
    <path d="M514,440 C680,400 820,370 930,360"/>
</g>
<g fill="#00E5FF">
    <circle cx="467" cy="205" r="7" filter="url(#softGlow)"><animate attributeName="r" values="7;11;7" dur="1.1s" repeatCount="indefinite"/></circle>
    <circle cx="559" cy="176" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="1;0.4;1" dur="1.6s" repeatCount="indefinite"/></circle>
    <circle cx="514" cy="440" r="6" filter="url(#softGlow)"><animate attributeName="opacity" values="0.4;1;0.4" dur="1.6s" begin="0.5s" repeatCount="indefinite"/></circle>
</g>
<polyline points="930,120 985,120 1000,96 1015,144 1030,108 1045,120 1140,120"
          fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linejoin="round" opacity="0.8"/>
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">
    <text x="935" y="80">HRV — 62 ms</text>
    <text x="955" y="228">SMA — 0.41</text>
    <text x="935" y="352">Sonno — 7h20</text>
</g>
</svg>"""

# STATISTICHE — il runner accanto a un istogramma delle sessioni con
# linea di tendenza.
SVG_STATS = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="statBarGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#12386B"/><stop offset="100%" stop-color="#7EC8FF"/>
    </linearGradient>
</defs>
{_backdrop(300, 260, 230, 200)}
{_runner(215, 110, 1.4)}
<g stroke="#7EC8FF" stroke-opacity="0.2">
    <line x1="600" y1="420" x2="1140" y2="420"/><line x1="600" y1="340" x2="1140" y2="340"/>
    <line x1="600" y1="260" x2="1140" y2="260"/><line x1="600" y1="180" x2="1140" y2="180"/>
</g>
<g fill="url(#statBarGrad)">
    <rect x="620" y="330" width="46" height="90" rx="4"/>
    <rect x="690" y="290" width="46" height="130" rx="4"/>
    <rect x="760" y="310" width="46" height="110" rx="4"/>
    <rect x="830" y="240" width="46" height="180" rx="4"/>
    <rect x="900" y="270" width="46" height="150" rx="4"/>
    <rect x="970" y="200" width="46" height="220" rx="4">
        <animate attributeName="height" values="220;245;220" dur="2.4s" repeatCount="indefinite"/>
        <animate attributeName="y" values="200;175;200" dur="2.4s" repeatCount="indefinite"/>
    </rect>
    <rect x="1040" y="230" width="46" height="190" rx="4"/>
</g>
<polyline points="643,320 713,280 783,295 853,225 923,250 993,180 1063,205"
          fill="none" stroke="#00E5FF" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<text x="600" y="460" font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">52 sessioni raccolte</text>
</svg>"""

# KPI DASHBOARD — indicatore circolare ("gauge") con la percentuale
# dell'indice di forma e il runner che lo attraversa.
SVG_KPI = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}
    <linearGradient id="gaugeG" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#2F8FE0"/><stop offset="100%" stop-color="#00E5FF"/>
    </linearGradient>
</defs>
{_backdrop(300, 260, 230, 200)}
{_runner(215, 110, 1.4)}
<g transform="translate(850,250) rotate(-90)">
    <circle r="150" fill="none" stroke="#12386B" stroke-width="26"/>
    <circle r="150" fill="none" stroke="url(#gaugeG)" stroke-width="26" stroke-linecap="round"
            stroke-dasharray="776 943" filter="url(#softGlow)"/>
</g>
<g stroke="#7EC8FF" stroke-opacity="0.35" stroke-dasharray="2,8"><path d="M470,260 L690,250"/></g>
<text x="850" y="248" text-anchor="middle" font-family="Inter, sans-serif" font-size="20" fill="#E8F6FF" opacity="0.85">Indice di forma</text>
<text x="850" y="298" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="52" font-weight="700" fill="#7EC8FF">82.4%</text>
</svg>"""

# ML / PREVISIONE — dal runner parte una rete neurale a tre livelli che
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
{_backdrop(200, 260, 170, 180)}
{_runner(130, 130, 1.2)}
<g stroke="#7EC8FF" stroke-opacity="0.35" stroke-width="1">
    <path d="M400,180 L520,130 M400,180 L520,250 M400,180 L520,370 M400,330 L520,130 M400,330 L520,250 M400,330 L520,370"/>
    <path d="M520,130 L640,190 M520,250 L640,190 M520,370 L640,190 M520,130 L640,320 M520,250 L640,320 M520,370 L640,320"/>
</g>
<g fill="#0B1F3F" stroke="#2F8FE0" stroke-width="2">
    <circle cx="400" cy="180" r="13"/><circle cx="400" cy="330" r="13"/>
    <circle cx="520" cy="130" r="13"/><circle cx="520" cy="250" r="13"/><circle cx="520" cy="370" r="13"/>
    <circle cx="640" cy="190" r="13"/><circle cx="640" cy="320" r="13"/>
</g>
<g fill="#00E5FF"><circle cx="640" cy="190" r="5"/><circle cx="640" cy="320" r="5"/></g>
<path d="M700,255 C800,225 880,185 950,155 C1010,132 1080,120 1170,100 L1170,180 C1080,200 1010,215 950,235 C880,270 800,285 700,290 Z" fill="url(#mlConeG)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="3s" repeatCount="indefinite"/>
</path>
<path d="M700,272 C800,250 880,215 950,195 C1010,175 1080,150 1170,130"
      fill="none" stroke="url(#mlForecastG)" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="2,12">
    <animate attributeName="stroke-dashoffset" values="0;-28" dur="1.6s" repeatCount="indefinite"/>
</path>
<text x="800" y="400" font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">Rischio overload: previsto in calo</text>
</svg>"""

# PIANO ALLENAMENTO — il runner in salita su un sentiero di montagna,
# con tappe intermedie fino alla bandiera del picco.
SVG_PLAN = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
<g fill="#0A1E3D" opacity="0.9">
    <polygon points="0,440 220,440 140,300"/><polygon points="140,300 220,440 320,350"/>
    <polygon points="220,440 480,440 320,350"/>
</g>
<g fill="#12386B" opacity="0.95">
    <polygon points="320,350 480,440 560,260"/><polygon points="480,440 820,440 560,260"/>
</g>
<g fill="#1B5FA8">
    <polygon points="560,260 820,440 780,190"/><polygon points="820,440 1160,440 780,190"/>
</g>
<polygon points="780,190 810,166 850,190 820,200" fill="#CFEBFF" opacity="0.85"/>
<path d="M120,455 C300,440 380,420 470,395 C560,370 600,340 660,320 C720,300 750,260 780,196"
      fill="none" stroke="#7EC8FF" stroke-width="3" stroke-dasharray="2,10" stroke-linecap="round" opacity="0.8"/>
<g fill="#00E5FF">
    <circle cx="470" cy="395" r="7"/><circle cx="660" cy="320" r="7"/>
</g>
<line x1="780" y1="196" x2="780" y2="120" stroke="#7EC8FF" stroke-width="3"/>
<path d="M780,120 L780,148 L826,134 Z" fill="#00E5FF" filter="url(#softGlow)">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="1.6s" repeatCount="indefinite"/>
</path>
{_ground(110, 290, 452)}
{_runner(110, 250, 0.95)}
<g font-family="Inter, sans-serif" font-size="18" fill="#E8F6FF" opacity="0.9">
    <text x="440" y="425">Base</text><text x="630" y="352">Build</text><text x="850" y="150">Picco</text>
</g>
</svg>"""

# COMPUTER VISION — il runner con lo scheletro biomeccanico in
# overlay, i giunti tracciati, l'angolo del ginocchio e i riquadri
# di motion-tracking.
SVG_CV = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500">
<defs>{RUNNER_GLOW_DEFS}</defs>
{_backdrop(460, 260, 300, 235)}
<g opacity="0.35">{_runner(330, 60, 1.7)}</g>
<g stroke="#00E5FF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#softGlow)">
    <polyline points="514,111 500,176 480,307"/>
    <polyline points="500,176 558,220 588,176"/>
    <polyline points="480,307 551,349 544,444"/>
    <polyline points="480,307 432,380 371,346"/>
    <polyline points="500,176 447,225 416,270"/>
</g>
<g fill="#0B1F3F" stroke="#00E5FF" stroke-width="2.5">
    <circle cx="514" cy="111" r="7"/><circle cx="500" cy="176" r="7"/><circle cx="558" cy="220" r="7"/>
    <circle cx="588" cy="176" r="7"/><circle cx="480" cy="307" r="7"/><circle cx="551" cy="349" r="7"/>
    <circle cx="544" cy="444" r="7"/><circle cx="432" cy="380" r="7"/><circle cx="371" cy="346" r="7"/>
</g>
<path d="M520,322 A45,45 0 0 1 528,372" fill="none" stroke="#7EC8FF" stroke-width="2"/>
<g stroke="#7EC8FF" stroke-width="3" fill="none" opacity="0.8">
    <path d="M340,90 h30 M340,90 v30"/><path d="M660,90 h-30 M660,90 v30"/>
    <path d="M340,470 h30 M340,470 v-30"/><path d="M660,470 h-30 M660,470 v-30"/>
</g>
<g stroke="#7EC8FF" stroke-width="1" opacity="0.5" stroke-dasharray="2,6">
    <path d="M580,349 C700,340 780,250 880,165"/><path d="M544,444 C700,380 780,300 880,335"/>
</g>
<g font-family="'JetBrains Mono', monospace" font-size="16" fill="#7EC8FF" opacity="0.9">
    <text x="890" y="160">GINOCCHIO — 128°</text>
    <text x="890" y="190">CARICO TIBIA — nominale</text>
    <text x="890" y="340">CADENZA — 176 spm</text>
</g>
</svg>"""
