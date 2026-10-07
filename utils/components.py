"""Hero generativi per RUNAI: particelle di luce 3D su canvas (zero dipendenze, niente CDN).

Uso nelle pagine Streamlit (al posto di image_url=SVG_xxx):

    from hero_engine import header_block
    header_block("TELEMETRIA", "Il tuo allenamento", "Dati che corrono", scene="home")

Scene disponibili: home, analisi, stats, kpi, ml, plan, cv
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

SCENES = ("home", "analisi", "stats", "kpi", "ml", "plan", "cv")
_JS = Path(__file__).with_name("hero_engine.js").read_text(encoding="utf-8")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:transparent;overflow:hidden}
canvas{display:block;width:100%;height:100%}</style></head>
<body><canvas id="c"></canvas><script>window.SCENE="__SCENE__";</script><script>__JS__</script></body></html>"""


def hero_html(scene="home"):
    return _PAGE.replace("__SCENE__", scene).replace("__JS__", _JS)


def hero(scene="home", height=460):
    """Rende un hero animato (iframe trasparente: si fonde con lo sfondo della pagina)."""
    components.html(hero_html(scene), height=height)


def header_block(kicker, title, subtitle, scene="home", height=460, **_legacy):
    """Sostituto drop-in del vecchio header_block: stessi parametri testuali, scena animata a destra."""
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    col_txt, col_img = st.columns([1.0, 1.6])
    with col_txt:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_img:
        hero(scene, height)


def preview_html():
    """Pagina unica con selettore di scena, per guardare tutto nel browser."""
    btns = "".join(f'<button onclick="setScene(\'{s}\')">{s}</button>' for s in SCENES)
    return (_PAGE.replace("__SCENE__", "home").replace("__JS__", _JS)
            .replace("<body>", "<body style='background:#04070D'>"
                     "<style>.bar{position:fixed;top:10px;left:10px;display:flex;gap:6px;z-index:9}"
                     ".bar button{background:#0b1730;color:#7EC8FF;border:1px solid #2F8FE0;border-radius:8px;"
                     "padding:6px 12px;font:600 12px Inter,sans-serif;cursor:pointer}"
                     "html,body{background:#04070D!important}</style>"
                     f"<div class='bar'>{btns}</div>"))


if __name__ == "__main__":
    Path("hero_preview.html").write_text(preview_html(), encoding="utf-8")
    print("scritto hero_preview.html")
