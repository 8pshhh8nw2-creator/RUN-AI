from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import pandas as pd

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, get_svg_url, SVG_ANALISI

st.set_page_config(page_title="Analisi Stato di Forma", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}
    st.session_state.device_connected = False
    st.session_state.diario_note = []

# from utils.components import header_block, get_svg_url, SVG_ANALISI
# ...
IMG_HERO_ANALISI = get_svg_url(SVG_ANALISI)


# ---------------------------------------------------------
# PAGINA 1: ANALISI STATO DI FORMA
# ---------------------------------------------------------
header_block(
    "Modulo 01 — Acquisizione Dati",
    "ANALISI STATO DI FORMA",
    "Inserisci i parametri fisiologici, annota le sensazioni soggettive e seleziona l'obiettivo odierno.",
    IMG_HERO_ANALISI, "Sport Tech Scan"
)

st.markdown("""
<div class='info-box'>
<strong>Configura i parametri odierni e compila il diario delle sensazioni prima di avviare l'analisi predittiva.</strong>
</div>
""", unsafe_allow_html=True)

with st.form("form_analisi"):
    st.markdown("### Obiettivi")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        obj_oggi = st.selectbox("Obiettivo Odierno", ["Leggero", "Medio", "Intermedio"])
    with col_o2:
        distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)

    st.markdown("#### Obiettivo Finale (Lungo Termine)")
    col_of1, col_of2, col_of3 = st.columns(3)
    with col_of1:
        obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
    with col_of2:
        data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
    with col_of3:
        km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)

    st.markdown("---")
    st.markdown("### Sonno e Recupero")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
    with col_s2:
        qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
    with col_s3:
        fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)

    st.markdown("---")
    st.markdown("### Stress Mentale & Diario Sensazioni")
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)
    with col_st2:
        nota_soggettiva = st.text_area("Diario Feedback Atleta (Sensazioni, dolori lievi, umore)", placeholder="Es: Gambe leggermente pesanti dopo il lavoro di ieri, ma buon focus mentale...")

    st.markdown("---")
    st.markdown("### Allenamento Previsto")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
    with col_a2:
        rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)

    st.markdown("---")
    bottone = st.form_submit_button("ANALIZZA STATO DI FORMA", use_container_width=True)

if bottone:
    st.session_state.analisi_fatta = True
    st.session_state.risultati_analisi = {
        'obj_oggi': obj_oggi, 'distanza_oggi': distanza_oggi, 'obj_finale': obj_finale, 'data_obj_finale': data_obj_finale,
        'km_obj_finale': km_obj_finale, 'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno, 'fc_riposo': fc_riposo,
        'stress_lavoro': stress_lavoro, 'ore_lavoro': ore_lavoro, 'tipo_allenamento': tipo_allenamento, 'rpe_previsto': rpe_previsto,
        'nota_soggettiva': nota_soggettiva, 'data_nota': pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')
    }
    if nota_soggettiva.strip():
        st.session_state.diario_note.append({'data': st.session_state.risultati_analisi['data_nota'], 'nota': nota_soggettiva})
    st.success("Stato di forma analizzato e note salvate con successo nel diario!")

if st.session_state.diario_note:
    st.markdown("---")
    st.subheader("Diario Storico Feedback Atleta")
    for item in reversed(st.session_state.diario_note[-5:]):
        st.markdown(f"""
        <div style='background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;'>
            <span style='color: var(--cyan); font-family: "JetBrains Mono", monospace; font-size: 0.8em;'>{item['data']}</span>
            <p style='color: var(--text-dim); margin: 4px 0 0 0; font-size: 0.95em;'>{item['nota']}</p>
        </div>
        """, unsafe_allow_html=True)
