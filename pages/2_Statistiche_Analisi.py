from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_KPI

st.set_page_config(page_title="KPI Dashboard", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

IMG_HERO_KPI = get_svg_url(SVG_KPI)

if not st.session_state.analisi_fatta:
    st.warning("Completa prima il questionario...")
else:
    r = st.session_state.risultati_analisi

# ---------------------------------------------------------
# PAGINA 3: KPI DASHBOARD — VERSIONE POTENZIATA
# ---------------------------------------------------------
header_block(
    "Modulo 03 — Live Monitoring",
    "KPI DASHBOARD",
    "Bilancio carico/recupero, indice di rischio e profilo atletico calcolati sui parametri odierni.",
    IMG_HERO_KPI, "Live Pulse Monitor"
)

if not st.session_state.analisi_fatta:
    st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
else:
    r = st.session_state.risultati_analisi
    df_base = st.session_state.dati.copy()

    # ---------------------------------------------------------------
    # HELPER: verdetto a semaforo e spiegazione semplice (coerenti con Pagina 4)
    # ---------------------------------------------------------------
    def verdetto_box(valore_pct, soglie=(50, 75), testo_basso="", testo_medio="", testo_alto="", spiegazione=""):
        if valore_pct >= soglie[1]:
            colore, emoji, etichetta = "#00F5A0", "✅", testo_alto
        elif valore_pct >= soglie[0]:
            colore, emoji, etichetta = "#FFB020", "⚠️", testo_medio
        else:
            colore, emoji, etichetta = "#FF6A3D", "🔴", testo_basso
        st.markdown(f"""
        <div style='border-left: 4px solid {colore}; background: rgba(255,255,255,0.03);
                    padding: 12px 16px; border-radius: 8px; margin: 10px 0;'>
            <span style='font-size: 1.05em; color:{colore}; font-weight:700;'>{emoji} {etichetta}</span>
            <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>{spiegazione}</p>
        </div>
        """, unsafe_allow_html=True)

    def in_pratica(testo):
        st.markdown(f"""
        <div style='background: rgba(0,229,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
        <span style='color:#00E5FF; font-weight:600;'>💡 In parole semplici:</span>
        <span style='color:#B8C2D0; font-family:"Inter",sans-serif;'> {testo}</span>
        </div>
        """, unsafe_allow_html=True)

    def trend_arrow(oggi, media_storica, invert=False, unita=""):
        """Restituisce freccia + delta colorato rispetto alla media storica."""
        delta = oggi - media_storica
        migliora = (delta < 0) if invert else (delta > 0)
        colore = "#00F5A0" if migliora else ("#8792A3" if abs(delta) < 0.01 else "#FF6A3D")
        freccia = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        return f"<span style='color:{colore}; font-weight:700;'>{freccia} {abs(delta):.1f}{unita}</span>"

    # ============================================================
    # BILANCIO CARICO VS RECUPERO
    # ============================================================
    st.markdown("### Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
    df_14 = df_base.tail(14).copy()

    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(
        x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)",
        fill='tozeroy', fillcolor='rgba(255, 106, 61, 0.18)', line=dict(color='#FF6A3D', width=3)
    ))
    fig_balance.add_trace(go.Scatter(
        x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero",
        line=dict(color='#00F5A0', width=4)
    ))
    fig_balance.update_layout(
        height=400, xaxis_title="Giorno", yaxis_title="Indice (base 100 = riferimento ottimale)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#E8ECF2", size=13), bgcolor="rgba(14,20,32,0.85)",
                    bordercolor="#1c2333", borderwidth=1)
    )
    st.plotly_chart(style_fig(fig_balance), use_container_width=True)
    st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Confronta visivamente la curva dello stress fisico (area arancione) con la capacità di recupero biologico (linea verde). Quando la linea verde sovrasta i picchi di carico, il corpo si trova in fase di supercompensazione ottimale.</div>", unsafe_allow_html=True)
    in_pratica("se l'area arancione supera spesso la linea verde, stai accumulando più fatica di quanta il sonno riesca a smaltirne: è il pattern tipico che precede cali di forma o infortuni da sovraccarico.")

    giorni_sforamento = int((df_14['RPE']*10 > (df_14['Ore Sonno']/8)*100).sum())
    st.caption(f"Negli ultimi {len(df_14)} giorni, il carico ha superato la capacità di recupero in **{giorni_sforamento} giorni**.")

    # ============================================================
    # CALCOLO SCORE
    # ============================================================
    risk_score = min(100,
        (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
        (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
        (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
        (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
    )
    recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
    sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

    if risk_score < 25:
        status_color, status_text = "#00F5A0", "OTTIMALE"
    elif risk_score < 60:
        status_color, status_text = "#FFB020", "MODERATO"
    else:
        status_color, status_text = "#FF6A3D", "CRITICO"

    st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 4px; font-family:\"Space Grotesk\",sans-serif;'>{status_text}</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # ============================================================
    # KPI CARDS CON CONFRONTO STORICO
    # ============================================================
    media_sonno_storica = df_base['Ore Sonno'].mean()
    media_rpe_storica = df_base['RPE'].mean()
    media_stress_storica = df_base['Stress Lavoro'].mean() if 'Stress Lavoro' in df_base.columns else r['stress_lavoro']
    sma_storico = ((df_base['Stress Lavoro'] * df_base['RPE']) / df_base['Ore Sonno'].replace(0, np.nan)).mean() if 'Stress Lavoro' in df_base.columns else sma

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {status_color};'>
            <div class='section-label'>Rischio Infortunio</div>
            <div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Combina sonno, stress e sforzo previsto in un unico indice 0-100.")
    with col_k2:
        rec_color = "#00F5A0" if recovery_score >= 75 else "#FFB020" if recovery_score >= 40 else "#FF6A3D"
        st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {rec_color};'>
            <div class='section-label'>Recovery Score</div>
            <div class='data-figure' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        st.caption(f"Sonno oggi: {r['ore_sonno']:.1f}h vs media storica {media_sonno_storica:.1f}h {trend_arrow(r['ore_sonno'], media_sonno_storica, unita='h')}", unsafe_allow_html=True)
    with col_k3:
        sma_color = "#00F5A0" if sma < 10 else "#FFB020" if sma < 15 else "#FF6A3D"
        st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {sma_color};'>
            <div class='section-label'>SMA Score</div>
            <div class='data-figure' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div>
        </div>""", unsafe_allow_html=True)
        if not np.isnan(sma_storico):
            st.caption(f"Oggi vs media storica ({sma_storico:.1f}): {trend_arrow(sma, sma_storico, invert=True)}", unsafe_allow_html=True)

    in_pratica("lo SMA (Stress x RPE / Sonno) sale quando accumuli stress e sforzo con poco sonno a bilanciarli: valori sotto 10 sono un buon equilibrio, sopra 15 indicano un sistema sotto pressione.")

    verdetto_box(
        100 - risk_score, soglie=(40, 75),
        testo_basso=f"Rischio elevato ({risk_score:.0f}%): oggi la combinazione di sonno, stress e sforzo previsto richiede prudenza — valuta di ridurre l'intensità o rimandare l'allenamento",
        testo_medio=f"Rischio moderato ({risk_score:.0f}%): allenati con attenzione, magari abbassando leggermente il carico previsto",
        testo_alto=f"Rischio basso ({risk_score:.0f}%): condizioni favorevoli per allenarti come da programma",
        spiegazione="Questo indice non sostituisce il buon senso: se percepisci dolore o fatica anomala, ascolta il corpo anche se il punteggio è basso."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # GAUGE + RADAR (scala corretta)
    # ============================================================
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#8792A3'}},
            gauge={'axis': {'range': [0, 100], 'tickcolor': "#E8ECF2"}, 'bar': {'color': status_color, 'thickness': 0.75},
                   'bgcolor': "#111827", 'borderwidth': 0,
                   'steps': [{'range': [0, 25], 'color': "rgba(0, 245, 160, 0.08)"},
                             {'range': [25, 60], 'color': "rgba(255, 176, 32, 0.08)"},
                             {'range': [60, 100], 'color': "rgba(255, 106, 61, 0.08)"}]},
            number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
        ))
        fig_gauge.update_layout(height=360)
        st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Tachimetro sintetico che quantifica il livello di pericolo sistemico attuale, associando fasce di colore (Verde = Sicuro, Giallo = Attenzione, Rosso = Rischio elevato).</div>", unsafe_allow_html=True)

    with col_g2:
        # --- Radar con scala normalizzata correttamente su 0-10 per tutti i 4 assi ---
        recovery_su_10 = recovery_score / 10  # 0-100 -> 0-10, coerente con le altre variabili
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_su_10],
            theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (/10)'],
            fill='toself', name='Oggi',
            marker=dict(color=status_color), line=dict(color=status_color)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[media_sonno_storica, media_stress_storica, media_rpe_storica, 7.5],
            theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (/10)'],
            fill='toself', name='Tua media storica',
            marker=dict(color='#8792A3'), line=dict(color='#8792A3', dash='dot'), opacity=0.5
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 12], gridcolor='#1c2333'),
                       angularaxis=dict(gridcolor='#1c2333')),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#E8ECF2", size=11))
        )
        st.plotly_chart(style_fig(fig_radar), use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione Grafico:</strong> Diagramma a ragnatela che mappa l'equilibrio tra i fattori di stress e le risorse di recupero attuali, confrontandoli con la tua media storica (linea tratteggiata grigia).</div>", unsafe_allow_html=True)
        in_pratica("se la forma colorata di 'Oggi' è molto più estesa di quella grigia sui lati Stress e RPE, oggi sei sopra i tuoi standard abituali di carico — motivo in più per monitorare il recupero.")

    st.markdown("---")

    # ============================================================
    # PROFILO ATLETICO AI
    # ============================================================
    st.markdown("### Il Tuo Profilo Atletico AI")
    in_pratica("l'archetipo non è un giudizio fisso: cambia nel tempo in base a come gestisci sonno, stress e carichi. Guarda l'indice di consistenza sotto per capire quanto puoi fidarti di questa etichetta.")

    cv_sonno = df_base['Ore Sonno'].std() / df_base['Ore Sonno'].mean() if df_base['Ore Sonno'].mean() > 0 else 0
    cv_rpe = df_base['RPE'].std() / df_base['RPE'].mean() if df_base['RPE'].mean() > 0 else 0
    consistenza = max(0, 100 - (cv_sonno + cv_rpe) * 100)

    if recovery_score >= 75 and sma < 10:
        archetipo, arch_col, arch_desc = "Il Bilanciato", "#00F5A0", "Gestisci sonno e carichi con grande equilibrio. Il tuo corpo lavora in supercompensazione costante: mantieni questa routine."
    elif r['stress_lavoro'] >= 7 and r['ore_sonno'] < 7:
        archetipo, arch_col, arch_desc = "Il Guerriero Stanco", "#FFB020", "Spingi forte nonostante stress e sonno limitato. Grande grinta, ma il conto arriva: pianifica un blocco di recupero prima possibile."
    elif sma >= 15:
        archetipo, arch_col, arch_desc = "L'Instancabile", "#FF6A3D", "Accumuli carico su carico. Ottimo motore, ma attenzione: senza pause il rischio di crollo fisico o mentale cresce rapidamente."
    else:
        archetipo, arch_col, arch_desc = "Il Costante", "#00E5FF", "Il tuo profilo è stabile e prevedibile: la base ideale su cui costruire progressi graduali e a basso rischio infortuni."

    col_arch1, col_arch2 = st.columns([1, 2])
    with col_arch1:
        st.markdown(f"""
        <div class='kpi-card' style='border-top: 2px solid {arch_col}; display:flex; flex-direction:column; justify-content:center;'>
            <h3 style='color:{arch_col}; margin:5px 0; font-size:1.3em;'>{archetipo}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col_arch2:
        st.markdown(f"""
        <div class='kpi-card' style='text-align:left; height:100%;'>
            <p style='color:#B8C2D0; font-size:1.02em; margin-bottom:15px; font-family:"Inter",sans-serif;'>{arch_desc}</p>
            <p style='color:#8792A3; margin-bottom:5px; font-family:"Inter",sans-serif; font-size:0.9em;'>Indice di Consistenza (90gg)</p>
            <div style='background:#111827; border-radius:8px; overflow:hidden; height:20px;'>
                <div style='background: linear-gradient(90deg, #00E5FF, #00F5A0); width:{min(consistenza,100):.0f}%; height:100%; text-align:right; padding-right:8px; color:#04121a; font-size:0.78em; font-weight:700; line-height:20px; font-family:"JetBrains Mono",monospace;'>{consistenza:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    verdetto_box(
        consistenza, soglie=(40, 70),
        testo_basso="La tua routine di sonno e sforzo varia molto da un giorno all'altro: l'archetipo attuale potrebbe cambiare presto",
        testo_medio="La tua routine è abbastanza regolare, con qualche oscillazione",
        testo_alto="La tua routine è molto stabile: puoi fidarti che questo archetipo rappresenti bene il tuo comportamento reale",
        spiegazione="L'indice si basa sulla variabilità di sonno e RPE negli ultimi 90 giorni: più bassa è la variabilità, più alto è il punteggio di consistenza."
    )
