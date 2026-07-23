"""
utils/kpi_components.py (o aggiungi queste funzioni al tuo utils/components.py)
--------------------------------------------------------------------------------
Componenti condivisi per mostrare QUALSIASI KPI (SMA, ISLR, IITR, IDET, Risk
Score, Recovery, Consistenza...) con lo stesso identico linguaggio visivo,
cosi' l'utente impara il pattern una volta e lo ritrova ovunque nell'app.

Ho spostato qui dentro verdetto_box(), in_pratica() e trend_arrow() che nella
Pagina 3 originale erano definite localmente dentro l'if — spostandole qui
puoi riusarle in tutte le pagine senza duplicare codice.
"""

import streamlit as st
import plotly.graph_objects as go
from .kpi_engine import FEATURE_IMPORTANCE_CHART_DATA


# ============================================================
# HELPER TESTUALI (spostati da Pagina 3, invariati)
# ============================================================
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


def azione_consigliata(testo):
    """NUOVO: box per il consiglio pratico pre/post allenamento legato al KPI."""
    st.markdown(f"""
    <div style='background: rgba(0,245,160,0.06); border: 1px solid rgba(0,245,160,0.2);
                border-radius: 8px; padding: 10px 14px; margin-top: 8px;'>
    <span style='color:#00F5A0; font-weight:600;'>🎯 Cosa fare oggi:</span>
    <span style='color:#B8C2D0; font-family:"Inter",sans-serif;'> {testo}</span>
    </div>
    """, unsafe_allow_html=True)


def trend_arrow(oggi, media_storica, invert=False, unita=""):
    delta = oggi - media_storica
    migliora = (delta < 0) if invert else (delta > 0)
    neutro = abs(delta) < 0.01
    colore = "#8792A3" if neutro else ("#00F5A0" if migliora else "#FF6A3D")
    freccia = "▬" if neutro else ("▲" if delta > 0 else "▼")
    return f"<span style='color:{colore}; font-weight:700;'>{freccia} {abs(delta):.1f}{unita}</span>"


# ============================================================
# CARD KPI RIUTILIZZABILE CON SPARKLINE
# ============================================================
def kpi_card_sparkline(titolo, valore, colore, serie_storica, giorni_asse, unita=""):
    """
    Card KPI con numero grande + mini-grafico andamento (senza assi, stile
    sparkline) inline nella stessa card. Gestisce in modo sicuro i colori rgba.
    """
    # Conversione sicura del colore esadecimale in rgba per il riempimento
    if colore.startswith("#") and len(colore) == 7:
        r = int(colore[1:3], 16)
        g = int(colore[3:5], 16)
        b = int(colore[5:7], 16)
        fill_colore = f"rgba({r}, {g}, {b}, 0.12)"
    else:
        fill_colore = "rgba(0, 245, 160, 0.12)"

    col_num, col_graf = st.columns([1, 1.4])
    with col_num:
        st.markdown(f"""<div class='kpi-card' style='border-top: 2px solid {colore}; height:100%;'>
            <div class='section-label'>{titolo}</div>
            <div class='data-figure' style='font-size:2em; font-weight:bold; color: {colore};'>{valore:.1f}{unita}</div>
        </div>""", unsafe_allow_html=True)
    with col_graf:
        fig = go.Figure(go.Scatter(
            x=giorni_asse, y=serie_storica, mode="lines", line=dict(color=colore, width=3),
            fill="tozeroy", fillcolor=fill_colore
        ))
        fig.update_layout(
            height=90, margin=dict(l=0, r=0, t=4, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# GRAFICO FEATURE IMPORTANCE (dal Random Forest della tesi)
# ============================================================
def feature_importance_chart(style_fig_fn=None):
    """
    Riproduce nell'app il grafico di Feature Importance del tuo Random
    Forest (Capitolo 2). style_fig_fn e' la tua funzione style_fig() di
    utils/components.py, passala per uniformare i colori con le altre pagine.
    """
    etichette = [x[0] for x in FEATURE_IMPORTANCE_CHART_DATA][::-1]
    valori = [x[1] for x in FEATURE_IMPORTANCE_CHART_DATA][::-1]

    fig = go.Figure(go.Bar(
        x=valori, y=etichette, orientation="h",
        marker=dict(color=["#FF6A3D" if v >= 10 else "#8792A3" for v in valori]),
        text=[f"{v:.1f}%" for v in valori], textposition="outside",
    ))
    fig.update_layout(
        height=340, xaxis_title="Contributo alla predizione di Overload (%)",
        margin=dict(l=10, r=40, t=20, b=30),
    )
    fig.add_vline(x=10, line_dash="dash", line_color="#8792A3",
                  annotation_text="Soglia di alta rilevanza (>10%)", annotation_position="top")
    return style_fig_fn(fig) if style_fig_fn else fig
