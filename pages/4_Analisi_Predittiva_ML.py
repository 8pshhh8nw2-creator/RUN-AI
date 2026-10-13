import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, get_svg_url, style_fig, SVG_ML
from utils.sidebar import sidebar_comune
from utils.ml_engine import (
    get_bundle_classificazione, get_bundle_regressione, get_bundle_cluster,
    stima_rischio_oggi, FEATURE_LABELS_CLASS,
)

# 1. Configurazione pagina
st.set_page_config(page_title="Rischio e Previsioni", layout="wide")
carica_css()

# 2. Stato base
if 'dati' not in st.session_state or st.session_state.dati is None:
    st.session_state.dati = genera_dati()
if 'device_connected' not in st.session_state:
    st.session_state.device_connected = False
if 'diario_note' not in st.session_state:
    st.session_state.diario_note = []
if 'analisi_fatta' not in st.session_state:
    st.session_state.analisi_fatta = False
if 'risultati_analisi' not in st.session_state:
    st.session_state.risultati_analisi = {}

# 3. Sidebar comune
sidebar_result = sidebar_comune()
if sidebar_result and isinstance(sidebar_result, tuple) and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati.copy()
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

IMG_HERO_ML = get_svg_url(SVG_ML)

BG_DARK, BG_DARK2, BD = "#0B1017", "#0E1420", "#202B3D"
TXT_PRI, TXT_SEC, TXT_TER = "#FFFFFF", "#B8C2D0", "#8792A3"
C_CYAN, C_AMBER, C_ORANGE, C_GREEN = "#00E5FF", "#FFB020", "#FF6A3D", "#00F5A0"
RISK_SCALE = [[0, C_GREEN], [0.3, C_AMBER], [1, C_ORANGE]]


def dot(c):
    return f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};margin-right:8px;vertical-align:middle;'></span>"


def colore_rischio(p):
    return C_ORANGE if p >= 60 else C_AMBER if p >= 25 else C_GREEN


header_block(
    "Modulo 04 — Le previsioni",
    "RISCHIO E PREVISIONI",
    "Il sistema studia i tuoi allenamenti passati e ti dice quando il rischio di farti male sale.",
    IMG_HERO_ML, "Previsioni"
)

df_base = st.session_state.dati.copy()

st.markdown(f"""
<div class='info-box'>
<h3>Cosa fa questa pagina, in parole semplici</h3>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema guarda tutte le tue uscite (km, sonno, stress, cuore, fatica percepita) e impara quali combinazioni hanno preceduto i giorni a rischio. Poi le usa per avvisarti prima. Sono gli stessi calcoli che decidono il verdetto nella pagina "Consiglio Finale".<br><br>
<strong style='color:#fff;'>Colori:</strong> &nbsp;{dot(C_GREEN)}verde = tutto ok &nbsp;&nbsp; {dot(C_AMBER)}giallo = attenzione &nbsp;&nbsp; {dot(C_ORANGE)}rosso = meglio rallentare</p>
</div>
""", unsafe_allow_html=True)

try:
    feature_names = FEATURE_LABELS_CLASS

    class_bundle = get_bundle_classificazione(df_base)
    rf_model = class_bundle["rf_model"]
    log_model = class_bundle["log_model"]
    scaler = class_bundle["scaler"]

    y_real = df_base['Rischio Infortunio'].values

    # Previsioni "oneste": ogni giorno è previsto da un modello che non l'ha mai visto
    y_pred_rf = class_bundle["y_pred_rf"]
    y_proba_rf = class_bundle["y_proba_rf"]
    y_proba_log = class_bundle["y_proba_log"]
    y_pred_log = (y_proba_log >= 0.5).astype(int)

    def rischio_rf(X):
        """Rischio (%) della giuria per una matrice [km, sonno, stress, fc, rpe]."""
        return rf_model.predict_proba(scaler.transform(np.asarray(X, dtype=float)))[:, 1] * 100

    st.markdown(f"""
    <style>
    @keyframes mlxReveal {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @keyframes mlxGlow {{ 0%,100% {{ box-shadow:0 0 0 rgba(0,229,255,0); }} 50% {{ box-shadow:0 0 22px rgba(0,229,255,0.22); }} }}
    .mlx-hero {{ background: linear-gradient(135deg, {BG_DARK2} 0%, #101A2E 100%); border:1px solid {BD}; border-radius:18px; padding:28px 32px; margin:6px 0 26px 0; box-shadow:0 8px 26px rgba(0,0,0,0.28); animation: mlxReveal .6s ease-out; }}
    .mlx-eyebrow {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:{TXT_TER}; font-weight:700; margin:0 0 8px 0; }}
    .mlx-hero-title {{ font-family:'Inter', sans-serif; font-weight:700; font-size:1.5rem; color:{TXT_PRI}; margin:0 0 6px 0; }}
    .mlx-hero-msg {{ font-family:'Inter', sans-serif; color:{TXT_SEC}; font-size:.95rem; line-height:1.6; margin:0 0 20px 0; }}
    .mlx-stat-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:16px; }}
    .mlx-stat-card {{ background: rgba(255,255,255,0.025); border:1px solid {BD}; border-top:3px solid var(--stat-color, {BD}); border-radius:12px; padding:16px 18px; animation: mlxReveal .7s ease-out both; }}
    .mlx-stat-card:nth-child(2) {{ animation-delay:.08s; }} .mlx-stat-card:nth-child(3) {{ animation-delay:.16s; }}
    .mlx-stat-card:nth-child(4) {{ animation-delay:.24s; }} .mlx-stat-card:nth-child(5) {{ animation-delay:.32s; }}
    .mlx-stat-card .v {{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.7rem; color: var(--stat-color, {TXT_PRI}); }}
    .mlx-stat-card .l {{ font-family:'Inter', sans-serif; font-size:.82rem; color:{TXT_SEC}; margin-top:7px; line-height:1.45; }}
    .mlx-section-head {{ display:flex; align-items:baseline; gap:14px; margin: 26px 0 12px 0; }}
    .mlx-section-head h3 {{ margin:0; font-family:'Inter',sans-serif; font-weight:700; color:{TXT_PRI}; font-size:1.15rem; }}
    .mlx-section-head .rule {{ flex:1; height:1px; background: linear-gradient(90deg, {BD} 0%, transparent 100%); align-self:center; }}
    .mlx-insight {{ border-left:3px solid var(--ic-color, {C_CYAN}); background: rgba(255,255,255,0.02); padding:12px 16px; border-radius:0 8px 8px 0; margin-top:10px; font-family:'Inter',sans-serif; font-size:.93rem; color:{TXT_SEC}; line-height:1.6; }}
    .mlx-insight strong {{ color:{TXT_PRI}; }}
    .mlx-box {{ background: rgba(255,255,255,0.03); border:1px solid var(--bc, {BD}); border-radius:12px; padding:16px 18px; text-align:center; animation: mlxReveal .6s ease-out both; }}
    .mlx-box .n {{ font-family:'JetBrains Mono', monospace; font-size:2rem; font-weight:700; color: var(--bc, {TXT_PRI}); }}
    .mlx-box .t {{ font-family:'Inter',sans-serif; font-size:.85rem; color:{TXT_SEC}; line-height:1.4; margin-top:4px; }}
    .mlx-rule {{ background: linear-gradient(160deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%); border:1px solid var(--bc, {BD}); border-radius:16px; padding:22px 20px; text-align:center; animation: mlxReveal .7s ease-out both; }}
    .mlx-rule .k {{ font-family:'JetBrains Mono', monospace; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:{TXT_TER}; }}
    .mlx-rule .big {{ font-family:'JetBrains Mono', monospace; font-size:2.6rem; font-weight:700; color: var(--bc, {TXT_PRI}); margin:8px 0 4px 0; }}
    .mlx-rule .d {{ font-family:'Inter',sans-serif; font-size:.88rem; color:{TXT_SEC}; line-height:1.5; }}
    .mlx-bigsafe {{ background: linear-gradient(135deg, #0E1F1B 0%, #0E1420 100%); border:1px solid {C_GREEN}55; border-radius:16px; padding:22px 26px; margin:16px 0; animation: mlxGlow 3.5s ease-in-out infinite; }}
    .mlx-bigsafe .big {{ font-family:'JetBrains Mono', monospace; font-size:2.8rem; font-weight:700; color:{C_GREEN}; }}
    .mlx-note {{ font-family:'Inter',sans-serif; font-size:.74rem; color:{TXT_TER}; margin-top:6px; }}
    </style>
    """, unsafe_allow_html=True)

    def mlx_section(title):
        st.markdown(f"<div class='mlx-section-head'><h3>{title}</h3><div class='rule'></div></div>", unsafe_allow_html=True)

    def mlx_insight(html_text, color=C_CYAN):
        st.markdown(f"<div class='mlx-insight' style='--ic-color:{color};'>{html_text}</div>", unsafe_allow_html=True)

    def su10(pct):
        return f"{pct/10:.0f} su 10"

    # -------- Numeri chiave --------
    n_giorni = len(y_real)
    acc_rf = (y_pred_rf == y_real).mean() * 100
    prec_rf = precision_score(y_real, y_pred_rf, zero_division=0) * 100
    rec_rf = recall_score(y_real, y_pred_rf, zero_division=0) * 100
    giorni_rischio_pct = y_real.sum() / n_giorni * 100

    df_t = pd.DataFrame({'Giorno': df_base['Giorno'].values, 'p': np.asarray(y_proba_rf) * 100}).sort_values('Giorno').reset_index(drop=True)
    p_ultimo = float(df_t['p'].iloc[-1])
    col_ultimo = colore_rischio(p_ultimo)
    lab_ultimo = "Alto" if p_ultimo >= 60 else "Attenzione" if p_ultimo >= 25 else "Basso"

    st.markdown(f"""
    <div class='mlx-hero'>
        <p class='mlx-eyebrow'>Quanto ci si può fidare</p>
        <h2 class='mlx-hero-title'>Come se la cava il sistema sui tuoi {n_giorni} allenamenti</h2>
        <p class='mlx-hero-msg'>Per essere onesti, ogni giorno è stato previsto "al buio": il sistema non aveva mai visto quell'allenamento quando ha dato il suo giudizio.</p>
        <div class='mlx-stat-grid'>
            <div class='mlx-stat-card' style='--stat-color:{col_ultimo};'>
                <div class='v'>{lab_ultimo}</div>
                <div class='l'>Rischio stimato per la tua ultima uscita ({p_ultimo:.0f}%).</div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_CYAN};'>
                <div class='v'>{su10(acc_rf)}</div>
                <div class='l'>Giorni giudicati correttamente (a rischio o sicuro).</div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_GREEN};'>
                <div class='v'>{su10(prec_rf)}</div>
                <div class='l'>Quando dà l'allarme, quante volte il rischio c'era davvero.</div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_AMBER};'>
                <div class='v'>{su10(rec_rf)}</div>
                <div class='l'>Dei giorni davvero a rischio, quanti ne ha scoperti.</div>
            </div>
            <div class='mlx-stat-card' style='--stat-color:{C_ORANGE};'>
                <div class='v'>{giorni_rischio_pct:.0f}%</div>
                <div class='l'>Quota di allenamenti a rischio nel tuo storico.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "Il tuo andamento", "Cosa ti mette a rischio", "Le tue regole personali", "Il tuo cuore sotto sforzo",
        "I tuoi tipi di allenamento", "Fatica accumulata", "E se domani...?", "Quale sistema scegliere"
    ])

    # =========================================================
    # TAB 1 — ANDAMENTO + CALENDARIO
    # =========================================================
    with t1:
        mlx_section("Il tuo rischio, allenamento dopo allenamento")
        mlx_insight("Ogni punto è una tua uscita. Più è in alto, più il sistema la considerava rischiosa. Le fasce colorate ti dicono subito dove sei: verde sicuro, giallo attenzione, rosso pericolo.")

        fig_t = go.Figure()
        fig_t.add_hrect(y0=0, y1=25, fillcolor=C_GREEN, opacity=0.07, line_width=0)
        fig_t.add_hrect(y0=25, y1=60, fillcolor=C_AMBER, opacity=0.07, line_width=0)
        fig_t.add_hrect(y0=60, y1=100, fillcolor=C_ORANGE, opacity=0.09, line_width=0)
        fig_t.add_trace(go.Scatter(
            x=df_t['Giorno'], y=df_t['p'], mode='lines+markers',
            line=dict(color=C_CYAN, width=2, shape='spline'),
            marker=dict(size=10, color=df_t['p'], colorscale=RISK_SCALE, cmin=0, cmax=100, line=dict(color=BG_DARK, width=1)),
            hovertemplate="%{x}<br>Rischio: %{y:.0f}%<extra></extra>"))
        fig_t.update_layout(height=360, title="Rischio stimato per ogni uscita", yaxis=dict(range=[0, 100], title="Rischio (%)"), showlegend=False)
        st.plotly_chart(style_fig(fig_t), use_container_width=True)

        n_rossi = int((df_t['p'] >= 60).sum())
        n_gialli = int(((df_t['p'] >= 25) & (df_t['p'] < 60)).sum())
        n_verdi = int((df_t['p'] < 25).sum())
        a, b, c = st.columns(3)
        a.markdown(f"<div class='mlx-box' style='--bc:{C_GREEN};'><div class='n'>{n_verdi}</div><div class='t'>uscite in zona verde</div></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='mlx-box' style='--bc:{C_AMBER};'><div class='n'>{n_gialli}</div><div class='t'>uscite in zona di attenzione</div></div>", unsafe_allow_html=True)
        c.markdown(f"<div class='mlx-box' style='--bc:{C_ORANGE};'><div class='n'>{n_rossi}</div><div class='t'>uscite in zona rossa</div></div>", unsafe_allow_html=True)

        # Calendario del rischio (stile "mappa di calore" settimanale)
        date = pd.to_datetime(df_t['Giorno'], errors='coerce')
        if date.notna().sum() >= 5:
            mlx_section("Il calendario del rischio")
            ok = date.notna()
            d0 = date[ok].min().normalize()
            start = d0 - pd.Timedelta(days=d0.weekday())
            cal = pd.DataFrame({
                'sett': ((date[ok] - start).dt.days // 7).values,
                'gs': date[ok].dt.weekday.values,
                'p': df_t.loc[ok, 'p'].values})
            piv = cal.pivot_table(index='gs', columns='sett', values='p', aggfunc='mean')
            piv = piv.reindex(index=range(7), columns=range(int(cal['sett'].max()) + 1))
            etich_sett = [(start + pd.Timedelta(weeks=int(w))).strftime('%d/%m') for w in piv.columns]
            fig_cal = go.Figure(go.Heatmap(
                z=piv.values, x=etich_sett, y=['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom'],
                colorscale=RISK_SCALE, zmin=0, zmax=100, xgap=4, ygap=4, hoverongaps=False,
                colorbar=dict(title="Rischio %"),
                hovertemplate="Settimana del %{x}, %{y}<br>Rischio: %{z:.0f}%<extra></extra>"))
            fig_cal.update_layout(height=300, title="Ogni quadratino è un giorno (vuoto = non hai corso)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig_cal), use_container_width=True)

            per_gs = cal.groupby('gs')['p'].mean()
            nomi_gs = ['lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì', 'sabato', 'domenica']
            peggiore_gs = nomi_gs[int(per_gs.idxmax())]
            mlx_insight(f"<strong>Cosa notare:</strong> cerca le strisce di quadratini caldi (arancioni) ravvicinati: sono i periodi in cui hai accumulato troppo. Il giorno in cui sei in media più a rischio è il <strong>{peggiore_gs}</strong>.", C_AMBER)

    # =========================================================
    # TAB 2 — COSA TI METTE A RISCHIO
    # =========================================================
    with t2:
        mlx_section("Cosa ti mette più a rischio")
        mlx_insight("Immagina una giuria di 100 allenatori esperti: ognuno guarda i tuoi dati e vota se quel giorno era a rischio. Vince la maggioranza. <span class='mlx-note'>(Nome tecnico: Random Forest)</span>")

        c1, c2 = st.columns(2)
        with c1:
            imp = sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
            fig_imp = go.Figure(go.Bar(
                y=[x[0] for x in imp], x=[x[1] * 100 for x in imp], orientation='h',
                marker=dict(color=[x[1] for x in imp], colorscale=[[0, '#00B8D4'], [1, C_CYAN]]),
                text=[f'{x[1]*100:.0f}%' for x in imp], textposition='auto'))
            fig_imp.update_traces(hovertemplate="%{y}: pesa il %{x:.0f}%<extra></extra>")
            fig_imp.update_layout(height=320, yaxis=dict(autorange="reversed"),
                                  title="Quanto conta ogni fattore nel giudizio", xaxis_title="Peso (%)")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            mlx_insight(f"<strong>In pratica:</strong> il fattore che pesa di più è <strong>{imp[0][0]}</strong>. Tienilo d'occhio prima di aumentare i carichi.", C_AMBER)

        with c2:
            cm = confusion_matrix(y_real, y_pred_rf, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            st.markdown("<p style='color:#fff; font-weight:700; font-family:Inter,sans-serif; margin:6px 0 10px 0;'>Com'è andata, giorno per giorno</p>", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.markdown(f"<div class='mlx-box' style='--bc:{C_GREEN};'><div class='n'>{tn}</div><div class='t'>Giorni sicuri,<br>riconosciuti come sicuri</div></div>", unsafe_allow_html=True)
            b.markdown(f"<div class='mlx-box' style='--bc:{C_CYAN};'><div class='n'>{tp}</div><div class='t'>Giorni a rischio,<br>scoperti in tempo</div></div>", unsafe_allow_html=True)
            st.write("")
            c, d = st.columns(2)
            c.markdown(f"<div class='mlx-box' style='--bc:{C_AMBER};'><div class='n'>{fp}</div><div class='t'>Falsi allarmi:<br>segnalato rischio, ma era ok</div></div>", unsafe_allow_html=True)
            d.markdown(f"<div class='mlx-box' style='--bc:{C_ORANGE};'><div class='n'>{fn}</div><div class='t'>Rischi sfuggiti:<br>detto ok, ma c'era rischio</div></div>", unsafe_allow_html=True)
            mlx_insight("<strong>Come leggere:</strong> le due caselle in alto sono le risposte giuste. Gli errori più pericolosi sono i <strong>rischi sfuggiti</strong>; i falsi allarmi sono solo prudenza in eccesso.", C_CYAN)

        mlx_section("Giorni a rischio contro giorni sicuri: cosa cambia?")
        fattori = {'Distanza (km)': 'Km corsi', 'Ore Sonno': 'Ore di sonno', 'Stress Lavoro': 'Stress',
                   'RPE': 'Fatica percepita', 'FC Media': 'Battiti medi'}
        g = df_base.groupby('Rischio Infortunio')[list(fattori)].mean()
        if 0 in g.index and 1 in g.index:
            diff = ((g.loc[1] - g.loc[0]) / g.loc[0].replace(0, np.nan) * 100).dropna()
            diff = diff.reindex(diff.abs().sort_values().index)
            fig_d = go.Figure(go.Bar(
                y=[fattori[k] for k in diff.index], x=diff.values, orientation='h',
                marker_color=[C_ORANGE if v > 0 else C_GREEN for v in diff.values],
                text=[f"{v:+.0f}%" for v in diff.values], textposition='outside'))
            fig_d.add_vline(x=0, line_color="#E8ECF2", line_width=1)
            fig_d.update_traces(hovertemplate="%{y}: %{x:+.0f}% rispetto ai giorni sicuri<extra></extra>")
            fig_d.update_layout(height=320, title="Nei giorni a rischio, rispetto a quelli sicuri, hai avuto...", xaxis_title="Differenza (%)")
            st.plotly_chart(style_fig(fig_d), use_container_width=True)
            top = diff.abs().idxmax()
            verso = "più" if diff[top] > 0 else "meno"
            mlx_insight(f"<strong>La differenza più grande:</strong> nei giorni a rischio hai avuto in media il <strong>{abs(diff[top]):.0f}% {verso}</strong> di <strong>{fattori[top].lower()}</strong> rispetto ai giorni sicuri.", C_AMBER)
        else:
            mlx_insight("Nel periodo selezionato non ci sono abbastanza giorni a rischio per fare questo confronto.", C_AMBER)

    # =========================================================
    # TAB 3 — REGOLE PERSONALI
    # =========================================================
    with t3:
        mlx_section("Le tue regole personali")
        mlx_insight("Non sono regole prese da un manuale: sono ricavate dai <strong>tuoi</strong> allenamenti. Il sistema ha cercato il punto in cui, per te, il rischio supera il 25%. Gli altri fattori sono tenuti sui tuoi valori tipici.", C_GREEN)

        med = df_base[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].median().values.astype(float)

        def griglia(idx, valori):
            X = np.tile(med, (len(valori), 1))
            X[:, idx] = valori
            return rischio_rf(X)

        s_vals = np.arange(4, 10.01, 0.25)
        p_s = griglia(1, s_vals)
        d_vals = np.arange(0, 42.01, 0.5)
        p_d = griglia(0, d_vals)
        st_vals = np.arange(1, 11, 1.0)
        p_st = griglia(2, st_vals)

        sopra_s = np.where(p_s >= 25)[0]
        sopra_d = np.where(p_d >= 25)[0]
        sopra_st = np.where(p_st >= 25)[0]

        r1, r2, r3 = st.columns(3)
        if len(sopra_s) and sopra_s.max() < len(s_vals) - 1:
            lim = s_vals[sopra_s.max()] + 0.25
            r1.markdown(f"<div class='mlx-rule' style='--bc:{C_CYAN};'><div class='k'>Sonno</div><div class='big'>{lim:.1f} h</div><div class='d'>Dormi meno di così e il rischio supera il 25%. Questa è la tua soglia minima.</div></div>", unsafe_allow_html=True)
        else:
            r1.markdown(f"<div class='mlx-rule' style='--bc:{C_GREEN};'><div class='k'>Sonno</div><div class='big'>nessun limite</div><div class='d'>Nel tuo storico il sonno da solo non porta il rischio sopra il 25%.</div></div>", unsafe_allow_html=True)
        if len(sopra_d):
            lim = d_vals[sopra_d.min()]
            r2.markdown(f"<div class='mlx-rule' style='--bc:{C_AMBER};'><div class='k'>Distanza</div><div class='big'>{lim:.0f} km</div><div class='d'>Oltre questa distanza il rischio supera il 25%. Questo è il tuo tetto con i valori tipici.</div></div>", unsafe_allow_html=True)
        else:
            r2.markdown(f"<div class='mlx-rule' style='--bc:{C_GREEN};'><div class='k'>Distanza</div><div class='big'>nessun limite</div><div class='d'>Nel tuo storico i km da soli non portano il rischio sopra il 25%.</div></div>", unsafe_allow_html=True)
        if len(sopra_st):
            lim = st_vals[sopra_st.min()]
            r3.markdown(f"<div class='mlx-rule' style='--bc:{C_ORANGE};'><div class='k'>Stress</div><div class='big'>{lim:.0f} / 10</div><div class='d'>Da questo livello di stress in su il rischio supera il 25%. Attenzione ai periodi pesanti al lavoro.</div></div>", unsafe_allow_html=True)
        else:
            r3.markdown(f"<div class='mlx-rule' style='--bc:{C_GREEN};'><div class='k'>Stress</div><div class='big'>nessun limite</div><div class='d'>Nel tuo storico lo stress da solo non porta il rischio sopra il 25%.</div></div>", unsafe_allow_html=True)

        mlx_section("Quanto pesa ogni fattore")
        mlx_insight("Qui il sistema è una semplice bilancia: ogni fattore o <strong>aumenta</strong> o <strong>riduce</strong> il rischio, e il grafico mostra di quanto. <span class='mlx-note'>(Nome tecnico: Regressione Logistica)</span>", C_AMBER)

        coefs = log_model.coef_[0]
        order = np.argsort(np.abs(coefs))
        nomi = [feature_names[i] for i in order]
        vals = [coefs[i] for i in order]
        fig_log = go.Figure(go.Bar(y=nomi, x=vals, orientation='h', marker_color=[C_ORANGE if v > 0 else C_GREEN for v in vals]))
        fig_log.update_traces(hovertemplate="%{y}<extra></extra>")
        fig_log.add_vline(x=0, line_color="#E8ECF2", line_width=1)
        fig_log.update_layout(height=320, title="Verso destra (arancione) aumenta il rischio, verso sinistra (verde) lo protegge",
                              xaxis=dict(showticklabels=False, title="Più la barra è lunga, più il fattore conta"))
        st.plotly_chart(style_fig(fig_log), use_container_width=True)

        peggiore = feature_names[int(np.argmax(coefs))]
        migliore = feature_names[int(np.argmin(coefs))]
        txt = f"Il fattore che <strong>spinge di più il rischio verso l'alto</strong> è <strong>{peggiore}</strong>."
        if coefs.min() < 0:
            txt += f" Quello che <strong>ti protegge di più</strong> è <strong>{migliore}</strong>."
        mlx_insight(txt, C_AMBER)

    # =========================================================
    # TAB 4 — CUORE SOTTO SFORZO
    # =========================================================
    with t4:
        mlx_section("Il tuo cuore: previsto vs reale")
        mlx_insight("Il sistema calcola quanti battiti al minuto dovresti avere in una corsa con quel passo, quel clima e quella distanza. Se il cuore va <strong>molto più veloce del previsto</strong>, può essere un segnale di stanchezza nascosta. <span class='mlx-note'>(Nome tecnico: Regressione Lineare)</span>")

        reg_bundle = get_bundle_regressione(df_base)
        df_base['FC_Predetta'] = reg_bundle["fc_predetta"]
        df_base['Residuo'] = reg_bundle["residui"]
        mae = reg_bundle["mae"]
        giorni_alti = int((df_base['Residuo'] > 2 * mae).sum())

        c1, c2 = st.columns(2)
        with c1:
            fig_lr = px.scatter(df_base, x='FC Media', y='FC_Predetta', color='RPE',
                                color_continuous_scale=[[0, C_CYAN], [1, C_ORANGE]],
                                labels={'FC_Predetta': 'Battiti previsti', 'FC Media': 'Battiti reali', 'RPE': 'Fatica percepita'})
            fig_lr.update_traces(hovertemplate="Reali: %{x} bpm<br>Previsti: %{y:.0f} bpm<extra></extra>")
            fig_lr.add_shape(type="line", x0=df_base['FC Media'].min(), y0=df_base['FC Media'].min(),
                             x1=df_base['FC Media'].max(), y1=df_base['FC Media'].max(), line=dict(color=C_GREEN, dash="dash"))
            fig_lr.update_layout(height=320, title="Battiti reali contro battiti previsti")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)
            mlx_insight("<strong>Come leggere:</strong> i punti sulla linea verde sono corse in cui il cuore ha fatto esattamente ciò che ci si aspettava. Più un punto è lontano, più la corsa è stata 'strana'.")
        with c2:
            fig_resid = px.scatter(df_base, x='Giorno', y='Residuo', color='Residuo',
                                   color_continuous_scale=[[0, C_GREEN], [0.5, TXT_TER], [1, C_ORANGE]],
                                   labels={'Residuo': 'Battiti in più/meno'})
            fig_resid.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            fig_resid.update_traces(hovertemplate="%{x}<br>%{y:+.1f} battiti rispetto al previsto<extra></extra>")
            fig_resid.update_layout(height=320, title="Giorno per giorno: cuore più o meno stanco del previsto", coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_resid), use_container_width=True)
            mlx_insight(f"Il sistema sbaglia in media di circa <strong>{mae:.0f} battiti al minuto</strong>. Punti sopra lo zero (arancioni) = cuore più affaticato del previsto. "
                        f"Giorni con scarto molto alto: <strong>{giorni_alti}</strong>.", C_AMBER)

    # =========================================================
    # TAB 5 — TIPI DI ALLENAMENTO
    # =========================================================
    with t5:
        mlx_section("I tuoi tipi di allenamento")
        mlx_insight("Il sistema ha raggruppato da solo le tue uscite in famiglie simili. Serve a capire se alterni davvero giorni facili e giorni duri, o se corri sempre alla stessa intensità 'media'. <span class='mlx-note'>(Nome tecnico: Clustering K-Means)</span>")

        cluster_bundle = get_bundle_cluster(df_base)
        df_base['Cluster_ID'] = cluster_bundle["cluster_id"]

        prof = df_base.groupby('Cluster_ID').agg(
            n=('Distanza (km)', 'size'), km=('Distanza (km)', 'mean'), fc=('FC Media', 'mean'),
            rpe=('RPE', 'mean'), sonno=('Ore Sonno', 'mean'), rischio=('Rischio Infortunio', 'mean')
        ).sort_values('km')
        nomi_base = ["Uscite brevi e leggere", "Uscite medie", "Uscite lunghe e impegnative"]
        nomi_cl = {cid: (nomi_base[i] if len(prof) == 3 else f"Gruppo {i+1}") for i, cid in enumerate(prof.index)}
        df_base['Tipo'] = df_base['Cluster_ID'].map(nomi_cl)

        fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Tipo', size='RPE',
                            color_discrete_sequence=[C_GREEN, C_AMBER, C_ORANGE],
                            category_orders={'Tipo': [nomi_cl[c] for c in prof.index]},
                            labels={'FC Media': 'Battiti medi', 'Tipo': ''})
        fig_km.update_traces(hovertemplate="%{x} km · %{y} bpm<extra></extra>")
        fig_km.update_layout(height=340, title="Le tue uscite, divise per famiglia (pallino grande = fatica alta)")
        st.plotly_chart(style_fig(fig_km), use_container_width=True)

        cols = st.columns(len(prof))
        for col, (cid, r) in zip(cols, prof.iterrows()):
            colore = C_GREEN if r['rischio'] < 0.25 else C_AMBER if r['rischio'] < 0.5 else C_ORANGE
            col.markdown(f"""
            <div class='mlx-box' style='--bc:{colore}; text-align:left;'>
                <div style='color:#fff; font-weight:700; font-family:Inter,sans-serif; margin-bottom:8px;'>{nomi_cl[cid]}</div>
                <div class='t'>{int(r['n'])} uscite · in media <strong style='color:#fff;'>{r['km']:.1f} km</strong><br>
                Cuore: {r['fc']:.0f} bpm · Fatica: {r['rpe']:.1f}/10<br>
                Sonno: {r['sonno']:.1f} h<br>
                <strong style='color:{colore};'>Giorni a rischio: {r['rischio']*100:.0f}%</strong></div>
            </div>""", unsafe_allow_html=True)
        mlx_insight("<strong>Come usarlo:</strong> guarda la percentuale di rischio di ogni famiglia. Se le uscite 'medie' hanno quasi sempre fatica alta, stai correndo troppo forte nei giorni che dovrebbero essere facili.")

    # =========================================================
    # TAB 6 — FATICA ACCUMULATA
    # =========================================================
    with t6:
        mlx_section("Fatica accumulata nel tempo")
        mlx_insight("Non conta solo il singolo allenamento: la stanchezza si somma giorno dopo giorno. Questo grafico mostra la fatica media degli ultimi 7 giorni, per accorgerti in anticipo di un periodo troppo carico.", C_ORANGE)

        df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').copy()
        df_stress['Media7'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
        sopra = int((df_stress['Media7'] > 15).sum())

        fig_sp = px.area(df_stress, x='Giorno', y='Media7', color_discrete_sequence=[C_ORANGE], labels={'Media7': 'Fatica accumulata'})
        fig_sp.update_traces(hovertemplate="%{x}<br>Fatica accumulata: %{y:.1f}<extra></extra>")
        fig_sp.add_hline(y=15, line_dash="dash", line_color=C_AMBER, annotation_text="Sopra questa linea: troppo carico")
        fig_sp.update_layout(height=340, title="Fatica accumulata (media degli ultimi 7 giorni)")
        st.plotly_chart(style_fig(fig_sp), use_container_width=True)
        mlx_insight(f"<strong>In numeri:</strong> sei stato sopra la linea per <strong>{sopra} giorni su {len(df_stress)}</strong> ({sopra/len(df_stress)*100:.0f}% del periodo). Più a lungo resti lassù, più serve inserire giorni di scarico.", C_AMBER)

    # =========================================================
    # TAB 7 — E SE DOMANI...?
    # =========================================================
    with t7:
        mlx_section("E se domani...? Prova uno scenario")
        mlx_insight("Muovi i cursori e guarda subito quanto sarebbe rischioso correre in quelle condizioni.")

        base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {
            'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6}

        s1, s2 = st.columns(2)
        with s1:
            sim_dist = st.slider("Quanti km vuoi correre?", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
            sim_sonno = st.slider("Quante ore hai dormito?", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
        with s2:
            sim_stress = st.slider("Stress da lavoro/vita (1 = zero, 10 = altissimo)", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
            sim_rpe = st.slider("Quanto ti sembrerà dura la corsa? (1 = facile, 10 = massimo)", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")

        stima = stima_rischio_oggi(class_bundle, sim_dist, sim_sonno, sim_stress, sim_rpe)
        sim_fc = stima["fc_media_stimata"]
        p = stima["probabilita_rf"]
        p_log = stima["probabilita_log"]
        col = colore_rischio(p)

        if p >= 60:
            msg = f"{dot(C_ORANGE)}<strong>RISCHIO ALTO ({p:.0f}%)</strong> — Con questi valori {sim_dist:.0f} km sono troppi. Meglio scendere a circa <strong>{sim_dist*0.4:.0f} km</strong> o riposare."
        elif p >= 25:
            msg = f"{dot(C_AMBER)}<strong>ATTENZIONE ({p:.0f}%)</strong> — C'è un po' di sovraccarico. Prova a ridurre da {sim_dist:.0f} a circa <strong>{sim_dist*0.7:.0f} km</strong>."
        else:
            msg = f"{dot(C_GREEN)}<strong>RISCHIO BASSO ({p:.0f}%)</strong> — Le condizioni reggono bene {sim_dist:.0f} km. Puoi procedere."
        st.markdown(f"<div class='info-box' style='border-left-color:{col};'>{msg}</div>", unsafe_allow_html=True)
        st.caption(f"Un secondo sistema, indipendente, stima {p_log:.0f}% per lo stesso scenario: se i due numeri sono vicini, il giudizio è più solido. "
                   f"Per i battiti si usa una stima basata sulla fatica che hai impostato (circa {sim_fc:.0f} bpm).")

        # Piano migliore: massimo di km restando in zona verde
        d_fine = np.arange(0, 42.01, 0.5)
        Xd = np.column_stack([d_fine, np.full_like(d_fine, sim_sonno), np.full_like(d_fine, sim_stress),
                              np.full_like(d_fine, sim_fc), np.full_like(d_fine, sim_rpe)])
        p_fine = rischio_rf(Xd)
        primo_sopra = np.where(p_fine >= 25)[0]
        if len(primo_sopra) == 0:
            km_sicuri, frase = 42.0, "Con queste condizioni puoi coprire qualsiasi distanza restando in zona verde."
        elif primo_sopra[0] == 0:
            km_sicuri, frase = 0.0, "Con queste condizioni, anche una corsa brevissima è già in zona di attenzione. Oggi meglio riposo o recupero attivo."
        else:
            km_sicuri = float(d_fine[primo_sopra[0] - 1])
            frase = f"Oltre questa distanza il rischio supera il 25%. Tu hai scelto {sim_dist:.0f} km."
        st.markdown(f"""
        <div class='mlx-bigsafe'>
            <div class='mlx-eyebrow'>Il tuo piano migliore per oggi</div>
            <div class='big'>{km_sicuri:.1f} km</div>
            <div style='font-family:Inter,sans-serif; color:{TXT_SEC}; font-size:.95rem; margin-top:4px;'>{frase}</div>
        </div>
        """, unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=p,
                title={'text': "Rischio di infortunio", 'font': {'color': TXT_TER}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': col}, 'bgcolor': "#111827", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': 'rgba(0,245,160,0.12)'},
                                 {'range': [25, 60], 'color': 'rgba(255,176,32,0.12)'},
                                 {'range': [60, 100], 'color': 'rgba(255,106,61,0.14)'}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}))
            fig_g.update_layout(height=300)
            st.plotly_chart(style_fig(fig_g), use_container_width=True)
        with g2:
            sr = np.linspace(4, 10, 25)
            Xs = np.column_stack([np.full_like(sr, sim_dist), sr, np.full_like(sr, sim_stress), np.full_like(sr, sim_fc), np.full_like(sr, sim_rpe)])
            fig_s = px.line(x=sr, y=rischio_rf(Xs), labels={'x': 'Ore di sonno', 'y': 'Rischio %'}, title="Se dormi di più (o di meno)")
            fig_s.update_traces(line_color=C_CYAN, line_width=3, hovertemplate="%{x:.1f} ore: rischio %{y:.0f}%<extra></extra>")
            fig_s.add_vline(x=sim_sonno, line_dash="dash", line_color=C_ORANGE)
            fig_s.update_layout(height=300)
            st.plotly_chart(style_fig(fig_s), use_container_width=True)

        mlx_section("La mappa del rischio: sonno contro km")
        mlx_insight("Ogni punto della mappa è una combinazione di ore di sonno (in orizzontale) e km (in verticale). Il rombo bianco è lo scenario che hai scelto: <strong>cerca di restare nelle zone verdi</strong>.", C_CYAN)
        s_ax = np.linspace(4, 10, 25)
        d_ax = np.linspace(0, 42, 25)
        S, D = np.meshgrid(s_ax, d_ax)
        Xm = np.column_stack([D.ravel(), S.ravel(), np.full(S.size, sim_stress, dtype=float),
                              np.full(S.size, sim_fc, dtype=float), np.full(S.size, sim_rpe, dtype=float)])
        Z = rischio_rf(Xm).reshape(S.shape)
        fig_map = go.Figure(go.Heatmap(
            x=s_ax, y=d_ax, z=Z, colorscale=RISK_SCALE, zmin=0, zmax=100, colorbar=dict(title="Rischio %"),
            hovertemplate="Sonno: %{x:.1f} h<br>Km: %{y:.0f}<br>Rischio: %{z:.0f}%<extra></extra>"))
        fig_map.add_trace(go.Scatter(
            x=[sim_sonno], y=[sim_dist], mode='markers', name="Il tuo scenario",
            marker=dict(symbol='diamond', size=16, color='#FFFFFF', line=dict(color=BG_DARK, width=2)),
            hovertemplate="Il tuo scenario<extra></extra>"))
        fig_map.update_layout(height=420, title="Mappa del rischio (stress e fatica come da cursori)",
                              xaxis_title="Ore di sonno", yaxis_title="Km corsi", showlegend=False)
        st.plotly_chart(style_fig(fig_map), use_container_width=True)

    # =========================================================
    # TAB 8 — QUALE SISTEMA SCEGLIERE
    # =========================================================
    with t8:
        mlx_section("Due sistemi a confronto")
        mlx_insight("Abbiamo due 'consulenti' che guardano gli stessi dati: la <strong>giuria di esperti</strong> (Random Forest), più precisa, e la <strong>bilancia dei fattori</strong> (Regressione Logistica), più facile da spiegare. Ecco come si sono comportati sui tuoi allenamenti.", C_AMBER)

        acc_log = (y_pred_log == y_real).mean() * 100
        prec_log = precision_score(y_real, y_pred_log, zero_division=0) * 100
        rec_log = recall_score(y_real, y_pred_log, zero_division=0) * 100

        etichette = ["Giorni giudicati bene", "Allarmi giusti", "Rischi scoperti"]
        v_rf = [acc_rf, prec_rf, rec_rf]
        v_log = [acc_log, prec_log, rec_log]
        s_rf, s_log = float(np.mean(v_rf)), float(np.mean(v_log))
        vince = "La giuria di esperti" if s_rf >= s_log else "La bilancia dei fattori"
        colv = C_CYAN if s_rf >= s_log else C_AMBER
        margine = abs(s_rf - s_log)
        nota = "di poco: i due si equivalgono quasi" if margine < 5 else "in modo netto"

        st.markdown(f"""
        <div class='mlx-hero'>
            <p class='mlx-eyebrow'>Risultato</p>
            <h2 class='mlx-hero-title'>{vince} se la cava meglio ({nota})</h2>
            <p class='mlx-hero-msg' style='margin:0;'>Voto medio: <strong style='color:{colv};'>{max(s_rf, s_log)/10:.1f}/10</strong>
            contro {min(s_rf, s_log)/10:.1f}/10 dell'altro.</p>
        </div>
        """, unsafe_allow_html=True)

        comp = pd.DataFrame({
            'Cosa misuriamo': etichette * 2,
            'Voto (su 10)': [v / 10 for v in v_rf + v_log],
            'Sistema': ['Giuria di esperti'] * 3 + ['Bilancia dei fattori'] * 3})
        fig_c = px.bar(comp, x='Cosa misuriamo', y='Voto (su 10)', color='Sistema', barmode='group',
                       color_discrete_map={'Giuria di esperti': C_CYAN, 'Bilancia dei fattori': C_AMBER})
        fig_c.update_traces(hovertemplate="%{x}: %{y:.1f}/10<extra></extra>")
        fig_c.update_layout(height=340, title="Voti dei due sistemi (più alto = meglio)", legend=dict(orientation="h", y=-0.25),
                            yaxis=dict(range=[0, 10]))
        st.plotly_chart(style_fig(fig_c), use_container_width=True)

        k1, k2 = st.columns(2)
        k1.markdown(f"""
        <div class='mlx-box' style='--bc:{C_CYAN}; text-align:left;'>
            <div style='color:{C_CYAN}; font-weight:700; font-size:1.1rem; margin-bottom:8px;'>Giuria di esperti</div>
            <div class='t'><strong style='color:#fff;'>Pro:</strong> coglie le combinazioni (es. poco sonno <em>e</em> molto stress insieme).<br>
            <strong style='color:#fff;'>Contro:</strong> difficile dire perché ha deciso così in un singolo giorno.<br>
            <strong style='color:#fff;'>Usala per:</strong> l'allarme più affidabile.</div>
        </div>""", unsafe_allow_html=True)
        k2.markdown(f"""
        <div class='mlx-box' style='--bc:{C_AMBER}; text-align:left;'>
            <div style='color:{C_AMBER}; font-weight:700; font-size:1.1rem; margin-bottom:8px;'>Bilancia dei fattori</div>
            <div class='t'><strong style='color:#fff;'>Pro:</strong> dice chiaramente quanto pesa ogni fattore.<br>
            <strong style='color:#fff;'>Contro:</strong> più semplice, può perdersi le combinazioni.<br>
            <strong style='color:#fff;'>Usala per:</strong> capire <em>perché</em> il rischio sale.</div>
        </div>""", unsafe_allow_html=True)

        mlx_insight("<strong>Il consiglio pratico:</strong> non sceglierne uno. La giuria ti avvisa, la bilancia ti spiega cosa cambiare (sonno, stress o km). Entrambi, insieme ai calcoli sul cuore e sui tipi di allenamento, alimentano il verdetto del \"Consiglio Finale\".", C_GREEN)

except Exception as e:
    st.error(f"Errore caricamento modelli ML: {str(e)}")
