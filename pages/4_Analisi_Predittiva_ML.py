import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, get_svg_url, style_fig, SVG_ML
from utils.sidebar import sidebar_comune
from utils.ml_engine import (
    get_bundle_classificazione,
    get_bundle_regressione,
    get_bundle_cluster,
    stima_rischio_oggi,
    FEATURE_LABELS_CLASS,
)

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="RUN AI - Modelli personali", layout="wide")
carica_css()

# =========================
# INIT SESSION STATE
# =========================
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

# =========================
# DATI / FILTRI
# =========================
sidebar_result = sidebar_comune()
if sidebar_result and isinstance(sidebar_result, tuple) and len(sidebar_result) == 3:
    df, df_full, filtro_tempo = sidebar_result
else:
    df_full = st.session_state.dati.copy()
    df = df_full
    filtro_tempo = "Ultimi 30 giorni"

IMG_HERO_ML = get_svg_url(SVG_ML)

df_base = st.session_state.dati.copy()

# =========================
# TEMA
# =========================
BG = "#0B1017"
BG2 = "#0E1420"
BD = "#202B3D"
TXT = "#FFFFFF"
TXT_SEC = "#B8C2D0"
TXT_MUTED = "#8792A3"
CYAN = "#00E5FF"
GREEN = "#00F5A0"
AMBER = "#FFB020"
ORANGE = "#FF6A3D"
RED = "#ff5b5b"

# =========================
# HEADER
# =========================
header_block(
    "Modulo 04 — I tuoi modelli",
    "COME RUN AI TI PROTEGGE",
    "I modelli imparano dal tuo storico per capire quando la fatica è troppo alta e quando la distanza è troppo forte per te.",
    IMG_HERO_ML,
    "Machine Learning Engine"
)

st.markdown(
    """
    <div style='background: linear-gradient(135deg, #0E1420 0%, #101A2E 100%);
                border: 1px solid #202B3D; border-radius: 16px; padding: 18px 22px; margin: 18px 0 24px 0;'>
        <h3 style='margin:0 0 10px 0; color:#00E5FF;'>Come lavora il sistema</h3>
        <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
            RUN AI usa il tuo storico di allenamenti per capire i tuoi pattern. Guarda <strong>distanza, sonno, stress, frequenza cardiaca e sforzo percepito</strong>,
            e poi stima il rischio di infortunio. In pratica: insegna a un coach virtuale a riconoscere i segnali del tuo corpo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    # =========================
    # LOAD MODELLI
    # =========================
    feature_names = FEATURE_LABELS_CLASS

    class_bundle = get_bundle_classificazione(df_base)
    rf_model = class_bundle["rf_model"]
    log_model = class_bundle["log_model"]
    scaler = class_bundle["scaler"]

    y_train_class = df_base['Rischio Infortunio'].values
    y_pred_rf = class_bundle["y_pred_rf"]
    y_proba_rf = class_bundle["y_proba_rf"]
    y_pred_log = class_bundle["y_pred_log"]
    y_proba_log = class_bundle["y_proba_log"]

    # =========================
    # KPI INIZIALI
    # =========================
    acc_rf = (y_pred_rf == y_train_class).mean() * 100
    prec_rf = precision_score(y_train_class, y_pred_rf, zero_division=0) * 100
    rec_rf = recall_score(y_train_class, y_pred_rf, zero_division=0) * 100
    giorni_rischio_pct = (df_base['Rischio Infortunio'].sum() / len(df_base)) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuratezza", f"{acc_rf:.0f}%", "quanto il modello indovina in media")
    with col2:
        st.metric("Sensibilità", f"{rec_rf:.0f}%", "stima i giorni davvero a rischio")
    with col3:
        st.metric("Precisione", f"{prec_rf:.0f}%", "quando dice rischio, ha ragione")
    with col4:
        st.metric("Giorni a rischio", f"{giorni_rischio_pct:.0f}%", "percentuale del tuo storico")

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Rischio infortunio",
        "FC e fatica",
        "Tipi di allenamento",
        "Carico cronico",
        "Simulatore What-If",
        "Confronto modelli"
    ])

    # =============================================================
    # TAB 1 - RANDOM FOREST: rischio infortunio
    # =============================================================
    with tab1:
        st.subheader("Come RUN AI capisce se oggi è rischioso correre")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #00E5FF; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Il modello Random Forest è come una giuria di tanti allenatori esperti. Accumula tanti piccoli giudizi
                    su distanza, sonno, stress e sforzo, e poi decide insieme: <strong>rischio alto o basso?</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Quali segnali contano di più?")
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)

            fig_imp = go.Figure(
                go.Bar(
                    y=[x[0] for x in imp_data],
                    x=[x[1] * 100 for x in imp_data],
                    orientation='h',
                    marker=dict(
                        color=[CYAN if x[1] > 0.18 else AMBER if x[1] > 0.12 else GREEN for x in imp_data],
                        line=dict(color='white', width=1)
                    ),
                    text=[f'{x[1]*100:.0f}%' for x in imp_data],
                    textposition='auto',
                )
            )
            fig_imp.update_layout(
                height=310,
                yaxis=dict(autorange='reversed'),
                xaxis_title='Peso nel modello',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC),
                title='Fattori che pesano di più'
            )
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)

            top_feat = imp_data[0][0]
            st.info(f"Il fattore più importante per te è: **{top_feat}**. Quando sale, il rischio cresce molto rapidamente.")

        with col_b:
            st.markdown("#### Quante volte ha azzeccato le previsioni?")
            cm = confusion_matrix(y_train_class, y_pred_rf)
            tn, fp, fn, tp = cm.ravel()

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=['Previsto: sicuro', 'Previsto: rischio'],
                    y=['Reale: sicuro', 'Reale: rischio'],
                    text=cm,
                    texttemplate='%{text}',
                    textfont={'size': 18, 'color': 'white'},
                    colorscale=[[0, '#0E1420'], [1, '#00E5FF']],
                )
            )
            fig_cm.update_layout(
                height=310,
                title='Matrice di confusioni',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)

            st.markdown(
                f"""
                **Come leggere il grafico:**
                - {tn} giorni: sicuro → sicuro
                - {fp} giorni: rischio previsto ma in realtà sicuro
                - {fn} giorni: sicuro previsto ma in realtà rischio
                - {tp} giorni: rischio → rischio
                """
            )

        st.markdown('---')
        st.markdown("#### Curva ROC: quanto distingue i giorni sicuri da quelli a rischio?")

        fpr, tpr, _ = roc_curve(y_train_class, y_proba_rf)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                line=dict(color=CYAN, width=3),
                fill='tozeroy',
                fillcolor='rgba(0,229,255,0.18)',
                name=f'Random Forest (AUC={roc_auc:.2f})',
                hovertemplate='Falsi allarmi: %{x:.2f}<br>Giorni corretti: %{y:.2f}<extra></extra>'
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color=TXT_MUTED, dash='dash'),
                name='Caso casuale'
            )
        )
        fig_roc.update_layout(
            height=350,
            xaxis_title='Probabilità di falso allarme',
            yaxis_title='Probabilità di rilevare il rischio',
            title='Curva ROC',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TXT_SEC)
        )
        st.plotly_chart(style_fig(fig_roc), use_container_width=True)

        if roc_auc >= 0.85:
            verdict = 'Molto buono — distingue bene i periodi critici.'
        elif roc_auc >= 0.75:
            verdict = 'Buono — utile e abbastanza affidabile.'
        else:
            verdict = 'Discreto — utile, ma non è un giudizio assoluto.'

        st.success(f"**AUC = {roc_auc:.3f}** → {verdict}")

    # =============================================================
    # TAB 2 - REGRESSIONE LINEARE: FC media
    # =============================================================
    with tab2:
        st.subheader("La frequenza cardiaca: come il modello capisce se stai lavorando troppo")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #FFB020; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Questo modulo cerca di capire quale dovrebbe essere la tua FC media in base a distanza, sforzo e condizione del giorno.
                    Se la tua FC reale è molto più alta del previsto, c'è un segnale di stress o affaticamento extra.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_bundle = get_bundle_regressione(df_base)
        df_base['FC_Predetta'] = reg_bundle['fc_predetta']
        df_base['Residuo'] = reg_bundle['residui']
        r2 = reg_bundle['r2']
        mae = reg_bundle['mae']

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### FC reale vs FC attesa")
            fig_lr = px.scatter(
                df_base,
                x='FC Media',
                y='FC_Predetta',
                color='RPE',
                color_continuous_scale=[[0, CYAN], [1, ORANGE]],
                labels={
                    'FC Media': 'FC reale (bpm)',
                    'FC_Predetta': 'FC stimata (bpm)',
                    'RPE': 'Sforzo percepito'
                },
                title='FC reale vs FC prevista'
            )
            min_fc = df_base['FC Media'].min()
            max_fc = df_base['FC Media'].max()
            fig_lr.add_shape(
                type='line',
                x0=min_fc, y0=min_fc,
                x1=max_fc, y1=max_fc,
                line=dict(color=GREEN, dash='dash', width=2),
            )
            fig_lr.update_layout(
                height=330,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)

            st.info(f"**R² = {r2*100:.0f}%** — il modello spiega circa il {r2*100:.0f}% della variazione della tua FC. Se un punto è lontano dalla linea verde, vuol dire che il tuo corpo sta lavorando oltre il solito.")

        with col_b:
            st.markdown("#### Scostamenti: quando la FC è fuori norma")
            fig_resid = px.scatter(
                df_base,
                x='Giorno',
                y='Residuo',
                color='Residuo',
                color_continuous_scale=[[0, GREEN], [0.5, '#8792A3'], [1, ORANGE]],
                labels={'Residuo': 'Scostamento (bpm)'}
            )
            fig_resid.add_hline(y=0, line_dash='dash', line_color='white')
            fig_resid.update_layout(
                height=330,
                title='Andamento degli scostamenti',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_resid), use_container_width=True)

            giorni_fuori = int((df_base['Residuo'] > 5).sum())
            st.warning(f"**Giorni fuori norma:** {giorni_fuori}.\n\nErrore medio del modello: ±{mae:.1f} bpm. Questi sono i giorni in cui il tuo corpo sta dicendo che è vicino al limite.")

    # =============================================================
    # TAB 3 - CLUSTERING: tipi di allenamento
    # =============================================================
    with tab3:
        st.subheader("I tuoi allenamenti si dividono in gruppi naturali")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #00F5A0; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Il clustering trova gruppi di allenamenti simili senza che tu glieli dica in anticipo:
                    giorni leggeri, misurati e intensi. Serve a capire se stai sempre facendo lo stesso tipo di lavoro.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cluster_bundle = get_bundle_cluster(df_base)
        X_clust_raw = df_base[['Distanza (km)', 'FC Media']]
        X_clust = cluster_bundle['scaler'].transform(X_clust_raw)
        df_base['Cluster_ID'] = cluster_bundle['cluster_id']
        df_base['Cluster_Name'] = df_base['Cluster_ID'].apply(
            lambda x: ['Giorni leggeri', 'Giorni medi', 'Giorni intensi'][x]
        )
        sil = cluster_bundle['silhouette']

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Perché 3 gruppi?")
            inertias = []
            for k in range(2, 7):
                km_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_clust)
                inertias.append(km_test.inertia_)

            fig_elbow = go.Figure()
            fig_elbow.add_trace(
                go.Scatter(
                    x=list(range(2, 7)),
                    y=inertias,
                    mode='lines+markers',
                    line=dict(color=CYAN, width=3),
                    marker=dict(size=9),
                )
            )
            fig_elbow.add_vline(x=3, line_dash='dash', line_color=ORANGE, annotation_text='Scelta ottima')
            fig_elbow.update_layout(
                height=310,
                title='Metodo del gomito',
                xaxis_title='Numero di gruppi',
                yaxis_title='Compattezza del gruppo',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_elbow), use_container_width=True)

        with col_b:
            st.markdown("#### I tuoi allenamenti nel grafico")
            fig_cluster = px.scatter(
                df_base,
                x='Distanza (km)',
                y='FC Media',
                color='Cluster_Name',
                color_discrete_map={
                    'Giorni leggeri': GREEN,
                    'Giorni medi': AMBER,
                    'Giorni intensi': ORANGE,
                },
                size='RPE',
                title='Segmentazione degli allenamenti'
            )
            fig_cluster.update_layout(
                height=310,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_cluster), use_container_width=True)

        st.markdown('---')
        st.markdown("#### Profilo medio di ogni gruppo")
        cluster_profile = df_base.groupby('Cluster_Name')[[
            'Distanza (km)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro'
        ]].mean().reset_index()

        fig_profile = go.Figure()
        for _, row in cluster_profile.iterrows():
            color = {'Giorni leggeri': GREEN, 'Giorni medi': AMBER, 'Giorni intensi': ORANGE}[row['Cluster_Name']]
            fig_profile.add_trace(go.Scatterpolar(
                r=[row['Distanza (km)'] / 5, row['FC Media'] / 30, row['RPE'], row['Ore Sonno'], row['Stress Lavoro'] / 2],
                theta=['Distanza', 'FC', 'RPE', 'Sonno', 'Stress'],
                fill='toself',
                name=row['Cluster_Name'],
                line=dict(color=color, width=2),
                fillcolor=color + '33'
            ))

        fig_profile.update_layout(
            height=380,
            title='Radar del profilo di allenamento',
            polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TXT_SEC)
        )
        st.plotly_chart(style_fig(fig_profile), use_container_width=True)

        if sil > 0.6:
            st.success(f"**Silhouette score: {sil:.2f}** — i gruppi sono ben separati e molto leggibili.")
        else:
            st.info(f"**Silhouette score: {sil:.2f}** — i gruppi esistono ma ci sono ancora sovrapposizioni tra tipi di allenamento.")

    # =============================================================
    # TAB 4 - STRESS CRONICO / FATICA nell'ultimo periodo
    # =============================================================
    with tab4:
        st.subheader("Il carico cronico: stai accumulando fatica nel tempo?")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #FF6A3D; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Non conta solo il singolo allenamento. Il tuo corpo accumula fatica settimana dopo settimana.
                    Se il carico medio resta troppo alto, aumentano i rischi di infortunio anche se oggi sembri "ok".
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_stress = df_base[['Giorno', 'SMA']].sort_values('Giorno').copy()
        df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Media mobile del carico negli ultimi 7 giorni")
            fig_stress = px.area(
                df_stress,
                x='Giorno',
                y='SMA_Rolling',
                color_discrete_sequence=[ORANGE],
                labels={'SMA_Rolling': 'Carico settimanale medio'}
            )
            fig_stress.add_hline(y=15, line_dash='dash', line_color=AMBER, annotation_text='Soglia critica')
            fig_stress.update_layout(
                height=330,
                title='Carico cronico',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_stress), use_container_width=True)

        with col_b:
            st.markdown("#### Distribuzione del carico giornaliero")
            fig_hist = px.histogram(
                df_stress,
                x='SMA',
                nbins=15,
                color_discrete_sequence=[CYAN],
                labels={'SMA': 'Carico giornaliero'}
            )
            fig_hist.add_vline(x=15, line_dash='dash', line_color=ORANGE, annotation_text='Soglia critica')
            fig_hist.update_layout(
                height=330,
                title='Istogramma del carico',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_hist), use_container_width=True)

        giorni_sopra = int((df_stress['SMA_Rolling'] > 15).sum())
        pct_sopra = (giorni_sopra / len(df_stress)) * 100
        st.warning(f"**Hai superato la soglia critica per {giorni_sopra} giorni** ({pct_sopra:.0f}% del periodo). Se questo numero è alto, il tuo corpo sta accumulando troppa fatica.")

    # =============================================================
    # TAB 5 - SIMULATORE WHAT-IF
    # =============================================================
    with tab5:
        st.subheader("Simulatore fast: cosa succederebbe se facessi oggi questa seduta?")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #00E5FF; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Muovi gli slider per provare uno scenario. Il sistema ti dice in tempo reale quanto il tuo rischio cambierebbe
                    con distanza, sonno, stress e sforzo diversi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {
            'distanza_oggi': 10.0,
            'ore_sonno': 7.5,
            'stress_lavoro': 5,
            'rpe_previsto': 6
        }

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sim_dist = st.slider('Distanza (km)', 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), step=0.5, key='sim_dist')
        with col2:
            sim_sonno = st.slider('Ore di sonno', 2.0, 12.0, float(base.get('ore_sonno', 7.5)), step=0.5, key='sim_sonno')
        with col3:
            sim_stress = st.slider('Stress lavorativo', 1, 10, int(base.get('stress_lavoro', 5)), key='sim_stress')
        with col4:
            sim_rpe = st.slider('RPE', 1, 10, int(base.get('rpe_previsto', 6)), key='sim_rpe')

        stima_sim = stima_rischio_oggi(class_bundle, sim_dist, sim_sonno, sim_stress, sim_rpe)
        sim_prob = stima_sim['probabilita_rf']
        sim_fc = stima_sim['fc_media_stimata']

        if sim_prob >= 60:
            color = ORANGE
            label = 'Rischio elevato'
            advice = f'Un giorno così pesante ({sim_prob:.0f}% di rischio) non è adatto a oggi. Riduci la distanza o fai un allenamento più leggero.'
        elif sim_prob >= 35:
            color = AMBER
            label = 'Rischio moderato'
            advice = f'Il rischio è presente ({sim_prob:.0f}%). Valuta di ridurre la distanza o di fare più recupero.'
        else:
            color = GREEN
            label = 'Rischio basso'
            advice = f'Questo scenario è sostenibile. Il rischio è basso ({sim_prob:.0f}%), quindi puoi procedere con prudenza.'

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                        border: 2px solid {color}; border-radius: 12px; padding: 20px; margin: 18px 0;'>
                <h3 style='margin:0 0 8px 0; color:{color};'>{label}</h3>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>{advice}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode='gauge+number',
                    value=sim_prob,
                    title={'text': 'Probabilità di rischio', 'font': {'color': TXT_SEC}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white'},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': GREEN + '22'},
                            {'range': [30, 60], 'color': AMBER + '22'},
                            {'range': [60, 100], 'color': ORANGE + '22'}
                        ]
                    }
                )
            )
            fig_gauge.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TXT_SEC))
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)

        with col_b:
            sonno_range = np.linspace(4, 10, 15)
            probs_range = [
                rf_model.predict_proba(scaler.transform(np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100
                for s in sonno_range
            ]
            fig_sens = px.line(
                x=sonno_range,
                y=probs_range,
                labels={'x': 'Ore di sonno', 'y': 'Rischio %'},
                title='Sensibilità: rischio vs sonno'
            )
            fig_sens.add_vline(x=sim_sonno, line_dash='dash', line_color=AMBER)
            fig_sens.update_traces(line_color=CYAN, line_width=3)
            fig_sens.update_layout(
                height=320,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_sens), use_container_width=True)

    # =============================================================
    # TAB 6 - CONFRONTO MODELLI
    # =============================================================
    with tab6:
        st.subheader("Random Forest vs Logistic Regression")
        st.markdown(
            """
            <div style='background: #0E1420; border-left: 4px solid #FFB020; padding: 14px 16px; border-radius: 8px;'>
                <p style='margin:0; color:#B8C2D0; line-height:1.6;'>
                    Due approcci diversi: <strong>Random Forest</strong> è più precisa e sofisticata, mentre <strong>Logistic Regression</strong>
                    è più semplice da leggere e capire. Entrambi hanno un ruolo utile nella tua analisi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        y_pred_log_bin = (y_proba_log >= 0.5).astype(int)

        acc_log = (y_pred_log_bin == y_train_class).mean() * 100
        prec_log = precision_score(y_train_class, y_pred_log_bin, zero_division=0) * 100
        rec_log = recall_score(y_train_class, y_pred_log_bin, zero_division=0) * 100
        f1_log = f1_score(y_train_class, y_pred_log_bin, zero_division=0) * 100

        f1_rf_final = f1_score(y_train_class, y_pred_rf, zero_division=0) * 100
        auc_rf_final = auc(*roc_curve(y_train_class, y_proba_rf)[:2])
        auc_log_final = auc(*roc_curve(y_train_class, y_proba_log)[:2])

        metriche_labels = ['Accuratezza', 'Precisione', 'Sensibilità', 'F1-Score', 'AUC']
        rf_values = [acc_rf, prec_rf, rec_rf, f1_rf_final, auc_rf_final * 100]
        log_values = [acc_log, prec_log, rec_log, f1_log, auc_log_final * 100]

        score_rf = float(np.mean(rf_values))
        score_log = float(np.mean(log_values))
        vincitore = 'Random Forest' if score_rf >= score_log else 'Logistic Regression'

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0E1420 0%, #0F1D30 100%);
                    border: 1px solid #202B3D; border-radius: 14px; padding: 18px; margin: 12px 0 22px 0;'>
            <h3 style='margin:0 0 8px 0; color:#00E5FF;'>{vincitore} vince sul tuo storico</h3>
            <p style='margin:0; color:#B8C2D0;'>
                Punteggio medio: <strong>{max(score_rf, score_log):.1f}/100</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=rf_values + [rf_values[0]],
                theta=metriche_labels + [metriche_labels[0]],
                fill='toself',
                name='Random Forest',
                line=dict(color=CYAN, width=2),
                fillcolor='rgba(0,229,255,0.15)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=log_values + [log_values[0]],
                theta=metriche_labels + [metriche_labels[0]],
                fill='toself',
                name='Logistic Regression',
                line=dict(color=AMBER, width=2),
                fillcolor='rgba(255,176,32,0.15)'
            ))
            fig_radar.update_layout(
                height=360,
                title='Confronto completo',
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)

        with col_b:
            comp_data = pd.DataFrame({
                'Metrica': metriche_labels * 2,
                'Valore': rf_values + log_values,
                'Modello': ['Random Forest'] * 5 + ['Logistic Regression'] * 5
            })
            fig_comp = px.bar(
                comp_data,
                x='Metrica',
                y='Valore',
                color='Modello',
                barmode='group',
                color_discrete_map={'Random Forest': CYAN, 'Logistic Regression': AMBER}
            )
            fig_comp.update_layout(
                height=360,
                title='Metriche fianco a fianco',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TXT_SEC)
            )
            st.plotly_chart(style_fig(fig_comp), use_container_width=True)

        st.markdown('---')
        col_rf, col_log = st.columns(2)

        with col_rf:
            st.markdown(
                """
                ### Random Forest
                **Punti forti:**
                - cattura relazioni complesse
                - molto utile per l'allarme di rischio
                - spesso più preciso sul tuo storico

                **Limiti:**
                - è meno trasparente
                - non spiega sempre il perché in modo semplice
                """
            )

        with col_log:
            st.markdown(
                """
                ### Logistic Regression
                **Punti forti:**
                - spiegabile
                - mostra quali fattori pesano di più
                - utile per capire la causa del rischio

                **Limiti:**
                - meno efficace su pattern molto complessi
                - più semplice e quindi meno ricca di dettagli
                """
            )

        st.success("**Regola pratica:** usa Random Forest come allarme principale e Logistic Regression per capire quale fattore sta spingendo il rischio verso l'alto. Insieme danno sia previsione che spiegazione.")

except Exception as e:
    st.error(f"Errore nel caricamento dei modelli ML: {str(e)}")
    st.caption("Controlla i dati del tuo storico e riprova. Se il problema persiste, riavvia la sessione o contatta il supporto.")
