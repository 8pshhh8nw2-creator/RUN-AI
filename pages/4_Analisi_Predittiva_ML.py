from utils.sidebar import sidebar_comune
df, df_full, filtro_tempo = sidebar_comune()
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.metrics import silhouette_score

from utils.style import carica_css
from utils.data import genera_dati
from utils.components import header_block, style_fig, get_svg_url, SVG_ML

st.set_page_config(page_title="Analisi Predittiva ML", layout="wide")
carica_css()

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()

IMG_HERO_ML = get_svg_url(SVG_ML)


# =========================================================
# PARTE 1 — DA AGGIUNGERE DENTRO genera_dati(), subito dopo
# la riga che calcola df['SMA'] = ...
# Serve per allineare il dataset ai 4 KPI proprietari della tesi
# (SMA era già presente, qui aggiungiamo ISLR, IITR, IDET)
# =========================================================

    # --- Vento, necessario per IITR (non presente prima) ---
    df['Vento (km/h)'] = np.round(np.random.uniform(0, 25, n), 1)

    # --- ISLR: Indice di Sforzo Lavorativo Residuo ---
    # ISLR = (Ore Lavoro x Stress Lavoro) / Distanza
    df['ISLR'] = np.where(
        df['Distanza (km)'] > 0,
        (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'],
        0
    )

    # --- IITR: Indice di Impatto Termico e Resistenza ---
    # IITR = (Temperatura x Vento) / Distanza
    df['IITR'] = np.where(
        df['Distanza (km)'] > 0,
        (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'],
        0
    )

    # --- IDET: Indice di Degradazione Termica ---
    # IDET = (FC Media x Temperatura) / Velocità
    df['IDET'] = np.where(
        df['Velocità (km/h)'] > 0,
        (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'],
        0
    )

    # --- Session-RPE (Foster), usata come baseline di confronto ---
    # Durata stimata in minuti = (Distanza / Velocità) x 60
    df['Durata (min)'] = np.where(
        df['Velocità (km/h)'] > 0,
        (df['Distanza (km)'] / df['Velocità (km/h)']) * 60,
        0
    )
    df['Session_RPE'] = df['RPE'] * df['Durata (min)']

    return df


# =========================================================
# PARTE 2 — SOSTITUISCE INTERAMENTE IL BLOCCO
#   elif pagina == "ANALISI PREDITTIVA ML":
#       ...
# Contiene SOLO i 5 modelli descritti nella proposta di tesi:
# Regressione Lineare, Regressione Logistica, Random Forest,
# Clustering K-Means, Stress/Overload Prediction (time series)
# =========================================================


header_block(
    "Modulo 04 — Model Explainability",
    "ANALISI PREDITTIVA ML",
    "I 5 modelli di Machine Learning descritti nella tesi, applicati al tuo storico di allenamento.",
    IMG_HERO_ML, "Machine Learning Engine"
)

df_base = st.session_state.dati.copy()

st.markdown("""
<div class='info-box'>
<h3>Come leggere questa pagina</h3>
<p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>
Ogni scheda qui sotto corrisponde a uno dei modelli descritti nel Capitolo 2 della tesi.
Per ciascuno trovi: una spiegazione in linguaggio semplice, il grafico principale,
e le metriche di validazione — incluso un confronto con una <strong>baseline</strong>
(la Session-RPE di Foster) per capire quanto i KPI proprietari (SMA, ISLR, IITR, IDET)
aggiungano davvero rispetto a uno standard già noto in letteratura.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HELPER riutilizzabili in questa pagina
# ---------------------------------------------------------------
def verdetto_box(valore_pct, soglie=(50, 75), testo_basso="Debole", testo_medio="Discreto", testo_alto="Solido", spiegazione=""):
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

try:
    # -----------------------------------------------------
    # PREPARAZIONE DATI COMUNE A TUTTI I MODELLI
    # -----------------------------------------------------
    feature_cols = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE',
                    'SMA', 'ISLR', 'IITR', 'IDET']
    feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE',
                     'SMA', 'ISLR', 'IITR', 'IDET']
    kpi_proprietari = {'SMA', 'ISLR', 'IITR', 'IDET'}  # per colorare diversamente nei grafici

    for col in feature_cols + ['Rischio Infortunio', 'Velocità (km/h)', 'Session_RPE']:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce').fillna(0)

    X_class = df_base[feature_cols].values
    y_class = df_base['Rischio Infortunio'].astype(int).values

    scaler = StandardScaler()
    X_scaled_class = scaler.fit_transform(X_class)

    n_sessioni = len(df_base)
    if n_sessioni < 60:
        st.markdown(f"""
        <div style='border-left: 4px solid #FFB020; background: rgba(255,176,32,0.08);
                    padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;'>
            <span style='color:#FFB020; font-weight:700;'>⚠️ Campione contenuto</span>
            <p style='color:#B8C2D0; margin-top:6px; font-family:"Inter",sans-serif; font-size:0.92em;'>
            Con <strong>{n_sessioni} sessioni</strong>, le metriche riportate qui sotto vanno lette come
            indicazioni di tendenza, non come valori assoluti. Per questo motivo, oltre al singolo
            split train/test, viene sempre mostrata anche una validazione incrociata (cross-validation)
            più robusta rispetto alla dimensione del campione.
            </p>
        </div>
        """, unsafe_allow_html=True)

    unique_classes = np.unique(y_class)
    has_multiple_classes = bool(len(unique_classes) > 1)
    has_enough_samples = bool(len(df_base) >= 10)
    stratify_arg = y_class if (has_multiple_classes and has_enough_samples) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_class, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
    )
    has_multiple_test_classes = bool(len(np.unique(y_test)) > 1)

    # -----------------------------------------------------
    # TAB DEFINITION — SOLO I 5 MODELLI DELLA TESI
    # -----------------------------------------------------
    t_lin, t_log, t_rf, t_clu, t_stress = st.tabs([
        " Regressione Lineare", " Regressione Logistica",
        " Random Forest", " Clustering K-Means", " Stress / Overload Prediction"
    ])

    # =====================================================
    # TAB 1 — REGRESSIONE LINEARE (Volume vs Performance)
    # =====================================================
    with t_lin:
        st.markdown("### Regressione Lineare — Volume di Allenamento vs Performance")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> stima la relazione tra il volume di allenamento (distanza) e un indicatore di performance (velocità media), assumendo una relazione lineare tra le due variabili.</div>", unsafe_allow_html=True)
        in_pratica("più km fai, in genere corri più veloce nel tempo — ma quanto esattamente? Questo modello traccia la 'retta di tendenza' che meglio riassume questa relazione nei tuoi dati.")

        X_lr = df_base[['Distanza (km)']].values
        y_lr = df_base['Velocità (km/h)'].values

        X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
        lr_model = LinearRegression().fit(X_lr_train, y_lr_train)
        y_lr_pred_test = lr_model.predict(X_lr_test)

        r2_test = r2_score(y_lr_test, y_lr_pred_test)
        rmse_test = mean_squared_error(y_lr_test, y_lr_pred_test) ** 0.5

        fig_lr = px.scatter(df_base, x='Distanza (km)', y='Velocità (km/h)', color='RPE',
                            color_continuous_scale=[[0, '#00E5FF'], [1, '#FF6A3D']])
        x_range = np.linspace(df_base['Distanza (km)'].min(), df_base['Distanza (km)'].max(), 50).reshape(-1, 1)
        y_range_pred = lr_model.predict(x_range)
        fig_lr.add_trace(go.Scatter(x=x_range.flatten(), y=y_range_pred, mode='lines',
                                    line=dict(color='#00F5A0', width=3, dash='dash'), name='Retta di regressione'))
        fig_lr.update_layout(height=380, title="Distanza vs Velocità media, con retta di tendenza")
        st.plotly_chart(style_fig(fig_lr), use_container_width=True)

        rc1, rc2 = st.columns(2)
        rc1.metric("R² (test)", f"{r2_test:.2f}", help="Quota di variabilità della velocità spiegata dal volume (1.0 = perfetto)")
        rc2.metric("RMSE (test)", f"{rmse_test:.2f} km/h", help="Errore medio di previsione")

        verdetto_box(
            max(r2_test, 0) * 100, soglie=(30, 60),
            testo_basso=f"Il volume da solo spiega poco la tua velocità (errore medio ±{rmse_test:.1f} km/h): altri fattori (sonno, stress, KPI) pesano di più",
            testo_medio=f"Il volume ha un'influenza parziale sulla velocità, con un margine di errore di ±{rmse_test:.1f} km/h",
            testo_alto=f"Il volume di allenamento spiega bene la tua velocità, con un errore medio di soli ±{rmse_test:.1f} km/h",
            spiegazione="Questo è il modello più semplice della tesi: serve come punto di partenza, non come risposta definitiva — è normale che da solo non spieghi tutto."
        )

    # =====================================================
    # TAB 2 — REGRESSIONE LOGISTICA (con baseline Session-RPE)
    # =====================================================
    with t_log:
        st.markdown("### Regressione Logistica — Predizione dello Stato di Overload")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> stima la probabilità che una sessione rientri in uno stato di sovraccarico (overload), a partire dalla combinazione delle variabili raccolte.</div>", unsafe_allow_html=True)
        in_pratica("prima confrontiamo il modello completo con una 'baseline' che usa solo la Session-RPE (il metodo classico di Foster) — così vediamo onestamente quanto i tuoi KPI aggiungono davvero.")

        # --- BASELINE: solo Session-RPE ---
        X_baseline = df_base[['Session_RPE']].values
        X_baseline_scaled = StandardScaler().fit_transform(X_baseline)
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(
            X_baseline_scaled, y_class, test_size=0.25, random_state=42, stratify=stratify_arg
        )
        baseline_model = LogisticRegression(random_state=42, max_iter=1000).fit(Xb_train, yb_train)
        y_pred_baseline = baseline_model.predict(Xb_test)
        acc_baseline = accuracy_score(yb_test, y_pred_baseline)
        f1_baseline = f1_score(yb_test, y_pred_baseline, zero_division=0)

        # --- MODELLO COMPLETO: tutte le feature + KPI ---
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        y_proba_log = log_model.predict_proba(X_test)[:, 1]
        acc_l = accuracy_score(y_test, y_pred_log)
        f1_l = f1_score(y_test, y_pred_log, zero_division=0)
        auc_l = roc_auc_score(y_test, y_proba_log) if has_multiple_test_classes else float('nan')

        st.markdown("#### Baseline (Session-RPE) vs Modello Completo (KPI proprietari)")
        fig_confronto = go.Figure()
        fig_confronto.add_trace(go.Bar(name='Baseline (solo Session-RPE)',
                                        x=['Accuracy', 'F1-Score'], y=[acc_baseline * 100, f1_baseline * 100],
                                        marker_color='#8792A3'))
        fig_confronto.add_trace(go.Bar(name='Modello Completo (+ KPI)',
                                        x=['Accuracy', 'F1-Score'], y=[acc_l * 100, f1_l * 100],
                                        marker_color='#00E5FF'))
        fig_confronto.update_layout(height=350, barmode='group', yaxis_title="%",
                                    title="Quanto aggiungono davvero i KPI proprietari?")
        st.plotly_chart(style_fig(fig_confronto), use_container_width=True)

        delta_acc = (acc_l - acc_baseline) * 100
        in_pratica(f"il modello con i KPI proprietari {'migliora' if delta_acc > 0 else 'non migliora'} l'accuracy di {abs(delta_acc):.1f} punti percentuali rispetto alla sola Session-RPE — questo è il numero da riportare nel Capitolo 4 della tesi (sezione 4.0) al posto dei placeholder.")

        st.markdown("#### Coefficienti del modello completo")
        coefs = log_model.coef_[0]
        colors = ['#FF6A3D' if fname in kpi_proprietari and c > 0 else
                  '#00F5A0' if fname in kpi_proprietari else
                  ('#FFB020' if c > 0 else '#8792A3')
                  for fname, c in zip(feature_names, coefs)]
        fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors,
                                   text=[f'{c:+.2f}' for c in coefs], textposition='auto'))
        fig_log.update_layout(height=380, title="Peso di ciascuna variabile (arancione/verde = KPI proprietari)",
                              yaxis_title="Coefficiente — positivo = aumenta il rischio")
        fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
        st.plotly_chart(style_fig(fig_log), use_container_width=True)

        fattore_peggiore = feature_names[int(np.argmax(coefs))]
        st.caption(f"La variabile che pesa di più sul rischio è **{fattore_peggiore}**.")

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Accuracy (modello completo)", f"{acc_l*100:.1f}%")
        lc2.metric("F1-Score (modello completo)", f"{f1_l*100:.1f}%")
        lc3.metric("ROC-AUC", f"{auc_l:.2f}" if not np.isnan(auc_l) else "N/D")

        verdetto_box(
            acc_l * 100, soglie=(60, 80),
            testo_basso="Il modello sbaglia spesso — utile solo come indizio di tendenza",
            testo_medio="Il modello coglie una tendenza reale, con margine d'errore",
            testo_alto="Il modello è affidabile su questi dati",
            spiegazione="Ricorda: il confronto con la baseline sopra è più importante del numero assoluto di accuracy."
        )

    # =====================================================
    # TAB 3 — RANDOM FOREST
    # =====================================================
    with t_rf:
        st.markdown("### Random Forest — Classificazione del Rischio")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> combina molti alberi decisionali indipendenti per catturare relazioni non lineari tra le variabili, che la Regressione Logistica da sola non riesce a vedere.</div>", unsafe_allow_html=True)
        in_pratica("immagina 100 allenatori diversi che guardano i tuoi dati e votano indipendentemente 'rischio' o 'sicuro': il modello prende la media dei loro pareri.")

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, min_samples_split=5)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        c1, c2 = st.columns(2)
        with c1:
            importances = rf_model.feature_importances_
            imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            colors_imp = ['#FF6A3D' if name in kpi_proprietari else '#00E5FF' for name, _ in imp_data]
            fig_imp = go.Figure(go.Bar(
                y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h',
                marker_color=colors_imp, text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'
            ))
            fig_imp.update_layout(height=380, yaxis=dict(autorange="reversed"),
                                  title="Importanza delle variabili (arancione = KPI proprietari)",
                                  xaxis_title="Importanza relativa (%)")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            var_top = imp_data[0][0]
            nota_kpi = " — ed è uno dei tuoi KPI proprietari!" if var_top in kpi_proprietari else ""
            st.caption(f"La variabile più determinante è **{var_top}**{nota_kpi}")
        with c2:
            cm = confusion_matrix(y_test, y_pred_rf)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"},
                colorscale=[[0, '#0E1420'], [1, '#00E5FF']], showscale=False
            ))
            fig_cm.update_layout(height=380, title="Matrice di Confusione (dati di TEST)")
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)
            st.caption("Diagonale = previsioni corrette. Fuori diagonale = errori del modello.")

        acc = accuracy_score(y_test, y_pred_rf)
        prec = precision_score(y_test, y_pred_rf, zero_division=0)
        rec = recall_score(y_test, y_pred_rf, zero_division=0)
        f1 = f1_score(y_test, y_pred_rf, zero_division=0)
        roc_auc_rf = roc_auc_score(y_test, y_proba_rf) if has_multiple_test_classes else float('nan')

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy", f"{acc*100:.1f}%")
        mc2.metric("Precision", f"{prec*100:.1f}%")
        mc3.metric("Recall", f"{rec*100:.1f}%")
        mc4.metric("F1-Score", f"{f1*100:.1f}%")
        mc5.metric("ROC-AUC", f"{roc_auc_rf:.2f}" if not np.isnan(roc_auc_rf) else "N/D")

        try:
            cv_scores = cross_val_score(
                RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, min_samples_split=5),
                X_scaled_class, y_class, cv=5, scoring='accuracy'
            )
            st.caption(f"Cross-validation a 5 fold su tutto il dataset: accuracy media {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}% — più questa oscillazione è piccola, più il risultato è stabile.")
        except ValueError:
            st.caption("Cross-validation non disponibile: servono più campioni per classe.")

        verdetto_box(
            acc * 100, soglie=(60, 80),
            testo_basso="Il modello sbaglia spesso — usalo solo come indizio, non come verdetto",
            testo_medio="Il modello coglie una tendenza reale ma con margine d'errore",
            testo_alto="Il modello è affidabile su questi dati",
            spiegazione=f"Su 100 sessioni di test, il modello ne classifica correttamente circa {acc*100:.0f}, individuando circa {rec*100:.0f} sessioni a rischio su 100 realmente a rischio."
        )

        # --- Analisi errori (sessioni classificate male) ---
        st.markdown("#### Quando il modello sbaglia")
        idx_test = np.arange(len(df_base))[-len(y_test):]  # approssimazione indici test se split non shuffle-safe
        errori_mask = y_pred_rf != y_test
        n_errori = int(errori_mask.sum())
        st.caption(f"Il modello ha classificato erroneamente **{n_errori} sessioni su {len(y_test)}** nel test set. Guardare queste sessioni una per una (data, KPI, condizioni) aiuta a capire i limiti del modello — utile per il Capitolo 4/5 della tesi.")

    # =====================================================
    # TAB 4 — CLUSTERING K-MEANS
    # =====================================================
    with t_clu:
        st.markdown("### Clustering K-Means — Profili di Allenamento")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> raggruppa automaticamente le sessioni simili tra loro, senza che tu debba etichettarle a mano.</div>", unsafe_allow_html=True)
        in_pratica("il modello non sa nulla di 'tipi di allenamento', ma guardando distanza, FC media e sforzo lavorativo residuo riesce a raggruppare da solo le sessioni simili tra loro.")

        X_clust = df_base[['Distanza (km)', 'FC Media', 'ISLR']].values
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_base['Cluster_ID'] = km.fit_predict(X_clust)
        df_base['Cluster_Type'] = df_base['Cluster_ID'].apply(lambda x: f"Cluster {x+1}")

        try:
            sil = silhouette_score(X_clust, df_base['Cluster_ID'])
        except ValueError:
            sil = float('nan')

        fig_km = px.scatter(df_base, x='Distanza (km)', y='FC Media', color='Cluster_Type',
                            color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='ISLR')
        fig_km.update_layout(height=420, title="Sessioni raggruppate per tipologia (dimensione bolla = ISLR)")
        st.plotly_chart(style_fig(fig_km), use_container_width=True)

        cluster_sizes = df_base['Cluster_Type'].value_counts()
        st.caption("Distribuzione sessioni per cluster: " + ", ".join([f"**{k}**: {v} sessioni" for k, v in cluster_sizes.items()]))

        if not np.isnan(sil):
            verdetto_box(
                (sil + 1) / 2 * 100, soglie=(50, 70),
                testo_basso=f"I gruppi si sovrappongono parecchio (silhouette {sil:.2f}): le tue sessioni non si dividono in categorie nettamente distinte",
                testo_medio=f"I gruppi sono discretamente separati (silhouette {sil:.2f})",
                testo_alto=f"I gruppi sono ben separati (silhouette {sil:.2f}): esistono profili di allenamento davvero distinti nel tuo storico",
                spiegazione="Il Silhouette Score misura quanto ogni sessione sia più simile al proprio gruppo che agli altri: valori vicino a 1 indicano cluster ben separati, vicino a 0 gruppi confusi tra loro."
            )

    # =====================================================
    # TAB 5 — STRESS / OVERLOAD PREDICTION (Time Series)
    # =====================================================
    with t_stress:
        st.markdown("### Stress / Overload Prediction — Andamento nel Tempo")
        st.markdown("<div class='explain-text'><strong>Cosa fa questo modello:</strong> osserva la tendenza degli ultimi giorni (non un singolo giorno isolato) per anticipare un possibile sovraccarico.</div>", unsafe_allow_html=True)
        in_pratica("uno stress alto isolato non preoccupa, ma una tendenza in salita costante sì — per questo guardiamo la media mobile, non il singolo dato del giorno.")

        df_stress = df_base[['Giorno', 'SMA', 'ISLR']].sort_values('Giorno').reset_index(drop=True).copy()
        df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
        df_stress['ISLR_Rolling'] = df_stress['ISLR'].rolling(7, min_periods=1).mean()

        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['SMA_Rolling'], name='SMA (media 7gg)',
                                    line=dict(color='#FF6A3D', width=2.5), fill='tozeroy', fillcolor='rgba(255,106,61,0.12)'))
        fig_sp.add_trace(go.Scatter(x=df_stress['Giorno'], y=df_stress['ISLR_Rolling'], name='ISLR (media 7gg)',
                                    line=dict(color='#FFB020', width=2.5)))
        fig_sp.add_hline(y=15, line_dash="dash", line_color="#8792A3", annotation_text="Soglia critica SMA")
        fig_sp.update_layout(height=400, title="Media Mobile a 7 giorni di SMA e ISLR",
                             xaxis_title="Giorno", yaxis_title="Livello")
        st.plotly_chart(style_fig(fig_sp), use_container_width=True)

        giorni_sopra_soglia = int((df_stress['SMA_Rolling'] > 15).sum())
        st.caption(f"Giorni con SMA sopra la soglia critica: **{giorni_sopra_soglia} su {len(df_stress)}**.")

        n_forecast = 7
        x_idx = np.arange(len(df_stress))
        valid_mask = df_stress['SMA_Rolling'].notna().to_numpy()

        if int(valid_mask.sum()) >= 3:
            coeffs = np.polyfit(x_idx[valid_mask], df_stress['SMA_Rolling'][valid_mask], deg=1)
            trend_fn = np.poly1d(coeffs)
            future_idx = np.arange(len(df_stress), len(df_stress) + n_forecast)
            future_vals = trend_fn(future_idx)
            residual_std = np.std(df_stress['SMA_Rolling'][valid_mask] - trend_fn(x_idx[valid_mask]))

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=list(range(len(df_stress))), y=df_stress['SMA_Rolling'],
                                              mode='lines', line=dict(color='#00E5FF'), name="Storico"))
            fig_forecast.add_trace(go.Scatter(x=list(future_idx), y=future_vals,
                                              mode='lines', line=dict(color='#FF6A3D', dash='dash'), name="Proiezione"))
            fig_forecast.add_trace(go.Scatter(
                x=list(future_idx) + list(future_idx)[::-1],
                y=list(future_vals + residual_std) + list(future_vals - residual_std)[::-1],
                fill='toself', fillcolor='rgba(255,106,61,0.15)', line=dict(color='rgba(0,0,0,0)'), name="Incertezza"
            ))
            fig_forecast.update_layout(height=380, title=f"Proiezione SMA — Prossimi {n_forecast} giorni")
            st.plotly_chart(style_fig(fig_forecast), use_container_width=True)

            trend_direzione = "in aumento 📈" if coeffs[0] > 0.01 else ("in diminuzione 📉" if coeffs[0] < -0.01 else "stabile ➡️")
            verdetto_box(
                max(0, 100 - abs(future_vals[-1] - 15) * 5) if future_vals[-1] <= 15 else 20,
                soglie=(40, 70),
                testo_basso=f"Il trend è {trend_direzione} verso livelli di rischio: valuta un blocco di recupero",
                testo_medio=f"Trend {trend_direzione}, da monitorare",
                testo_alto=f"Trend {trend_direzione}, lontano dalla soglia critica",
                spiegazione="La banda arancione indica l'incertezza della proiezione: più è larga, meno la previsione è precisa."
            )
        else:
            st.info("Servono almeno 3 punti validi per calcolare una proiezione.")

except Exception as e:
    st.error(f"Errore caricamento modelli ML: {str(e)}")
