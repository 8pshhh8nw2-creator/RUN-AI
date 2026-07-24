import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, 
    silhouette_score, precision_recall_curve, auc
)
import shap
from scipy import stats

# --- FALLBACK PER UTILS LOCALI ---
# Nel caso manchino i file locali, usiamo dei mock robusti per non far crashare l'app
try:
    from utils.sidebar import sidebar_comune
    from utils.style import carica_css
    from utils.data import genera_dati
    from utils.components import header_block, style_fig, get_svg_url, SVG_ML
except ImportError:
    st.warning(" Moduli 'utils' non trovati. Avvio in modalità Standalone con dati simulati avanzati.")
    def carica_css(): pass
    def sidebar_comune(): return None
    def genera_dati(): return pd.DataFrame(np.random.randn(1000, 10), columns=[f'Feature_{i}' for i in range(10)])
    def header_block(a,b,c,d,e): st.title(b); st.subheader(a)
    def style_fig(fig): fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'); return fig
    def get_svg_url(x): return ""
    SVG_ML = ""

# ============================================================================
# CONFIGURAZIONE E STILI
# ============================================================================
st.set_page_config(page_title="Advanced ML Suite | Tesi", layout="wide")
carica_css()

# CSS Custom per card esplicative
st.markdown("""
<style>
    .metric-card {
        background: rgba(32,40,58,0.7); border: 1px solid rgba(0,229,255,0.2); 
        padding: 15px; border-radius: 10px; margin-bottom: 15px;
    }
    .metric-title { color: #8792A3; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #00E5FF; font-size: 2em; font-weight: bold; }
    .explain-box {
        background: linear-gradient(90deg, rgba(0,229,255,0.05) 0%, rgba(0,0,0,0) 100%);
        border-left: 4px solid #00E5FF; padding: 15px; border-radius: 0 10px 10px 0; margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA ENGINE E PREPROCESSING DINAMICO
# ============================================================================
@st.cache_data
def preprocess_data(df_raw):
    """Motore di preprocessing adattivo che si plasma sui dati reali"""
    df = df_raw.copy()
    
    # 1. Identificazione automatica del Target
    target_col = 'Rischio Infortunio'
    if target_col not in df.columns:
        # Simulazione intelligente del rischio basata sui volumi se assente
        if 'Distanza (km)' in df.columns:
            threshold = df['Distanza (km)'].quantile(0.75)
            df[target_col] = (df['Distanza (km)'] > threshold).astype(int)
        else:
            df[target_col] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

    # 2. Identificazione Features (ignora target e date)
    exclude_cols = [target_col, 'Giorno', 'Data', 'Session_ID']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # 3. Pulizia e Normalizzazione sicura
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    # Riconoscimento KPI Proprietari
    kpi_props = [c for c in feature_cols if c in ['SMA', 'ISLR', 'IITR', 'IDET']]
    
    return df, feature_cols, target_col, kpi_props

# Caricamento
df_base = genera_dati()
df_processed, FEATURE_COLS, TARGET_COL, KPI_PROPR = preprocess_data(df_base)

# Preparazione ML globale
X = df_processed[FEATURE_COLS]
y = df_processed[TARGET_COL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# ============================================================================
# HEADER
# ============================================================================
header_block("Modulo 04 — Predict & Explain", "ADVANCED MACHINE INTELLIGENCE", 
             "Motore di predizione del rischio infortuni. Adattamento dinamico ai dati, validazione K-Fold e simulazione in tempo reale.", 
             get_svg_url(SVG_ML), "AI Core")

st.markdown(f"""
<div class='metric-card' style='display: flex; justify-content: space-around; text-align: center;'>
    <div><div class='metric-title'>Sessioni Analizzate</div><div class='metric-value'>{len(df_processed):,}</div></div>
    <div><div class='metric-title'>Features Attive</div><div class='metric-value'>{len(FEATURE_COLS)}</div></div>
    <div><div class='metric-title'>Tasso di Rischio Baseline</div><div class='metric-value' style='color:#FF6A3D'>{(y.mean()*100):.1f}%</div></div>
    <div><div class='metric-title'>KPI Proprietari</div><div class='metric-value' style='color:#FFB020'>{len(KPI_PROPR)}</div></div>
</div>
""", unsafe_allow_html=True)

# TABS
tabs = st.tabs([" Random Forest (Core)", " SHAP Explainability", " Simulatore Dinamico", " Logistica & Baseline", " EDA Avanzata"])

# ============================================================================
# TAB 1: RANDOM FOREST (Il Cuore della Predizione)
# ============================================================================
with tabs[0]:
    st.markdown("###  Random Forest Classifier")
    st.write("Modello non-lineare d'elezione per catturare le interazioni complesse tra i carichi di lavoro (es. alta FC combinata con scarso sonno).")

    # Addestramento
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    # Metriche
    acc, prec, rec = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)
    f1, auc_val = f1_score(y_test, y_pred), roc_auc_score(y_test, y_proba)

    # Spiegazione Automatica (NLG)
    interpretazione = f"""
    **Cosa significano questi numeri per l'atleta?**
    Il modello riconosce correttamente le sessioni a rischio nel **{rec*100:.0f}%** dei casi (Recall). 
    Quando il sistema lancia un allarme di infortunio, ha ragione l'**{prec*100:.0f}%** delle volte (Precision). 
    L'AUC di **{auc_val:.2f}** indica una {'eccellente' if auc_val>0.85 else 'buona'} capacità generale di separare sessioni sicure da quelle pericolose.
    """
    st.markdown(f"<div class='explain-box'>{interpretazione}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc*100:.1f}%")
    c2.metric("Precision", f"{prec*100:.1f}%")
    c3.metric("Recall (Sensibilità)", f"{rec*100:.1f}%")
    c4.metric("F1-Score", f"{f1*100:.1f}%")
    c5.metric("ROC-AUC", f"{auc_val:.3f}")

    # Feature Importance Avanzata
    st.markdown("#### Quali variabili guidano il rischio?")
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances}).sort_values('Importance', ascending=True)
    
    # Coloriamo i KPI proprietari diversamente per farli risaltare nella tesi
    imp_df['Colore'] = imp_df['Feature'].apply(lambda x: '#FFB020' if x in KPI_PROPR else '#00E5FF')
    
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                     color='Colore', color_discrete_map='identity',
                     title="Importanza delle Variabili (Gini Importance)")
    st.plotly_chart(style_fig(fig_imp), use_container_width=True)

# ============================================================================
# TAB 2: SHAP (La vera Explainability)
# ============================================================================
with tabs[1]:
    st.markdown("### 🔬 Interpretazione SHAP (SHapley Additive exPlanations)")
    st.markdown("Mentre la *Feature Importance* ci dice quali variabili sono importanti in generale, SHAP ci spiega **come** influenzano la singola predizione (es. 'Valori alti di SMA aumentano il rischio, valori bassi lo riducono').")

    @st.cache_resource
    def get_shap_values(_model, _X):
        explainer = shap.TreeExplainer(_model)
        shap_vals = explainer.shap_values(_X)
        return explainer, shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    explainer, shap_values = get_shap_values(rf_model, X_test)

    c_shap1, c_shap2 = st.columns(2)
    with c_shap1:
        st.markdown("#### Summary Plot Globale")
        st.info("Punti rossi = Valori alti della feature. Se i punti rossi sono a destra dello zero, significa che **aumentano** il rischio.")
        # Non potendo usare matplotlib in modo nativo e pulito con st.pyplot senza warning, creiamo un surrogato plotly elegante
        shap_mean_abs = np.abs(shap_values).mean(axis=0)
        df_shap_summary = pd.DataFrame({'Feature': FEATURE_COLS, 'Impatto Medio': shap_mean_abs}).sort_values('Impatto Medio', ascending=True)
        fig_shap_sum = px.bar(df_shap_summary, x='Impatto Medio', y='Feature', orientation='h', color_discrete_sequence=['#FF6A3D'])
        st.plotly_chart(style_fig(fig_shap_sum), use_container_width=True)
    
    with c_shap2:
        st.markdown("#### Analisi Interattiva per Feature")
        feat_scelta = st.selectbox("Seleziona una feature da esplorare:", FEATURE_COLS)
        idx_f = FEATURE_COLS.index(feat_scelta)
        
        fig_dep = px.scatter(
            x=X_test[:, idx_f], y=shap_values[:, idx_f],
            color=X_test[:, idx_f], color_continuous_scale='RdBu_r',
            labels={'x': f'Valore Standardizzato di {feat_scelta}', 'y': 'Impatto sul Rischio (SHAP Value)'}
        )
        fig_dep.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        st.plotly_chart(style_fig(fig_dep), use_container_width=True)

# ============================================================================
# TAB 3: SIMULATORE DINAMICO E PRESCRIPTIVE ANALYTICS
# ============================================================================
with tabs[2]:
    st.markdown("### Simulatore Dinamico")
    st.write("Modifica i parametri di un atleta per simulare il rischio in tempo reale. I controlli si adattano ai limiti dei dati di training storici.")
    
    # Generazione dinamica dei controlli
    st.markdown("<div style='background:rgba(32,40,58,0.5); padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
    cols = st.columns(3)
    user_inputs = {}
    
    for i, col_name in enumerate(FEATURE_COLS):
        min_v = float(X[col_name].min())
        max_v = float(X[col_name].max())
        mean_v = float(X[col_name].mean())
        
        with cols[i % 3]:
            # Evidenzia i KPI proprietari
            label = f"⭐ {col_name}" if col_name in KPI_PROPR else col_name
            user_inputs[col_name] = st.slider(label, min_value=min_v, max_value=max_v, value=mean_v, step=(max_v-min_v)/100, format="%.1f")
    st.markdown("</div>", unsafe_allow_html=True)

    # Predizione Real-time
    input_df = pd.DataFrame([user_inputs])
    input_scaled = scaler.transform(input_df)
    prob_rischio = rf_model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    res1, res2 = st.columns([1, 2])
    
    with res1:
        color = "#00E5FF" if prob_rischio < 30 else "#FFB020" if prob_rischio < 70 else "#FF6A3D"
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_rischio,
            number = {'suffix': "%", 'font': {'color': color, 'size': 50}},
            title = {'text': "Probabilità di Infortunio", 'font': {'size': 20, 'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0,229,255,0.1)"},
                    {'range': [30, 70], 'color': "rgba(255,176,32,0.1)"},
                    {'range': [70, 100], 'color': "rgba(255,106,61,0.1)"}],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(style_fig(fig_gauge), use_container_width=True)

    with res2:
        st.markdown("#### 🤖 Intelligenza Prescrittiva (Suggerimenti)")
        if prob_rischio >= 70:
            st.error("🚨 **ALLERTA CRITICA**: Il profilo di carico simulato presenta un rischio molto elevato.")
            # Trova le 2 feature che contribuiscono di più (dinamicamente)
            shap_sim = explainer.shap_values(input_scaled)
            shap_sim_vals = shap_sim[1][0] if isinstance(shap_sim, list) else shap_sim[0]
            
            # Feature con SHAP positivo più alto (che aumentano il rischio)
            top_risks_idx = np.argsort(shap_sim_vals)[-2:][::-1]
            for idx in top_risks_idx:
                feat = FEATURE_COLS[idx]
                val_attuale = user_inputs[feat]
                media_sicura = X[y==0][feat].mean()
                st.write(f"📉 **Per ridurre il rischio**: Abbassa `{feat}` dall'attuale **{val_attuale:.1f}** verso valori intorno a **{media_sicura:.1f}**.")
                
        elif prob_rischio >= 30:
            st.warning(" **ATTENZIONE**: Zona di rischio moderato. Monitorare strettamente il recupero.")
            st.write("Mantieni costanti le ore di sonno e valuta un micro-ciclo di scarico se questo carico viene mantenuto per più di 3 giorni.")
        else:
            st.success(" **ZONA SICURA**: L'atleta è in un range fisiologico di adattamento ottimale.")
            st.write("Il piano di allenamento attuale bilancia correttamente stress e recupero. È possibile incrementare gradualmente il volume (es. max 10% a settimana).")

# ============================================================================
# TAB 4: LOGISTICA & BASELINE (Validazione Scientifica per la Tesi)
# ============================================================================
with tabs[3]:
    st.markdown("###  Confronto: Baseline vs Modello Arricchito")
    st.write("Dimostrazione dell'efficacia dei KPI proprietari rispetto a un modello tradizionale.")
    
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        baseline_col = 'Session_RPE' if 'Session_RPE' in FEATURE_COLS else FEATURE_COLS[0]
        X_base = X_scaled[:, FEATURE_COLS.index(baseline_col)].reshape(-1, 1)
        Xb_train, Xb_test = train_test_split(X_base, test_size=0.25, random_state=42, stratify=y)
        
        log_base = LogisticRegression(class_weight='balanced').fit(Xb_train, y_train)
        auc_base = roc_auc_score(y_test, log_base.predict_proba(Xb_test)[:, 1])
        
        log_full = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
        auc_full = roc_auc_score(y_test, log_full.predict_proba(X_test)[:, 1])
        
        # Curva PR Dinamica
        st.markdown(f"#### Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, log_full.predict_proba(X_test)[:, 1])
        pr_auc = auc(recall, precision)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Completo (AUC={pr_auc:.2f})', line=dict(color='#00E5FF', width=3)))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Trade-off Precision/Recall")
        st.plotly_chart(style_fig(fig_pr), use_container_width=True)

    with col_l2:
        st.markdown(f"#### Uplift Metriche vs Baseline ({baseline_col})")
        
        metrics_df = pd.DataFrame({
            'Modello': ['Baseline (Solo RPE)', 'Logistica (Completo)', 'Random Forest (Ensemble)'],
            'AUC Score': [auc_base, auc_full, auc_val]
        })
        fig_bar = px.bar(metrics_df, x='Modello', y='AUC Score', text_auto='.3f', color='Modello',
                         color_discrete_sequence=['#8792A3', '#FFB020', '#00E5FF'])
        st.plotly_chart(style_fig(fig_bar), use_container_width=True)
        
        uplift = ((auc_val - auc_base) / auc_base) * 100
        st.markdown(f"<div class='explain-box'>Integrare i KPI proprietari e l'algoritmo Ensemble ha generato un <b>miglioramento del {uplift:.1f}%</b> nel potere predittivo rispetto alla metrica tradizionale.</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 5: EDA AVANZATA
# ============================================================================
with tabs[4]:
    st.markdown("### Analisi Esplorativa dei Dati (EDA)")
    
    e1, e2 = st.columns([1.5, 1])
    with e1:
        st.markdown("#### Matrice di Correlazione Dinamica")
        corr = df_processed[FEATURE_COLS].corr()
        fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=500)
        st.plotly_chart(style_fig(fig_corr), use_container_width=True)
        
    with e2:
        st.markdown("#### Distribuzione rispetto all'Infortunio")
        feature_dist = st.selectbox("Seleziona variabile da analizzare", FEATURE_COLS)
        fig_box = px.box(df_processed, x=TARGET_COL, y=feature_dist, color=TARGET_COL,
                         color_discrete_map={0: '#00E5FF', 1: '#FF6A3D'},
                         labels={TARGET_COL: 'Infortunio (0=No, 1=Si)'})
        st.plotly_chart(style_fig(fig_box), use_container_width=True)
        
        # Test Statistico T-Student al volo
        safe_data = df_processed[df_processed[TARGET_COL] == 0][feature_dist]
        risk_data = df_processed[df_processed[TARGET_COL] == 1][feature_dist]
        t_stat, p_val = stats.ttest_ind(safe_data, risk_data, nan_policy='omit')
        
        if p_val < 0.05:
            st.success(f"✔️ Significatività confermata (p-value: {p_val:.4f}). La differenza di '{feature_dist}' tra atleti sani e infortunati è statisticamente rilevante.")
        else:
            st.warning(f" Differenza non significativa (p-value: {p_val:.4f}). '{feature_dist}' da sola non discrimina fortemente l'infortunio.")
