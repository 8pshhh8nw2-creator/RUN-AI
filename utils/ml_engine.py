"""
utils/ml_engine.py
------------------
Motore ML condiviso tra la pagina "Analisi Predittiva ML" e la pagina
"Consiglio Finale". I modelli vengono addestrati UNA sola volta (grazie
a st.cache_resource) sui dati storici e vengono poi consultati da
entrambe le pagine: così il verdetto del Consiglio Finale è sempre
coerente con ciò che l'utente vede nella pagina di analisi.

Modelli inclusi:
  - Random Forest       -> classificazione del rischio infortunio
  - Logistic Regression -> classificazione trasparente/interpretabile
  - Linear Regression   -> FC attesa vs FC reale (anomalie fisiologiche)
  - K-Means             -> profilo/cluster dell'allenamento odierno
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc,
    r2_score, mean_absolute_error, silhouette_score
)

FEATURE_COLS_CLASS = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']
FEATURE_LABELS_CLASS = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']

FEATURE_COLS_REG = ['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']
FEATURE_COLS_CLUST = ['Distanza (km)', 'FC Media']

# Pesi usati per combinare i segnali dei diversi modelli nel punteggio di
# rischio finale mostrato nel Consiglio Finale. Modificabili liberamente
# per ricalibrare quanto conta ciascun modello (la somma dovrebbe fare 1.0).
PESI_RISCHIO = {
    "euristica": 0.25,          # regole cliniche su sonno / stress / RPE
    "random_forest": 0.30,      # pattern non lineari appresi dallo storico
    "logistic": 0.15,           # relazione lineare, trasparente
    "cluster": 0.15,            # quanto è rischioso storicamente il tuo "tipo" di allenamento odierno
    "trend_fisiologico": 0.15,  # anomalie recenti di FC rispetto all'atteso (Linear Regression)
}


def _hash_df(df: pd.DataFrame, cols) -> int:
    """Firma del dataset, usata solo per invalidare la cache se i dati cambiano."""
    return int(pd.util.hash_pandas_object(df[cols]).sum())


# =====================================================================
# 1) CLASSIFICAZIONE — Random Forest + Logistic Regression
# =====================================================================
@st.cache_resource(show_spinner=False)
def _addestra_classificazione(df_hash: int, df: pd.DataFrame):
    X = df[FEATURE_COLS_CLASS].values
    y = df['Rischio Infortunio'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled, y)

    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_scaled, y)

    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos > 0 and n_neg > 0:
        cv = StratifiedKFold(n_splits=max(2, min(5, n_pos, n_neg)), shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_rf = cross_val_predict(rf_model, X_scaled, y, cv=cv, method='predict')
    y_proba_rf = cross_val_predict(rf_model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_proba_log = cross_val_predict(log_model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred_log = (y_proba_log >= 0.5).astype(int)

    fpr_rf, tpr_rf, _ = roc_curve(y, y_proba_rf)
    fpr_log, tpr_log, _ = roc_curve(y, y_proba_log)

    metriche = {
        "rf": {
            "accuratezza": (y_pred_rf == y).mean() * 100,
            "precisione": precision_score(y, y_pred_rf, zero_division=0) * 100,
            "sensibilita": recall_score(y, y_pred_rf, zero_division=0) * 100,
            "f1": f1_score(y, y_pred_rf, zero_division=0) * 100,
            "auc": auc(fpr_rf, tpr_rf),
        },
        "log": {
            "accuratezza": (y_pred_log == y).mean() * 100,
            "precisione": precision_score(y, y_pred_log, zero_division=0) * 100,
            "sensibilita": recall_score(y, y_pred_log, zero_division=0) * 100,
            "f1": f1_score(y, y_pred_log, zero_division=0) * 100,
            "auc": auc(fpr_log, tpr_log),
        },
    }

    return {
        "rf_model": rf_model, "log_model": log_model, "scaler": scaler,
        "y_pred_rf": y_pred_rf, "y_proba_rf": y_proba_rf,
        "y_pred_log": y_pred_log, "y_proba_log": y_proba_log,
        "metriche": metriche,
    }


def get_bundle_classificazione(df: pd.DataFrame):
    return _addestra_classificazione(_hash_df(df, FEATURE_COLS_CLASS + ['Rischio Infortunio']), df)


# =====================================================================
# 2) REGRESSIONE — Linear Regression sulla FC Media
# =====================================================================
@st.cache_resource(show_spinner=False)
def _addestra_regressione(df_hash: int, df: pd.DataFrame):
    X = df[FEATURE_COLS_REG]
    y = df['FC Media']

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    fc_predetta = lr_model.predict(X)
    residui = y.values - fc_predetta

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(lr_model, X, y, cv=cv)
    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)

    return {
        "lr_model": lr_model, "fc_predetta": fc_predetta, "residui": residui,
        "r2": r2, "mae": mae,
    }


def get_bundle_regressione(df: pd.DataFrame):
    return _addestra_regressione(_hash_df(df, FEATURE_COLS_REG + ['FC Media']), df)


def trend_residuo_recente(reg_bundle: dict, giorni: int = 14) -> float:
    """
    Media dei residui (FC reale - FC prevista) negli ultimi N giorni.
    Un valore positivo e alto indica che il cuore lavora più del previsto
    rispetto a ritmo/clima: un segnale di affaticamento non spiegato
    dallo sforzo esterno.
    """
    residui = reg_bundle["residui"]
    if len(residui) == 0:
        return 0.0
    ultimi = residui[-giorni:] if len(residui) >= giorni else residui
    return float(np.mean(ultimi))


# =====================================================================
# 3) CLUSTERING — K-Means sui profili di allenamento
# =====================================================================
@st.cache_resource(show_spinner=False)
def _addestra_cluster(df_hash: int, df: pd.DataFrame, n_clusters: int = 3):
    X_raw = df[FEATURE_COLS_CLUST]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_id = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, cluster_id) if len(set(cluster_id)) > 1 else 0.0

    df_tmp = df.copy()
    df_tmp['Cluster_ID'] = cluster_id
    # % di giorni a rischio infortunio storicamente osservata in ciascun
    # cluster: dice "il tipo di allenamento di oggi, in passato, quanto
    # spesso ha portato a un giorno a rischio?"
    rischio_per_cluster = df_tmp.groupby('Cluster_ID')['Rischio Infortunio'].mean() * 100

    return {
        "km_model": km, "scaler": scaler, "cluster_id": cluster_id,
        "silhouette": sil, "rischio_per_cluster": rischio_per_cluster,
    }


def get_bundle_cluster(df: pd.DataFrame, n_clusters: int = 3):
    return _addestra_cluster(_hash_df(df, FEATURE_COLS_CLUST), df, n_clusters)


def rischio_storico_del_cluster(cluster_bundle: dict, distanza: float, fc_media: float) -> dict:
    """Assegna lo scenario di oggi al cluster più vicino e restituisce
    la percentuale storica di giorni a rischio osservata in quel cluster."""
    x_scaled = cluster_bundle["scaler"].transform([[distanza, fc_media]])
    cluster_oggi = int(cluster_bundle["km_model"].predict(x_scaled)[0])
    rischio_pct = float(cluster_bundle["rischio_per_cluster"].get(cluster_oggi, 0.0))
    return {"cluster_oggi": cluster_oggi + 1, "rischio_storico_cluster": rischio_pct}


# =====================================================================
# 4) STIMA PER OGGI — interroga classificazione (RF + Logistic)
# =====================================================================
def stima_rischio_oggi(class_bundle: dict, distanza: float, ore_sonno: float,
                        stress: float, rpe: float, fc_media: float = None) -> dict:
    """Random Forest + Logistic Regression sullo scenario di oggi
    (stessa logica usata dal Simulatore What-If della pagina ML)."""
    if fc_media is None:
        fc_media = 100 + rpe * 10  # proxy dichiarato, coerente col simulatore ML

    x = np.array([[distanza, ore_sonno, stress, fc_media, rpe]])
    x_scaled = class_bundle["scaler"].transform(x)

    prob_rf = class_bundle["rf_model"].predict_proba(x_scaled)[0][1] * 100
    prob_log = class_bundle["log_model"].predict_proba(x_scaled)[0][1] * 100

    contributi_rf = sorted(zip(FEATURE_LABELS_CLASS, class_bundle["rf_model"].feature_importances_),
                            key=lambda t: t[1], reverse=True)
    coefs_log = sorted(zip(FEATURE_LABELS_CLASS, class_bundle["log_model"].coef_[0]),
                        key=lambda t: abs(t[1]), reverse=True)

    return {
        "fc_media_stimata": fc_media,
        "probabilita_rf": prob_rf,
        "probabilita_log": prob_log,
        "fattore_principale_rf": contributi_rf[0][0],
        "fattore_principale_log": coefs_log[0][0],
        "verso_log": "aumenta" if coefs_log[0][1] > 0 else "riduce",
    }


# =====================================================================
# 5) ANALISI COMPLETA — combina TUTTI i modelli per il Consiglio Finale
# =====================================================================
def analisi_completa_oggi(df: pd.DataFrame, distanza: float, ore_sonno: float,
                           stress: float, rpe: float) -> dict:
    """
    Interroga TUTTI i modelli ML addestrati sullo storico (classificazione,
    clustering, regressione) per lo scenario di oggi e restituisce le
    componenti pronte per essere combinate in un punteggio di rischio
    unico nella pagina "Consiglio Finale".
    """
    class_bundle = get_bundle_classificazione(df)
    cluster_bundle = get_bundle_cluster(df)
    reg_bundle = get_bundle_regressione(df)

    stima_class = stima_rischio_oggi(class_bundle, distanza, ore_sonno, stress, rpe)
    fc_oggi = stima_class["fc_media_stimata"]

    stima_cluster = rischio_storico_del_cluster(cluster_bundle, distanza, fc_oggi)
    residuo_recente = trend_residuo_recente(reg_bundle)
    # Il residuo (in bpm) viene convertito in un punteggio 0-100: circa 12
    # bpm sopra l'atteso in modo sistematico equivale già a un segnale forte.
    punteggio_residuo = float(np.clip(residuo_recente * 8, 0, 100))

    componenti = {
        "random_forest": stima_class["probabilita_rf"],
        "logistic": stima_class["probabilita_log"],
        "cluster": stima_cluster["rischio_storico_cluster"],
        "trend_fisiologico": punteggio_residuo,
    }

    return {
        "componenti": componenti,
        "fattore_principale_rf": stima_class["fattore_principale_rf"],
        "fattore_principale_log": stima_class["fattore_principale_log"],
        "verso_log": stima_class["verso_log"],
        "cluster_oggi": stima_cluster["cluster_oggi"],
        "residuo_recente_bpm": residuo_recente,
        "auc_rf": class_bundle["metriche"]["rf"]["auc"],
        "auc_log": class_bundle["metriche"]["log"]["auc"],
        "r2_regressione": reg_bundle["r2"],
        "silhouette_cluster": cluster_bundle["silhouette"],
    }


def punteggio_rischio_combinato(componenti: dict, risk_score_euristico: float) -> float:
    """Media pesata di euristica clinica + segnali di tutti i modelli ML (PESI_RISCHIO)."""
    tutti = {"euristica": risk_score_euristico, **componenti}
    totale = sum(tutti[k] * PESI_RISCHIO[k] for k in PESI_RISCHIO)
    return round(min(100, max(0, totale)))
