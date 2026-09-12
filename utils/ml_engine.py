import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

FEATURE_COLS = ['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']
FEATURE_LABELS = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']


def _hash_df(df: pd.DataFrame) -> int:
    """Firma del dataset, usata solo per invalidare la cache se i dati cambiano."""
    return int(pd.util.hash_pandas_object(df[FEATURE_COLS + ['Rischio Infortunio']]).sum())


@st.cache_resource(show_spinner=False)
def _addestra(df_hash: int, df: pd.DataFrame):
    X = df[FEATURE_COLS].values
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

    fpr, tpr, _ = roc_curve(y, y_proba_rf)
    metriche = {
        "accuratezza": (y_pred_rf == y).mean() * 100,
        "precisione": precision_score(y, y_pred_rf, zero_division=0) * 100,
        "sensibilita": recall_score(y, y_pred_rf, zero_division=0) * 100,
        "f1": f1_score(y, y_pred_rf, zero_division=0) * 100,
        "auc": auc(fpr, tpr),
    }

    return {
        "rf_model": rf_model, "log_model": log_model, "scaler": scaler,
        "y_pred_rf": y_pred_rf, "y_proba_rf": y_proba_rf, "y_proba_log": y_proba_log,
        "metriche": metriche,
    }


def get_bundle(df: pd.DataFrame):
    """Punto di ingresso unico: entrambe le pagine chiamano questa funzione."""
    return _addestra(_hash_df(df), df)


def stima_rischio_oggi(bundle, distanza, ore_sonno, stress, rpe, fc_media=None):
    """Stessa logica del Simulatore What-If, riusabile ovunque."""
    if fc_media is None:
        fc_media = 100 + rpe * 10  # proxy dichiarato nel simulatore ML

    x = np.array([[distanza, ore_sonno, stress, fc_media, rpe]])
    x_scaled = bundle["scaler"].transform(x)
    prob = bundle["rf_model"].predict_proba(x_scaled)[0][1] * 100

    contributi = sorted(zip(FEATURE_LABELS, bundle["rf_model"].feature_importances_),
                         key=lambda t: t[1], reverse=True)

    return {"probabilita_rf": prob, "fattore_principale": contributi[0][0], "contributi": contributi}
