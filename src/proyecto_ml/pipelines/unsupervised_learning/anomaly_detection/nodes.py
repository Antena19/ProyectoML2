import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def _numeric(df: pd.DataFrame) -> np.ndarray:
    X = df.select_dtypes(include=[np.number]).dropna()
    return StandardScaler().fit_transform(X.values)

def run_isolation_forest(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xs = _numeric(dataset_con_features_temporales)
    p = params.get("no_supervisado", {}).get("anomaly", {}).get("isoforest", {})
    model = IsolationForest(n_estimators=p.get("n_estimators", 200), contamination=p.get("contamination", "auto"), random_state=p.get("random_state", 42))
    model.fit(Xs)
    scores = model.score_samples(Xs)
    labels = model.predict(Xs)
    return pd.DataFrame({"score": scores}), pd.DataFrame({"label": labels})

def run_lof(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xs = _numeric(dataset_con_features_temporales)
    p = params.get("no_supervisado", {}).get("anomaly", {}).get("lof", {})
    model = LocalOutlierFactor(n_neighbors=p.get("n_neighbors", 20), contamination=p.get("contamination", "auto"))
    labels = model.fit_predict(Xs)
    scores = model.negative_outlier_factor_
    return pd.DataFrame({"score": scores}), pd.DataFrame({"label": labels})

def run_oneclass(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xs = _numeric(dataset_con_features_temporales)
    p = params.get("no_supervisado", {}).get("anomaly", {}).get("oneclass", {})
    model = OneClassSVM(kernel=p.get("kernel", "rbf"), gamma=p.get("gamma", "scale"), nu=p.get("nu", 0.1))
    labels = model.fit_predict(Xs)
    scores = model.decision_function(Xs)
    return pd.DataFrame({"score": scores}), pd.DataFrame({"label": labels})