import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage


def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).dropna()


def run_kmeans(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    grid = [k for k in params["no_supervisado"]["kmeans"]["n_clusters_grid"] if k <= max(2, Xs.shape[0] - 1)]
    random_state = params["no_supervisado"]["kmeans"].get("random_state", 42)
    max_iter = params["no_supervisado"]["kmeans"].get("max_iter", 300)
    metrics_rows = []
    elbow_rows = []
    best_labels = None
    best_sil = -np.inf
    for k in grid:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto", max_iter=max_iter)
        labels = km.fit_predict(Xs)
        wcss = km.inertia_
        valid = 1 < len(set(labels)) < Xs.shape[0]
        sil = silhouette_score(Xs, labels) if valid else -1
        dbi = davies_bouldin_score(Xs, labels) if valid else np.inf
        chi = calinski_harabasz_score(Xs, labels) if valid else -1
        metrics_rows.append({"n_clusters": k, "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi})
        elbow_rows.append({"n_clusters": k, "wcss": wcss})
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
    labels_df = pd.DataFrame({"cluster": best_labels})
    metrics_df = pd.DataFrame(metrics_rows)
    elbow_df = pd.DataFrame(elbow_rows)
    return labels_df, metrics_df, elbow_df


def run_dbscan(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["dbscan"]
    model = DBSCAN(eps=p["eps"], min_samples=p["min_samples"], metric=p.get("metric", "euclidean"))
    labels = model.fit_predict(Xs)
    unique = set(labels)
    if len(unique) > 1 and not (len(unique) == 1 and (-1 in unique)):
        valid = 1 < len(unique) < Xs.shape[0]
        if valid:
            sil = silhouette_score(Xs, labels)
            dbi = davies_bouldin_score(Xs, labels)
            chi = calinski_harabasz_score(Xs, labels)
        else:
            sil, dbi, chi = -1, np.inf, -1
    else:
        sil, dbi, chi = -1, np.inf, -1
    labels_df = pd.DataFrame({"cluster": labels})
    metrics_df = pd.DataFrame([{"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}])
    return labels_df, metrics_df


def run_hierarchical(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["hierarchical"]
    n_clust = min(p["n_clusters"], max(2, Xs.shape[0] - 1))
    model = AgglomerativeClustering(n_clusters=n_clust, linkage=p["linkage"], metric=p.get("affinity", "euclidean"))
    labels = model.fit_predict(Xs)
    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else -1
    dbi = davies_bouldin_score(Xs, labels) if len(set(labels)) > 1 else np.inf
    chi = calinski_harabasz_score(Xs, labels) if len(set(labels)) > 1 else -1
    Z = linkage(Xs, method=p["linkage"], metric=p.get("affinity", "euclidean"))
    dendro_obj = {"linkage": Z}
    labels_df = pd.DataFrame({"cluster": labels})
    metrics_df = pd.DataFrame([{"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}])
    return labels_df, metrics_df, dendro_obj


def run_gmm(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["gmm"]
    best_labels = None
    best_sil = -np.inf
    metrics_rows = []
    comp_grid = [k for k in p["n_components_grid"] if k <= max(2, Xs.shape[0] - 1)]
    for k in comp_grid:
        gm = GaussianMixture(n_components=k, covariance_type=p.get("covariance_type", "full"), random_state=p.get("random_state", 42))
        gm.fit(Xs)
        labels = gm.predict(Xs)
        valid = 1 < len(set(labels)) < Xs.shape[0]
        sil = silhouette_score(Xs, labels) if valid else -1
        dbi = davies_bouldin_score(Xs, labels) if valid else np.inf
        chi = calinski_harabasz_score(Xs, labels) if valid else -1
        metrics_rows.append({"n_components": k, "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi})
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
    labels_df = pd.DataFrame({"cluster": best_labels})
    metrics_df = pd.DataFrame(metrics_rows)
    return labels_df, metrics_df