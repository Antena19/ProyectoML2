import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import TruncatedSVD


def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).dropna()


def run_pca(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = _select_numeric_features(dataset_con_features_temporales)
    cols = list(X.columns)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["pca"]
    n_components = p.get("n_components", 2)
    model = PCA(n_components=n_components, whiten=p.get("whiten", False))
    emb = model.fit_transform(Xs)
    df_emb = pd.DataFrame(emb, columns=[f"pc{i+1}" for i in range(emb.shape[1])])
    var_ratio = pd.DataFrame({"component": [f"pc{i+1}" for i in range(model.explained_variance_ratio_.shape[0])], "explained_variance_ratio": model.explained_variance_ratio_})
    load = (model.components_.T * np.sqrt(model.explained_variance_))
    load_df = pd.DataFrame(load, columns=[f"pc{i+1}" for i in range(model.components_.shape[0])])
    load_df.insert(0, "feature", cols[: load_df.shape[0]])
    fig_path = Path("data/07_model_output/reduction/pca_biplot.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5))
    if df_emb.shape[1] >= 2 and load_df.shape[1] >= 3:
        plt.scatter(df_emb.iloc[:,0], df_emb.iloc[:,1], s=10, alpha=0.6)
        for _, r in load_df.iterrows():
            x = r["pc1"]
            y = r["pc2"]
            plt.arrow(0, 0, x, y, color="red", alpha=0.5, head_width=0.02, length_includes_head=True)
            plt.text(x, y, str(r["feature"]), fontsize=8)
        plt.xlabel("pc1"); plt.ylabel("pc2")
        plt.title("PCA Biplot")
        plt.grid(True)
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    return df_emb, var_ratio, load_df


def run_tsne(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["tsne"]
    perplex = p.get("perplexity", 30)
    perplex = min(perplex, max(2, Xs.shape[0] - 1))
    model = TSNE(n_components=p.get("n_components", 2), perplexity=perplex, learning_rate=p.get("learning_rate", 200))
    emb = model.fit_transform(Xs)
    df_emb = pd.DataFrame(emb, columns=[f"tsne{i+1}" for i in range(emb.shape[1])])
    return df_emb


def run_umap(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["umap"]
    model = umap.UMAP(n_components=p.get("n_components", 2), n_neighbors=p.get("n_neighbors", 15), min_dist=p.get("min_dist", 0.1), random_state=42)
    emb = model.fit_transform(Xs)
    df_emb = pd.DataFrame(emb, columns=[f"umap{i+1}" for i in range(emb.shape[1])])
    return df_emb


def run_tsne3d(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"]["tsne"]
    perplex = p.get("perplexity", 30)
    perplex = min(perplex, max(2, Xs.shape[0] - 1))
    model = TSNE(n_components=3, perplexity=perplex, learning_rate=p.get("learning_rate", 200))
    emb = model.fit_transform(Xs)
    df_emb = pd.DataFrame(emb, columns=["tsne1", "tsne2", "tsne3"])
    return df_emb


def run_truncated_svd(dataset_con_features_temporales: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = _select_numeric_features(dataset_con_features_temporales)
    Xs = StandardScaler().fit_transform(X.values)
    p = params["no_supervisado"].get("svd", {})
    n_components = p.get("n_components", 2)
    model = TruncatedSVD(n_components=n_components, random_state=42)
    emb = model.fit_transform(Xs)
    df_emb = pd.DataFrame(emb, columns=[f"svd{i+1}" for i in range(emb.shape[1])])
    var_ratio = pd.DataFrame({"component": [f"svd{i+1}" for i in range(model.explained_variance_ratio_.shape[0])], "explained_variance_ratio": model.explained_variance_ratio_})
    return df_emb, var_ratio
"""
Nodos de reducción de dimensionalidad:
- PCA: embeddings, varianza explicada y loadings para interpretar variables.
- t-SNE (2D/3D): estructura local y microgrupos.
- UMAP (2D): relaciones locales y globales.
- SVD truncado: baseline lineal para alta dimensionalidad.

Salidas (catálogo `data/07_model_output/reduction/`):
- `embeddings_pca.csv`, `pca_varianza_explicada.csv`, `pca_loadings.csv`.
- `embeddings_tsne.csv`, `embeddings_tsne3d.csv`, `embeddings_umap.csv`.
- `embeddings_svd.csv`, `svd_varianza_explicada.csv`.
"""