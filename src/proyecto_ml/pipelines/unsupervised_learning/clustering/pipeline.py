from kedro.pipeline import Pipeline, node
from .nodes import run_kmeans, run_dbscan, run_hierarchical, run_gmm


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(run_kmeans, ["dataset_con_features_temporales", "parameters"], ["clust_kmeans_labels", "metrics_kmeans", "elbow_kmeans"], name="run_kmeans"),
            node(run_dbscan, ["dataset_con_features_temporales", "parameters"], ["clust_dbscan_labels", "metrics_dbscan"], name="run_dbscan"),
            node(run_hierarchical, ["dataset_con_features_temporales", "parameters"], ["clust_hier_labels", "metrics_hier", "dendrogram_hier"], name="run_hierarchical"),
            node(run_gmm, ["dataset_con_features_temporales", "parameters"], ["clust_gmm_labels", "metrics_gmm"], name="run_gmm"),
        ],
        tags=["no_supervisado", "clustering"],
    )
"""
Pipeline de Clustering (no supervisado).

Objetivo:
- Probar K-Means, DBSCAN, Jerárquico y GMM, medir su calidad y guardar
  etiquetas y métricas para análisis.

Salidas principales (catálogo):
- Etiquetas: `data/07_model_output/clustering/*_labels.csv`.
- Métricas: `data/07_model_output/clustering/metrics_*.csv`.
- Elbow (K-Means): `data/07_model_output/clustering/elbow_kmeans.csv`.
- Dendrograma (Jerárquico): `data/07_model_output/clustering/dendrogram_hier.pkl`.

Cómo interpretarlo:
- Silhouette alto y Davies-Bouldin bajo indican separación clara.
- Elbow sugiere K óptimo; el dendrograma ayuda a elegir cortes.
"""