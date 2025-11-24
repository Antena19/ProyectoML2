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