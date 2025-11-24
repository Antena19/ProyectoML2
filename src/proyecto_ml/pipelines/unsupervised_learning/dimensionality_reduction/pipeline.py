from kedro.pipeline import Pipeline, node
from .nodes import run_pca, run_tsne, run_umap, run_tsne3d, run_truncated_svd


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(run_pca, ["dataset_con_features_temporales", "parameters"], ["embeddings_pca", "pca_varianza_explicada", "pca_loadings"], name="run_pca"),
            node(run_tsne, ["dataset_con_features_temporales", "parameters"], "embeddings_tsne", name="run_tsne"),
            node(run_umap, ["dataset_con_features_temporales", "parameters"], "embeddings_umap", name="run_umap"),
            node(run_tsne3d, ["dataset_con_features_temporales", "parameters"], "embeddings_tsne3d", name="run_tsne3d"),
            node(run_truncated_svd, ["dataset_con_features_temporales", "parameters"], ["embeddings_svd", "svd_varianza_explicada"], name="run_truncated_svd"),
        ],
        tags=["no_supervisado", "dimensionality_reduction"],
    )