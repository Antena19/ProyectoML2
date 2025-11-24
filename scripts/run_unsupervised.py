import argparse
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["prep", "kmeans", "dbscan", "hier", "gmm", "pca", "tsne", "umap", "tsne3d", "svd", "all"], default="all")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    bootstrap_project(project_path)
    import sys
    sys.path.append(str(project_path))

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog

        import yaml
        params_path = project_path / "conf" / "base" / "parameters.yml"
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        if args.step == "prep":
            session.run()
            return

        from proyecto_ml.pipelines.unsupervised_learning.clustering.nodes import (
            run_kmeans,
            run_dbscan,
            run_hierarchical,
            run_gmm,
        )
        from proyecto_ml.pipelines.unsupervised_learning.dimensionality_reduction.nodes import (
            run_pca,
            run_tsne,
            run_umap,
            run_tsne3d,
            run_truncated_svd,
        )

        X = catalog.load("dataset_con_features_temporales")

        if args.step == "kmeans":
            labels_df, metrics_df, elbow_df = run_kmeans(X, params)
            catalog.save("clust_kmeans_labels", labels_df)
            catalog.save("metrics_kmeans", metrics_df)
            catalog.save("elbow_kmeans", elbow_df)
        elif args.step == "dbscan":
            labels_df, metrics_df = run_dbscan(X, params)
            catalog.save("clust_dbscan_labels", labels_df)
            catalog.save("metrics_dbscan", metrics_df)
        elif args.step == "hier":
            labels_df, metrics_df, dendro_obj = run_hierarchical(X, params)
            catalog.save("clust_hier_labels", labels_df)
            catalog.save("metrics_hier", metrics_df)
            catalog.save("dendrogram_hier", dendro_obj)
        elif args.step == "gmm":
            labels_df, metrics_df = run_gmm(X, params)
            catalog.save("clust_gmm_labels", labels_df)
            catalog.save("metrics_gmm", metrics_df)
        elif args.step == "pca":
            emb, var, loadings = run_pca(X, params)
            catalog.save("embeddings_pca", emb)
            catalog.save("pca_varianza_explicada", var)
            catalog.save("pca_loadings", loadings)
        elif args.step == "tsne":
            emb = run_tsne(X, params)
            catalog.save("embeddings_tsne", emb)
        elif args.step == "tsne3d":
            emb = run_tsne3d(X, params)
            catalog.save("embeddings_tsne3d", emb)
        elif args.step == "umap":
            emb = run_umap(X, params)
            catalog.save("embeddings_umap", emb)
        elif args.step == "svd":
            emb, var = run_truncated_svd(X, params)
            catalog.save("embeddings_svd", emb)
            catalog.save("svd_varianza_explicada", var)
        elif args.step == "all":
            emb_pca, var_pca, loadings_pca = run_pca(X, params)
            catalog.save("embeddings_pca", emb_pca)
            catalog.save("pca_varianza_explicada", var_pca)
            catalog.save("pca_loadings", loadings_pca)

            emb_tsne = run_tsne(X, params)
            catalog.save("embeddings_tsne", emb_tsne)

            emb_tsne3d = run_tsne3d(X, params)
            catalog.save("embeddings_tsne3d", emb_tsne3d)

            emb_umap = run_umap(X, params)
            catalog.save("embeddings_umap", emb_umap)

            emb_svd, var_svd = run_truncated_svd(X, params)
            catalog.save("embeddings_svd", emb_svd)
            catalog.save("svd_varianza_explicada", var_svd)


if __name__ == "__main__":
    main()