from kedro.pipeline import Pipeline, node
from .nodes import run_isolation_forest, run_lof, run_oneclass

def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(run_isolation_forest, ["dataset_con_features_temporales", "parameters"], ["anomaly_isoforest_scores", "anomaly_isoforest_labels"], name="run_isolation_forest"),
            node(run_lof, ["dataset_con_features_temporales", "parameters"], ["anomaly_lof_scores", "anomaly_lof_labels"], name="run_lof"),
            node(run_oneclass, ["dataset_con_features_temporales", "parameters"], ["anomaly_oneclass_scores", "anomaly_oneclass_labels"], name="run_oneclass"),
        ],
        tags=["no_supervisado", "anomaly_detection"],
    )