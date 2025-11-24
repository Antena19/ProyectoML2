from kedro.pipeline import Pipeline, node
from .nodes import run_apriori, run_fpgrowth

def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(run_apriori, ["dataset_con_features_temporales", "parameters"], "assoc_apriori", name="run_apriori"),
            node(run_fpgrowth, ["dataset_con_features_temporales", "parameters"], "assoc_fpgrowth", name="run_fpgrowth"),
        ],
        tags=["no_supervisado", "association_rules"],
    )