from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="ml_pipeline_master",
    start_date=datetime(2025, 11, 23),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:
    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && kedro run --pipeline ingenieria_datos",
    )

    supervised_classification = BashOperator(
        task_id="supervised_classification",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && kedro run --pipeline clasificacion",
    )

    unsup_kmeans = BashOperator(
        task_id="unsup_kmeans",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step kmeans",
    )

    unsup_dbscan = BashOperator(
        task_id="unsup_dbscan",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step dbscan",
    )

    unsup_hier = BashOperator(
        task_id="unsup_hier",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step hier",
    )

    unsup_gmm = BashOperator(
        task_id="unsup_gmm",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step gmm",
    )

    red_pca = BashOperator(
        task_id="red_pca",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step pca",
    )

    red_tsne = BashOperator(
        task_id="red_tsne",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step tsne",
    )

    red_umap = BashOperator(
        task_id="red_umap",
        bash_command="source /opt/airflow/.venvs/tools/bin/activate && cd /opt/airflow/proyecto-ml && python scripts/run_unsupervised.py --step umap",
    )

    data_engineering >> supervised_classification
    supervised_classification >> [unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm]
    for r in [red_pca, red_tsne, red_umap]:
        [unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm] >> r