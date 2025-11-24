FROM apache/airflow:2.10.3

USER root
RUN apt-get update && apt-get install -y git python3-venv && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt /tmp/requirements_docker.txt

USER airflow
RUN python -m venv /opt/airflow/.venvs/tools \
 && /opt/airflow/.venvs/tools/bin/pip install --upgrade pip setuptools wheel \
 && /opt/airflow/.venvs/tools/bin/pip install "numpy>=2.0.0" "pandas>=2.2.0" "scikit-learn>=1.4.0" "scipy>=1.11.0" "matplotlib>=3.8.0" "PyYAML>=6.0" \
 && /opt/airflow/.venvs/tools/bin/pip install "kedro==1.0.0" "kedro-datasets[pandas]>=1.7.0" "dvc[gdrive]==3.41.0" \
 && /opt/airflow/.venvs/tools/bin/pip install "seaborn>=0.12.0" "plotly>=5.0.0" "umap-learn>=0.5.0" "hdbscan>=0.8.0" \
    "mlxtend>=0.22.0" "pyod>=1.1.0" "shap>=0.42.0"