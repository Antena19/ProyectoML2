# Arquitectura del proyecto

- Configuración en `conf/base` con `catalog.yml` y `parameters.yml`.
- Orquestación en Airflow con DAG `ml_pipeline_master.py`.
- Pipelines en `src/proyecto_ml/pipelines`, incluyendo `unsupervised_learning`.
- Artefactos versionados en `data/06_models`, `data/07_model_output`, `data/08_reporting`.