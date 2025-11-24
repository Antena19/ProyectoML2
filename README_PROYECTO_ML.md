# Proyecto ML — Predicción y Análisis Exploratorio de Defunciones

## Resumen Ejecutivo
- Objetivo: entender y predecir defunciones anuales en Chile y descubrir patrones.
- Enfoque: pipelines de ingeniería, ciencia de datos y modelado (Kedro), orquestados por Airflow, versionados con DVC.
- Resultados:
  - Supervisado (regresión): Regresión Lineal promovida a producción (`models/production/model.pkl`) con métricas en `data/07_model_output/`.
  - No supervisado: clustering (K-Means, Jerárquico, GMM, DBSCAN), reducción (PCA, t-SNE, UMAP, SVD), anomalías (IsolationForest/LOF/One-Class), reglas de asociación (Apriori/FP-Growth).
  - Notebooks con narrativa y visualizaciones interactivas (`Presentacion_Pauta_Visual.ipynb`, `06_final_analysis.ipynb`).

## Arquitectura y Pipelines
- Ingeniería de Datos: limpieza y estandarización.
- Ciencia de Datos: integración, features temporales, escalado y codificación.
- Modelado: entrenamiento, evaluación, selección y promoción a producción.
- No supervisado: clustering, reducción dimensional, anomalías, asociación.
- Reportes: visualizaciones y resúmenes.

## Ejecución de Pipelines
- Orden recomendado:
  - `kedro run --pipeline=ingenieria_datos`
  - `kedro run --pipeline=ciencia_datos`
  - `kedro run --pipeline=reportes`
  - `kedro run --pipeline=modelado`
- No supervisado con script:
  - `python scripts/run_unsupervised.py --step kmeans|dbscan|hier|gmm|pca|tsne|umap|tsne3d|svd|isoforest|lof|oneclass|apriori|fpgrowth|all`

## Artefactos y Resultados
- Supervisado:
  - Métricas y comparaciones: `data/07_model_output/metricas_modelos.csv`, `metricas_resumen.csv`, `comparacion_y_real_vs_pred.csv`.
  - Producción: `models/production/model.pkl`, `fig_prediccion.png`, `prediccion_actual.csv`.
- No supervisado:
  - Clustering: `data/07_model_output/clustering/*` (labels, métricas, elbow, dendrograma).
  - Reducción: `data/07_model_output/reduction/*` (embeddings, varianza, loadings).
  - Anomalías: `data/07_model_output/anomaly/*` (scores y labels).
  - Asociación: `data/07_model_output/association/*` (rules con support/confidence/lift).

## DVC — Versionado 100%
- Etapas declaradas en `dvc.yaml` para todos los pipelines (supervisado y no supervisado).
- `metrics:` marcado en etapas clave (`metricas_modelos.csv`, `elbow_kmeans.csv`, `pca_varianza_explicada.csv`, `svd_varianza_explicada.csv`).
- Directorios versionados: `data/03_primary.dvc`, `models/production.dvc`.
- Remoto configurado en `.dvc/config`.

## Airflow — Orquestación
- Levantar: `docker compose -f docker\docker-compose.yml up -d`
- UI: `http://localhost:8080` (admin/admin).
- Ejecutar DAG principal: `ml_modelado_kedro_dvc` (UI o CLI).
- Artefactos versionados y notificaciones opcionales (Slack).

## Notebooks — Narrativa Profesional
- `notebooks/Presentacion_Pauta_Visual.ipynb`: historia del proyecto, visualizaciones clave (PCA, biplot, t-SNE/UMAP, clustering), explicaciones simples.
- `notebooks/06_final_analysis.ipynb`: resultados finales, comparativas baseline vs con clusters, análisis de patrones por cluster, anomalías y asociación.
- Interactividad: usa `plotly` si está disponible; fallback a `matplotlib`.

## Docstrings — Cobertura
- Módulos con docstrings: pipelines y nodos de modelado, clustering, reducción, anomalías, asociación, ciencia de datos, ingeniería de datos, reportes y clasificación.

# Navegar a la raiz del proyecto
cd proyecto-ml

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (PowerShell)
.\.venv\Scripts\Activate

# *Desactivar
deactivate

# Instalar dependencias desde requirements.txt
uv pip install -r requirements.txt

# Abrir sesión interactiva de Kedro
kedro ipython

# 1 Importar Kedro y crear la sesión del proyecto
from kedro.framework.session import KedroSession

# Crear sesión del proyecto y cargar el catálogo
session = KedroSession.create()
catalog = session.load_context().catalog

# 2 Cargar los datasets crudos (01_raw)
datos_historicos = catalog.load("datos_historicos_nacimientos_defunciones")
datos_filtrados_defunciones = catalog.load("datos_filtrados_defunciones")
nacimientos_por_sexo = catalog.load("nacimientos_defunciones_por_sexo")
nacimientos_por_edad_madre = catalog.load("nacimientos_por_edad_madre")
defunciones_por_edad_fallecido = catalog.load("defunciones_por_edad_fallecido")

# 3 Mostrar un vistazo rápido de los CSV principales
print("=== Datos Históricos de Nacimientos y Defunciones ===")
print(datos_historicos.head())

print("\n=== Datos Filtrados de Defunciones (2014-2023) ===")
print(datos_filtrados_defunciones.head())

print("\n=== Nacimientos y Defunciones por Sexo ===")
print(nacimientos_por_sexo.head())

print("\n=== Nacimientos por Edad de la Madre ===")
print(nacimientos_por_edad_madre.head())

print("\n=== Defunciones por Edad del Fallecido ===")
print(defunciones_por_edad_fallecido.head())

# Trabajar en Jupyter Notebook
pip install notebook

# Abrir Notebook
kedro jupyter notebook

# 1. Ejecutar pipeline en orden
kedro run --pipeline=ingenieria_datos

kedro run --pipeline=ciencia_datos

kedro run --pipeline=reportes

### Visualizar Pipelines

# Abrir Kedro Viz en el navegador
kedro viz

# Ejecutar todos los tests
pytest

# Ejecutar tests específicos
pytest tests/pipelines/test_ingenieria_datos.py
pytest tests/pipelines/test_ciencia_datos.py
pytest tests/pipelines/test_reportes.py

# Ejecutar con verbose
pytest -v

# Ejecutar con cobertura
pytest --cov=src/proyecto_ml

## Airflow - Guía rápida (Windows)

### 1) Pre-requisitos

- Docker Desktop instalado y ejecutándose
- Webhook de Slack (opcional) para notificaciones

### 2) Variables y secretos

```powershell
setx SLACK_WEBHOOK_URL "https://hooks.slack.com/services/XXXXXXXX/XXXXXXXX/XXXXXXXXXXXXXXXX"
setx AIRFLOW_BASE_URL "http://localhost:8080"
```

Ubica los secretos de DVC en `c:\ProyectoML2\secrets`:

- `proyecto-ml-479118-c0e8199326a2.json`
- `gdrive-user-creds.json`
- `client_secret_*.json` (opcional)

### 3) Levantar Airflow

```powershell
docker compose -f docker\docker-compose.yml up -d
```

UI: `http://localhost:8080` (usuario `admin`, contraseña `admin`).

### 4) Despausar y ejecutar el DAG principal

Desde la UI: DAGs → `ml_modelado_kedro_dvc` → Unpause → Trigger.

Vía CLI:

```powershell
docker compose -f docker\docker-compose.yml exec airflow bash -lc "airflow dags list"
docker compose -f docker\docker-compose.yml exec airflow bash -lc "airflow dags unpause ml_modelado_kedro_dvc"
docker compose -f docker\docker-compose.yml exec airflow bash -lc "airflow dags trigger ml_modelado_kedro_dvc"
```

### 5) Artefactos y versionado

- Supervisado:
  - `models/production/model.pkl`, `fig_prediccion.png`, `prediccion_actual.csv`
- No supervisado:
  - `data/07_model_output/clustering/*`, `data/07_model_output/reduction/*`

Los artefactos se versionan automáticamente con DVC y se suben al remoto configurado.

### 6) Notificaciones Slack

Si `SLACK_WEBHOOK_URL` está definido:

- Éxito: resumen con rutas y métricas
- Falla: aviso con enlace a la UI si configuraste `AIRFLOW_BASE_URL`