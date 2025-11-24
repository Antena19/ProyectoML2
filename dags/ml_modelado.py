from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# ========== CONFIGURACIÓN ==========
DEFAULT_ENV = {
    "TZ": "America/Santiago",
    "HOME": "/home/airflow",
    "PYTHONUNBUFFERED": "1",
}

VENV_ACTIVATE = "source /opt/airflow/.venvs/tools/bin/activate"
REPO_DIR = "/opt/airflow/proyecto-ml"
GIT_USER = "Airflow Bot"
GIT_EMAIL = "airflow@tu-empresa.com"

# ========== DAG ==========
default_args = {
    "owner": "carolina",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_modelado_kedro_dvc",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["kedro", "dvc", "modelo", "ml"],
    description="Pipeline de entrenamiento y despliegue de modelos ML con Kedro y DVC",
) as dag:

    # ========== TASK 1 ==========
    validar_entorno = BashOperator(
        task_id="validar_entorno",
        bash_command=f"""
            set -e
            echo "=== Validando entorno ==="
            if [ ! -d "{REPO_DIR}" ]; then echo "❌ {REPO_DIR} no existe"; exit 1; fi
            if [ ! -d "{REPO_DIR}/.git" ]; then echo "❌ Falta .git"; exit 1; fi
            if [ ! -d "{REPO_DIR}/.dvc" ]; then echo "❌ Falta .dvc"; exit 1; fi
            echo "✅ Entorno validado correctamente"
        """,
        env=DEFAULT_ENV,
    )

    # ========== TASK 2 ==========
    actualizar_repo = BashOperator(
        task_id="actualizar_repo",
        bash_command=f"""
            set -e
            echo "=== Actualizando repositorio ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            git config --global --add safe.directory "{REPO_DIR}"
            git config --global user.name "{GIT_USER}"
            git config --global user.email "{GIT_EMAIL}"
            git fetch origin
            git pull --rebase origin main || git pull --rebase origin master || true
            dvc pull || echo "⚠️ DVC pull parcial"
            echo "✅ Repositorio actualizado"
        """,
        env=DEFAULT_ENV,
    )

    # ========== TASK 3 ==========
    ejecutar_kedro = BashOperator(
        task_id="ejecutar_kedro",
        bash_command=f"""
            set -e
            echo "=== Ejecutando pipeline Kedro ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            kedro run --only-missing-outputs
            echo "✅ Pipeline ejecutado correctamente"
        """,
        env=DEFAULT_ENV,
    )

    # ========== TASK 4 ==========
    validar_modelo = BashOperator(
        task_id="validar_modelo",
        bash_command=f"""
            set -e
            echo "=== Validando modelo ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            if [ ! -f "data/06_models/mejor_modelo.pkl" ]; then
                echo "❌ Modelo no encontrado"; exit 1; fi
            echo "✅ Modelo válido encontrado"
        """,
        env=DEFAULT_ENV,
    )

    # ========== TASK 5 ==========
    promover_modelo = BashOperator(
        task_id="promover_modelo",
        bash_command=f"""
            set -e
            echo "=== Promoviendo modelo ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            mkdir -p models/production
            if [ -f "models/production/model.pkl" ]; then
                TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                cp models/production/model.pkl "models/production/model_backup_$TIMESTAMP.pkl"
            fi
            cp data/06_models/mejor_modelo.pkl models/production/model.pkl
            echo "✅ Modelo promovido"
        """,
        env=DEFAULT_ENV,
    )

    unsup_kmeans = BashOperator(
        task_id="unsup_kmeans",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step kmeans
        """,
        env=DEFAULT_ENV,
    )

    unsup_dbscan = BashOperator(
        task_id="unsup_dbscan",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step dbscan
        """,
        env=DEFAULT_ENV,
    )

    unsup_hier = BashOperator(
        task_id="unsup_hier",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step hier
        """,
        env=DEFAULT_ENV,
    )

    unsup_gmm = BashOperator(
        task_id="unsup_gmm",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step gmm
        """,
        env=DEFAULT_ENV,
    )

    red_pca = BashOperator(
        task_id="red_pca",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step pca
        """,
        env=DEFAULT_ENV,
    )

    red_tsne = BashOperator(
        task_id="red_tsne",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step tsne
        """,
        env=DEFAULT_ENV,
    )

    red_umap = BashOperator(
        task_id="red_umap",
        bash_command=f"""
            set -e
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            python scripts/run_unsupervised.py --step umap
        """,
        env=DEFAULT_ENV,
    )

    # ========== NUEVO TASK: graficar_prediccion ==========
    graficar_prediccion = BashOperator(
    task_id="graficar_prediccion",
    bash_command="""
{% raw %}
set -e
echo "=== Generando gráfico ==="
source /opt/airflow/.venvs/tools/bin/activate
cd /opt/airflow/proyecto-ml

python - << 'PYCODE'
import os, pickle
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = "/opt/airflow/proyecto-ml"
DATA_CSV = os.path.join(REPO_DIR, "data/03_primary/dataset_unificado.csv")
MODEL_PKL = os.path.join(REPO_DIR, "models/production/model.pkl")
OUT_DIR   = os.path.join(REPO_DIR, "models/production")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Leer dataset completo
df = pd.read_csv(DATA_CSV, encoding="utf-8")

# 2) Preparar datos para el gráfico
df_plot = df.rename(columns={"año": "anio", "defunciones_totales": "defunciones"})
if {"anio", "defunciones"}.issubset(df_plot.columns):
    df_plot["anio"] = df_plot["anio"].astype(int)
    df_plot = df_plot.sort_values("anio")
    ultimos5 = df_plot.tail(5).copy()
else:
    raise ValueError("No se encontraron las columnas 'año' o 'defunciones_totales' en el dataset.")

# 3) Cargar modelo y preparar features
with open(MODEL_PKL, "rb") as f:
    modelo = pickle.load(f)

feature_cols = getattr(modelo, "feature_names_in_", None)

if feature_cols is not None:
    # Asegurar todas las columnas que el modelo espera
    faltantes = [c for c in feature_cols if c not in df.columns]
    for c in faltantes:
        df[c] = 0
    X_pred_full = df[feature_cols].copy()
else:
    excluir = {"defunciones_totales", "defunciones", "anio", "año"}
    X_cols = [c for c in df.columns if c not in excluir]
    X_pred_full = df[X_cols].copy()

# Convertir a numérico y manejar NaN
X_pred_full = X_pred_full.apply(pd.to_numeric, errors="coerce")

# Tomar la última fila COMPLETA si existe; si no, imputar (ffill -> bfill -> 0) y tomar la última
ultima_completa = X_pred_full.dropna().tail(1)
if not ultima_completa.empty:
    X_pred = ultima_completa
else:
    X_pred = X_pred_full.fillna(method="ffill").fillna(method="bfill").fillna(0).tail(1)

if X_pred.empty:
    raise ValueError("⚠️ No hay columnas válidas para predecir (X_pred está vacío).")

# 4) Predicción
anio_actual = datetime.now().year
y_pred = modelo.predict(X_pred)
pred = int(round(float(y_pred[0])))

# 5) Guardar CSV con la predicción
pd.DataFrame({"anio": [anio_actual], "defunciones_predichas": [pred]}).to_csv(
    os.path.join(OUT_DIR, "prediccion_actual.csv"), index=False
)

# 6) Graficar
plt.figure()
plt.plot(ultimos5["anio"], ultimos5["defunciones"], marker="o", label="Reales (últimos 5)")
plt.plot([anio_actual], [pred], marker="o", linestyle="--", label=f"Predicción {anio_actual}")
plt.title("Defunciones en Chile: últimos 5 años + predicción actual")
plt.xlabel("Año"); plt.ylabel("Defunciones")
plt.legend(); plt.grid(True)
out_png = os.path.join(OUT_DIR, "fig_prediccion.png")
plt.savefig(out_png, bbox_inches="tight")
print(f"✅ Gráfico guardado en: {out_png}")
PYCODE

{% endraw %}
    """,
    env=DEFAULT_ENV,
)

    # ========== TASK 6 ==========
    versionar_y_subir = BashOperator(
        task_id="versionar_y_subir",
        bash_command=f"""
            set -e
            echo "=== Versionando y subiendo ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            git config --global --add safe.directory "{REPO_DIR}"
            git add models/production.dvc .gitignore || true
            git commit -m "Actualiza modelo y gráfico" || true
            git push origin main || true
            dvc push || true
            echo "✅ Versionado completo"
        """,
        env=DEFAULT_ENV,
    )

    versionar_no_supervisado = BashOperator(
        task_id="versionar_no_supervisado",
        bash_command=f"""
            set -e
            echo "=== Versionando artefactos no supervisados ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            git config --global --add safe.directory "{REPO_DIR}"
            if [ -d "data/07_model_output/clustering" ]; then
                dvc add data/07_model_output/clustering || true
            else
                echo "⚠️ No se encontró directorio de clustering"
            fi
            if [ -d "data/07_model_output/reduction" ]; then
                dvc add data/07_model_output/reduction || true
            else
                echo "⚠️ No se encontró directorio de reducción"
            fi
            if [ -d "data/08_reporting" ]; then
                dvc add data/08_reporting || true
            else
                echo "⚠️ No se encontró directorio de reportes"
            fi
            if [ -f "data/07_model_output/metricas_modelos.csv" ]; then dvc add data/07_model_output/metricas_modelos.csv || true; fi
            if [ -f "data/07_model_output/metricas_resumen.csv" ]; then dvc add data/07_model_output/metricas_resumen.csv || true; fi
            if [ -f "data/07_model_output/comparacion_y_real_vs_pred.csv" ]; then dvc add data/07_model_output/comparacion_y_real_vs_pred.csv || true; fi
            if [ -f "data/07_model_output/probabilidades_raw.csv" ]; then dvc add data/07_model_output/probabilidades_raw.csv || true; fi
            if [ -f "data/07_model_output/probabilidades.csv" ]; then dvc add data/07_model_output/probabilidades.csv || true; fi
            if [ -f "data/07_model_output/importancias_features.csv" ]; then dvc add data/07_model_output/importancias_features.csv || true; fi
            git add data/07_model_output/*.dvc || true
            git commit -m "Versiona artefactos no supervisados (clustering y reducción)" || true
            git push origin main || true
            dvc push || true
            echo "✅ Versionado no supervisado completo"
        """,
        env=DEFAULT_ENV,
    )

    # ========== TASK 7 ==========
    notificar_exito = BashOperator(
        task_id="notificar_exito",
        bash_command=f"""
            echo "✅ Pipeline completado exitosamente"
            echo "📊 Modelo: {REPO_DIR}/models/production/model.pkl"
            echo "🖼️ Gráfico: {REPO_DIR}/models/production/fig_prediccion.png"
            echo "📄 CSV: {REPO_DIR}/models/production/prediccion_actual.csv"
        """,
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    notificar_slack_exito = BashOperator(
        task_id="notificar_slack_exito",
        bash_command="""
            set -e
            if [ -z "$SLACK_WEBHOOK_URL" ]; then echo "SLACK_WEBHOOK_URL no configurado"; exit 0; fi
            python - << 'PYCODE'
import json, os, urllib.request, csv
url = os.environ.get('SLACK_WEBHOOK_URL', '')
if not url:
    raise SystemExit(0)
repo = os.environ.get('REPO_DIR', '')
airflow_url = os.environ.get('AIRFLOW_BASE_URL', '')
modelo_path = f"{repo}/models/production/model.pkl"
grafico_path = f"{repo}/models/production/fig_prediccion.png"
csv_pred_path = f"{repo}/models/production/prediccion_actual.csv"
met_res_path = os.path.join(repo, "data/07_model_output/metricas_resumen.csv")
met_mod_path = os.path.join(repo, "data/07_model_output/metricas_modelos.csv")
clust_dir = os.path.join(repo, "data/07_model_output/clustering")
red_dir = os.path.join(repo, "data/07_model_output/reduction")
def count_files(d):
    try:
        return len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    except Exception:
        return 0
clust_count = count_files(clust_dir)
red_count = count_files(red_dir)
def read_head(path, n=3):
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(", ".join(row))
                if i+1 >= n:
                    break
            return "\n".join(rows)
    except Exception:
        return ""
met_res_head = read_head(met_res_path, 3)
met_mod_head = read_head(met_mod_path, 3)
blocks = [
    {"type": "section", "text": {"type": "mrkdwn", "text": "✅ Pipeline completado exitosamente"}},
    {"type": "section", "fields": [
        {"type": "mrkdwn", "text": f"*Modelo:* \n{modelo_path}"},
        {"type": "mrkdwn", "text": f"*Gráfico:* \n{grafico_path}"},
        {"type": "mrkdwn", "text": f"*CSV:* \n{csv_pred_path}"},
        {"type": "mrkdwn", "text": f"*Clustering archivos:* \n{clust_count}"},
        {"type": "mrkdwn", "text": f"*Reducción archivos:* \n{red_count}"}
    ]}
]
if met_res_head:
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Métricas resumen (top):*\n```{met_res_head}```"}})
if met_mod_head:
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Métricas modelos (top):*\n```{met_mod_head}```"}})
if airflow_url:
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Airflow:* {airflow_url}"}})
payload = {"text": "Pipeline ML completado", "blocks": blocks}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    resp.read()
PYCODE
        """,
        env={**DEFAULT_ENV, "REPO_DIR": REPO_DIR},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ========== TASK 8 ==========
    notificar_fallo = BashOperator(
        task_id="notificar_fallo",
        bash_command="""
            echo "❌ Pipeline falló. Revisa logs."
        """,
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    notificar_slack_fallo = BashOperator(
        task_id="notificar_slack_fallo",
        bash_command="""
            set -e
            if [ -z "$SLACK_WEBHOOK_URL" ]; then echo "SLACK_WEBHOOK_URL no configurado"; exit 0; fi
            python - << 'PYCODE'
import json, os, urllib.request
url = os.environ.get('SLACK_WEBHOOK_URL', '')
if not url:
    raise SystemExit(0)
airflow_url = os.environ.get('AIRFLOW_BASE_URL', '')
text = "❌ Pipeline falló. Revisa logs."
if airflow_url:
    text += f"\nAirflow: {airflow_url}"
payload = {"text": "Pipeline ML falló", "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    resp.read()
PYCODE
        """,
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ========== DEPENDENCIAS ==========
    validar_entorno >> actualizar_repo >> ejecutar_kedro >> validar_modelo >> promover_modelo >> graficar_prediccion >> versionar_y_subir >> notificar_exito >> notificar_slack_exito
    ejecutar_kedro >> [unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm]
    for r in [red_pca, red_tsne, red_umap]:
        [unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm] >> r
    [red_pca, red_tsne, red_umap] >> versionar_no_supervisado >> notificar_exito
    [versionar_y_subir, versionar_no_supervisado, ejecutar_kedro, validar_modelo, promover_modelo, graficar_prediccion, unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm, red_pca, red_tsne, red_umap] >> notificar_fallo
    [versionar_y_subir, versionar_no_supervisado, ejecutar_kedro, validar_modelo, promover_modelo, graficar_prediccion, unsup_kmeans, unsup_dbscan, unsup_hier, unsup_gmm, red_pca, red_tsne, red_umap] >> notificar_slack_fallo

