from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# ========== CONFIGURACIÓN ==========
DEFAULT_ENV = {
    "TZ": "America/Santiago",
    "HOME": "/home/airflow",      # Fix para git
    "PYTHONUNBUFFERED": "1",      # Logs en tiempo real
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

    # ========== TASK 1: Validar Entorno ==========
    validar_entorno = BashOperator(
        task_id="validar_entorno",
        bash_command=f'''
            set -e
            echo "=== Validando entorno ==="

            if [ ! -d "{REPO_DIR}" ]; then
                echo "❌ ERROR: Directorio {REPO_DIR} no existe"; exit 1; fi
            if [ ! -d "{REPO_DIR}/.git" ]; then
                echo "❌ ERROR: No es un repositorio git válido"; exit 1; fi
            if [ ! -d "{REPO_DIR}/.dvc" ]; then
                echo "❌ ERROR: No es un repositorio DVC válido"; exit 1; fi
            if [ ! -w "{REPO_DIR}" ]; then
                echo "❌ ERROR: Sin permisos de escritura en {REPO_DIR}"; exit 1; fi

            echo "✅ Validación exitosa"
        ''',
        env=DEFAULT_ENV,
    )

    # ========== TASK 2: Actualizar Repositorio ==========
    actualizar_repo = BashOperator(
        task_id="actualizar_repo",
        bash_command=f'''
            set -e
            echo "=== Actualizando repositorio ==="

            {VENV_ACTIVATE}
            git config --global --add safe.directory "{REPO_DIR}"
            git config --global user.name "{GIT_USER}"
            git config --global user.email "{GIT_EMAIL}"

            cd "{REPO_DIR}"

            if ! git diff-index --quiet HEAD --; then
                echo "⚠️  Cambios locales detectados; guardando stash..."
                git stash
            fi

            echo "📥 Descargando últimos cambios..."
            git fetch origin
            git pull --rebase origin main || git pull --rebase origin master || true

            echo "📦 Actualizando artefactos DVC..."
            dvc pull || echo "⚠️  Advertencia: dvc pull falló parcialmente"

            echo "✅ Repositorio actualizado"
        ''',
        env=DEFAULT_ENV,
    )

    # ========== TASK 3: Ejecutar Pipeline Kedro ==========
    ejecutar_kedro = BashOperator(
        task_id="ejecutar_kedro",
        bash_command=f'''
            set -e
            echo "=== Ejecutando pipeline Kedro ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"

            # kedro clean   # opcional
            echo "🚀 Iniciando pipeline de modelado..."
            kedro run --pipeline=modelado
            echo "✅ Pipeline ejecutado exitosamente"
        ''',
        env=DEFAULT_ENV,
        execution_timeout=timedelta(hours=2),
    )

    # ========== TASK 4: Validar Modelo ==========
    validar_modelo = BashOperator(
        task_id="validar_modelo",
        bash_command=f'''
            set -e
            echo "=== Validando modelo generado ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"

            if [ ! -f "data/06_models/mejor_modelo.pkl" ]; then
                echo "❌ ERROR: Modelo no encontrado en data/06_models/mejor_modelo.pkl"; exit 1; fi

            MODEL_SIZE=$(stat -f%z "data/06_models/mejor_modelo.pkl" 2>/dev/null || stat -c%s "data/06_models/mejor_modelo.pkl")
            if [ "$MODEL_SIZE" -lt 1024 ]; then
                echo "❌ ERROR: Modelo demasiado pequeño ($MODEL_SIZE bytes)"; exit 1; fi

            echo "✅ Modelo válido ($(($MODEL_SIZE / 1024)) KB)"
        ''',
        env=DEFAULT_ENV,
    )

    # ========== TASK 5: Promover Modelo ==========
    promover_modelo = BashOperator(
        task_id="promover_modelo",
        bash_command=f'''
            set -e
            echo "=== Promoviendo modelo a producción ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"

            mkdir -p models/production

            if [ -f "models/production/model.pkl" ]; then
                TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                echo "📦 Respaldando modelo anterior..."
                cp models/production/model.pkl "models/production/model_backup_$TIMESTAMP.pkl"
            fi

            echo "🎯 Copiando nuevo modelo a producción..."
            cp data/06_models/mejor_modelo.pkl models/production/model.pkl

            if [ -f "data/06_models/metricas.json" ]; then
                cp data/06_models/metricas.json models/production/metricas.json
            fi

            echo "✅ Modelo promovido exitosamente"
        ''',
        env=DEFAULT_ENV,
    )

    # ========== NUEVO: TASK 5.5 Graficar Predicción ==========
    graficar_prediccion = BashOperator(
        task_id="graficar_prediccion",
        bash_command=f'''
            set -e
            echo "=== Generando gráfico: últimos 5 años + predicción año actual ==="
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"

            python - << 'PYCODE'
import os, pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

REPO_DIR = "{REPO_DIR}"
DATA_CSV = os.path.join(REPO_DIR, "data/03_primary/dataset_unificado.csv")
MODEL_PKL = os.path.join(REPO_DIR, "models/production/model.pkl")
OUT_DIR   = os.path.join(REPO_DIR, "models/production")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Cargar datos (usa columnas: 'año', 'defunciones_totales')
df = pd.read_csv(DATA_CSV, encoding="utf-8")
# Normalizamos nombres a 'anio' y 'defunciones'
df = df.rename(columns={"año": "anio", "defunciones_totales": "defunciones"})
df = df[['anio', 'defunciones']].dropna()
df['anio'] = df['anio'].astype(int)
df = df.sort_values('anio')

# 2) Últimos 5 años reales
ultimos5 = df.tail(5).copy()

# 3) Predicción año actual
anio_actual = datetime.now().year
with open(MODEL_PKL, "rb") as f:
    modelo = pickle.load(f)

X_new = pd.DataFrame({"anio": [anio_actual]})
y_pred = modelo.predict(X_new)
pred = int(round(float(y_pred[0])))

# 4) Guardar predicción en CSV
pd.DataFrame({"anio":[anio_actual], "defunciones_predichas":[pred]}) \
  .to_csv(os.path.join(OUT_DIR, "prediccion_actual.csv"), index=False)

# 5) Graficar (5 reales + 1 predicho)
plt.figure()
plt.plot(ultimos5["anio"], ultimos5["defunciones"], marker="o", label="Reales (últimos 5)")
plt.plot([anio_actual], [pred], marker="o", linestyle="--", label=f"Predicción {anio_actual}")
plt.title("Defunciones en Chile: últimos 5 años + predicción año actual")
plt.xlabel("Año")
plt.ylabel("Defunciones")
plt.legend()
plt.grid(True)

out_png = os.path.join(OUT_DIR, "fig_prediccion.png")
plt.savefig(out_png, bbox_inches="tight")
print(f"✅ Gráfico guardado en: {out_png}")
PYCODE
        ''',
        env=DEFAULT_ENV,
    )

    # ========== TASK 6: Versionar y Subir ==========
    versionar_y_subir = BashOperator(
        task_id="versionar_y_subir",
        bash_command=f'''
            set -e
            echo "=== Versionando y subiendo artefactos ==="
            {VENV_ACTIVATE}

            git config --global --add safe.directory "{REPO_DIR}"
            git config --global user.name "{GIT_USER}"
            git config --global user.email "{GIT_EMAIL}"

            cd "{REPO_DIR}"

            echo "📝 dvc add models/production ..."
            dvc add models/production

            if git diff --cached --quiet && git diff --quiet; then
                echo "ℹ️  No hay cambios para versionar"
            else
                echo "💾 Guardando cambios en git..."
                git add models/production.dvc .gitignore
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                COMMIT_MSG="🤖 Promueve modelo productivo y gráfico de predicción

Fecha: $TIMESTAMP
Pipeline: ml_modelado_kedro_dvc
Ejecutado por: {GIT_USER}"
                git commit -m "$COMMIT_MSG" || echo "ℹ️  Nada que commitear"
                echo "⬆️  git push ..."
                git push origin main || git push origin master || echo "⚠️  Git push falló"
            fi

            echo "⬆️  dvc push ..."
            dvc push || echo "⚠️  DVC push falló parcialmente"

            echo "✅ Versionado completado"
        ''',
        env=DEFAULT_ENV,
    )

    # ========== TASK 7: Notificación Final ==========
    notificar_exito = BashOperator(
        task_id="notificar_exito",
        bash_command=f'''
            echo "✅ ============================================"
            echo "✅ Pipeline ML completado exitosamente"
            echo "✅ ============================================"
            echo "📊 Modelo: {REPO_DIR}/models/production/model.pkl"
            echo "🖼️ Gráfico: {REPO_DIR}/models/production/fig_prediccion.png"
            echo "📄 CSV:     {REPO_DIR}/models/production/prediccion_actual.csv"
            echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "============================================"
        ''',
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ========== TASK 8: Manejo de Errores ==========
    notificar_fallo = BashOperator(
        task_id="notificar_fallo",
        bash_command='''
            echo "❌ ============================================"
            echo "❌ Pipeline ML falló"
            echo "❌ ============================================"
            echo "⚠️  Revisa los logs de Airflow para más detalles"
            echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "============================================"
        ''',
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ========== DEPENDENCIAS ==========
    validar_entorno >> actualizar_repo >> ejecutar_kedro >> validar_modelo >> promover_modelo >> graficar_prediccion >> versionar_y_subir >> notificar_exito
    [versionar_y_subir, ejecutar_kedro, validar_modelo, promover_modelo, graficar_prediccion] >> notificar_fallo
