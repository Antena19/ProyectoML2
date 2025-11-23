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

    # ========== TASK 8 ==========
    notificar_fallo = BashOperator(
        task_id="notificar_fallo",
        bash_command="""
            echo "❌ Pipeline falló. Revisa logs."
        """,
        env=DEFAULT_ENV,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ========== DEPENDENCIAS ==========
    validar_entorno >> actualizar_repo >> ejecutar_kedro >> validar_modelo >> promover_modelo >> graficar_prediccion >> versionar_y_subir >> notificar_exito
    [versionar_y_subir, ejecutar_kedro, validar_modelo, promover_modelo, graficar_prediccion] >> notificar_fallo

