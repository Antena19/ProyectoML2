from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

# ========== CONFIGURACIÓN ==========
DEFAULT_ENV = {
    "TZ": "America/Santiago",
    "HOME": "/home/airflow",  # Fix para git
    "PYTHONUNBUFFERED": "1",  # Para ver logs en tiempo real
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
            set -e  # Detener en cualquier error
            
            echo "=== Validando entorno ==="
            
            # Verificar directorio
            if [ ! -d "{REPO_DIR}" ]; then
                echo "❌ ERROR: Directorio {REPO_DIR} no existe"
                exit 1
            fi
            
            # Verificar repositorio git
            if [ ! -d "{REPO_DIR}/.git" ]; then
                echo "❌ ERROR: No es un repositorio git válido"
                exit 1
            fi
            
            # Verificar repositorio DVC
            if [ ! -d "{REPO_DIR}/.dvc" ]; then
                echo "❌ ERROR: No es un repositorio DVC válido"
                exit 1
            fi
            
            # Verificar permisos
            if [ ! -w "{REPO_DIR}" ]; then
                echo "❌ ERROR: Sin permisos de escritura en {REPO_DIR}"
                exit 1
            fi
            
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
            
            # Configurar git
            git config --global --add safe.directory "{REPO_DIR}"
            git config --global user.name "{GIT_USER}"
            git config --global user.email "{GIT_EMAIL}"
            
            cd "{REPO_DIR}"
            
            # Guardar cambios locales si existen
            if ! git diff-index --quiet HEAD --; then
                echo "⚠️  Hay cambios locales, guardando stash..."
                git stash
            fi
            
            # Actualizar código
            echo "📥 Descargando últimos cambios..."
            git fetch origin
            git pull --rebase origin main || git pull --rebase origin master || true
            
            # Actualizar datos y modelos con DVC
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
            
            # Limpiar cache si es necesario
            # kedro clean
            
            echo "🚀 Iniciando pipeline de modelado..."
            kedro run --pipeline=modelado
            
            echo "✅ Pipeline ejecutado exitosamente"
        ''',
        env=DEFAULT_ENV,
        execution_timeout=timedelta(hours=2),  # Timeout para pipelines largos
    )

    # ========== TASK 4: Validar Modelo ==========
    validar_modelo = BashOperator(
        task_id="validar_modelo",
        bash_command=f'''
            set -e
            
            echo "=== Validando modelo generado ==="
            
            {VENV_ACTIVATE}
            cd "{REPO_DIR}"
            
            # Verificar que el modelo existe
            if [ ! -f "data/06_models/mejor_modelo.pkl" ]; then
                echo "❌ ERROR: Modelo no encontrado en data/06_models/mejor_modelo.pkl"
                exit 1
            fi
            
            # Verificar tamaño del modelo (debe ser > 1KB)
            MODEL_SIZE=$(stat -f%z "data/06_models/mejor_modelo.pkl" 2>/dev/null || stat -c%s "data/06_models/mejor_modelo.pkl")
            if [ "$MODEL_SIZE" -lt 1024 ]; then
                echo "❌ ERROR: Modelo demasiado pequeño ($MODEL_SIZE bytes)"
                exit 1
            fi
            
            echo "✅ Modelo válido ($(($MODEL_SIZE / 1024)) KB)"
            
            # Opcional: Ejecutar tests del modelo
            # kedro test --pipeline=validacion_modelo
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
            
            # Crear directorio de producción
            mkdir -p models/production
            
            # Backup del modelo anterior si existe
            if [ -f "models/production/model.pkl" ]; then
                TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                echo "📦 Respaldando modelo anterior..."
                cp models/production/model.pkl "models/production/model_backup_$TIMESTAMP.pkl"
            fi
            
            # Copiar nuevo modelo
            echo "🎯 Copiando nuevo modelo a producción..."
            cp data/06_models/mejor_modelo.pkl models/production/model.pkl
            
            # Copiar metadatos si existen
            if [ -f "data/06_models/metricas.json" ]; then
                cp data/06_models/metricas.json models/production/metricas.json
            fi
            
            echo "✅ Modelo promovido exitosamente"
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
            
            # Configurar git
            git config --global --add safe.directory "{REPO_DIR}"
            git config --global user.name "{GIT_USER}"
            git config --global user.email "{GIT_EMAIL}"
            
            cd "{REPO_DIR}"
            
            # Versionar con DVC
            echo "📝 Versionando modelo con DVC..."
            dvc add models/production
            
            # Verificar si hay cambios para commit
            if git diff --cached --quiet && git diff --quiet; then
                echo "ℹ️  No hay cambios para versionar"
            else
                echo "💾 Guardando cambios en git..."
                git add models/production.dvc .gitignore
                
                # Crear mensaje de commit con metadata
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                COMMIT_MSG="🤖 Promueve modelo productivo desde Airflow

Fecha: $TIMESTAMP
Pipeline: ml_modelado_kedro_dvc
Ejecutado por: {GIT_USER}"
                
                git commit -m "$COMMIT_MSG" || echo "ℹ️  Nada que commitear"
                
                # Push a git
                echo "⬆️  Subiendo a repositorio git..."
                git push origin main || git push origin master || echo "⚠️  Git push falló"
            fi
            
            # Push a DVC remote
            echo "⬆️  Subiendo artefactos a DVC remote..."
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
            echo "📊 Modelo actualizado en: {REPO_DIR}/models/production/model.pkl"
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
    validar_entorno >> actualizar_repo >> ejecutar_kedro >> validar_modelo >> promover_modelo >> versionar_y_subir >> notificar_exito
    
    # Tareas finales se ejecutan siempre
    [versionar_y_subir, ejecutar_kedro, validar_modelo, promover_modelo] >> notificar_fallo