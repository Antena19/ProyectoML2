"""Pipeline de Regresión.

Este pipeline entrena 6 modelos de regresión con GridSearchCV y CrossValidation
para predecir variables continuas en los datos de defunciones.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodos import (
    preparar_datos_regresion,
    entrenar_modelos_regresion,
    generar_tabla_comparativa_regresion,
    guardar_modelos_regresion
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de regresión.
    
    Este pipeline implementa el flujo completo de machine learning para regresión:
    
    1. Preparar datos: Extrae features y targets, divide en train/test
    2. Entrenar modelos: 6 modelos con GridSearchCV + CrossValidation (k=5)
       - Linear Regression (Ridge)
       - Random Forest Regressor
       - Gradient Boosting Regressor
       - SVR
       - KNN Regressor
       - Decision Tree Regressor
    3. Generar tabla comparativa: Métricas de todos los modelos (mean±std)
    4. Guardar modelos: Persistir modelos entrenados en disco
    
    Returns:
        Pipeline de regresión configurado
    """
    return Pipeline(
        [
            # Nodo 1: Preparar datos para regresión
            # Extrae features y targets, divide en train/test
            # USA dataset_regresion_ml (LIMPIO, sin data leakage)
            node(
                func=preparar_datos_regresion,
                inputs=["dataset_regresion_ml", "params:modelado"],
                outputs="datos_regresion",
                name="preparar_datos_regresion",
                tags=["regresion", "preparacion"]
            ),
            
            # Nodo 2: Entrenar 6 modelos de regresión
            # Cada modelo con GridSearchCV + CrossValidation (k=5)
            # Calcula métricas: MAE, MSE, RMSE, R², MAPE
            node(
                func=entrenar_modelos_regresion,
                inputs=["datos_regresion", "params:modelado"],
                outputs="resultados_regresion",
                name="entrenar_modelos_regresion",
                tags=["regresion", "entrenamiento", "modelos"]
            ),
            
            # Nodo 3: Generar tabla comparativa
            # Crea tabla con métricas de todos los modelos (formato mean±std)
            # Esencial para el reporte de experimentos
            node(
                func=generar_tabla_comparativa_regresion,
                inputs="resultados_regresion",
                outputs="tabla_comparativa_regresion",
                name="generar_tabla_comparativa_regresion",
                tags=["regresion", "comparacion", "metricas"]
            ),
            
            # Nodo 4: Guardar modelos entrenados
            # Persiste los modelos en disco para uso posterior
            # Necesario para DVC y deployment
            node(
                func=guardar_modelos_regresion,
                inputs=["resultados_regresion", "params:modelado"],
                outputs="rutas_modelos_regresion",
                name="guardar_modelos_regresion",
                tags=["regresion", "guardado", "modelos"]
            )
        ],
        tags="regresion"
    )

