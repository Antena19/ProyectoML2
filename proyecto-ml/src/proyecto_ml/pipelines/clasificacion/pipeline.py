"""Pipeline de Clasificación.

Este pipeline entrena 6 modelos de clasificación con GridSearchCV y CrossValidation
para predecir variables categóricas en los datos de defunciones.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodos import (
    preparar_datos_clasificacion,
    entrenar_modelos_clasificacion,
    generar_tabla_comparativa_clasificacion,
    guardar_modelos_clasificacion
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de clasificación.
    
    Este pipeline implementa el flujo completo de machine learning para clasificación:
    
    1. Preparar datos: Extrae features y targets, divide en train/test
    2. Entrenar modelos: 6 modelos con GridSearchCV + CrossValidation (k=5)
       - Logistic Regression
       - Random Forest
       - Gradient Boosting
       - SVM
       - KNN
       - Decision Tree
    3. Generar tabla comparativa: Métricas de todos los modelos (mean±std)
    4. Guardar modelos: Persistir modelos entrenados en disco
    
    Returns:
        Pipeline de clasificación configurado
    """
    return Pipeline(
        [
            # Nodo 1: Preparar datos para clasificación
            # Extrae features y targets, codifica variables categóricas,
            # divide en train/test estratificado
            # USA dataset_individual_ml con 1.2M registros (sexo, region, edad)
            node(
                func=preparar_datos_clasificacion,
                inputs=["dataset_individual_ml", "params:modelado"],
                outputs="datos_clasificacion",
                name="preparar_datos_clasificacion",
                tags=["clasificacion", "preparacion"]
            ),
            
            # Nodo 2: Entrenar 6 modelos de clasificación
            # Cada modelo con GridSearchCV + CrossValidation (k=5)
            # Calcula métricas: accuracy, precision, recall, f1, roc-auc
            node(
                func=entrenar_modelos_clasificacion,
                inputs=["datos_clasificacion", "params:modelado"],
                outputs="resultados_clasificacion",
                name="entrenar_modelos_clasificacion",
                tags=["clasificacion", "entrenamiento", "modelos"]
            ),
            
            # Nodo 3: Generar tabla comparativa
            # Crea tabla con métricas de todos los modelos (formato mean±std)
            # Esencial para el reporte de experimentos
            node(
                func=generar_tabla_comparativa_clasificacion,
                inputs="resultados_clasificacion",
                outputs="tabla_comparativa_clasificacion",
                name="generar_tabla_comparativa_clasificacion",
                tags=["clasificacion", "comparacion", "metricas"]
            ),
            
            # Nodo 4: Guardar modelos entrenados
            # Persiste los modelos en disco para uso posterior
            # Necesario para DVC y deployment
            node(
                func=guardar_modelos_clasificacion,
                inputs=["resultados_clasificacion", "params:modelado"],
                outputs="rutas_modelos_clasificacion",
                name="guardar_modelos_clasificacion",
                tags=["clasificacion", "guardado", "modelos"]
            )
        ],
        tags="clasificacion"
    )

