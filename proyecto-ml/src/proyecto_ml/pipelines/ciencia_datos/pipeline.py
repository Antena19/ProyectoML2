"""Pipeline de Ciencia de Datos.

Este pipeline implementa la fase de ciencia de datos de la metodología CRISP-DM.
Se encarga de integrar datasets, crear features y preparar datos para modelado.
"""

from kedro.pipeline import Pipeline, node
from .nodos import (
    crear_features_temporales_avanzadas,
    normalizar_datos_para_modelado,
    crear_datasets_finales_para_modelado
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ciencia de datos.
    
    Este pipeline procesa los datos limpios y los prepara para modelado de
    machine learning, incluyendo integración, feature engineering y escalado.
    
    Returns:
        Pipeline de ciencia de datos configurado
    """
    return Pipeline(
        [
            # Nodo 1: Crear features temporales avanzadas
            # Toma datasets_estandarizados del pipeline de ingeniería de datos
            node(
                func=crear_features_temporales_avanzadas,
                inputs=["datasets_estandarizados", "params:features_temporales"],
                outputs="dataset_con_features_temporales",
                name="crear_features_temporales_avanzadas",
                tags=["ciencia_datos", "feature_engineering"]
            ),
            
            # Nodo 2: Normalizar datos para modelado
            # Aplica diferentes tipos de normalización (StandardScaler, MinMaxScaler, RobustScaler)
            node(
                func=normalizar_datos_para_modelado,
                inputs=["dataset_con_features_temporales", "params:normalizacion"],
                outputs="datasets_normalizados",
                name="normalizar_datos_para_modelado",
                tags=["ciencia_datos", "normalizacion"]
            ),
            
            # Nodo 3: Crear datasets finales para modelado
            # Crea versiones específicas para diferentes tipos de modelos ML
            node(
                func=crear_datasets_finales_para_modelado,
                inputs=["datasets_normalizados", "params:datasets_finales"],
                outputs="datasets_finales_modelado",
                name="crear_datasets_finales_para_modelado",
                tags=["ciencia_datos", "preparacion_modelado"]
            )
        ],
        tags="ciencia_datos"
    )
