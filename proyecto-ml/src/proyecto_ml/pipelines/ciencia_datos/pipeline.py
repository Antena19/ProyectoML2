"""Pipeline de Ciencia de Datos.

Este pipeline implementa la fase de ciencia de datos de la metodología CRISP-DM.
Se encarga de integrar datasets, crear features y preparar datos para modelado.
"""

from kedro.pipeline import Pipeline, node
from .nodos import (
    integrar_datasets,
    crear_features_temporales_avanzadas,
    normalizar_datos_para_modelado,
    crear_datasets_finales_para_modelado,
    preparar_dataset_individual_para_ml,
    preparar_dataset_para_regresion
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ciencia de datos.
    
    Este pipeline procesa los datos limpios y los prepara para modelado de
    machine learning, incluyendo integración, feature engineering y escalado.
    
    Incluye dos flujos:
    1. Flujo de agregación (para análisis temporal por año)
    2. Flujo de datos individuales (para clasificación ML con 1.2M registros)
    
    Returns:
        Pipeline de ciencia de datos configurado
    """
    return Pipeline(
        [
            # ============================================================
            # FLUJO 1: DATOS AGREGADOS (Análisis temporal por año)
            # ============================================================
            
            # Nodo 1: Integrar datasets de múltiples fuentes
            # Combina datos históricos con información por sexo y crea variables derivadas
            node(
                func=integrar_datasets,
                inputs=["datasets_estandarizados", "datos_historicos_nacimientos_defunciones"],
                outputs="dataset_unificado",
                name="integrar_datasets",
                tags=["ciencia_datos", "integracion"]
            ),
            
            # Nodo 2: Crear features temporales avanzadas
            # Usa el dataset_unificado para crear features de tendencia, volatilidad y promedios móviles
            node(
                func=crear_features_temporales_avanzadas,
                inputs=["dataset_unificado", "params:features_temporales"],
                outputs="dataset_con_features_temporales",
                name="crear_features_temporales_avanzadas",
                tags=["ciencia_datos", "feature_engineering"]
            ),
            
            # Nodo 3: Normalizar datos para modelado
            # Aplica diferentes tipos de normalización (StandardScaler, MinMaxScaler, RobustScaler)
            node(
                func=normalizar_datos_para_modelado,
                inputs=["dataset_con_features_temporales", "params:normalizacion"],
                outputs="datasets_normalizados",
                name="normalizar_datos_para_modelado",
                tags=["ciencia_datos", "normalizacion"]
            ),
            
            # Nodo 4: Crear datasets finales para modelado
            # Crea versiones específicas para diferentes tipos de modelos ML
            node(
                func=crear_datasets_finales_para_modelado,
                inputs=["datasets_normalizados", "params:datasets_finales"],
                outputs="datasets_finales_modelado",
                name="crear_datasets_finales_para_modelado",
                tags=["ciencia_datos", "preparacion_modelado"]
            ),
            
            # ============================================================
            # FLUJO 2: DATOS INDIVIDUALES (Para clasificación ML)
            # ============================================================
            
            # Nodo 5: Preparar dataset individual para CLASIFICACIÓN
            # Mantiene 1.2M registros individuales con sexo, region, edad
            # Agrega features temporales cíclicos + features derivados de edad
            node(
                func=preparar_dataset_individual_para_ml,
                inputs=["datasets_estandarizados", "params:modelado"],
                outputs="dataset_individual_ml",
                name="preparar_dataset_individual_ml",
                tags=["ciencia_datos", "ml", "clasificacion"]
            ),
            
            # Nodo 6: Preparar dataset LIMPIO para REGRESIÓN
            # Dataset SIN features derivados de edad (evita data leakage)
            # Solo features independientes: sexo, region, codigo_diagnostico, temporales
            node(
                func=preparar_dataset_para_regresion,
                inputs=["datasets_estandarizados", "params:modelado"],
                outputs="dataset_regresion_ml",
                name="preparar_dataset_para_regresion",
                tags=["ciencia_datos", "ml", "regresion"]
            )
        ],
        tags="ciencia_datos"
    )
