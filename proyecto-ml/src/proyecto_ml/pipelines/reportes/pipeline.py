"""
Pipeline de reportes para generar visualizaciones y métricas de calidad.
"""

from kedro.pipeline import Pipeline, node
from .nodos import (
    generar_reporte_calidad_datos,
    generar_visualizaciones_calidad,
    generar_reporte_features_temporales,
    generar_visualizaciones_features,
    generar_reporte_final
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de reportes.
    
    Returns:
        Pipeline de reportes configurado
    """
    return Pipeline(
        [
            # Generar reporte de calidad de datos
            node(
                func=generar_reporte_calidad_datos,
                inputs="metricas_calidad_datos",
                outputs="reporte_calidad_datos",
                name="generar_reporte_calidad_datos",
                tags=["reportes", "calidad"]
            ),
            
            # Generar visualizaciones de calidad
            node(
                func=generar_visualizaciones_calidad,
                inputs="metricas_calidad_datos",
                outputs="visualizaciones_calidad",
                name="generar_visualizaciones_calidad",
                tags=["reportes", "visualizaciones"]
            ),
            
            # Generar reporte de features temporales
            node(
                func=generar_reporte_features_temporales,
                inputs="dataset_con_features_temporales",
                outputs="reporte_features_temporales",
                name="generar_reporte_features_temporales",
                tags=["reportes", "features"]
            ),
            
            # Generar visualizaciones de features
            node(
                func=generar_visualizaciones_features,
                inputs="dataset_con_features_temporales",
                outputs="visualizaciones_features",
                name="generar_visualizaciones_features",
                tags=["reportes", "visualizaciones"]
            ),
            
            # Generar reporte final consolidado
            node(
                func=generar_reporte_final,
                inputs=[
                    "reporte_calidad_datos",
                    "reporte_features_temporales", 
                    "visualizaciones_calidad",
                    "visualizaciones_features"
                ],
                outputs="reporte_final_consolidado",
                name="generar_reporte_final",
                tags=["reportes", "consolidado"]
            )
        ],
        tags="reportes"
    )
