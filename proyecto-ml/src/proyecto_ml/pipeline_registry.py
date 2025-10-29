"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    
    Pipelines disponibles:
    - ingenieria_datos: Limpieza y estandarización de datos
    - ciencia_datos: Feature engineering y normalización
    - reportes: Generación de reportes de calidad
    - clasificacion: Modelos de clasificación (6 modelos + GridSearchCV + CV)
    - regresion: Modelos de regresión (6 modelos + GridSearchCV + CV)
    - modelado_completo: Ejecuta clasificación + regresión juntos
    - __default__: Ejecuta todos los pipelines en orden
    """
    pipelines = find_pipelines()
    
    # Pipeline de modelado completo: clasificación + regresión
    if "clasificacion" in pipelines and "regresion" in pipelines:
        pipelines["modelado_completo"] = (
            pipelines["clasificacion"] + pipelines["regresion"]
        )
    
    # Pipeline por defecto: ejecuta todo en orden
    # ingenieria_datos -> ciencia_datos -> clasificacion -> regresion -> reportes
    pipelines["__default__"] = sum(pipelines.values())
    
    return pipelines
