"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from proyecto_ml.pipelines.modelado import pipeline as modelado_pipeline
<<<<<<< HEAD
#Paso 5: A) Importamos el pipeline de clasificación
from proyecto_ml.pipelines.clasificacion.pipeline import create_pipeline as clasificacion_pipeline

=======
>>>>>>> 90c46579fb7a25e44638a7cd92a52497ee8b57bf


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    
    # Pipeline por defecto: solo ingeniería de datos por ahora
    pipelines["__default__"] = sum(pipelines.values())
    
    return pipelines

def register_pipelines():
    return {
        "modelado": modelado_pipeline.create_pipeline(),
        "__default__": modelado_pipeline.create_pipeline(),
<<<<<<< HEAD
        #B) Registramos el pipeline de clasificación
        "clasificacion": clasificacion_pipeline(),

=======
>>>>>>> 90c46579fb7a25e44638a7cd92a52497ee8b57bf
    }

