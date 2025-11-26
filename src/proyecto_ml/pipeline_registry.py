"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from proyecto_ml.pipelines.modelado import pipeline as modelado_pipeline
from proyecto_ml.pipelines.ciencia_datos import pipeline as ciencia_pipeline
from proyecto_ml.pipelines.ingenieria_datos import pipeline as ingenieria_pipeline
from proyecto_ml.pipelines.clasificacion.pipeline import create_pipeline as clasificacion_pipeline
from proyecto_ml.pipelines.reportes import pipeline as reportes_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "modelado": modelado_pipeline.create_pipeline(),
        "ciencia_datos": ciencia_pipeline.create_pipeline(),
        "ingenieria_datos": ingenieria_pipeline.create_pipeline(),
        "clasificacion": clasificacion_pipeline(),
        "reportes": reportes_pipeline.create_pipeline(),
        "__default__": modelado_pipeline.create_pipeline(),
    }

