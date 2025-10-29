"""Pipeline de Clasificación para Machine Learning.

Este pipeline implementa modelos de clasificación para predecir variables categóricas
como sexo y región geográfica en los datos de defunciones.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

