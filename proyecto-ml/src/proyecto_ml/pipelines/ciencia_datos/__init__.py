"""Pipeline de Ciencia de Datos.

Este pipeline implementa la fase de ciencia de datos de la metodología CRISP-DM.
Se encarga de:
- Integrar múltiples datasets en uno unificado
- Crear features para modelado
- Codificar variables categóricas
- Escalar y normalizar datos
- Preparar datos para entrenamiento de modelos
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
