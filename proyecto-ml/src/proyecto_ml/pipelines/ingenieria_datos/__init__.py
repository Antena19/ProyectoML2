"""Pipeline de Ingeniería de Datos.

Este pipeline se encarga de la limpieza y preparación inicial de los datos crudos.
Incluye funciones para:
- Cargar datos del catálogo
- Limpiar dataset de defunciones
- Estandarizar nombres de columnas
- Validar calidad de datos
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
