#Paso 4: Creamos el pipeline para cargar el dataset
from kedro.pipeline import Pipeline, node
from .nodes import cargar_datos

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=cargar_datos,
                inputs="nacimientos_raw",
                outputs="datos_cargados",
                name="cargar_datos_node"
            )
        ]
    )
"""
Pipeline simple de Clasificación (carga de dataset de nacimientos).

Objetivo:
- Demostrar carga y paso de datos crudos para futuros nodos de clasificación.

Salida:
- `datos_cargados` (en memoria) para nodos posteriores.
"""
