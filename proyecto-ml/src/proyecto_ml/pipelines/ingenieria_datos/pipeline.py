"""Pipeline de Ingeniería de Datos.

Este pipeline implementa la fase de preparación de datos de la metodología CRISP-DM.
Se encarga de limpiar, estandarizar y validar los datos crudos para prepararlos
para la siguiente fase de ciencia de datos.
"""

from kedro.pipeline import Pipeline, node
from .nodos import (
    cargar_datos_crudos,
    limpiar_defunciones,
    estandarizar_columnas,
    validar_calidad_datos
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ingeniería de datos.
    
    Este pipeline procesa los datos crudos siguiendo el flujo:
    1. Cargar datos del catálogo
    2. Limpiar dataset de defunciones
    3. Estandarizar nombres de columnas
    4. Validar calidad de datos
    
    Returns:
        Pipeline de Kedro configurado
    """
    return Pipeline(
        [
            # Nodo 1: Cargar todos los datos crudos del catálogo
            node(
                func=cargar_datos_crudos,
                inputs=[
                    "datos_historicos_nacimientos_defunciones",
                    "datos_filtrados_defunciones", 
                    "nacimientos_defunciones_por_sexo",
                    "nacimientos_por_edad_madre",
                    "defunciones_por_edad_fallecido"
                ],
                outputs="datasets_crudos_cargados",
                name="cargar_datos_crudos",
                tags=["carga", "datos_crudos"]
            ),
            
            # Nodo 2: Limpiar dataset de defunciones (el más crítico)
            node(
                func=limpiar_defunciones,
                inputs="datos_filtrados_defunciones",
                outputs="defunciones_limpias",
                name="limpiar_defunciones",
                tags=["limpieza", "defunciones"]
            ),
            
            # Nodo 3: Estandarizar nombres de columnas
            node(
                func=estandarizar_columnas,
                inputs=[
                    "defunciones_limpias",
                    "nacimientos_defunciones_por_sexo"
                ],
                outputs="datasets_estandarizados",
                name="estandarizar_columnas",
                tags=["estandarizacion", "columnas"]
            ),
            
            # Nodo 4: Validar calidad de datos procesados
            node(
                func=validar_calidad_datos,
                inputs="datasets_estandarizados",
                outputs="metricas_calidad_datos",
                name="validar_calidad_datos",
                tags=["validacion", "calidad"]
            )
        ],
        tags="ingenieria_datos"
    )
