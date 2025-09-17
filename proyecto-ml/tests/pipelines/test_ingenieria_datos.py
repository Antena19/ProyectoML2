"""
Tests para el pipeline de ingeniería de datos.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para tests

class TestIngenieriaDatos:
    """Tests para el pipeline de ingeniería de datos."""
    
    def test_limpiar_defunciones_basico(self):
        """Test básico de limpieza de defunciones."""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'AÑO': [2014, 2015, 2016, 2014],  # Duplicado intencional
            'FECHA_DEF': ['2014-01-01', '2015-01-01', '2016-01-01', '2014-01-01'],
            'SEXO_NOMBRE': ['Hombre', 'Mujer', 'Hombre', 'Hombre'],
            'EDAD_TIPO': ['Años', 'Años', 'Años', 'Años'],
            'EDAD_CANT': [25, 30, 35, 25],
            'COD_COMUNA': ['13101', '13102', '13103', '13101'],
            'COMUNA': ['Santiago', 'Providencia', 'Las Condes', 'Santiago'],
            'NOMBRE_REGION': ['Metropolitana', 'Metropolitana', 'Metropolitana', 'Metropolitana'],
            'CAPITULO_DIAG1': ['A00-B99', 'C00-D48', 'E00-E90', 'A00-B99'],
            'GLOSA_CAPITULO_DIAG1': ['Enfermedades infecciosas', 'Neoplasias', 'Endocrinas', 'Enfermedades infecciosas']
        })
        
        # Importar función de limpieza
        from src.proyecto_ml.pipelines.ingenieria_datos.nodos import limpiar_defunciones
        
        # Parámetros mock
        params = {
            "eliminar_duplicados": True,
            "eliminar_geograficos_nulos": True,
            "estrategia_fechas_nulas": "imputacion_media_anual"
        }
        
        # Ejecutar función
        result = limpiar_defunciones(test_data, params)
        
        # Verificar resultados
        assert not result.empty, "El resultado no debe estar vacío"
        assert 'AÑO_FECHA' in result.columns, "Debe tener columna AÑO_FECHA"
        assert 'MES' in result.columns, "Debe tener columna MES"
        assert 'DIA_SEMANA' in result.columns, "Debe tener columna DIA_SEMANA"
        assert 'TRIMESTRE' in result.columns, "Debe tener columna TRIMESTRE"
        assert 'DIA_AÑO' in result.columns, "Debe tener columna DIA_AÑO"
        
        # Verificar que se eliminaron duplicados
        assert len(result) < len(test_data), "Debe haber eliminado duplicados"
        
        # Verificar tipos de datos
        assert pd.api.types.is_datetime64_any_dtype(result['FECHA_DEF']), "FECHA_DEF debe ser datetime"
    
    def test_estandarizar_columnas(self):
        """Test de estandarización de columnas."""
        # Crear datos de prueba
        defunciones_data = pd.DataFrame({
            'AÑO': [2014, 2015],
            'FECHA_DEF': ['2014-01-01', '2015-01-01'],
            'SEXO_NOMBRE': ['Hombre', 'Mujer']
        })
        
        nacimientos_data = pd.DataFrame({
            'Año': [2014, 2015],
            'Nacimiento (Hombre)': [100, 110],
            'Nacimiento (Mujer)': [95, 105]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.ingenieria_datos.nodos import estandarizar_columnas
        
        # Ejecutar función
        result = estandarizar_columnas(defunciones_data, nacimientos_data)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'defunciones_estandarizado' in result, "Debe tener defunciones estandarizado"
        assert 'nacimientos_por_sexo_estandarizado' in result, "Debe tener nacimientos estandarizado"
        
        # Verificar columnas estandarizadas
        def_std = result['defunciones_estandarizado']
        assert 'año' in def_std.columns, "Debe tener columna 'año' estandarizada"
        assert 'fecha_defuncion' in def_std.columns, "Debe tener columna 'fecha_defuncion' estandarizada"
        assert 'sexo' in def_std.columns, "Debe tener columna 'sexo' estandarizada"
    
    def test_validar_calidad_datos(self):
        """Test de validación de calidad de datos."""
        # Crear datos de prueba
        datasets_estandarizados = {
            'defunciones_estandarizado': pd.DataFrame({
                'año': [2014, 2015, 2016],
                'fecha_defuncion': ['2014-01-01', '2015-01-01', '2016-01-01'],
                'sexo': ['Hombre', 'Mujer', 'Hombre'],
                'region': ['Metropolitana', 'Valparaíso', 'Metropolitana']
            }),
            'nacimientos_por_sexo_estandarizado': pd.DataFrame({
                'año': [2014, 2015],
                'nacimientos_hombres': [100, 110],
                'nacimientos_mujeres': [95, 105]
            })
        }
        
        # Importar función
        from src.proyecto_ml.pipelines.ingenieria_datos.nodos import validar_calidad_datos
        
        # Parámetros mock
        params = {
            "checks_calidad": ["verificar_completitud", "verificar_duplicados"],
            "umbrales_validacion": {
                "completitud_minima": 0.95,
                "duplicados_maximo": 0.05
            }
        }
        
        # Ejecutar función
        result = validar_calidad_datos(datasets_estandarizados, params)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'defunciones' in result, "Debe tener métricas de defunciones"
        assert 'nacimientos' in result, "Debe tener métricas de nacimientos"
        assert 'estado' in result, "Debe tener estado de validación"
        
        # Verificar métricas
        def_metrics = result['defunciones']
        assert 'total_registros' in def_metrics, "Debe tener total_registros"
        assert 'total_columnas' in def_metrics, "Debe tener total_columnas"
        assert 'valores_nulos' in def_metrics, "Debe tener valores_nulos"
        assert 'duplicados' in def_metrics, "Debe tener duplicados"
