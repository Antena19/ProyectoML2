"""
Tests para el pipeline de reportes.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para tests

class TestReportes:
    """Tests para el pipeline de reportes."""
    
    def test_generar_reporte_calidad_datos(self):
        """Test de generación de reporte de calidad."""
        # Crear métricas de prueba
        metricas_calidad_datos = {
            'defunciones': {
                'total_registros': 1000,
                'completitud': 0.95,
                'duplicados': 5,
                'outliers': 10
            },
            'nacimientos_por_sexo': {
                'total_registros': 100,
                'completitud': 0.98,
                'duplicados': 2,
                'outliers': 3
            }
        }
        
        # Importar función
        from src.proyecto_ml.pipelines.reportes.nodos import generar_reporte_calidad_datos
        
        # Ejecutar función
        result = generar_reporte_calidad_datos(metricas_calidad_datos)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'resumen_ejecutivo' in result, "Debe tener resumen ejecutivo"
        assert 'metricas_detalladas' in result, "Debe tener métricas detalladas"
        assert 'recomendaciones' in result, "Debe tener recomendaciones"
        assert 'estado_general' in result, "Debe tener estado general"
        
        # Verificar resumen ejecutivo
        resumen = result['resumen_ejecutivo']
        assert 'defunciones' in resumen, "Debe tener métricas de defunciones"
        assert 'nacimientos' in resumen, "Debe tener métricas de nacimientos"
        
        # Verificar recomendaciones
        assert isinstance(result['recomendaciones'], list), "Recomendaciones debe ser una lista"
        assert len(result['recomendaciones']) > 0, "Debe tener al menos una recomendación"
        
        # Verificar estado general
        assert result['estado_general'] in ['BUENO', 'REQUIERE_ATENCION'], "Estado debe ser válido"
    
    def test_generar_reporte_features_temporales(self):
        """Test de generación de reporte de features temporales."""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'año': [2014, 2015, 2016, 2017],
            'mes': [1, 2, 3, 4],
            'mes_sin': [0.5, 0.8, 0.9, 0.7],
            'mes_cos': [0.8, 0.6, 0.4, 0.7],
            'dia_año_sin': [0.1, 0.3, 0.5, 0.7],
            'dia_año_cos': [0.9, 0.8, 0.6, 0.4],
            'trimestre_sin': [0.2, 0.4, 0.6, 0.8],
            'trimestre_cos': [0.8, 0.7, 0.5, 0.3],
            'dia_semana_sin': [0.3, 0.5, 0.7, 0.9],
            'dia_semana_cos': [0.7, 0.6, 0.4, 0.2],
            'es_fin_de_semana': [False, True, False, True],
            'es_mes_inicio_ano': [True, False, False, False],
            'es_mes_fin_ano': [False, False, False, False]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.reportes.nodos import generar_reporte_features_temporales
        
        # Ejecutar función
        result = generar_reporte_features_temporales(test_data)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'resumen_features' in result, "Debe tener resumen de features"
        assert 'analisis_ciclicos' in result, "Debe tener análisis cíclicos"
        assert 'distribucion_temporal' in result, "Debe tener distribución temporal"
        assert 'recomendaciones' in result, "Debe tener recomendaciones"
        
        # Verificar resumen de features
        resumen = result['resumen_features']
        assert 'total_features' in resumen, "Debe tener total_features"
        assert 'features_ciclicos' in resumen, "Debe tener features_ciclicos"
        assert 'features_especiales' in resumen, "Debe tener features_especiales"
        assert 'total_registros' in resumen, "Debe tener total_registros"
        
        # Verificar que cuenta correctamente
        assert resumen['total_features'] == len(test_data.columns), "Total features debe coincidir"
        assert resumen['features_ciclicos'] == 8, "Debe tener 8 features cíclicos"
        assert resumen['features_especiales'] == 3, "Debe tener 3 features especiales"
        assert resumen['total_registros'] == len(test_data), "Total registros debe coincidir"
        
        # Verificar distribución temporal
        dist_temporal = result['distribucion_temporal']
        assert 'años_cubiertos' in dist_temporal, "Debe tener años_cubiertos"
        assert 'rango_años' in dist_temporal, "Debe tener rango_años"
    
    def test_generar_visualizaciones_calidad(self):
        """Test de generación de visualizaciones de calidad."""
        # Crear métricas de prueba
        metricas_calidad_datos = {
            'defunciones': {
                'total_registros': 1000,
                'valores_nulos': 50,
                'duplicados': 5,
                'total_columnas': 10
            },
            'nacimientos': {
                'total_registros': 100,
                'valores_nulos': 2,
                'duplicados': 1,
                'total_columnas': 5
            }
        }
        
        # Importar función
        from src.proyecto_ml.pipelines.reportes.nodos import generar_visualizaciones_calidad
        
        # Ejecutar función
        result = generar_visualizaciones_calidad(metricas_calidad_datos)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        
        # Verificar que se generaron visualizaciones
        assert len(result) > 0, "Debe generar al menos una visualización"
        
        # Verificar tipos de visualizaciones esperadas
        visualizaciones_esperadas = ['completitud', 'duplicados', 'valores_nulos']
        for viz in visualizaciones_esperadas:
            if viz in result:
                assert isinstance(result[viz], str), f"{viz} debe ser una ruta de archivo"
                assert result[viz].endswith('.png'), f"{viz} debe ser un archivo PNG"
    
    def test_generar_visualizaciones_features(self):
        """Test de generación de visualizaciones de features."""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'año': [2014, 2015, 2016, 2017],
            'mes': [1, 2, 3, 4],
            'mes_sin': [0.5, 0.8, 0.9, 0.7],
            'mes_cos': [0.8, 0.6, 0.4, 0.7],
            'dia_año_sin': [0.1, 0.3, 0.5, 0.7],
            'dia_año_cos': [0.9, 0.8, 0.6, 0.4],
            'trimestre_sin': [0.2, 0.4, 0.6, 0.8],
            'trimestre_cos': [0.8, 0.7, 0.5, 0.3],
            'dia_semana_sin': [0.3, 0.5, 0.7, 0.9],
            'dia_semana_cos': [0.7, 0.6, 0.4, 0.2]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.reportes.nodos import generar_visualizaciones_features
        
        # Ejecutar función
        result = generar_visualizaciones_features(test_data)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        
        # Verificar que se generaron visualizaciones
        assert len(result) > 0, "Debe generar al menos una visualización"
        
        # Verificar tipos de visualizaciones esperadas
        visualizaciones_esperadas = ['features_ciclicos', 'distribucion_temporal']
        for viz in visualizaciones_esperadas:
            if viz in result:
                assert isinstance(result[viz], str), f"{viz} debe ser una ruta de archivo"
                assert result[viz].endswith('.png'), f"{viz} debe ser un archivo PNG"
    
    def test_generar_reporte_final(self):
        """Test de generación de reporte final consolidado."""
        # Crear datos de prueba
        reporte_calidad = {
            'estado_general': 'BUENO',
            'recomendaciones': ['Datos listos para análisis']
        }
        
        reporte_features = {
            'resumen_features': {'total_features': 10},
            'recomendaciones': ['Features completos']
        }
        
        visualizaciones_calidad = {'completitud': 'path1.png'}
        visualizaciones_features = {'features_ciclicos': 'path2.png'}
        
        # Importar función
        from src.proyecto_ml.pipelines.reportes.nodos import generar_reporte_final
        
        # Ejecutar función
        result = generar_reporte_final(
            reporte_calidad, reporte_features, 
            visualizaciones_calidad, visualizaciones_features
        )
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'informacion_general' in result, "Debe tener información general"
        assert 'resumen_ejecutivo' in result, "Debe tener resumen ejecutivo"
        assert 'reportes_detallados' in result, "Debe tener reportes detallados"
        assert 'archivos_generados' in result, "Debe tener archivos generados"
        assert 'recomendaciones_generales' in result, "Debe tener recomendaciones generales"
        
        # Verificar información general
        info_general = result['informacion_general']
        assert 'fecha_generacion' in info_general, "Debe tener fecha de generación"
        assert 'version' in info_general, "Debe tener versión"
        assert 'metodologia' in info_general, "Debe tener metodología"
        
        # Verificar resumen ejecutivo
        resumen = result['resumen_ejecutivo']
        assert 'estado_calidad' in resumen, "Debe tener estado de calidad"
        assert 'features_generados' in resumen, "Debe tener features generados"
        assert 'visualizaciones_generadas' in resumen, "Debe tener visualizaciones generadas"
