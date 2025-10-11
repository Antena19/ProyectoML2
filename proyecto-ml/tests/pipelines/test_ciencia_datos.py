"""
Tests para el pipeline de ciencia de datos.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para tests

class TestCienciaDatos:
    """Tests para el pipeline de ciencia de datos."""
    
    def test_integrar_datasets(self):
        """Test de integración de datasets múltiples."""
        # Crear datos de prueba
        datasets_estandarizados = {
            'nacimientos_por_sexo_estandarizado': pd.DataFrame({
                'año': [2015, 2016, 2017],
                'nacimientos_hombres': [124000, 118000, 112000],
                'nacimientos_mujeres': [120000, 114000, 108000],
                'defunciones_hombres': [55000, 55000, 56000],
                'defunciones_mujeres': [49000, 49000, 51000]
            })
        }
        
        datos_historicos = pd.DataFrame({
            'año': [2014, 2015, 2016, 2017, 2018],
            'nacimientos_totales': [250000, 244000, 232000, 220000, 222000],
            'defunciones_totales': [102000, 103000, 104000, 107000, 107000]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.ciencia_datos.nodos import integrar_datasets
        
        # Ejecutar función
        result = integrar_datasets(datasets_estandarizados, datos_historicos)
        
        # Verificar resultados
        assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
        assert not result.empty, "El resultado no debe estar vacío"
        assert len(result) == len(datos_historicos), "Debe tener todos los años históricos"
        
        # Verificar columnas base
        assert 'año' in result.columns, "Debe tener columna año"
        assert 'nacimientos_totales' in result.columns, "Debe tener nacimientos_totales"
        assert 'defunciones_totales' in result.columns, "Debe tener defunciones_totales"
        
        # Verificar variables derivadas
        assert 'tasa_natalidad' in result.columns, "Debe tener tasa_natalidad"
        assert 'tasa_mortalidad' in result.columns, "Debe tener tasa_mortalidad"
        assert 'crecimiento_natural' in result.columns, "Debe tener crecimiento_natural"
        assert 'porcentaje_crecimiento_natural' in result.columns, "Debe tener porcentaje_crecimiento_natural"
        
        # Verificar integración por sexo (solo para algunos años)
        assert 'nacimientos_hombres' in result.columns, "Debe tener nacimientos_hombres"
        assert 'ratio_nacimientos_sexo' in result.columns, "Debe tener ratio_nacimientos_sexo"
    
    def test_crear_features_temporales_avanzadas(self):
        """Test de creación de features temporales avanzadas."""
        # Crear datos de prueba (datos agregados por año)
        dataset_unificado = pd.DataFrame({
            'año': [2014, 2015, 2016, 2017, 2018],
            'nacimientos_totales': [250000, 244000, 232000, 220000, 222000],
            'defunciones_totales': [102000, 103000, 104000, 107000, 107000],
            'nacimientos_hombres': [128000, 125000, 119000, 112000, 113000],
            'nacimientos_mujeres': [122000, 119000, 113000, 108000, 109000],
            'defunciones_hombres': [55000, 55000, 55000, 56000, 56000],
            'defunciones_mujeres': [47000, 48000, 49000, 51000, 51000],
            'crecimiento_natural': [148000, 141000, 128000, 113000, 115000]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.ciencia_datos.nodos import crear_features_temporales_avanzadas
        
        # Parámetros mock
        params = {
            "ciclicos": ["ciclo_5_anos", "ciclo_10_anos"],
            "especiales": ["tendencia_lineal", "tendencia_cuadratica"],
            "basicas": ["año_normalizado", "decada", "siglo"]
        }
        
        # Ejecutar función
        result = crear_features_temporales_avanzadas(dataset_unificado, params)
        
        # Verificar resultados
        assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
        assert not result.empty, "El resultado no debe estar vacío"
        
        # Verificar features básicos
        assert 'año_normalizado' in result.columns, "Debe tener año_normalizado"
        assert 'decada' in result.columns, "Debe tener decada"
        assert 'siglo' in result.columns, "Debe tener siglo"
        
        # Verificar features de tendencia
        assert 'tendencia_lineal' in result.columns, "Debe tener tendencia_lineal"
        assert 'tendencia_cuadratica' in result.columns, "Debe tener tendencia_cuadratica"
        
        # Verificar features cíclicos
        assert 'ciclo_5_anos' in result.columns, "Debe tener ciclo_5_anos"
        assert 'ciclo_10_anos' in result.columns, "Debe tener ciclo_10_anos"
        
        # Verificar features de cambio
        assert 'cambio_nacimientos' in result.columns, "Debe tener cambio_nacimientos"
        assert 'cambio_defunciones' in result.columns, "Debe tener cambio_defunciones"
        assert 'cambio_crecimiento_poblacional' in result.columns, "Debe tener cambio_crecimiento_poblacional"
        
        # Verificar promedios móviles
        assert 'promedio_movil_nacimientos_3' in result.columns, "Debe tener promedio_movil_nacimientos_3"
        assert 'promedio_movil_defunciones_3' in result.columns, "Debe tener promedio_movil_defunciones_3"
        assert 'promedio_movil_crecimiento_3' in result.columns, "Debe tener promedio_movil_crecimiento_3"
        
        # Verificar volatilidad
        assert 'volatilidad_nacimientos_3' in result.columns, "Debe tener volatilidad_nacimientos_3"
        assert 'volatilidad_defunciones_3' in result.columns, "Debe tener volatilidad_defunciones_3"
        
        # Verificar percentiles
        assert 'percentil_nacimientos' in result.columns, "Debe tener percentil_nacimientos"
        assert 'percentil_defunciones' in result.columns, "Debe tener percentil_defunciones"
        
        # Verificar rangos de valores
        assert result['año_normalizado'].min() >= 0, "año_normalizado debe tener mínimo 0"
        assert result['año_normalizado'].max() <= 1, "año_normalizado debe tener máximo 1"
        assert result['percentil_nacimientos'].min() >= 0, "percentil_nacimientos debe tener mínimo 0"
        assert result['percentil_nacimientos'].max() <= 1, "percentil_nacimientos debe tener máximo 1"
    
    def test_normalizar_datos_para_modelado(self):
        """Test de normalización de datos."""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'año': [2014, 2015, 2016, 2017],
            'mes': [1, 2, 3, 4],
            'edad_cantidad': [25, 30, 35, 40],
            'mes_sin': [0.5, 0.8, 0.9, 0.7],
            'mes_cos': [0.8, 0.6, 0.4, 0.7]
        })
        
        # Importar función
        from src.proyecto_ml.pipelines.ciencia_datos.nodos import normalizar_datos_para_modelado
        
        # Parámetros mock
        params = {
            "metodos": ["StandardScaler", "MinMaxScaler", "RobustScaler"],
            "variables_numericas": ["año", "mes", "edad_cantidad"],
            "metodo_principal": "StandardScaler"
        }
        
        # Ejecutar función
        result = normalizar_datos_para_modelado(test_data, params)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        assert 'dataset_final_modelado' in result, "Debe tener dataset final para modelado"
        assert 'info_normalizacion' in result, "Debe tener información de normalización"
        
        # Verificar que el dataset final es un DataFrame
        assert isinstance(result['dataset_final_modelado'], pd.DataFrame), "dataset_final_modelado debe ser un DataFrame"
        assert not result['dataset_final_modelado'].empty, "dataset_final_modelado no debe estar vacío"
        
        # Verificar información de normalización
        info = result['info_normalizacion']
        assert 'total_variables' in info, "Debe tener total_variables"
        assert 'metodo_principal' in info, "Debe tener metodo_principal"
        assert 'shape_dataset' in info, "Debe tener shape_dataset"
    
    def test_crear_datasets_finales_para_modelado(self):
        """Test de creación de datasets finales."""
        # Crear datos de prueba
        datasets_normalizados = {
            'dataset_final_modelado': pd.DataFrame({
                'año': [2014, 2015, 2016, 2017],
                'mes': [1, 2, 3, 4],
                'edad_cantidad': [25, 30, 35, 40],
                'mes_sin': [0.5, 0.8, 0.9, 0.7],
                'mes_cos': [0.8, 0.6, 0.4, 0.7],
                'año_normalizado': [0.0, 0.33, 0.67, 1.0],
                'epoca_año_codificada': [1, 1, 2, 2],
                'es_fin_semana': [0, 0, 0, 0],
                'es_invierno': [1, 1, 0, 0],
                'es_verano': [0, 0, 0, 0],
                'trimestre_fiscal': [1, 1, 1, 2]
            }),
            'info_normalizacion': {
                'total_variables': 3,
                'metodo_principal': 'StandardScaler',
                'shape_dataset': (4, 11)
            },
            'scalers': {}
        }
        
        # Importar función
        from src.proyecto_ml.pipelines.ciencia_datos.nodos import crear_datasets_finales_para_modelado
        
        # Parámetros mock
        params = {
            "regresion": {
                "variables_objetivo": ["edad_cantidad"],
                "variables_predictoras": ["mes_sin", "mes_cos"]
            },
            "clasificacion": {
                "variables_objetivo": ["sexo"],
                "variables_predictoras": ["edad_cantidad", "mes_sin"]
            }
        }
        
        # Ejecutar función
        result = crear_datasets_finales_para_modelado(datasets_normalizados, params)
        
        # Verificar resultados
        assert isinstance(result, dict), "Debe retornar un diccionario"
        
        # Verificar datasets finales esperados
        datasets_esperados = [
            'dataset_regresion', 'dataset_temporal', 'dataset_indexado',
            'dataset_resumido', 'dataset_completo'
        ]
        
        for dataset in datasets_esperados:
            assert dataset in result, f"Debe tener dataset: {dataset}"
            assert isinstance(result[dataset], pd.DataFrame), f"{dataset} debe ser un DataFrame"
            assert not result[dataset].empty, f"{dataset} no debe estar vacío"
        
        # Verificar que todos los datasets tienen datos válidos
        for dataset_name, dataset in result.items():
            if isinstance(dataset, pd.DataFrame):
                assert len(dataset) > 0, f"{dataset_name} debe tener registros"
                assert len(dataset.columns) > 0, f"{dataset_name} debe tener columnas"
