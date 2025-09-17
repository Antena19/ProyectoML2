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
    
    def test_crear_features_temporales_avanzadas(self):
        """Test de creación de features temporales avanzadas."""
        # Crear datos de prueba
        datasets_estandarizados = {
            'defunciones_estandarizado': pd.DataFrame({
                'año': [2014, 2015, 2016, 2017],
                'mes': [1, 2, 3, 4],
                'dia_año': [1, 32, 60, 91],
                'trimestre': [1, 1, 1, 2],
                'dia_semana': [1, 2, 3, 4],
                'fecha_defuncion': pd.to_datetime(['2014-01-01', '2015-02-01', '2016-03-01', '2017-04-01']),
                'tipo_edad': ['Años', 'Años', 'Años', 'Años'],
                'edad_cantidad': [25, 30, 35, 40]
            })
        }
        
        # Importar función
        from src.proyecto_ml.pipelines.ciencia_datos.nodos import crear_features_temporales_avanzadas
        
        # Ejecutar función
        result = crear_features_temporales_avanzadas(datasets_estandarizados)
        
        # Verificar resultados
        assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
        assert not result.empty, "El resultado no debe estar vacío"
        
        # Verificar features cíclicos
        ciclicos_esperados = ['mes_sin', 'mes_cos', 'dia_año_sin', 'dia_año_cos', 
                             'trimestre_sin', 'trimestre_cos', 'dia_semana_sin', 'dia_semana_cos']
        
        for feature in ciclicos_esperados:
            assert feature in result.columns, f"Debe tener feature cíclico: {feature}"
        
        # Verificar features especiales
        assert 'es_fin_semana' in result.columns, "Debe tener feature es_fin_semana"
        assert 'es_invierno' in result.columns, "Debe tener feature es_invierno"
        assert 'es_verano' in result.columns, "Debe tener feature es_verano"
        
        # Verificar que los valores cíclicos están en el rango correcto (solo si no son NaN)
        for feature in ciclicos_esperados:
            if not result[feature].isna().all():
                assert result[feature].min() >= -1, f"{feature} debe tener mínimo -1"
                assert result[feature].max() <= 1, f"{feature} debe tener máximo 1"
    
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
        
        # Ejecutar función
        result = normalizar_datos_para_modelado(test_data)
        
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
        
        # Ejecutar función
        result = crear_datasets_finales_para_modelado(datasets_normalizados)
        
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
