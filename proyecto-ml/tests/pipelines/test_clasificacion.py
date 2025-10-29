"""Tests para el pipeline de Clasificación."""

import pytest
import pandas as pd
import numpy as np
from proyecto_ml.pipelines.clasificacion.nodos import (
    preparar_datos_clasificacion,
    entrenar_modelos_clasificacion,
    generar_tabla_comparativa_clasificacion,
    guardar_modelos_clasificacion
)


class TestClasificacion:
    """Tests para nodos del pipeline de clasificación."""
    
    def test_preparar_datos_clasificacion(self):
        """Test de preparación de datos para clasificación."""
        # Crear datos de prueba
        np.random.seed(42)
        n_samples = 100
        
        dataset_prueba = pd.DataFrame({
            'sexo': np.random.choice(['Hombre', 'Mujer'], n_samples),
            'region': np.random.choice(['Region1', 'Region2', 'Region3'], n_samples),
            'edad_cantidad': np.random.randint(0, 100, n_samples),
            'mes_sin': np.random.rand(n_samples),
            'mes_cos': np.random.rand(n_samples),
            'dia_año_sin': np.random.rand(n_samples),
            'dia_año_cos': np.random.rand(n_samples)
        })
        
        datos_finales = {'dataset_final': dataset_prueba}
        
        params = {
            'clasificacion': {
                'variables_objetivo': ['sexo'],
                'variables_predictoras': ['edad_cantidad', 'mes_sin', 'mes_cos']
            },
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Ejecutar función
        resultado = preparar_datos_clasificacion(datos_finales, params)
        
        # Verificaciones
        assert isinstance(resultado, dict)
        assert 'sexo' in resultado
        assert 'X_train' in resultado['sexo']
        assert 'X_test' in resultado['sexo']
        assert 'y_train' in resultado['sexo']
        assert 'y_test' in resultado['sexo']
        
        print("✓ Test preparar_datos_clasificacion pasado")
    
    def test_generar_tabla_comparativa_clasificacion(self):
        """Test de generación de tabla comparativa."""
        # Crear resultados de prueba simulados
        resultados_prueba = {
            'sexo': {
                'modelo_test': {
                    'nombre': 'Test Model',
                    'test_metrics': {
                        'accuracy': 0.85,
                        'precision': 0.83,
                        'recall': 0.82,
                        'f1_score': 0.84,
                        'roc_auc': 0.86
                    },
                    'cv_metrics': {
                        'mean_accuracy': 0.84,
                        'std_accuracy': 0.02
                    },
                    'tiempo_entrenamiento': 1.5
                }
            }
        }
        
        # Ejecutar función
        tabla = generar_tabla_comparativa_clasificacion(resultados_prueba)
        
        # Verificaciones
        assert isinstance(tabla, pd.DataFrame)
        assert len(tabla) > 0
        assert 'Problema' in tabla.columns
        assert 'Modelo' in tabla.columns
        assert 'Accuracy (Test)' in tabla.columns
        
        print("✓ Test generar_tabla_comparativa_clasificacion pasado")
    
    def test_tipos_salida_correctos(self):
        """Test que verifica los tipos de salida."""
        # Verificar que las funciones retornan tipos correctos
        np.random.seed(42)
        n_samples = 50
        
        dataset = pd.DataFrame({
            'sexo': np.random.choice(['Hombre', 'Mujer'], n_samples),
            'edad': np.random.randint(20, 80, n_samples),
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples)
        })
        
        datos_finales = {'dataset_final': dataset}
        params = {
            'clasificacion': {
                'variables_objetivo': ['sexo'],
                'variables_predictoras': ['edad', 'feature1', 'feature2']
            },
            'test_size': 0.2,
            'random_state': 42
        }
        
        resultado = preparar_datos_clasificacion(datos_finales, params)
        
        assert isinstance(resultado, dict)
        assert all(isinstance(v, dict) for v in resultado.values())
        
        print("✓ Test tipos_salida_correctos pasado")


# Ejecutar tests si se corre directamente
if __name__ == "__main__":
    test = TestClasificacion()
    test.test_preparar_datos_clasificacion()
    test.test_generar_tabla_comparativa_clasificacion()
    test.test_tipos_salida_correctos()
    print("\n✓ Todos los tests de clasificación pasaron correctamente")

