"""Tests para el pipeline de Regresión."""

import pytest
import pandas as pd
import numpy as np
from proyecto_ml.pipelines.regresion.nodos import (
    preparar_datos_regresion,
    entrenar_modelos_regresion,
    generar_tabla_comparativa_regresion,
    guardar_modelos_regresion
)


class TestRegresion:
    """Tests para nodos del pipeline de regresión."""
    
    def test_preparar_datos_regresion(self):
        """Test de preparación de datos para regresión."""
        # Crear datos de prueba
        np.random.seed(42)
        n_samples = 100
        
        dataset_prueba = pd.DataFrame({
            'edad_cantidad': np.random.randint(0, 100, n_samples).astype(float),
            'año_normalizado': np.random.rand(n_samples),
            'mes_sin': np.random.rand(n_samples),
            'mes_cos': np.random.rand(n_samples),
            'dia_año_sin': np.random.rand(n_samples),
            'dia_año_cos': np.random.rand(n_samples)
        })
        
        datos_finales = {'dataset_final': dataset_prueba}
        
        params = {
            'regresion': {
                'variables_objetivo': ['edad_cantidad'],
                'variables_predictoras': ['mes_sin', 'mes_cos', 'dia_año_sin', 'dia_año_cos']
            },
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Ejecutar función
        resultado = preparar_datos_regresion(datos_finales, params)
        
        # Verificaciones
        assert isinstance(resultado, dict)
        assert 'edad_cantidad' in resultado
        assert 'X_train' in resultado['edad_cantidad']
        assert 'X_test' in resultado['edad_cantidad']
        assert 'y_train' in resultado['edad_cantidad']
        assert 'y_test' in resultado['edad_cantidad']
        
        print("✓ Test preparar_datos_regresion pasado")
    
    def test_generar_tabla_comparativa_regresion(self):
        """Test de generación de tabla comparativa."""
        # Crear resultados de prueba simulados
        resultados_prueba = {
            'edad_cantidad': {
                'modelo_test': {
                    'nombre': 'Test Model',
                    'test_metrics': {
                        'mae': 5.2,
                        'mse': 42.3,
                        'rmse': 6.5,
                        'r2': 0.75,
                        'mape': 12.5
                    },
                    'cv_metrics': {
                        'mean_r2': 0.73,
                        'std_r2': 0.03
                    },
                    'tiempo_entrenamiento': 2.1
                }
            }
        }
        
        # Ejecutar función
        tabla = generar_tabla_comparativa_regresion(resultados_prueba)
        
        # Verificaciones
        assert isinstance(tabla, pd.DataFrame)
        assert len(tabla) > 0
        assert 'Problema' in tabla.columns
        assert 'Modelo' in tabla.columns
        assert 'R² (Test)' in tabla.columns
        assert 'RMSE (Test)' in tabla.columns
        
        print("✓ Test generar_tabla_comparativa_regresion pasado")
    
    def test_tipos_salida_correctos(self):
        """Test que verifica los tipos de salida."""
        np.random.seed(42)
        n_samples = 50
        
        dataset = pd.DataFrame({
            'target': np.random.rand(n_samples) * 100,
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples),
            'feature3': np.random.rand(n_samples)
        })
        
        datos_finales = {'dataset_final': dataset}
        params = {
            'regresion': {
                'variables_objetivo': ['target'],
                'variables_predictoras': ['feature1', 'feature2', 'feature3']
            },
            'test_size': 0.2,
            'random_state': 42
        }
        
        resultado = preparar_datos_regresion(datos_finales, params)
        
        assert isinstance(resultado, dict)
        assert all(isinstance(v, dict) for v in resultado.values())
        
        print("✓ Test tipos_salida_correctos pasado")


# Ejecutar tests si se corre directamente
if __name__ == "__main__":
    test = TestRegresion()
    test.test_preparar_datos_regresion()
    test.test_generar_tabla_comparativa_regresion()
    test.test_tipos_salida_correctos()
    print("\n✓ Todos los tests de regresión pasaron correctamente")


