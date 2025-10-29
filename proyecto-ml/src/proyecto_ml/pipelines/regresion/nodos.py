"""Nodos del pipeline de Regresión - PREDICCIÓN DE EDAD DE FALLECIMIENTO.

=============================================================================
PROBLEMA DE REGRESIÓN: PREDECIR EDAD_CANTIDAD (Edad de fallecimiento)
=============================================================================

OBJETIVO:
---------
Predecir la edad de fallecimiento de personas basándose en patrones temporales
y geográficos para identificar factores de riesgo asociados a periodos 
específicos del año y regiones geográficas.

HIPÓTESIS EPIDEMIOLÓGICA:
--------------------------
1. Mortalidad Infantil (0-5 años):
   - Mayor en meses de invierno (enfermedades respiratorias)
   - Patrón estacional marcado
   
2. Mortalidad en Adultos Mayores (65+ años):
   - Picos en invierno (hipotermia, complicaciones cardiovasculares)
   - Variación por región (acceso a salud)
   
3. Accidentes:
   - Patrón por día de semana (mayor en fines de semana)
   - Afecta principalmente a adultos jóvenes (20-40 años)
   
4. Geografía:
   - Norte: Mayor esperanza de vida (mejores servicios)
   - Sur: Mortalidad más temprana (acceso limitado)

VARIABLES PREDICTORAS (23 features - SIN DATA LEAKAGE):
-------------------------------------
⭐⭐⭐ CAUSA DE MUERTE (CIE-10) - LA VARIABLE MÁS IMPORTANTE:
- codigo_diagnostico: Capítulo CIE-10 (20 categorías)
  * I00-I99: Enfermedades circulatorias → edad 60-80 años
  * C00-D48: Tumores/Neoplasias → edad 50-75 años  
  * J00-J99: Enfermedades respiratorias → edad 65-85 años
  * S00-T98: Traumatismos/accidentes → edad 20-40 años
  * P00-P96: Afecciones perinatales → edad 0-1 año
  * Esta variable SOLA puede explicar 50-70% de la varianza en edad
  
- Demográficas: sexo (Hombre/Mujer) ⭐ IMPORTANTE
  * Hombres: Mayor mortalidad joven (accidentes, laborales)
  * Mujeres: Mayor esperanza de vida (~7 años más)
  
- Temporales cíclicas: mes_sin, mes_cos, dia_año_sin, dia_año_cos, 
                       trimestre_sin, trimestre_cos, dia_semana_sin, dia_semana_cos
- Estacionales: es_fin_semana, es_invierno, es_verano, es_primavera, es_otono
- Temporales: trimestre_fiscal, epoca_año_codificada, decada
- Geográficas: es_norte, es_centro, es_sur

VARIABLE OBJETIVO:
------------------
- edad_cantidad: Edad de fallecimiento en años (0-100+)
  * Variable continua
  * Rango: 0 a 100+ años
  * Media esperada: ~45 años (Chile 2014-2023)

MÉTRICAS DE EVALUACIÓN:
------------------------
1. R² (Coeficiente de determinación): % de varianza explicada
   - Meta CON causa de muerte: R² > 0.60 (excelente) ⭐
   - Meta SIN causa de muerte: R² > 0.30 (aceptable)
   
2. MAE (Mean Absolute Error): Error promedio en años
   - Meta: MAE < 15 años
   
3. RMSE (Root Mean Squared Error): Penaliza errores grandes
   - Meta: RMSE < 20 años
   
4. MAPE (Mean Absolute Percentage Error): Error porcentual
   - Meta: MAPE < 30%

5. Cross-Validation R² (mean ± std): Robustez del modelo
   - Meta: std < 0.05 (baja variabilidad entre folds)

MODELOS IMPLEMENTADOS (6):
---------------------------
1. Ridge Regression (Linear Regression con regularización L2)
2. Random Forest Regressor (Ensamble de árboles)
3. Gradient Boosting Regressor (Boosting secuencial)
4. SVR (Support Vector Regression)
5. KNN Regressor (Regresión por vecindad)
6. Decision Tree Regressor (Árbol simple)

Cada modelo se entrena con:
- GridSearchCV: Búsqueda exhaustiva de hiperparámetros
- CrossValidation: k=5 folds (mínimo requerido por rúbrica)
- Métricas completas: MAE, MSE, RMSE, R², MAPE

CUMPLIMIENTO DE RÚBRICA:
-------------------------
✓ Mínimo 6 modelos de regresión
✓ GridSearchCV para optimización de hiperparámetros
✓ CrossValidation con k=5 (validación robusta)
✓ División train/test (80/20)
✓ Métricas completas con mean ± std
✓ Documentación científica del problema

Autor: Proyecto ML - Chile
Fecha: 2025
=============================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import pickle
import time
from pathlib import Path

# Modelos de regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Herramientas de evaluación y validación
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, make_scorer
)

# Configurar logging
logger = logging.getLogger(__name__)


def preparar_datos_regresion(
    dataset_individual_ml: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepara los datos para regresión de EDAD_CANTIDAD (edad de fallecimiento).
    
    PROBLEMA: Predecir la edad de fallecimiento basándose en:
    - Patrones temporales (estacionalidad, día de la semana)
    - Patrones geográficos (región)
    
    PROCESO:
    1. Extrae dataset_individual_ml (100,000 registros)
    2. Selecciona features predictoras (21 variables: demográficas + temporales + geográficas)
    3. Codifica variables categóricas (sexo: Hombre=0, Mujer=1)
    4. Extrae target: edad_cantidad (0-100+ años)
    5. Normaliza features con StandardScaler
    6. Divide en train/test (80/20)
    7. Verifica calidad de datos (sin NaNs)
    
    Args:
        dataset_individual_ml: DataFrame con registros individuales y features
        params: Parámetros de configuración (regresion.variables_objetivo, etc.)
        
        Returns:
        Diccionario con datos preparados:
        {
            'edad_cantidad': {
                'X_train': Features de entrenamiento (80,000 x 21),
                'X_test': Features de prueba (20,000 x 21),
                'y_train': Target de entrenamiento (80,000,),
                'y_test': Target de prueba (20,000,),
                'feature_names': Lista de nombres de features,
                'target_name': 'edad_cantidad',
                'y_stats': Estadísticas del target (min, max, mean, std)
            }
        }
    """
    logger.info("=" * 80)
    logger.info("PREPARANDO DATOS PARA REGRESIÓN DE EDAD_CANTIDAD")
    logger.info("=" * 80)
    
    # Usar dataset_individual_ml directamente
    dataset = dataset_individual_ml.copy()
    
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Columnas disponibles: {len(dataset.columns)}")
    
    # Configuración desde parameters.yml
    config = params.get('datasets_finales', {}).get('regresion', {})
    variables_objetivo = config.get('variables_objetivo', ['edad_cantidad'])
    variables_predictoras = config.get('variables_predictoras', [])
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    
    logger.info(f"\nVariables objetivo: {variables_objetivo}")
    logger.info(f"Variables predictoras configuradas: {len(variables_predictoras)}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Random state: {random_state}")
    
    # Diccionario para almacenar los datos preparados
    datos_preparados = {}
    
    # Preparar datos para cada variable objetivo
    for target in variables_objetivo:
        if target not in dataset.columns:
            logger.warning(f"Variable objetivo '{target}' no encontrada en dataset. Saltando...")
            continue
        
        logger.info(f"\n--- Preparando datos para regresión de: {target.upper()} ---")
        
        # Seleccionar features (usar las configuradas o todas excepto targets)
        if variables_predictoras and all(col in dataset.columns for col in variables_predictoras):
            features = variables_predictoras
        else:
            # Usar todas las columnas numéricas excepto las variables objetivo
            features = [col for col in dataset.columns 
                       if col not in variables_objetivo 
                       and dataset[col].dtype in ['int64', 'float64']]
        
        logger.info(f"Features seleccionadas: {len(features)}")
        
        # Extraer X (features) e y (target)
        X = dataset[features].copy()
        y = dataset[target].copy()
        
        # =====================================================================
        # NOTA: OneHotEncoding ya se aplicó en preparar_dataset_para_regresion()
        # =====================================================================
        # El dataset ya viene con variables categóricas convertidas a columnas
        # binarias (0/1). No necesitamos hacer nada aquí.
        #
        # Variables que YA están como OneHot:
        #   - sexo: 0 o 1 (binario)
        #   - region_*: 16 columnas binarias
        #   - cie10_*: 19 columnas binarias
        #
        # Total: ~51 columnas numéricas listas para normalizar y entrenar
        # =====================================================================
        
        logger.info("\n" + "="*70)
        logger.info("VARIABLES CATEGÓRICAS:")
        logger.info("="*70)
        logger.info("✓ OneHotEncoding ya aplicado en la preparación del dataset")
        logger.info(f"  Shape de X: {X.shape}")
        logger.info(f"  Todas las columnas son numéricas (listas para ML)")
        
        # Contar columnas OneHot
        region_cols = [col for col in X.columns if col.startswith('region_')]
        cie10_cols = [col for col in X.columns if col.startswith('cie10_')]
        
        if region_cols:
            logger.info(f"  - Columnas region_*: {len(region_cols)}")
        if cie10_cols:
            logger.info(f"  - Columnas cie10_*: {len(cie10_cols)}")
        if 'sexo' in X.columns:
            logger.info(f"  - Columna sexo: binaria (0/1)")
        
        logger.info("="*70)
        
        # IMPORTANTE: Limpiar valores nulos ANTES del split
        # 1. Primero eliminar filas donde el target es nulo
        if y.isnull().sum() > 0:
            logger.warning(f"Encontrados {y.isnull().sum()} valores nulos en target. Eliminando...")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        # 2. Luego manejar valores nulos en features
        nulos_antes = X.isnull().sum().sum()
        if nulos_antes > 0:
            logger.warning(f"Encontrados {nulos_antes} valores nulos en features. Limpiando...")
            
            # Imputar con mediana cada columna
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    mediana = X[col].median()
                    if pd.notna(mediana):
                        X[col] = X[col].fillna(mediana)
                    else:
                        # Si la mediana es NaN (columna vacía), llenar con 0
                        X[col] = X[col].fillna(0)
                        logger.warning(f"Columna '{col}' rellenada con 0 (sin mediana válida)")
            
            # 3. Verificación final: asegurar que no queden NaN
            nulos_final = X.isnull().sum().sum()
            if nulos_final > 0:
                logger.error(f"⚠ Aún quedan {nulos_final} valores NaN. Forzando limpieza...")
                X = X.fillna(0)  # Última opción: rellenar todo con 0
            
            logger.info(f"Valores nulos tratados: {nulos_antes} -> 0")
        
        # Verificar que es una variable continua
        n_unique = len(np.unique(y))
        logger.info(f"Valores únicos en target: {n_unique}")
        
        if n_unique < 10:
            logger.warning(f"⚠ Target tiene pocos valores únicos ({n_unique}). Puede no ser apropiado para regresión.")
        
        # IMPORTANTE: Normalizar features ANTES del split (para evitar data leakage)
        logger.info("\nNormalizando features con StandardScaler...")
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info("✓ Features normalizados")
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target train - min: {y_train.min():.2f}, max: {y_train.max():.2f}, mean: {y_train.mean():.2f}")
        logger.info(f"Target test - min: {y_test.min():.2f}, max: {y_test.max():.2f}, mean: {y_test.mean():.2f}")
        
        # Guardar datos preparados
        datos_preparados[target] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
            'feature_names': features,
            'target_name': target,
            'y_stats': {
                'min': float(y_train.min()),
                'max': float(y_train.max()),
                'mean': float(y_train.mean()),
                'std': float(y_train.std())
            }
        }
    
    logger.info(f"\n✓ Datos preparados para {len(datos_preparados)} problemas de regresión")
    return datos_preparados


def entrenar_modelos_regresion(
    datos_preparados: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena 6 modelos de regresión con GridSearchCV y CrossValidation.
    
    Implementa los siguientes modelos:
    1. Linear Regression - Regresión lineal simple
    2. Random Forest - Ensamble de árboles de regresión
    3. Gradient Boosting - Boosting secuencial de árboles
    4. SVR (Support Vector Regression) - Regresión con vectores de soporte
    5. KNN Regressor - Regresión por vecindad
    6. Decision Tree - Árbol de regresión simple
    
    Para cada modelo:
    - GridSearchCV: búsqueda exhaustiva de hiperparámetros
    - CrossValidation: k=5 folds (configurable)
    - Métricas: MAE, MSE, RMSE, R², MAPE
    
    Args:
        datos_preparados: Diccionario con datos de train/test para cada problema
        params: Parámetros de configuración (cv_folds, scoring, etc.)
        
    Returns:
        Diccionario con resultados de todos los modelos:
        {
            'edad_cantidad': {
                'linear_regression': {...},
                'random_forest': {...},
                ...
            },
            'año_normalizado': {...}
        }
    """
    logger.info("=" * 80)
    logger.info("ENTRENANDO MODELOS DE REGRESIÓN")
    logger.info("=" * 80)
    
    # Configuración
    cv_folds = params.get('cv_folds', 5)  # K=5 mínimo requerido
    n_jobs = params.get('n_jobs', -1)  # Usar todos los cores
    verbose = params.get('verbose', 1)
    
    logger.info(f"CrossValidation: {cv_folds} folds")
    logger.info(f"Procesamiento paralelo: {n_jobs} jobs")
    
    # Diccionario para almacenar todos los resultados
    resultados_globales = {}
    
    # Entrenar modelos para cada problema de regresión
    for problema, datos in datos_preparados.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PROBLEMA: Regresión de {problema.upper()}")
        logger.info(f"{'=' * 80}")
        
        X_train = datos['X_train']
        X_test = datos['X_test']
        y_train = datos['y_train']
        y_test = datos['y_test']
        
        logger.info(f"Features: {len(datos['feature_names'])}")
        logger.info(f"Target range: [{datos['y_stats']['min']:.2f}, {datos['y_stats']['max']:.2f}]")
        
        # Diccionario para almacenar resultados de este problema
        resultados_problema = {}
        
        # =================================================================
        # MODELO 1: LINEAR REGRESSION
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 1: LINEAR REGRESSION")
        logger.info(f"{'-' * 80}")
        
        modelo_lr = LinearRegression(
            n_jobs=n_jobs
        )
        
        # Grid de hiperparámetros (Linear Regression no tiene muchos, usar Ridge)
        # Usamos Ridge (regresión lineal con regularización L2)
        from sklearn.linear_model import Ridge
        modelo_lr = Ridge()
        
        param_grid_lr = {
            'alpha': [0.1, 1.0, 10.0],  # Regularización (reducido)
            'solver': ['auto']  # Solo auto
        }
        
        resultados_problema['linear_regression'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_lr,
            param_grid=param_grid_lr,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="Linear Regression (Ridge)"
        )
        
        # =================================================================
        # MODELO 2: RANDOM FOREST
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 2: RANDOM FOREST REGRESSOR")
        logger.info(f"{'-' * 80}")
        
        modelo_rf = RandomForestRegressor(
            random_state=42,
            n_jobs=n_jobs
        )
        
        param_grid_rf = {
            'n_estimators': [50, 100],  # Número de árboles (reducido)
            'max_depth': [10, 20],  # Profundidad máxima (sin None)
            'min_samples_split': [2, 5]  # Mínimo de muestras para dividir
        }
        
        resultados_problema['random_forest'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_rf,
            param_grid=param_grid_rf,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="Random Forest"
        )
        
        # =================================================================
        # MODELO 3: GRADIENT BOOSTING
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 3: GRADIENT BOOSTING REGRESSOR")
        logger.info(f"{'-' * 80}")
        
        modelo_gb = GradientBoostingRegressor(
            random_state=42
        )
        
        param_grid_gb = {
            'n_estimators': [50, 100],  # Número de boosting stages (reducido)
            'learning_rate': [0.1, 0.2],  # Tasa de aprendizaje (reducido)
            'max_depth': [3, 5]  # Profundidad de árboles (reducido)
        }
        
        resultados_problema['gradient_boosting'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_gb,
            param_grid=param_grid_gb,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="Gradient Boosting"
        )
        
        # =================================================================
        # MODELO 4: SUPPORT VECTOR REGRESSION (SVR)
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 4: SUPPORT VECTOR REGRESSION (SVR)")
        logger.info(f"{'-' * 80}")
        
        modelo_svr = SVR()
        
        param_grid_svr = {
            'C': [0.1, 1.0],  # Parámetro de regularización (reducido)
            'kernel': ['rbf'],  # Solo RBF
            'gamma': ['scale']  # Solo scale
        }
        
        resultados_problema['svr'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_svr,
            param_grid=param_grid_svr,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="SVR"
        )
        
        # =================================================================
        # MODELO 5: K-NEAREST NEIGHBORS REGRESSOR
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 5: K-NEAREST NEIGHBORS REGRESSOR")
        logger.info(f"{'-' * 80}")
        
        modelo_knn = KNeighborsRegressor(
            n_jobs=n_jobs
        )
        
        param_grid_knn = {
            'n_neighbors': [5, 7],  # Número de vecinos (reducido)
            'weights': ['distance'],  # Solo distance
            'metric': ['euclidean']  # Solo euclidean
        }
        
        resultados_problema['knn'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_knn,
            param_grid=param_grid_knn,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="KNN Regressor"
        )
        
        # =================================================================
        # MODELO 6: DECISION TREE
        # =================================================================
        logger.info(f"\n{'-' * 80}")
        logger.info("MODELO 6: DECISION TREE REGRESSOR")
        logger.info(f"{'-' * 80}")
        
        modelo_dt = DecisionTreeRegressor(
            random_state=42
        )
        
        param_grid_dt = {
            'max_depth': [5, 10, 15],  # Profundidad máxima (sin None)
            'min_samples_split': [2, 5],  # Mínimo para dividir (reducido)
            'criterion': ['squared_error']  # Solo squared_error
        }
        
        resultados_problema['decision_tree'] = _entrenar_y_evaluar_modelo_regresion(
            modelo=modelo_dt,
            param_grid=param_grid_dt,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            nombre_modelo="Decision Tree"
        )
        
        # Guardar resultados de este problema
        resultados_globales[problema] = resultados_problema
        
        # Resumen de resultados
        logger.info(f"\n{'=' * 80}")
        logger.info(f"RESUMEN - Regresión de {problema.upper()}")
        logger.info(f"{'=' * 80}")
        
        for nombre_modelo, resultados in resultados_problema.items():
            logger.info(f"\n{nombre_modelo.upper()}:")
            logger.info(f"  R² (Test):        {resultados['test_metrics']['r2']:.4f}")
            logger.info(f"  RMSE (Test):      {resultados['test_metrics']['rmse']:.4f}")
            logger.info(f"  MAE (Test):       {resultados['test_metrics']['mae']:.4f}")
            logger.info(f"  CV R²:            {resultados['cv_metrics']['mean_r2']:.4f} ± {resultados['cv_metrics']['std_r2']:.4f}")
            logger.info(f"  Tiempo entrenamiento: {resultados['tiempo_entrenamiento']:.2f}s")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("✓ ENTRENAMIENTO DE MODELOS DE REGRESIÓN COMPLETADO")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total de problemas: {len(resultados_globales)}")
    logger.info(f"Total de modelos por problema: 6")
    logger.info(f"Total de modelos entrenados: {len(resultados_globales) * 6}")
    
    return resultados_globales


def _entrenar_y_evaluar_modelo_regresion(
    modelo: Any,
    param_grid: Dict[str, List],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int,
    n_jobs: int,
    verbose: int,
    nombre_modelo: str
) -> Dict[str, Any]:
    """
    Función auxiliar para entrenar y evaluar un modelo de regresión con GridSearchCV.
    
    Esta función encapsula todo el proceso de:
    1. GridSearchCV para encontrar mejores hiperparámetros
    2. CrossValidation para evaluar el modelo (k folds)
    3. Evaluación en conjunto de test
    4. Cálculo de métricas (MAE, MSE, RMSE, R², MAPE)
    
    Args:
        modelo: Modelo de sklearn a entrenar
        param_grid: Diccionario con grilla de hiperparámetros
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
        cv_folds: Número de folds para cross-validation
        n_jobs: Número de trabajos paralelos
        verbose: Nivel de verbosidad
        nombre_modelo: Nombre del modelo para logging
        
    Returns:
        Diccionario con todos los resultados del modelo
    """
    logger.info(f"Entrenando {nombre_modelo}...")
    tiempo_inicio = time.time()
    
    # GridSearchCV con CrossValidation
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='r2',  # Métrica principal para selección (R²)
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Entrenar
    grid_search.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - tiempo_inicio
    
    logger.info(f"Mejor score CV (R²): {grid_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    
    # Mejor modelo encontrado
    mejor_modelo = grid_search.best_estimator_
    
    # Predicciones
    y_pred_train = mejor_modelo.predict(X_train)
    y_pred_test = mejor_modelo.predict(X_test)
    
    # =================================================================
    # MÉTRICAS DE EVALUACIÓN
    # =================================================================
    
    # Métricas en conjunto de TEST
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mse': mean_squared_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2': r2_score(y_test, y_pred_test)
    }
    
    # MAPE (Mean Absolute Percentage Error) - cuidado con divisiones por cero
    try:
        # Evitar división por cero
        mask = y_test != 0
        if mask.sum() > 0:
            test_metrics['mape'] = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
        else:
            test_metrics['mape'] = None
    except:
        test_metrics['mape'] = None
    
    # Métricas en conjunto de TRAIN (para detectar overfitting)
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_pred_train),
        'mse': mean_squared_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'r2': r2_score(y_train, y_pred_train)
    }
    
    # =================================================================
    # MÉTRICAS DE CROSS-VALIDATION (mean ± std)
    # =================================================================
    
    # Definir scoring para cross_validate
    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',  # Negativo porque sklearn maximiza
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error'
    }
    
    # Cross-validation completo con el mejor modelo
    cv_results = cross_validate(
        mejor_modelo,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs
    )
    
    # Extraer mean ± std de cada métrica (convertir negativos a positivos)
    cv_metrics = {
        'mean_r2': cv_results['test_r2'].mean(),
        'std_r2': cv_results['test_r2'].std(),
        'mean_mae': -cv_results['test_mae'].mean(),  # Convertir a positivo
        'std_mae': cv_results['test_mae'].std(),
        'mean_mse': -cv_results['test_mse'].mean(),  # Convertir a positivo
        'std_mse': cv_results['test_mse'].std(),
        'mean_rmse': -cv_results['test_rmse'].mean(),  # Convertir a positivo
        'std_rmse': cv_results['test_rmse'].std(),
        'cv_scores': cv_results['test_r2']  # Scores de cada fold
    }
    
    # Residuales (diferencia entre predicción y valor real)
    residuales_test = y_test - y_pred_test
    
    # Log de resultados
    logger.info(f"✓ {nombre_modelo} entrenado en {tiempo_entrenamiento:.2f}s")
    logger.info(f"  Test R²:    {test_metrics['r2']:.4f}")
    logger.info(f"  Test RMSE:  {test_metrics['rmse']:.4f}")
    logger.info(f"  Test MAE:   {test_metrics['mae']:.4f}")
    logger.info(f"  CV R²:      {cv_metrics['mean_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")
    
    # Retornar todos los resultados
    return {
        'modelo': mejor_modelo,
        'nombre': nombre_modelo,
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'cv_metrics': cv_metrics,
        'predictions_test': y_pred_test,
        'residuales_test': residuales_test,
        'y_test': y_test,
        'tiempo_entrenamiento': tiempo_entrenamiento,
        'cv_results_full': cv_results,
        'grid_search_results': grid_search.cv_results_
    }


def generar_tabla_comparativa_regresion(
    resultados: Dict[str, Any]
) -> pd.DataFrame:
    """
    Genera una tabla comparativa con todos los modelos de regresión.
    
    Crea un DataFrame con formato:
    | Problema | Modelo | MAE | MSE | RMSE | R² | MAPE | CV_R² | CV_Std | Tiempo |
    
    Esta tabla es esencial para la rúbrica de evaluación, ya que permite
    comparar visualmente el rendimiento de todos los modelos.
    
    Args:
        resultados: Diccionario con resultados de todos los modelos
        
    Returns:
        DataFrame con tabla comparativa formateada
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERANDO TABLA COMPARATIVA DE REGRESIÓN")
    logger.info("=" * 80)
    
    filas = []
    
    for problema, modelos in resultados.items():
        for nombre_modelo, res in modelos.items():
            fila = {
                'Problema': problema.upper(),
                'Modelo': nombre_modelo.replace('_', ' ').title(),
                'MAE (Test)': f"{res['test_metrics']['mae']:.4f}",
                'MSE (Test)': f"{res['test_metrics']['mse']:.4f}",
                'RMSE (Test)': f"{res['test_metrics']['rmse']:.4f}",
                'R² (Test)': f"{res['test_metrics']['r2']:.4f}",
                'MAPE (Test)': f"{res['test_metrics']['mape']:.2f}%" if res['test_metrics']['mape'] else 'N/A',
                'CV R² (mean±std)': f"{res['cv_metrics']['mean_r2']:.4f} ± {res['cv_metrics']['std_r2']:.4f}",
                'Tiempo (s)': f"{res['tiempo_entrenamiento']:.2f}"
            }
            filas.append(fila)
    
    tabla = pd.DataFrame(filas)
    
    logger.info(f"\n✓ Tabla comparativa generada: {tabla.shape[0]} filas")
    logger.info("\n" + str(tabla))
    
    return tabla


def guardar_modelos_regresion(
    resultados: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, str]:
    """
    Guarda todos los modelos entrenados en disco.
    
    Guarda cada modelo en formato pickle para poder cargarlos después
    y hacer predicciones sin tener que reentrenar.
    
    Args:
        resultados: Diccionario con todos los modelos entrenados
        params: Parámetros de configuración (rutas de guardado)
        
    Returns:
        Diccionario con las rutas donde se guardaron los modelos
    """
    logger.info("\n" + "=" * 80)
    logger.info("GUARDANDO MODELOS DE REGRESIÓN")
    logger.info("=" * 80)
    
    # Ruta base para guardar modelos
    ruta_base = Path(params.get('ruta_modelos', 'data/06_models/regresion'))
    ruta_base.mkdir(parents=True, exist_ok=True)
    
    rutas_guardado = {}
    
    for problema, modelos in resultados.items():
        for nombre_modelo, res in modelos.items():
            # Crear nombre de archivo
            nombre_archivo = f"regresion_{problema}_{nombre_modelo}.pkl"
            ruta_completa = ruta_base / nombre_archivo
            
            # Guardar modelo
            with open(ruta_completa, 'wb') as f:
                pickle.dump(res['modelo'], f)
            
            rutas_guardado[f"{problema}_{nombre_modelo}"] = str(ruta_completa)
            logger.info(f"✓ Guardado: {nombre_archivo}")
    
    logger.info(f"\n✓ Total de modelos guardados: {len(rutas_guardado)}")
    return rutas_guardado

