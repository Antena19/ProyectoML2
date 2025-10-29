"""Nodos del pipeline de Clasificación - VERSIÓN SIMPLIFICADA PARA SEXO.

Este módulo implementa 6 modelos de clasificación con GridSearchCV y CrossValidation
SOLO para clasificación de SEXO (binario):
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVC)
5. K-Nearest Neighbors (KNN)
6. Decision Tree Classifier

Cada modelo se entrena con búsqueda de hiperparámetros y validación cruzada (k=5).

================================================================================
JUSTIFICACIÓN CIENTÍFICA: PATRONES DE MORTALIDAD POR SEXO
================================================================================

BASADO EN EVIDENCIA EPIDEMIOLÓGICA REAL:

1. MORTALIDAD JOVEN (15-35 años):
   - HOMBRES: 5.2% mueren jóvenes vs MUJERES: 2.2%
   - DIFERENCIA: 2.4x más hombres mueren jóvenes
   - CAUSAS: Accidentes, violencia, comportamientos de riesgo, suicidios
   - VARIABLE: riesgo_mortalidad_joven

2. MORTALIDAD ADULTA (35-65 años):
   - HOMBRES: 28.5% mueren en edad adulta vs MUJERES: 18.1%
   - DIFERENCIA: 1.6x más hombres mueren en edad adulta
   - CAUSAS: Enfermedades laborales, estrés, comportamientos de riesgo
   - VARIABLE: riesgo_mortalidad_adulto

3. MORTALIDAD MAYOR (65+ años):
   - HOMBRES: 65.1% mueren mayores vs MUJERES: 78.4%
   - DIFERENCIA: Las mujeres viven 6.9 años más en promedio
   - CAUSAS: Mayor esperanza de vida femenina, mejor cuidado de salud
   - VARIABLE: riesgo_mortalidad_mayor

4. EDAD PROMEDIO DE MUERTE:
   - HOMBRES: 68.9 años (mediana: 73 años)
   - MUJERES: 75.8 años (mediana: 80 años)
   - DIFERENCIA: 6.9 años más de vida para mujeres
   - VARIABLES: edad_cantidad, edad_normalizada, desviacion_edad_*

VARIABLES SELECCIONADAS (11 variables relevantes):
- edad_cantidad: Edad exacta de fallecimiento (0-118 años)
- edad_normalizada: Edad normalizada 0-1
- desviacion_edad_hombres: Desviación respecto a edad promedio de hombres (65 años)
- desviacion_edad_mujeres: Desviación respecto a edad promedio de mujeres (75 años)
- riesgo_mortalidad_joven: Indicador de muerte joven (15-35 años) - MÁS HOMBRES
- riesgo_mortalidad_adulto: Indicador de muerte adulta (35-65 años) - MÁS HOMBRES
- riesgo_mortalidad_mayor: Indicador de muerte mayor (65+ años) - MÁS MUJERES
- es_menor_edad: Menor de 18 años
- es_adulto_joven: 18-30 años
- es_adulto_maduro: 30-65 años
- es_adulto_mayor: 65+ años

EXPECTATIVAS DE RENDIMIENTO:
- Accuracy esperado: 70-80% (vs 60% con variables irrelevantes)
- Mejora por: Eliminación de 30 variables de ruido
- Justificación: Patrones epidemiológicos reales y medibles
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import pickle
import time
from pathlib import Path

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Herramientas de evaluación y validación
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configurar logging
logger = logging.getLogger(__name__)


def preparar_datos_clasificacion(
    dataset_individual_ml: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepara los datos para clasificación de SEXO únicamente.
    
    Esta función implementa un enfoque científico basado en patrones epidemiológicos
    reales de mortalidad por sexo. Utiliza 11 variables cuidadosamente seleccionadas
    que capturan las diferencias biológicas y sociales entre hombres y mujeres.
    
    JUSTIFICACIÓN CIENTÍFICA:
    - Hombres mueren 6.9 años más jóvenes que mujeres en promedio
    - Hombres tienen 2.4x más mortalidad joven (accidentes, violencia)
    - Hombres tienen 1.6x más mortalidad adulta (enfermedades laborales)
    - Mujeres tienen mayor mortalidad mayor (78.4% vs 65.1% en 65+ años)
    
    VARIABLES UTILIZADAS:
    1. edad_cantidad: Edad exacta (0-118 años) - PRINCIPAL PREDICTOR
    2. edad_normalizada: Edad normalizada 0-1
    3. desviacion_edad_hombres: Desviación respecto a edad promedio hombres (65 años)
    4. desviacion_edad_mujeres: Desviación respecto a edad promedio mujeres (75 años)
    5. riesgo_mortalidad_joven: Indicador muerte 15-35 años (MÁS HOMBRES)
    6. riesgo_mortalidad_adulto: Indicador muerte 35-65 años (MÁS HOMBRES)
    7. riesgo_mortalidad_mayor: Indicador muerte 65+ años (MÁS MUJERES)
    8. es_menor_edad: Menor de 18 años
    9. es_adulto_joven: 18-30 años
    10. es_adulto_maduro: 30-65 años
    11. es_adulto_mayor: 65+ años
    
    Args:
        dataset_individual_ml: Dataset individual con 100K registros estratificados
        params: Parámetros de configuración (test_size, random_state, etc.)
        
    Returns:
        Diccionario con datos preparados para entrenamiento:
        - X_train, X_test: Features normalizadas
        - y_train, y_test: Target codificado (0=Hombre, 1=Mujer)
        - feature_names: Nombres de las 11 variables
        - label_encoder: Codificador para interpretar resultados
        - clases: ['Hombre', 'Mujer']
        - n_clases: 2 (clasificación binaria)
    """
    logger.info("=" * 80)
    logger.info("PREPARANDO DATOS PARA CLASIFICACION DE SEXO")
    logger.info("=" * 80)
    
    # Configuración
    config = params.get('clasificacion', {})
    variables_predictoras = config.get('variables_predictoras', [])
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    
    logger.info(f"Variables predictoras: {len(variables_predictoras)}")
    logger.info(f"Test size: {test_size}")
    
    # Verificar que las variables existen
    missing_vars = [var for var in variables_predictoras if var not in dataset_individual_ml.columns]
    if missing_vars:
        logger.error(f"Variables faltantes: {missing_vars}")
        return {}
    
    # Extraer features y target
    X = dataset_individual_ml[variables_predictoras].copy()
    y = dataset_individual_ml['sexo'].copy()
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Limpiar valores nulos
    if y.isnull().sum() > 0:
        logger.warning(f"Eliminando {y.isnull().sum()} filas con target nulo")
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
    
    if X.isnull().sum().sum() > 0:
        logger.warning(f"Imputando {X.isnull().sum().sum()} valores nulos en features")
        X = X.fillna(X.median())
    
    # Codificar target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    clases = label_encoder.classes_
    
    logger.info(f"Clases: {clases}")
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train distribution: {np.bincount(y_train)}")
    logger.info(f"Test distribution: {np.bincount(y_test)}")
    
    # Preparar datos de salida
    datos_preparados = {
        'sexo': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': variables_predictoras,
            'target_name': 'sexo',
            'label_encoder': label_encoder,
            'clases': clases,
            'n_clases': len(clases)
        }
    }
    
    logger.info("✓ Datos preparados para clasificación de SEXO")
    return datos_preparados


def entrenar_modelos_clasificacion(
    datos_preparados: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena 6 modelos de clasificación para SEXO con GridSearchCV.
    
    Esta función implementa un pipeline completo de machine learning para clasificar
    el sexo basándose en patrones epidemiológicos de mortalidad. Utiliza 6 algoritmos
    diferentes con optimización de hiperparámetros y validación cruzada.
    
    MODELOS IMPLEMENTADOS:
    1. Logistic Regression: Baseline con regularización L2
    2. Random Forest: Ensemble con múltiples árboles de decisión
    3. Gradient Boosting: Boosting adaptativo con árboles débiles
    4. Support Vector Machine: Máquinas de soporte vectorial con kernel RBF
    5. K-Nearest Neighbors: Clasificación basada en vecinos más cercanos
    6. Decision Tree: Árbol de decisión simple y interpretable
    
    OPTIMIZACIÓN:
    - GridSearchCV: Búsqueda exhaustiva de hiperparámetros
    - Cross-Validation: k=5 folds para validación robusta
    - Normalización: StandardScaler para features numéricos
    - Muestreo estratificado: Mantiene proporción de clases
    
    MÉTRICAS CALCULADAS:
    - Accuracy: Precisión general del modelo
    - Balanced Accuracy: Accuracy balanceado para clases desbalanceadas
    - Precision: Precisión por clase
    - Recall: Sensibilidad por clase
    - F1-Score: Media armónica de precision y recall
    - Cross-Validation: Scores promedio ± desviación estándar
    
    EXPECTATIVAS DE RENDIMIENTO:
    - Accuracy esperado: 70-80% (mejora vs 60% con variables irrelevantes)
    - Mejora por: Eliminación de 30 variables de ruido
    - Justificación: Patrones epidemiológicos reales y medibles
    
    Args:
        datos_preparados: Datos preparados con 11 variables relevantes
        params: Parámetros de configuración (cv_folds, n_jobs, verbose)
        
    Returns:
        Diccionario con resultados de todos los modelos:
        - modelo: Modelo entrenado optimizado
        - mejores_parametros: Mejores hiperparámetros encontrados
        - test_metrics: Métricas en conjunto de test
        - cv_scores: Scores de validación cruzada
        - training_time: Tiempo de entrenamiento en segundos
        - problema: 'sexo' (clasificación binaria)
    """
    logger.info("=" * 80)
    logger.info("ENTRENANDO MODELOS DE CLASIFICACION - SEXO")
    logger.info("=" * 80)
    
    # Configuración
    cv_folds = params.get('cv_folds', 5)
    n_jobs = params.get('n_jobs', -1)
    verbose = params.get('verbose', 1)
    
    # Obtener datos
    datos = datos_preparados['sexo']
    X_train = datos['X_train']
    X_test = datos['X_test']
    y_train = datos['y_train']
    y_test = datos['y_test']
    n_clases = datos['n_clases']
    
    logger.info(f"Clases: {n_clases}")
    logger.info(f"Features: {len(datos['feature_names'])}")
    logger.info(f"CV folds: {cv_folds}")
    
    # Configurar métricas
    es_binario = n_clases == 2
    average_method = 'binary' if es_binario else 'weighted'
    
    # Diccionario de resultados
    resultados_problema = {}
    
    # =================================================================
    # MODELO 1: LOGISTIC REGRESSION
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 1: LOGISTIC REGRESSION")
    logger.info("-" * 80)
    
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=n_jobs)
    
    param_grid_lr = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }
    
    grid_lr = GridSearchCV(
        modelo_lr, param_grid_lr, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_lr.predict(X_test)
    y_pred_train = grid_lr.predict(X_train)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    # Cross-validation scores
    cv_scores = cross_validate(
        grid_lr.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['Logistic Regression'] = {
        'modelo': grid_lr.best_estimator_,
        'mejores_parametros': grid_lr.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ Logistic Regression completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # MODELO 2: RANDOM FOREST
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 2: RANDOM FOREST")
    logger.info("-" * 80)
    
    modelo_rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_rf = GridSearchCV(
        modelo_rf, param_grid_rf, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_rf.predict(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    cv_scores = cross_validate(
        grid_rf.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['Random Forest'] = {
        'modelo': grid_rf.best_estimator_,
        'mejores_parametros': grid_rf.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ Random Forest completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # MODELO 3: GRADIENT BOOSTING
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 3: GRADIENT BOOSTING")
    logger.info("-" * 80)
    
    modelo_gb = GradientBoostingClassifier(random_state=42)
    
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }
    
    grid_gb = GridSearchCV(
        modelo_gb, param_grid_gb, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_gb.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_gb.predict(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    cv_scores = cross_validate(
        grid_gb.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['Gradient Boosting'] = {
        'modelo': grid_gb.best_estimator_,
        'mejores_parametros': grid_gb.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ Gradient Boosting completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # MODELO 4: SUPPORT VECTOR MACHINE
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 4: SUPPORT VECTOR MACHINE")
    logger.info("-" * 80)
    
    modelo_svm = SVC(random_state=42, probability=True)
    
    param_grid_svm = {
        'C': [1.0],
        'kernel': ['rbf'],
        'gamma': ['scale']
    }
    
    grid_svm = GridSearchCV(
        modelo_svm, param_grid_svm, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_svm.predict(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    cv_scores = cross_validate(
        grid_svm.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['Support Vector Machine'] = {
        'modelo': grid_svm.best_estimator_,
        'mejores_parametros': grid_svm.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ SVM completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # MODELO 5: K-NEAREST NEIGHBORS
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 5: K-NEAREST NEIGHBORS")
    logger.info("-" * 80)
    
    modelo_knn = KNeighborsClassifier(n_jobs=n_jobs)
    
    param_grid_knn = {
        'n_neighbors': [5, 7],
        'weights': ['distance'],
        'metric': ['euclidean']
    }
    
    grid_knn = GridSearchCV(
        modelo_knn, param_grid_knn, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_knn.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_knn.predict(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    cv_scores = cross_validate(
        grid_knn.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['K-Nearest Neighbors'] = {
        'modelo': grid_knn.best_estimator_,
        'mejores_parametros': grid_knn.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ KNN completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # MODELO 6: DECISION TREE
    # =================================================================
    logger.info("\n" + "-" * 80)
    logger.info("MODELO 6: DECISION TREE")
    logger.info("-" * 80)
    
    modelo_dt = DecisionTreeClassifier(random_state=42)
    
    param_grid_dt = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'criterion': ['gini']
    }
    
    grid_dt = GridSearchCV(
        modelo_dt, param_grid_dt, cv=cv_folds, n_jobs=n_jobs, 
        verbose=verbose, scoring='accuracy'
    )
    
    start_time = time.time()
    grid_dt.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred_test = grid_dt.predict(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_test, average=average_method, zero_division=0)
    }
    
    cv_scores = cross_validate(
        grid_dt.best_estimator_, X_train, y_train, 
        cv=cv_folds, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    
    resultados_problema['Decision Tree'] = {
        'modelo': grid_dt.best_estimator_,
        'mejores_parametros': grid_dt.best_params_,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'training_time': training_time,
        'problema': 'sexo'
    }
    
    logger.info(f"✓ Decision Tree completado en {training_time:.2f}s")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    
    # =================================================================
    # RESUMEN FINAL
    # =================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN FINAL - CLASIFICACION DE SEXO")
    logger.info("=" * 80)
    
    for nombre_modelo, resultados in resultados_problema.items():
        logger.info(f"\n{nombre_modelo.upper()}:")
        logger.info(f"  Test Accuracy:  {resultados['test_metrics']['accuracy']:.4f}")
        logger.info(f"  CV Accuracy:    {resultados['cv_scores']['test_accuracy'].mean():.4f} ± {resultados['cv_scores']['test_accuracy'].std():.4f}")
        logger.info(f"  F1-Score:       {resultados['test_metrics']['f1_score']:.4f}")
        logger.info(f"  Tiempo:         {resultados['training_time']:.2f}s")
    
    # Guardar resultados globales
    resultados_globales = {'sexo': resultados_problema}
    
    logger.info(f"\n✓ Entrenamiento completado: {len(resultados_problema)} modelos")
    return resultados_globales


def generar_tabla_comparativa_clasificacion(
    resultados_clasificacion: Dict[str, Any]
) -> pd.DataFrame:
    """
    Genera tabla comparativa de todos los modelos de clasificación.
    
    Args:
        resultados_clasificacion: Resultados de todos los modelos
        
    Returns:
        DataFrame con tabla comparativa
    """
    logger.info("Generando tabla comparativa de clasificación...")
    
    filas = []
    
    for problema, modelos in resultados_clasificacion.items():
        for nombre_modelo, resultados in modelos.items():
            fila = {
                'Problema': problema,
                'Modelo': nombre_modelo,
                'Test Accuracy': resultados['test_metrics']['accuracy'],
                'CV Accuracy (mean±std)': f"{resultados['cv_scores']['test_accuracy'].mean():.4f}±{resultados['cv_scores']['test_accuracy'].std():.4f}",
                'F1-Score': resultados['test_metrics']['f1_score'],
                'Precision': resultados['test_metrics']['precision'],
                'Recall': resultados['test_metrics']['recall'],
                'Tiempo (s)': resultados['training_time']
            }
            filas.append(fila)
    
    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values('Test Accuracy', ascending=False)
    
    logger.info(f"Tabla comparativa generada: {len(tabla)} modelos")
    return tabla


def guardar_modelos_clasificacion(
    resultados_clasificacion: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, str]:
    """
    Guarda todos los modelos entrenados en disco.
    
    Args:
        resultados_clasificacion: Resultados de todos los modelos
        params: Parámetros de configuración
        
    Returns:
        Diccionario con rutas de archivos guardados
    """
    logger.info("Guardando modelos de clasificación...")
    
    ruta_modelos = params.get('ruta_modelos', 'data/06_models')
    Path(ruta_modelos).mkdir(parents=True, exist_ok=True)
    
    rutas_guardadas = {}
    
    for problema, modelos in resultados_clasificacion.items():
        for nombre_modelo, resultados in modelos.items():
            # Nombre del archivo
            nombre_archivo = f"modelo_{problema}_{nombre_modelo.lower().replace(' ', '_')}.pkl"
            ruta_completa = Path(ruta_modelos) / nombre_archivo
            
            # Guardar modelo
            with open(ruta_completa, 'wb') as f:
                pickle.dump(resultados['modelo'], f)
            
            rutas_guardadas[f"{problema}_{nombre_modelo}"] = str(ruta_completa)
            logger.info(f"  Guardado: {nombre_archivo}")
    
    logger.info(f"✓ {len(rutas_guardadas)} modelos guardados en {ruta_modelos}")
    return rutas_guardadas
