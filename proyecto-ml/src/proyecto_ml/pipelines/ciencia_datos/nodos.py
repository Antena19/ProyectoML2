"""Nodos del pipeline de Ciencia de Datos.

Este módulo contiene todas las funciones de procesamiento de datos
para la fase de ciencia de datos del proyecto CRISP-DM.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Configurar logging
logger = logging.getLogger(__name__)


def integrar_datasets(defunciones_limpias: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un dataset unificado basado en defunciones limpias para análisis y modelado.
    
    Esta función toma el dataset de defunciones ya limpio y crea un dataset consolidado
    con métricas agregadas por año, preparado para feature engineering y modelado.
    
    Args:
        defunciones_limpias: Dataset de defunciones ya limpio (resultado del pipeline de ingeniería de datos)
        
    Returns:
        Dataset unificado con métricas agregadas por año
    """
    logger.info("Iniciando creación de dataset unificado...")
    
    # 1. Crear métricas agregadas de defunciones por año
    logger.info("Agregando defunciones por año y región...")
    logger.info(f"Usando dataset de defunciones limpias: {defunciones_limpias.shape}")
    
    # Crear métricas agregadas de defunciones por año
    defunciones_por_año = defunciones_limpias.groupby('año').agg({
        'fecha_defuncion': 'count',  # Total de defunciones por año
        'sexo': lambda x: (x == 'Hombre').sum(),  # Defunciones de hombres
        'edad_cantidad': ['mean', 'median', 'std'],  # Estadísticas de edad
        'region': 'nunique'  # Número de regiones con defunciones
    }).reset_index()
    
    # Aplanar nombres de columnas
    defunciones_por_año.columns = [
        'año', 'total_defunciones_año', 'defunciones_hombres_año',
        'edad_promedio_defunciones', 'edad_mediana_defunciones', 
        'edad_std_defunciones', 'regiones_con_defunciones'
    ]
    
    logger.info(f"Defunciones por año: {defunciones_por_año.shape}")
    
    # 2. Crear métricas derivadas básicas
    logger.info("Creando métricas derivadas...")
    
    # Calcular proporción de defunciones por sexo
    defunciones_por_año['proporcion_defunciones_hombres'] = (
        defunciones_por_año['defunciones_hombres_año'] / 
        defunciones_por_año['total_defunciones_año']
    )
    
    defunciones_por_año['proporcion_defunciones_mujeres'] = (
        1 - defunciones_por_año['proporcion_defunciones_hombres']
    )
    
    # Calcular densidad de defunciones por región
    defunciones_por_año['densidad_defunciones_por_region'] = (
        defunciones_por_año['total_defunciones_año'] / 
        defunciones_por_año['regiones_con_defunciones']
    )
    
    logger.info("Métricas derivadas creadas exitosamente")
    
    # 3. Resumen final
    logger.info("=== RESUMEN DE INTEGRACIÓN ===")
    logger.info(f"Dataset unificado: {defunciones_por_año.shape}")
    logger.info(f"Años cubiertos: {sorted(defunciones_por_año['año'].unique())}")
    logger.info(f"Columnas totales: {defunciones_por_año.shape[1]}")
    
    # Verificar valores nulos
    nulos_por_columna = defunciones_por_año.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    if len(columnas_con_nulos) > 0:
        logger.warning(f"Columnas con valores nulos: {len(columnas_con_nulos)}")
        logger.warning(f"Columnas: {list(columnas_con_nulos.index)}")
    else:
        logger.info("No se encontraron valores nulos")
    
    logger.info("Integración de datasets completada exitosamente")
    return defunciones_por_año


def crear_features_temporales_avanzadas(datasets_estandarizados: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features temporales avanzadas para análisis de machine learning.
    
    Esta función toma los datasets estandarizados del pipeline de ingeniería
    y aplica feature engineering avanzado basado en el notebook:
    1. Features cíclicos (sin, cos) para capturar estacionalidad
    2. Codificación de variables categóricas temporales
    3. Features de días especiales y épocas del año
    
    Args:
        datasets_estandarizados: Diccionario con datasets estandarizados
        
    Returns:
        Dataset con features temporales avanzadas
    """
    logger.info("Iniciando creación de features temporales avanzadas...")
    
    # Extraer dataset de defunciones del diccionario
    defunciones_estandarizado = datasets_estandarizados["defunciones_estandarizado"]
    logger.info(f"Usando dataset de defunciones estandarizado: {defunciones_estandarizado.shape}")
    
    # Crear copia para trabajar
    dataset_con_features = defunciones_estandarizado.copy()
    
    # 1. Features cíclicos para capturar estacionalidad
    logger.info("Creando features cíclicos...")
    
    # Verificar que tenemos las columnas temporales necesarias
    columnas_temporales = ['mes', 'dia_año', 'trimestre']
    columnas_disponibles = [col for col in columnas_temporales if col in dataset_con_features.columns]
    logger.info(f"Columnas temporales disponibles: {columnas_disponibles}")
    
    # Features cíclicos para mes (si está disponible)
    if 'mes' in dataset_con_features.columns:
        dataset_con_features['mes_sin'] = np.sin(2 * np.pi * dataset_con_features['mes'] / 12)
        dataset_con_features['mes_cos'] = np.cos(2 * np.pi * dataset_con_features['mes'] / 12)
        logger.info(" Features cíclicos de mes creados")
    
    # Features cíclicos para día del año (si está disponible)
    if 'dia_año' in dataset_con_features.columns:
        dataset_con_features['dia_año_sin'] = np.sin(2 * np.pi * dataset_con_features['dia_año'] / 365)
        dataset_con_features['dia_año_cos'] = np.cos(2 * np.pi * dataset_con_features['dia_año'] / 365)
        logger.info(" Features cíclicos de día del año creados")
    
    # Features cíclicos para trimestre (si está disponible)
    if 'trimestre' in dataset_con_features.columns:
        dataset_con_features['trimestre_sin'] = np.sin(2 * np.pi * dataset_con_features['trimestre'] / 4)
        dataset_con_features['trimestre_cos'] = np.cos(2 * np.pi * dataset_con_features['trimestre'] / 4)
        logger.info(" Features cíclicos de trimestre creados")
    
    # 2. Codificación de día de la semana
    logger.info("Codificando día de la semana...")
    if 'dia_semana' in dataset_con_features.columns:
        # Mapeo de días de la semana
        mapeo_dias_semana = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        dataset_con_features['dia_semana_codificado'] = dataset_con_features['dia_semana'].map(mapeo_dias_semana)
        
        # Features cíclicos para día de la semana
        dataset_con_features['dia_semana_sin'] = np.sin(2 * np.pi * dataset_con_features['dia_semana_codificado'] / 7)
        dataset_con_features['dia_semana_cos'] = np.cos(2 * np.pi * dataset_con_features['dia_semana_codificado'] / 7)
        logger.info(" Día de la semana codificado y features cíclicos creados")
    else:
        logger.warning(" Columna 'dia_semana' no encontrada")
    
    # 3. Features de días especiales y épocas del año
    logger.info("Creando features de días especiales...")
    
    # Fin de semana
    if 'dia_semana_codificado' in dataset_con_features.columns:
        dataset_con_features['es_fin_semana'] = dataset_con_features['dia_semana_codificado'].isin([6, 7]).astype(int)
        logger.info(" Feature 'es_fin_semana' creado")
    
    # Estaciones del año (basado en mes)
    if 'mes' in dataset_con_features.columns:
        # Invierno: dic, ene, feb (12, 1, 2)
        dataset_con_features['es_invierno'] = dataset_con_features['mes'].isin([12, 1, 2]).astype(int)
        # Verano: jun, jul, ago (6, 7, 8)
        dataset_con_features['es_verano'] = dataset_con_features['mes'].isin([6, 7, 8]).astype(int)
        logger.info(" Features de estaciones creados")
    
    # Trimestre fiscal (ajustado para Chile: abril-marzo)
    if 'mes' in dataset_con_features.columns:
        dataset_con_features['trimestre_fiscal'] = ((dataset_con_features['mes'] - 4) % 12) // 3 + 1
        logger.info(" Feature 'trimestre_fiscal' creado")
    
    # Época del año (4 épocas)
    if 'mes' in dataset_con_features.columns:
        def obtener_epoca_año(mes):
            if mes in [12, 1, 2]: return 1  # Verano
            elif mes in [3, 4, 5]: return 2  # Otoño
            elif mes in [6, 7, 8]: return 3  # Invierno
            else: return 4  # Primavera
        
        dataset_con_features['epoca_año_codificada'] = dataset_con_features['mes'].apply(obtener_epoca_año)
        logger.info(" Época del año codificada")
    
    # 4. Features básicas temporales
    logger.info("Creando features básicas temporales...")
    
    # Año normalizado (para modelos que requieren escalado)
    dataset_con_features['año_normalizado'] = (
        dataset_con_features['año'] - dataset_con_features['año'].min()
    ) / (dataset_con_features['año'].max() - dataset_con_features['año'].min())
    
    # Década (para análisis de tendencias a largo plazo)
    dataset_con_features['decada'] = (dataset_con_features['año'] // 10) * 10
    
    # Resumen de features creados
    logger.info("=== RESUMEN DE FEATURES TEMPORALES CREADAS ===")
    features_ciclicos = [col for col in dataset_con_features.columns if '_sin' in col or '_cos' in col]
    features_especiales = [col for col in dataset_con_features.columns if col.startswith('es_') or 'fiscal' in col or 'epoca' in col]
    
    logger.info(f"Features cíclicos creados: {len(features_ciclicos)}")
    logger.info(f"Features especiales creados: {len(features_especiales)}")
    logger.info(f"Dataset final con features temporales: {dataset_con_features.shape}")
    
    logger.info("Features temporales avanzadas creadas exitosamente")
    return dataset_con_features


def normalizar_datos_para_modelado(dataset_con_features: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Normaliza los datos para prepararlos para modelos de machine learning.
    
    Esta función aplica diferentes tipos de normalización basados en el notebook:
    1. StandardScaler para la mayoría de variables
    2. MinMaxScaler para variables que requieren rango [0,1]
    3. RobustScaler para variables con outliers
    
    Args:
        dataset_con_features: Dataset con features temporales avanzadas
        
    Returns:
        Diccionario con datasets normalizados usando diferentes métodos
    """
    logger.info("Iniciando normalización de datos para modelado...")
    
    # Importar librerías de normalización
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Crear copia para trabajar
    dataset_normalizado = dataset_con_features.copy()
    
    # 1. Identificar variables que necesitan normalización
    logger.info("Identificando variables para normalización...")
    
    # Variables numéricas (excluir códigos categóricos y variables ya normalizadas)
    variables_numericas = dataset_normalizado.select_dtypes(include=[np.number]).columns.tolist()
    variables_a_excluir = ['año', 'codigo_comuna', 'region_codificada', 'sexo_codificado', 
                          'rango_edad_codificado', 'categoria_diagnostico_codificada',
                          'dia_semana_codificado', 'epoca_año_codificada', 'decada']
    
    variables_a_normalizar = [col for col in variables_numericas if col not in variables_a_excluir]
    logger.info(f"Variables a normalizar: {len(variables_a_normalizar)}")
    logger.info(f"Variables: {variables_a_normalizar}")
    
    # 2. Aplicar StandardScaler (normalización estándar)
    logger.info("Aplicando StandardScaler...")
    scaler_std = StandardScaler()
    dataset_std = dataset_normalizado.copy()
    dataset_std[variables_a_normalizar] = scaler_std.fit_transform(dataset_std[variables_a_normalizar])
    
    # 3. Aplicar MinMaxScaler (normalización a rango [0,1])
    logger.info("Aplicando MinMaxScaler...")
    scaler_minmax = MinMaxScaler()
    dataset_minmax = dataset_normalizado.copy()
    dataset_minmax[variables_a_normalizar] = scaler_minmax.fit_transform(dataset_minmax[variables_a_normalizar])
    
    # 4. Aplicar RobustScaler (robusto a outliers)
    logger.info("Aplicando RobustScaler...")
    scaler_robust = RobustScaler()
    dataset_robust = dataset_normalizado.copy()
    dataset_robust[variables_a_normalizar] = scaler_robust.fit_transform(dataset_robust[variables_a_normalizar])
    
    # 5. Crear dataset final para modelado (usando StandardScaler por defecto)
    logger.info("Creando dataset final para modelado...")
    dataset_final_modelado = dataset_std.copy()
    
    # Agregar información de normalización
    info_normalizacion = {
        "variables_normalizadas": variables_a_normalizar,
        "metodo_principal": "StandardScaler",
        "total_variables": len(variables_a_normalizar),
        "shape_dataset": dataset_final_modelado.shape
    }
    
    # Compilar datasets normalizados
    datasets_normalizados = {
        "dataset_modelado_standard": dataset_std,
        "dataset_modelado_minmax": dataset_minmax,
        "dataset_modelado_robust": dataset_robust,
        "dataset_final_modelado": dataset_final_modelado,
        "info_normalizacion": info_normalizacion
    }
    
    logger.info("=== RESUMEN DE NORMALIZACIÓN ===")
    logger.info(f"Variables normalizadas: {len(variables_a_normalizar)}")
    logger.info(f"Dataset final: {dataset_final_modelado.shape}")
    logger.info(f"Métodos aplicados: StandardScaler, MinMaxScaler, RobustScaler")
    
    logger.info("Normalización de datos completada exitosamente")
    return datasets_normalizados


def crear_datasets_finales_para_modelado(datasets_normalizados: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Crea datasets finales optimizados para diferentes tipos de modelos de ML.
    
    Esta función toma los datasets normalizados y crea versiones específicas
    para diferentes algoritmos de machine learning:
    1. Dataset para modelos de regresión
    2. Dataset para modelos de clasificación
    3. Dataset para análisis temporal
    4. Dataset con índices únicos
    
    Args:
        datasets_normalizados: Diccionario con datasets normalizados
        
    Returns:
        Diccionario con datasets finales para modelado
    """
    logger.info("Iniciando creación de datasets finales para modelado...")
    
    # Extraer dataset principal
    dataset_final = datasets_normalizados["dataset_final_modelado"]
    logger.info(f"Dataset base para modelado: {dataset_final.shape}")
    
    # 1. Dataset para modelos de regresión (predicción de cantidades)
    logger.info("Creando dataset para modelos de regresión...")
    dataset_regresion = dataset_final.copy()
    
    # Seleccionar features relevantes para regresión
    features_regresion = [
        'año', 'mes', 'trimestre', 'dia_año',
        'mes_sin', 'mes_cos', 'dia_año_sin', 'dia_año_cos',
        'trimestre_sin', 'trimestre_cos', 'dia_semana_sin', 'dia_semana_cos',
        'es_fin_semana', 'es_invierno', 'es_verano', 'trimestre_fiscal',
        'epoca_año_codificada', 'año_normalizado', 'decada'
    ]
    
    # Filtrar solo las columnas que existen
    features_regresion_disponibles = [col for col in features_regresion if col in dataset_regresion.columns]
    dataset_regresion_final = dataset_regresion[features_regresion_disponibles]
    logger.info(f"Dataset regresión: {dataset_regresion_final.shape}")
    
    # 2. Dataset para análisis temporal (series de tiempo)
    logger.info("Creando dataset para análisis temporal...")
    dataset_temporal = dataset_final.copy()
    
    # Agregar índice temporal si no existe
    if 'fecha_defuncion' in dataset_temporal.columns:
        dataset_temporal['fecha_defuncion'] = pd.to_datetime(dataset_temporal['fecha_defuncion'])
        dataset_temporal = dataset_temporal.sort_values('fecha_defuncion')
        dataset_temporal['indice_temporal'] = range(len(dataset_temporal))
        logger.info(" Índice temporal creado")
    
    # 3. Dataset con índices únicos para identificación
    logger.info("Creando dataset con índices únicos...")
    dataset_indexado = dataset_final.copy()
    
    # Crear índice único combinando año, mes, región, sexo
    if all(col in dataset_indexado.columns for col in ['año', 'mes', 'region', 'sexo']):
        dataset_indexado['id_unico'] = (
            dataset_indexado['año'].astype(str) + '_' +
            dataset_indexado['mes'].astype(str) + '_' +
            dataset_indexado['region'].astype(str) + '_' +
            dataset_indexado['sexo'].astype(str)
        )
        logger.info(" ID único creado")
    
    # 4. Dataset resumido por agregación temporal
    logger.info("Creando dataset resumido por agregación...")
    if 'año' in dataset_final.columns and 'mes' in dataset_final.columns:
        # Agregar por año y mes
        dataset_resumido = dataset_final.groupby(['año', 'mes']).agg({
            'mes_sin': 'mean',
            'mes_cos': 'mean',
            'es_fin_semana': 'mean',
            'es_invierno': 'mean',
            'es_verano': 'mean',
            'trimestre_fiscal': 'mean',
            'epoca_año_codificada': 'mean',
            'año_normalizado': 'mean'
        }).reset_index()
        logger.info(f"Dataset resumido: {dataset_resumido.shape}")
    else:
        dataset_resumido = dataset_final.copy()
        logger.info("Dataset resumido: usando dataset completo")
    
    # Compilar datasets finales
    datasets_finales = {
        "dataset_regresion": dataset_regresion_final,
        "dataset_temporal": dataset_temporal,
        "dataset_indexado": dataset_indexado,
        "dataset_resumido": dataset_resumido,
        "dataset_completo": dataset_final
    }
    
    logger.info("=== RESUMEN DE DATASETS FINALES ===")
    for nombre, df in datasets_finales.items():
        logger.info(f"{nombre}: {df.shape}")
    
    logger.info("Datasets finales para modelado creados exitosamente")
    return datasets_finales
    dataset_con_features['tendencia_lineal'] = (
        dataset_con_features['año'] - dataset_con_features['año'].min()
    )
    
    # Tendencia cuadrática (para capturar aceleraciones)
    dataset_con_features['tendencia_cuadratica'] = (
        dataset_con_features['tendencia_lineal'] ** 2
    )
    
    # 3. Features cíclicas
    logger.info("Creando features cíclicas...")
    
    # Ciclo de 5 años (para capturar ciclos económicos)
    dataset_con_features['ciclo_5_anos'] = (
        dataset_con_features['año'] % 5
    )
    
    # Ciclo de 10 años (para capturar ciclos demográficos)
    dataset_con_features['ciclo_10_anos'] = (
        dataset_con_features['año'] % 10
    )
    
    # 4. Features de cambio y crecimiento
    logger.info("Creando features de cambio...")
    
    # Calcular cambios año a año
    dataset_con_features = dataset_con_features.sort_values('año')
    
    # Cambio en nacimientos
    dataset_con_features['cambio_nacimientos'] = (
        dataset_con_features['total_nacimientos_año'].diff()
    )
    
    # Cambio en defunciones
    dataset_con_features['cambio_defunciones'] = (
        dataset_con_features['total_defunciones_año'].diff()
    )
    
    # Cambio en crecimiento poblacional
    dataset_con_features['cambio_crecimiento_poblacional'] = (
        dataset_con_features['crecimiento_poblacional'].diff()
    )
    
    # 5. Features de promedio móvil (ventana de 3 años)
    logger.info("Creando promedios móviles...")
    
    # Promedio móvil de nacimientos
    dataset_con_features['promedio_movil_nacimientos_3'] = (
        dataset_con_features['total_nacimientos_año'].rolling(window=3, min_periods=1).mean()
    )
    
    # Promedio móvil de defunciones
    dataset_con_features['promedio_movil_defunciones_3'] = (
        dataset_con_features['total_defunciones_año'].rolling(window=3, min_periods=1).mean()
    )
    
    # Promedio móvil de crecimiento poblacional
    dataset_con_features['promedio_movil_crecimiento_3'] = (
        dataset_con_features['crecimiento_poblacional'].rolling(window=3, min_periods=1).mean()
    )
    
    # 6. Features de volatilidad
    logger.info("Creando features de volatilidad...")
    
    # Volatilidad de nacimientos (desviación estándar móvil)
    dataset_con_features['volatilidad_nacimientos_3'] = (
        dataset_con_features['total_nacimientos_año'].rolling(window=3, min_periods=1).std()
    )
    
    # Volatilidad de defunciones
    dataset_con_features['volatilidad_defunciones_3'] = (
        dataset_con_features['total_defunciones_año'].rolling(window=3, min_periods=1).std()
    )
    
    # 7. Features de posición relativa
    logger.info("Creando features de posición relativa...")
    
    # Percentil de nacimientos en el año
    dataset_con_features['percentil_nacimientos'] = (
        dataset_con_features['total_nacimientos_año'].rank(pct=True)
    )
    
    # Percentil de defunciones en el año
    dataset_con_features['percentil_defunciones'] = (
        dataset_con_features['total_defunciones_año'].rank(pct=True)
    )
    
    # 8. Resumen de features creadas
    logger.info("=== RESUMEN DE FEATURES TEMPORALES ===")
    features_temporales = [
        'año_normalizado', 'decada', 'siglo', 'tendencia_lineal', 'tendencia_cuadratica',
        'ciclo_5_anos', 'ciclo_10_anos', 'cambio_nacimientos', 'cambio_defunciones',
        'cambio_crecimiento_poblacional', 'promedio_movil_nacimientos_3',
        'promedio_movil_defunciones_3', 'promedio_movil_crecimiento_3',
        'volatilidad_nacimientos_3', 'volatilidad_defunciones_3',
        'percentil_nacimientos', 'percentil_defunciones'
    ]
    
    logger.info(f"Features temporales creadas: {len(features_temporales)}")
    logger.info(f"Dataset final: {dataset_con_features.shape}")
    
    # Verificar valores nulos en features nuevas
    nulos_features = dataset_con_features[features_temporales].isnull().sum().sum()
    logger.info(f"Valores nulos en features temporales: {nulos_features}")
    
    logger.info("Creación de features temporales completada exitosamente")
    return dataset_con_features


def codificar_variables_categoricas(dataset_con_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Codifica variables categóricas usando diferentes estrategias según el tipo de variable.
    
    Esta función aplica codificación apropiada a variables categóricas:
    - Label Encoding para variables ordinales
    - One-Hot Encoding para variables nominales
    - Codificación personalizada para variables específicas
    
    Args:
        dataset_con_features: Dataset con features temporales
        
    Returns:
        Tuple con (dataset codificado, mapeos de codificación)
    """
    logger.info("Iniciando codificación de variables categóricas...")
    
    # Crear copia para trabajar
    dataset_codificado = dataset_con_features.copy()
    mapeos_codificacion = {}
    
    # 1. Codificar década (Label Encoding - ordinal)
    logger.info("Codificando década...")
    le_decada = LabelEncoder()
    dataset_codificado['decada_codificada'] = le_decada.fit_transform(dataset_codificado['decada'])
    mapeos_codificacion['decada'] = {
        'encoder': le_decada,
        'tipo': 'label_encoding',
        'mapeo': dict(zip(le_decada.classes_, le_decada.transform(le_decada.classes_)))
    }
    
    # 2. Codificar siglo (Label Encoding - ordinal)
    logger.info("Codificando siglo...")
    le_siglo = LabelEncoder()
    dataset_codificado['siglo_codificada'] = le_siglo.fit_transform(dataset_codificado['siglo'])
    mapeos_codificacion['siglo'] = {
        'encoder': le_siglo,
        'tipo': 'label_encoding',
        'mapeo': dict(zip(le_siglo.classes_, le_siglo.transform(le_siglo.classes_)))
    }
    
    # 3. Codificar ciclos (One-Hot Encoding - nominal)
    logger.info("Codificando ciclos con One-Hot Encoding...")
    
    # Ciclo de 5 años
    ciclo_5_dummies = pd.get_dummies(dataset_codificado['ciclo_5_anos'], prefix='ciclo_5')
    dataset_codificado = pd.concat([dataset_codificado, ciclo_5_dummies], axis=1)
    mapeos_codificacion['ciclo_5_anos'] = {
        'tipo': 'one_hot_encoding',
        'columnas': list(ciclo_5_dummies.columns)
    }
    
    # Ciclo de 10 años
    ciclo_10_dummies = pd.get_dummies(dataset_codificado['ciclo_10_anos'], prefix='ciclo_10')
    dataset_codificado = pd.concat([dataset_codificado, ciclo_10_dummies], axis=1)
    mapeos_codificacion['ciclo_10_anos'] = {
        'tipo': 'one_hot_encoding',
        'columnas': list(ciclo_10_dummies.columns)
    }
    
    # 4. Crear variables categóricas derivadas
    logger.info("Creando variables categóricas derivadas...")
    
    # Categorizar crecimiento poblacional
    dataset_codificado['categoria_crecimiento'] = pd.cut(
        dataset_codificado['crecimiento_poblacional'],
        bins=[-np.inf, -1000, 0, 1000, np.inf],
        labels=['decrecimiento_alto', 'decrecimiento_bajo', 'crecimiento_bajo', 'crecimiento_alto']
    )
    
    # Codificar categoría de crecimiento
    le_crecimiento = LabelEncoder()
    dataset_codificado['categoria_crecimiento_codificada'] = le_crecimiento.fit_transform(
        dataset_codificado['categoria_crecimiento'].astype(str)
    )
    mapeos_codificacion['categoria_crecimiento'] = {
        'encoder': le_crecimiento,
        'tipo': 'label_encoding',
        'mapeo': dict(zip(le_crecimiento.classes_, le_crecimiento.transform(le_crecimiento.classes_)))
    }
    
    # Categorizar tasa de natalidad
    dataset_codificado['categoria_tasa_natalidad'] = pd.cut(
        dataset_codificado['tasa_natalidad'],
        bins=[0, 0.5, 1.0, 1.5, np.inf],
        labels=['muy_baja', 'baja', 'media', 'alta']
    )
    
    # Codificar categoría de tasa de natalidad
    le_natalidad = LabelEncoder()
    dataset_codificado['categoria_tasa_natalidad_codificada'] = le_natalidad.fit_transform(
        dataset_codificado['categoria_tasa_natalidad'].astype(str)
    )
    mapeos_codificacion['categoria_tasa_natalidad'] = {
        'encoder': le_natalidad,
        'tipo': 'label_encoding',
        'mapeo': dict(zip(le_natalidad.classes_, le_natalidad.transform(le_natalidad.classes_)))
    }
    
    # 5. Crear variables binarias
    logger.info("Creando variables binarias...")
    
    # Indicador de años con crecimiento positivo
    dataset_codificado['crecimiento_positivo'] = (
        dataset_codificado['crecimiento_poblacional'] > 0
    ).astype(int)
    
    # Indicador de años con más nacimientos que defunciones
    dataset_codificado['mas_nacimientos_que_defunciones'] = (
        dataset_codificado['total_nacimientos_año'] > dataset_codificado['total_defunciones_año']
    ).astype(int)
    
    # Indicador de años con alta volatilidad en nacimientos
    dataset_codificado['alta_volatilidad_nacimientos'] = (
        dataset_codificado['volatilidad_nacimientos_3'] > 
        dataset_con_features['volatilidad_nacimientos_3'].quantile(0.75)
    ).astype(int)
    
    # 6. Resumen de codificación
    logger.info("=== RESUMEN DE CODIFICACIÓN ===")
    
    # Contar tipos de codificación
    tipos_codificacion = {}
    for variable, info in mapeos_codificacion.items():
        tipo = info['tipo']
        tipos_codificacion[tipo] = tipos_codificacion.get(tipo, 0) + 1
    
    logger.info(f"Variables codificadas: {len(mapeos_codificacion)}")
    logger.info(f"Tipos de codificación: {tipos_codificacion}")
    
    # Contar columnas nuevas
    columnas_originales = len(dataset_con_features.columns)
    columnas_finales = len(dataset_codificado.columns)
    columnas_nuevas = columnas_finales - columnas_originales
    
    logger.info(f"Columnas originales: {columnas_originales}")
    logger.info(f"Columnas finales: {columnas_finales}")
    logger.info(f"Columnas nuevas: {columnas_nuevas}")
    
    # Verificar valores nulos
    nulos_codificacion = dataset_codificado.isnull().sum().sum()
    logger.info(f"Valores nulos después de codificación: {nulos_codificacion}")
    
    logger.info("Codificación de variables categóricas completada exitosamente")
    return dataset_codificado, mapeos_codificacion


def escalar_caracteristicas(
    dataset_codificado: pd.DataFrame, 
    mapeos_codificacion: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Escala las características numéricas usando diferentes métodos de normalización.
    
    Esta función aplica escalado a las variables numéricas para prepararlas
    para modelos de machine learning que requieren datos normalizados.
    
    Args:
        dataset_codificado: Dataset con variables codificadas
        mapeos_codificacion: Diccionario con mapeos de codificación
        
    Returns:
        Tuple con (dataset escalado, scalers utilizados)
    """
    logger.info("Iniciando escalado de características...")
    
    # Crear copia para trabajar
    dataset_escalado = dataset_codificado.copy()
    scalers = {}
    
    # 1. Identificar columnas numéricas para escalar
    logger.info("Identificando columnas numéricas...")
    
    # Excluir columnas que no deben escalarse
    columnas_excluidas = [
        'año', 'decada', 'siglo', 'ciclo_5_anos', 'ciclo_10_anos',
        'categoria_crecimiento', 'categoria_tasa_natalidad',
        'crecimiento_positivo', 'mas_nacimientos_que_defunciones', 'alta_volatilidad_nacimientos'
    ]
    
    # Excluir columnas codificadas (ya están en escala apropiada)
    columnas_codificadas = [
        'decada_codificada', 'siglo_codificada', 'categoria_crecimiento_codificada',
        'categoria_tasa_natalidad_codificada'
    ]
    
    # Excluir columnas one-hot
    columnas_one_hot = []
    for variable, info in mapeos_codificacion.items():
        if info['tipo'] == 'one_hot_encoding':
            columnas_one_hot.extend(info['columnas'])
    
    # Identificar columnas numéricas
    columnas_numericas = []
    for col in dataset_escalado.columns:
        if (col not in columnas_excluidas and 
            col not in columnas_codificadas and 
            col not in columnas_one_hot and
            dataset_escalado[col].dtype in ['int64', 'float64']):
            columnas_numericas.append(col)
    
    logger.info(f"Columnas numéricas identificadas: {len(columnas_numericas)}")
    logger.info(f"Columnas: {columnas_numericas}")
    
    # 2. Aplicar diferentes métodos de escalado según el tipo de variable
    logger.info("Aplicando métodos de escalado...")
    
    # StandardScaler para variables con distribución normal
    variables_standard = [
        'edad_promedio_defunciones', 'edad_mediana_defunciones', 'edad_std_defunciones',
        'tasa_natalidad', 'tasa_mortalidad', 'proporcion_nacimientos_hombres',
        'proporcion_defunciones_hombres', 'año_normalizado', 'tendencia_lineal',
        'tendencia_cuadratica', 'percentil_nacimientos', 'percentil_defunciones'
    ]
    
    variables_standard = [col for col in variables_standard if col in columnas_numericas]
    
    if variables_standard:
        logger.info(f"Aplicando StandardScaler a {len(variables_standard)} variables...")
        scaler_standard = StandardScaler()
        dataset_escalado[variables_standard] = scaler_standard.fit_transform(
            dataset_escalado[variables_standard]
        )
        scalers['standard'] = {
            'scaler': scaler_standard,
            'variables': variables_standard
        }
    
    # MinMaxScaler para variables que deben estar en rango [0,1]
    variables_minmax = [
        'promedio_movil_nacimientos_3', 'promedio_movil_defunciones_3',
        'promedio_movil_crecimiento_3', 'volatilidad_nacimientos_3',
        'volatilidad_defunciones_3'
    ]
    
    variables_minmax = [col for col in variables_minmax if col in columnas_numericas]
    
    if variables_minmax:
        logger.info(f"Aplicando MinMaxScaler a {len(variables_minmax)} variables...")
        scaler_minmax = MinMaxScaler()
        dataset_escalado[variables_minmax] = scaler_minmax.fit_transform(
            dataset_escalado[variables_minmax]
        )
        scalers['minmax'] = {
            'scaler': scaler_minmax,
            'variables': variables_minmax
        }
    
    # RobustScaler para variables con outliers
    variables_robust = [
        'total_defunciones_año', 'total_nacimientos_año', 'crecimiento_poblacional',
        'cambio_nacimientos', 'cambio_defunciones', 'cambio_crecimiento_poblacional'
    ]
    
    variables_robust = [col for col in variables_robust if col in columnas_numericas]
    
    if variables_robust:
        logger.info(f"Aplicando RobustScaler a {len(variables_robust)} variables...")
        scaler_robust = RobustScaler()
        dataset_escalado[variables_robust] = scaler_robust.fit_transform(
            dataset_escalado[variables_robust]
        )
        scalers['robust'] = {
            'scaler': scaler_robust,
            'variables': variables_robust
        }
    
    # 3. Crear versiones escaladas con sufijos
    logger.info("Creando versiones escaladas con sufijos...")
    
    # Renombrar columnas escaladas para identificar el método usado
    for metodo, info in scalers.items():
        variables = info['variables']
        for var in variables:
            nueva_columna = f"{var}_{metodo}"
            dataset_escalado[nueva_columna] = dataset_escalado[var]
            # Mantener la original también
            dataset_escalado[f"{var}_original"] = dataset_codificado[var]
    
    # 4. Resumen de escalado
    logger.info("=== RESUMEN DE ESCALADO ===")
    
    logger.info(f"Métodos de escalado aplicados: {len(scalers)}")
    for metodo, info in scalers.items():
        logger.info(f"  {metodo}: {len(info['variables'])} variables")
    
    logger.info(f"Dataset final: {dataset_escalado.shape}")
    
    # Verificar que no hay valores infinitos o NaN
    valores_infinitos = np.isinf(dataset_escalado.select_dtypes(include=[np.number])).sum().sum()
    valores_nan = dataset_escalado.isnull().sum().sum()
    
    logger.info(f"Valores infinitos: {valores_infinitos}")
    logger.info(f"Valores NaN: {valores_nan}")
    
    if valores_infinitos > 0:
        logger.warning("Se encontraron valores infinitos después del escalado")
    
    if valores_nan > 0:
        logger.warning("Se encontraron valores NaN después del escalado")
    
    logger.info("Escalado de características completado exitosamente")
    return dataset_escalado, scalers


def preparar_datos_modelado(
    dataset_escalado: pd.DataFrame,
    scalers: Dict[str, Any],
    mapeos_codificacion: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepara los datos finales para modelado de machine learning.
    
    Esta función organiza los datos en formatos apropiados para diferentes
    tipos de modelos y identifica variables objetivo para regresión y clasificación.
    
    Args:
        dataset_escalado: Dataset con características escaladas
        scalers: Diccionario con scalers utilizados
        mapeos_codificacion: Diccionario con mapeos de codificación
        
    Returns:
        Diccionario con datos preparados para modelado
    """
    logger.info("Iniciando preparación de datos para modelado...")
    
    # Crear copia para trabajar
    dataset_modelado = dataset_escalado.copy()
    
    # 1. Identificar variables objetivo
    logger.info("Identificando variables objetivo...")
    
    # Variables objetivo para regresión
    targets_regresion = {
        'total_nacimientos_año': 'Regresión - Total de nacimientos por año',
        'total_defunciones_año': 'Regresión - Total de defunciones por año',
        'crecimiento_poblacional': 'Regresión - Crecimiento poblacional',
        'tasa_natalidad': 'Regresión - Tasa de natalidad',
        'edad_promedio_defunciones': 'Regresión - Edad promedio de defunciones'
    }
    
    # Variables objetivo para clasificación
    targets_clasificacion = {
        'crecimiento_positivo': 'Clasificación - Crecimiento poblacional positivo',
        'mas_nacimientos_que_defunciones': 'Clasificación - Más nacimientos que defunciones',
        'alta_volatilidad_nacimientos': 'Clasificación - Alta volatilidad en nacimientos',
        'categoria_crecimiento_codificada': 'Clasificación - Categoría de crecimiento',
        'categoria_tasa_natalidad_codificada': 'Clasificación - Categoría de tasa de natalidad'
    }
    
    logger.info(f"Variables objetivo para regresión: {len(targets_regresion)}")
    logger.info(f"Variables objetivo para clasificación: {len(targets_clasificacion)}")
    
    # 2. Identificar features para modelado
    logger.info("Identificando features para modelado...")
    
    # Excluir variables objetivo y identificadores
    columnas_excluidas = [
        'año', 'decada', 'siglo', 'ciclo_5_anos', 'ciclo_10_anos',
        'categoria_crecimiento', 'categoria_tasa_natalidad'
    ] + list(targets_regresion.keys()) + list(targets_clasificacion.keys())
    
    # Identificar features
    features = [col for col in dataset_modelado.columns if col not in columnas_excluidas]
    
    logger.info(f"Features identificadas: {len(features)}")
    
    # 3. Crear datasets de entrenamiento y prueba
    logger.info("Creando datasets de entrenamiento y prueba...")
    
    # Ordenar por año para mantener orden temporal
    dataset_modelado = dataset_modelado.sort_values('año')
    
    # Dividir en entrenamiento (80%) y prueba (20%)
    split_index = int(len(dataset_modelado) * 0.8)
    
    dataset_entrenamiento = dataset_modelado.iloc[:split_index].copy()
    dataset_prueba = dataset_modelado.iloc[split_index:].copy()
    
    logger.info(f"Dataset entrenamiento: {dataset_entrenamiento.shape}")
    logger.info(f"Dataset prueba: {dataset_prueba.shape}")
    
    # 4. Preparar matrices X e Y para cada tipo de modelo
    logger.info("Preparando matrices X e Y...")
    
    datos_modelado = {
        'features': features,
        'targets_regresion': targets_regresion,
        'targets_clasificacion': targets_clasificacion,
        'dataset_completo': dataset_modelado,
        'dataset_entrenamiento': dataset_entrenamiento,
        'dataset_prueba': dataset_prueba,
        'scalers': scalers,
        'mapeos_codificacion': mapeos_codificacion
    }
    
    # Crear matrices X e Y para regresión
    for target_name, target_desc in targets_regresion.items():
        if target_name in dataset_modelado.columns:
            # Datos completos
            X_completo = dataset_modelado[features]
            y_completo = dataset_modelado[target_name]
            
            # Datos de entrenamiento
            X_entrenamiento = dataset_entrenamiento[features]
            y_entrenamiento = dataset_entrenamiento[target_name]
            
            # Datos de prueba
            X_prueba = dataset_prueba[features]
            y_prueba = dataset_prueba[target_name]
            
            datos_modelado[f'X_{target_name}'] = X_completo
            datos_modelado[f'y_{target_name}'] = y_completo
            datos_modelado[f'X_train_{target_name}'] = X_entrenamiento
            datos_modelado[f'y_train_{target_name}'] = y_entrenamiento
            datos_modelado[f'X_test_{target_name}'] = X_prueba
            datos_modelado[f'y_test_{target_name}'] = y_prueba
    
    # Crear matrices X e Y para clasificación
    for target_name, target_desc in targets_clasificacion.items():
        if target_name in dataset_modelado.columns:
            # Datos completos
            X_completo = dataset_modelado[features]
            y_completo = dataset_modelado[target_name]
            
            # Datos de entrenamiento
            X_entrenamiento = dataset_entrenamiento[features]
            y_entrenamiento = dataset_entrenamiento[target_name]
            
            # Datos de prueba
            X_prueba = dataset_prueba[features]
            y_prueba = dataset_prueba[target_name]
            
            datos_modelado[f'X_{target_name}'] = X_completo
            datos_modelado[f'y_{target_name}'] = y_completo
            datos_modelado[f'X_train_{target_name}'] = X_entrenamiento
            datos_modelado[f'y_train_{target_name}'] = y_entrenamiento
            datos_modelado[f'X_test_{target_name}'] = X_prueba
            datos_modelado[f'y_test_{target_name}'] = y_prueba
    
    # 5. Resumen final
    logger.info("=== RESUMEN DE PREPARACIÓN PARA MODELADO ===")
    
    logger.info(f"Dataset completo: {dataset_modelado.shape}")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Variables objetivo regresión: {len(targets_regresion)}")
    logger.info(f"Variables objetivo clasificación: {len(targets_clasificacion)}")
    logger.info(f"Entrenamiento: {dataset_entrenamiento.shape[0]} registros")
    logger.info(f"Prueba: {dataset_prueba.shape[0]} registros")
    
    # Verificar calidad de datos
    logger.info("Verificando calidad de datos...")
    
    # Verificar valores faltantes
    valores_faltantes = dataset_modelado[features].isnull().sum().sum()
    logger.info(f"Valores faltantes en features: {valores_faltantes}")
    
    # Verificar valores infinitos
    valores_infinitos = np.isinf(dataset_modelado[features].select_dtypes(include=[np.number])).sum().sum()
    logger.info(f"Valores infinitos en features: {valores_infinitos}")
    
    # Verificar correlaciones altas
    correlaciones = dataset_modelado[features].corr().abs()
    correlaciones_altas = (correlaciones > 0.95).sum().sum() - len(features)  # Restar diagonal
    logger.info(f"Correlaciones altas (>0.95): {correlaciones_altas}")
    
    logger.info("Preparación de datos para modelado completada exitosamente")
    return datos_modelado
