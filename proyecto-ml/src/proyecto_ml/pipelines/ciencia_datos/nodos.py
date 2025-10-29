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


def preparar_dataset_individual_para_ml(
    datasets_estandarizados: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prepara dataset de defunciones INDIVIDUALES para Machine Learning.
    
    A diferencia del flujo principal que agrega datos por año (50 filas),
    este nodo mantiene los registros INDIVIDUALES (1.2M filas) necesarios
    para problemas de clasificación como:
    - Predecir SEXO del fallecido
    - Predecir REGIÓN geográfica
    - Predecir EDAD de fallecimiento
    
    Args:
        datasets_estandarizados: Diccionario con datasets del pipeline ingeniería_datos
        params: Parámetros de configuración
        
    Returns:
        DataFrame con 1.2M registros y features para ML:
        - sexo: Variable para clasificación
        - region: Variable para clasificación  
        - edad_cantidad: Variable para regresión
        - Features temporales cíclicos (mes_sin, mes_cos, etc.)
        - Features derivados
    """
    logger.info("=" * 80)
    logger.info("PREPARANDO DATASET INDIVIDUAL PARA MACHINE LEARNING")
    logger.info("=" * 80)
    
    # 1. Extraer dataset de defunciones individuales
    df = datasets_estandarizados['defunciones_estandarizado'].copy()
    logger.info(f"Dataset inicial: {df.shape}")
    logger.info(f"Columnas disponibles: {list(df.columns)}")
    
    # 2. Convertir columnas temporales a numérico si es necesario
    logger.info("\nConvirtiendo columnas temporales a numérico...")
    columnas_numericas = ['año', 'mes', 'dia_semana', 'trimestre', 'dia_año', 'tipo_edad', 'edad_cantidad']
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Rellenar valores nulos en columnas temporales
            if df[col].isnull().sum() > 0:
                # Para columnas temporales, usar el valor más común (moda) o 0
                if col in ['dia_semana', 'mes', 'trimestre']:
                    moda = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                    df[col] = df[col].fillna(moda)
                    logger.info(f"  '{col}': {df[col].isnull().sum()} nulos rellenados con moda ({moda})")
                else:
                    mediana = df[col].median()
                    df[col] = df[col].fillna(mediana if pd.notna(mediana) else 0)
                    logger.info(f"  '{col}': nulos rellenados con mediana")
    
    # 3. Crear features temporales CÍCLICOS (para capturar estacionalidad)
    logger.info("\nCreando features temporales cíclicos...")
    
    # Mes (1-12) → seno y coseno
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Día del año (1-365) → seno y coseno
    df['dia_año_sin'] = np.sin(2 * np.pi * df['dia_año'] / 365)
    df['dia_año_cos'] = np.cos(2 * np.pi * df['dia_año'] / 365)
    
    # Trimestre (1-4) → seno y coseno
    df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    
    # Día de la semana (0-6) → seno y coseno
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    
    logger.info("✓ Features cíclicos creados")
    
    # 4. Crear features derivados MEJORADOS para ML
    logger.info("\nCreando features derivados MEJORADOS...")
    
    # === FEATURES TEMPORALES ===
    # Indicador de fin de semana
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    # Indicador de estación (aproximado para Chile)
    df['es_invierno'] = df['mes'].isin([6, 7, 8]).astype(int)  # Invierno en Chile
    df['es_verano'] = df['mes'].isin([12, 1, 2]).astype(int)   # Verano en Chile
    df['es_primavera'] = df['mes'].isin([9, 10, 11]).astype(int)  # Primavera
    df['es_otono'] = df['mes'].isin([3, 4, 5]).astype(int)     # Otoño
    
    # Trimestre fiscal
    df['trimestre_fiscal'] = ((df['mes'] - 1) // 3) + 1
    
    # Año normalizado (para regresión)
    df['año_normalizado'] = (df['año'] - df['año'].min()) / (df['año'].max() - df['año'].min())
    
    # Década
    df['decada'] = (df['año'] // 10) * 10
    
    # Época del año codificada (0-3 para cada estación)
    df['epoca_año_codificada'] = pd.cut(df['mes'], bins=[0, 3, 6, 9, 12], 
                                         labels=[0, 1, 2, 3]).astype(int)
    
    # === FEATURES DE EDAD MEJORADOS ===
    # Rango de edad (más informativo que edad exacta)
    df['rango_edad'] = pd.cut(df['edad_cantidad'], 
                              bins=[0, 1, 5, 18, 30, 50, 65, 80, 120], 
                              labels=['bebe', 'infante', 'adolescente', 'joven', 'adulto', 'maduro', 'adulto_mayor', 'anciano'])
    
    # Edad normalizada (0-1)
    df['edad_normalizada'] = (df['edad_cantidad'] - df['edad_cantidad'].min()) / (df['edad_cantidad'].max() - df['edad_cantidad'].min())
    
    # Indicadores de edad específicos
    df['es_menor_edad'] = (df['edad_cantidad'] < 18).astype(int)
    df['es_adulto_joven'] = ((df['edad_cantidad'] >= 18) & (df['edad_cantidad'] < 30)).astype(int)
    df['es_adulto_maduro'] = ((df['edad_cantidad'] >= 30) & (df['edad_cantidad'] < 65)).astype(int)
    df['es_adulto_mayor'] = (df['edad_cantidad'] >= 65).astype(int)
    
    # === FEATURES DE REGIÓN MEJORADOS ===
    # Agrupar regiones por zona geográfica (más balanceado)
    regiones_norte = ['De Arica y Parinacota', 'De Tarapacá', 'De Antofagasta', 'De Atacama', 'De Coquimbo']
    regiones_centro = ['De Valparaíso', 'Metropolitana de Santiago', 'Del Libertador B. O\'Higgins', 'Del Maule', 'De Ñuble']
    regiones_sur = ['Del Bíobío', 'De La Araucanía', 'De Los Ríos', 'De Los Lagos', 'De Aisén del Gral. C. Ibáñez del Campo', 'De Magallanes y de La Antártica Chilena']
    
    df['zona_geografica'] = 'Ignorada'  # Default
    df.loc[df['region'].isin(regiones_norte), 'zona_geografica'] = 'Norte'
    df.loc[df['region'].isin(regiones_centro), 'zona_geografica'] = 'Centro'
    df.loc[df['region'].isin(regiones_sur), 'zona_geografica'] = 'Sur'
    
    # Indicadores de zona
    df['es_norte'] = (df['zona_geografica'] == 'Norte').astype(int)
    df['es_centro'] = (df['zona_geografica'] == 'Centro').astype(int)
    df['es_sur'] = (df['zona_geografica'] == 'Sur').astype(int)
    
    # === FEATURES COMBINADOS ===
    # Interacciones edad-temporales
    df['edad_mes_interaccion'] = df['edad_cantidad'] * df['mes']
    df['edad_trimestre_interaccion'] = df['edad_cantidad'] * df['trimestre']
    
    # Interacciones región-temporales
    df['region_mes_interaccion'] = df['region'].astype('category').cat.codes * df['mes']
    
    # === FEATURES ESPECÍFICOS PARA CLASIFICACIÓN DE SEXO ===
    # Patrones de mortalidad por edad y género (basado en estadísticas reales)
    # Hombres tienden a morir más jóvenes, mujeres más mayores
    
    # Edad promedio de mortalidad por género (aproximado)
    df['edad_promedio_hombres'] = 65  # Aproximado
    df['edad_promedio_mujeres'] = 75  # Aproximado
    
    # Desviación respecto a edad promedio de mortalidad
    df['desviacion_edad_hombres'] = df['edad_cantidad'] - df['edad_promedio_hombres']
    df['desviacion_edad_mujeres'] = df['edad_cantidad'] - df['edad_promedio_mujeres']
    
    # Indicadores de riesgo por edad (basado en patrones epidemiológicos)
    df['riesgo_mortalidad_joven'] = ((df['edad_cantidad'] >= 15) & (df['edad_cantidad'] <= 35)).astype(int)  # Más hombres
    df['riesgo_mortalidad_adulto'] = ((df['edad_cantidad'] >= 35) & (df['edad_cantidad'] <= 65)).astype(int)  # Equilibrado
    df['riesgo_mortalidad_mayor'] = (df['edad_cantidad'] > 65).astype(int)  # Más mujeres
    
    # Patrones estacionales por género (hipótesis epidemiológica)
    df['patron_invierno'] = (df['es_invierno'] * df['edad_cantidad']).astype(int)  # Enfermedades respiratorias
    df['patron_verano'] = (df['es_verano'] * df['edad_cantidad']).astype(int)  # Accidentes, golpes de calor
    
    # Features de interacción edad-región (diferentes patrones por zona)
    df['edad_norte'] = df['edad_cantidad'] * df['es_norte']
    df['edad_centro'] = df['edad_cantidad'] * df['es_centro'] 
    df['edad_sur'] = df['edad_cantidad'] * df['es_sur']
    
    # === FEATURES ESTADÍSTICOS ===
    # Percentiles de edad por región
    df['edad_percentil_region'] = df.groupby('region')['edad_cantidad'].transform(lambda x: x.rank(pct=True))
    
    # Edad promedio por región
    df['edad_promedio_region'] = df.groupby('region')['edad_cantidad'].transform('mean')
    
    # Desviación de edad respecto al promedio regional
    df['edad_desviacion_region'] = df['edad_cantidad'] - df['edad_promedio_region']
    
    logger.info("✓ Features derivados MEJORADOS creados")
    
    # 5. Limpiar y preparar columnas finales
    logger.info("\nSeleccionando columnas finales para ML...")
    
    # Columnas finales que necesitamos para ML (MEJORADAS)
    columnas_finales = [
        # Variables objetivo para CLASIFICACIÓN
        'sexo',                    # Clasificar sexo (Hombre/Mujer)
        'region',                  # Clasificar región geográfica
        'zona_geografica',         # Clasificar zona geográfica (Norte/Centro/Sur)
        
        # Variables objetivo para REGRESIÓN
        'edad_cantidad',           # Predecir edad de fallecimiento
        'año_normalizado',         # Predecir año normalizado
        
        # CAUSA DE MUERTE (CIE-10) - VARIABLE MÁS IMPORTANTE  
        'codigo_diagnostico',      # Código capítulo CIE-10 (ej: I00-I99)
        
        # Features temporales cíclicos
        'mes_sin', 'mes_cos',
        'dia_año_sin', 'dia_año_cos',
        'trimestre_sin', 'trimestre_cos',
        'dia_semana_sin', 'dia_semana_cos',
        
        # Features temporales mejorados
        'es_fin_semana',
        'es_invierno', 'es_verano', 'es_primavera', 'es_otono',
        'trimestre_fiscal',
        'epoca_año_codificada',
        'decada',
        
        # Features de edad mejorados
        'rango_edad',              # Rango de edad categórico
        'edad_normalizada',        # Edad normalizada (0-1)
        'es_menor_edad', 'es_adulto_joven', 'es_adulto_maduro', 'es_adulto_mayor',
        
        # Features de región mejorados
        'es_norte', 'es_centro', 'es_sur',
        
        # Features combinados
        'edad_mes_interaccion',
        'edad_trimestre_interaccion',
        'region_mes_interaccion',
        
        # Features específicos para clasificación de SEXO
        'desviacion_edad_hombres',
        'desviacion_edad_mujeres',
        'riesgo_mortalidad_joven',
        'riesgo_mortalidad_adulto',
        'riesgo_mortalidad_mayor',
        'patron_invierno',
        'patron_verano',
        'edad_norte',
        'edad_centro',
        'edad_sur',
        
        # Features estadísticos
        'edad_percentil_region',
        'edad_promedio_region',
        'edad_desviacion_region',
        
        # Features adicionales útiles
        'tipo_edad',               # Tipo de edad (años, meses, días)
        'año',                     # Año original (para referencia)
        'mes',                     # Mes original
        'trimestre',               # Trimestre original
    ]
    
    # Verificar que todas las columnas existen
    columnas_disponibles = [col for col in columnas_finales if col in df.columns]
    columnas_faltantes = set(columnas_finales) - set(columnas_disponibles)
    
    if columnas_faltantes:
        logger.warning(f"Columnas faltantes: {columnas_faltantes}")
    
    df_final = df[columnas_disponibles].copy()
    
    # 6. Limpieza final
    logger.info("\nRealizando limpieza final...")
    
    # Eliminar filas con valores nulos en variables críticas
    antes = len(df_final)
    df_final = df_final.dropna(subset=['sexo', 'region', 'edad_cantidad'])
    despues = len(df_final)
    logger.info(f"Filas eliminadas por nulos: {antes - despues}")
    
    # Convertir edad_cantidad a numérico si no lo es
    df_final['edad_cantidad'] = pd.to_numeric(df_final['edad_cantidad'], errors='coerce')
    
    # Eliminar filas con edad_cantidad nula después de conversión
    df_final = df_final.dropna(subset=['edad_cantidad'])
    
    # IMPORTANTE: Eliminar sexo "Indeterminado" para clasificación binaria limpia
    logger.info("\nFiltrando datos para clasificación binaria...")
    antes_sexo = len(df_final)
    df_final = df_final[df_final['sexo'].isin(['Hombre', 'Mujer'])]
    despues_sexo = len(df_final)
    logger.info(f"Registros con sexo 'Indeterminado' eliminados: {antes_sexo - despues_sexo}")
    logger.info(f"Dataset después de limpieza: {df_final.shape}")
    
    # Tomar muestra estratificada de 100,000 registros
    # Esto acelera el entrenamiento manteniendo la proporción de clases
    sample_size = params.get('sample_size_ml', 100000)
    
    if len(df_final) > sample_size:
        logger.info(f"\nCreando muestra estratificada de {sample_size:,} registros...")
        logger.info("Esto mantiene la proporción de clases y acelera el entrenamiento")
        
        # Muestra estratificada por sexo y región
        from sklearn.model_selection import train_test_split
        df_final, _ = train_test_split(
            df_final,
            train_size=sample_size,
            stratify=df_final['sexo'],  # Mantener proporción de sexo
            random_state=42
        )
        logger.info(f"Muestra estratificada creada: {df_final.shape}")
    else:
        logger.info(f"Dataset ya tiene menos de {sample_size:,} registros, usando todos")
    
    # 7. Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("DATASET PARA ML PREPARADO")
    logger.info("=" * 80)
    logger.info(f"Shape final: {df_final.shape}")
    logger.info(f"Columnas finales: {len(df_final.columns)}")
    logger.info(f"\nVariables objetivo:")
    logger.info(f"  - sexo: {df_final['sexo'].nunique()} categorías")
    logger.info(f"  - region: {df_final['region'].nunique()} categorías")
    logger.info(f"  - edad_cantidad: min={df_final['edad_cantidad'].min():.0f}, max={df_final['edad_cantidad'].max():.0f}")
    logger.info(f"\nDistribución de sexo:")
    for sexo, count in df_final['sexo'].value_counts().items():
        logger.info(f"  {sexo}: {count:,}")
    logger.info(f"\nMemoria usada: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_final


def preparar_dataset_para_regresion(
    datasets_estandarizados: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prepara dataset LIMPIO para regresión de edad (SIN data leakage).
    
    A diferencia de preparar_dataset_individual_para_ml, este dataset:
    - NO incluye variables derivadas de edad (rango_edad, edad_normalizada, etc.)
    - Solo incluye features INDEPENDIENTES de la edad
    - Diseñado específicamente para predecir edad_cantidad
    
    Variables incluidas:
    - sexo, region, codigo_diagnostico (demográficas y causa de muerte)
    - Features temporales cíclicos (mes_sin, mes_cos, etc.)
    - Features estacionales (es_invierno, es_verano, etc.)
    - Features geográficos (es_norte, es_centro, es_sur)
    
    Variables EXCLUIDAS (data leakage):
    - rango_edad, edad_normalizada
    - es_menor_edad, es_adulto_joven, es_adulto_maduro, es_adulto_mayor
    - edad_mes_interaccion, edad_trimestre_interaccion
    - desviacion_edad_hombres/mujeres
    - edad_norte/centro/sur, edad_percentil_region, edad_promedio_region
    - tipo_edad
    
    Args:
        datasets_estandarizados: Diccionario con datasets del pipeline ingeniería_datos
        params: Parámetros de configuración
        
    Returns:
        DataFrame limpio para regresión de edad (sin data leakage)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PREPARANDO DATASET LIMPIO PARA REGRESIÓN DE EDAD")
    logger.info("=" * 80)
    
    # 1. Extraer dataset de defunciones individuales
    df = datasets_estandarizados['defunciones_estandarizado'].copy()
    logger.info(f"Dataset inicial: {df.shape}")
    
    # 2. Convertir columnas temporales a numérico si es necesario
    logger.info("\nConvirtiendo columnas temporales a numérico...")
    for col in ['dia_semana', 'trimestre', 'dia_año', 'mes']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            moda = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            nulos = df[col].isnull().sum()
            df[col] = df[col].fillna(moda)
            logger.info(f"  '{col}': {nulos} nulos rellenados con moda ({moda})")
    
    # 3. Crear features temporales cíclicos
    logger.info("\nCreando features temporales cíclicos...")
    import numpy as np
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_año_sin'] = np.sin(2 * np.pi * df['dia_año'] / 365)
    df['dia_año_cos'] = np.cos(2 * np.pi * df['dia_año'] / 365)
    df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    
    logger.info("✓ Features cíclicos creados")
    
    # 4. Crear features estacionales
    logger.info("\nCreando features estacionales...")
    
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['es_invierno'] = df['mes'].isin([6, 7, 8]).astype(int)
    df['es_verano'] = df['mes'].isin([12, 1, 2]).astype(int)
    df['es_primavera'] = df['mes'].isin([9, 10, 11]).astype(int)
    df['es_otono'] = df['mes'].isin([3, 4, 5]).astype(int)
    df['trimestre_fiscal'] = ((df['mes'] - 1) // 3) + 1
    df['epoca_año_codificada'] = (df['mes'] - 1) // 3
    df['decada'] = (df['año'] // 10) * 10
    
    logger.info("✓ Features estacionales creados")
    
    # 5. Crear features geográficos
    logger.info("\nCreando features geográficos...")
    
    # Zonas geográficas de Chile
    regiones_norte = ['Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo']
    regiones_centro = ['Valparaíso', 'Metropolitana de Santiago', "Libertador General Bernardo O'Higgins", 'Maule', 'Ñuble', 'Biobío']
    regiones_sur = ['La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén del General Carlos Ibáñez del Campo', 'De Magallanes y de La Antártica Chilena']
    
    df['es_norte'] = df['region'].isin(regiones_norte).astype(int)
    df['es_centro'] = df['region'].isin(regiones_centro).astype(int)
    df['es_sur'] = df['region'].isin(regiones_sur).astype(int)
    
    logger.info("✓ Features geográficos creados")
    
    # 6. Seleccionar SOLO columnas sin data leakage
    logger.info("\nSeleccionando columnas LIMPIAS (sin data leakage)...")
    
    columnas_finales = [
        # Variable objetivo
        'edad_cantidad',
        
        # Variables demográficas y causa de muerte
        'sexo',
        'region',
        'codigo_diagnostico',  # CIE-10
        
        # Features temporales cíclicos
        'mes_sin', 'mes_cos',
        'dia_año_sin', 'dia_año_cos',
        'trimestre_sin', 'trimestre_cos',
        'dia_semana_sin', 'dia_semana_cos',
        
        # Features estacionales
        'es_fin_semana',
        'es_invierno', 'es_verano', 'es_primavera', 'es_otono',
        'trimestre_fiscal',
        'epoca_año_codificada',
        'decada',
        
        # Features geográficos
        'es_norte', 'es_centro', 'es_sur',
    ]
    
    # Verificar columnas disponibles
    columnas_disponibles = [col for col in columnas_finales if col in df.columns]
    columnas_faltantes = set(columnas_finales) - set(columnas_disponibles)
    
    if columnas_faltantes:
        logger.warning(f"Columnas faltantes: {columnas_faltantes}")
    
    df_final = df[columnas_disponibles].copy()
    logger.info(f"Columnas seleccionadas: {len(columnas_disponibles)}")
    
    # =========================================================================
    # 6.5 APLICAR ONEHOT ENCODING A VARIABLES CATEGÓRICAS
    # =========================================================================
    # EXPLICACIÓN:
    # Convertimos variables categóricas (region, codigo_diagnostico) en columnas
    # binarias (0/1) para que el modelo NO piense que hay orden entre categorías.
    #
    # Ejemplo ANTES:
    #   region = "Arica"           → LabelEncoding → 0
    #   region = "Metropolitana"   → LabelEncoding → 13
    #   Problema: El modelo piensa "Metropolitana es 13 veces más que Arica" ❌
    #
    # Ejemplo DESPUÉS (OneHot):
    #   region = "Arica"           → region_Arica=1, region_RM=0, ...
    #   region = "Metropolitana"   → region_Arica=0, region_RM=1, ...
    #   Ventaja: Cada región es independiente, sin orden artificial ✅
    #
    # drop_first=True: Elimina la primera columna para evitar multicolinealidad
    #   Si region_Arica=0 y region_Tarapaca=0 y ... → entonces es la región omitida
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("APLICANDO ONEHOT ENCODING A VARIABLES CATEGÓRICAS")
    logger.info("="*70)
    
    columnas_antes = len(df_final.columns)
    logger.info(f"\nColumnas ANTES de OneHot: {columnas_antes}")
    
    # 1. SEXO: Convertir a binario (0=Hombre, 1=Mujer)
    # NO necesita OneHot porque solo tiene 2 categorías
    if 'sexo' in df_final.columns:
        logger.info("\n1. SEXO (convertir a binario 0/1):")
        logger.info(f"   Valores únicos: {df_final['sexo'].unique()}")
        sexo_map = {'Hombre': 0, 'Mujer': 1}
        df_final['sexo'] = df_final['sexo'].map(sexo_map)
        logger.info("   ✓ Convertido: Hombre=0, Mujer=1")
        logger.info(f"   Distribución: {df_final['sexo'].value_counts().to_dict()}")
    
    # 2. REGION: Aplicar OneHotEncoding
    # 17 regiones → 16 columnas (drop_first=True)
    if 'region' in df_final.columns:
        logger.info("\n2. REGION (OneHotEncoding):")
        logger.info(f"   Regiones únicas: {df_final['region'].nunique()}")
        logger.info(f"   Ejemplos: {list(df_final['region'].unique()[:3])}")
        
        # pd.get_dummies crea columnas: region_Arica y Parinacota, region_Tarapacá, etc.
        df_final = pd.get_dummies(df_final, columns=['region'], prefix='region', drop_first=True)
        
        region_cols = [col for col in df_final.columns if col.startswith('region_')]
        logger.info(f"   ✓ Columnas creadas: {len(region_cols)}")
        logger.info(f"   Ejemplos: {region_cols[:3]}")
        logger.info("   Interpretación:")
        logger.info("     - region_Arica y Parinacota=1 significa 'es de Arica'")
        logger.info("     - Cada región aprende su patrón de mortalidad independiente")
    
    # 3. CODIGO_DIAGNOSTICO (CIE-10): Aplicar OneHotEncoding
    # 20 códigos → 19 columnas (drop_first=True)
    # ESTE ES EL MÁS IMPORTANTE: cada causa tiene patrón de edad diferente
    if 'codigo_diagnostico' in df_final.columns:
        logger.info("\n3. CODIGO_DIAGNOSTICO / CIE-10 (OneHotEncoding):")
        logger.info(f"   Códigos únicos: {df_final['codigo_diagnostico'].nunique()}")
        
        # Mostrar distribución de causas
        top_codes = df_final['codigo_diagnostico'].value_counts().head(5)
        logger.info("   Top 5 causas de muerte:")
        for code, count in top_codes.items():
            logger.info(f"     * {code}: {count:,} casos ({count/len(df_final)*100:.1f}%)")
        
        # Aplicar OneHot
        df_final = pd.get_dummies(df_final, columns=['codigo_diagnostico'], prefix='cie10', drop_first=True)
        
        cie10_cols = [col for col in df_final.columns if col.startswith('cie10_')]
        logger.info(f"   ✓ Columnas creadas: {len(cie10_cols)}")
        logger.info(f"   Ejemplos: {cie10_cols[:3]}")
        logger.info("   Interpretación:")
        logger.info("     - cie10_I00-I99=1 significa 'causa cardiovascular'")
        logger.info("     - cie10_S00-T98=1 significa 'accidentes/traumatismos'")
        logger.info("     - Cada causa aprenderá su patrón de edad específico:")
        logger.info("       * I00-I99: edad alta (~75 años)")
        logger.info("       * S00-T98: edad baja (~45 años)")
    
    columnas_despues = len(df_final.columns)
    logger.info("\n" + "="*70)
    logger.info("RESUMEN ONEHOT ENCODING:")
    logger.info("="*70)
    logger.info(f"  Columnas ANTES:  {columnas_antes}")
    logger.info(f"  Columnas DESPUÉS: {columnas_despues}")
    logger.info(f"  Nuevas columnas:  {columnas_despues - columnas_antes}")
    logger.info(f"  Shape final: {df_final.shape}")
    logger.info("✓ Variables categóricas convertidas a numéricas")
    logger.info("="*70)
    
    # 7. Limpieza final
    logger.info("\nRealizando limpieza final...")
    
    # Eliminar filas con valores nulos en edad_cantidad y sexo
    # NOTA: codigo_diagnostico ya no existe, se convirtió en cie10_*
    nulos_antes = df_final.shape[0]
    df_final = df_final.dropna(subset=['edad_cantidad', 'sexo'])
    nulos_despues = df_final.shape[0]
    logger.info(f"Filas eliminadas por nulos: {nulos_antes - nulos_despues}")
    
    # Eliminar sexo 'Indeterminado' (si existe)
    if 'Indeterminado' in df_final['sexo'].unique():
        indeterminados = (df_final['sexo'] == 'Indeterminado').sum()
        df_final = df_final[df_final['sexo'] != 'Indeterminado']
        logger.info(f"Registros con sexo 'Indeterminado' eliminados: {indeterminados}")
    
    # 8. Muestra estratificada (100K registros)
    sample_size = params.get('sample_size_ml', 100000)
    
    if len(df_final) > sample_size:
        logger.info(f"\nCreando muestra estratificada de {sample_size:,} registros...")
        from sklearn.model_selection import train_test_split
        df_final, _ = train_test_split(
            df_final,
            train_size=sample_size,
            stratify=df_final['sexo'],
            random_state=42
        )
        logger.info(f"Muestra estratificada creada: {df_final.shape}")
    
    # 9. Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("DATASET LIMPIO PARA REGRESIÓN PREPARADO")
    logger.info("=" * 80)
    logger.info(f"Shape final: {df_final.shape}")
    logger.info(f"Columnas finales: {len(df_final.columns)}")
    logger.info(f"\nVariable objetivo:")
    logger.info(f"  - edad_cantidad: min={df_final['edad_cantidad'].min():.0f}, max={df_final['edad_cantidad'].max():.0f}, mean={df_final['edad_cantidad'].mean():.1f}")
    logger.info(f"\nDistribución de sexo:")
    for sexo, count in df_final['sexo'].value_counts().items():
        logger.info(f"  {sexo}: {count:,}")
    
    # Contar columnas OneHot
    cie10_cols = [col for col in df_final.columns if col.startswith('cie10_')]
    region_cols = [col for col in df_final.columns if col.startswith('region_')]
    logger.info(f"\nColumnas OneHot creadas:")
    logger.info(f"  - CIE-10: {len(cie10_cols)} columnas")
    logger.info(f"  - Regiones: {len(region_cols)} columnas")
    logger.info(f"Memoria usada: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_final


def integrar_datasets(
    datasets_estandarizados: Dict[str, pd.DataFrame],
    datos_historicos: pd.DataFrame
) -> pd.DataFrame:
    """
    Crea un dataset unificado integrando múltiples fuentes de datos.
    
    Esta función implementa la lógica del notebook (sección 6: Integración de Datasets):
    1. Usa datos históricos como base (1974-2023)
    2. Integra información por sexo (2015-2023)
    3. Crea variables derivadas: tasas, ratios, crecimiento natural
    
    Args:
        datasets_estandarizados: Diccionario con datasets estandarizados del pipeline de ingeniería
                                 (incluye 'defunciones_estandarizado', 'nacimientos_por_sexo_estandarizado')
        datos_historicos: Dataset histórico de nacimientos y defunciones (1974-2023)
        
    Returns:
        Dataset unificado con información integrada y variables derivadas
    """
    logger.info("Iniciando integración de datasets...")
    
    # Estandarizar columnas de datos históricos primero
    logger.info("Estandarizando columnas de datos históricos...")
    datos_historicos_std = datos_historicos.copy()
    
    # Mapeo de columnas de setdedatos.csv
    mapeo_historicos = {
        'año': 'año',  # Ya está bien
        'a�o': 'año',  # Encoding problem
        'Año': 'año',  # Mayúscula
        'Nacimientos': 'nacimientos_totales',
        'Defunciones': 'defunciones_totales'
    }
    
    # Renombrar solo las columnas que existen
    columnas_existentes = {k: v for k, v in mapeo_historicos.items() if k in datos_historicos_std.columns}
    datos_historicos_std = datos_historicos_std.rename(columns=columnas_existentes)
    logger.info(f"Columnas estandarizadas: {list(datos_historicos_std.columns)}")
    
    # Extraer datasets del diccionario
    nacimientos_por_sexo = datasets_estandarizados.get("nacimientos_por_sexo_estandarizado")
    
    if nacimientos_por_sexo is None:
        logger.warning("No se encontró 'nacimientos_por_sexo_estandarizado', usando solo datos históricos")
        dataset_unificado = datos_historicos_std.copy()
    else:
        logger.info(f"Dataset histórico: {datos_historicos_std.shape}")
        logger.info(f"Dataset por sexo: {nacimientos_por_sexo.shape}")
        
        # 1. Crear dataset base con información histórica (serie más larga: 1974-2023)
        dataset_unificado = datos_historicos_std.copy()
        logger.info(f"Dataset base (datos históricos): {dataset_unificado.shape}")
        
        # 2. Integrar información por sexo para años 2015-2023
        logger.info("Integrando información por sexo...")
        dataset_unificado = dataset_unificado.merge(
            nacimientos_por_sexo,
            on='año',
            how='left',
            suffixes=('_total', '_sexo')
        )
        logger.info(f"Después de integrar por_sexo: {dataset_unificado.shape}")
        
        # 3. Generar variables derivadas (según notebook sección 6.3)
        logger.info("Generando variables derivadas...")
        
        # Tasas de natalidad y mortalidad (aproximación sin población exacta)
        dataset_unificado['tasa_natalidad'] = (
            dataset_unificado['nacimientos_totales'] / 1000
        ).round(2)
        
        dataset_unificado['tasa_mortalidad'] = (
            dataset_unificado['defunciones_totales'] / 1000
        ).round(2)
        
        # Ratio de nacimientos por sexo (solo para años con datos completos)
        if 'nacimientos_hombres' in dataset_unificado.columns:
            dataset_unificado['ratio_nacimientos_sexo'] = (
                dataset_unificado['nacimientos_hombres'] / dataset_unificado['nacimientos_mujeres']
            ).round(3)
        
        # Ratio de defunciones por sexo
        if 'defunciones_hombres' in dataset_unificado.columns:
            dataset_unificado['ratio_defunciones_sexo'] = (
                dataset_unificado['defunciones_hombres'] / dataset_unificado['defunciones_mujeres']
            ).round(3)
        
        # Crecimiento natural (nacimientos - defunciones)
        dataset_unificado['crecimiento_natural'] = (
            dataset_unificado['nacimientos_totales'] - dataset_unificado['defunciones_totales']
        )
        
        # Porcentaje de crecimiento natural
        dataset_unificado['porcentaje_crecimiento_natural'] = (
            (dataset_unificado['crecimiento_natural'] / dataset_unificado['nacimientos_totales']) * 100
        ).round(2)
        
        # Proporción de nacimientos por sexo
        if 'nacimientos_hombres' in dataset_unificado.columns:
            dataset_unificado['proporcion_nacimientos_hombres'] = (
                dataset_unificado['nacimientos_hombres'] / dataset_unificado['nacimientos_totales']
            ).round(3)
            
            dataset_unificado['proporcion_nacimientos_mujeres'] = (
                dataset_unificado['nacimientos_mujeres'] / dataset_unificado['nacimientos_totales']
            ).round(3)
        
        # Proporción de defunciones por sexo
        if 'defunciones_hombres' in dataset_unificado.columns:
            dataset_unificado['proporcion_defunciones_hombres'] = (
                dataset_unificado['defunciones_hombres'] / dataset_unificado['defunciones_totales']
            ).round(3)
            
            dataset_unificado['proporcion_defunciones_mujeres'] = (
                dataset_unificado['defunciones_mujeres'] / dataset_unificado['defunciones_totales']
            ).round(3)
        
        logger.info("Variables derivadas creadas exitosamente")
    
    # 4. Resumen final
    logger.info("=== RESUMEN DE INTEGRACIÓN ===")
    logger.info(f"Dataset unificado: {dataset_unificado.shape}")
    logger.info(f"Años cubiertos: {sorted(dataset_unificado['año'].unique())}")
    logger.info(f"Rango temporal: {dataset_unificado['año'].min()} - {dataset_unificado['año'].max()}")
    logger.info(f"Columnas totales: {dataset_unificado.shape[1]}")
    logger.info(f"Columnas: {list(dataset_unificado.columns)}")
    
    # Verificar valores nulos
    nulos_por_columna = dataset_unificado.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    if len(columnas_con_nulos) > 0:
        logger.info(f"Columnas con valores nulos: {len(columnas_con_nulos)}")
        logger.info(f"Detalle: {dict(columnas_con_nulos)}")
    else:
        logger.info("No se encontraron valores nulos")
    
    logger.info("Integración de datasets completada exitosamente")
    return dataset_unificado


def crear_features_temporales_avanzadas(
    dataset_unificado: pd.DataFrame,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Crea features temporales avanzadas para análisis de machine learning.
    
    Esta función implementa el feature engineering completo del notebook (sección 8.3):
    1. Features básicos temporales (año normalizado, década, siglo)
    2. Features de tendencia (lineal, cuadrática, cíclica)
    3. Promedios móviles (ventanas de 3 años)
    4. Features de volatilidad (desviación estándar móvil)
    5. Features de cambio año a año
    6. Features de posición relativa (percentiles)
    
    Args:
        dataset_unificado: Dataset unificado con datos agregados por año
        params: Parámetros de configuración para features temporales
        
    Returns:
        Dataset con features temporales avanzadas
    """
    logger.info("Iniciando creación de features temporales avanzadas...")
    logger.info(f"Dataset unificado recibido: {dataset_unificado.shape}")
    
    # Crear copia para trabajar
    dataset_con_features = dataset_unificado.copy()
    
    # Ordenar por año para cálculos temporales
    dataset_con_features = dataset_con_features.sort_values('año').reset_index(drop=True)
    logger.info("Dataset ordenado por año")
    
    # 1. Features básicas temporales
    logger.info("Creando features básicas temporales...")
    
    # Año normalizado (para modelos que requieren escalado)
    dataset_con_features['año_normalizado'] = (
        dataset_con_features['año'] - dataset_con_features['año'].min()
    ) / (dataset_con_features['año'].max() - dataset_con_features['año'].min())
    
    # Década (para análisis de tendencias a largo plazo)
    dataset_con_features['decada'] = (dataset_con_features['año'] // 10) * 10
    
    # Siglo (para análisis de tendencias muy largas)
    dataset_con_features['siglo'] = (dataset_con_features['año'] // 100) * 100
    
    logger.info(" Features básicas temporales creadas")
    
    # 2. Features de tendencia
    logger.info("Creando features de tendencia...")
    
    # Tendencia lineal (años desde el inicio)
    dataset_con_features['tendencia_lineal'] = (
        dataset_con_features['año'] - dataset_con_features['año'].min()
    )
    
    # Tendencia cuadrática (para capturar aceleraciones)
    dataset_con_features['tendencia_cuadratica'] = (
        dataset_con_features['tendencia_lineal'] ** 2
    )
    
    logger.info(" Features de tendencia creadas")
    
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
    
    logger.info(" Features cíclicas creadas")
    
    # 4. Features de cambio y crecimiento
    logger.info("Creando features de cambio año a año...")
    
    # Verificar que las columnas existen antes de calcular diferencias
    if 'nacimientos_totales' in dataset_con_features.columns:
        # Cambio en nacimientos
        dataset_con_features['cambio_nacimientos'] = (
            dataset_con_features['nacimientos_totales'].diff()
        )
        logger.info(" Feature 'cambio_nacimientos' creado")
    
    if 'defunciones_totales' in dataset_con_features.columns:
        # Cambio en defunciones
        dataset_con_features['cambio_defunciones'] = (
            dataset_con_features['defunciones_totales'].diff()
        )
        logger.info(" Feature 'cambio_defunciones' creado")
    
    if 'crecimiento_natural' in dataset_con_features.columns:
        # Cambio en crecimiento poblacional
        dataset_con_features['cambio_crecimiento_poblacional'] = (
            dataset_con_features['crecimiento_natural'].diff()
        )
        logger.info(" Feature 'cambio_crecimiento_poblacional' creado")
    
    # 5. Features de promedio móvil (ventana de 3 años)
    logger.info("Creando promedios móviles...")
    
    if 'nacimientos_totales' in dataset_con_features.columns:
        # Promedio móvil de nacimientos
        dataset_con_features['promedio_movil_nacimientos_3'] = (
            dataset_con_features['nacimientos_totales'].rolling(window=3, min_periods=1).mean()
        )
        logger.info(" Feature 'promedio_movil_nacimientos_3' creado")
    
    if 'defunciones_totales' in dataset_con_features.columns:
        # Promedio móvil de defunciones
        dataset_con_features['promedio_movil_defunciones_3'] = (
            dataset_con_features['defunciones_totales'].rolling(window=3, min_periods=1).mean()
        )
        logger.info(" Feature 'promedio_movil_defunciones_3' creado")
    
    if 'crecimiento_natural' in dataset_con_features.columns:
        # Promedio móvil de crecimiento poblacional
        dataset_con_features['promedio_movil_crecimiento_3'] = (
            dataset_con_features['crecimiento_natural'].rolling(window=3, min_periods=1).mean()
        )
        logger.info(" Feature 'promedio_movil_crecimiento_3' creado")
    
    # 6. Features de volatilidad
    logger.info("Creando features de volatilidad...")
    
    if 'nacimientos_totales' in dataset_con_features.columns:
        # Volatilidad de nacimientos (desviación estándar móvil)
        dataset_con_features['volatilidad_nacimientos_3'] = (
            dataset_con_features['nacimientos_totales'].rolling(window=3, min_periods=1).std()
        )
        logger.info(" Feature 'volatilidad_nacimientos_3' creado")
    
    if 'defunciones_totales' in dataset_con_features.columns:
        # Volatilidad de defunciones
        dataset_con_features['volatilidad_defunciones_3'] = (
            dataset_con_features['defunciones_totales'].rolling(window=3, min_periods=1).std()
        )
        logger.info(" Feature 'volatilidad_defunciones_3' creado")
    
    # 7. Features de posición relativa
    logger.info("Creando features de posición relativa...")
    
    if 'nacimientos_totales' in dataset_con_features.columns:
        # Percentil de nacimientos en el año
        dataset_con_features['percentil_nacimientos'] = (
            dataset_con_features['nacimientos_totales'].rank(pct=True)
        )
        logger.info(" Feature 'percentil_nacimientos' creado")
    
    if 'defunciones_totales' in dataset_con_features.columns:
        # Percentil de defunciones en el año
        dataset_con_features['percentil_defunciones'] = (
            dataset_con_features['defunciones_totales'].rank(pct=True)
        )
        logger.info(" Feature 'percentil_defunciones' creado")
    
    # 8. Resumen de features creadas
    logger.info("=== RESUMEN DE FEATURES TEMPORALES ===")
    
    # Contar diferentes tipos de features
    features_temporales = [
        'año_normalizado', 'decada', 'siglo', 'tendencia_lineal', 'tendencia_cuadratica',
        'ciclo_5_anos', 'ciclo_10_anos', 'cambio_nacimientos', 'cambio_defunciones',
        'cambio_crecimiento_poblacional', 'promedio_movil_nacimientos_3',
        'promedio_movil_defunciones_3', 'promedio_movil_crecimiento_3',
        'volatilidad_nacimientos_3', 'volatilidad_defunciones_3',
        'percentil_nacimientos', 'percentil_defunciones'
    ]
    
    # Contar cuántas se crearon realmente
    features_creadas = [f for f in features_temporales if f in dataset_con_features.columns]
    
    logger.info(f"Features temporales creadas: {len(features_creadas)}/{len(features_temporales)}")
    logger.info(f"Dataset final: {dataset_con_features.shape}")
    
    # Verificar valores nulos en features nuevas
    if features_creadas:
        nulos_features = dataset_con_features[features_creadas].isnull().sum().sum()
        logger.info(f"Valores nulos en features temporales: {nulos_features}")
    
    logger.info("Creación de features temporales completada exitosamente")
    return dataset_con_features


def normalizar_datos_para_modelado(dataset_con_features: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Normaliza los datos para prepararlos para modelos de machine learning.
    
    Esta función aplica diferentes tipos de normalización según el tipo de variable
    y el algoritmo de ML que se usará posteriormente. Implementa tres métodos:
    
    1. StandardScaler (media=0, desviación=1):
       - Para variables con distribución aproximadamente normal
       - Ideal para: Regresión Lineal, SVM, Regresión Logística
       - Ejemplo: edad_promedio, tasas, proporciones
    
    2. MinMaxScaler (escala [0,1]):
       - Para variables que deben estar en un rango específico
       - Ideal para: Redes Neuronales, K-means, algoritmos de distancia
       - Ejemplo: promedios móviles, volatilidad
    
    3. RobustScaler (resistente a outliers):
       - Para variables con outliers significativos
       - Usa mediana y rango intercuartílico en lugar de media
       - Ideal para: Datos demográficos con valores extremos
       - Ejemplo: totales de nacimientos/defunciones, cambios año a año
    
    Nota: Las variables categóricas codificadas (0, 1, 2...) no se normalizan,
    ya que representan categorías, no magnitudes.
    
    Args:
        dataset_con_features: Dataset con features temporales avanzadas (del nodo anterior)
        params: Parámetros de configuración (métodos, variables, etc.)
        
    Returns:
        Diccionario con datasets normalizados:
        - 'dataset_modelado_standard': Normalizado con StandardScaler
        - 'dataset_modelado_minmax': Normalizado con MinMaxScaler
        - 'dataset_modelado_robust': Normalizado con RobustScaler
        - 'dataset_final_modelado': Dataset principal para modelado
        - 'info_normalizacion': Metadatos sobre la normalización
    
    Raises:
        ValueError: Si no hay variables numéricas para normalizar
    
    Example:
        >>> datasets_norm = normalizar_datos_para_modelado(dataset_features, params)
        >>> # Para regresión lineal, usar:
        >>> dataset_std = datasets_norm['dataset_modelado_standard']
        >>> # Para redes neuronales, usar:
        >>> dataset_minmax = datasets_norm['dataset_modelado_minmax']
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
    
    Esta función toma los datasets normalizados y crea versiones especializadas
    para diferentes algoritmos y casos de uso de machine learning. Es el último
    paso antes del modelado.
    
    Datasets creados:
    
    1. dataset_regresion:
       - Selecciona features relevantes para predicción de valores continuos
       - Incluye: features temporales cíclicos, features especiales, año normalizado
       - Uso: Predecir cantidades (nacimientos, defunciones, tasas)
       - Algoritmos sugeridos: Linear Regression, Random Forest Regressor, XGBoost
    
    2. dataset_temporal:
       - Dataset ordenado cronológicamente con índice temporal
       - Incluye columna 'indice_temporal' para series de tiempo
       - Uso: Forecasting, análisis de tendencias, ARIMA, LSTM
       - Importante: Mantiene orden temporal estricto
    
    3. dataset_indexado:
       - Dataset con identificador único compuesto: año_mes_región_sexo
       - Facilita tracking de predicciones individuales
       - Uso: Producción, auditoría, validación cruzada personalizada
    
    4. dataset_resumido:
       - Agregado por año y mes (reduce dimensionalidad)
       - Promedios de features para análisis de alto nivel
       - Uso: Dashboards, reportes ejecutivos, análisis exploratorio
    
    5. dataset_completo:
       - Dataset sin modificaciones (todas las features)
       - Uso: Experimentación, feature selection, análisis ad-hoc
    
    Cada dataset está diseñado para un propósito específico, optimizando
    el rendimiento y la interpretabilidad de los modelos.
    
    Args:
        datasets_normalizados: Diccionario con datasets normalizados del nodo anterior
                               (debe contener 'dataset_final_modelado')
        params: Parámetros de configuración con variables objetivo y predictoras
        
    Returns:
        Diccionario con 5 datasets especializados:
        - 'dataset_regresion': Para modelos de regresión
        - 'dataset_temporal': Para análisis de series de tiempo
        - 'dataset_indexado': Con IDs únicos para tracking
        - 'dataset_resumido': Agregado por período
        - 'dataset_completo': Dataset sin filtrar
    
    Example:
        >>> datasets_finales = crear_datasets_finales_para_modelado(datasets_norm, params)
        >>> # Para modelo de regresión:
        >>> X = datasets_finales['dataset_regresion']
        >>> # Para forecasting:
        >>> ts_data = datasets_finales['dataset_temporal']
        >>> # Para producción con tracking:
        >>> prod_data = datasets_finales['dataset_indexado']
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


def codificar_variables_categoricas(dataset_con_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Codifica variables categóricas usando diferentes estrategias según el tipo de variable.
    
    Esta función implementa tres estrategias de codificación según la naturaleza
    de cada variable categórica:
    
    1. Label Encoding (para variables ORDINALES):
       - Variables con orden natural: década (1970 < 1980 < 1990)
       - Asigna números secuenciales: 0, 1, 2, 3...
       - Preserva el orden inherente de las categorías
       - Ejemplos: década, siglo, categorías de crecimiento
    
    2. One-Hot Encoding (para variables NOMINALES):
       - Variables sin orden natural: ciclo_5_anos, ciclo_10_anos
       - Crea columnas binarias (0/1) para cada categoría
       - Evita que el modelo asuma orden donde no existe
       - Aumenta dimensionalidad pero mejora precisión
    
    3. Codificación Personalizada:
       - Categorías derivadas de bins numéricos
       - Ejemplo: 'crecimiento_poblacional' → 'decrecimiento_alto', 'crecimiento_bajo', etc.
    
    La función también crea variables binarias útiles para clasificación:
    - crecimiento_positivo (1 si crecimiento > 0)
    - mas_nacimientos_que_defunciones
    - alta_volatilidad_nacimientos
    
    Args:
        dataset_con_features: Dataset con features temporales y variables a codificar
        
    Returns:
        Tuple con:
        - DataFrame codificado con nuevas columnas de codificación
        - Diccionario de mapeos con los encoders y transformaciones aplicadas
          (útil para interpretar resultados y aplicar a datos nuevos)
    
    Note:
        Los mapeos retornados se deben guardar para aplicar las mismas
        transformaciones a datos de producción o test.
    
    Example:
        >>> dataset_cod, mapeos = codificar_variables_categoricas(dataset_features)
        >>> # Para interpretar un valor codificado:
        >>> mapeos['decada']['mapeo']  # {1970: 0, 1980: 1, 1990: 2, ...}
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
    Escala las características numéricas usando diferentes métodos según el tipo de variable.
    
    Esta función aplica escalado inteligente, seleccionando el método de normalización
    más apropiado para cada tipo de variable numérica. Esto es crítico porque:
    
    Por qué es importante escalar:
    - Algoritmos basados en distancia (KNN, K-means) son muy sensibles a la escala
    - Gradiente descendente converge más rápido con features escaladas
    - Evita que variables con rangos grandes dominen el modelo
    - Facilita interpretación de coeficientes en modelos lineales
    - Mejora estabilidad numérica en cálculos matriciales
    
    Estrategia de escalado por tipo de variable:
    
    1. StandardScaler (para variables con distribución normal):
       Variables: edad_promedio, edad_mediana, tasas, proporciones, tendencias
       Razón: Centrado en 0 ayuda a modelos lineales y SVM
       Fórmula: z = (x - μ) / σ
    
    2. MinMaxScaler (para variables de rango específico):
       Variables: promedios móviles, volatilidad, percentiles
       Razón: Escala [0,1] ideal para redes neuronales y sigmoide
       Fórmula: x_scaled = (x - min) / (max - min)
    
    3. RobustScaler (para variables con outliers):
       Variables: totales, cambios año a año, crecimiento
       Razón: Usa mediana e IQR, resistente a valores extremos
       Fórmula: x_scaled = (x - median) / IQR
    
    Variables NO escaladas (se mantienen originales):
    - Variables categóricas codificadas (año, década, códigos)
    - Variables binarias (0/1)
    - Variables ya normalizadas (percentiles, proporciones)
    - Columnas one-hot encoded
    
    La función crea versiones con sufijos para preservar originales:
    - columna_standard (escalada con StandardScaler)
    - columna_minmax (escalada con MinMaxScaler)  
    - columna_robust (escalada con RobustScaler)
    - columna_original (valor sin escalar)
    
    Args:
        dataset_codificado: Dataset con variables categóricas ya codificadas
        mapeos_codificacion: Diccionario con mapeos (para identificar columnas one-hot)
        
    Returns:
        Tuple con:
        - DataFrame escalado con columnas adicionales (originales + escaladas)
        - Diccionario de scalers con:
          * 'standard': {'scaler': StandardScaler, 'variables': [...]}
          * 'minmax': {'scaler': MinMaxScaler, 'variables': [...]}
          * 'robust': {'scaler': RobustScaler, 'variables': [...]}
    
    Warning:
        Los scalers retornados se deben guardar (pickle) para aplicar
        las mismas transformaciones a datos de producción.
    
    Example:
        >>> dataset_esc, scalers = escalar_caracteristicas(dataset_cod, mapeos)
        >>> # Para datos nuevos en producción:
        >>> X_new_scaled = scalers['standard']['scaler'].transform(X_new)
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
    
    Esta función es el paso final del pipeline de preparación de datos. Organiza
    los datos en formatos listos para entrenar modelos, identificando claramente:
    
    1. Variables Objetivo (Targets):
       a) Para REGRESIÓN (predecir valores continuos):
          - total_nacimientos_año: Predecir cantidad de nacimientos
          - total_defunciones_año: Predecir cantidad de defunciones
          - crecimiento_poblacional: Predecir cambio poblacional
          - tasa_natalidad: Predecir tasa de nacimientos
          - edad_promedio_defunciones: Predecir edad promedio
       
       b) Para CLASIFICACIÓN (predecir categorías):
          - crecimiento_positivo: Clasificar si hay crecimiento (binario)
          - mas_nacimientos_que_defunciones: Clasificar balance demográfico
          - alta_volatilidad_nacimientos: Detectar períodos volátiles
          - categoria_crecimiento_codificada: Clasificar tipo de crecimiento
          - categoria_tasa_natalidad_codificada: Clasificar nivel de natalidad
    
    2. Features (Variables Predictoras):
       - Identifica automáticamente todas las variables que NO son targets
       - Excluye identificadores y variables auxiliares
    
    3. División Temporal (80/20):
       - Entrenamiento: 80% primeros años
       - Prueba: 20% últimos años
       - Respeta el orden temporal (no aleatorio)
    
    4. Matrices X e Y:
       - Crea matrices separadas para cada variable objetivo
       - Formato listo para sklearn: X_train, X_test, y_train, y_test
    
    La función retorna un diccionario completo con:
    - Datasets de entrenamiento y prueba
    - Matrices X e Y para cada target
    - Scalers y mapeos de codificación
    - Lista de features y targets
    
    Args:
        dataset_escalado: Dataset con todas las características escaladas
        scalers: Diccionario con scalers de normalización (para aplicar a datos nuevos)
        mapeos_codificacion: Diccionario con mapeos de variables categóricas
        
    Returns:
        Diccionario exhaustivo con:
        - 'features': Lista de nombres de features
        - 'targets_regresion': Dict de targets para regresión
        - 'targets_clasificacion': Dict de targets para clasificación
        - 'dataset_completo': Dataset completo ordenado por año
        - 'dataset_entrenamiento': Dataset de entrenamiento (80%)
        - 'dataset_prueba': Dataset de prueba (20%)
        - 'X_{target_name}': Features para cada target
        - 'y_{target_name}': Target específico
        - 'X_train_{target_name}': Features de entrenamiento
        - 'y_train_{target_name}': Target de entrenamiento
        - 'X_test_{target_name}': Features de prueba
        - 'y_test_{target_name}': Target de prueba
        - 'scalers': Scalers usados
        - 'mapeos_codificacion': Mapeos de codificación
    
    Note:
        La división temporal (no aleatoria) es crítica para datos de series
        de tiempo. Esto evita "data leakage" donde el modelo vería el futuro.
    
    Example:
        >>> datos = preparar_datos_modelado(dataset_escalado, scalers, mapeos)
        >>> # Para entrenar modelo de regresión:
        >>> X_train = datos['X_train_total_nacimientos_año']
        >>> y_train = datos['y_train_total_nacimientos_año']
        >>> # Para entrenar modelo de clasificación:
        >>> X_train = datos['X_train_crecimiento_positivo']
        >>> y_train = datos['y_train_crecimiento_positivo']
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
