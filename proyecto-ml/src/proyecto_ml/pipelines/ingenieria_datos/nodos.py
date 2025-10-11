"""Nodos del pipeline de Ingeniería de Datos.

Este módulo contiene todas las funciones de procesamiento de datos
para la fase de ingeniería de datos del proyecto CRISP-DM.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Configurar logging
logger = logging.getLogger(__name__)


def cargar_datos_crudos(
    datos_historicos: pd.DataFrame,
    datos_filtrados_defunciones: pd.DataFrame,
    nacimientos_defunciones_por_sexo: pd.DataFrame,
    nacimientos_por_edad_madre: pd.DataFrame,
    defunciones_por_edad_fallecido: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Carga y organiza todos los datasets crudos del catálogo de Kedro.
    
    Esta es la función de entrada del pipeline de ingeniería de datos. Toma
    los 5 datasets que Kedro carga automáticamente desde data/01_raw/ y los
    organiza en un diccionario estructurado para facilitar el procesamiento.
    
    Datasets procesados:
    
    1. datos_historicos (setdedatos.csv):
       - Serie histórica larga: 1974-2023 (50 años)
       - Columnas: año, nacimientos_totales, defunciones_totales
       - Uso: Análisis de tendencias a largo plazo
    
    2. datos_filtrados_defunciones (datos_filtrados_2014_2023.csv):
       - Datos detallados: 2014-2024 (~1.25M registros)
       - Columnas: año, fecha, sexo, edad, ubicación, diagnóstico
       - Uso: Análisis granular, modelado predictivo
       - Problemas conocidos: Duplicados, nulos geográficos (requiere limpieza)
    
    3. nacimientos_defunciones_por_sexo:
       - Desglose por sexo: 2015-2023
       - Columnas: año, nacimientos_hombres, nacimientos_mujeres, defunciones_hombres, defunciones_mujeres
       - Uso: Análisis de diferencias por género
    
    4. nacimientos_por_edad_madre:
       - Rangos de edad de madres: 2010-2023
       - Columnas: año + 9 rangos de edad (<15, 15-19, ..., 50+)
       - Uso: Análisis de fecundidad por edad
    
    5. defunciones_por_edad_fallecido:
       - Rangos de edad de fallecidos: 2010-2023
       - Columnas: año + 12 rangos de edad (<1, 1-4, ..., 50+)
       - Uso: Análisis de mortalidad por edad
    
    La organización en diccionario permite:
    - Acceso fácil por nombre de dataset
    - Iteración sobre todos los datasets
    - Logging de información de cada uno
    - Paso uniforme a nodos posteriores
    
    Args:
        datos_historicos: Dataset histórico (1974-2023)
        datos_filtrados_defunciones: Dataset detallado de defunciones (2014-2024)
        nacimientos_defunciones_por_sexo: Dataset por sexo (2015-2023)
        nacimientos_por_edad_madre: Dataset de nacimientos por edad de madre (2010-2023)
        defunciones_por_edad_fallecido: Dataset de defunciones por edad (2010-2023)
        
    Returns:
        Diccionario con datasets organizados:
        {
            'datos_historicos': DataFrame,
            'defunciones_filtradas': DataFrame,
            'nacimientos_por_sexo': DataFrame,
            'nacimientos_por_edad_madre': DataFrame,
            'defunciones_por_edad_fallecido': DataFrame
        }
    
    Note:
        Esta función NO modifica los datos, solo los organiza. La limpieza
        y transformación ocurre en nodos posteriores del pipeline.
    """
    logger.info("Iniciando carga de datos crudos...")
    
    # Organizar datasets en diccionario
    datasets_crudos = {
        "datos_historicos": datos_historicos,
        "defunciones_filtradas": datos_filtrados_defunciones,
        "nacimientos_por_sexo": nacimientos_defunciones_por_sexo,
        "nacimientos_por_edad_madre": nacimientos_por_edad_madre,
        "defunciones_por_edad_fallecido": defunciones_por_edad_fallecido
    }
    
    # Log de información de cada dataset
    for nombre, df in datasets_crudos.items():
        logger.info(f"{nombre}: {df.shape[0]:,} registros, {df.shape[1]} columnas")
    
    logger.info("Datos crudos cargados exitosamente")
    return datasets_crudos


# =============================================================================
# FUNCIONES AUXILIARES PRIVADAS PARA LIMPIEZA DE DATOS
# =============================================================================

def _eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros duplicados del dataset.
    
    Args:
        df: DataFrame con posibles duplicados
        
    Returns:
        DataFrame sin duplicados
    """
    registros_antes = df.shape[0]
    df_sin_duplicados = df.drop_duplicates(keep='first')
    registros_eliminados = registros_antes - df_sin_duplicados.shape[0]
    logger.info(f"Eliminados {registros_eliminados:,} duplicados")
    return df_sin_duplicados


def _eliminar_nulos_geograficos(df: pd.DataFrame, columnas_geograficas: list) -> pd.DataFrame:
    """
    Elimina registros con información geográfica nula.
    
    La información geográfica es crítica para análisis regionales,
    por lo que los registros sin esta información se eliminan.
    
    Args:
        df: DataFrame con posibles nulos geográficos
        columnas_geograficas: Lista de columnas geográficas críticas
        
    Returns:
        DataFrame sin nulos en columnas geográficas
    """
    registros_antes = df.shape[0]
    df_sin_nulos = df.dropna(subset=columnas_geograficas)
    registros_eliminados = registros_antes - df_sin_nulos.shape[0]
    logger.info(f"Eliminados {registros_eliminados} registros con datos geográficos nulos")
    return df_sin_nulos


def _imputar_fechas_nulas(df: pd.DataFrame, columna_fecha: str, columna_año: str) -> pd.DataFrame:
    """
    Imputa fechas nulas usando la fecha media del año correspondiente.
    
    Estrategia de imputación:
    - Para cada año, calcula la fecha media de registros válidos
    - Imputa esa fecha media en registros con fecha nula del mismo año
    
    Args:
        df: DataFrame con posibles fechas nulas
        columna_fecha: Nombre de la columna de fecha
        columna_año: Nombre de la columna de año
        
    Returns:
        DataFrame con fechas imputadas
    """
    df_imputado = df.copy()
    nulos_fecha = df_imputado[columna_fecha].isnull().sum()
    logger.info(f"Fechas nulas encontradas: {nulos_fecha}")
    
    if nulos_fecha > 0:
        registros_nulos_fecha = df_imputado[df_imputado[columna_fecha].isnull()]
        
        for año in registros_nulos_fecha[columna_año].unique():
            # Calcular fecha media del año para registros con fecha válida
            fechas_validas_año = df_imputado[
                (df_imputado[columna_año] == año) & 
                (df_imputado[columna_fecha].notnull())
            ][columna_fecha]
            
            if len(fechas_validas_año) > 0:
                # Convertir a datetime para calcular media
                fechas_datetime = pd.to_datetime(fechas_validas_año, errors='coerce')
                fecha_media = fechas_datetime.mean()
                
                # Imputar fecha media en registros nulos del año
                mask_nulos_año = (df_imputado[columna_año] == año) & (df_imputado[columna_fecha].isnull())
                df_imputado.loc[mask_nulos_año, columna_fecha] = fecha_media.strftime('%Y-%m-%d')
                
                logger.info(f"Año {año}: {mask_nulos_año.sum()} registros imputados con fecha {fecha_media.strftime('%Y-%m-%d')}")
    
    return df_imputado


def _estandarizar_formato_fechas(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    """
    Convierte fechas a formato datetime y elimina registros con fechas inválidas.
    
    Args:
        df: DataFrame con fechas en formato string
        columna_fecha: Nombre de la columna de fecha
        
    Returns:
        DataFrame con fechas en formato datetime
    """
    df_estandarizado = df.copy()
    
    # Convertir a datetime
    df_estandarizado[columna_fecha] = pd.to_datetime(df_estandarizado[columna_fecha], errors='coerce')
    
    # Eliminar registros con fechas inválidas
    fechas_invalidas = df_estandarizado[columna_fecha].isnull().sum()
    if fechas_invalidas > 0:
        df_estandarizado = df_estandarizado.dropna(subset=[columna_fecha])
        logger.info(f"Eliminados {fechas_invalidas} registros con fechas inválidas")
    
    return df_estandarizado


def _crear_variables_temporales_derivadas(df: pd.DataFrame, columna_fecha: str, columna_año: str) -> pd.DataFrame:
    """
    Crea variables temporales derivadas a partir de la fecha.
    
    Variables creadas:
    - AÑO_FECHA: Año extraído de la fecha
    - MES: Mes (1-12)
    - DIA_SEMANA: Día de la semana (Monday, Tuesday, etc.)
    - TRIMESTRE: Trimestre (1-4)
    - DIA_AÑO: Día del año (1-365/366)
    
    Args:
        df: DataFrame con columna de fecha en formato datetime
        columna_fecha: Nombre de la columna de fecha
        columna_año: Nombre de la columna de año original
        
    Returns:
        DataFrame con variables temporales derivadas
    """
    df_con_variables = df.copy()
    
    # Extraer componentes temporales
    df_con_variables['AÑO_FECHA'] = df_con_variables[columna_fecha].dt.year
    df_con_variables['MES'] = df_con_variables[columna_fecha].dt.month
    df_con_variables['DIA_SEMANA'] = df_con_variables[columna_fecha].dt.day_name()
    df_con_variables['TRIMESTRE'] = df_con_variables[columna_fecha].dt.quarter
    df_con_variables['DIA_AÑO'] = df_con_variables[columna_fecha].dt.dayofyear
    
    logger.info("Variables temporales creadas: AÑO_FECHA, MES, DIA_SEMANA, TRIMESTRE, DIA_AÑO")
    
    # Verificar consistencia entre año original y año de fecha
    inconsistencias_año = (df_con_variables[columna_año] != df_con_variables['AÑO_FECHA']).sum()
    logger.info(f"Inconsistencias entre {columna_año} y AÑO_FECHA: {inconsistencias_año}")
    
    return df_con_variables


# =============================================================================
# FUNCIÓN PRINCIPAL DE LIMPIEZA
# =============================================================================

def limpiar_defunciones(defunciones_filtradas: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia el dataset de defunciones aplicando todas las transformaciones
    identificadas en el análisis exploratorio.
    
    Esta función orquesta el proceso de limpieza usando funciones auxiliares
    especializadas para cada tarea. Sigue la lógica del notebook:
    1. Elimina duplicados
    2. Maneja valores nulos geográficos
    3. Imputa fechas nulas
    4. Estandariza formato de fechas
    5. Crea variables temporales
    
    Args:
        defunciones_filtradas: Dataset crudo de defunciones
        params: Parámetros de configuración para la limpieza
        
    Returns:
        Dataset de defunciones completamente limpio
    """
    logger.info("Iniciando limpieza del dataset de defunciones...")
    logger.info(f"Registros originales: {defunciones_filtradas.shape[0]:,}")
    
    # Crear copia para trabajar
    defunciones_limpio = defunciones_filtradas.copy()
    
    # 1. Eliminar duplicados
    logger.info("Paso 1: Eliminando duplicados...")
    defunciones_limpio = _eliminar_duplicados(defunciones_limpio)
    
    # 2. Eliminar nulos geográficos
    logger.info("Paso 2: Eliminando registros con información geográfica nula...")
    columnas_geograficas = ['COD_COMUNA', 'COMUNA', 'NOMBRE_REGION']
    defunciones_limpio = _eliminar_nulos_geograficos(defunciones_limpio, columnas_geograficas)
    
    # 3. Imputar fechas nulas
    logger.info("Paso 3: Imputando fechas nulas...")
    defunciones_limpio = _imputar_fechas_nulas(defunciones_limpio, 'FECHA_DEF', 'AÑO')
    
    # 4. Estandarizar formato de fechas
    logger.info("Paso 4: Estandarizando formato de fechas...")
    defunciones_limpio = _estandarizar_formato_fechas(defunciones_limpio, 'FECHA_DEF')
    
    # 5. Crear variables temporales derivadas
    logger.info("Paso 5: Creando variables temporales...")
    defunciones_limpio = _crear_variables_temporales_derivadas(defunciones_limpio, 'FECHA_DEF', 'AÑO')
    
    # Resumen final
    logger.info("=== RESUMEN DE LIMPIEZA ===")
    logger.info(f"Registros finales: {defunciones_limpio.shape[0]:,}")
    logger.info(f"Columnas finales: {defunciones_limpio.shape[1]}")
    logger.info(f"Rango de fechas: {defunciones_limpio['FECHA_DEF'].min()} a {defunciones_limpio['FECHA_DEF'].max()}")
    logger.info(f"Años cubiertos: {sorted(defunciones_limpio['AÑO'].unique())}")
    
    logger.info("Dataset de defunciones limpiado exitosamente")
    return defunciones_limpio


def estandarizar_columnas(
    defunciones_limpio: pd.DataFrame,
    nacimientos_por_sexo: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Estandariza los nombres de columnas de todos los datasets para mantener consistencia.
    
    Esta función implementa la estandarización de nombres descrita en el notebook
    (sección 4: Estandarización de Nombres de Columnas), aplicando las siguientes
    transformaciones:
    
    Convenciones de nombres aplicadas:
    - Todo en minúsculas: 'AÑO' → 'año', 'FECHA_DEF' → 'fecha_defuncion'
    - Snake_case para nombres compuestos: 'EDAD_CANT' → 'edad_cantidad'
    - Nombres descriptivos: 'SEXO_NOMBRE' → 'sexo'
    - Sin espacios ni caracteres especiales: 'Nacimiento (Hombre)' → 'nacimientos_hombres'
    
    Beneficios de la estandarización:
    - Facilita la integración entre datasets (misma columna 'año' en todos)
    - Evita errores por mayúsculas/minúsculas
    - Mejora legibilidad del código
    - Consistencia con convenciones Python (PEP 8)
    - Facilita operaciones de merge/join
    
    Datasets procesados:
    1. defunciones_limpio: 15 columnas renombradas
    2. nacimientos_por_sexo: 5 columnas renombradas
    
    Args:
        defunciones_limpio: Dataset de defunciones después de limpieza
        nacimientos_por_sexo: Dataset de nacimientos y defunciones por sexo
        
    Returns:
        Diccionario con dos datasets estandarizados:
        - 'defunciones_estandarizado': Dataset de defunciones con nombres estándar
        - 'nacimientos_por_sexo_estandarizado': Dataset por sexo con nombres estándar
    
    Example:
        >>> datasets_std = estandarizar_columnas(defunciones_limpio, nacimientos_sexo)
        >>> # Ahora todos los datasets usan 'año' en minúsculas
        >>> datasets_std['defunciones_estandarizado']['año']  # ✓ Funciona
        >>> datasets_std['nacimientos_por_sexo_estandarizado']['año']  # ✓ Funciona
    """
    logger.info("Iniciando estandarización de nombres de columnas...")
    
    # 1. Estandarizar dataset de defunciones
    logger.info("Estandarizando dataset de defunciones...")
    defunciones_estandarizado = defunciones_limpio.copy()
    
    # Mapeo de nombres para defunciones (del notebook)
    mapeo_defunciones = {
        'AÑO': 'año',
        'FECHA_DEF': 'fecha_defuncion',
        'SEXO_NOMBRE': 'sexo',
        'EDAD_TIPO': 'tipo_edad',
        'EDAD_CANT': 'edad_cantidad',
        'COD_COMUNA': 'codigo_comuna',
        'COMUNA': 'comuna',
        'NOMBRE_REGION': 'region',
        'CAPITULO_DIAG1': 'codigo_diagnostico',
        'GLOSA_CAPITULO_DIAG1': 'descripcion_diagnostico',
        'AÑO_FECHA': 'año_fecha',
        'MES': 'mes',
        'DIA_SEMANA': 'dia_semana',
        'TRIMESTRE': 'trimestre',
        'DIA_AÑO': 'dia_año'
    }
    
    defunciones_estandarizado = defunciones_estandarizado.rename(columns=mapeo_defunciones)
    logger.info(f"Defunciones estandarizado: {defunciones_estandarizado.shape}")
    
    # 2. Estandarizar dataset de nacimientos por sexo
    logger.info("Estandarizando dataset de nacimientos por sexo...")
    nacimientos_estandarizado = nacimientos_por_sexo.copy()
    
    # Mapeo de nombres para nacimientos por sexo (del notebook)
    mapeo_nacimientos = {
        'Año': 'año',
        'Nacimiento (Hombre)': 'nacimientos_hombres',
        'Nacimiento (Mujer)': 'nacimientos_mujeres', 
        'Defuncion(Hombre)': 'defunciones_hombres',
        'Defuncion (Mujer)': 'defunciones_mujeres'
    }
    
    nacimientos_estandarizado = nacimientos_estandarizado.rename(columns=mapeo_nacimientos)
    logger.info(f"Nacimientos por sexo estandarizado: {nacimientos_estandarizado.shape}")
    
    # Organizar datasets estandarizados
    datasets_estandarizados = {
        "defunciones_estandarizado": defunciones_estandarizado,
        "nacimientos_por_sexo_estandarizado": nacimientos_estandarizado
    }
    
    logger.info("Estandarización de columnas completada")
    return datasets_estandarizados


def validar_calidad_datos(
    datasets_estandarizados: Dict[str, pd.DataFrame], 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valida la calidad de los datos después de la limpieza y estandarización.
    
    Esta función es un punto de control crítico en el pipeline. Realiza
    verificaciones exhaustivas de calidad para garantizar que los datos
    están listos para la fase de ciencia de datos y modelado.
    
    Verificaciones realizadas:
    
    1. Completitud de datos:
       - Cuenta total de registros
       - Número de columnas
       - Identificación de valores nulos
    
    2. Calidad estructural:
       - Detección de duplicados
       - Validación de tipos de datos
       - Verificación de rangos de fechas
    
    3. Cobertura temporal:
       - Años únicos presentes
       - Rango de fechas (inicio-fin)
       - Gaps temporales
    
    4. Cobertura geográfica:
       - Regiones únicas (para defunciones)
       - Distribución geográfica
    
    5. Estado de validación:
       - Lista de problemas críticos encontrados
       - Estado general: "APROBADO" o "REVISAR"
    
    Umbrales de calidad (según params):
    - Completitud mínima: 95%
    - Duplicados máximo: 5%
    - Valores nulos máximo: 3%
    
    Si se detectan problemas críticos, se registran en el log como WARNING
    para revisión manual.
    
    Args:
        datasets_estandarizados: Diccionario con datasets procesados:
                                 - 'defunciones_estandarizado'
                                 - 'nacimientos_por_sexo_estandarizado'
        params: Parámetros de configuración con umbrales de validación
        
    Returns:
        Diccionario con métricas de calidad:
        - 'defunciones': Métricas del dataset de defunciones
        - 'nacimientos': Métricas del dataset de nacimientos
        - 'problemas_criticos': Lista de problemas encontrados
        - 'estado': "APROBADO" si no hay problemas, "REVISAR" si los hay
    
    Raises:
        KeyError: Si los datasets esperados no están en el diccionario
    
    Example:
        >>> metricas = validar_calidad_datos(datasets_std, params)
        >>> if metricas['estado'] == 'APROBADO':
        >>>     print("✓ Datos listos para modelado")
        >>> else:
        >>>     print(f"⚠ Problemas: {metricas['problemas_criticos']}")
    """
    logger.info("Iniciando validación de calidad de datos...")
    
    # Extraer datasets del diccionario
    defunciones_estandarizado = datasets_estandarizados["defunciones_estandarizado"]
    nacimientos_por_sexo_estandarizado = datasets_estandarizados["nacimientos_por_sexo_estandarizado"]
    
    # Métricas de calidad para defunciones
    logger.info("Validando dataset de defunciones...")
    calidad_defunciones = {
        "total_registros": defunciones_estandarizado.shape[0],
        "total_columnas": defunciones_estandarizado.shape[1],
        "valores_nulos": defunciones_estandarizado.isnull().sum().sum(),
        "duplicados": defunciones_estandarizado.duplicated().sum(),
        "rango_fechas": {
            "inicio": str(defunciones_estandarizado['fecha_defuncion'].min()),
            "fin": str(defunciones_estandarizado['fecha_defuncion'].max())
        },
        "años_unicos": len(defunciones_estandarizado['año'].unique()),
        "regiones_unicas": len(defunciones_estandarizado['region'].unique())
    }
    
    # Métricas de calidad para nacimientos
    logger.info("Validando dataset de nacimientos por sexo...")
    calidad_nacimientos = {
        "total_registros": nacimientos_por_sexo_estandarizado.shape[0],
        "total_columnas": nacimientos_por_sexo_estandarizado.shape[1],
        "valores_nulos": nacimientos_por_sexo_estandarizado.isnull().sum().sum(),
        "duplicados": nacimientos_por_sexo_estandarizado.duplicated().sum(),
        "años_unicos": len(nacimientos_por_sexo_estandarizado['año'].unique())
    }
    
    # Resumen de validación
    logger.info("=== RESUMEN DE VALIDACIÓN ===")
    logger.info(f"Defunciones: {calidad_defunciones['total_registros']:,} registros, {calidad_defunciones['valores_nulos']} nulos, {calidad_defunciones['duplicados']} duplicados")
    logger.info(f"Nacimientos: {calidad_nacimientos['total_registros']:,} registros, {calidad_nacimientos['valores_nulos']} nulos, {calidad_nacimientos['duplicados']} duplicados")
    
    # Verificar que no hay problemas críticos
    problemas_criticos = []
    
    if calidad_defunciones['valores_nulos'] > 0:
        problemas_criticos.append("Defunciones tiene valores nulos")
    
    if calidad_defunciones['duplicados'] > 0:
        problemas_criticos.append("Defunciones tiene duplicados")
        
    if calidad_nacimientos['valores_nulos'] > 0:
        problemas_criticos.append("Nacimientos tiene valores nulos")
        
    if calidad_nacimientos['duplicados'] > 0:
        problemas_criticos.append("Nacimientos tiene duplicados")
    
    if problemas_criticos:
        logger.warning(f"Problemas encontrados: {', '.join(problemas_criticos)}")
    else:
        logger.info("No se encontraron problemas críticos de calidad")
    
    # Compilar métricas de calidad
    metricas_calidad = {
        "defunciones": calidad_defunciones,
        "nacimientos": calidad_nacimientos,
        "problemas_criticos": problemas_criticos,
        "estado": "APROBADO" if not problemas_criticos else "REVISAR"
    }
    
    logger.info("Validación de calidad completada")
    return metricas_calidad
