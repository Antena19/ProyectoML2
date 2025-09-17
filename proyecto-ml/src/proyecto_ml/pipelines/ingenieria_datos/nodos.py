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
    Carga todos los datasets crudos del catálogo de Kedro.
    
    Esta función toma los datasets ya cargados por Kedro y los organiza
    en un diccionario para facilitar el procesamiento posterior.
    
    Args:
        datos_historicos: Dataset histórico de nacimientos y defunciones
        datos_filtrados_defunciones: Dataset de defunciones filtradas (2014-2023)
        nacimientos_defunciones_por_sexo: Dataset por sexo
        nacimientos_por_edad_madre: Dataset de nacimientos por edad de madre
        defunciones_por_edad_fallecido: Dataset de defunciones por edad
        
    Returns:
        Dict con todos los datasets organizados por nombre
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


def limpiar_defunciones(defunciones_filtradas: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia el dataset de defunciones aplicando todas las transformaciones
    identificadas en el análisis exploratorio.
    
    Esta función replica exactamente la lógica de limpieza del notebook:
    1. Elimina duplicados
    2. Maneja valores nulos geográficos
    3. Imputa fechas nulas
    4. Estandariza formato de fechas
    5. Crea variables temporales
    
    Args:
        defunciones_filtradas: Dataset crudo de defunciones
        
    Returns:
        Dataset de defunciones completamente limpio
    """
    logger.info("Iniciando limpieza del dataset de defunciones...")
    logger.info(f"Registros originales: {defunciones_filtradas.shape[0]:,}")
    
    # Crear copia para trabajar
    defunciones_limpio = defunciones_filtradas.copy()
    
    # 1. Eliminar duplicados manteniendo la primera ocurrencia
    logger.info("Eliminando duplicados...")
    registros_antes = defunciones_limpio.shape[0]
    defunciones_limpio = defunciones_limpio.drop_duplicates(keep='first')
    registros_eliminados = registros_antes - defunciones_limpio.shape[0]
    logger.info(f"Eliminados {registros_eliminados:,} duplicados")
    
    # 2. Manejar valores nulos en información geográfica crítica
    logger.info("Eliminando registros con información geográfica nula...")
    columnas_geograficas = ['COD_COMUNA', 'COMUNA', 'NOMBRE_REGION']
    registros_antes_geo = defunciones_limpio.shape[0]
    defunciones_limpio = defunciones_limpio.dropna(subset=columnas_geograficas)
    registros_eliminados_geo = registros_antes_geo - defunciones_limpio.shape[0]
    logger.info(f"Eliminados {registros_eliminados_geo} registros con datos geográficos nulos")
    
    # 3. Manejar valores nulos en FECHA_DEF mediante imputación
    logger.info("Imputando fechas nulas...")
    nulos_fecha = defunciones_limpio['FECHA_DEF'].isnull().sum()
    logger.info(f"Fechas nulas encontradas: {nulos_fecha}")
    
    if nulos_fecha > 0:
        # Imputar con fecha media del año correspondiente
        registros_nulos_fecha = defunciones_limpio[defunciones_limpio['FECHA_DEF'].isnull()]
        
        for año in registros_nulos_fecha['AÑO'].unique():
            # Calcular fecha media del año para registros con fecha válida
            fechas_validas_año = defunciones_limpio[
                (defunciones_limpio['AÑO'] == año) & 
                (defunciones_limpio['FECHA_DEF'].notnull())
            ]['FECHA_DEF']
            
            if len(fechas_validas_año) > 0:
                # Convertir a datetime para calcular media
                fechas_datetime = pd.to_datetime(fechas_validas_año, errors='coerce')
                fecha_media = fechas_datetime.mean()
                
                # Imputar fecha media en registros nulos del año
                mask_nulos_año = (defunciones_limpio['AÑO'] == año) & (defunciones_limpio['FECHA_DEF'].isnull())
                defunciones_limpio.loc[mask_nulos_año, 'FECHA_DEF'] = fecha_media.strftime('%Y-%m-%d')
                
                logger.info(f"Año {año}: {mask_nulos_año.sum()} registros imputados con fecha {fecha_media.strftime('%Y-%m-%d')}")
    
    # 4. Estandarizar formato de fechas
    logger.info("Estandarizando formato de fechas...")
    defunciones_limpio['FECHA_DEF'] = pd.to_datetime(defunciones_limpio['FECHA_DEF'], errors='coerce')
    
    # Eliminar registros con fechas inválidas (si los hay)
    fechas_invalidas = defunciones_limpio['FECHA_DEF'].isnull().sum()
    if fechas_invalidas > 0:
        defunciones_limpio = defunciones_limpio.dropna(subset=['FECHA_DEF'])
        logger.info(f"Eliminados {fechas_invalidas} registros con fechas inválidas")
    
    # 5. Crear variables temporales derivadas
    logger.info("Creando variables temporales...")
    defunciones_limpio['AÑO_FECHA'] = defunciones_limpio['FECHA_DEF'].dt.year
    defunciones_limpio['MES'] = defunciones_limpio['FECHA_DEF'].dt.month
    defunciones_limpio['DIA_SEMANA'] = defunciones_limpio['FECHA_DEF'].dt.day_name()
    defunciones_limpio['TRIMESTRE'] = defunciones_limpio['FECHA_DEF'].dt.quarter
    defunciones_limpio['DIA_AÑO'] = defunciones_limpio['FECHA_DEF'].dt.dayofyear
    
    logger.info("Variables temporales creadas: AÑO_FECHA, MES, DIA_SEMANA, TRIMESTRE, DIA_AÑO")
    
    # Verificar consistencia entre AÑO y AÑO_FECHA
    inconsistencias_año = (defunciones_limpio['AÑO'] != defunciones_limpio['AÑO_FECHA']).sum()
    logger.info(f"Inconsistencias entre AÑO y AÑO_FECHA: {inconsistencias_año}")
    
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
    Estandariza los nombres de columnas de todos los datasets
    para mantener consistencia en el proyecto.
    
    Esta función aplica el mapeo de nombres definido en el notebook
    para unificar la nomenclatura de columnas.
    
    Args:
        defunciones_limpio: Dataset de defunciones limpio
        nacimientos_por_sexo: Dataset de nacimientos por sexo
        
    Returns:
        Dict con datasets con columnas estandarizadas
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
    
    Esta función realiza verificaciones de calidad para asegurar que
    los datos están listos para la siguiente fase del pipeline.
    
    Args:
        datasets_estandarizados: Diccionario con datasets estandarizados
        
    Returns:
        Dict con métricas de calidad de datos
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
