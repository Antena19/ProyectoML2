"""
Nodos del pipeline de reportes para generar visualizaciones y métricas de calidad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def generar_reporte_calidad_datos(
    metricas_calidad_datos: Dict[str, Any], 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Genera reporte de calidad de datos basado en las métricas calculadas.
    
    Args:
        metricas_calidad_datos: Diccionario con métricas de calidad
        
    Returns:
        Diccionario con reporte de calidad estructurado
    """
    logger.info("Generando reporte de calidad de datos")
    
    reporte = {
        "resumen_ejecutivo": {},
        "metricas_detalladas": {},
        "recomendaciones": [],
        "estado_general": "BUENO"
    }
    
    # Analizar métricas de defunciones
    if "defunciones" in metricas_calidad_datos:
        def_metrics = metricas_calidad_datos["defunciones"]
        
        reporte["resumen_ejecutivo"]["defunciones"] = {
            "total_registros": def_metrics.get("total_registros", 0),
            "completitud": def_metrics.get("completitud", 0),
            "duplicados": def_metrics.get("duplicados", 0),
            "outliers": def_metrics.get("outliers", 0)
        }
        
        # Evaluar calidad
        if def_metrics.get("completitud", 0) < 0.8:
            reporte["recomendaciones"].append("Defunciones: Completitud baja, revisar valores faltantes")
            reporte["estado_general"] = "REQUIERE_ATENCION"
    
    # Analizar métricas de nacimientos
    if "nacimientos_por_sexo" in metricas_calidad_datos:
        nac_metrics = metricas_calidad_datos["nacimientos_por_sexo"]
        
        reporte["resumen_ejecutivo"]["nacimientos"] = {
            "total_registros": nac_metrics.get("total_registros", 0),
            "completitud": nac_metrics.get("completitud", 0),
            "duplicados": nac_metrics.get("duplicados", 0),
            "outliers": nac_metrics.get("outliers", 0)
        }
    
    # Generar recomendaciones generales
    if not reporte["recomendaciones"]:
        reporte["recomendaciones"].append("Calidad de datos aceptable para análisis")
    
    reporte["metricas_detalladas"] = metricas_calidad_datos
    
    logger.info(f"Reporte de calidad generado - Estado: {reporte['estado_general']}")
    return reporte

def generar_visualizaciones_calidad(
    metricas_calidad_datos: Dict[str, Any]
) -> Dict[str, str]:
    """
    Genera visualizaciones de calidad de datos.
    
    Args:
        metricas_calidad_datos: Diccionario con métricas de calidad
        
    Returns:
        Diccionario con rutas de archivos de visualizaciones generadas
    """
    logger.info("Generando visualizaciones de calidad de datos")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    visualizaciones = {}
    
    # Crear directorio para reportes
    reportes_dir = Path("data/08_reporting")
    reportes_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfico de completitud por dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = []
    completitud = []
    
    for dataset_name, metrics in metricas_calidad_datos.items():
        if isinstance(metrics, dict) and "valores_nulos" in metrics and "total_registros" in metrics:
            # Calcular completitud como (1 - valores_nulos / total_registros)
            total_registros = metrics["total_registros"]
            valores_nulos = metrics["valores_nulos"]
            completitud_calculada = 1 - (valores_nulos / (total_registros * metrics.get("total_columnas", 1)))
            
            datasets.append(dataset_name.replace("_", " ").title())
            completitud.append(completitud_calculada)
    
    if datasets:
        bars = ax.bar(datasets, completitud, color=['#2E8B57', '#4169E1', '#DC143C'])
        ax.set_title('Completitud de Datos por Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Completitud (%)', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, completitud):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_completitud = reportes_dir / "completitud_datos.png"
        plt.savefig(ruta_completitud, dpi=300, bbox_inches='tight')
        plt.close()
        visualizaciones["completitud"] = str(ruta_completitud)
    
    # 2. Gráfico de duplicados por dataset
    if datasets:  # Solo crear si hay datasets
        fig, ax = plt.subplots(figsize=(10, 6))
        
        duplicados = []
        for dataset_name, metrics in metricas_calidad_datos.items():
            if isinstance(metrics, dict) and "duplicados" in metrics:
                duplicados.append(metrics["duplicados"])
        
        if duplicados and len(duplicados) == len(datasets):
            bars = ax.bar(datasets, duplicados, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Registros Duplicados por Dataset', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Duplicados', fontsize=12)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, duplicados):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(value)}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Guardar gráfico
            ruta_duplicados = reportes_dir / "duplicados_datos.png"
            plt.savefig(ruta_duplicados, dpi=300, bbox_inches='tight')
            plt.close()
            visualizaciones["duplicados"] = str(ruta_duplicados)
    
    # 3. Gráfico de valores nulos por dataset
    if datasets:  # Solo crear si hay datasets
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valores_nulos = []
        for dataset_name, metrics in metricas_calidad_datos.items():
            if isinstance(metrics, dict) and "valores_nulos" in metrics:
                valores_nulos.append(metrics["valores_nulos"])
        
        if valores_nulos and len(valores_nulos) == len(datasets):
            bars = ax.bar(datasets, valores_nulos, color=['#FF9F43', '#10AC84', '#EE5A24'])
            ax.set_title('Valores Nulos por Dataset', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Valores Nulos', fontsize=12)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, valores_nulos):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(value)}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Guardar gráfico
            ruta_nulos = reportes_dir / "valores_nulos_datos.png"
            plt.savefig(ruta_nulos, dpi=300, bbox_inches='tight')
            plt.close()
            visualizaciones["valores_nulos"] = str(ruta_nulos)
    
    logger.info(f"Visualizaciones generadas: {len(visualizaciones)} archivos")
    return visualizaciones

def generar_reporte_features_temporales(
    dataset_con_features_temporales: pd.DataFrame
) -> Dict[str, Any]:
    """
    Genera reporte de features temporales creados.
    
    Args:
        dataset_con_features_temporales: Dataset con features temporales
        
    Returns:
        Diccionario con reporte de features temporales
    """
    logger.info("Generando reporte de features temporales")
    
    reporte = {
        "resumen_features": {},
        "analisis_ciclicos": {},
        "distribucion_temporal": {},
        "recomendaciones": []
    }
    
    # Analizar features cíclicos
    features_ciclicos = [col for col in dataset_con_features_temporales.columns 
                        if '_sin' in col or '_cos' in col]
    
    reporte["resumen_features"] = {
        "total_features": len(dataset_con_features_temporales.columns),
        "features_ciclicos": len(features_ciclicos),
        "features_especiales": len([col for col in dataset_con_features_temporales.columns 
                                   if col.startswith('es_')]),
        "total_registros": len(dataset_con_features_temporales)
    }
    
    # Analizar distribución temporal
    if 'año' in dataset_con_features_temporales.columns:
        años_unicos = dataset_con_features_temporales['año'].nunique()
        reporte["distribucion_temporal"] = {
            "años_cubiertos": años_unicos,
            "rango_años": f"{dataset_con_features_temporales['año'].min()}-{dataset_con_features_temporales['año'].max()}"
        }
    
    # Generar recomendaciones
    if len(features_ciclicos) >= 4:
        reporte["recomendaciones"].append("Features cíclicos completos para modelado temporal")
    else:
        reporte["recomendaciones"].append("Considerar agregar más features cíclicos")
    
    logger.info(f"Reporte de features temporales generado - {len(features_ciclicos)} features cíclicos")
    return reporte

def generar_visualizaciones_features(
    dataset_con_features_temporales: pd.DataFrame
) -> Dict[str, str]:
    """
    Genera visualizaciones de features temporales.
    
    Args:
        dataset_con_features_temporales: Dataset con features temporales
        
    Returns:
        Diccionario con rutas de archivos de visualizaciones generadas
    """
    logger.info("Generando visualizaciones de features temporales")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    visualizaciones = {}
    
    # Crear directorio para reportes
    reportes_dir = Path("data/08_reporting")
    reportes_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfico de features cíclicos
    features_ciclicos = [col for col in dataset_con_features_temporales.columns 
                        if '_sin' in col or '_cos' in col]
    
    if len(features_ciclicos) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Features Cíclicos Generados', fontsize=16, fontweight='bold')
        
        # Mapear features cíclicos
        ciclicos_map = {
            'mes': ['mes_sin', 'mes_cos'],
            'dia_año': ['dia_año_sin', 'dia_año_cos'],
            'trimestre': ['trimestre_sin', 'trimestre_cos'],
            'dia_semana': ['dia_semana_sin', 'dia_semana_cos']
        }
        
        positions = [(0,0), (0,1), (1,0), (1,1)]
        titles = ['Mes Cíclico', 'Día del Año Cíclico', 'Trimestre Cíclico', 'Día de Semana Cíclico']
        
        for i, (key, cols) in enumerate(ciclicos_map.items()):
            if cols[0] in dataset_con_features_temporales.columns and cols[1] in dataset_con_features_temporales.columns:
                row, col = positions[i]
                axes[row, col].scatter(dataset_con_features_temporales[cols[0]], 
                                     dataset_con_features_temporales[cols[1]], 
                                     alpha=0.1, s=1)
                axes[row, col].set_title(titles[i], fontweight='bold')
                axes[row, col].set_xlabel(cols[0])
                axes[row, col].set_ylabel(cols[1])
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_ciclicos = reportes_dir / "features_ciclicos.png"
        plt.savefig(ruta_ciclicos, dpi=300, bbox_inches='tight')
        plt.close()
        visualizaciones["features_ciclicos"] = str(ruta_ciclicos)
    
    # 2. Gráfico de distribución temporal
    if 'año' in dataset_con_features_temporales.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Contar registros por año
        conteo_anual = dataset_con_features_temporales['año'].value_counts().sort_index()
        
        bars = ax.bar(conteo_anual.index, conteo_anual.values, color='#2E8B57')
        ax.set_title('Distribución de Registros por Año', fontsize=14, fontweight='bold')
        ax.set_xlabel('Año', fontsize=12)
        ax.set_ylabel('Número de Registros', fontsize=12)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, conteo_anual.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_temporal = reportes_dir / "distribucion_temporal.png"
        plt.savefig(ruta_temporal, dpi=300, bbox_inches='tight')
        plt.close()
        visualizaciones["distribucion_temporal"] = str(ruta_temporal)
    
    logger.info(f"Visualizaciones de features generadas: {len(visualizaciones)} archivos")
    return visualizaciones

def generar_reporte_final(
    reporte_calidad_datos: Dict[str, Any],
    reporte_features_temporales: Dict[str, Any],
    visualizaciones_calidad: Dict[str, str],
    visualizaciones_features: Dict[str, str]
) -> Dict[str, Any]:
    """
    Genera reporte final consolidado del proyecto.
    
    Args:
        reporte_calidad_datos: Reporte de calidad de datos
        reporte_features_temporales: Reporte de features temporales
        visualizaciones_calidad: Visualizaciones de calidad
        visualizaciones_features: Visualizaciones de features
        
    Returns:
        Diccionario con reporte final consolidado
    """
    logger.info("Generando reporte final consolidado")
    
    reporte_final = {
        "informacion_general": {
            "fecha_generacion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "metodologia": "CRISP-DM - Fase 3: Preparación de Datos"
        },
        "resumen_ejecutivo": {
            "estado_calidad": reporte_calidad_datos.get("estado_general", "DESCONOCIDO"),
            "features_generados": reporte_features_temporales.get("resumen_features", {}).get("total_features", 0),
            "visualizaciones_generadas": len(visualizaciones_calidad) + len(visualizaciones_features)
        },
        "reportes_detallados": {
            "calidad_datos": reporte_calidad_datos,
            "features_temporales": reporte_features_temporales
        },
        "archivos_generados": {
            "visualizaciones_calidad": visualizaciones_calidad,
            "visualizaciones_features": visualizaciones_features
        },
        "recomendaciones_generales": []
    }
    
    # Consolidar recomendaciones
    recomendaciones = []
    
    if "recomendaciones" in reporte_calidad_datos:
        recomendaciones.extend(reporte_calidad_datos["recomendaciones"])
    
    if "recomendaciones" in reporte_features_temporales:
        recomendaciones.extend(reporte_features_temporales["recomendaciones"])
    
    reporte_final["recomendaciones_generales"] = recomendaciones
    
    # Agregar recomendación final
    if reporte_calidad_datos.get("estado_general") == "BUENO":
        reporte_final["recomendaciones_generales"].append("Datos listos para modelado de Machine Learning")
    else:
        reporte_final["recomendaciones_generales"].append("Revisar calidad de datos antes del modelado")
    
    logger.info("Reporte final consolidado generado exitosamente")
    return reporte_final
