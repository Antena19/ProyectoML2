import os
import time
from datetime import datetime

def verificar_progreso_pipeline():
    """Verifica el progreso del pipeline de clasificación"""
    
    print("VERIFICADOR DE PROGRESO - PIPELINE DE CLASIFICACION")
    print("=" * 60)
    
    # Archivos que se generan durante el pipeline
    archivos_progreso = {
        'dataset_individual_ml.csv': 'Dataset preparado para ML',
        'datos_clasificacion': 'Datos preparados para clasificación (en memoria)',
        'resultados_clasificacion.pkl': 'Resultados de modelos entrenados',
        'tabla_comparativa_clasificacion.csv': 'Tabla comparativa generada',
        'rutas_modelos_clasificacion': 'Rutas de modelos guardados (en memoria)'
    }
    
    # Directorios a verificar
    directorios = [
        'data/04_feature/',
        'data/07_model_output/',
        'data/06_models/'
    ]
    
    print("ARCHIVOS DE PROGRESO:")
    print("-" * 40)
    
    for archivo, descripcion in archivos_progreso.items():
        encontrado = False
        ubicacion = ""
        
        for directorio in directorios:
            ruta_completa = os.path.join(directorio, archivo)
            if os.path.exists(ruta_completa):
                encontrado = True
                ubicacion = directorio
                break
        
        if encontrado:
            tamaño = os.path.getsize(os.path.join(ubicacion, archivo))
            modificado = datetime.fromtimestamp(os.path.getmtime(os.path.join(ubicacion, archivo)))
            print(f"✓ {archivo:30s} | {descripcion}")
            print(f"  Ubicación: {ubicacion}")
            print(f"  Tamaño: {tamaño:,} bytes")
            print(f"  Modificado: {modificado.strftime('%H:%M:%S')}")
        else:
            print(f"✗ {archivo:30s} | {descripcion}")
        print()
    
    # Verificar logs de Kedro
    print("LOGS DE KEDRO:")
    print("-" * 40)
    
    log_files = ['info.log', 'kedro.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            tamaño = os.path.getsize(log_file)
            modificado = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"✓ {log_file:15s} | Tamaño: {tamaño:,} bytes | Modificado: {modificado.strftime('%H:%M:%S')}")
            
            # Mostrar últimas líneas del log
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lineas = f.readlines()
                    if lineas:
                        print(f"  Última línea: {lineas[-1].strip()}")
            except:
                print(f"  No se pudo leer el archivo de log")
        else:
            print(f"✗ {log_file:15s} | No encontrado")
        print()
    
    # Verificar procesos de Python
    print("PROCESOS DE PYTHON ACTIVOS:")
    print("-" * 40)
    
    try:
        import psutil
        procesos_python = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'kedro' in cmdline.lower():
                        procesos_python += 1
                        print(f"✓ PID {proc.pid:6d} | {cmdline[:80]}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if procesos_python == 0:
            print("✗ No hay procesos de Kedro ejecutándose")
        else:
            print(f"Total de procesos Kedro: {procesos_python}")
            
    except ImportError:
        print("✗ psutil no disponible - no se puede verificar procesos")
    
    print()
    print("ESTADO DEL PROGRESO:")
    print("-" * 40)
    
    # Determinar estado basado en archivos
    if os.path.exists('data/04_feature/dataset_individual_ml.csv'):
        if os.path.exists('data/07_model_output/resultados_clasificacion.pkl'):
            print("✓ PIPELINE COMPLETADO - Todos los modelos entrenados")
        elif os.path.exists('data/07_model_output/tabla_comparativa_clasificacion.csv'):
            print("✓ MODELOS ENTRENADOS - Generando tabla comparativa")
        else:
            print("🔄 ENTRENANDO MODELOS - En progreso...")
    else:
        print("✗ PIPELINE NO INICIADO - Ejecuta el pipeline primero")

if __name__ == "__main__":
    verificar_progreso_pipeline()
