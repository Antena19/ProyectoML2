import time
import psutil
import os
from datetime import datetime, timedelta

def monitorear_progreso():
    """Monitorea el progreso del entrenamiento de modelos"""
    
    print("MONITOR DE PROGRESO - ENTRENAMIENTO DE MODELOS")
    print("=" * 60)
    print("Presiona Ctrl+C para detener el monitoreo")
    print()
    
    inicio = datetime.now()
    proceso_python = None
    
    try:
        while True:
            # Buscar proceso de Python que ejecuta Kedro
            procesos_python = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'kedro' in cmdline.lower() and 'run' in cmdline.lower():
                            procesos_python.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Mostrar información del sistema
            tiempo_transcurrido = datetime.now() - inicio
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Tiempo: {str(tiempo_transcurrido).split('.')[0]} | "
                  f"CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {memory.percent:5.1f}% | "
                  f"Procesos Kedro: {len(procesos_python)}", end="")
            
            # Si hay procesos de Kedro, mostrar más detalles
            if procesos_python:
                print()
                for i, proc in enumerate(procesos_python):
                    try:
                        cpu = proc.cpu_percent()
                        memory_mb = proc.memory_info().rss / 1024 / 1024
                        print(f"  Proceso {i+1}: PID {proc.pid} | CPU {cpu:5.1f}% | RAM {memory_mb:6.1f}MB")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            time.sleep(5)  # Actualizar cada 5 segundos
            
    except KeyboardInterrupt:
        print("\n\nMonitoreo detenido por el usuario")
        print(f"Tiempo total de monitoreo: {str(datetime.now() - inicio).split('.')[0]}")

if __name__ == "__main__":
    monitorear_progreso()
