# 🎯 PROYECTO MACHINE LEARNING - ANÁLISIS DE DEFUNCIONES

## 📋 Información General

**Proyecto**: Evaluación Parcial 2 - Machine Learning Pipelines
**Asignatura**: Machine Learning
**Modalidad**: Parejas
**Duración**: 4 semanas
**Ponderación**: 40%

---

## 🎯 Objetivo del Proyecto

Este proyecto implementa pipelines de Machine Learning para análisis de datos de defunciones en Chile, utilizando el framework Kedro para la gestión de pipelines de datos.

### **Problema Principal:**
**Predecir el SEXO de una persona fallecida** basándose únicamente en características de su muerte (edad, fecha, región).

### **Hipótesis Epidemiológica:**
- **HOMBRES** tienden a morir más jóvenes que mujeres
- **MUJERES** tienden a morir más mayores que hombres
- Existen **patrones temporales y regionales** distintos por género

### **Objetivos de Rendimiento:**
- **Accuracy > 80%**: Predicción correcta en 8 de cada 10 casos
- **F1-Score > 75%**: Buen balance entre precisión y recall
- **ROC-AUC > 0.8**: Excelente capacidad discriminativa

---

## 🚀 Instalación y Configuración

# Navegar a la raiz del proyecto
cd proyecto-ml

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (PowerShell)
.\.venv\Scripts\Activate

# *Desactivar
deactivate

# Instalar dependencias desde requirements.txt
uv pip install -r requirements.txt

# Abrir sesión interactiva de Kedro
kedro ipython

# 1 Importar Kedro y crear la sesión del proyecto
from kedro.framework.session import KedroSession

# Crear sesión del proyecto y cargar el catálogo
session = KedroSession.create()
catalog = session.load_context().catalog

# 2 Cargar los datasets crudos (01_raw)
datos_historicos = catalog.load("datos_historicos_nacimientos_defunciones")
datos_filtrados_defunciones = catalog.load("datos_filtrados_defunciones")
nacimientos_por_sexo = catalog.load("nacimientos_defunciones_por_sexo")
nacimientos_por_edad_madre = catalog.load("nacimientos_por_edad_madre")
defunciones_por_edad_fallecido = catalog.load("defunciones_por_edad_fallecido")

# 3 Mostrar un vistazo rápido de los CSV principales
print("=== Datos Históricos de Nacimientos y Defunciones ===")
print(datos_historicos.head())

print("\n=== Datos Filtrados de Defunciones (2014-2023) ===")
print(datos_filtrados_defunciones.head())

print("\n=== Nacimientos y Defunciones por Sexo ===")
print(nacimientos_por_sexo.head())

print("\n=== Nacimientos por Edad de la Madre ===")
print(nacimientos_por_edad_madre.head())

print("\n=== Defunciones por Edad del Fallecido ===")
print(defunciones_por_edad_fallecido.head())

# Trabajar en Jupyter Notebook
pip install notebook

# Abrir Notebook
kedro jupyter notebook

# 1. Ejecutar pipeline en orden
kedro run --pipeline=ingenieria_datos

kedro run --pipeline=ciencia_datos

kedro run --pipeline=reportes

### Visualizar Pipelines

# Abrir Kedro Viz en el navegador
kedro viz

# Ejecutar todos los tests
pytest

# Ejecutar tests específicos
pytest tests/pipelines/test_ingenieria_datos.py
pytest tests/pipelines/test_ciencia_datos.py
pytest tests/pipelines/test_reportes.py

# Ejecutar con verbose
pytest -v

# Ejecutar con cobertura
pytest --cov=src/proyecto_ml