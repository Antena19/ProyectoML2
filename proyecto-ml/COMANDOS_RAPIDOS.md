# ⚡ COMANDOS RÁPIDOS - EVALUACIÓN 2

## 🚀 EJECUCIÓN DE PIPELINES

```bash
# 1. Activar entorno virtual
.\.venv\Scripts\Activate

# 2. Ejecutar pipeline de clasificación (6 modelos)
kedro run --pipeline=clasificacion

# 3. Ejecutar pipeline de regresión (6 modelos)
kedro run --pipeline=regresion

# 4. Ejecutar ambos pipelines de ML
kedro run --pipeline=modelado_completo

# 5. Ejecutar TODO el proyecto (ingeniería + ciencia + ML + reportes)
kedro run
```

## 📊 VISUALIZACIÓN

```bash
# Abrir Kedro Viz (ver estructura de pipelines)
kedro viz

# Abrir Jupyter Notebooks
kedro jupyter notebook
# → Navegar a notebooks/modelado/
# → Abrir 6_Clasificacion_Modelos.ipynb
# → Abrir 7_Regresion_Modelos.ipynb
```

## 🧪 TESTS

```bash
# Ejecutar todos los tests
pytest

# Tests de clasificación
pytest tests/pipelines/test_clasificacion.py -v

# Tests de regresión
pytest tests/pipelines/test_regresion.py -v

# Con cobertura
pytest --cov=src/proyecto_ml
```

## 📁 VER RESULTADOS

```bash
# Listar modelos entrenados
dir data\06_models\clasificacion
dir data\06_models\regresion

# Ver tablas comparativas (CSV)
type data\07_model_output\tabla_comparativa_clasificacion.csv
type data\07_model_output\tabla_comparativa_regresion.csv
```

## 🐍 DESDE PYTHON

```python
# Cargar resultados de clasificación
from kedro.framework.session import KedroSession

with KedroSession.create() as session:
    catalog = session.load_context().catalog
    resultados = catalog.load("resultados_clasificacion")
    tabla = catalog.load("tabla_comparativa_clasificacion")
    
print(tabla)
```

## 🔍 DEBUGGING

```bash
# Ver logs
type info.log

# Ejecutar con más verbosidad
kedro run --pipeline=clasificacion --env=local

# Ver solo un nodo específico
kedro run --node=entrenar_modelos_clasificacion
```

## 📋 INFORMACIÓN DEL PROYECTO

```bash
# Listar todos los pipelines
kedro registry list

# Listar datasets en el catálogo
kedro catalog list

# Ver información del proyecto
kedro info
```

## 🛠️ DESARROLLO

```bash
# Formatear código
ruff format src/

# Linting
ruff check src/

# Ejecutar en modo interactivo
kedro ipython
```

## 📦 INSTALACIÓN (si cambias de máquina)

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
.\.venv\Scripts\Activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
kedro --version
```

## 💾 EXPORTAR RESULTADOS

```bash
# Copiar tablas comparativas al directorio actual
copy data\07_model_output\tabla_comparativa_clasificacion.csv .
copy data\07_model_output\tabla_comparativa_regresion.csv .

# Comprimir modelos para compartir
tar -czf modelos_clasificacion.tar.gz data/06_models/clasificacion/
tar -czf modelos_regresion.tar.gz data/06_models/regresion/
```

## 🎯 FLUJO DE TRABAJO TÍPICO

```bash
# 1. Activar entorno
.\.venv\Scripts\Activate

# 2. Ejecutar pipelines ML
kedro run --pipeline=modelado_completo

# 3. Abrir notebooks para visualizar
kedro jupyter notebook

# 4. Ejecutar tests
pytest -v

# 5. Verificar resultados
dir data\07_model_output
```

## 🚨 SOLUCIÓN DE PROBLEMAS

```bash
# Si falla por falta de datos:
kedro run --pipeline=ingenieria_datos
kedro run --pipeline=ciencia_datos
kedro run --pipeline=modelado_completo

# Si falla un modelo específico:
# Revisar logs en info.log
type info.log | Select-String -Pattern "ERROR"

# Limpiar caché
kedro clean

# Reinstalar dependencias
pip install -r requirements.txt --upgrade
```

## 📊 DURANTE LA DEFENSA

```bash
# 1. Mostrar pipelines visualmente
kedro viz

# 2. Mostrar tabla comparativa
python -c "import pandas as pd; df = pd.read_csv('data/07_model_output/tabla_comparativa_clasificacion.csv'); print(df)"

# 3. Abrir notebook
kedro jupyter notebook

# 4. Ejecutar pipeline en vivo (si te lo piden)
kedro run --pipeline=clasificacion
```

---

**💡 TIP:** Practica estos comandos antes de la defensa para que los tengas en músculo!

