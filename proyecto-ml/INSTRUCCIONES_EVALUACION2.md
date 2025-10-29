# 📋 INSTRUCCIONES - EVALUACIÓN 2 MACHINE LEARNING

## ✅ LO QUE SE HA IMPLEMENTADO

### 1. **Pipelines de Machine Learning** ✓

#### **Pipeline de Clasificación** (6 modelos)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

#### **Pipeline de Regresión** (6 modelos)
- Linear Regression (Ridge)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- KNN Regressor
- Decision Tree Regressor

### 2. **GridSearchCV + CrossValidation** ✓
- Búsqueda exhaustiva de hiperparámetros
- CrossValidation con **k=5 folds** (mínimo requerido)
- Métricas con formato **mean±std**

### 3. **Métricas Implementadas** ✓

**Clasificación:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matrices de confusión

**Regresión:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### 4. **Estructura de Archivos Creada** ✓

```
src/proyecto_ml/pipelines/
├── clasificacion/
│   ├── __init__.py
│   ├── nodos.py          # 6 modelos de clasificación
│   └── pipeline.py       # Pipeline Kedro
├── regresion/
│   ├── __init__.py
│   ├── nodos.py          # 6 modelos de regresión
│   └── pipeline.py       # Pipeline Kedro
notebooks/modelado/
├── 6_Clasificacion_Modelos.ipynb
└── 7_Regresion_Modelos.ipynb
tests/pipelines/
├── test_clasificacion.py
└── test_regresion.py
```

### 5. **Configuración Actualizada** ✓
- `conf/base/catalog.yml`: Datasets de ML agregados
- `conf/base/parameters.yml`: Configuración de modelos
- `src/proyecto_ml/pipeline_registry.py`: Pipelines registrados

---

## 🚀 CÓMO EJECUTAR LOS PIPELINES

### Opción 1: Ejecutar desde línea de comandos

```bash
# Activar entorno virtual
.\.venv\Scripts\Activate  # Windows PowerShell

# Ejecutar pipeline de clasificación
kedro run --pipeline=clasificacion

# Ejecutar pipeline de regresión
kedro run --pipeline=regresion

# Ejecutar ambos pipelines de ML
kedro run --pipeline=modelado_completo

# Ejecutar todo el proyecto completo
kedro run
```

### Opción 2: Ejecutar desde notebooks

1. Abrir Jupyter:
```bash
kedro jupyter notebook
```

2. Navegar a `notebooks/modelado/`

3. Ejecutar los notebooks:
   - `6_Clasificacion_Modelos.ipynb`
   - `7_Regresion_Modelos.ipynb`

### Opción 3: Ejecutar desde Python

```python
from kedro.framework.session import KedroSession

# Ejecutar pipeline de clasificación
with KedroSession.create() as session:
    session.run(pipeline_name="clasificacion")

# Ejecutar pipeline de regresión
with KedroSession.create() as session:
    session.run(pipeline_name="regresion")
```

---

## 📊 RESULTADOS GENERADOS

### Archivos de Salida

Los pipelines generan automáticamente:

1. **Modelos entrenados** → `data/06_models/`
   - `clasificacion_sexo_logistic_regression.pkl`
   - `clasificacion_sexo_random_forest.pkl`
   - `regresion_edad_cantidad_linear_regression.pkl`
   - etc. (12+ modelos)

2. **Métricas y resultados** → `data/07_model_output/`
   - `resultados_clasificacion.pkl`
   - `resultados_regresion.pkl`
   - `tabla_comparativa_clasificacion.csv` ⭐ **IMPORTANTE PARA EVALUACIÓN**
   - `tabla_comparativa_regresion.csv` ⭐ **IMPORTANTE PARA EVALUACIÓN**

3. **Visualizaciones** (generadas en notebooks)
   - `comparacion_accuracy_clasificacion.png`
   - `comparacion_r2_regresion.png`
   - `matrices_confusion_*.png`
   - `cv_boxplot_*.png`

---

## 🧪 EJECUTAR TESTS

```bash
# Ejecutar todos los tests
pytest

# Ejecutar tests específicos
pytest tests/pipelines/test_clasificacion.py
pytest tests/pipelines/test_regresion.py

# Con verbose
pytest -v

# Con cobertura
pytest --cov=src/proyecto_ml
```

---

## 📑 CHECKLIST EVALUACIÓN 2

### ✅ COMPLETADOS

- [x] **Pipelines Kedro modulares y ejecutables** (8%)
  - Pipeline de clasificación ✓
  - Pipeline de regresión ✓
  - Nodos bien documentados ✓

- [x] **Cobertura de modelos + Tuning + CV** (24%)
  - ≥5 modelos de clasificación (6 implementados) ✓
  - ≥5 modelos de regresión (6 implementados) ✓
  - GridSearchCV configurado ✓
  - CrossValidation k=5 ✓
  - Tabla comparativa con mean±std ✓

- [x] **Métricas y visualizaciones** (10%)
  - Métricas apropiadas (accuracy, f1, roc-auc, r², rmse, etc.) ✓
  - Gráficos comparativos ✓
  - Matrices de confusión ✓
  - Boxplots de CV ✓

- [x] **Reproducibilidad (Git)** (Parcial 7%)
  - Código versionado ✓
  - Configuración externa (parameters.yml) ✓
  - Tests unitarios ✓

- [x] **Documentación técnica** (5%)
  - README con instrucciones ✓
  - Código bien comentado ✓
  - Docstrings completos ✓

- [x] **Reporte de experimentos** (Parcial 5%)
  - Notebooks con análisis ✓
  - Tablas comparativas ✓
  - Visualizaciones ✓

### ❌ PENDIENTES (Próximas fases)

- [ ] **DVC** (7%) - Fase 2
  - Crear `dvc.yaml`
  - Versionar datos, features y modelos
  - Configurar remote storage

- [ ] **Airflow** (7%) - Fase 3
  - Crear DAG que ejecute ambos pipelines
  - Configurar Airflow local o en Docker
  - Orquestar ejecución

- [ ] **Docker** (Parcial 7%) - Fase 4
  - Ya existe Dockerfile básico
  - Necesita actualización para incluir Airflow y DVC
  - Probar ejecución completa en contenedor

- [ ] **Defensa técnica** (20%) - Final
  - Preparar presentación (10 min)
  - Explicar flujo Kedro→Airflow→DVC→Docker
  - Practicar respuestas a preguntas

---

## 🎯 PRÓXIMOS PASOS

### Fase 2: Implementar DVC

1. Inicializar DVC:
```bash
dvc init
```

2. Crear `dvc.yaml` con stages

3. Versionar datasets y modelos

### Fase 3: Implementar Airflow

1. Instalar Airflow
2. Crear DAG en `dags/`
3. Configurar ejecución de pipelines

### Fase 4: Integrar todo en Docker

1. Actualizar Dockerfile
2. Agregar docker-compose con Airflow
3. Probar ejecución completa

---

## 📝 NOTAS IMPORTANTES

### Para la Defensa Técnica

**Prepara explicar:**
1. **¿Por qué GridSearchCV?** - Para encontrar mejores hiperparámetros automáticamente
2. **¿Por qué k=5?** - Balance entre sesgo-varianza, estándar en la industria
3. **¿Cómo funciona el flujo?** - Kedro organiza el código, Airflow orquesta, DVC versiona
4. **¿Mejor modelo?** - Consultar tablas comparativas generadas
5. **¿Evidencia de overfitting?** - Comparar métricas train vs test vs CV

### Comandos Útiles

```bash
# Ver pipelines disponibles
kedro registry list

# Visualizar pipeline
kedro viz

# Ver catálogo de datos
kedro catalog list

# Ejecutar nodo específico
kedro run --node=entrenar_modelos_clasificacion
```

---

## 🔗 RECURSOS

- **Kedro Docs**: https://docs.kedro.org
- **Scikit-learn GridSearchCV**: https://scikit-learn.org/stable/modules/grid_search.html
- **DVC**: https://dvc.org/doc
- **Airflow**: https://airflow.apache.org/docs/

---

## ✨ RESUMEN EJECUTIVO

**Tienes implementado:**
- ✅ 12 modelos de ML entrenados (6 clasificación + 6 regresión)
- ✅ GridSearchCV + CrossValidation (k=5)
- ✅ Tablas comparativas con mean±std
- ✅ Pipelines Kedro modulares
- ✅ Notebooks para exploración y defensa
- ✅ Tests unitarios
- ✅ Documentación completa

**Te falta:**
- ❌ DVC (Fase 2)
- ❌ Airflow (Fase 3)
- ❌ Docker actualizado (Fase 4)

**Siguiente paso:** Implementar DVC cuando estés listo.

---

**¡Éxito en tu evaluación!** 🎓🚀

