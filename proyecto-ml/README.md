# Proyecto ML - Análisis de Defunciones y Nacimientos

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CRISP-DM](https://img.shields.io/badge/metodología-CRISP--DM-green.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)

## Descripción del Proyecto

Este proyecto implementa un análisis completo de datos de defunciones y nacimientos en Chile utilizando la metodología **CRISP-DM** y el framework **Kedro**. El objetivo es desarrollar pipelines modulares y reproducibles para el procesamiento de datos demográficos y la preparación de datasets para modelado de Machine Learning.

### Datasets Incluidos

**IMPORTANTE: Debes colocar manualmente los siguientes 5 datasets en la carpeta `data/01_raw/`:**

1. **`dataset_nacimiento-defuncion_por_sexo.csv`** - Estadísticas de nacimientos y defunciones por género
2. **`datos_filtrados_2014_2023.csv`** - Registros de defunciones filtrados (2014-2023)
3. **`nacimiento_rango_edad_madre.csv`** - Distribución de nacimientos según edad materna
4. **`rango_edad_fallecido.csv`** - Registros de defunciones clasificados por edad del fallecido
5. **`setdedatos.csv`** - Dataset histórico de nacimientos y defunciones

**Nota:** Estos archivos estan en el Drive de ML

## Instalación y Configuración

### Opción A: Instalación Tradicional (Recomendada)

#### 1. Clonar el Repositorio

**Opción A: Clonar rama principal (main)**
```bash
git clone https://github.com/Antena19/proyecto-ml.git
cd proyecto-ml
```

**Opción B: Clonar rama específica**
```bash
git clone -b pipelines_nodos https://github.com/Antena19/proyecto-ml.git
cd proyecto-ml
```

#### 2. Crear Ambiente Virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### 4. Verificar Instalación

```bash
kedro info
```

### Opción B: Instalación con Docker (Alternativa)

#### Prerrequisitos
- **Docker Desktop** instalado y ejecutándose
- **Docker Compose** (incluido con Docker Desktop)

#### 1. Clonar el Repositorio
```bash
git clone https://github.com/Antena19/proyecto-ml.git
cd proyecto-ml
```

#### 2. Construir Imagen Docker
```bash
# Windows
scripts\build.bat

# Linux/Mac
./scripts/build.sh

# O directamente
docker build -t proyecto-ml:latest .
```

#### 3. Verificar Imagen
```bash
docker images proyecto-ml:latest
```

#### 4. Ejecutar Tests en Docker
```bash
docker run --rm proyecto-ml:latest python -m pytest -v
```

## Estructura del Proyecto

```
proyecto-ml/
├── conf/                    # Configuración de Kedro
│   ├── base/               # Configuración base
│   │   ├── catalog.yml        # Catálogo de datasets
│   │   └── parameters.yml     # Parámetros del proyecto
│   └── local/              # Configuración local (no versionar)
├── data/                   # Datos del proyecto
│   ├── 01_raw/            # Datos originales
│   ├── 02_intermediate/   # Datos procesados parcialmente
│   ├── 03_primary/       # Datos limpios y estandarizados
│   └── 08_reporting/      # Reportes y visualizaciones
├── notebooks/              # Análisis exploratorio
│   ├── 1_Comprensión_del_negocio.ipynb
│   ├── 2_Comprensión_de_los_Datos.ipynb
│   ├── 3_Preparación_de_Datos,.ipynb
│   ├── ciencia_datos/     # Notebooks de análisis avanzado
│   │   └── 4_Exploracion_Datasets_Finales.ipynb
│   └── reportes/          # Notebooks de reportes
│       └── 5_Exploracion_Reportes.ipynb
├── scripts/                # Scripts de automatización
│   ├── build.sh           # Script de construcción (Linux/Mac)
│   ├── build.bat          # Script de construcción (Windows)
│   ├── run.sh             # Script de ejecución (Linux/Mac)
│   └── run.bat            # Script de ejecución (Windows)
├── src/proyecto_ml/       # Código fuente
│   └── pipelines/         # Pipelines de Kedro
│       ├── ingenieria_datos/  # Pipeline de ingeniería
│       ├── ciencia_datos/     # Pipeline de ciencia de datos
│       └── reportes/          # Pipeline de reportes
├── tests/                 # Tests unitarios
│   └── pipelines/         # Tests de pipelines
├── Dockerfile             # Configuración Docker
├── docker-compose.yml     # Configuración Docker Compose
├── .dockerignore          # Archivos a ignorar en Docker
├── DOCKER_README.md       # Documentación específica de Docker
└── README.md              # Este archivo
```

## Orden de Ejecución Recomendado

### FASE 1: Análisis Exploratorio (Notebooks)

**1. Comprensión del Negocio**
```bash
# Abrir Jupyter con Kedro
kedro jupyter notebook

# Ejecutar notebook:
# notebooks/exploracion/1_Comprensión_del_negocio.ipynb
```

**2. Comprensión de los Datos**
```bash
# Ejecutar notebook:
# notebooks/exploracion/2_Comprensión_de_los_Datos.ipynb
```

**3. Preparación de Datos**
```bash
# Ejecutar notebook:
# notebooks/exploracion/3_Preparación_de_Datos.ipynb
```

### FASE 2: Pipelines de Kedro

**1. Pipeline de Ingeniería de Datos**
```bash
# Ejecutar pipeline de limpieza y estandarización
kedro run --pipeline=ingenieria_datos
```

**2. Pipeline de Ciencia de Datos**
```bash
# Ejecutar pipeline de feature engineering y normalización
kedro run --pipeline=ciencia_datos
```

**3. Pipeline de Reportes**
```bash
# Ejecutar pipeline de generación de reportes
kedro run --pipeline=reportes
```

### FASE 3: Análisis de Resultados

**1. Exploración de Datasets Finales**
```bash
# Ejecutar notebook:
# notebooks/ciencia_datos/4_Exploracion_Datasets_Finales.ipynb
```

**2. Exploración de Reportes**
```bash
# Ejecutar notebook:
# notebooks/reportes/5_Exploracion_Reportes.ipynb
```

## Comandos Principales

### Ejecutar Pipelines

#### Instalación Tradicional
```bash
# Ejecutar todos los pipelines
kedro run

# Ejecutar pipeline específico
kedro run --pipeline=ingenieria_datos
kedro run --pipeline=ciencia_datos
kedro run --pipeline=reportes

# Ejecutar con logs detallados
kedro run --verbose
```

#### Con Docker
```bash
# Ejecutar pipeline completo
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest

# Ejecutar pipeline específico
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ingenieria_datos
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ciencia_datos
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=reportes

# Usar scripts automatizados
# Windows
scripts\run.bat
scripts\run.bat ingenieria
scripts\run.bat ciencia
scripts\run.bat reportes
scripts\run.bat tests

# Linux/Mac
./scripts/run.sh
./scripts/run.sh ingenieria
./scripts/run.sh ciencia
./scripts/run.sh reportes
./scripts/run.sh tests
```

### Visualizar Pipelines

```bash
# Abrir Kedro Viz en el navegador
kedro viz
```

### Trabajar con Notebooks

```bash
# Abrir Jupyter con contexto de Kedro
kedro jupyter notebook

# Abrir JupyterLab
kedro jupyter lab

# Abrir IPython con Kedro
kedro ipython
```

### Explorar Datasets

```bash
# Listar todos los datasets disponibles
kedro catalog list

# Ver información de un dataset específico
kedro catalog describe dataset_name
```

## Pipelines Implementados

### Pipeline de Ingeniería de Datos
- **Carga de datos crudos** desde múltiples fuentes
- **Limpieza de defunciones** (eliminación de duplicados, imputación de fechas)
- **Estandarización de columnas** para consistencia
- **Validación de calidad** de datos

### Pipeline de Ciencia de Datos
- **Feature engineering avanzado** (features cíclicos, temporales)
- **Normalización múltiple** (StandardScaler, MinMaxScaler, RobustScaler)
- **Creación de datasets finales** para modelado
- **Preparación para ML** (regresión, clasificación)

### Pipeline de Reportes
- **Generación de reportes de calidad** automáticos
- **Visualizaciones de datos** (completitud, duplicados, valores nulos)
- **Análisis de features temporales** con gráficos cíclicos
- **Reporte final consolidado** del proyecto

## Testing

### Instalación Tradicional
```bash
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
```

### Con Docker
```bash
# Ejecutar todos los tests en Docker
docker run --rm proyecto-ml:latest python -m pytest -v

# Ejecutar tests específicos
docker run --rm proyecto-ml:latest python -m pytest tests/pipelines/test_ingenieria_datos.py -v
docker run --rm proyecto-ml:latest python -m pytest tests/pipelines/test_ciencia_datos.py -v
docker run --rm proyecto-ml:latest python -m pytest tests/pipelines/test_reportes.py -v

# Usar script automatizado
# Windows
scripts\run.bat tests

# Linux/Mac
./scripts/run.sh tests
```

### Tests Implementados

**Tests de Ingeniería de Datos:**
- `test_limpiar_defunciones_basico()` - Verifica limpieza de datos
- `test_estandarizar_columnas()` - Verifica estandarización de columnas
- `test_validar_calidad_datos()` - Verifica validación de calidad

**Tests de Ciencia de Datos:**
- `test_crear_features_temporales_avanzadas()` - Verifica feature engineering
- `test_normalizar_datos_para_modelado()` - Verifica normalización de datos
- `test_crear_datasets_finales_para_modelado()` - Verifica creación de datasets finales

**Tests de Reportes:**
- `test_generar_reporte_calidad_datos()` - Verifica reportes de calidad
- `test_generar_reporte_features_temporales()` - Verifica reportes de features
- `test_generar_visualizaciones_calidad()` - Verifica visualizaciones de calidad
- `test_generar_visualizaciones_features()` - Verifica visualizaciones de features
- `test_generar_reporte_final()` - Verifica reporte consolidado

## Requisitos del Sistema

- **Python**: 3.8+
- **Kedro**: 1.0.0
- **Memoria RAM**: Mínimo 4GB (recomendado 8GB)
- **Espacio en disco**: 2GB libres

## Dependencias Principales

### Instalación Tradicional
- `kedro>=0.18.0` - Framework principal
- `pandas>=1.3.0` - Manipulación de datos
- `numpy>=1.21.0` - Computación numérica
- `matplotlib>=3.4.0` - Visualizaciones
- `seaborn>=0.11.0` - Visualizaciones estadísticas
- `scikit-learn>=1.0.0` - Machine Learning
- `jupyter>=1.0.0` - Notebooks interactivos

### Docker
- **Imagen base**: Python 3.8-slim
- **Tamaño**: ~1.75GB
- **Todas las dependencias** preinstaladas
- **Entorno consistente** en cualquier sistema

## Solución de Problemas

### Error: "No such command 'list'"
```bash
# Usar comando correcto para Kedro 1.0.0
kedro catalog list
```

### Error: "Could not find pyproject.toml"
```bash
# Asegurarse de estar en el directorio raíz del proyecto
cd proyecto-ml
kedro run
```

### Error: "Dataset not found"
```bash
# Verificar que los datos estén en data/01_raw/
ls data/01_raw/

# Ejecutar pipeline de ingeniería primero
kedro run --pipeline=ingenieria_datos
```

### Problemas con Docker

#### Error: "Docker Desktop no está ejecutándose"
```bash
# Iniciar Docker Desktop manualmente
# Verificar que el servicio esté ejecutándose
docker --version
docker info
```

#### Error: "Imagen no encontrada"
```bash
# Construir la imagen primero
docker build -t proyecto-ml:latest .

# O usar script automatizado
# Windows
scripts\build.bat

# Linux/Mac
./scripts/build.sh
```

#### Error: "Permisos denegados" (Linux/Mac)
```bash
# Dar permisos de ejecución a los scripts
chmod +x scripts/*.sh
```

#### Error: "Version mismatch" en Docker
```bash
# Este error es normal y no afecta la funcionalidad
# Los pipelines funcionan correctamente en Docker
# Solo afecta el test de bootstrap
```

## Soporte

Para reportar problemas o solicitar ayuda:

1. **Revisar logs**: `kedro run --verbose`
2. **Verificar estructura**: `kedro info`
3. **Consultar documentación**: [Kedro Docs](https://docs.kedro.org)

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## Contribuidores

- **Desarrollador Principal**: [Tu Nombre]
- **Metodología**: CRISP-DM
- **Framework**: Kedro 1.0.0

---
