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

# 1️⃣ Importar Kedro y crear la sesión del proyecto
from kedro.framework.session import KedroSession

# Crear sesión del proyecto y cargar el catálogo
session = KedroSession.create()
catalog = session.load_context().catalog

# 2️⃣ Cargar los datasets crudos (01_raw)
datos_historicos = catalog.load("datos_historicos_nacimientos_defunciones")
datos_filtrados_defunciones = catalog.load("datos_filtrados_defunciones")
nacimientos_por_sexo = catalog.load("nacimientos_defunciones_por_sexo")
nacimientos_por_edad_madre = catalog.load("nacimientos_por_edad_madre")
defunciones_por_edad_fallecido = catalog.load("defunciones_por_edad_fallecido")

# 3️⃣ Mostrar un vistazo rápido de los CSV principales
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