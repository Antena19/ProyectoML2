# Dockerfile para Proyecto ML con Kedro
# Imagen base: Python 3.8 slim (ligera y eficiente)
FROM python:3.8-slim

# Metadatos de la imagen
LABEL maintainer="Proyecto ML Team"
LABEL description="Proyecto de Machine Learning con Kedro para análisis de defunciones"
LABEL version="1.0"

# Establecer directorio de trabajo dentro del container
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY . .

# Crear directorios necesarios para datos
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary data/04_feature data/05_model_input data/06_models data/07_model_output data/08_reporting

# Establecer variables de entorno
ENV KEDRO_ENV=local
ENV PYTHONPATH=/app/src
ENV KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Exponer puerto (para futuras funcionalidades web)
EXPOSE 8000

# Comando por defecto: ejecutar pipeline completo
CMD ["python", "-m", "kedro", "run"]

# Comandos alternativos disponibles:
# CMD ["python", "-m", "kedro", "run", "--pipeline=ingenieria_datos"]
# CMD ["python", "-m", "kedro", "run", "--pipeline=ciencia_datos"]
# CMD ["python", "-m", "kedro", "run", "--pipeline=reportes"]
# CMD ["python", "-m", "pytest", "-v"]
