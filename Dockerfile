FROM python:3.8-slim

LABEL maintainer="Proyecto ML Team"
WORKDIR /app

# Sistema
RUN apt-get update && apt-get install -y gcc g++ git \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    KEDRO_ENV=local \
    KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Dependencias (usar archivo compatible con Py3.8)
COPY requirements_docker.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Código del proyecto
COPY . .

# Directorios de datos (si no monta volúmenes)
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary \
             data/04_feature data/05_model_input data/06_models \
             data/07_model_output data/08_reporting

EXPOSE 8000
CMD ["python", "-m", "kedro", "run"]
