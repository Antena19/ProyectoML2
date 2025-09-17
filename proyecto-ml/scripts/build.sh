#!/bin/bash
# Script para construir la imagen Docker del proyecto ML

set -e  # Salir si hay errores

echo "🐳 Construyendo imagen Docker para Proyecto ML..."

# Verificar que Docker esté instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

# Verificar que estamos en el directorio correcto
if [ ! -f "Dockerfile" ]; then
    echo "❌ No se encontró Dockerfile. Ejecuta este script desde la raíz del proyecto."
    exit 1
fi

# Construir la imagen
echo "📦 Construyendo imagen Docker..."
docker build -t proyecto-ml:latest .

# Verificar que la imagen se construyó correctamente
if [ $? -eq 0 ]; then
    echo "✅ Imagen Docker construida exitosamente!"
    echo "📊 Información de la imagen:"
    docker images proyecto-ml:latest
    echo ""
    echo "🚀 Comandos disponibles:"
    echo "  docker run proyecto-ml:latest                    # Ejecutar pipeline completo"
    echo "  docker run proyecto-ml:latest python -m pytest -v  # Ejecutar tests"
    echo "  docker-compose up                                # Usar docker-compose"
else
    echo "❌ Error al construir la imagen Docker"
    exit 1
fi
