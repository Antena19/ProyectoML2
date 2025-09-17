#!/bin/bash
# Script para ejecutar el proyecto ML con Docker

set -e  # Salir si hay errores

# Función para mostrar ayuda
show_help() {
    echo "🐳 Script de ejecución Docker para Proyecto ML"
    echo ""
    echo "Uso: $0 [OPCIÓN]"
    echo ""
    echo "Opciones:"
    echo "  all          Ejecutar pipeline completo (por defecto)"
    echo "  ingenieria   Ejecutar solo pipeline de ingeniería de datos"
    echo "  ciencia      Ejecutar solo pipeline de ciencia de datos"
    echo "  reportes     Ejecutar solo pipeline de reportes"
    echo "  tests        Ejecutar tests unitarios"
    echo "  interactive  Ejecutar container en modo interactivo"
    echo "  help         Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0                    # Pipeline completo"
    echo "  $0 ingenieria         # Solo ingeniería de datos"
    echo "  $0 tests              # Solo tests"
    echo "  $0 interactive        # Modo interactivo"
}

# Verificar que Docker esté instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

# Verificar que la imagen existe
if ! docker images proyecto-ml:latest &> /dev/null; then
    echo "❌ Imagen Docker no encontrada. Ejecuta primero: ./scripts/build.sh"
    exit 1
fi

# Obtener comando
COMMAND=${1:-all}

case $COMMAND in
    "all")
        echo "🚀 Ejecutando pipeline completo..."
        docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest
        ;;
    "ingenieria")
        echo "🔧 Ejecutando pipeline de ingeniería de datos..."
        docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ingenieria_datos
        ;;
    "ciencia")
        echo "📊 Ejecutando pipeline de ciencia de datos..."
        docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ciencia_datos
        ;;
    "reportes")
        echo "📈 Ejecutando pipeline de reportes..."
        docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=reportes
        ;;
    "tests")
        echo "🧪 Ejecutando tests unitarios..."
        docker run --rm proyecto-ml:latest python -m pytest -v
        ;;
    "interactive")
        echo "💻 Iniciando container en modo interactivo..."
        docker run --rm -it -v "$(pwd):/app" proyecto-ml:latest /bin/bash
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Opción no válida: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "✅ Comando ejecutado exitosamente!"
