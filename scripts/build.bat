@echo off
REM Script para construir la imagen Docker del proyecto ML (Windows)

echo 🐳 Construyendo imagen Docker para Proyecto ML...

REM Verificar que Docker esté instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker no está instalado. Por favor instala Docker primero.
    exit /b 1
)

REM Verificar que estamos en el directorio correcto
if not exist "Dockerfile" (
    echo ❌ No se encontró Dockerfile. Ejecuta este script desde la raíz del proyecto.
    exit /b 1
)

REM Construir la imagen
echo 📦 Construyendo imagen Docker...
docker build -t proyecto-ml:latest .

REM Verificar que la imagen se construyó correctamente
if %errorlevel% equ 0 (
    echo ✅ Imagen Docker construida exitosamente!
    echo 📊 Información de la imagen:
    docker images proyecto-ml:latest
    echo.
    echo 🚀 Comandos disponibles:
    echo   docker run proyecto-ml:latest                    # Ejecutar pipeline completo
    echo   docker run proyecto-ml:latest python -m pytest -v  # Ejecutar tests
    echo   docker-compose up                                # Usar docker-compose
) else (
    echo ❌ Error al construir la imagen Docker
    exit /b 1
)
