@echo off
REM Script para ejecutar el proyecto ML con Docker (Windows)

setlocal enabledelayedexpansion

REM Función para mostrar ayuda
if "%1"=="help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

REM Verificar que Docker esté instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker no está instalado. Por favor instala Docker primero.
    exit /b 1
)

REM Verificar que la imagen existe
docker images proyecto-ml:latest >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Imagen Docker no encontrada. Ejecuta primero: scripts\build.bat
    exit /b 1
)

REM Obtener comando
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=all

if "%COMMAND%"=="all" (
    echo 🚀 Ejecutando pipeline completo...
    docker run --rm -v "%cd%\data:/app/data" proyecto-ml:latest
) else if "%COMMAND%"=="ingenieria" (
    echo 🔧 Ejecutando pipeline de ingeniería de datos...
    docker run --rm -v "%cd%\data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ingenieria_datos
) else if "%COMMAND%"=="ciencia" (
    echo 📊 Ejecutando pipeline de ciencia de datos...
    docker run --rm -v "%cd%\data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ciencia_datos
) else if "%COMMAND%"=="reportes" (
    echo 📈 Ejecutando pipeline de reportes...
    docker run --rm -v "%cd%\data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=reportes
) else if "%COMMAND%"=="tests" (
    echo 🧪 Ejecutando tests unitarios...
    docker run --rm proyecto-ml:latest python -m pytest -v
) else if "%COMMAND%"=="interactive" (
    echo 💻 Iniciando container en modo interactivo...
    docker run --rm -it -v "%cd%:/app" proyecto-ml:latest /bin/bash
) else (
    echo ❌ Opción no válida: %COMMAND%
    goto :show_help
)

echo ✅ Comando ejecutado exitosamente!
goto :eof

:show_help
echo 🐳 Script de ejecución Docker para Proyecto ML
echo.
echo Uso: %0 [OPCIÓN]
echo.
echo Opciones:
echo   all          Ejecutar pipeline completo (por defecto)
echo   ingenieria   Ejecutar solo pipeline de ingeniería de datos
echo   ciencia      Ejecutar solo pipeline de ciencia de datos
echo   reportes     Ejecutar solo pipeline de reportes
echo   tests        Ejecutar tests unitarios
echo   interactive  Ejecutar container en modo interactivo
echo   help         Mostrar esta ayuda
echo.
echo Ejemplos:
echo   %0                    # Pipeline completo
echo   %0 ingenieria         # Solo ingeniería de datos
echo   %0 tests              # Solo tests
echo   %0 interactive        # Modo interactivo
