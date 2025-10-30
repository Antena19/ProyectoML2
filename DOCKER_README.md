# 🐳 Docker para Proyecto ML

## 📋 Instrucciones de Uso

### **Prerrequisitos**
1. **Docker Desktop instalado** y ejecutándose
2. **Docker Compose** (incluido con Docker Desktop)

### **🚀 Comandos Rápidos**

#### **Construir la imagen:**
```bash
# Linux/Mac
./scripts/build.sh

# Windows
scripts\build.bat

# O directamente
docker build -t proyecto-ml:latest .
```

#### **Ejecutar el proyecto:**
```bash
# Linux/Mac
./scripts/run.sh

# Windows
scripts\run.bat

# O directamente
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest
```

### **📊 Opciones de Ejecución**

#### **Pipeline completo:**
```bash
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest
```

#### **Pipeline específico:**
```bash
# Ingeniería de datos
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ingenieria_datos

# Ciencia de datos
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=ciencia_datos

# Reportes
docker run --rm -v "$(pwd)/data:/app/data" proyecto-ml:latest python -m kedro run --pipeline=reportes
```

#### **Tests:**
```bash
docker run --rm proyecto-ml:latest python -m pytest -v
```

#### **Modo interactivo:**
```bash
docker run --rm -it -v "$(pwd):/app" proyecto-ml:latest /bin/bash
```

### **🔧 Docker Compose**

#### **Pipeline completo:**
```bash
docker-compose up
```

#### **Pipeline específico:**
```bash
# Ingeniería de datos
docker-compose --profile ingenieria up

# Ciencia de datos
docker-compose --profile ciencia up

# Reportes
docker-compose --profile reportes up

# Tests
docker-compose --profile testing up
```

### **📁 Estructura de Archivos Docker**

```
proyecto-ml/
├── Dockerfile              # Imagen base del proyecto
├── .dockerignore           # Archivos a ignorar
├── docker-compose.yml      # Configuración de servicios
├── scripts/
│   ├── build.sh           # Script de construcción (Linux/Mac)
│   ├── build.bat          # Script de construcción (Windows)
│   ├── run.sh             # Script de ejecución (Linux/Mac)
│   └── run.bat            # Script de ejecución (Windows)
└── DOCKER_README.md        # Este archivo
```

### **🎯 Beneficios de Docker**

1. **Consistencia:** Mismo entorno en cualquier lugar
2. **Portabilidad:** Funciona en Windows, Mac, Linux
3. **Aislamiento:** No interfiere con tu sistema
4. **Reproducibilidad:** Mismo resultado siempre
5. **Escalabilidad:** Fácil despliegue en producción

### **🚨 Solución de Problemas**

#### **Error: "Docker Desktop no está ejecutándose"**
```bash
# Iniciar Docker Desktop manualmente
# O verificar que el servicio esté ejecutándose
```

#### **Error: "Imagen no encontrada"**
```bash
# Construir la imagen primero
docker build -t proyecto-ml:latest .
```

#### **Error: "Permisos denegados"**
```bash
# En Linux/Mac, dar permisos de ejecución
chmod +x scripts/*.sh
```

### **📈 Comandos Útiles**

#### **Ver imágenes:**
```bash
docker images
```

#### **Ver containers:**
```bash
docker ps -a
```

#### **Limpiar imágenes no utilizadas:**
```bash
docker system prune -a
```

#### **Ver logs del container:**
```bash
docker logs <container_id>
```

### **🎉 ¡Listo para usar!**

Una vez que Docker Desktop esté ejecutándose, puedes usar cualquiera de los comandos anteriores para ejecutar tu proyecto ML de forma consistente y reproducible.

**¿Problemas?** Revisa que Docker Desktop esté ejecutándose y que tengas los permisos necesarios.
