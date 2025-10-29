# DOCUMENTACIÓN - REGRESIÓN DE EDAD DE FALLECIMIENTO

## 📋 RESUMEN EJECUTIVO

**Problema**: Predecir la edad de fallecimiento basándose en patrones temporales y geográficos

**Objetivo**: Identificar factores de riesgo asociados a periodos específicos del año y regiones geográficas

**Dataset**: 100,000 registros de defunciones en Chile (2014-2023)

---

## 🎯 DEFINICIÓN DEL PROBLEMA

### **Variable Objetivo**
- **edad_cantidad**: Edad de fallecimiento (0-100+ años)
  - Variable **continua**
  - Rango: 0 a 100+ años
  - Media esperada: ~45 años (Chile 2014-2023)
  - Valores únicos: ~100+ (alta variabilidad)

### **Tipo de Problema**
- **Regresión** (predecir valor numérico continuo)
- **Aprendizaje Supervisado**

---

## 🔬 HIPÓTESIS EPIDEMIOLÓGICA

### **1. Mortalidad Infantil (0-5 años)**
**Patrón**: Mayor mortalidad en meses de invierno

**Justificación Científica**:
- Enfermedades respiratorias infantiles (RSV, neumonía) aumentan en invierno
- Sistema inmune infantil más vulnerable al frío
- Mayor contagio en espacios cerrados

**Variables relevantes**: `es_invierno`, `mes_sin`, `mes_cos`

---

### **2. Mortalidad en Adultos Mayores (65+ años)**
**Patrón**: Picos de mortalidad en invierno, variación por región

**Justificación Científica**:
- Hipotermia y complicaciones cardiovasculares en invierno
- Enfermedades crónicas (EPOC, cardiopatías) se agravan con el frío
- Norte de Chile: Mejor acceso a salud → Mayor esperanza de vida
- Sur de Chile: Acceso limitado → Mortalidad más temprana

**Variables relevantes**: `es_invierno`, `es_norte`, `es_centro`, `es_sur`

---

### **3. Accidentes (20-40 años)**
**Patrón**: Mayor mortalidad en fines de semana

**Justificación Científica**:
- Accidentes de tránsito aumentan en fines de semana (alcohol)
- Mayor actividad recreativa riesgosa
- Afecta principalmente a adultos jóvenes

**Variables relevantes**: `es_fin_semana`, `dia_semana_sin`, `dia_semana_cos`

---

### **4. Geografía y Acceso a Salud**
**Patrón**: Diferencias significativas por región

**Justificación Científica**:
- **Norte** (Arica, Iquique): Mejores servicios de salud, clima favorable
- **Centro** (Santiago, Valparaíso): Población urbana, contaminación
- **Sur** (Temuco, Punta Arenas): Acceso limitado, clima extremo

**Variables relevantes**: `es_norte`, `es_centro`, `es_sur`

---

## 📊 VARIABLES PREDICTORAS (21 features)

### **⭐ Demográficas (1) - MUY IMPORTANTE**
**Variable más predictiva después de causa de muerte:**
- `sexo` → Hombre=0, Mujer=1

**Justificación Epidemiológica**:
- **Mujeres**: Esperanza de vida ~7 años mayor que hombres (Chile)
  - Menor mortalidad por causas externas (accidentes, violencia)
  - Menor mortalidad cardiovascular temprana
  - Mejor adherencia a tratamientos médicos
  
- **Hombres**: Mayor mortalidad en edades jóvenes y medias
  - Mayor exposición a riesgos laborales
  - Mayor mortalidad por accidentes de tránsito
  - Mayor consumo de alcohol y tabaco
  - Mayor mortalidad por enfermedades cardiovasculares (40-60 años)

**Impacto esperado**: Esta variable sola podría explicar 15-25% de la varianza en edad de fallecimiento.

---

### **Temporales Cíclicas (8)**
Capturan estacionalidad y ciclos naturales:
- `mes_sin`, `mes_cos` → Ciclo anual
- `dia_año_sin`, `dia_año_cos` → Día del año (1-365)
- `trimestre_sin`, `trimestre_cos` → Trimestre (Q1-Q4)
- `dia_semana_sin`, `dia_semana_cos` → Día de la semana

### **Estacionales (5)**
Capturan patrones por estación del año:
- `es_fin_semana` → Binaria (0/1)
- `es_invierno` → Binaria (0/1)
- `es_verano` → Binaria (0/1)
- `es_primavera` → Binaria (0/1)
- `es_otono` → Binaria (0/1)

### **Temporales Adicionales (3)**
Capturan tendencias de largo plazo:
- `trimestre_fiscal` → Trimestre fiscal (1-4)
- `epoca_año_codificada` → Época del año codificada
- `decada` → Década (2010, 2020)

### **Geográficas (3)**
Capturan diferencias regionales:
- `es_norte` → Binaria (0/1)
- `es_centro` → Binaria (0/1)
- `es_sur` → Binaria (0/1)

---

### **❌ VARIABLE NO DISPONIBLE (sería la más importante)**
- **Causa de muerte / CIE-10**: Si estuviera disponible, sería la variable MÁS predictiva
  - Accidentes → edad joven (20-30 años)
  - Cáncer → edad media-alta (50-70 años)
  - Cardiovascular → edad alta (60-80 años)
  - Causas perinatales → edad 0-1 año

---

## 📈 MÉTRICAS DE EVALUACIÓN

### **1. R² (Coeficiente de Determinación)**
- **Definición**: % de varianza del target explicada por el modelo
- **Rango**: 0 a 1 (mayor es mejor)
- **Meta**: **R² > 0.30** (aceptable para datos epidemiológicos)
- **Interpretación**:
  - R² = 0.30 → El modelo explica 30% de la varianza en la edad
  - R² = 0.50 → Muy bueno para datos de mortalidad (factores complejos)

### **2. MAE (Mean Absolute Error)**
- **Definición**: Error promedio en años
- **Unidad**: Años
- **Meta**: **MAE < 15 años**
- **Interpretación**:
  - MAE = 10 años → El modelo se equivoca en promedio ±10 años
  - MAE = 15 años → Error razonable dado el rango 0-100 años

### **3. RMSE (Root Mean Squared Error)**
- **Definición**: Raíz del error cuadrático medio (penaliza errores grandes)
- **Unidad**: Años
- **Meta**: **RMSE < 20 años**
- **Interpretación**:
  - RMSE > MAE → Hay algunos errores muy grandes
  - RMSE ≈ MAE → Errores distribuidos uniformemente

### **4. MAPE (Mean Absolute Percentage Error)**
- **Definición**: Error porcentual promedio
- **Unidad**: Porcentaje (%)
- **Meta**: **MAPE < 30%**
- **Interpretación**:
  - MAPE = 20% → El modelo se equivoca en promedio 20% de la edad real
  - **Problema**: MAPE falla con edades cercanas a 0 (división por cero)

### **5. Cross-Validation R² (mean ± std)**
- **Definición**: R² promedio en 5 folds con desviación estándar
- **Meta**: **std < 0.05** (baja variabilidad)
- **Interpretación**:
  - std = 0.02 → Modelo muy estable
  - std > 0.10 → Modelo inestable (overfitting o datos problemáticos)

---

## 🤖 MODELOS IMPLEMENTADOS (6)

### **1. Ridge Regression**
- **Tipo**: Linear Regression con regularización L2
- **Ventajas**: Simple, interpretable, rápido
- **Desventajas**: Asume relación lineal
- **Hiperparámetros**: `alpha` (fuerza de regularización)

### **2. Random Forest Regressor**
- **Tipo**: Ensamble de árboles de decisión
- **Ventajas**: No lineal, robusto, maneja interacciones
- **Desventajas**: Lento, requiere mucha memoria
- **Hiperparámetros**: `n_estimators`, `max_depth`, `min_samples_split`

### **3. Gradient Boosting Regressor**
- **Tipo**: Boosting secuencial de árboles
- **Ventajas**: Muy preciso, maneja no linealidad
- **Desventajas**: Prone a overfitting, lento
- **Hiperparámetros**: `n_estimators`, `learning_rate`, `max_depth`

### **4. SVR (Support Vector Regression)**
- **Tipo**: Regresión con vectores de soporte
- **Ventajas**: Efectivo en espacios de alta dimensión
- **Desventajas**: MUY LENTO (O(n²)), difícil de interpretar
- **Hiperparámetros**: `C`, `kernel`, `gamma`

### **5. KNN Regressor**
- **Tipo**: Regresión por vecindad (k-nearest neighbors)
- **Ventajas**: Simple, no paramétrico
- **Desventajas**: Sensible a escala, lento en predicción
- **Hiperparámetros**: `n_neighbors`, `weights`, `metric`

### **6. Decision Tree Regressor**
- **Tipo**: Árbol de decisión simple
- **Ventajas**: Interpretable, visual, rápido
- **Desventajas**: Prone a overfitting
- **Hiperparámetros**: `max_depth`, `min_samples_split`, `criterion`

---

## 🔧 CONFIGURACIÓN DE ENTRENAMIENTO

### **GridSearchCV**
- **Objetivo**: Encontrar mejores hiperparámetros
- **Método**: Búsqueda exhaustiva en grilla
- **Scoring**: R² (maximizar)

### **CrossValidation**
- **Método**: K-Fold (k=5)
- **Objetivo**: Validación robusta
- **Ventaja**: Reduce overfitting

### **División de Datos**
- **Train**: 80% (80,000 registros)
- **Test**: 20% (20,000 registros)
- **Método**: Aleatorio con `random_state=42`

### **Normalización**
- **Método**: StandardScaler (z-score)
- **Fórmula**: `(x - mean) / std`
- **Aplicación**: Antes del train/test split

---

## ✅ CUMPLIMIENTO DE RÚBRICA

| **Requisito** | **Cumplimiento** | **Evidencia** |
|---------------|------------------|---------------|
| Mínimo 6 modelos de regresión | ✅ SÍ | 6 modelos implementados |
| GridSearchCV para hiperparámetros | ✅ SÍ | Todos los modelos usan GridSearchCV |
| CrossValidation con k≥5 | ✅ SÍ | k=5 folds configurado |
| División train/test | ✅ SÍ | 80/20 implementado |
| Métricas completas (mean ± std) | ✅ SÍ | MAE, MSE, RMSE, R², MAPE con CV |
| Documentación científica | ✅ SÍ | Este documento |
| Justificación epidemiológica | ✅ SÍ | Hipótesis fundamentadas |

---

## ⏱️ ESTIMACIÓN DE TIEMPO

### **Tiempos Estimados por Modelo**
1. **Ridge Regression**: ~5-10 segundos
2. **Random Forest**: ~60-90 segundos
3. **Gradient Boosting**: ~80-120 segundos
4. **SVR**: **~1-2 HORAS** ⚠️ (MUY LENTO)
5. **KNN**: ~40-60 segundos
6. **Decision Tree**: ~1-2 segundos

### **Tiempo Total Estimado**
- **Con SVR**: ~2-3 horas
- **Sin SVR**: ~3-5 minutos

---

## 📋 ARCHIVOS GENERADOS

Al finalizar el entrenamiento:

1. **resultados_regresion.pkl**: Resultados completos de todos los modelos
2. **tabla_comparativa_regresion.csv**: Tabla con métricas de todos los modelos
3. **data/06_models/regresion_edad_cantidad_*.pkl**: 6 modelos entrenados

---

## 🚀 COMANDO DE EJECUCIÓN

```bash
python -m kedro run --pipeline=regresion
```

Para monitorear el progreso durante el entrenamiento, ejecutar en otra terminal:

```bash
python monitor_progreso.py
```

---

## 📊 RESULTADOS ESPERADOS

### **Escenario Optimista** (R² > 0.40) ⭐ **MEJORADO CON SEXO**
- La variable SEXO aporta significativamente a la predicción
- Patrones claros de mortalidad diferencial por género
- Variables temporales y geográficas complementan bien
- Error promedio < 12 años

### **Escenario Realista** (R² = 0.25-0.40)
- SEXO explica ~20% de varianza
- Patrones temporales/geográficos añaden ~10-20% adicional
- Error promedio 12-18 años
- Random Forest y Gradient Boosting superan a modelos lineales

### **Escenario Pesimista** (R² < 0.25)
- SEXO tiene efecto pero limitado
- Patrones temporales débiles
- Necesidad crítica de agregar causa de muerte (CIE-10)

---

## 💡 MEJORAS FUTURAS

1. **Agregar features de CIE-10**: Código de causa de muerte (muy predictivo)
2. **Interacciones**: edad_cantidad × región, edad_cantidad × mes
3. **Features de contexto**: Tasa de mortalidad histórica por región
4. **Modelos avanzados**: XGBoost, LightGBM, CatBoost
5. **Ensemble**: Combinar predicciones de múltiples modelos

---

**Autor**: Proyecto ML - Chile  
**Fecha**: Octubre 2025  
**Versión**: 1.0

