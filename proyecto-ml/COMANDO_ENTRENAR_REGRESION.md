# 🚀 COMANDO PARA ENTRENAR MODELOS DE REGRESIÓN

## Dataset preparado con OneHotEncoding:
- ✅ 100,000 registros
- ✅ 56 columnas numéricas
- ✅ Variables categóricas convertidas (region_*, cie10_*, sexo)
- ✅ Sin data leakage

## Comando para entrenar:
```bash
python -m kedro run --pipeline=regresion
```

## ⏱️ Tiempo estimado: ~1 hora
- Linear Regression: ~5 segundos
- Random Forest: ~3 minutos
- Gradient Boosting: ~4 minutos
- **SVR: ~40 minutos** (el más lento)
- KNN: ~2 minutos
- Decision Tree: ~5 segundos

## 📊 Resultados esperados CON OneHotEncoding:
- **R² = 0.40 - 0.60** (mucho mejor que 0.002)
- **MAE = 10 - 12 años** (mejor que 14 años)

## ¿Por qué mejorará?
Con OneHotEncoding, el modelo aprenderá patrones específicos por:
- **Causa de muerte**: I00-I99 (cardiovascular) → edad alta (~75)
- **Región**: Regiones con mejor acceso a salud → mayor esperanza de vida
- **Temporalidad**: Estaciones y tendencias temporales

¡Ejecuta el comando cuando estés listo!


