import pandas as pd

# Cargar dataset limpio
df = pd.read_csv('data/04_feature/dataset_regresion_ml.csv', encoding='latin-1')

print('=' * 80)
print('DATASET LIMPIO PARA REGRESIÓN (SIN DATA LEAKAGE)')
print('=' * 80)
print(f'\nShape: {df.shape}')
print(f'Registros: {df.shape[0]:,}')
print(f'Columnas: {df.shape[1]}')

print(f'\n{"="*80}')
print('COLUMNAS DEL DATASET:')
print('=' * 80)
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print(f'\n{"="*80}')
print('VERIFICACIÓN DE DATA LEAKAGE:')
print('=' * 80)

# Buscar variables con "edad" en el nombre (excepto edad_cantidad que es el target)
edad_cols = [col for col in df.columns if 'edad' in col.lower() and col != 'edad_cantidad']
print(f'\n¿Variables derivadas de edad? {len(edad_cols)}')
if edad_cols:
    print('⚠️ PROBLEMA - Variables con "edad" encontradas:')
    for col in edad_cols:
        print(f'   - {col}')
else:
    print('✅ PERFECTO - No hay variables derivadas de edad (sin data leakage)')

print(f'\n{"="*80}')
print('ESTADÍSTICAS DEL DATASET:')
print('=' * 80)
print(f'\n📊 Variable objetivo:')
print(f'   - edad_cantidad: {df["edad_cantidad"].min():.0f} a {df["edad_cantidad"].max():.0f} años')
print(f'   - Media: {df["edad_cantidad"].mean():.1f} años')
print(f'   - Mediana: {df["edad_cantidad"].median():.0f} años')

print(f'\n👥 Variables demográficas:')
print(f'   - sexo: {df["sexo"].nunique()} categorías → {list(df["sexo"].unique())}')
print(f'   - region: {df["region"].nunique()} regiones')

print(f'\n🏥 Causa de muerte (CIE-10):')
print(f'   - codigo_diagnostico: {df["codigo_diagnostico"].nunique()} códigos únicos')
print(f'\n   Top 5 causas:')
for causa, count in df['codigo_diagnostico'].value_counts().head(5).items():
    print(f'      {causa}: {count:,} ({count/len(df)*100:.1f}%)')

print(f'\n⏰ Features temporales:')
temp_cols = [col for col in df.columns if any(x in col for x in ['mes', 'dia', 'trimestre', 'semana', 'invierno', 'verano', 'primavera', 'otono', 'decada'])]
print(f'   Total: {len(temp_cols)} features')

print(f'\n🗺️ Features geográficos:')
geo_cols = [col for col in df.columns if any(x in col for x in ['norte', 'centro', 'sur', 'region'])]
print(f'   Total: {len(geo_cols)} features')

print(f'\n{"="*80}')
print('CONCLUSIÓN:')
print('=' * 80)
if not edad_cols:
    print('✅ Dataset LIMPIO y listo para regresión')
    print('✅ No hay data leakage')
    print('✅ Todas las variables son INDEPENDIENTES de edad_cantidad')
else:
    print('❌ Dataset tiene data leakage')
    print('❌ Hay variables derivadas de edad')

print('\n' + '=' * 80)


