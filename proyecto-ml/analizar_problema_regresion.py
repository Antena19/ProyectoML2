import pandas as pd

# Cargar dataset
df = pd.read_csv('data/04_feature/dataset_regresion_ml.csv', encoding='latin-1')

print('=' * 80)
print('DIAGNÓSTICO DEL PROBLEMA DE REGRESIÓN')
print('=' * 80)

print(f'\nShape del dataset: {df.shape}')
print(f'\nColumnas totales: {len(df.columns)}')
print(f'Primeras 20 columnas:')
for i, col in enumerate(df.columns[:20], 1):
    print(f'  {i:2d}. {col}')

# Detectar columnas OneHot
region_cols = [col for col in df.columns if col.startswith('region_')]
cie10_cols = [col for col in df.columns if col.startswith('cie10_')]

print(f'\n{"="*80}')
print('ANÁLISIS DE VARIABLES CATEGÓRICAS:')
print('=' * 80)

print(f'\nColumnas region_*: {len(region_cols)}')
if region_cols:
    print(f'  Ejemplos: {region_cols[:3]}')

print(f'\nColumnas cie10_*: {len(cie10_cols)}')
if cie10_cols:
    print(f'  Ejemplos: {cie10_cols[:3]}')

# Verificar distribución de CIE-10
print(f'\n{"="*80}')
print('DISTRIBUCIÓN DE CAUSAS DE MUERTE (CIE-10):')
print('=' * 80)

# Contar cuántos partecipan en cada categoría
top_5 = df['cie10_I00-I99'].sum(), df['cie10_C00-D48'].sum(), df['cie10_J00-J99'].sum()
print(f'\nTop 3 causas de muerte:')
print(f'  - cie10_I00-I99: {df["cie10_I00-I99"].sum():,} casos ({df["cie10_I00-I99"].sum()/len(df)*100:.1f}%)')
print(f'  - cie10_C00-D48: {df["cie10_C00-D48"].sum():,} casos ({df["cie10_C00-D48"].sum()/len(df)*100:.1f}%)')
print(f'  - cie10_J00-J99: {df["cie10_J00-J99"].sum():,} casos ({df["cie10_J00-J99"].sum()/len(df)*100:.1f}%)')

# Calcular edad promedio por causa
print(f'\n{"="*80}')
print('EDAD PROMEDIO POR CAUSA DE MUERTE:')
print('=' * 80)

if 'cie10_I00-I99' in df.columns:
    df_i00 = df[df['cie10_I00-I99'] == 1]
    print(f'  - I00-I99 (cardiovascular): {df_i00["edad_cantidad"].mean():.1f} años (std: {df_i00["edad_cantidad"].std():.1f})')

if 'cie10_C00-D48' in df.columns:
    df_c00 = df[df['cie10_C00-D48'] == 1]
    print(f'  - C00-D48 (cáncer): {df_c00["edad_cantidad"].mean():.1f} años (std: {df_c00["edad_cantidad"].std():.1f})')

if 'cie10_J00-J99' in df.columns:
    df_j00 = df[df['cie10_J00-J99'] == 1]
    print(f'  - J00-J99 (respiratorio): {df_j00["edad_cantidad"].mean():.1f} años (std: {df_j00["edad_cantidad"].std():.1f})')

if 'cie10_S00-T98' in df.columns:
    df_s00 = df[df['cie10_S00-T98'] == 1]
    print(f'  - S00-T98 (accidentes): {df_s00["edad_cantidad"].mean():.1f} años (std: {df_s00["edad_cantidad"].std():.1f})')

print(f'\n{"="*80}')
print('CONCLUSIÓN:')
print('=' * 80)
print('\nSi las edades promedio por causa son MUY SIMILARES (>70 años),')
print('entonces la causa de muerte NO predice bien la edad.')
print('\nEsto explicaría por qué R² es bajo (0.04) aunque usamos OneHot.')


