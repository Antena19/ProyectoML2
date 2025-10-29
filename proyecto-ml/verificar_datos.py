import pickle
import pandas as pd

# Verificar datasets_estandarizados
print("=" * 80)
print("DATASETS ESTANDARIZADOS")
print("=" * 80)
datos = pickle.load(open('data/03_primary/datasets_estandarizados.pkl', 'rb'))
print(f"Keys: {list(datos.keys())}\n")

for k, v in datos.items():
    if isinstance(v, pd.DataFrame):
        print(f"{k}:")
        print(f"  Shape: {v.shape}")
        print(f"  Columnas: {list(v.columns)[:20]}")
        print()


