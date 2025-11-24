# src/proyecto_ml/pipelines/modelado/nodes.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# Split
from sklearn.model_selection import train_test_split

# Modelos de regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Métricas
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# Nodo previo: preparar dataset de modelado (X + y)
# ──────────────────────────────────────────────────────────────────────────────
def preparar_model_input(
    df_features: pd.DataFrame,
    params: dict,  # keys: target_col, drop_cols (opc.), one_hot_max_uniques (opc.)
) -> pd.DataFrame:
    """
    Limpia, imputa y codifica categóricas (one-hot) con cardinalidad controlada.
    Devuelve un DataFrame con X + y (target al final).
    """
    target_col = params.get("target_col", "defunciones_totales")
    drop_cols = params.get("drop_cols", [])
    one_hot_max_uniques = params.get("one_hot_max_uniques", 40)

    df = df_features.copy()

    # Integrar etiquetas de clustering como features categóricas (opcional)
    try:
        add_clusters = params.get("use_cluster_features", True)
        if add_clusters:
            import os
            def _read_csv_robusto(p):
                import pandas as pd
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        return pd.read_csv(p, encoding=enc)
                    except UnicodeDecodeError:
                        pass
                return pd.read_csv(p, encoding="latin-1")

            base_paths = {
                "kmeans": "data/07_model_output/clustering/kmeans_labels.csv",
                "hier":   "data/07_model_output/clustering/hier_labels.csv",
                "gmm":    "data/07_model_output/clustering/gmm_labels.csv",
                "dbscan": "data/07_model_output/clustering/dbscan_labels.csv",
            }
            for name, path in base_paths.items():
                if os.path.exists(path):
                    lab = _read_csv_robusto(path)
                    if "cluster" in lab.columns:
                        serie = lab["cluster"].astype(str)
                        if len(serie) == len(df):
                            df[f"cluster_{name}"] = serie.values
                        else:
                            # Alinear por posición hasta el mínimo común
                            n = min(len(serie), len(df))
                            df[f"cluster_{name}"] = pd.Series(serie.iloc[:n].tolist() + [None] * (len(df) - n))
    except Exception:
        pass

    # 1) Validación de target
    if target_col not in df.columns:
        primeras = ", ".join(df.columns[:25])
        raise ValueError(
            f"Target '{target_col}' no existe en el dataset de entrada. "
            f"Columnas detectadas (parcial): {primeras}"
        )
    df = df[~df[target_col].isna()].reset_index(drop=True)

    # 2) Quitar columnas no deseadas (sin tocar el target)
    cols_quitar = [c for c in drop_cols if c in df.columns and c != target_col]
    if cols_quitar:
        df = df.drop(columns=cols_quitar, errors="ignore")

    # 3) Tipos
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = [c for c in df.columns if c not in numericas and c != target_col]

    # 4) Imputación simple
    for c in numericas:
        if c == target_col:
            continue
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in categoricas:
        if df[c].isna().any():
            moda = df[c].mode(dropna=True)
            df[c] = df[c].fillna(moda.iloc[0] if len(moda) > 0 else "desconocido")

    # 5) One-hot con límite de cardinalidad
    cols_one_hot: List[str] = []
    for c in categoricas:
        if df[c].nunique(dropna=True) <= one_hot_max_uniques:
            cols_one_hot.append(c)
        else:
            df = df.drop(columns=[c])

    if cols_one_hot:
        df = pd.get_dummies(df, columns=cols_one_hot, drop_first=True)

    # 6) Sanitizar
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7) Target al final
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def _separar_X_y(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X y y; elimina de X las columnas en drop_cols y el target."""
    drop_cols = drop_cols or []
    cols_quitar = list(set(drop_cols + [target_col]))
    X = df.drop(columns=cols_quitar, errors="ignore")
    y = df[target_col]
    return X, y


def _modelos_por_tipo(problem_type: str) -> Dict[str, object]:
    """Modelos base, robustos y rápidos para un primer barrido."""
    if problem_type.lower() == "regresion":
        return {
            "linreg": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.001, max_iter=10000),
            "rf": RandomForestRegressor(n_estimators=300, random_state=42),
            "gbr": GradientBoostingRegressor(random_state=42),
            "svr": SVR(),
        }
    elif problem_type.lower() == "clasificacion":
        return {
            "logreg": LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"),
            "dt": DecisionTreeClassifier(random_state=42),
            "rf": RandomForestClassifier(n_estimators=300, random_state=42),
            "gbc": GradientBoostingClassifier(random_state=42),
            "svc": SVC(probability=True, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=7),
        }
    else:
        raise ValueError("problem_type debe ser 'regresion' o 'clasificacion'.")


def _metricas_regresion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)  # sin 'squared'
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _metricas_clasificacion(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> Dict[str, float]:
    """Métricas robustas; evita errores si falta alguna clase."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # ROC-AUC (si hay probas)
    try:
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                )
    except Exception:
        pass
    return metrics


def _importancias(best_model, feature_names: list[str]) -> Optional[pd.DataFrame]:
    """Importancias o coeficientes si el modelo las expone."""
    try:
        if hasattr(best_model, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": feature_names, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)
            return imp
        elif hasattr(best_model, "coef_"):
            coefs = np.ravel(best_model.coef_)
            imp = pd.DataFrame(
                {"feature": feature_names, "importance": coefs}
            ).sort_values("importance", ascending=False)
            return imp
    except Exception:
        return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Nodos principales
# ──────────────────────────────────────────────────────────────────────────────
def split_train_test(
    df_model: pd.DataFrame,
    params: Dict,  # keys: target_col, test_size, random_state, drop_cols (opc.)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Split aleatorio (útil para datos i.i.d.)."""
    target_col = params.get("target_col", "defunciones_totales")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    drop_cols = params.get("drop_cols", [])

    X, y = _separar_X_y(df_model, target_col=target_col, drop_cols=drop_cols)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def split_train_test_temporal(
    df_model: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Split temporal: ordena por 'año' y reserva el último tramo como test.
    Adecuado para series anuales (evita fuga temporal).
    """
    target_col = params.get("target_col", "defunciones_totales")
    test_size = params.get("test_size", 0.2)
    drop_cols = params.get("drop_cols", [])

    if "año" in df_model.columns:
        df_model = df_model.sort_values("año").reset_index(drop=True)

    X = df_model.drop(columns=list(set(drop_cols + [target_col])), errors="ignore")
    y = df_model[target_col]
    feature_names = list(X.columns)

    n = len(df_model)
    n_test = max(1, int(round(n * test_size)))
    split_idx = n - n_test

    X_train, X_test = X.iloc[:split_idx, :], X.iloc[split_idx:, :]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_names


def entrenar_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict,  # key: problem_type
) -> Dict[str, object]:
    """Entrena un conjunto breve de modelos según el tipo de problema."""
    problem_type = params.get("problem_type", "regresion").lower()
    modelos = _modelos_por_tipo(problem_type)
    for m in modelos.values():
        m.fit(X_train, y_train)
    return modelos


def evaluar_modelos(
    modelos: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,  # key: problem_type
) -> pd.DataFrame:
    """Evalúa todos los modelos y devuelve una tabla de métricas."""
    problem_type = params.get("problem_type", "regresion").lower()
    resultados = []

    for nombre, modelo in modelos.items():
        if problem_type == "regresion":
            y_hat = modelo.predict(X_test)
            met = _metricas_regresion(y_test, y_hat)
        else:
            y_pred = modelo.predict(X_test)
            y_proba = None
            if hasattr(modelo, "predict_proba"):
                proba = modelo.predict_proba(X_test)
                y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.ravel(proba)
            met = _metricas_clasificacion(y_test, y_pred, y_proba)

        fila = {"modelo": nombre}
        fila.update(met)
        resultados.append(fila)

    df_metricas = pd.DataFrame(resultados) \
        .sort_values("modelo") \
        .reset_index(drop=True)      # ← no usamos set_index("modelo")
    return df_metricas


def seleccionar_mejor_modelo(metricas: pd.DataFrame, modelos: Dict[str, object], params: Dict) -> object:
    import numpy as np
    problem_type = params.get("problem_type", "regresion").lower()
    criterio = params.get("criterio") or ("RMSE" if problem_type == "regresion" else "f1_macro")

    # limpiar columnas fantasma
    if "Unnamed: 0" in metricas.columns:
        metricas = metricas.drop(columns=["Unnamed: 0"])

    # si viene "modelo" como columna, úsala como índice
    if "modelo" in metricas.columns:
        metricas = metricas.set_index("modelo")

    # si el criterio no está, elegir primer numérico disponible
    if criterio not in metricas.columns:
        num_cols = metricas.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("La tabla de métricas no contiene columnas numéricas.")
        criterio = num_cols[0]

    serie = metricas[criterio].astype(float)

    mejor = serie.idxmin() if problem_type == "regresion" else serie.idxmax()

    if mejor not in modelos:
        raise KeyError(
            f"No encuentro el modelo '{mejor}' en {list(modelos.keys())}. "
            "Asegúrate de que la columna 'modelo' exista en 'metricas_modelos' o guarda el índice."
        )
    return modelos[mejor]



def predecir_con_mejor_modelo(
    best_model: object,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Genera predicciones del mejor modelo y, si existen, probabilidades.
    Devuelve siempre un DataFrame para 'probabilidades' (vacío si no aplica).
    """
    import numpy as np
    import pandas as pd

    y_pred = pd.Series(best_model.predict(X_test), index=X_test.index, name="pred")

    proba_df = pd.DataFrame(columns=["proba"])  # por defecto vacío
    try:
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X_test)
            if proba.ndim == 1 or proba.shape[1] == 1:
                proba_df = pd.DataFrame({"proba": np.ravel(proba)}, index=X_test.index).reset_index(drop=True)
            else:
                # multiclase: guardar todas las columnas de probas
                cols = [f"proba_{i}" for i in range(proba.shape[1])]
                proba_df = pd.DataFrame(proba, index=X_test.index, columns=cols).reset_index(drop=True)
    except Exception:
        pass

    return y_pred, proba_df



def importancia_features(
    best_model: object,
    feature_names: list[str],
) -> Optional[pd.DataFrame]:
    """Importancias/coeficientes si el modelo lo soporta."""
    return _importancias(best_model, feature_names)


def consolidar_resultados(
    metricas: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.DataFrame | None = None,
    params: dict | None = None,
) -> dict[str, pd.DataFrame]:
    comparacion = pd.DataFrame({"y_real": y_test, "y_pred": y_pred}).reset_index(drop=True)
    out = {"metricas": metricas.reset_index(), "comparacion": comparacion}

    if y_proba is not None and not y_proba.empty:
        out["probabilidades"] = y_proba
    else:
        # opcional: guarde CSV vacío con encabezado estándar
        out["probabilidades"] = pd.DataFrame(columns=["proba"])

    # Copia opcional a carpeta de comparación según 'run_tag'
    try:
        run_tag = None
        if params and isinstance(params, dict):
            run_tag = params.get("run_tag")
        if run_tag:
            import os
            base_dir = "data/07_model_output"
            cmp_dir = os.path.join(base_dir, "compare", str(run_tag))
            os.makedirs(cmp_dir, exist_ok=True)

            # Guardar copias en CSV
            out["metricas"].to_csv(os.path.join(cmp_dir, "metricas_resumen.csv"), index=False)
            out["comparacion"].to_csv(os.path.join(cmp_dir, "comparacion_y_real_vs_pred.csv"), index=False)
            out["probabilidades"].to_csv(os.path.join(cmp_dir, "probabilidades.csv"), index=False)

            # Copiar metricas_modelos si existe
            try:
                import pandas as pd
                mm_path = os.path.join(base_dir, "metricas_modelos.csv")
                if os.path.exists(mm_path):
                    df_mm = pd.read_csv(mm_path)
                    df_mm.to_csv(os.path.join(cmp_dir, "metricas_modelos.csv"), index=False)
            except Exception:
                pass
    except Exception:
        pass

    return out

"""
Nodos del pipeline de Modelado Supervisado.

Propósito:
- Preparar datos para entrenamiento, entrenar un conjunto breve de modelos,
  evaluar sus métricas, seleccionar el mejor y producir predicciones y
  probabilidades, junto a importancias de variables.

Selección del mejor:
- Por defecto usa `RMSE` si el problema es de regresión y `f1_macro` si es
  clasificación. Con los datos actuales de defunciones (regresión), el mejor
  suele ser `linreg` (Regresión Lineal).

Salidas y dónde verlas:
- `data/06_models/mejor_modelo.pkl`: modelo ganador (promovido por Airflow a
  `models/production/model.pkl`).
- `data/07_model_output/metricas_modelos.csv` y `metricas_resumen.csv`:
  rendimiento por modelo y resumen.
- `data/07_model_output/comparacion_y_real_vs_pred.csv`: reales vs predichos.
- `data/07_model_output/importancias_features.csv`: importancias/coeficientes si aplica.
- En producción: `models/production/fig_prediccion.png` y `prediccion_actual.csv`.
"""

