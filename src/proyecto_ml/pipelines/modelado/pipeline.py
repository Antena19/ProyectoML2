# src/proyecto_ml/pipelines/modelado/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    preparar_model_input,
    split_train_test_temporal,   # usamos split temporal
    entrenar_modelos,
    evaluar_modelos,
    seleccionar_mejor_modelo,
    predecir_con_mejor_modelo,
    importancia_features,
    consolidar_resultados,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # 0) Preparar dataset de modelado (03_primary → memoria)
        node(
            func=preparar_model_input,
            inputs=dict(
                df_features="features_temporales",   # definido en catalog.yml
                params="params:modelado",
            ),
            outputs="model_input_prepared",
            name="preparar_model_input",
        ),

        # 1) Split temporal (evita fuga por año)
        node(
            func=split_train_test_temporal,
            inputs=dict(
                df_model="model_input_prepared",
                params="params:modelado",
            ),
            outputs=["X_train", "X_test", "y_train", "y_test", "feature_names"],
            name="split_train_test_temporal",
        ),

        # 2) Entrenar varios modelos
        node(
            func=entrenar_modelos,
            inputs=dict(
                X_train="X_train",
                y_train="y_train",
                params="params:modelado",
            ),
            outputs="modelos_entrenados",
            name="entrenar_modelos",
        ),

        # 3) Evaluar todos
        node(
            func=evaluar_modelos,
            inputs=dict(
                modelos="modelos_entrenados",
                X_test="X_test",
                y_test="y_test",
                params="params:modelado",
            ),
            outputs="metricas_modelos",
            name="evaluar_modelos",
        ),

        # 4) Seleccionar el mejor
        node(
            func=seleccionar_mejor_modelo,
            inputs=dict(
                metricas="metricas_modelos",
                modelos="modelos_entrenados",
                params="params:modelado",
            ),
            outputs="mejor_modelo",
            name="seleccionar_mejor_modelo",
        ),

        # 5) Importancias/coeficientes (si aplica)
        node(
            func=importancia_features,
            inputs=dict(
                best_model="mejor_modelo",
                feature_names="feature_names",
            ),
            outputs="importancias",
            name="importancia_features",
        ),

        # 6) Predicciones (y probabilidades si hay)
        node(
            func=predecir_con_mejor_modelo,
            inputs=dict(
                best_model="mejor_modelo",
                X_test="X_test",
            ),
            outputs=["predicciones", "probabilidades"],
            name="predecir_con_mejor_modelo",
        ),

        # 7) Consolidados finales
        node(
            func=consolidar_resultados,
            inputs=dict(
                metricas="metricas_modelos",
                y_test="y_test",
                y_pred="predicciones",
                y_proba="probabilidades",
            ),
            outputs=dict(
                metricas="metricas",
                comparacion="comparacion",
                probabilidades="probabilidades_final",
            ),
            name="consolidar_resultados",
        ),
    ])
