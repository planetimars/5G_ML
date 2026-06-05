"""SHAP-based explainability helpers for trained models."""

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def _transform_features_for_model(model_pipeline, x_df):
    if not hasattr(model_pipeline, "named_steps"):
        return x_df.to_numpy(), list(x_df.columns), model_pipeline

    steps = list(model_pipeline.named_steps.items())
    estimator_name, estimator = steps[-1]
    transformed = x_df.copy()

    for step_name, step_obj in steps[:-1]:
        transformed = step_obj.transform(transformed)

    feature_names = list(x_df.columns)
    if hasattr(transformed, "shape"):
        transformed = np.asarray(transformed)
        if transformed.ndim == 2 and transformed.shape[1] != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    return transformed, feature_names, estimator


def compute_shap_importance(model_pipeline, x_df, max_samples=300):
    if not SHAP_AVAILABLE:
        return None

    sampled = x_df.sample(min(max_samples, len(x_df)), random_state=42) if len(x_df) > max_samples else x_df

    transformed, feature_names, estimator = _transform_features_for_model(model_pipeline, sampled)

    try:
        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)

        values = shap_values.values
        if values.ndim == 3:
            values = np.mean(np.abs(values), axis=(0, 2))
        else:
            values = np.mean(np.abs(values), axis=0)

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "MeanAbsSHAP": values,
            }
        ).sort_values("MeanAbsSHAP", ascending=False)
        return importance_df
    except Exception:
        return None


def build_shap_importance_plot(importance_df):
    if importance_df is None or importance_df.empty:
        return None

    return px.bar(
        importance_df.head(20),
        x="Feature",
        y="MeanAbsSHAP",
        title="SHAP Global Feature Importance (Top 20)",
        color="MeanAbsSHAP",
        text_auto=".4f",
    )
