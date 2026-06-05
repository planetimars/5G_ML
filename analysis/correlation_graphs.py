"""Correlation plotting utilities for numeric network features."""

import numpy as np
import plotly.express as px


def render_correlation_heatmap(dataset):
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    if numeric_dataset.empty:
        return None

    corr_matrix = numeric_dataset.corr()
    return px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
    )


def render_target_correlation_bar(dataset, target_col, top_n=12):
    numeric_dataset = dataset.select_dtypes(include=[np.number])
    if target_col not in numeric_dataset.columns:
        return None

    corr_series = numeric_dataset.corr()[target_col].drop(labels=[target_col], errors="ignore")
    if corr_series.empty:
        return None

    corr_df = corr_series.abs().sort_values(ascending=False).head(top_n).reset_index()
    corr_df.columns = ["Feature", "AbsoluteCorrelationWithTarget"]
    return px.bar(
        corr_df,
        x="Feature",
        y="AbsoluteCorrelationWithTarget",
        color="AbsoluteCorrelationWithTarget",
        title=f"Top {min(top_n, len(corr_df))} Feature Correlations with {target_col}",
        text_auto=".3f",
    )


def render_scatter_matrix(dataset, selected_features, color_col=None, max_points=500):
    numeric_cols = list(dataset.select_dtypes(include=[np.number]).columns)
    dims = [col for col in selected_features if col in numeric_cols][:6]
    if len(dims) < 2:
        return None

    plot_df = dataset[dims + ([color_col] if color_col and color_col in dataset.columns else [])].copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    fig = px.scatter_matrix(
        plot_df,
        dimensions=dims,
        color=color_col if color_col in plot_df.columns else None,
        title="Feature Scatter Matrix (sampled)",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return fig
