"""Main Streamlit app for 5G prediction, optimization, and model research workflows."""

import io
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from analysis.correlation_graphs import render_correlation_heatmap, render_scatter_matrix, render_target_correlation_bar
from analysis.advanced_research import (
    compute_calibration_and_confidence,
    compute_learning_curve_data,
    compute_permutation_importance,
    compute_stability_analysis,
)
from analysis.ai_recommendations import generate_ai_optimization_plan, has_groq_api_key
from analysis.explainability import SHAP_AVAILABLE, build_shap_importance_plot, compute_shap_importance
from analysis.feature_selection import run_feature_selection
from analysis.research_plots import classification_diagnostics, multiclass_roc_auc, regression_diagnostics
from analysis.run_comparison import (
    best_model_by_metric,
    discover_run_directories,
    infer_primary_metric,
    load_run_table,
)
from analysis.pdf_export import build_pdf_report
from models.model_registry import evaluate_models, get_model_registry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRETRAINED_DIR = BASE_DIR / "artifacts" / "pretrained"

st.set_page_config(
    page_title="5G ML Optimization System",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --bg-base: #f2f7ff;
    --surface: #ffffff;
    --ink-strong: #0d1b2a;
    --ink-soft: #375a7f;
    --brand: #0b66c3;
    --ok: #1f8a5b;
    --warn: #ca8a04;
    --bad: #c0392b;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, #dbeafe 0%, transparent 38%),
        radial-gradient(circle at 85% 10%, #c7e0ff 0%, transparent 33%),
        linear-gradient(140deg, #f7fbff 0%, #eef6ff 42%, #f8fbff 100%);
}

.big-title {
    font-size: 38px;
    font-weight: 800;
    color: var(--ink-strong);
    letter-spacing: 0.2px;
}

.subtitle {
    font-size: 18px;
    color: var(--ink-soft);
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #12385f;
    margin-top: 20px;
    margin-bottom: 10px;
}

.recommendation {
    background-color: #0f2747;
    padding: 16px;
    border-left: 6px solid #0b66c3;
    border-radius: 10px;
    color: #e8f1ff;
    margin-bottom: 10px;
}

.kpi-card {
    background-color: var(--surface);
    border: 1px solid #d7e8ff;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 8px 26px rgba(16, 57, 104, 0.08);
}

.step-pill {
    display: inline-block;
    background: #dcecff;
    color: #0b3d71;
    border: 1px solid #b9d4f8;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 5px 10px;
    margin-bottom: 6px;
}

.status-pill {
    border-radius: 12px;
    color: white;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}

.status-ok {
    background: linear-gradient(135deg, #1f8a5b 0%, #1b6f4a 100%);
}

.status-warn {
    background: linear-gradient(135deg, #ca8a04 0%, #a56f03 100%);
}

.status-bad {
    background: linear-gradient(135deg, #c0392b 0%, #9f2d22 100%);
}

.action-card {
    background: #ffffff;
    border: 1px solid #dbe9ff;
    border-left: 6px solid #0b66c3;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
    box-shadow: 0 8px 22px rgba(15, 39, 71, 0.07);
}

.action-priority-high {
    border-left-color: #c0392b;
}

.action-priority-medium {
    border-left-color: #ca8a04;
}

.action-priority-low {
    border-left-color: #1f8a5b;
}

.action-priority-info {
    border-left-color: #0b66c3;
}

.api-badge {
    border-radius: 10px;
    padding: 10px 12px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 8px;
    color: #ffffff;
}

.api-ready {
    background: linear-gradient(135deg, #1f8a5b 0%, #1b6f4a 100%);
}

.api-fallback {
    background: linear-gradient(135deg, #ca8a04 0%, #a56f03 100%);
}

.footer-note {
    text-align: center;
    color: #4a6b92;
    font-size: 13px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pretrained_models():
    return (
        joblib.load(PRETRAINED_DIR / "latency_model.pkl"),
        joblib.load(PRETRAINED_DIR / "throughput_model.pkl"),
        joblib.load(PRETRAINED_DIR / "qos_model.pkl"),
    )


@st.cache_data
def load_default_dataset(path):
    return pd.read_csv(path)


def render_step_label(step_text):
    st.markdown(f'<div class="step-pill">{step_text}</div>', unsafe_allow_html=True)


def apply_plot_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        font={"family": "Trebuchet MS, Verdana, sans-serif", "color": "#163a62"},
        title={"font": {"size": 18, "color": "#13375f"}},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def render_status_pills(avg_latency, avg_throughput, most_common_qos):
    latency_state = "status-ok"
    latency_text = f"Latency Risk: Healthy ({avg_latency:.1f} ms)"
    if avg_latency >= 70:
        latency_state = "status-bad"
        latency_text = f"Latency Risk: Critical ({avg_latency:.1f} ms)"
    elif avg_latency >= 50:
        latency_state = "status-warn"
        latency_text = f"Latency Risk: Watch ({avg_latency:.1f} ms)"

    throughput_state = "status-ok"
    throughput_text = f"Throughput Health: Strong ({avg_throughput:.1f} Mbps)"
    if avg_throughput < 60:
        throughput_state = "status-bad"
        throughput_text = f"Throughput Health: Weak ({avg_throughput:.1f} Mbps)"
    elif avg_throughput < 100:
        throughput_state = "status-warn"
        throughput_text = f"Throughput Health: Moderate ({avg_throughput:.1f} Mbps)"

    qos_state = "status-ok" if str(most_common_qos).lower() in ["good", "medium"] else "status-bad"
    qos_text = f"QoS Confidence: {most_common_qos}"

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="status-pill {latency_state}">{latency_text}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="status-pill {throughput_state}">{throughput_text}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="status-pill {qos_state}">{qos_text}</div>', unsafe_allow_html=True)


def render_strategy_cards(strategy_df, high_only=False):
    display_df = strategy_df.copy()
    if high_only:
        display_df = display_df[display_df["Priority"] == "High"]
        if display_df.empty:
            display_df = strategy_df.copy()

    for _, row in display_df.iterrows():
        priority = str(row["Priority"]).lower()
        priority_class = f"action-priority-{priority}" if priority in ["high", "medium", "low", "info"] else "action-priority-info"
        st.markdown(
            f"""
            <div class="action-card {priority_class}">
                <strong>{row['Priority']} Priority</strong><br/>
                <strong>Issue:</strong> {row['Issue']}<br/>
                <strong>Current:</strong> {row['Current']}<br/>
                <strong>Target:</strong> {row['Target']}<br/>
                <strong>Action:</strong> {row['PreciseAction']}<br/>
                <strong>Estimated Impact:</strong> {row['EstimatedImpact']}
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_manual_input_frame():
    """Collect manual prediction inputs in a cleaner main-panel layout."""
    st.markdown("#### Manual Parameter Entry")
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_strength = st.slider("Signal Strength (dBm)", -120, -60, -85, key="signal_strength")
        sinr = st.slider("SINR", 0, 30, 15, key="sinr")
        connected_users = st.slider("Connected Users", 10, 500, 100, key="connected_users")
    with col2:
        bandwidth_mhz = st.selectbox("Bandwidth (MHz)", [20, 40, 60, 80, 100], key="bandwidth_mhz")
        packet_loss = st.slider("Packet Loss (%)", 0.0, 5.0, 1.0, key="packet_loss")
    with col3:
        jitter = st.slider("Jitter (ms)", 1.0, 30.0, 5.0, key="jitter")
        mobility_speed = st.slider("Mobility Speed (km/h)", 0.0, 120.0, 20.0, key="mobility_speed")

    return pd.DataFrame({
        "signal_strength": [signal_strength],
        "sinr": [sinr],
        "connected_users": [connected_users],
        "bandwidth_mhz": [bandwidth_mhz],
        "packet_loss": [packet_loss],
        "jitter": [jitter],
        "mobility_speed": [mobility_speed],
    })


def render_individual_model_graphs(task_type, model_name, y_true, y_pred, class_labels=None, class_names=None):
    """Render model-specific diagnostics so each trained model has dedicated graphs."""
    st.markdown(f"##### Individual Graphs: {model_name}")

    if task_type == "regression":
        model_diag = regression_diagnostics(y_true, y_pred)
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.plotly_chart(apply_plot_theme(model_diag["scatter"]), use_container_width=True)
            st.plotly_chart(apply_plot_theme(model_diag["residual_vs_pred"]), use_container_width=True)
        with model_col2:
            st.plotly_chart(apply_plot_theme(model_diag["residual_hist"]), use_container_width=True)
            model_abs = np.abs(model_diag["table"]["Residual"])
            model_quant = model_abs.quantile([0.5, 0.75, 0.9, 0.95]).reset_index()
            model_quant.columns = ["Quantile", "AbsoluteResidual"]
            st.dataframe(model_quant, use_container_width=True)
        render_graph_interpretation_notes("diagnostics")
    else:
        cls_diag = classification_diagnostics(y_true, y_pred, class_labels=class_labels, display_labels=class_names)
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.plotly_chart(apply_plot_theme(cls_diag["confusion"]), use_container_width=True)
        with model_col2:
            st.plotly_chart(apply_plot_theme(cls_diag["class_accuracy"]), use_container_width=True)
        render_graph_interpretation_notes("diagnostics")


def render_metric_explanations(task_type):
    """Explain what each metric means for easier interpretation."""
    with st.expander("What each metric means", expanded=False):
        if task_type == "regression":
            metric_df = pd.DataFrame(
                [
                    {"Metric": "R2", "Meaning": "Explained variance ratio. Higher is better; 1.0 is perfect."},
                    {"Metric": "MAE", "Meaning": "Average absolute error in target units. Lower is better."},
                    {"Metric": "MAPE", "Meaning": "Average percentage error. Lower is better."},
                    {"Metric": "RMSE", "Meaning": "Square-root of mean squared error; penalizes large errors."},
                    {"Metric": "FitTimeSec", "Meaning": "Model training time in seconds."},
                    {"Metric": "PredictTimeSec", "Meaning": "Inference time on test set in seconds."},
                    {"Metric": "CVMean", "Meaning": "Average cross-validation score across folds."},
                    {"Metric": "CVStd", "Meaning": "Cross-validation variability; lower means more stable."},
                ]
            )
        else:
            metric_df = pd.DataFrame(
                [
                    {"Metric": "Accuracy", "Meaning": "Overall fraction of correct predictions."},
                    {"Metric": "F1", "Meaning": "Balance of precision and recall (weighted by class)."},
                    {"Metric": "Precision", "Meaning": "When model predicts a class, how often it is correct."},
                    {"Metric": "Recall", "Meaning": "How much of each true class the model recovers."},
                    {"Metric": "BalancedAccuracy", "Meaning": "Average recall across classes; helps with imbalance."},
                    {"Metric": "FitTimeSec", "Meaning": "Model training time in seconds."},
                    {"Metric": "PredictTimeSec", "Meaning": "Inference time on test set in seconds."},
                    {"Metric": "CVMean", "Meaning": "Average cross-validation score across folds."},
                    {"Metric": "CVStd", "Meaning": "Cross-validation variability; lower means more stable."},
                ]
            )
        st.dataframe(metric_df, use_container_width=True)


def render_graph_interpretation_notes(context):
    """Provide concise interpretation hints near graphs and heatmaps."""
    notes = {
        "correlation": [
            "High absolute correlation (close to 1) means stronger linear association.",
            "Negative correlation means feature increase tends to lower the target/output.",
            "Use correlation as signal, not causation; validate with model metrics.",
        ],
        "importance": [
            "Higher feature importance means model performance depends more on that feature.",
            "Near-zero importance features can often be deprioritized.",
            "Compare with permutation/SHAP importance for robustness.",
        ],
        "diagnostics": [
            "Tighter Actual-vs-Predicted alignment indicates better predictive fit.",
            "Residuals centered around zero indicate lower systematic bias.",
            "Wide residual spread suggests instability or missing signal.",
        ],
    }

    selected = notes.get(context, [])
    if selected:
        st.caption("Interpretation guide")
        for item in selected:
            st.markdown(f"- {item}")


def render_api_status_badge():
    """Show Groq API configuration and latest recommendation-source status."""
    key_ready = has_groq_api_key()
    last_mode = st.session_state.get("last_recommendation_mode")

    if key_ready:
        text = "Groq API: Configured"
        if last_mode == "ai":
            text += " | Last recommendation: AI"
        elif last_mode == "fallback":
            text += " | Last recommendation: Fallback"
        st.sidebar.markdown(f'<div class="api-badge api-ready">{text}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            '<div class="api-badge api-fallback">Groq API: Not configured | Using fallback recommendations</div>',
            unsafe_allow_html=True,
        )


def render_model_correlation_analysis(model_name, x_features, y_pred, task_type, key_prefix):
    """Run per-model feature correlation analysis using model outputs."""
    with st.expander(f"Correlation analysis for {model_name}", expanded=False):
        corr_df = x_features.copy()
        if task_type == "classification":
            y_numeric = pd.Series(y_pred).astype("category").cat.codes
            corr_df["model_output"] = y_numeric
        else:
            corr_df["model_output"] = pd.Series(y_pred)

        corr_tab1, corr_tab2, corr_tab3 = st.tabs(["Model Heatmap", "Feature vs Model Output", "Model Scatter Matrix"])

        with corr_tab1:
            fig_h = render_correlation_heatmap(corr_df)
            if fig_h is not None:
                st.plotly_chart(apply_plot_theme(fig_h), use_container_width=True)
                render_graph_interpretation_notes("correlation")

        with corr_tab2:
            fig_b = render_target_correlation_bar(corr_df, target_col="model_output", top_n=12)
            if fig_b is not None:
                st.plotly_chart(apply_plot_theme(fig_b), use_container_width=True)
                render_graph_interpretation_notes("correlation")

        with corr_tab3:
            scatter_dims = st.multiselect(
                "Select features for model scatter matrix",
                options=list(x_features.columns),
                default=list(x_features.columns)[: min(4, len(x_features.columns))],
                key=f"{key_prefix}_model_scatter_dims",
            )
            fig_s = render_scatter_matrix(corr_df, selected_features=scatter_dims, color_col="model_output")
            if fig_s is not None:
                st.plotly_chart(apply_plot_theme(fig_s), use_container_width=True)
                render_graph_interpretation_notes("correlation")

st.markdown('<div class="big-title">AI-Driven 5G Network Optimization System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A Machine Learning prototype for predicting latency, throughput, and QoS classification in 5G networks.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">System Purpose</div>', unsafe_allow_html=True)

st.write("""
This application demonstrates how Machine Learning can support smarter 5G network management.

In a real network, conditions change constantly. The number of users, signal quality, packet loss,
mobility, and available bandwidth can all affect the quality of service. Instead of waiting until
the network becomes slow or unstable, this system tries to predict performance in advance.

The prototype takes 5G network parameters as input, predicts latency and throughput, classifies the
network quality, and provides simple optimization recommendations.
""")

predict_tab, ml_lab_tab, run_compare_tab = st.tabs(
    [
        "Performance Prediction",
        "ML Lab: Train, Compare, Select Features",
        "Run Comparison",
    ]
)

st.sidebar.markdown("### Experience Mode")
experience_mode = st.sidebar.radio(
    "Choose your workflow",
    ["Operator Mode", "Research Mode"],
    help="Operator Mode keeps UI concise for field operations. Research Mode enables full diagnostics and explainability.",
)
render_api_status_badge()

required_columns = [
    "signal_strength",
    "sinr",
    "connected_users",
    "bandwidth_mhz",
    "packet_loss",
    "jitter",
    "mobility_speed",
]


with predict_tab:
    render_step_label("Step 1 - Input")
    latency_model, throughput_model, qos_model = load_pretrained_models()

    input_method = st.radio(
        "Choose input method",
        ["Manual input", "Upload file"],
        horizontal=True,
        key="predict_input_method",
    )

    input_data = None
    if input_method == "Manual input":
        input_data = get_manual_input_frame()
    else:
        st.markdown("#### Upload Parameter File")
        uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"], key="predict_uploader")
        st.caption(
            "Required columns: signal_strength, sinr, connected_users, bandwidth_mhz, packet_loss, jitter, mobility_speed"
        )
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                missing_columns = [col for col in required_columns if col not in input_data.columns]
                if missing_columns:
                    st.error("Uploaded file is missing required columns: " + ", ".join(missing_columns))
                    input_data = None
                else:
                    input_data = input_data[required_columns]
            except Exception:
                st.error("Could not read uploaded file. Please upload a valid CSV/TXT file.")
                input_data = None
        else:
            st.info("Upload a file to run predictions for multiple scenarios.")

    if input_data is not None:
        with st.expander("Review Input Summary", expanded=True):
            st.dataframe(input_data, use_container_width=True)

    if st.button("Predict Network Performance", key="predict_button", disabled=input_data is None):
        render_step_label("Step 2 - Prediction")
        latency_predictions = latency_model.predict(input_data)
        throughput_predictions = throughput_model.predict(input_data)
        qos_predictions = qos_model.predict(input_data)

        results = input_data.copy()
        results["predicted_latency_ms"] = latency_predictions
        results["predicted_throughput_mbps"] = throughput_predictions
        results["qos_classification"] = qos_predictions

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

        avg_latency = results["predicted_latency_ms"].mean()
        avg_throughput = results["predicted_throughput_mbps"].mean()
        most_common_qos = results["qos_classification"].mode()[0]

        render_step_label("Step 3 - Status")
        render_status_pills(avg_latency, avg_throughput, most_common_qos)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Latency", f"{avg_latency:.2f} ms")
        col2.metric("Average Throughput", f"{avg_throughput:.2f} Mbps")
        col3.metric("Most Common QoS", most_common_qos)

        st.dataframe(results, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            latency_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_latency,
                title={"text": "Average Latency Level"},
                gauge={
                    "axis": {"range": [0, 120]},
                    "bar": {"color": "#0b66c3"},
                    "steps": [
                        {"range": [0, 40], "color": "#d9f2ff"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 120], "color": "#f8d7da"},
                    ],
                },
            ))
            latency_fig = apply_plot_theme(latency_fig)
            st.plotly_chart(latency_fig, use_container_width=True)

        with chart_col2:
            throughput_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_throughput,
                title={"text": "Average Throughput Level"},
                gauge={
                    "axis": {"range": [0, 250]},
                    "bar": {"color": "#0b66c3"},
                    "steps": [
                        {"range": [0, 60], "color": "#f8d7da"},
                        {"range": [60, 100], "color": "#fff3cd"},
                        {"range": [100, 250], "color": "#d9f2ff"},
                    ],
                },
            ))
            throughput_fig = apply_plot_theme(throughput_fig)
            st.plotly_chart(throughput_fig, use_container_width=True)

        render_step_label("Step 4 - Action Plan")
        st.markdown('<div class="section-title">Precise Optimization Strategy Plan</div>', unsafe_allow_html=True)
        strategy_df, strategy_mode = generate_ai_optimization_plan(input_data, results)
        st.session_state["last_recommendation_mode"] = strategy_mode
        render_api_status_badge()
        if strategy_mode == "ai":
            st.success("Optimization plan generated with Groq AI.")
        else:
            st.info("Using fallback rule-based strategy (Groq key unavailable or request failed).")

        if experience_mode == "Operator Mode":
            high_only = st.toggle("Show high-priority actions only", value=True, key="strategy_high_only")
            render_strategy_cards(strategy_df, high_only=high_only)
        else:
            render_strategy_cards(strategy_df, high_only=False)
            with st.expander("Strategy table (research view)", expanded=False):
                st.dataframe(strategy_df, use_container_width=True)

        strategy_csv = strategy_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Optimization Plan",
            data=strategy_csv,
            file_name="5g_optimization_plan.csv",
            mime="text/csv",
        )

        strategy_pdf = build_pdf_report(
            title="5G Optimization Recommendation Report",
            tables={"optimization_plan": strategy_df, "prediction_results": results},
            figures={"latency_gauge": latency_fig, "throughput_gauge": throughput_fig},
        )
        st.download_button(
            label="Download Optimization Report (PDF)",
            data=strategy_pdf,
            file_name="5g_optimization_report.pdf",
            mime="application/pdf",
        )

        st.markdown('<div class="section-title">Social and Technical Impact</div>', unsafe_allow_html=True)
        st.write(
            "Reliable 5G networks are important for modern society because they support healthcare systems, "
            "online education, emergency communication, smart cities, industrial automation, autonomous "
            "transportation, and IoT services."
        )

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="5g_prediction_results.csv",
            mime="text/csv",
        )


with ml_lab_tab:
    render_step_label("Step 1 - Dataset")
    st.markdown('<div class="section-title">Model Training and Feature Engineering Lab</div>', unsafe_allow_html=True)
    st.write(
        "Train and compare multiple models (linear regression, tree models, SVM, KNN, neural networks), "
        "run feature selection, inspect correlations, and generate advanced diagnostics."
    )

    dataset_path = DATA_DIR / "5g_network_data.csv"
    uploaded_dataset = st.file_uploader("Upload a dataset for training (CSV)", type=["csv"], key="training_uploader")

    if uploaded_dataset is not None:
        dataset = pd.read_csv(uploaded_dataset)
    elif dataset_path.exists():
        dataset = load_default_dataset(str(dataset_path))
    else:
        st.warning("No dataset found. Upload a CSV file to continue.")
        st.stop()

    with st.expander("Dataset preview", expanded=True):
        st.dataframe(dataset.head(20), use_container_width=True)

    available_targets = [col for col in dataset.columns if col not in required_columns] or list(dataset.columns)
    default_target = "latency" if "latency" in dataset.columns else available_targets[0]

    col_a, col_b = st.columns(2)
    with col_a:
        target_col = st.selectbox("Target column", options=available_targets, index=available_targets.index(default_target))
    with col_b:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    excluded_cols = {target_col}
    feature_options = [col for col in dataset.columns if col not in excluded_cols]
    default_features = [col for col in required_columns if col in feature_options] or feature_options
    selected_features = st.multiselect("Feature columns", options=feature_options, default=default_features)

    if not selected_features:
        st.warning("Select at least one feature column.")
        st.stop()

    # Detect whether the selected target should be handled as regression or classification.
    # QoS class columns can appear either as text labels (Good/Poor) or small numeric codes.
    target_name = str(target_col).lower()
    target_series = dataset[target_col]
    # Pandas extension dtypes, such as string[python] / string[pyarrow], cannot be
    # passed safely to numpy.issubdtype. Use pandas dtype helpers instead.
    is_text_target = (
        pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
        or pd.api.types.is_categorical_dtype(target_series)
    )
    is_qos_like_target = any(token in target_name for token in ["qos", "class", "category", "label", "quality"])
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
    is_float_target = pd.api.types.is_float_dtype(target_series)
    is_small_discrete_target = (
        target_series.nunique(dropna=True) <= 10
        and (not is_numeric_target or not is_float_target)
    )
    task_type = "classification" if (is_text_target or is_qos_like_target or is_small_discrete_target) else "regression"

    st.info(f"Detected task type: {task_type.title()}")

    render_step_label("Step 2 - Correlation and Feature Design")
    corr_tab1, corr_tab2, corr_tab3 = st.tabs(["Heatmap", "Target Correlation", "Scatter Matrix"])

    with corr_tab1:
        corr_fig = render_correlation_heatmap(dataset)
        if corr_fig is not None:
            corr_fig = apply_plot_theme(corr_fig)
            st.plotly_chart(corr_fig, use_container_width=True)
            render_graph_interpretation_notes("correlation")
        else:
            st.warning("No numeric columns available to compute a correlation matrix.")

    with corr_tab2:
        target_corr_fig = render_target_correlation_bar(dataset, target_col=target_col, top_n=12)
        if target_corr_fig is not None:
            st.plotly_chart(apply_plot_theme(target_corr_fig), use_container_width=True)
            render_graph_interpretation_notes("correlation")
        else:
            st.info("Target correlation bar is available when selected target is numeric.")

    with corr_tab3:
        scatter_dims = st.multiselect(
            "Select features for scatter matrix (2-6)",
            options=selected_features,
            default=selected_features[: min(4, len(selected_features))],
            key="scatter_dims",
        )
        scatter_fig = render_scatter_matrix(dataset, selected_features=scatter_dims, color_col=target_col)
        if scatter_fig is not None:
            st.plotly_chart(apply_plot_theme(scatter_fig), use_container_width=True)
            render_graph_interpretation_notes("correlation")
        else:
            st.info("Select at least two numeric features for scatter matrix.")

    st.markdown("Feature selection")
    feature_method = st.selectbox("Selection method", ["Mutual Information", "F-Score", "RFE"])
    k_features = st.slider("Number of selected features", 1, len(selected_features), min(4, len(selected_features)))

    st.markdown("Research options")
    default_cv = True if experience_mode == "Research Mode" else False
    run_cv = st.checkbox("Run cross-validation for each model", value=default_cv)
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5) if run_cv else 5
    run_stability = st.checkbox("Run model stability analysis across random seeds", value=(experience_mode == "Research Mode"))
    stability_seeds = st.slider("Stability seeds", 3, 12, 6) if run_stability else 6
    run_learning_curves = st.checkbox("Generate learning curves for top models", value=False)
    learning_curve_top_n = st.slider("Top models for learning curves", 1, 6, 3) if run_learning_curves else 3
    run_permutation_importance = st.checkbox("Compute permutation importance for best model", value=(experience_mode == "Research Mode"))
    default_shap = True if experience_mode == "Research Mode" else False
    generate_shap = st.checkbox("Generate SHAP explainability plot (can be slower)", value=default_shap)
    if generate_shap and not SHAP_AVAILABLE:
        st.warning("SHAP package not available. Install dependencies from requirements.txt.")
        generate_shap = False

    if st.button("Train and Compare Models", key="train_compare"):
        render_step_label("Step 3 - Train and Evaluate")
        train_df = dataset[selected_features + [target_col]].dropna().copy()
        x = train_df[selected_features]
        y = train_df[target_col]
        export_tables = {}
        export_figures = {}

        label_encoder = None
        if task_type == "classification" and y.dtype == "object":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        stratify_target = y if task_type == "classification" else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_target,
        )

        selected_after_fs = run_feature_selection(
            x_train,
            y_train,
            x_columns=selected_features,
            task_type=task_type,
            method=feature_method,
            k_features=k_features,
        )

        x_train = x_train[selected_after_fs]
        x_test = x_test[selected_after_fs]
        x_full_selected = train_df[selected_after_fs]

        st.markdown("Selected features")
        st.write(", ".join(selected_after_fs))

        model_registry = get_model_registry(task_type)
        results_df, best_model_name, best_model, _all_trained_models, model_predictions = evaluate_models(
            model_registry,
            x_train,
            x_test,
            y_train,
            y_test,
            task_type,
            run_cv=run_cv,
            cv_folds=cv_folds,
        )

        st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)
        export_tables["leaderboard"] = results_df.copy()
        render_metric_explanations(task_type)

        leaderboard_csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Leaderboard",
            data=leaderboard_csv,
            file_name=f"leaderboard_{target_col}.csv",
            mime="text/csv",
        )

        metric_col = "R2" if task_type == "regression" else "F1"
        bar_fig = px.bar(
            results_df,
            x="Model",
            y=metric_col,
            color=metric_col,
            title=f"Model Comparison by {metric_col}",
            text_auto=".3f",
        )
        bar_fig = apply_plot_theme(bar_fig)
        st.plotly_chart(bar_fig, use_container_width=True)
        export_figures["leaderboard"] = bar_fig

        if run_stability:
            st.markdown('<div class="section-title">Model Stability Across Random Seeds</div>', unsafe_allow_html=True)
            stability_df = compute_stability_analysis(
                model_registry=model_registry,
                x_full=x_full_selected,
                y_full=y,
                task_type=task_type,
                test_size=test_size,
                num_seeds=stability_seeds,
            )
            st.dataframe(stability_df, use_container_width=True)
            stability_fig = px.bar(
                stability_df,
                x="Model",
                y="Mean",
                error_y="Std",
                color="Mean",
                title="Model Stability (Mean +/- Std across seeds)",
                text_auto=".3f",
            )
            stability_fig = apply_plot_theme(stability_fig)
            st.plotly_chart(stability_fig, use_container_width=True)
            export_tables["stability"] = stability_df.copy()
            export_figures["stability"] = stability_fig

        if run_learning_curves:
            st.markdown('<div class="section-title">Learning Curves (Top Models)</div>', unsafe_allow_html=True)
            top_model_names = results_df["Model"].head(learning_curve_top_n).tolist()
            lc_df = compute_learning_curve_data(
                model_registry=model_registry,
                x_full=x_full_selected,
                y_full=y,
                model_names=top_model_names,
                task_type=task_type,
                cv_folds=cv_folds if run_cv else 5,
            )
            if lc_df is not None and not lc_df.empty:
                st.dataframe(lc_df, use_container_width=True)
                lc_plot_df = lc_df.melt(
                    id_vars=["Model", "TrainSize"],
                    value_vars=["TrainMean", "ValidationMean"],
                    var_name="Curve",
                    value_name="Score",
                )
                learning_fig = px.line(
                    lc_plot_df,
                    x="TrainSize",
                    y="Score",
                    color="Model",
                    line_dash="Curve",
                    markers=True,
                    title="Learning Curves by Model",
                )
                learning_fig = apply_plot_theme(learning_fig)
                st.plotly_chart(learning_fig, use_container_width=True)
                export_tables["learning_curves"] = lc_df.copy()
                export_figures["learning_curves"] = learning_fig
            else:
                st.info("Learning curve data could not be generated for current selection.")

        st.markdown('<div class="section-title">Individual Model Explorer</div>', unsafe_allow_html=True)
        available_model_names = list(model_predictions.keys())
        render_all_models = st.toggle(
            "Render graphs for every trained model",
            value=False,
            key="render_all_model_graphs",
            help="Can be heavy when many models are included.",
        )
        run_corr_per_model = st.toggle(
            "Enable feature correlation analysis for each model",
            value=False,
            key="run_corr_per_model",
            help="Adds model-specific correlation plots between features and model outputs.",
        )

        if task_type == "classification":
            class_labels = list(label_encoder.transform(label_encoder.classes_)) if label_encoder is not None else sorted(list(np.unique(y_test)))
            class_names = list(label_encoder.classes_) if label_encoder is not None else [str(c) for c in class_labels]
        else:
            class_labels = None
            class_names = None

        if render_all_models:
            for model_name in available_model_names:
                with st.expander(f"Model: {model_name}", expanded=False):
                    render_individual_model_graphs(
                        task_type=task_type,
                        model_name=model_name,
                        y_true=y_test,
                        y_pred=model_predictions[model_name],
                        class_labels=class_labels,
                        class_names=class_names,
                    )
                    if run_corr_per_model:
                        render_model_correlation_analysis(
                            model_name=model_name,
                            x_features=x_test,
                            y_pred=model_predictions[model_name],
                            task_type=task_type,
                            key_prefix=f"all_{model_name}",
                        )
        else:
            selected_model_for_graph = st.selectbox(
                "Select model to inspect",
                options=available_model_names,
                index=0,
                key="selected_model_graph",
            )
            render_individual_model_graphs(
                task_type=task_type,
                model_name=selected_model_for_graph,
                y_true=y_test,
                y_pred=model_predictions[selected_model_for_graph],
                class_labels=class_labels,
                class_names=class_names,
            )
            if run_corr_per_model:
                render_model_correlation_analysis(
                    model_name=selected_model_for_graph,
                    x_features=x_test,
                    y_pred=model_predictions[selected_model_for_graph],
                    task_type=task_type,
                    key_prefix="single_model",
                )

        st.markdown('<div class="section-title">Best Classical Model</div>', unsafe_allow_html=True)
        st.write(f"Best model: {best_model_name}")

        y_pred_best = best_model.predict(x_test)
        diagnostics_export_df = None

        estimator = best_model.named_steps.get("model") if isinstance(best_model, Pipeline) else best_model
        if hasattr(estimator, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": selected_after_fs,
                "Importance": estimator.feature_importances_,
            }).sort_values("Importance", ascending=False)
            importance_fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
            importance_fig = apply_plot_theme(importance_fig)
            st.plotly_chart(importance_fig, use_container_width=True)
            render_graph_interpretation_notes("importance")
            export_tables["feature_importance"] = importance_df.copy()
            export_figures["feature_importance"] = importance_fig
        elif hasattr(estimator, "coef_"):
            coefficients = estimator.coef_[0] if np.ndim(estimator.coef_) > 1 else estimator.coef_
            coef_df = pd.DataFrame({
                "Feature": selected_after_fs,
                "Coefficient": coefficients,
            }).sort_values("Coefficient", key=np.abs, ascending=False)
            coef_fig = px.bar(coef_df, x="Feature", y="Coefficient", title="Model Coefficients")
            coef_fig = apply_plot_theme(coef_fig)
            st.plotly_chart(coef_fig, use_container_width=True)
            render_graph_interpretation_notes("importance")
            export_tables["coefficients"] = coef_df.copy()
            export_figures["coefficients"] = coef_fig

        if run_permutation_importance:
            st.markdown('<div class="section-title">Permutation Importance (Best Model)</div>', unsafe_allow_html=True)
            perm_df = compute_permutation_importance(
                best_model=best_model,
                x_test=x_test,
                y_test=y_test,
                task_type=task_type,
                n_repeats=12,
            )
            st.dataframe(perm_df, use_container_width=True)
            perm_fig = px.bar(
                perm_df.head(20),
                x="Feature",
                y="ImportanceMean",
                error_y="ImportanceStd",
                color="ImportanceMean",
                title="Permutation Importance (Top 20)",
                text_auto=".4f",
            )
            perm_fig = apply_plot_theme(perm_fig)
            st.plotly_chart(perm_fig, use_container_width=True)
            export_tables["permutation_importance"] = perm_df.copy()
            export_figures["permutation_importance"] = perm_fig

        show_diagnostics = True if experience_mode == "Research Mode" else st.toggle(
            "Show advanced diagnostics",
            value=False,
            key="operator_show_diagnostics",
        )

        if task_type == "regression" and show_diagnostics:
            st.markdown('<div class="section-title">Regression Diagnostics</div>', unsafe_allow_html=True)
            reg_diag = regression_diagnostics(y_test, y_pred_best)
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                reg_diag["scatter"] = apply_plot_theme(reg_diag["scatter"])
                reg_diag["residual_vs_pred"] = apply_plot_theme(reg_diag["residual_vs_pred"])
                st.plotly_chart(reg_diag["scatter"], use_container_width=True)
                st.plotly_chart(reg_diag["residual_vs_pred"], use_container_width=True)
            with diag_col2:
                reg_diag["residual_hist"] = apply_plot_theme(reg_diag["residual_hist"])
                st.plotly_chart(reg_diag["residual_hist"], use_container_width=True)
                error_quantiles = reg_diag["table"]["Residual"].abs().quantile([0.5, 0.75, 0.9, 0.95]).reset_index()
                error_quantiles.columns = ["Quantile", "AbsoluteResidual"]
                st.dataframe(error_quantiles, use_container_width=True)
            render_graph_interpretation_notes("diagnostics")

            diag_csv = reg_diag["table"].to_csv(index=False).encode("utf-8")
            diagnostics_export_df = reg_diag["table"]
            export_tables["regression_diagnostics"] = reg_diag["table"].copy()
            export_figures["reg_actual_vs_pred"] = reg_diag["scatter"]
            export_figures["reg_residual_vs_pred"] = reg_diag["residual_vs_pred"]
            export_figures["reg_residual_hist"] = reg_diag["residual_hist"]
            st.download_button(
                label="Download Regression Diagnostics",
                data=diag_csv,
                file_name=f"diagnostics_{target_col}.csv",
                mime="text/csv",
            )
        elif task_type == "classification" and show_diagnostics:
            st.markdown('<div class="section-title">Classification Diagnostics</div>', unsafe_allow_html=True)
            class_labels = list(label_encoder.transform(label_encoder.classes_)) if label_encoder is not None else sorted(list(np.unique(y_test)))
            class_names = list(label_encoder.classes_) if label_encoder is not None else [str(c) for c in class_labels]
            cls_diag = classification_diagnostics(y_test, y_pred_best, class_labels, display_labels=class_names)
            cls_col1, cls_col2 = st.columns(2)
            with cls_col1:
                cls_diag["confusion"] = apply_plot_theme(cls_diag["confusion"])
                st.plotly_chart(cls_diag["confusion"], use_container_width=True)
            with cls_col2:
                cls_diag["class_accuracy"] = apply_plot_theme(cls_diag["class_accuracy"])
                st.plotly_chart(cls_diag["class_accuracy"], use_container_width=True)
            render_graph_interpretation_notes("diagnostics")

            roc_auc_df = multiclass_roc_auc(best_model, x_test, y_test)
            if roc_auc_df is not None:
                st.markdown("ROC-AUC indicators")
                st.dataframe(roc_auc_df, use_container_width=True)

            cm_csv = cls_diag["confusion_table"].to_csv().encode("utf-8")
            diagnostics_export_df = cls_diag["confusion_table"].copy()
            export_tables["confusion_matrix"] = cls_diag["confusion_table"].copy()
            export_figures["confusion_matrix"] = cls_diag["confusion"]
            export_figures["per_class_accuracy"] = cls_diag["class_accuracy"]
            st.download_button(
                label="Download Confusion Matrix",
                data=cm_csv,
                file_name=f"confusion_matrix_{target_col}.csv",
                mime="text/csv",
            )

            calib_df, calib_summary_df = compute_calibration_and_confidence(best_model, x_test, y_test)
            if calib_df is not None and not calib_df.empty:
                st.markdown('<div class="section-title">Calibration and Confidence Analysis</div>', unsafe_allow_html=True)
                st.dataframe(calib_summary_df, use_container_width=True)
                calib_line_fig = px.line(
                    calib_df.melt(id_vars=["Bin", "Count"], value_vars=["Accuracy", "Confidence"], var_name="Metric", value_name="Value"),
                    x="Bin",
                    y="Value",
                    color="Metric",
                    markers=True,
                    title="Calibration Curve (Accuracy vs Confidence by Bin)",
                )
                calib_line_fig = apply_plot_theme(calib_line_fig)
                st.plotly_chart(calib_line_fig, use_container_width=True)

                conf_hist_fig = px.bar(
                    calib_df,
                    x="Bin",
                    y="Count",
                    title="Prediction Confidence Histogram",
                    color="Count",
                    text_auto=True,
                )
                conf_hist_fig = apply_plot_theme(conf_hist_fig)
                st.plotly_chart(conf_hist_fig, use_container_width=True)

                export_tables["calibration_bins"] = calib_df.copy()
                export_tables["calibration_summary"] = calib_summary_df.copy()
                export_figures["calibration_curve"] = calib_line_fig
                export_figures["confidence_histogram"] = conf_hist_fig

        if generate_shap and SHAP_AVAILABLE and show_diagnostics:
            st.markdown('<div class="section-title">Model Explainability (SHAP)</div>', unsafe_allow_html=True)
            shap_importance = compute_shap_importance(best_model, x_test)
            shap_fig = build_shap_importance_plot(shap_importance)
            if shap_fig is not None:
                shap_fig = apply_plot_theme(shap_fig)
                st.plotly_chart(shap_fig, use_container_width=True)
                shap_csv = shap_importance.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download SHAP Importance",
                    data=shap_csv,
                    file_name=f"shap_importance_{target_col}.csv",
                    mime="text/csv",
                )
                export_tables["shap_importance"] = shap_importance.copy()
                export_figures["shap_importance"] = shap_fig
            else:
                st.warning("SHAP could not generate results for this model. Try a tree-based or linear model.")

        experiment_summary = pd.DataFrame(
            [
                {
                    "Target": target_col,
                    "TaskType": task_type,
                    "FeatureMethod": feature_method,
                    "SelectedFeatureCount": len(selected_after_fs),
                    "RunCrossValidation": run_cv,
                    "CVFolds": cv_folds if run_cv else 0,
                    "BestModel": best_model_name,
                    "TestSize": test_size,
                }
            ]
        )
        with st.expander("Experiment summary", expanded=(experience_mode == "Research Mode")):
            st.dataframe(experiment_summary, use_container_width=True)
        export_tables["experiment_summary"] = experiment_summary.copy()

        summary_csv = experiment_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Experiment Summary",
            data=summary_csv,
            file_name=f"experiment_summary_{target_col}.csv",
            mime="text/csv",
        )

        if st.button("Save Experiment Package to artifacts/reports", key="save_experiment_package"):
            reports_dir = BASE_DIR / "artifacts" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = reports_dir / f"run_{target_col}_{run_id}"
            tables_dir = run_dir / "tables"
            figures_dir = run_dir / "figures"
            tables_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"SelectedFeature": selected_after_fs}).to_csv(
                tables_dir / "selected_features.csv", index=False
            )

            for table_name, table_df in export_tables.items():
                table_df.to_csv(tables_dir / f"{table_name}.csv", index=False)

            for fig_name, fig_obj in export_figures.items():
                fig_obj.write_html(str(figures_dir / f"{fig_name}.html"), include_plotlyjs="cdn")

            if diagnostics_export_df is not None and "diagnostics" not in export_tables:
                diagnostics_export_df.to_csv(tables_dir / "diagnostics.csv", index=False)

            st.success(f"Saved full experiment bundle to {run_dir}")

        experiment_pdf = build_pdf_report(
            title=f"Experiment Report - {target_col}",
            tables=export_tables,
            figures=export_figures,
        )
        st.download_button(
            label="Download Experiment Report (PDF)",
            data=experiment_pdf,
            file_name=f"experiment_report_{target_col}.pdf",
            mime="application/pdf",
        )

        model_bytes = io.BytesIO()
        joblib.dump(best_model, model_bytes)
        st.download_button(
            label="Download Best Classical Model",
            data=model_bytes.getvalue(),
            file_name=f"best_{target_col}_model.pkl",
            mime="application/octet-stream",
        )


with run_compare_tab:
    render_step_label("Step 1 - Select Runs")
    st.markdown('<div class="section-title">Experiment Run Comparison</div>', unsafe_allow_html=True)
    st.write(
        "Compare saved experiment runs from artifacts/reports to analyze performance drift, "
        "best-model consistency, and model ranking changes over time."
    )

    reports_dir = BASE_DIR / "artifacts" / "reports"
    run_dirs = discover_run_directories(reports_dir)

    if not run_dirs:
        st.info("No saved runs found yet. Use 'Save Experiment Package to artifacts/reports' in ML Lab first.")
    else:
        run_name_map = {p.name: p for p in run_dirs}
        selected_run_names = st.multiselect(
            "Select runs to compare",
            options=list(run_name_map.keys()),
            default=list(run_name_map.keys())[: min(4, len(run_name_map))],
        )

        if not selected_run_names:
            st.warning("Select at least one run to compare.")
        else:
            selected_dirs = [run_name_map[name] for name in selected_run_names]

            leaderboard_frames = []
            summary_frames = []

            for run_dir in selected_dirs:
                lb_df = load_run_table(run_dir, "leaderboard")
                if lb_df is not None and not lb_df.empty:
                    lb_df = lb_df.copy()
                    lb_df["Run"] = run_dir.name
                    leaderboard_frames.append(lb_df)

                summary_df = load_run_table(run_dir, "experiment_summary")
                if summary_df is not None and not summary_df.empty:
                    summary_df = summary_df.copy()
                    summary_df["Run"] = run_dir.name
                    summary_frames.append(summary_df)

            if not leaderboard_frames:
                st.error("Selected runs do not contain leaderboard tables.")
            else:
                render_step_label("Step 2 - Compare Metrics")
                combined_leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
                st.dataframe(combined_leaderboard, use_container_width=True)

                inferred = infer_primary_metric(combined_leaderboard)
                numeric_cols = [
                    c
                    for c in combined_leaderboard.select_dtypes(include=[np.number]).columns
                    if c not in ["TestSize", "CVFolds"]
                ]
                default_metric = inferred if inferred in numeric_cols else (numeric_cols[0] if numeric_cols else None)

                if default_metric is None:
                    st.warning("No numeric metric columns found for comparison.")
                else:
                    metric_col = st.selectbox("Metric for comparison", options=numeric_cols, index=numeric_cols.index(default_metric))

                    top_n_models = st.slider("Top models per run", 1, 10, 5)
                    comparison_rows = []
                    best_rows = []

                    for run_dir in selected_dirs:
                        run_name = run_dir.name
                        run_lb = combined_leaderboard[combined_leaderboard["Run"] == run_name].copy()
                        if metric_col not in run_lb.columns:
                            continue

                        run_lb = run_lb.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False)
                        comparison_rows.append(run_lb.head(top_n_models))

                        best = best_model_by_metric(run_lb, metric_col)
                        if best is not None:
                            best_rows.append(
                                {
                                    "Run": run_name,
                                    "BestModel": best["Model"],
                                    "Metric": best["Metric"],
                                    "Score": best["Score"],
                                }
                            )

                    if comparison_rows:
                        top_models_df = pd.concat(comparison_rows, ignore_index=True)

                        st.markdown("#### Model Ranking by Run")
                        ranking_fig = px.bar(
                            top_models_df,
                            x="Model",
                            y=metric_col,
                            color="Run",
                            barmode="group",
                            title=f"Top {top_n_models} Models per Run by {metric_col}",
                            text_auto=".3f",
                        )
                        st.plotly_chart(apply_plot_theme(ranking_fig), use_container_width=True)

                        st.markdown("#### Metric Distribution Across Runs")
                        dist_fig = px.box(
                            combined_leaderboard,
                            x="Run",
                            y=metric_col,
                            color="Run",
                            points="all",
                            title=f"{metric_col} Distribution by Run",
                        )
                        st.plotly_chart(apply_plot_theme(dist_fig), use_container_width=True)

                    if best_rows:
                        best_df = pd.DataFrame(best_rows).sort_values("Score", ascending=False)
                        st.markdown("#### Best Model Snapshot per Run")
                        st.dataframe(best_df, use_container_width=True)

                        best_fig = px.bar(
                            best_df,
                            x="Run",
                            y="Score",
                            color="BestModel",
                            title=f"Best Model Score per Run ({metric_col})",
                            text_auto=".3f",
                        )
                        st.plotly_chart(apply_plot_theme(best_fig), use_container_width=True)

                        best_csv = best_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Run Comparison Summary",
                            data=best_csv,
                            file_name=f"run_comparison_{metric_col}.csv",
                            mime="text/csv",
                        )

                        comparison_tables = {
                            "best_model_snapshot": best_df,
                            "combined_leaderboard": combined_leaderboard,
                        }
                        comparison_figures = {
                            "ranking_by_run": ranking_fig,
                            "distribution_by_run": dist_fig,
                            "best_model_scores": best_fig,
                        }
                        comparison_pdf = build_pdf_report(
                            title=f"Run Comparison Report - {metric_col}",
                            tables=comparison_tables,
                            figures=comparison_figures,
                        )
                        st.download_button(
                            label="Download Run Comparison (PDF)",
                            data=comparison_pdf,
                            file_name=f"run_comparison_{metric_col}.pdf",
                            mime="application/pdf",
                        )

                if summary_frames:
                    render_step_label("Step 3 - Experiment Setup Audit")
                    summary_combined = pd.concat(summary_frames, ignore_index=True)
                    st.markdown("#### Experiment Setup Summary")
                    st.dataframe(summary_combined, use_container_width=True)

st.markdown("---")
st.markdown('<div class="footer-note">MB 2026</div>', unsafe_allow_html=True)
