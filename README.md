# 5G ML Prototype

Streamlit-based 5G machine learning prototype with two workflows:

1. Performance Prediction (pretrained inference)
2. ML Lab (train/compare models, feature selection, correlation analysis, advanced diagnostics)

Additional capabilities:
- Precise optimization strategy generation with target-based actions
- SHAP explainability plots (optional)
- Research-grade diagnostics (cross-validation, timing, residual/confusion analysis, ROC-AUC indicators)

## Project Layout

```text
.
├── app.py
├── requirements.txt
├── analysis/
│   ├── correlation_graphs.py
│   ├── explainability.py
│   ├── feature_selection.py
│   ├── optimization_strategies.py
│   └── research_plots.py
├── models/
│   ├── model_registry.py
│   ├── linear_regression_model.py
│   ├── logistic_regression_model.py
│   ├── random_forest_regressor_model.py
│   ├── random_forest_classifier_model.py
│   ├── gradient_boosting_regressor_model.py
│   ├── gradient_boosting_classifier_model.py
│   ├── svr_model.py
│   ├── svc_model.py
│   ├── knn_regressor_model.py
│   ├── knn_classifier_model.py
│   ├── mlp_regressor_model.py
│   └── mlp_classifier_model.py
├── data/
│   ├── 5g_network_data.csv
│   └── sample_input.csv
├── artifacts/
│   └── pretrained/
│       ├── latency_model.pkl
│       ├── throughput_model.pkl
│       └── qos_model.pkl
└── docs/
		├── QUICKSTART.md
		└── SYSTEM_EXPLANATION.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run app.py
```

## Are the PKL files required?

Short answer: depends on tab.

- Required for Performance Prediction tab:
	- artifacts/pretrained/latency_model.pkl
	- artifacts/pretrained/throughput_model.pkl
	- artifacts/pretrained/qos_model.pkl
- Not required for ML Lab tab, because ML Lab trains models from data/5g_network_data.csv or uploaded CSV.

If you only use ML Lab, you can keep the pretrained files archived elsewhere. If you use Prediction, keep them in artifacts/pretrained/.

## Documentation

- Quick start: docs/QUICKSTART.md
- User manual: docs/USER_MANUAL.md
- Full architecture, diagrams, and flowcharts: docs/SYSTEM_EXPLANATION.md