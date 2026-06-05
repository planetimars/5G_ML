# Quick Start

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run the app

```bash
streamlit run app.py
```

## 3) Use the two tabs

1. Performance Prediction
- Uses pretrained artifacts from artifacts/pretrained/.
- Predicts latency, throughput, and QoS from input features.

2. ML Lab: Train, Compare, Select Features
- Uses dataset from data/5g_network_data.csv by default.
- Lets you select target/features, run feature selection, compare many models, and run advanced diagnostics.
- Supports optional SHAP explainability plots for the best model.

## 4) Optional: Bring your own files

- Prediction input samples can follow data/sample_input.csv.
- ML Lab accepts CSV uploads directly from UI.

## 5) Important note about PKL files

- Required for Performance Prediction tab:
  - artifacts/pretrained/latency_model.pkl
  - artifacts/pretrained/throughput_model.pkl
  - artifacts/pretrained/qos_model.pkl
- Not required for ML Lab tab (it trains models live from your dataset).

## 6) Cloud deployment notes

- Keep these paths in the deployed repo exactly:
  - artifacts/pretrained/*.pkl for Prediction tab
  - data/5g_network_data.csv for default ML Lab dataset
- If Prediction tab is needed, pretrained PKL files must be present in artifacts/pretrained/.
- For research runs, download CSV outputs from the UI or save to artifacts/reports using the in-app button.

## 7) Groq API optimization recommendations

- The app tries AI-based optimization recommendations first using Groq.
- Set API key in Streamlit Cloud:
  - Secrets key: GROQ_API_KEY
- If Groq is unavailable or key is missing, app automatically falls back to rule-based recommendations.

## 8) PDF export

- The app supports PDF downloads for:
  - Optimization report
  - Experiment report
  - Run comparison report
- Graph rendering in PDF uses Plotly static image export via kaleido.
