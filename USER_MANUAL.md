# User Manual: 5G ML Optimization System

This manual explains each part of the app, what it means, and how to use it for operations and research.

## 1) Who this app is for

- Operations users: monitor network quality, predict QoS/latency/throughput, and get precise optimization actions.
- Research users: compare many models, inspect diagnostics, analyze explainability, and export publication-ready artifacts.

## 2) App modes

### Operator Mode
Use this for fast decision support.

What you get:
- Cleaner, concise UI
- Prediction + action plan focus
- Optional advanced diagnostics hidden by default

### Research Mode
Use this for model comparison and paper experiments.

What you get:
- Full diagnostics enabled
- Cross-validation/stability tools
- Learning curves, calibration, permutation importance, SHAP
- Full experiment bundle export

## 3) Performance Prediction tab

### Step 1: Input
You can provide parameters in two ways:

1. Manual input
- Enter values for signal strength, SINR, users, bandwidth, packet loss, jitter, mobility speed.

2. Upload file
- Upload CSV/TXT containing required columns:
  - signal_strength
  - sinr
  - connected_users
  - bandwidth_mhz
  - packet_loss
  - jitter
  - mobility_speed

### Step 2: Prediction
The app uses pretrained models from artifacts/pretrained/ to predict:
- predicted_latency_ms
- predicted_throughput_mbps
- qos_classification

### Step 3: Status panel
Status pills summarize current risk:
- Latency Risk
- Throughput Health
- QoS Confidence

### Step 4: Precise Optimization Strategy Plan
The app provides targeted actions with:
- Priority
- Issue and current KPI
- Target KPI
- PreciseAction
- EstimatedImpact

Recommendation source order:
1. Groq AI recommendation engine (primary)
2. Rule-based strategy engine (fallback only)

You can download:
- Prediction results CSV
- Optimization plan CSV
- Optimization report PDF

## 4) ML Lab tab

### Step 1: Dataset and target
- Use default dataset data/5g_network_data.csv or upload your own CSV.
- Choose target column.
- Choose test set size.
- Choose features.

Task type is auto-detected:
- Regression for numeric continuous targets
- Classification for class-like targets

### Step 2: Correlation and feature design
You get 3 correlation views:

1. Heatmap
- Full numeric correlation matrix.

2. Target Correlation
- Top absolute correlations with selected target (numeric targets).

3. Scatter Matrix
- Pairwise feature interactions (choose 2-6 numeric features).

Feature selection options:
- Mutual Information
- F-Score
- RFE

### Step 3: Train and evaluate
Click Train and Compare Models to run the model registry.

Regression models include:
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, AdaBoost
- SVR, KNN, MLP

Classification models include:
- Logistic Regression
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, AdaBoost
- SVC, Gaussian Naive Bayes, KNN, MLP

## 4.1) Run Comparison tab

Purpose:
- Compare multiple saved experiment runs from artifacts/reports.
- Track metric drift over time.
- Check if best-model choice is stable across runs.

How to use:
1. First save at least 2 runs from ML Lab using Save Experiment Package.
2. Open Run Comparison tab.
3. Select run folders to compare.
4. Choose metric (for example R2, F1, Accuracy, CVMean).
5. Review:
  - Model Ranking by Run chart
  - Metric Distribution Across Runs chart
  - Best Model Snapshot per Run table/chart
  - Experiment Setup Summary table

Outputs:
- Downloadable run comparison summary CSV.

## 5) What each research option means

### Run cross-validation
- Re-evaluates each model over CV folds.
- Outputs CVMean and CVStd in leaderboard.

### Model stability analysis across random seeds
- Repeats train/test splits with different seeds.
- Reports Mean, Std, Min, Max performance by model.
- Use this to assess robustness.

### Learning curves (top models)
- Shows Train vs Validation score as data size grows.
- Helps detect underfitting or overfitting.

### Permutation importance (best model)
- Measures how prediction quality drops when each feature is shuffled.
- Model-agnostic importance estimate.

### SHAP explainability
- Provides feature contribution summary for best model.
- Useful for explaining model behavior.

## 6) Leaderboard metrics

### Regression
- R2: higher is better
- MAE: lower is better
- MAPE: lower is better
- RMSE: lower is better
- FitTimeSec, PredictTimeSec
- CVMean, CVStd (if CV enabled)

### Classification
- Accuracy: higher is better
- F1 (weighted): higher is better
- Precision, Recall
- BalancedAccuracy
- FitTimeSec, PredictTimeSec
- CVMean, CVStd (if CV enabled)

## 7) Individual Model Explorer

Purpose:
- Inspect each trained model separately, not just the best one.

Modes:
- Select one model to inspect
- Render graphs for all models (heavier)

Regression view:
- Actual vs Predicted
- Residuals vs Predicted
- Residual histogram
- Error quantiles

Classification view:
- Confusion matrix
- Per-class accuracy

## 8) Advanced diagnostics

### Regression diagnostics
- Residual-based quality checks
- Downloadable diagnostics table

### Classification diagnostics
- Confusion matrix
- Per-class accuracy
- ROC-AUC indicators (when available)

### Calibration and confidence (classification)
- Calibration curve: bin accuracy vs confidence
- Confidence histogram
- ECE and Brier score summary

## 9) Exports and artifacts

Download buttons in UI provide:
- Leaderboard CSV
- Experiment summary CSV
- Diagnostics CSV
- SHAP CSV (if available)
- Optimization plan CSV
- Optimization report PDF
- Experiment report PDF
- Run comparison report PDF

### Save Experiment Package to artifacts/reports
This creates a run folder with:
- tables/*.csv
- figures/*.html

Use this for reproducibility and paper appendix materials.

## 10) Typical workflows

### A) Fast operations workflow
1. Switch to Operator Mode.
2. Use Prediction tab with manual/file input.
3. Run prediction.
4. Apply high-priority actions from strategy cards.
5. Download prediction and optimization CSV.

### B) Research-paper workflow
1. Switch to Research Mode.
2. Open ML Lab.
3. Pick target/features and feature-selection method.
4. Enable CV, stability, learning curves, permutation importance, SHAP.
5. Train and compare.
6. Review leaderboard + individual model explorer + diagnostics.
7. Save full experiment package.
8. Repeat for each target (latency, throughput, qos_class).

## 11) Cloud deployment notes

Required files for full behavior:
- artifacts/pretrained/latency_model.pkl
- artifacts/pretrained/throughput_model.pkl
- artifacts/pretrained/qos_model.pkl
- data/5g_network_data.csv

Prediction tab depends on pretrained pkl files.
ML Lab can still train from uploaded datasets without them.

Groq API setup (for AI optimization recommendations):
- Add GROQ_API_KEY in Streamlit Cloud secrets.
- If missing/unreachable, app automatically switches to fallback rules.

## 12) Troubleshooting

- No dataset found:
  - Ensure data/5g_network_data.csv exists or upload a CSV.
- SHAP unavailable:
  - Install dependencies from requirements.txt.
- Deep-learning benchmark not included in Streamlit build:
  - Current cloud-friendly setup focuses on classical models + explainability diagnostics.
- Slow runs:
  - Reduce number of features, disable render-all-models, lower stability seeds, disable learning curves/SHAP.
