"""Advanced research helpers: stability, calibration, learning curves, and permutation importance."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import learning_curve, train_test_split


def _primary_metric(task_type, y_true, y_pred):
    if task_type == "regression":
        return r2_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _primary_metric_name(task_type):
    return "R2" if task_type == "regression" else "F1"


def compute_stability_analysis(model_registry, x_full, y_full, task_type, test_size, num_seeds):
    """Evaluate model metric stability across multiple random seeds."""
    seeds = list(range(42, 42 + num_seeds))
    rows = []

    for model_name, base_model in model_registry.items():
        metric_values = []

        for seed in seeds:
            stratify_target = y_full if task_type == "classification" else None
            x_train, x_test, y_train, y_test = train_test_split(
                x_full,
                y_full,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_target,
            )

            model = clone(base_model)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            metric_values.append(_primary_metric(task_type, y_test, y_pred))

        rows.append(
            {
                "Model": model_name,
                "Metric": _primary_metric_name(task_type),
                "Mean": float(np.mean(metric_values)),
                "Std": float(np.std(metric_values)),
                "Min": float(np.min(metric_values)),
                "Max": float(np.max(metric_values)),
            }
        )

    stability_df = pd.DataFrame(rows)
    stability_df = stability_df.sort_values("Mean", ascending=False)
    return stability_df


def compute_learning_curve_data(model_registry, x_full, y_full, model_names, task_type, cv_folds=5):
    """Generate learning curve aggregates for selected models."""
    scoring = "r2" if task_type == "regression" else "f1_weighted"
    train_sizes = np.linspace(0.2, 1.0, 5)
    rows = []

    for model_name in model_names:
        if model_name not in model_registry:
            continue

        model = clone(model_registry[model_name])
        sizes, train_scores, val_scores = learning_curve(
            model,
            x_full,
            y_full,
            train_sizes=train_sizes,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=None,
        )

        for idx, size in enumerate(sizes):
            rows.append(
                {
                    "Model": model_name,
                    "TrainSize": int(size),
                    "TrainMean": float(train_scores[idx].mean()),
                    "TrainStd": float(train_scores[idx].std()),
                    "ValidationMean": float(val_scores[idx].mean()),
                    "ValidationStd": float(val_scores[idx].std()),
                }
            )

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_calibration_and_confidence(best_model, x_test, y_test):
    """Create calibration curve data and confidence metrics for classification models."""
    if not hasattr(best_model, "predict_proba"):
        return None, None

    probs = best_model.predict_proba(x_test)
    pred_labels = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)
    correctness = (pred_labels == np.asarray(y_test)).astype(int)

    # Confidence calibration bins (works for binary and multiclass)
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidences, bins) - 1

    calib_rows = []
    total = len(confidences)
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue

        acc = float(correctness[mask].mean())
        conf = float(confidences[mask].mean())
        frac = count / total
        ece += abs(acc - conf) * frac
        calib_rows.append(
            {
                "Bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "Accuracy": acc,
                "Confidence": conf,
                "Count": count,
            }
        )

    calib_df = pd.DataFrame(calib_rows)

    # Brier score (binary exact, multiclass generalized)
    classes = np.unique(y_test)
    if probs.shape[1] == 2:
        y_binary = (np.asarray(y_test) == classes.max()).astype(float)
        brier = float(np.mean((probs[:, 1] - y_binary) ** 2))
    else:
        y_int = np.asarray(y_test, dtype=int)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y_int)), y_int] = 1.0
        brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    summary_df = pd.DataFrame(
        [
            {
                "ECE": ece,
                "BrierScore": brier,
                "AvgConfidence": float(confidences.mean()),
                "AvgAccuracy": float(correctness.mean()),
            }
        ]
    )

    return calib_df, summary_df


def compute_permutation_importance(best_model, x_test, y_test, task_type, n_repeats=12):
    """Compute permutation-based feature importance on the test set."""
    scoring = "r2" if task_type == "regression" else "f1_weighted"
    perm = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring,
    )

    feature_names = list(x_test.columns)
    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "ImportanceMean": perm.importances_mean,
            "ImportanceStd": perm.importances_std,
        }
    ).sort_values("ImportanceMean", ascending=False)
    return perm_df
