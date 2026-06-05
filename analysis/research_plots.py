"""Research visualization and diagnostics helpers for regression and classification models."""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


def regression_diagnostics(y_true, y_pred):
    diagnostics_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    diagnostics_df["Residual"] = diagnostics_df["Actual"] - diagnostics_df["Predicted"]

    fig_scatter = px.scatter(
        diagnostics_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted",
    )
    fig_residual_hist = px.histogram(
        diagnostics_df,
        x="Residual",
        nbins=30,
        title="Residual Distribution",
    )
    fig_residual_vs_pred = px.scatter(
        diagnostics_df,
        x="Predicted",
        y="Residual",
        title="Residuals vs Predicted",
    )

    return {
        "scatter": fig_scatter,
        "residual_hist": fig_residual_hist,
        "residual_vs_pred": fig_residual_vs_pred,
        "table": diagnostics_df,
    }


def classification_diagnostics(y_true, y_pred, class_labels, display_labels=None):
    if display_labels is None:
        display_labels = class_labels

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    fig_confusion = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix",
        color_continuous_scale="Blues",
    )

    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    class_accuracy = []
    for label_value, label_name in zip(class_labels, display_labels):
        mask = y_true_series == label_value
        if mask.sum() == 0:
            score = np.nan
        else:
            score = (y_pred_series[mask] == label_value).mean()
        class_accuracy.append({"Class": label_name, "ClassAccuracy": score})

    class_acc_df = pd.DataFrame(class_accuracy)
    fig_class_accuracy = px.bar(
        class_acc_df,
        x="Class",
        y="ClassAccuracy",
        title="Per-Class Accuracy",
        text_auto=".3f",
    )

    return {
        "confusion": fig_confusion,
        "class_accuracy": fig_class_accuracy,
        "confusion_table": cm_df,
    }


def multiclass_roc_auc(model, x_test, y_test):
    if not hasattr(model, "predict_proba"):
        return None

    y_prob = model.predict_proba(x_test)
    classes = np.unique(y_test)

    if len(classes) < 2:
        return None

    y_bin = label_binarize(y_test, classes=classes)

    try:
        if y_prob.shape[1] == 2:
            auc_score = roc_auc_score(y_test, y_prob[:, 1])
            return pd.DataFrame([{"Type": "BinaryROC_AUC", "Score": auc_score}])

        auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
        return pd.DataFrame(
            [
                {"Type": "MultiClassROC_AUC_Macro", "Score": auc_macro},
                {"Type": "MultiClassROC_AUC_Weighted", "Score": auc_weighted},
            ]
        )
    except Exception:
        return None
