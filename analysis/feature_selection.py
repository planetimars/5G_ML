"""Feature selection helpers used by the ML Lab workflow."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder


def _is_numeric_target(y):
    """Return True only when y can safely be used as a continuous regression target."""
    try:
        pd.to_numeric(pd.Series(y), errors="raise")
        return True
    except Exception:
        return False


def _encode_classification_target(y):
    """Feature selectors need numeric class labels, so encode text classes when needed."""
    series = pd.Series(y)
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return LabelEncoder().fit_transform(series.astype(str))
    return np.asarray(y)


def run_feature_selection(x_train, y_train, x_columns, task_type, method, k_features):
    """
    Select the top k features for either regression or classification.

    Safety note:
    QoS class targets may be stored as text labels such as Good/Medium/Poor.
    In that case we must use classification selectors. Using regression selectors on
    text labels causes Streamlit Cloud to crash because sklearn tries to convert the
    labels to float.
    """
    k_value = max(1, min(k_features, len(x_columns)))

    # Defensive task detection. The UI passes task_type, but this prevents crashes
    # when a classification target is accidentally detected as regression.
    is_regression = task_type == "regression" and _is_numeric_target(y_train)

    if is_regression:
        y_for_selector = pd.to_numeric(pd.Series(y_train), errors="coerce")
        if method == "Mutual Information":
            selector = SelectKBest(score_func=mutual_info_regression, k=k_value)
        elif method == "F-Score":
            selector = SelectKBest(score_func=f_regression, k=k_value)
        else:
            selector = RFE(estimator=LinearRegression(), n_features_to_select=k_value)
    else:
        y_for_selector = _encode_classification_target(y_train)
        if method == "Mutual Information":
            selector = SelectKBest(score_func=mutual_info_classif, k=k_value)
        elif method == "F-Score":
            selector = SelectKBest(score_func=f_classif, k=k_value)
        else:
            selector = RFE(estimator=LogisticRegression(max_iter=2000), n_features_to_select=k_value)

    selector.fit(x_train, y_for_selector)
    return list(np.array(x_columns)[selector.get_support()])
