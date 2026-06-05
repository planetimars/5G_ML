"""Feature selection helpers used by the ML Lab workflow."""

import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression


def run_feature_selection(x_train, y_train, x_columns, task_type, method, k_features):
    k_value = max(1, min(k_features, len(x_columns)))

    if method == "Mutual Information":
        score_fn = mutual_info_regression if task_type == "regression" else mutual_info_classif
        selector = SelectKBest(score_func=score_fn, k=k_value)
    elif method == "F-Score":
        score_fn = f_regression if task_type == "regression" else f_classif
        selector = SelectKBest(score_func=score_fn, k=k_value)
    else:
        estimator = LinearRegression() if task_type == "regression" else LogisticRegression(max_iter=1000)
        selector = RFE(estimator=estimator, n_features_to_select=k_value)

    selector.fit(x_train, y_train)
    return list(np.array(x_columns)[selector.get_support()])
