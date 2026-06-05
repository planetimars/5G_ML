"""Central model registry and evaluation metrics for regression and classification experiments."""

import pandas as pd
import time
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from models.gradient_boosting_classifier_model import gradient_boosting_classifier_pipeline
from models.gradient_boosting_regressor_model import gradient_boosting_regressor_pipeline
from models.knn_classifier_model import knn_classifier_pipeline
from models.knn_regressor_model import knn_regressor_pipeline
from models.linear_regression_model import linear_regression_pipeline
from models.logistic_regression_model import logistic_regression_pipeline
from models.mlp_classifier_model import mlp_classifier_pipeline
from models.mlp_regressor_model import mlp_regressor_pipeline
from models.random_forest_classifier_model import random_forest_classifier_pipeline
from models.random_forest_regressor_model import random_forest_regressor_pipeline
from models.svc_model import svc_pipeline
from models.svr_model import svr_pipeline


def get_model_registry(task_type):
    if task_type == "regression":
        return {
            "Linear Regression": linear_regression_pipeline(),
            "Ridge Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=1.0)),
            ]),
            "Lasso Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", Lasso(alpha=0.01, max_iter=5000)),
            ]),
            "Decision Tree Regressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]),
            "Random Forest Regressor": random_forest_regressor_pipeline(),
            "Extra Trees Regressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=400, random_state=42)),
            ]),
            "Gradient Boosting Regressor": gradient_boosting_regressor_pipeline(),
            "AdaBoost Regressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", AdaBoostRegressor(random_state=42)),
            ]),
            "SVR": svr_pipeline(),
            "KNN Regressor": knn_regressor_pipeline(),
            "Neural Network (MLP)": mlp_regressor_pipeline(),
        }

    return {
        "Logistic Regression": logistic_regression_pipeline(),
        "Decision Tree Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest Classifier": random_forest_classifier_pipeline(),
        "Extra Trees Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=400, random_state=42)),
        ]),
        "Gradient Boosting Classifier": gradient_boosting_classifier_pipeline(),
        "AdaBoost Classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", AdaBoostClassifier(random_state=42)),
        ]),
        "SVC": svc_pipeline(),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GaussianNB()),
        ]),
        "KNN Classifier": knn_classifier_pipeline(),
        "Neural Network (MLP)": mlp_classifier_pipeline(),
    }


def evaluate_models(model_registry, x_train, x_test, y_train, y_test, task_type, run_cv=False, cv_folds=5):
    evaluation_rows = []
    trained_models = {}
    prediction_details = {}

    for model_name, model in model_registry.items():
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_pred = model.predict(x_test)
        predict_time = time.perf_counter() - pred_start

        trained_models[model_name] = model
        prediction_details[model_name] = y_pred

        cv_mean = float("nan")
        cv_std = float("nan")
        if run_cv:
            scoring = "r2" if task_type == "regression" else "f1_weighted"
            cv_scores = cross_val_score(model, x_train, y_train, cv=cv_folds, scoring=scoring)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

        if task_type == "regression":
            evaluation_rows.append({
                "Model": model_name,
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
                "FitTimeSec": fit_time,
                "PredictTimeSec": predict_time,
                "CVMean": cv_mean,
                "CVStd": cv_std,
            })
        else:
            evaluation_rows.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "BalancedAccuracy": balanced_accuracy_score(y_test, y_pred),
                "FitTimeSec": fit_time,
                "PredictTimeSec": predict_time,
                "CVMean": cv_mean,
                "CVStd": cv_std,
            })

    results_df = pd.DataFrame(evaluation_rows)
    if task_type == "regression":
        results_df = results_df.sort_values("R2", ascending=False)
    else:
        results_df = results_df.sort_values("F1", ascending=False)

    best_model_name = results_df.iloc[0]["Model"]
    return results_df, best_model_name, trained_models[best_model_name], trained_models, prediction_details
