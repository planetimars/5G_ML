"""MLP regressor pipeline definition."""

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mlp_regressor_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)),
    ])
