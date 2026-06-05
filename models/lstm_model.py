"""LSTM training helper for optional deep-learning regression benchmarking."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM

    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


def train_lstm_regressor(x_train, x_test, y_train, y_test, epochs):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_lstm = x_train_scaled.reshape(x_train_scaled.shape[0], x_train_scaled.shape[1], 1)
    x_test_lstm = x_test_scaled.reshape(x_test_scaled.shape[0], x_test_scaled.shape[1], 1)

    model = Sequential([
        LSTM(32, input_shape=(x_train_lstm.shape[1], 1)),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0)

    y_pred = model.predict(x_test_lstm, verbose=0).flatten()
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    }
