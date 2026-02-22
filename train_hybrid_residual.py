# train_hybrid_residual.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "./data/uploaded_data.csv"
MODEL_DIR = "./models"
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_lstm_model.h5")
RESIDUAL_SCALER_PATH = os.path.join(MODEL_DIR, "residual_scaler.pkl")
ARIMA_MODEL_PATH = os.path.join(MODEL_DIR, "best_arima_model.pkl")
WINDOW = 7
EPOCHS = 80
BATCH = 16
VALID_SPLIT = 0.15

def load_series(path):
    if not os.path.exists(path):
        raise FileNotFoundError("uploaded_data.csv not found: " + path)
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    if 'prec' in cols:
        col = df.columns[cols.index('prec')]
    elif 'curah_hujan' in cols:
        col = df.columns[cols.index('curah_hujan')]
    else:
        if len(df.columns) >= 2:
            col = df.columns[1]
        else:
            raise ValueError("uploaded_data.csv doesn't contain expected columns.")
    y = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values.astype(float)
    return y, df

def compute_arima_fitted(y):
    # Try to load pmdarima/statsmodels ARIMA fitted model; if not available, fallback to rolling mean
    if os.path.exists(ARIMA_MODEL_PATH):
        try:
            import joblib
            arima_m = joblib.load(ARIMA_MODEL_PATH)
            # pmdarima case
            if hasattr(arima_m, "predict_in_sample"):
                try:
                    import pmdarima as pm
                    order = getattr(arima_m, "order", (1,0,0))
                    seasonal_order = getattr(arima_m, "seasonal_order", (0,0,0,0))
                    m2 = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order)
                    m2.fit(y)
                    y_arima = m2.predict_in_sample(start=0, end=len(y)-1)
                    return np.asarray(y_arima, dtype=float)
                except Exception:
                    pass
            # statsmodels SARIMAXResults fallback
            if hasattr(arima_m, "model"):
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
                    res = arima_m
                    res2 = res.model.smooth(res.model.filter(y).params)
                    fv = np.asarray(res2.fittedvalues, dtype=float)
                    if len(fv) < len(y):
                        fv = np.pad(fv, (len(y)-len(fv), 0), mode='edge')
                    else:
                        fv = fv[-len(y):]
                    return fv
                except Exception:
                    pass
        except Exception:
            pass
    # fallback
    if len(y) >= 3:
        return pd.Series(y).rolling(3, min_periods=1).mean().values
    return y.copy()

def series_to_windows(arr, win=7):
    X, Y = [], []
    for i in range(len(arr) - win + 1):
        X.append(arr[i:i+win])
        Y.append(arr[i+win-1])
    X = np.array(X).reshape(-1, win, 1)
    Y = np.array(Y)
    return X, Y

def build_small_lstm(win=7):
    model = Sequential([
        LSTM(32, input_shape=(win,1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    y, df = load_series(DATA_PATH)
    print("Loaded series length:", len(y))
    y_arima = compute_arima_fitted(y)
    resid = (y - y_arima).astype(float)
    print("Residual stats: mean {:.6f}, std {:.6f}".format(resid.mean(), resid.std()))

    # scale residuals
    scaler = MinMaxScaler()
    resid_sc = scaler.fit_transform(resid.reshape(-1,1)).flatten()

    X, Y = series_to_windows(resid_sc, WINDOW)
    print("Residual windows:", X.shape, Y.shape)
    n = len(X)
    val_n = max(1, int(n * VALID_SPLIT))
    train_n = n - val_n
    X_train, Y_train = X[:train_n], Y[:train_n]
    X_val, Y_val = X[train_n:], Y[train_n:]

    model = build_small_lstm(WINDOW)
    model.summary()

    ckpt = ModelCheckpoint(HYBRID_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[ckpt, es], verbose=2)

    # save scaler & ensure model saved
    joblib.dump(scaler, RESIDUAL_SCALER_PATH)
    print("Saved residual scaler to", RESIDUAL_SCALER_PATH)
    if not os.path.exists(HYBRID_MODEL_PATH):
        model.save(HYBRID_MODEL_PATH)
        print("Saved hybrid model to", HYBRID_MODEL_PATH)
    else:
        print("Hybrid model saved by checkpoint:", HYBRID_MODEL_PATH)

    # evaluate full
    X_full, _ = series_to_windows(resid_sc, WINDOW)
    if len(X_full) > 0:
        res_hat_sc = model.predict(X_full, verbose=0).reshape(-1)
        res_hat_full = np.concatenate([np.repeat(res_hat_sc[0], WINDOW-1), res_hat_sc])
        try:
            res_hat = scaler.inverse_transform(res_hat_full.reshape(-1,1)).flatten()
        except Exception:
            res_hat = res_hat_full
    else:
        res_hat = np.zeros_like(resid)

    y_hybrid = y_arima + res_hat
    rmse = np.sqrt(mean_squared_error(y[:len(y_hybrid)], y_hybrid))
    mae = mean_absolute_error(y[:len(y_hybrid)], y_hybrid)
    print("HYBRID EVAL -> RMSE: {:.6f}, MAE: {:.6f}".format(rmse, mae))

    # save debug csv
    out_df = pd.DataFrame({"date": df.iloc[:len(y_hybrid),0].astype(str), "actual": y[:len(y_hybrid)], "y_arima": y_arima[:len(y_hybrid)], "res_hat": res_hat, "y_hybrid": y_hybrid})
    out_df.to_csv(os.path.join(MODEL_DIR, "hybrid_result_debug.csv"), index=False)
    print("Saved hybrid debug CSV.")

if __name__ == "__main__":
    main()
