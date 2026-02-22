# train_lstm.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
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
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_modell.h5")   # double-l
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_minmax.pkl")
WINDOW = 7
EPOCHS = 100
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
        # fallback to second column
        if len(df.columns) >= 2:
            col = df.columns[1]
        else:
            raise ValueError("uploaded_data.csv doesn't contain expected columns.")
    y = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values.astype(float)
    return y, df

def series_to_windows(y, win=7):
    X, Y = [], []
    for i in range(len(y) - win + 1):
        window = y[i:i+win]
        X.append(window)
        Y.append(y[i+win-1])  # predict last of window (one-step)
    X = np.array(X)
    Y = np.array(Y)
    # reshape X to (n_samples, win, 1)
    X = X.reshape((-1, win, 1))
    return X, Y

def build_lstm(win=7):
    model = Sequential([
        LSTM(64, input_shape=(win,1), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading data...")
    y, df = load_series(DATA_PATH)
    print("Series length:", len(y))

    # scaler fit on entire y (training-time). We'll save it and use for inference.
    scaler = MinMaxScaler()
    y_resh = y.reshape(-1,1)
    y_sc = scaler.fit_transform(y_resh).flatten()

    X, Y = series_to_windows(y_sc, WINDOW)
    print("Windows:", X.shape, Y.shape)

    # train/val split (temporal split: last VALID_SPLIT portion for val)
    n = len(X)
    val_n = max(1, int(n * VALID_SPLIT))
    train_n = n - val_n
    X_train, Y_train = X[:train_n], Y[:train_n]
    X_val, Y_val = X[train_n:], Y[train_n:]

    print("Train samples:", X_train.shape[0], "Val samples:", X_val.shape[0])

    model = build_lstm(WINDOW)
    model.summary()

    # callbacks
    ckpt = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[ckpt, es],
        verbose=2
    )

    # ensure best model saved
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)
        print("Saved model to", MODEL_PATH)
    else:
        print("Model saved by checkpoint to", MODEL_PATH)

    # save scaler
    joblib.dump(scaler, SCALER_PATH)
    print("Saved scaler to", SCALER_PATH)

    # Evaluate on full series (inverse transform)
    # produce predictions using sliding windows and invert scaler
    X_full, _ = series_to_windows(y_sc, WINDOW)
    preds_sc = model.predict(X_full, verbose=0).reshape(-1)
    # expand to full length by prepending repeated first preds
    preds_sc_full = np.concatenate([np.repeat(preds_sc[0], WINDOW-1), preds_sc])
    try:
        preds = scaler.inverse_transform(preds_sc_full.reshape(-1,1)).flatten()
    except Exception:
        preds = preds_sc_full

    # align ground truth
    y_true = y[:len(preds)]
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    print("FINAL EVAL on full sequence -> RMSE: {:.6f}, MAE: {:.6f}".format(rmse, mae))

    # save a quick CSV for inspection
    out_df = pd.DataFrame({"date": df.iloc[:len(preds), 0].astype(str), "actual": y_true, "pred": preds})
    out_df.to_csv(os.path.join(MODEL_DIR, "lstm_result_debug.csv"), index=False)
    print("Saved debug CSV:", os.path.join(MODEL_DIR, "lstm_result_debug.csv"))

if __name__ == "__main__":
    main()
