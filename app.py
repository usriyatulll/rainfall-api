from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, re
from functools import lru_cache


# OPTIONAL IMPORTS
try:
    import joblib
except:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except:
    load_model = None

try:
    from sklearn.preprocessing import MinMaxScaler
except:
    MinMaxScaler = None


# APP INIT
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "./data"
MODEL_FOLDER = "./models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# HELPERS 
def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def normalize_columns(df):
    df.columns = [_norm(c) for c in df.columns]
    return df

def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9\.\-eE]+", "", regex=True),
        errors="coerce"
    )


# LOAD UPLOADED DATA
def uploaded_series():
    path = os.path.join(UPLOAD_FOLDER, "uploaded_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Belum ada data upload")

    df = pd.read_csv(path)
    df = normalize_columns(df)

    if not {"tanggal", "prec"} <= set(df.columns):
        raise ValueError("uploaded_data.csv harus ada kolom tanggal & prec")

    dates = df["tanggal"].astype(str).tolist()
    y = clean_numeric(df["prec"]).fillna(0).values
    return dates, y


# UPLOAD ENDPOINT
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(status="error", message="File tidak ditemukan"), 400

    f = request.files["file"]
    ext = os.path.splitext(f.filename.lower())[1]

    if ext not in {".csv", ".xlsx", ".xls"}:
        return jsonify(status="error", message="Format tidak didukung"), 400

    df = pd.read_excel(f) if ext != ".csv" else pd.read_csv(f)
    df = normalize_columns(df)

    date_col = next((c for c in ["tanggal", "date"] if c in df.columns), None)
    prec_col = next((c for c in ["prec", "rr", "curah_hujan"] if c in df.columns), None)

    if not date_col or not prec_col:
        return jsonify(status="error", message="Kolom tanggal / curah hujan tidak ditemukan"), 400

    out = df[[date_col, prec_col]].copy()
    out.columns = ["tanggal", "prec"]
    out["prec"] = clean_numeric(out["prec"])

    out.to_csv(os.path.join(UPLOAD_FOLDER, "uploaded_data.csv"), index=False)

    return jsonify(
        status="success",
        message="Upload berhasil",
        data=out.to_dict(orient="records")
    )


# DATA STATUS
@app.route("/api/data/status")
def data_status():
    return jsonify(has_upload=os.path.exists(os.path.join(UPLOAD_FOLDER, "uploaded_data.csv")))


# LOAD MODELS 
@lru_cache(maxsize=1)
def load_arima_model():
    if joblib is None:
        raise RuntimeError("joblib belum tersedia")
    return joblib.load("./models/best_arima_model.pkl")

@lru_cache(maxsize=1)
def load_lstm_model():
    model = load_model("./models/lstm_model.h5")
    scaler = joblib.load("./models/scaler_minmax.pkl")
    return model, scaler

@lru_cache(maxsize=1)
def load_hybrid_models():
    arima = load_arima_model()
    lstm = load_model("./models/hybrid_lstm_model.h5")
    scaler = joblib.load("./models/residual_scaler.pkl")
    return arima, lstm, scaler


# PREDICTION HELPERS
def predict_arima(y):
    model = load_arima_model()
    return model.predict_in_sample()

def predict_lstm(y, window=7):
    model, scaler = load_lstm_model()
    y = y.reshape(-1, 1)
    y_sc = scaler.transform(y)

    X = [y_sc[i:i+window] for i in range(len(y_sc)-window+1)]
    X = np.array(X)

    y_hat_sc = model.predict(X, verbose=0).reshape(-1)
    y_hat_sc = np.concatenate([np.repeat(y_hat_sc[0], window-1), y_hat_sc])

    return scaler.inverse_transform(y_hat_sc.reshape(-1,1)).flatten()

def predict_hybrid(y, window=7):
    arima, lstm, scaler = load_hybrid_models()
    y = np.asarray(y, dtype=float)

    # ===== ARIMA =====
    y_arima = predict_arima(y)

    # ===== Residual =====
    resid = (y - y_arima).reshape(-1, 1)
    resid_sc = scaler.transform(resid)

    # windowing
    Xr = []
    for i in range(len(resid_sc) - window + 1):
        Xr.append(resid_sc[i:i+window])
    Xr = np.array(Xr)

    # prediksi residual
    res_hat_sc = lstm.predict(Xr, verbose=0).reshape(-1)

    # ===== PADDING DI DEPAN =====
    pad = np.zeros(window - 1)
    res_hat_sc_full = np.concatenate([pad, res_hat_sc])

    # inverse scale
    res_hat = scaler.inverse_transform(
        res_hat_sc_full.reshape(-1, 1)
    ).flatten()

    # ===== PANJANG DIJAMIN SAMA =====
    res_hat = res_hat[:len(y_arima)]

    return y_arima + res_hat

# RUN MODEL ENDPOINT
@app.route("/api/run/<model>", methods=["POST"])
def run_model(model):
    if model not in {"arima", "lstm", "hybrid"}:
        return jsonify(status="error", message="Model tidak dikenal"), 404

    try:
        dates, y = uploaded_series()
        y = np.asarray(y, dtype=float)
    except Exception as e:
        return jsonify(status="error", message=str(e)), 400

    try:
        if model == "arima":
            y_hat = predict_arima(y)
            df = pd.DataFrame({"date": dates, "arima_forecast": y_hat})
            out = "./models/arima_forecast_results.csv"

        elif model == "lstm":
            y_hat = predict_lstm(y)
            df = pd.DataFrame({"date": dates, "lstm_forecast": y_hat})
            out = "./models/lstm_forecast_results.csv"

        else:
            y_hat = predict_hybrid(y)
            df = pd.DataFrame({"date": dates, "hybrid_forecast": y_hat})
            out = "./models/hybrid_forecast_results.csv"

        df.to_csv(out, index=False)

        return jsonify(
            status="success",
            message=f"{model.upper()} selesai",
            preview=df.tail(5).to_dict(orient="records")
        )

    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

# READ RESULT (LARAVEL)
@app.route("/api/prediksi/<model>", methods=["GET"])
def get_prediction(model):
    MODEL_FOLDER = "./models"

    if model not in {"arima", "lstm", "hybrid"}:
        return jsonify(
            status="error",
            message="Model tidak dikenal"
        ), 404

    path = os.path.join(MODEL_FOLDER, f"{model}_forecast_results.csv")
    if not os.path.exists(path):
        return jsonify(
            status="error",
            message=f"Hasil {model.upper()} belum tersedia"
        ), 400

    try:
        # BACA CSV HASIL MODEL
        df = pd.read_csv(path, sep=None, engine="python")
        df = normalize_columns(df)

        # buang index pandas kalau ada
        if "unnamed_0" in df.columns:
            df = df.drop(columns=["unnamed_0"])

        cols = set(df.columns)

        # KANDIDAT KOLOM
        pred_candidates = [
            "prediksi", "prediction", "forecast", "predicted_mean",
            "nilai_prediksi",
            "arima_forecast", "arima_forecast_prec",
            "lstm_forecast", "lstm_forecast_prec",
            "hybrid_forecast", "hybrid_forecast_prec"
        ]

        actual_candidates = [
            "aktual", "actual", "actual_prec",
            "nilai_aktual", "observed", "y_true"
        ]

        date_candidates = [
            "tanggal", "date", "time", "periode"
        ]

        def pick(cands):
            return next((c for c in cands if c in cols), None)

        col_pred = pick(pred_candidates)
        col_act  = pick(actual_candidates)
        col_date = pick(date_candidates)

        if not col_pred:
            return jsonify(
                status="error",
                message=f"Kolom prediksi tidak ditemukan: {list(df.columns)}"
            ), 400

        # RENAME KE STANDAR
        rename_map = {col_pred: "prediksi"}
        if col_act:
            rename_map[col_act] = "aktual"
        if col_date:
            rename_map[col_date] = "tanggal"

        df = df.rename(columns=rename_map)

        # ALIGN DATA TERAKHIR (DASHBOARD)
        # sumber tanggal & aktual HANYA dari uploaded_data.csv
        dates, y = uploaded_series()

        n = min(len(df), len(dates))

        # ambil DATA TERAKHIR → tanggal terakhir = 30-04-2025 (jadi sesuai panjang result dihitung dari terakhir ke depan)
        df = df.iloc[-n:].reset_index(drop=True)

        df["tanggal"] = dates[-n:]
        df["aktual"]  = y[-n:]

        # FORMAT TANGGAL SERAGAM
        try:
            df["tanggal"] = (
                pd.to_datetime(df["tanggal"], dayfirst=True, errors="coerce")
                .dt.strftime("%d/%m/%Y")
            )
        except Exception:
            df["tanggal"] = df["tanggal"].astype(str)


        # CLEAN NUMERIK
        df["prediksi"] = clean_numeric(df["prediksi"]).round(2)
        df["aktual"]   = clean_numeric(df["aktual"]).round(2)

        # RESPONSE FINAL
        return jsonify(
            status="success",
            data=df[["tanggal", "aktual", "prediksi"]].to_dict(orient="records")
        )

    except Exception as e:
        return jsonify(
            status="error",
            message=str(e)
        ), 500



@app.route("/api/metrics/hybrid", methods=["GET"])
def hybrid_metrics():
    path = "./models/error_metrics_hybrid_arima_lstm.csv"

    if not os.path.exists(path):
        return jsonify(
            status="error",
            message="File metrik Hybrid tidak ditemukan"
        ), 400

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df = normalize_columns(df)

        if "unnamed_0" in df.columns:
            df = df.drop(columns=["unnamed_0"])

        metrics = {}

        # CASE 1: ADA HEADER metric,value
        if {"metric", "value"} <= set(df.columns):
            for _, row in df.iterrows():
                key = str(row["metric"]).upper()
                try:
                    metrics[key] = round(float(row["value"]), 2)
                except:
                    metrics[key] = None

        # CASE 2: TANPA HEADER (kolom 0,1)
        elif len(df.columns) >= 2:
            for _, row in df.iterrows():
                key = str(row.iloc[0]).upper()
                try:
                    metrics[key] = round(float(row.iloc[1]), 2)
                except:
                    metrics[key] = None

        else:
            return jsonify(
                status="error",
                message="Format file metrik tidak dikenali"
            ), 400

        # PASTIKAN RMSE & MAE ADA
        return jsonify(
            status="success",
            data={
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE")
            }
        )

    except Exception as e:
        return jsonify(
            status="error",
            message=str(e)
        ), 500


# ROOT
@app.route("/")
def root():
    return jsonify(status="success", message="Rainfall Prediction API running")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
