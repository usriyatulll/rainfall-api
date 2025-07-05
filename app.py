# app.py
from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

#Help3r function untuk validasi CSV
def validate_forecast_csv(filepath, allowed_sets):
    try:
        df = pd.read_csv(filepath)

        # normalisasi nama kolom
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if any(set(df.columns) == allowed for allowed in allowed_sets):
            return df, None
        return None, f"Kolom CSV tidak sesuai: {list(df.columns)}"

    except FileNotFoundError:
        return None, f"[Errno 2] File tidak ditemukan: '{filepath}'"
    except Exception as e:
        return None, str(e)



#data historis endpoint
@app.route("/api/prediksi", methods=["GET"])
def get_historical():
    try:
        df = pd.read_excel("./data/DATA 2020-2025.xlsx")
        df.columns = [c.strip().upper() for c in df.columns]
        return jsonify(message="Berhasil memuat data historis",
                       status="success",
                       data=df.to_dict(orient="records"))
    except Exception as e:
        return jsonify(message=str(e), status="error"), 500


# konfigurasi model
MODELS = {
    "arima": dict(
        file="./models/arima_forecast_results.csv",
        allowed=[
            {'tanggal', 'aktual', 'prediksi'},
            {'nilai_aktual', 'nilai_prediksi'},
            {'date', 'actual_prec', 'arima_forecast_prec'},
            {'date', 'actual', 'hybrid_forecast'},
                ],
        msg="Berhasil memuat data ARIMA"
    ),
    "lstm": dict(
        file="./models/lstm_result.csv",
        allowed=[
            {"tanggal", "aktual", "prediksi"},
            {"nilai_aktual", "nilai_prediksi"},
            {"date", "actual_prec", "lstm_forecast_prec"},
            {"date", "actual", "lstm_forecast"},
        ],
        msg="Berhasil memuat data LSTM"
    ),
    "hybrid": dict(
        file="./models/hybrid_forecast_results.csv",
        allowed=[
            {"tanggal", "aktual", "prediksi"},
            {"nilai_aktual", "nilai_prediksi"},
            {"date", "actual_prec", "hybrid_forecast_prec"},
            {"date", "actual", "hybrid_forecast"},
        ],
        msg="Berhasil memuat data Hybrid"
    ),
}


# factory endpoint untuk mendapatkan prediksi berdasarkan model
@app.route("/api/prediksi/<model>", methods=["GET"])
def get_forecast(model):
    if model not in MODELS:
        return jsonify(message="Model tidak dikenal", status="error"), 404

    cfg = MODELS[model]
    df, error = validate_forecast_csv(cfg["file"], cfg["allowed"])
    if error:
        return jsonify(message=error, status="error"), 400

    return jsonify(message=cfg["msg"], status="success", data=df.to_dict(orient="records"))


#Root
@app.route("/")
def root():
    return jsonify(message="Rainfall Prediction API is running", status="success")


if __name__ == "__main__":
   
    app.run(debug=True, port=5000)
