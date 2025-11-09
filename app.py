# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import re

app = Flask(__name__)
CORS(app)  # aktifkan CORS setelah app dibuat

UPLOAD_FOLDER = "./data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================
# Helpers
# ======================
def normalize_columns(df):
    """Normalisasi kolom: lowercase & semua non-alnum jadi underscore."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.strip('_')
    )
    return df

def find_columns(df, aliases):
    """Cari kolom sesuai alias (setelah normalisasi)."""
    found = {}
    cols_set = set(df.columns)
    for key, possible_names in aliases.items():
        col = next((name for name in possible_names if name in cols_set), None)
        if not col:
            return None, f"Kolom {key} tidak ditemukan. Kolom tersedia: {sorted(df.columns)}"
        found[key] = col
    return found, None

def _normalize_header(s: str) -> str:
    """Normalisasi nama kolom bebas: lowercase + underscore."""
    return re.sub(r'[^a-z0-9]+', '_', s.strip().lower()).strip('_')

def _clean_numeric(series):
    """Bersihkan nilai numerik: ganti koma→titik, buang karakter selain digit/.-eE, lalu to_numeric."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(',', '.', regex=False)
              .str.replace(r'[^0-9\.\-eE]+', '', regex=True)
              .str.strip(),
        errors="coerce"
    )

# ======================
# Upload Data Curah Hujan
# ======================
@app.route("/api/upload", methods=["POST"])
def upload_data():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    try:
        # Baca file excel/csv
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            return jsonify({"error": "Format file harus .xlsx atau .csv"}), 400

        df = normalize_columns(df)

        # Alias kolom (sudah ternormalisasi)
        aliases = {
            "tanggal": ["tanggal", "date", "hari", "tgl"],
            "curah_hujan": [
                "curah_hujan", "curah_hujan_mm", "curah_hujanmm",
                "rr", "rainfall", "rainfall_mm", "prec", "precip", "precipitation"
            ],
        }

        cols, err = find_columns(df, aliases)
        if err:
            return jsonify({"error": err}), 400

        # Ambil 2 kolom & standarkan untuk frontend
        df_out = df[[cols["tanggal"], cols["curah_hujan"]]].copy()
        df_out.columns = ["tanggal", "prec"]

        # Format tanggal & numerik
        try:
            # dayfirst=True untuk format ID (dd-mm-yyyy) bila ada
            df_out["tanggal"] = pd.to_datetime(df_out["tanggal"], dayfirst=True, errors="coerce")
            if df_out["tanggal"].isna().all():
                df_out["tanggal"] = df_out["tanggal"].astype(str)
            else:
                df_out["tanggal"] = df_out["tanggal"].dt.strftime("%Y-%m-%d")
        except Exception:
            df_out["tanggal"] = df_out["tanggal"].astype(str)

        df_out["prec"] = _clean_numeric(df_out["prec"])

        # Simpan
        save_path = os.path.join(UPLOAD_FOLDER, "uploaded_data.csv")
        df_out.to_csv(save_path, index=False)

        return jsonify({
            "message": "Upload berhasil",
            "status": "success",
            "data": df_out.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# Data Historis Endpoint
# ======================
@app.route("/api/prediksi", methods=["GET"])
def get_historical():
    try:
        filepath = os.path.join(UPLOAD_FOLDER, "uploaded_data.csv")
        if not os.path.exists(filepath):
            return jsonify({"error": "Belum ada data yang diupload"}), 404

        df = pd.read_csv(filepath)
        # Pastikan konsisten untuk frontend
        df.columns = ["tanggal", "prec"]
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


# ======================
# Validasi & Loader Forecast
# ======================
def validate_forecast_csv(filepath, allowed_sets):
    """Baca CSV dan cek apakah ADA salah satu skema allowed sebagai subset kolom CSV."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [_normalize_header(c) for c in df.columns]
        cols = set(df.columns)
        # diterima jika salah satu 'allowed' adalah subset dari kolom yang ada
        if any(allowed <= cols for allowed in allowed_sets):
            return df, None
        return None, f"Kolom CSV tidak sesuai (ditemukan): {list(df.columns)}"
    except FileNotFoundError:
        return None, f"File tidak ditemukan: {filepath}"
    except Exception as e:
        return None, str(e)

def _load_and_unify_forecast(model_key):
    """Baca CSV hasil model, lalu samakan kolom ke: tanggal, aktual, prediksi."""
    cfg = MODELS[model_key]
    df, error = validate_forecast_csv(cfg["file"], cfg["allowed"])
    if error:
        return None, error

    cols = set(df.columns)

    # Kandidat nama kolom generik
    date_candidates     = ["tanggal", "date", "time", "periode"]
    actual_candidates   = ["aktual", "actual", "actual_prec", "nilai_aktual", "y_true", "observed"]
    pred_candidates_map = {
        "arima":  ["prediksi", "arima_forecast", "arima_forecast_prec", "forecast", "prediction", "y_pred"],
        "lstm":   ["prediksi", "lstm_forecast", "forecast", "prediction", "y_pred", "nilai_prediksi"],
        "hybrid": ["prediksi", "hybrid_forecast", "hybrid_forecast_prec", "forecast", "prediction", "y_pred", "nilai_prediksi"],
    }

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    col_date = pick(date_candidates)
    col_act  = pick(actual_candidates)
    col_pred = pick(pred_candidates_map.get(model_key, ["prediksi", "forecast", "prediction", "y_pred"]))

    # Skema lawas spesifik → rename cepat
    rename_map = {}
    if {"tanggal", "aktual", "prediksi"} <= cols:
        pass
    elif {"date", "actual_prec", "arima_forecast_prec"} <= cols:
        rename_map = {"date": "tanggal", "actual_prec": "aktual", "arima_forecast_prec": "prediksi"}
    elif {"date", "actual_prec", "hybrid_forecast_prec"} <= cols:
        rename_map = {"date": "tanggal", "actual_prec": "aktual", "hybrid_forecast_prec": "prediksi"}
    elif {"date", "actual", "lstm_forecast"} <= cols:
        rename_map = {"date": "tanggal", "actual": "aktual", "lstm_forecast": "prediksi"}
    elif {"date", "actual", "hybrid_forecast"} <= cols:
        rename_map = {"date": "tanggal", "actual": "aktual", "hybrid_forecast": "prediksi"}
    # Atau pakai kandidat dinamis
    elif col_act and col_pred:
        if not col_date:
            # tidak ada tanggal → buat index 1..N
            df["tanggal"] = range(1, len(df) + 1)
            col_date = "tanggal"
        rename_map = {col_date: "tanggal", col_act: "aktual", col_pred: "prediksi"}
    else:
        return None, f"Tidak menemukan kolom aktual/prediksi. Kolom tersedia: {sorted(df.columns)}"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ambil kolom final & rapikan tipe
    df = df[["tanggal", "aktual", "prediksi"]].copy()

    # Tanggal → ke string YYYY-MM-DD (support dd-mm-yyyy via dayfirst)
    try:
        parsed = pd.to_datetime(df["tanggal"], dayfirst=True, errors="coerce")
        if parsed.isna().all():
            df["tanggal"] = df["tanggal"].astype(str)
        else:
            df["tanggal"] = parsed.dt.strftime("%Y-%m-%d")
    except Exception:
        df["tanggal"] = df["tanggal"].astype(str)

    # Bersihkan numerik: ini mencegah 'prediksi' jadi NaN akibat koma/teks
    df["aktual"]   = _clean_numeric(df["aktual"])
    df["prediksi"] = _clean_numeric(df["prediksi"])

    return df, None

# factory endpoint untuk mendapatkan prediksi berdasarkan model
@app.route("/api/prediksi/<model>", methods=["GET"])
def get_forecast(model):
    if model not in MODELS:
        return jsonify(message="Model tidak dikenal", status="error"), 404

    df, error = _load_and_unify_forecast(model)  
    if error:
        return jsonify(message=error, status="error"), 400

    df["aktual"] = df["aktual"].round(2)
    df["prediksi"] = df["prediksi"].round(2)
    data = df.where(pd.notnull(df), None).to_dict(orient="records")
    return jsonify(message=MODELS[model]["msg"], status="success", data=data)


# Alias agar cocok dengan Laravel: POST/GET /predict/<model>
@app.route("/predict/<model>", methods=["GET", "POST"])
def predict_alias(model):
    if model not in MODELS:
        return jsonify(message="Model tidak dikenal", status="error"), 404

    df, error = _load_and_unify_forecast(model)
    if error:
        return jsonify(message=error, status="error"), 400

    return jsonify(message=MODELS[model]["msg"], status="success", data=df.to_dict(orient="records"))

# ======================
# Root
# ======================
@app.route("/")
def root():
    return jsonify(message="Rainfall Prediction API is running", status="success")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8001)
