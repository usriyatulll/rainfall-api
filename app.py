# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import re
from functools import lru_cache

try:
    import joblib
except Exception:
    joblib = None

try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
except Exception:
    SARIMAXResults = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    MinMaxScaler = None

app = Flask(__name__)
CORS(app)  # aktifkan CORS setelah app dibuat

UPLOAD_FOLDER = "./data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helpers umum
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
def _use_uploaded_dates(df):
    """
    Ganti 'tanggal' dengan tanggal dari uploaded_data.csv jika:
    - kolom 'tanggal' tidak ada, atau
    - nilainya angka murni (1,2,3,...) atau banyak kosong/'-'.
    Diselaraskan dari BELAKANG agar pairing akhir-akhir tetap sesuai.
    """
    try:
        up_dates, _ = _uploaded_series()  # tanggal & aktual dari upload user
    except Exception:
        return df  # kalau belum ada upload, biarkan

    need = len(df)

    if "tanggal" not in df.columns:
        if len(up_dates) >= need:
            df["tanggal"] = up_dates[-need:]
        else:
            df["tanggal"] = list(range(1, need + 1))
        return df

    cand = df["tanggal"].astype(str).str.strip()
    all_numeric = cand.str.fullmatch(r"\d+").all()
    many_empty  = cand.isin(["", "-", "nan", "NaT"]).mean() > 0.3  # >30% kosong

    if all_numeric or many_empty:
        if len(up_dates) >= need:
            df["tanggal"] = up_dates[-need:]
        else:
            # kalau upload lebih pendek, pad di depan dengan index
            pad = [str(i) for i in range(1, need - len(up_dates) + 1)]
            df["tanggal"] = pad + up_dates

    return df

# Upload Data Curah Hujan
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


# ======================
# Konfigurasi file model & hasil prediksi
# ======================
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
# Validasi & Loader Forecast CSV → unified kolom
# ======================
def validate_forecast_csv(filepath, allowed_sets):
    """Baca CSV dan cek apakah ADA salah satu skema allowed sebagai subset kolom CSV."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [_normalize_header(c) for c in df.columns]
        cols = set(df.columns)
        if any(allowed <= cols for allowed in allowed_sets):
            return df, None
        return None, f"Kolom CSV tidak sesuai (ditemukan): {list(df.columns)}"
    except FileNotFoundError:
        return None, f"File tidak ditemukan: {filepath}"
    except Exception as e:
        return None, str(e)

def _load_and_unify_forecast(model_key):
    cfg = MODELS[model_key]
    df, error = validate_forecast_csv(cfg["file"], cfg["allowed"])
    if error:
        return None, error

    # normalisasi header DULU
    df.columns = [_normalize_header(c) for c in df.columns]

    # buang kolom 'no' bila ada
    if "no" in df.columns:
        df = df.drop(columns=["no"])

    cols = set(df.columns)

    # kandidat kolom
    date_candidates   = ["tanggal", "date", "time", "periode"]
    actual_candidates = ["aktual", "actual", "actual_prec", "nilai_aktual", "y_true", "observed"]
    pred_candidates   = [
        "prediksi","forecast","prediction","y_pred",
        "arima_forecast","arima_forecast_prec",
        "lstm_forecast","lstm_forecast_prec",
        "hybrid_forecast","hybrid_forecast_prec",
        "curah_hujan_prediksi_mm","nilai_prediksi"
    ]

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    col_date = pick(date_candidates)
    col_pred = pick(pred_candidates)
    col_act  = pick(actual_candidates)

    rename_map = {}

    # skema spesifik lama (tetap didukung)
    if {"tanggal","aktual","prediksi"} <= cols:
        pass
    elif {"date","actual_prec","arima_forecast_prec"} <= cols and model_key=="arima":
        rename_map = {"date":"tanggal","actual_prec":"aktual","arima_forecast_prec":"prediksi"}
    elif {"date","actual_prec","hybrid_forecast_prec"} <= cols and model_key=="hybrid":
        rename_map = {"date":"tanggal","actual_prec":"aktual","hybrid_forecast_prec":"prediksi"}
    elif {"date","actual","lstm_forecast"} <= cols and model_key=="lstm":
        rename_map = {"date":"tanggal","actual":"aktual","lstm_forecast":"prediksi"}

    # HANYA PREDIKSI (kasus tabel kamu)
    elif col_pred and not col_act:
        if not col_date:
            try:
                _dates, _ = _uploaded_series()
                df["tanggal"] = _dates[-len(df):] if len(_dates) >= len(df) else list(range(1,len(df)+1))
            except Exception:
                df["tanggal"] = list(range(1,len(df)+1))
            col_date = "tanggal"
        rename_map = {col_date:"tanggal", col_pred:"prediksi"}

    # ADA AKTUAL & PREDIKSI
    elif col_pred and col_act:
        if not col_date:
            df["tanggal"] = list(range(1,len(df)+1))
            col_date = "tanggal"
        rename_map = {col_date:"tanggal", col_pred:"prediksi", col_act:"aktual"}

    else:
        return None, f"Tidak menemukan kolom prediksi. Kolom: {sorted(df.columns)}"

    if rename_map:
        df = df.rename(columns=rename_map)

    # pastikan ada kolom 'prediksi'
    if "prediksi" not in df.columns:
        return None, "Kolom prediksi tidak ditemukan setelah normalisasi."

    # isi tanggal kosong/angka dengan tanggal upload (force gunakan upload dates)
    df = _use_uploaded_dates(df)

    # siapkan aktual: kalau belum ada, ambil dari upload (selaraskan dari belakang)
    try:
        _dates, _y = _uploaded_series()
    except Exception:
        _dates, _y = None, None

    if "aktual" not in df.columns:
        if _y is not None and len(_y) >= len(df):
            df["aktual"] = np.asarray(_y[-len(df):], dtype=float)
        else:
            df["aktual"] = np.nan

    # parse & format tanggal
    try:
        parsed = pd.to_datetime(df["tanggal"], dayfirst=True, errors="coerce")
        df["tanggal"] = parsed.dt.strftime("%Y-%m-%d").fillna(df["tanggal"].astype(str))
    except Exception:
        df["tanggal"] = df["tanggal"].astype(str)

    # rapikan numerik
    df["aktual"]   = _clean_numeric(df["aktual"])
    df["prediksi"] = _clean_numeric(df["prediksi"])

    # final
    df = df[["tanggal","aktual","prediksi"]].copy()
    return df, None




# ======================
# Loader data upload & model tersimpan
# ======================
def _uploaded_series():
    """
    Baca ./data/uploaded_data.csv → (dates, y) untuk kolom 'prec'.
    """
    fp = os.path.join(UPLOAD_FOLDER, "uploaded_data.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError("Belum ada data yang diupload")
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"tanggal", "prec"} <= set(df.columns):
        raise ValueError(f"Skema uploaded_data.csv tidak valid: {df.columns}")
    dates = df["tanggal"].astype(str).tolist()
    y = pd.to_numeric(df["prec"], errors="coerce").fillna(0.0).values.astype(float)
    return dates, y

@lru_cache(maxsize=1)
def _load_arima_model():
    """
    Load model ARIMA tersimpan dari ./models/best_arima_model.pkl
    Support pmdarima atau statsmodels SARIMAXResults.
    """
    path = "./models/best_arima_model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model ARIMA tidak ditemukan: {path}")
    if joblib is None:
        raise RuntimeError("joblib tidak tersedia, install joblib.")
    m = joblib.load(path)
    return m

@lru_cache(maxsize=1)
def _load_lstm_model():
    """
    Load Keras LSTM + scaler (opsional).
    - File model: ./models/lstm_model.h5
    - Scaler (opsional): ./models/scaler_minmax.pkl
    """
    model_path = "./models/lstm_model.h5"
    scaler_path = "./models/scaler_minmax.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model LSTM tidak ditemukan: {model_path}")
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras belum terpasang.")

    model = load_model(model_path)
    scaler = None
    if os.path.exists(scaler_path) and joblib is not None:
        scaler = joblib.load(scaler_path)

    return model, scaler

@lru_cache(maxsize=1)
def _load_hybrid_models():
    """
    Hybrid = ARIMA residual → LSTM.
    - ARIMA: best_arima_model.pkl
    - LSTM residual: ./models/lstm_residual_model.h5
    - Scaler residual (opsional): ./models/residual_scaler.pkl
    """
    arima = _load_arima_model()
    lstm_path = "./models/lstm_residual_model.h5"
    scaler_path = "./models/residual_scaler.pkl"
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"Model LSTM residual tidak ditemukan: {lstm_path}")
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras belum terpasang.")
    lstm = load_model(lstm_path)

    scaler = None
    if os.path.exists(scaler_path) and joblib is not None:
        scaler = joblib.load(scaler_path)

    return arima, lstm, scaler


# ======================
# Prediksi helper per model
# ======================
def _predict_arima_on(y):
    """
    Kembalikan y_hat (panjang sama dengan y) dari model ARIMA tersimpan.
    Best-effort untuk pmdarima/statsmodels. Fallback: rolling mean 3.
    """
    m = _load_arima_model()
    y = np.asarray(y, dtype=float)

    # pmdarima?
    if pm is not None and hasattr(m, "predict_in_sample"):
        try:
            # re-fit ringan supaya domain cocok dengan data upload
            order = getattr(m, "order", (1,0,0))
            seasonal_order = getattr(m, "seasonal_order", (0,0,0,0))
            m2 = pm.arima.ARIMA(order=order, seasonal_order=seasonal_order)
            m2.fit(y)
            y_hat = m2.predict_in_sample(start=0, end=len(y)-1)
            return np.asarray(y_hat, dtype=float)
        except Exception:
            pass

    # statsmodels SARIMAXResults?
    if SARIMAXResults is not None and hasattr(m, "model"):
        try:
            res = m
            res2 = res.model.smooth(res.model.filter(y).params)
            fv = np.asarray(res2.fittedvalues, dtype=float)
            if len(fv) < len(y):
                fv = np.pad(fv, (len(y)-len(fv), 0), mode='edge')
            else:
                fv = fv[-len(y):]
            return fv
        except Exception:
            pass

    # fallback: rolling mean
    if len(y) >= 3:
        return pd.Series(y).rolling(3, min_periods=1).mean().values
    return y.astype(float)

def _series_to_windows(arr, win=7):
    X, y = [], []
    for i in range(len(arr) - win + 1):
        X.append(arr[i:i+win])
        y.append(arr[i+win-1])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((-1, win, 1)), y

def _predict_lstm_on(y, window=7):
    """
    Inferensi LSTM one-step (pseudo fitted):
    - skala dengan scaler jika ada; jika tidak, fit ke data upload.
    """
    model, scaler = _load_lstm_model()
    if MinMaxScaler is None and scaler is None:
        raise RuntimeError("scikit-learn tidak tersedia untuk scaler.")
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler()
        y_sc = scaler.fit_transform(y)
    else:
        y_sc = scaler.transform(y)

    X, _ = _series_to_windows(y_sc.flatten(), window)
    if X.size == 0:
        # data terlalu pendek → semua prediksi = nilai pertama
        return np.repeat(float(y[0]), len(y))
    y_hat_sc = model.predict(X, verbose=0).reshape(-1)
    y_hat_sc_full = np.concatenate([np.repeat(y_hat_sc[0], window-1), y_hat_sc])
    y_hat = scaler.inverse_transform(y_hat_sc_full.reshape(-1,1)).flatten()
    return y_hat

def _predict_hybrid_on(y, window=7):
    """
    Hybrid: y_hat = y_arima + y_residual_lstm
    """
    arima, lstm_res, scaler = _load_hybrid_models()
    if MinMaxScaler is None and scaler is None:
        raise RuntimeError("scikit-learn tidak tersedia untuk scaler residual.")
    y = np.asarray(y, dtype=float)

    # ARIMA component
    y_arima = _predict_arima_on(y)

    # residual
    resid = (y - y_arima).reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler()
        resid_sc = scaler.fit_transform(resid)
    else:
        resid_sc = scaler.transform(resid)

    Xr, _ = _series_to_windows(resid_sc.flatten(), window)
    if Xr.size == 0:
        res_hat = np.zeros_like(y)
    else:
        res_hat_sc = lstm_res.predict(Xr, verbose=0).reshape(-1)
        res_hat_sc_full = np.concatenate([np.repeat(res_hat_sc[0], window-1), res_hat_sc])
        res_hat = scaler.inverse_transform(res_hat_sc_full.reshape(-1,1)).flatten()

    return y_arima + res_hat


# ======================
# Endpoint MENJALANKAN model & simpan CSV hasil
# ======================
@app.route("/api/run/<model>", methods=["POST"])
def run_model(model):
    """
    Jalankan model tersimpan terhadap data upload, simpan CSV hasil ke ./models/*_forecast_results.csv
    model: arima | lstm | hybrid | all
    """
    if model not in MODELS and model != "all":
        return jsonify(message="Model tidak dikenal", status="error"), 404

    try:
        dates, y = _uploaded_series()
    except Exception as e:
        return jsonify(message=str(e), status="error"), 400

    results = {}

    def _save(df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    # ARIMA
    try:
        if model in ("arima", "all"):
            y_hat = _predict_arima_on(y)
            df = pd.DataFrame({
                "date": dates,
                "actual_prec": np.asarray(y, dtype=float),
                "arima_forecast_prec": np.asarray(y_hat, dtype=float)
            })
            _save(df, "./models/arima_forecast_results.csv")
            results["arima"] = df.tail(5).to_dict(orient="records")
    except Exception as e:
        results["arima_error"] = str(e)

    # LSTM
    try:
        if model in ("lstm", "all"):
            y_hat = _predict_lstm_on(y, window=7)
            df = pd.DataFrame({
                "date": dates,
                "actual_prec": np.asarray(y, dtype=float),
                "lstm_forecast_prec": np.asarray(y_hat, dtype=float)
            })
            _save(df, "./models/lstm_result.csv")
            results["lstm"] = df.tail(5).to_dict(orient="records")
    except Exception as e:
        results["lstm_error"] = str(e)

    # Hybrid
    try:
        if model in ("hybrid", "all"):
            y_hat = _predict_hybrid_on(y, window=7)
            df = pd.DataFrame({
                "date": dates,
                "actual_prec": np.asarray(y, dtype=float),
                "hybrid_forecast_prec": np.asarray(y_hat, dtype=float)
            })
            _save(df, "./models/hybrid_forecast_results.csv")
            results["hybrid"] = df.tail(5).to_dict(orient="records")
    except Exception as e:
        results["hybrid_error"] = str(e)

    if not results:
        return jsonify(message="Gagal menjalankan model apa pun", status="error"), 500

    return jsonify(message="Berhasil menjalankan model", status="success", results=results)


# ======================
# Endpoint pembaca hasil (untuk Laravel)
# ======================
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
# buat data blm/udh diup
@app.get("/api/data/status")
def data_status():
    # anggap “ada upload” jika ada minimal 1 file csv/xlsx di folder data
    has_upload = False
    num_files = 0
    if os.path.isdir(UPLOAD_FOLDER):
        for fn in os.listdir(UPLOAD_FOLDER):
            _, ext = os.path.splitext(fn.lower())
            if ext in ALLOWED_EXT:
                has_upload = True
                num_files += 1
                break
    return jsonify({"has_upload": has_upload, "num_files": num_files})

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
    # ganti port jika perlu
    app.run(host="0.0.0.0", debug=True, port=8001)
    