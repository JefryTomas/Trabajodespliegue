from pathlib import Path
import joblib
import pandas as pd

UMBRAL_BUENO = 6  # calidad >= 6 => bueno/malo

BASE_DIR = Path(__file__).resolve().parent
PACK_PATH = BASE_DIR / "wine_model_pack.pkl"

_pack = joblib.load(PACK_PATH)

# pack
modelo = _pack["modelo"]
scaler = _pack["scaler"]
cols_escalar = _pack["cols_escalar"]
cols_no_escalar = _pack["cols_no_escalar"]

# ✅ orden exacto de columnas con el que el modelo fue entrenado
FEATURES_MODEL = list(getattr(modelo, "feature_names_in_", []))
if not FEATURES_MODEL:
    FEATURES_MODEL = cols_escalar + cols_no_escalar


def predecir_vino(data: dict):
    # Permite JSON de 1 vino (dict)
    if not isinstance(data, dict):
        return {"ok": False, "msg": "El JSON debe ser un objeto (dict) con las columnas del vino."}

    df = pd.DataFrame([data])

    # Validar columnas faltantes
    faltan = [c for c in FEATURES_MODEL if c not in df.columns]
    if faltan:
        return {
            "ok": False,
            "msg": "Faltan columnas para predecir.",
            "faltan": faltan,
            "esperadas": FEATURES_MODEL
        }

    # ✅ Reordenar EXACTO como el modelo fue entrenado
    X = df.reindex(columns=FEATURES_MODEL).copy()

    # Convertir a numérico (por si llega como string)
    for c in FEATURES_MODEL:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Validar NaN (faltantes o inválidos)
    if X.isna().any().any():
        cols_nan = X.columns[X.isna().any()].tolist()
        return {
            "ok": False,
            "msg": "Hay valores inválidos o vacíos (no numéricos) en el JSON.",
            "columnas_con_problema": cols_nan
        }

    # Escalar SOLO las columnas numéricas seleccionadas
    X_scaled = X.copy()
    X_scaled[cols_escalar] = scaler.transform(X[cols_escalar])

    # ✅ Predecir como numpy para evitar problemas por nombres/orden
    calidad_pred = float(modelo.predict(X_scaled.to_numpy())[0])

    vino = "bueno" if calidad_pred >= UMBRAL_BUENO else "malo"

    return {
        "ok": True,
        "calidad_predicha": round(calidad_pred, 3),
        "vino": vino,
        "umbral": UMBRAL_BUENO
    }
