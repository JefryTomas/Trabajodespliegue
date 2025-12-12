# models/predicion.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# En tu caso, parece que ganar = 0 (si reentrenas y cambia, pon 1)
LABEL_GANAR = 0

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_futbol.pkl"

_pack = joblib.load(MODEL_PATH)
pipeline = _pack["pipeline"] if isinstance(_pack, dict) and "pipeline" in _pack else _pack

num_features = _pack.get("num_features") if isinstance(_pack, dict) else None
cat_features = _pack.get("cat_features") if isinstance(_pack, dict) else None

FEATURES = None
if num_features and cat_features:
    FEATURES = list(num_features) + list(cat_features)
else:
    FEATURES = list(getattr(pipeline, "feature_names_in_", [])) or None


def _json_safe(x):
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_json_safe(i) for i in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_json_safe(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return x


def _get_classes():
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        try:
            classes = pipeline[-1].classes_
        except Exception:
            classes = None
    return list(classes) if classes is not None else None


def _prob_label(df, label):
    if not hasattr(pipeline, "predict_proba"):
        return None

    probs = pipeline.predict_proba(df)[0]
    classes = _get_classes()

    if classes and label in classes:
        return float(probs[classes.index(label)])

    # fallback binario típico [0,1]
    if len(probs) >= 2:
        return float(probs[0]) if label == 0 else float(probs[1])

    return None


def predecir(data: dict):
    if not isinstance(data, dict):
        return {"ok": False, "msg": "Manda un JSON válido (objeto)."}

    if FEATURES:
        faltan = [c for c in FEATURES if c not in data]
        if faltan:
            return {"ok": False, "msg": f"Faltan campos: {faltan}"}
        df = pd.DataFrame([{c: data.get(c) for c in FEATURES}])
    else:
        df = pd.DataFrame([data])

    try:
        pred = pipeline.predict(df)[0]
        pred_int = int(pred) if isinstance(pred, (np.integer, int)) else int(pred)

        prob_ganar = _prob_label(df, LABEL_GANAR)
        gana = (pred_int == LABEL_GANAR)

        out = {
            "ok": True,
            "resultado": "TIENE posibilidad de ganar" if gana else "NO tiene posibilidad de ganar",
            "prediccion": pred_int,
            "prob_ganar": prob_ganar,
            "porcentaje_ganar": None if prob_ganar is None else round(prob_ganar * 100, 2),
        }
        return _json_safe(out)

    except Exception as e:
        return {"ok": False, "msg": "Error prediciendo", "error": str(e)}


def ejemplo_input():
    """
    GET /predict -> ejemplo NO quemado:
    - Si el pipeline tiene scaler/encoder, usa medias + categoría conocida.
    - Si no, devuelve 0s con las llaves correctas.
    """
    if not FEATURES:
        return {}

    ejemplo = {k: 0 for k in FEATURES}

    try:
        pre = getattr(pipeline, "named_steps", {}).get("preprocess", None)

        # num: medias del entrenamiento
        if pre is not None and num_features:
            num_tr = pre.named_transformers_.get("num", None)
            if num_tr is not None and hasattr(num_tr, "mean_"):
                for col, mean in zip(list(num_features), list(num_tr.mean_)):
                    ejemplo[col] = round(float(mean), 4)

        # cat: primera categoría conocida
        if pre is not None and cat_features:
            cat_tr = pre.named_transformers_.get("cat", None)
            if cat_tr is not None and hasattr(cat_tr, "categories_"):
                cats = cat_tr.categories_[0]
                if len(cats) > 0:
                    ejemplo[list(cat_features)[0]] = str(cats[0])

    except Exception:
        pass

    return _json_safe(ejemplo)
