from flask import Flask, request, jsonify
from models.predicion import predecir_vino, FEATURES_MODEL

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/features")
def features():
    # ✅ devuelve las columnas en el orden EXACTO que el modelo espera
    return jsonify({"ok": True, "features": FEATURES_MODEL})

@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"ok": False, "msg": "Debes enviar JSON"}), 400

    data = request.get_json()
    result = predecir_vino(data)

    status = 200 if result.get("ok") else 400
    return jsonify(result), status

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
