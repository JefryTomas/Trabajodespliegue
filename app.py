from flask import Flask, request, jsonify
from models.predicion import predecir, ejemplo_input

app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify(ejemplo_input()), 200

    data = request.get_json(silent=True)
    result = predecir(data)
    return jsonify(result), (200 if result.get("ok") else 400)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
