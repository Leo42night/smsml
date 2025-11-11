from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

MODEL_URI = "runs:/<RUN_ID>/model"  # ganti RUN_ID
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    preds = model.predict(df).tolist()
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)