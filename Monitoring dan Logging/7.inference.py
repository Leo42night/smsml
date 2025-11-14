from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5001/")
# mlruns\models\<nama_model>\<version>
MODEL_URI = "models:/NCF_ManualLogging/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

# ==========================
# Prometheus metrics
# ==========================
REQUEST_COUNT = Counter(
    "mlflow_request_count", "Total number of requests", ["endpoint"]
)
REQUEST_LATENCY = Histogram(
    "mlflow_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
MODEL_OUTPUT = Gauge(
    "mlflow_model_output", "Predicted value of the model"
)

# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    import time
    start = time.time()
    
    data = request.json
    df = pd.DataFrame(data)
    preds = model.predict(df)
    preds = np.array(preds).flatten()  # pastikan jadi array float
    
    # Update metrics
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
    if len(preds) > 0:
        MODEL_OUTPUT.set(float(preds[0]))  # ambil prediksi pertama untuk contoh

    return jsonify({"predictions": preds.tolist()})

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}

@app.route("/", methods=["GET"])
def index():
    return "Server Inference Model berjalan!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
