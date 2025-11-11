from prometheus_client import start_http_server, Summary, Counter, Gauge
import requests
import time

REQUEST_TIME = Summary(
    "inference_request_latency_seconds", "Latency untuk request inference"
)
REQUEST_COUNT = Counter("inference_requests_total", "Jumlah total request inference")
LAST_PREDICTION = Gauge("last_prediction_timestamp", "Waktu prediksi terakhir")

INFERENCE_URL = "http://localhost:5000/predict"


@REQUEST_TIME.time()
def make_request(payload):
    REQUEST_COUNT.inc()
    r = requests.post(INFERENCE_URL, json=payload)
    LAST_PREDICTION.set(time.time())
    return r


if __name__ == "__main__":
    start_http_server(8000)  # Prometheus scrape port
    while True:
        payload = [{"user": 1, "item": 1}]  # contoh input
        try:
            make_request(payload)
        except Exception as e:
            print("Inference gagal:", e)
        time.sleep(10)
