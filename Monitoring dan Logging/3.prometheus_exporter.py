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
    print("🚀 Prometheus exporter berjalan di http://localhost:8000/metrics")
    print("📡 Mengirim request ke inference API setiap 10 detik...\n")
    
    while True:
        payload = [{"user": 1, "item": 15},
                   {"user": 20, "item": 50},
                   {"user": 50, "item": 20}]  # contoh input
        try:
            response = make_request(payload)
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Request sukses - Status: {response.status_code}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Inference gagal:", e)
        time.sleep(10)
