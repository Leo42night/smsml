# Submisi Membangun Sistem Machine Learning
Modul pembelajaran dari kelas [Dicoding](https://www.dicoding.com/): **Membangun Sistem Machine Learning**

## Kriteria
1. Basic, Skilled, Advance Terpenuhi: File automasi preprocessing (`preprocessing/`) mengembalikan data latih, run di Github Action save di repo.
2. Advance Terpenuhi: DeepLearning Model Tuning (`Membanung_model/modelling_tuning.py`) pakai MLFlow Track UI & save di DagsHub. Pakai autolog dan minimal 2 metrik tambahan.
3. Advance Terpenuhi: Automasi modelling (`MLProject/`) save ke Github repo dan DockerHub.
4. Advance Terpenuhi: 3 metrik Prometheus, 10 Metrik Grafana, 3 Alert Grafana.

## Cara Pakai
NOTE: Saya menggunakan WSL yang berbasis Linux (Window peru penyesuaian script untuk slash `/`)
### Setup
```bash
# setup lingkungan
conda create -n ncf-env python=3.12
conda activate ncf-env

# numpy pandas scikit-learn tensorflow scikeras mlflow uvicorn fastapi
pip install -r requirements.txt
# --- Didapat dari pip freeze > requirements.txt
# jika gagal pakai file requiremets.txt
conda env create -f environment.yml
# didapat dari conda env export > environment.yml 
# Jika masih gagal, manual install.
pip install numpy pandas scikit-learn tensorflow scikeras mlflow uvicorn fastapi
# warning: tensorflow butuh 600MB. Total size env adalah 3.0G
```

### Kriteria 1: Preprocessing
```bash
# test preprocessing (kriteria 1 Advance)
python preprocessing/automate_LeoPrangsT.py 
```
Ketika push ke Github (auto run Github Action, save ke Github Repo).

### Kriteria 2: Modelling
Perlu run Mlflow Server lebih dulu:
```bash
mlflow server --host 127.0.0.1 --port 5001
```
- Single modelling
```bash
# run script (mode Autolog)
# -- sklearn.randomForest
python Membangun_model/modelling.py 
# -- tensorflow
python Membangun_model/modelling-ncf.py 

```
- Hypertuning ManualLog, auto up ke dagshub. Gunakan contoh di `.env.production`, buat salinan di `.env`. Gunakan token Dagshub sebagai password.
```bash 
# run script tuning (tensorflow)
python Membangun_model/modelling_tuning.py 
```

### Kriteria 3: CI/CD
- Uji modelling di local `MLProject\modelling.py` (hapus folder `/mlruns` untuk reset config mlflow-artifacts URI, jika tidak akan dapat error Tracking URI [lihat **log.txt**]):
```bash
mlflow run MLProject --env-manager=local
# akan membuat folder dan data model mlruns/
# === Run (ID 'f5974476ca3740ae92a4e5d1fb34b62e') succeeded ===
```
- Uji docker di local:
```bash
# build image (5.6GB)
mlflow models build-docker \
  -m models:/CF_Mahasiswa_Sklearn/1 \
  -n <dockerhub_username>/cf_rekomendasi:latest

# run image (tidak perlu setup conda env, karena mlflow dan semua dependency load dari container)
# - berhasil jika tampil pesan: "INFO:     Application startup complete."
docker run -p 5000:8000 \
  --entrypoint mlflow \
  <dockerhub_username>/cf_rekomendasi:latest \
  models serve -m /opt/ml/model -h 0.0.0.0 -p 8000

# info: 5000 adalah host, 8000 adalah internal docker

# test inferensi (localhost komputer akses host docker)
curl -X POST http://127.0.0.1:5000/invocations \
-H "Content-Type: application/json" \
-d '{
      "dataframe_split": {
        "columns": ["user", "item"],
        "data": [[1, 208]]
      }
    }'

# (test) deploy manual to Docker Hub
docker login # koneksi device, buat DockerHub repo 'cf_rekomendasi'
docker push <dockerhub_username>/cf_rekomendasi:latest # makan waktu +10 menit (5.6GB)
```
- Sebelum push ke Repo Github, tambahkan secret di repo github:
  - **USERNAME** (username github)
  - **EMAIL** (email akun github)
  - **DOCKER_USERNAME**
  - **DOCKER_PASSWORD**

### Kriteria 4: Log & Alert
Beberapa server yang akan djalankan (**MLflow Model Serve** hanya opsi untuk test, dapat di-skip):
| Komponen                   | Perintah                                      | Fungsi                                       | Port   |
| -------------------------- | --------------------------------------------- | -------------------------------------------- | -------|
| **MLflow Tracking Server** | `mlflow server --host 127.0.0.1 --port 5001`  | UI untuk eksperimen, log, dan registry model | `5001` |
| *MLflow Model Serve*       | *`mlflow models serve ... -p 8000`*           | *Menyajikan model untuk prediksi (testing)*  |*`8000`*|
| **Flask Inference API**    | `python inference.py`                         | REST API untuk prediksi model                | `5000` |
| **Prometheus Exporter**    | `python prometheus_exporter.py`               | Setup endpoint `/metrics` untuk Prometheus   | `8000` |
| **Prometheus**             | `prometheus.exe --config.file=prometheus.yml` | Scrape metrics dari API lain                 | `9090` |
| **Grafana**                | (jalankan grafana)                            | Dashboard visualisasi                        | `3000` |

### 1. Run Model
Sebelum run model, run dulu mlflow tracking server:
```bash
conda activate ncf-env
mlflow server --host 127.0.0.1 --port 5001

# Set tracking URI lebih dulu
set MLFLOW_TRACKING_URI=http://127.0.0.1:5001 # atau lewat Setting Windows Environment Variable
# cek dengan echo %MLFLOW_TRACKING_URI%
```
PENTING: Karena model path C:\... valid di Windows, kita perlu run model serve di CMD/Powershell.
MLflow menyediakan dua pendekatan berbeda untuk menyajikan (serve) model:
| Cara                                | Command                                | Yang dilakukan                                                                                                                  | Kapan digunakan                                                          |
| ----------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **A. MLflow Model Serve (builtin)** | `mlflow models serve -m "models:/..."` | Menjalankan **Uvicorn/FastAPI server internal MLflow** yang otomatis memuat model dan expose endpoint `/invocations`            | Cepat untuk *demo/testing* atau *API sederhana*                          |
| **B. Flask/Custom Inference API**   | `python inference.py`                  | Kamu sendiri membuat server Flask, dan di dalamnya memanggil `mlflow.pyfunc.load_model()` untuk load model dari registry MLflow | Digunakan saat ingin punya API fleksibel, custom logic, monitoring, dll. |

- **(Opsional) Test Mlflow Model Serve**: Akan membuat REST API yang diakses melalui endpoint `/invocations`.
```bash
# Pilih versi (lihat di `mlruns\models\<nama_model>\version-x`). Silakan pilih diantara 2 model berikut:

# 1. Model CF: Random Forest
# Pastikan model sudah dibuat mengikuti kriteria 2 (modelling.py) atau Kriteria 3.
mlflow models serve -m "models:/CF_Mahasiswa_Sklearn/1" -p 8000 --no-conda

# 1. Model NCF (🌟 MODEL UTAMA)
# Pastikan model sudah dibuat mengikuti Kriteria 2 (**modelling-ncf** atau **modelling_tuning.py**)
mlflow models serve -m "models:/NCF_ManualLogging/1" -p 8000 --no-conda
```
Test API MLFlow Model Serve:
```bash
curl -X POST http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["user", "item"],
          "data": [[1, 208]]
        }
      }'
# {"predictions": [{"0": 0.89720219373703}]}(base)
```

- Server Inferensi
Server Inferensi menggunakan REST API dari server model **models:/NCF_ManualLogging/1** 
```bash 
python "Monitoring dan Logging/7.inference.py" 
# inferensi run in port 5000
```
Test API Server Inferensi:
```bash
curl -X GET http://127.0.0.1:5000
curl -X GET http://127.0.0.1:5000/metrics
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '[
        {"user": 1, "item": 10},
        {"user": 2, "item": 20},
        {"user": 3, "item": 30}
      ]'
# {"predictions":[0.7424633502960205,0.8271910548210144,0.7336715459823608]}
```

### 2. Prometheus
[Download](https://prometheus.io/download/) & setup path. Lihat Hasil: http://localhost:9090/targets
```bash
# sebelum run ini, pasitikan model server sudah run.
python "Monitoring dan Logging/3.prometheus_exporter.py"
C:\prometheus\prometheus-3.5.0.windows-amd64\prometheus.exe --config.file="Monitoring dan Logging\2.prometheus.yml"
```

### 2.a. Gambaran alur data

```
Prometheus (9090)
     ↑
     │  scrape every 10s
     │
Exporter (8000) ──> panggil inference API (5000)
                    kirim payload ke model
```
| Komponen                         | Port             | Peran                                            | Mengakses siapa                           |
| -------------------------------- | ---------------- | ------------------------------------------------ | ----------------------------------------- |
| 🧠 `inference.py`                | `5000`           | API model inference                              | menerima request dari client/exporter     |
| 📊 `3.prometheus_exporter.py`    | `8000`           | Menyediakan endpoint `/metrics` untuk Prometheus | mengirim request ke `inference.py` (5000) |
| 📈 Prometheus (`prometheus.exe`) | `9090` (default) | Mengumpulkan (scrape) data metrik                | mengakses `exporter` di `8000`            |

### 2.b. Metrik Utama

| Nama Metrik                      | Jenis         | Label       | Deskripsi                                                                                  | Contoh Output di Prometheus                                     |
| -------------------------------- | ------------- | ----------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `mlflow_request_count`           | **Counter**   | `endpoint`  | Menghitung total jumlah request yang diterima oleh endpoint tertentu (misalnya `/predict`) | `mlflow_request_count{endpoint="/predict"} 42`                  |
| `mlflow_request_latency_seconds` | **Histogram** | `endpoint`  | Mengukur distribusi waktu (latency) eksekusi permintaan per endpoint dalam detik           | `mlflow_request_latency_seconds_sum{endpoint="/predict"} 4.532` |
| `mlflow_model_output`            | **Gauge**     | (tidak ada) | Menyimpan nilai **prediksi terakhir** dari model MLflow                                    | `mlflow_model_output 0.8721`                                    |

### 3. grafana
- [Download program](https://grafana.com/grafana/download) & setup path.
- http://localhost:3000/ adalah default, jika tidak bisa, pakai port lain.
```bash
cd "C:\Program Files\GrafanaLabs\grafana\bin"
grafana-server.exe --homepath "C:\Program Files\GrafanaLabs\grafana"
# --homepath? Grafana butuh mengetahui home directory untuk menemukan file konfigurasi default.
``` 
- By default, kredensial masuk dengan username “admin” dan password “admin”.
> Grafana bisa saja otomatis berjalan di latar belakang, untuk mengeceknya anda dapat menjalankan `netstat -ano | findstr "<nomor_port>"` atau `netstat -ano | find /i "<nomor_port>"`

**📃 Note:** Alert hanya 1, karena ketika menambahkan alert kedua, dapat error uuid kosong (lihat **Log.txt**).

## Disk Usage Managemet
```bash
# check disk usave of some (env, etc.)
du -hs /<path>

# Info: Setiap pertama run `mlflow models serve` dengan model baru akan membuat environment python (Conda) baru dari conda.yaml
# Periksa env dan remove jika sudah yakin test berhasil dan ingin hemat ruang.
conda env list # lihat list env conda
conda remove --name <nama_env> --all
```
> Kita hanya punya 3 metrik dasar:
>
> * `mlflow_request_count`
> * `mlflow_request_latency_seconds`
> * `mlflow_model_output`
>
> Tapi dari kombinasi query PromQL, kamu bisa menurunkannya jadi **10 visualisasi berbeda** 🔥



## Tips
- Di CMD: Untuk terminate program jika **Ctrl+C** tidak bekerja, bisa pakai **Ctrl+Pause/Break**.
- Grafana berjalan di latar belakang, jika ingin restart (ada perubahan konfigurasi) atau ingin stop server nya, tinggal cari `services` pada pencarian windows.
