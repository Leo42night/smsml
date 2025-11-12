# Submisi Membangun Sistem Machine Learning
Modul pembelajaran dari kelas [Dicoding](https://www.dicoding.com/): **Membangun Sistem Machine Learning**

## Kriteria
1. Basic, Skilled, Advance Terpenuhi: File automasi preprocessing (`preprocessing/`) mengembalikan data latih, run di Github Action save di repo.
2. Advance Terpenuhi: DeepLearning Model Tuning (`Membanung_model/modelling_tuning.py`) pakai MLFlow Track UI & save di DagsHub. Pakai autolog dan minimal 2 metrik tambahan.
3. Advance Terpenuhi: Automasi modelling (`MLProject/`) save ke Github repo dan DockerHub
4. Advance Terpenuhi: 

## Cara Pakai
### Setup
```bash
# setup lingkungan
conda create -n ncf-env python=3.12
conda activate ncf-env

# numpy pandas scikit-learn tensorflow scikeras mlflow uvicorn fastapi
pip install -r requirements.txt
```

### Kriteria 1: Preprocessing
```bash
# test preprocessing (kriteria 1 Advance)
python preprocessing\automate_LeoPrangsT.py
```
Ketika push ke Github (auto run Github Action, save ke Github Repo).

### Kriteria 2: Modelling
```bash
# jalankan Server mlflow (dibutuhkan ketika run script Membangun_model/[modelling-ncf.py, modelling.py] dan run model server NCF)
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Single modelling (model sklearn.random rorest & Neural Network)
# -- pakai Autolog
python Membangun_model\modelling.py
python Membangun_model\modelling-ncf.py

# Hypertuning ManualLog, auto up ke dagshub 
python Membangun_model\modelling_tuning.py
```

### Kriteria 3: CI/CD
`MLProject\modelling.py` run di GithubAction (local windows bermasalah [Lihat **Log 1**]). 
- Github action akan menjalankan simpan model di repo Github dan Docker Hub. 
- Anda pelu login ke docker hub dan install docker (bagian ini anda perlu setup sendiri, saya menggunakan Docker Desktop for Windows).
- Uji docker di local:
```bash
# build image
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
1. Run Model
```bash
# Model CF (pilih versi) 
mlflow models serve -m "models:/CF_Mahasiswa_Sklearn/1" -p 8080 --no-conda

# Model NCF. perlu set tracking URI lebih dulu
# di cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000 
# cek dengan echo %MLFLOW_TRACKING_URI%
# pilih versi
mlflow models serve -m "models:/NCF_ManualLogging/1" -p 8080 --no-conda

# Pakai File Server Inference
python "Monitoring dan Logging\7.inference.py" 
```

2. test inference (run di terminal basis unix)
```bash
curl -X POST http://127.0.0.1:8080/invocations \
-H "Content-Type: application/json" \
-d '{
      "dataframe_split": {
        "columns": ["user", "item"],
        "data": [[1, 208]]
      }
    }'
# contoh hasil
{"predictions": [{"0": 0.89720219373703}]}(base)
```

## Log

1. `MLProject\modelling.py` belum bisa run di local windows (`conda activate ncf-env && mlflow run MLProject --env-manager=local`) karena track_uri tidak bisa detect path ini, tapi aman di github action yang pakai ubuntu.
```bash
mlflow.exceptions.MlflowException: When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI, but it was currently set to file:///C:/Users/.../SMSML_LeoPrangsT/mlruns. Perhaps you forgot to set the tracking URI to the running MLflow server. To set the tracking URI, use either of the following methods:
1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. `export MLFLOW_TRACKING_URI=http://localhost:5000`
2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. `mlflow.set_tracking_uri('http://localhost:5000')
```