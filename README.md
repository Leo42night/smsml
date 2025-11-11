# Submisi Membangun Sistem Machine Learning
Modul pembelajaran dari kelas [Dicoding](https://www.dicoding.com/): **Membangun Sistem Machine Learning**

## Kriteria
1. Basic, Skilled, Advance Terpenuhi: File automasi preprocessing (`preprocessing/`) mengembalikan data latih, run di Github Action save di repo.
2. Advance Terpenuhi: DeepLearning Model Tuning (`Membanung_model/modelling_tuning.py`) pakai MLFlow Track UI & save di DagsHub. Pakai autolog dan minimal 2 metrik tambahan.
3. Advance Terpenuhi: Automasi modelling (`MLProject/`) save ke Github repo dan DockerHub
4. Advance Terpenuhi: 

## Cara Pakai
```bash
# setup lingkungan
conda create -n ncf-env python=3.12
conda activate ncf-env
# numpy pandas scikit-learn tensorflow scikeras mlflow
pip install -r requirements.txt

# test preprocessing (kriteria 1 Advance)
python preprocessing\automate_LeoPrangsT.py

# jalankan Server mlflow (untuk Membangun_model/)
mlflow server --host 127.0.0.1 --port 5000

# test modelling (kriteria 2 Advance)
python Membangun_model\modelling.py
python Membangun_model\modelling_tuning.py

# test Monitoring dan Loging

# Push ke github (auto run Github Action)
# Kriteria 1: run preprocessing\automate_LeoPrangsT.py
# Kriteria 3: run MLProject\modelling.py 


# ---

# --- For New User/Computer ---
```

## Log
- `MLProject\modelling.py` belum bisa run di local windows (`conda activate ncf-env && mlflow run MLProject --env-manager=local`) karena track_uri tidak bisa detect path ini, tapi aman di github action yang pakai ubuntu
```
mlflow.exceptions.MlflowException: When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI, but it was currently set to file:///C:/.../SMSML_LeoPrangsT/mlruns. Perhaps you forgot to set the tracking URI to the running MLflow server. To set the tracking URI, use either of the following methods:
1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. `export MLFLOW_TRACKING_URI=http://localhost:5000`
2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. `mlflow.set_tracking_uri('http://localhost:5000')
```