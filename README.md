# Submisi Membangun Sistem Machine Learning
Modul pembelajaran dari kelas [Dicoding](https://www.dicoding.com/): **Membangun Sistem Machine Learning**

## Cara Pakai
```bash
# setup lingkungan
conda create -n ncf-env --file requirements.txt
conda activate ncf-env

# test preprocessing
python preprocessing\automate_LeoPrangsT.py

# jalankan Server mlflow (untuk Membangun_model/ dll.)
mlflow server --host 127.0.0.1 --port 5000

# test modelling
python Membangun_model\modelling.py
python Membangun_model\modelling_tuning.py

# test Monitoring dan Loging

# ---

# --- For New User/Computer ---
```