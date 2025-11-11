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
conda create -n ncf-env --file requirements.txt
conda activate ncf-env

# test preprocessing (kriteria 1 Advance)
python preprocessing\automate_LeoPrangsT.py

# jalankan Server mlflow (untuk Membangun_model/)
mlflow server --host 127.0.0.1 --port 5000

# test modelling (kriteria 2 Advance)
python Membangun_model\modelling.py
python Membangun_model\modelling_tuning.py

# test Monitoring dan Loging

# Push ke github (auto run Github Action)
# Kriteria 3
# ---

# --- For New User/Computer ---
```