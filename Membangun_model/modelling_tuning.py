import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

# === PATH FILE DATA ===
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

file_path = os.path.join(base_dir, "nilai_mahasiswa-preprocessed.csv")
print(f"✅ File CSV: {file_path}")
df = pd.read_csv(file_path)

# === SPLIT DATA ===
X = df[["user", "item"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === MODEL BUILDER ===
def build_ncf_model(n_users, n_items, embed_dim=16, hidden=[32, 16, 8], lr=0.001):
    inputs = keras.Input(shape=(2,), name="user_item_input")

    user_input = layers.Lambda(lambda x: x[:, 0])(inputs)
    item_input = layers.Lambda(lambda x: x[:, 1])(inputs)
    user_input = layers.Reshape((1,))(user_input)
    item_input = layers.Reshape((1,))(item_input)

    user_emb = layers.Embedding(n_users, embed_dim)(user_input)
    item_emb = layers.Embedding(n_items, embed_dim)(item_input)

    user_vec = layers.Flatten()(user_emb)
    item_vec = layers.Flatten()(item_emb)
    x = layers.Concatenate()([user_vec, item_vec])

    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)

    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# === HITUNG JUMLAH USER DAN ITEM ===
n_users = df["user"].nunique()
n_items = df["item"].nunique()

# === WRAPPER KE KERASREGRESSOR ===
regressor = KerasRegressor(
    model=lambda embed_dim, hidden, lr: build_ncf_model(
        n_users=n_users,
        n_items=n_items,
        embed_dim=embed_dim,
        hidden=hidden,
        lr=lr
    ),
    epochs=10,
    batch_size=32,
    verbose=0
)

# === HYPERPARAMETER GRID ===
# 2025-11-11 11:15:17 sampai 2025/11/11 11:17:19 (2 menit)
param_grid = {
    "model__embed_dim": [16],
    "model__hidden": [[64, 32, 16], [32, 16, 8]],
    "model__lr": [0.001],
    "batch_size": [32],
    "epochs": [5, 8]
}

# === MLflow SETUP ===
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("NCF_Rekomendasi_Mahasiswa")

# === DISABLE AUTOLOG (opsional jika kamu pakai manual log) ===
mlflow.keras.autolog(log_models=False)

with mlflow.start_run(run_name="GridSearch_Optimized"):

    # --- GridSearch dengan parallel processing
    grid = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1  # Gunakan semua core CPU
    )

    # --- Fit tanpa validation_split (biar cepat)
    grid_result = grid.fit(X_train, y_train)

    # --- Ambil hasil terbaik
    best_params = grid_result.best_params_
    best_score = grid_result.best_score_
    best_model = grid_result.best_estimator_

    # --- Evaluasi di test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # === Logging ke MLflow ===
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_neg_mse", best_score)
    mlflow.log_metric("test_rmse", rmse)

    # Simpan model terbaik
    mlflow.keras.log_model(
        best_model.model_,
        artifact_path="model",
        registered_model_name="NCF_Rekomendasi_Mahasiswa"
    )

    print("\n🏆 Best Params:", best_params)
    print(f"📉 Best Score (neg MSE): {best_score:.4f}")
    print(f"✅ Test RMSE: {rmse:.4f}")

print("\n🎯 MLflow experiment selesai. Cek di MLflow UI (http://127.0.0.1:5000/).")
