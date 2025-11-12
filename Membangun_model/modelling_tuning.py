# DeepLearning HyperTuning, Manual Logging (model than autolog)
# Up Ke Dagshub
# 2025-11-12 05:25:50 sampai 2025/11/12 05:27:32 (2 menit)
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor

load_dotenv() # untuk seting token Dagshub (auto handle username & password)

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

class SliceLayer(layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index]


# === MODEL BUILDER ===
def build_ncf_model(n_users, n_items, embed_dim=16, hidden=[32, 16, 8], lr=0.001):
    inputs = keras.Input(shape=(2,), name="user_item_input")

    user_input = SliceLayer(0)(inputs)
    item_input = SliceLayer(1)(inputs)
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
        n_users=n_users, n_items=n_items, embed_dim=embed_dim, hidden=hidden, lr=lr
    ),
    epochs=10,
    batch_size=32,
    verbose=0,
)

# === HYPERPARAMETER GRID (efisien) ===
param_grid = {
    "model__embed_dim": [16],
    "model__hidden": [[64, 32, 16], [32, 16, 8]],
    "model__lr": [0.001],
    "batch_size": [32],
    "epochs": [5, 8],
}

# === MLflow SETUP ===
# Set konfigurasi MLflow ke DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("NCF_ManualLogging")

with mlflow.start_run(run_name="GridSearch_ManualLogging"):

    # --- GridSearch dengan parallel processing
    grid = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1,  # Gunakan semua core CPU
    )

    # --- Fit tanpa validation_split (biar cepat)
    grid_result = grid.fit(X_train, y_train)

    # === AMBIL MODEL TERBAIK ===
    best_params = grid_result.best_params_
    best_score = grid_result.best_score_
    best_model = grid_result.best_estimator_

    # === EVALUASI DI TEST SET ===
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    eval_result = best_model.model_.evaluate(X_test, y_test, verbose=0)

    if isinstance(eval_result, (list, tuple)) and len(eval_result) == 3:
        test_loss, test_mae, test_mse = eval_result
    elif isinstance(eval_result, (list, tuple)) and len(eval_result) == 2:
        test_loss, test_mae = eval_result
        test_mse = np.nan
    else:
        test_loss = eval_result
        test_mae = np.nan
        test_mse = np.nan
    
    # === LOG SEMUA PARAMETER (setara autolog) ===
    mlflow.log_params(
        {
            "embed_dim": best_params["model__embed_dim"],
            "hidden_layers": best_params["model__hidden"],
            "learning_rate": best_params["model__lr"],
            "batch_size": best_params["batch_size"],
            "epochs": best_params["epochs"],
            "n_users": n_users,
            "n_items": n_items,
            "optimizer": "Adam",
            "loss": "mse",
            "metrics": ["mae", "mse"],
        }
    )

    # === LOG METRIK TRAINING (SETARA AUTOLOG) ===
    history = getattr(best_model.model_, "history", None)
    if history and hasattr(history, "history"):
        hist_dict = history.history
        for metric_name, values in hist_dict.items():
            if len(values) > 0:
                mlflow.log_metric(f"train_final_{metric_name}", float(values[-1]))
    else:
        # Fallback jika tidak ada history
        mlflow.log_metric("train_final_loss", np.nan)
        mlflow.log_metric("train_final_mae", np.nan)
        mlflow.log_metric("train_final_mse", np.nan)

    # === LOG 2 METRIK TAMBAHAN ===
    mlflow.log_metric("best_cv_neg_mse", best_score)
    mlflow.log_metric("test_rmse", rmse)

    # === LOG METRIK TEST SET (seperti autolog) ===
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)

    # === SIMPAN MODEL KE MLflow === bermasalah dengan dagshub
    # mlflow.keras.log_model(
    #     best_model.model_,
    #     artifact_path="model",
    #     # registered_model_name="NCF_Rekomendasi_Mahasiswa"
    # )
    mlflow.tensorflow.log_model(
        best_model.model_,  # scikeras wrapper punya atribut .model_
        name="model",
        registered_model_name="NCF_ManualLogging",
    )
    # == LOG Manual Model ==
    # best_model.model_.save("ncf_model_best.h5")
    # mlflow.log_artifact("ncf_model_best.h5", artifact_path="model")

    # === OUTPUT KE TERMINAL ===
    print("\n🏆 Best Params:", best_params)
    print(f"📉 Best Score (CV neg MSE): {best_score:.4f}")
    print(f"✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")

print("🚀 Training, manual logging, dan push ke DagsHub selesai.")
# print("\n🎯 MLflow experiment selesai. Cek di MLflow UI (http://127.0.0.1:5000/).")
