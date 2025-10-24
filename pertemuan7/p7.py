# ANN_WORKFLOW.PY
# Artificial Neural Network untuk Klasifikasi Kelulusan Mahasiswa
# Run: python ann_workflow.py | Output: model_ann.h5 + plots + report

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# -----------------------
# KONFIGURASI & SEED
# -----------------------
print("🚀 Starting ANN Workflow...")
tf.random.set_seed(42)
np.random.seed(42)
RANDOM_STATE = 42
SAVE_PLOTS = True

# -----------------------
# LANGKAH 1 — SIPKAN DATA
# -----------------------
print("\n📊 Langkah 1: Data Preparation")
if os.path.exists("pertemuan5/processed_kelulusan.csv"):
    df = pd.read_csv("pertemuan5/processed_kelulusan.csv")
    X = df.drop("Lulus", axis=1)
    y = df["Lulus"]
    print(f"Dataset: {df.shape} | Class: {y.value_counts().to_dict()}")
else:
    raise FileNotFoundError("Download pertemuan5/processed_kelulusan.csv dulu!")

# StandardScaler
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split 70-15-15 (try stratified, fallback if not possible)
try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        Xs, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)
except ValueError:
    print("⚠️  Stratified split not possible, using random split")
    X_train, X_temp, y_train, y_temp = train_test_split(
        Xs, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

print(f"✅ Shapes: Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

# -----------------------
# LANGKAH 2 — BANGUN MODEL ANN
# -----------------------
print("\n🧠 Langkah 2: Build ANN Model")
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),  # ↑ Eksperimen: 64 neurons
    layers.BatchNormalization(),          # + BatchNorm
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # Binary
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Adam default
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

print(model.summary())

# -----------------------
# LANGKAH 3 — TRAINING + CALLBACKS
# -----------------------
print("\n🎯 Langkah 3: Training with Early Stopping")
es = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es, lr],
    verbose=1
)

print(f"✅ Training selesai! Epochs: {len(history.history['loss'])}")

# -----------------------
# LANGKAH 4 — EVALUASI TEST SET
# -----------------------
print("\n📈 Langkah 4: Test Evaluation")
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Results: Acc={acc:.3f} | AUC={auc:.3f} | Loss={loss:.3f}")

y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# LANGKAH 5 — VISUALISASI
# -----------------------
print("\n📊 Langkah 5: Plotting")

# Learning Curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history["auc"], label="Train AUC")
plt.plot(history.history["val_auc"], label="Val AUC")
plt.title("AUC Curve")
plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.legend(); plt.grid(True)

plt.tight_layout()
if SAVE_PLOTS: plt.savefig("learning_curve.png", dpi=120)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--", alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(); plt.grid(True)
if SAVE_PLOTS: plt.savefig("roc_curve.png", dpi=120)
plt.show()

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
if SAVE_PLOTS: plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# -----------------------
# LANGKAH 6 — SIMPAN MODEL
# -----------------------
print("\n💾 Langkah 6: Save Model")
model.save("ann_model.h5")
joblib.dump(sc, "scaler.pkl")  # Save scaler too!
print("✅ Saved: ann_model.h5 + scaler.pkl")

# -----------------------
# LANGKAH 7 — INFERENCE CONTOH
# -----------------------
print("\n🔮 Langkah 7: Example Prediction")
# Use a sample from the actual test data to ensure correct number of features
sample_idx = 0
sample_features = X_test[sample_idx:sample_idx+1]  # Keep as numpy array
pred_proba = model.predict(sample_features, verbose=0)[0,0]
pred_class = int(pred_proba >= 0.5)
print(f"Sample from test set at index {sample_idx}")
print(f"Prediction: {pred_class} ({'Lulus' if pred_class else 'Tidak Lulus'})")
print(f"Probability: {pred_proba:.3f}")

# -----------------------
# FINAL REPORT
# -----------------------
print("\n" + "="*50)
print("🏆 ANN FINAL REPORT")
print("="*50)
print(f"✅ Architecture: 64-32-16 (ReLU + BatchNorm + Dropout)")
print(f"✅ Optimizer: Adam (lr=0.001)")
print(f"✅ Test Accuracy: {acc:.3f}")
print(f"✅ Test F1-Score: {f1:.3f}")
print(f"✅ Test AUC-ROC: {auc:.3f}")
print(f"✅ Epochs Used: {len(history.history['loss'])}")
print(f"✅ Files Saved: ann_model.h5 | scaler.pkl")
print(f"✅ Plots: learning_curves.png | roc_ann.png | confusion_matrix_ann.png")
print("\n🎉 SUCCESS! Model ready for production!")
print("="*50)