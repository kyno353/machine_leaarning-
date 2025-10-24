# rf_workflow.py
# Run di Python 3.10+ dengan paket: pandas, scikit-learn, matplotlib, seaborn, joblib, numpy

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)

# -----------------------
# Konfigurasi
# -----------------------
RANDOM_STATE = 42
SAVE_PLOTS = True

# -----------------------
# Langkah 1 — Muat data
# Jika kamu punya X_train.csv dsb, ubah bagian ini ke Pilihan B.
# -----------------------
if os.path.exists("pertemuan5/processed_kelulusan.csv"):
    df = pd.read_csv("pertemuan5/processed_kelulusan.csv")
    X = df.drop("Lulus", axis=1)
    y = df["Lulus"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
else:
    # Jika kamu sudah punya file split hasil pertemuan 5, pakai ini:
    X_train = pd.read_csv("X_train.csv")
    X_val   = pd.read_csv("X_val.csv")
    X_test  = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze("columns")
    y_val   = pd.read_csv("y_val.csv").squeeze("columns")
    y_test  = pd.read_csv("y_test.csv").squeeze("columns")

print("Shapes -> Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Label distribution (train):")
print(y_train.value_counts())

# -----------------------
# Langkah 2 — Pipeline & Baseline Random Forest
# -----------------------
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print("Numeric columns used:", num_cols)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("\nBaseline RF — Validation results")
print("F1(val, macro):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# -----------------------
# Langkah 3 — Validasi Silang (di train)
# -----------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
try:
    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    print("\nCV F1-macro (train): {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))
except Exception as e:
    print("\nCross-val gagal:", e)

# -----------------------
# Langkah 4 — Tuning Ringkas (GridSearch)
# -----------------------
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("\nGridSearch selesai")
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("\nBest RF — Validation results")
print("F1(val, macro):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3))

# -----------------------
# Langkah 5 — Evaluasi Akhir (Test Set)
# -----------------------
final_model = best_model
y_test_pred = final_model.predict(X_test)

print("\nEvaluasi akhir pada test set")
print("F1(test, macro):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):\n", confusion_matrix(y_test, y_test_pred))

# ROC-AUC & plotting
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc)
    except Exception as e:
        print("ROC-AUC error:", e)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    if SAVE_PLOTS: plt.savefig("roc_test.png", dpi=120)
    plt.show()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    if SAVE_PLOTS: plt.savefig("pr_test.png", dpi=120)
    plt.show()

# -----------------------
# Langkah 6 — Feature importance
# -----------------------
print("\nFeature importance (native RandomForest, Gini):")
try:
    importances = final_model.named_steps["clf"].feature_importances_
    # dapatkan nama fitur dari preprocessor
    try:
        fn = final_model.named_steps["pre"].get_feature_names_out()
    except Exception:
        # alternatif: gunakan num_cols
        fn = num_cols
    items = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    for name, val in items:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Tidak bisa ambil feature importance:", e)

# -----------------------
# Langkah 7 — Simpan model
# -----------------------
joblib.dump(final_model, "rf_model.pkl")
print("\nModel disimpan sebagai rf_model.pkl")

# -----------------------
# Langkah 8 — Cek inference lokal contoh
# -----------------------
sample = pd.DataFrame([{
  # sesuaikan nama kolom dan nilai contoh
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])
pred = final_model.predict(sample)[0]
proba = None
if hasattr(final_model, "predict_proba"):
    proba = float(final_model.predict_proba(sample)[:,1][0])
print("\nContoh inference -> Prediksi:", int(pred), "Probabilitas:", proba)