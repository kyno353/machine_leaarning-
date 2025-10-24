##Langkah 1 — Muat Data

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("pertemuan5/processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

print(f"Dataset shape: {df.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# For small datasets, remove stratify parameter or use simple split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)


##Langkah 2 - Baseline model dan pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))


## Langkah 3 - Model Alternatif (Random Forest)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))

## Langkah 4 — Validasi Silang & Tuning Ringkas

from sklearn.model_selection import StratifiedKFold, GridSearchCV

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))


## Langkah 4.5 - Perbandingan Model
import matplotlib.pyplot as plt

logreg_f1 = f1_score(y_val, y_val_pred, average="macro")
rf_f1 = f1_score(y_val, y_val_best, average="macro")

models = ['Logistic Regression', 'Random Forest']
f1_scores = [logreg_f1, rf_f1]

plt.figure()
plt.bar(models, f1_scores, color=['blue', 'green'])
plt.ylabel('F1 Score (Macro)')
plt.title('Perbandingan F1 Score Model (Validation Set)')
plt.ylim(0, 1.1)

for i, score in enumerate(f1_scores):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
print("Gambar perbandingan model disimpan ke model_comparison.png")







## Langkah 5 - Evaluasi Akhir (Test set)

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

print(f"\n🏆 FINAL MODEL: RandomForest")
print(f"   Alasan: CV F1 {gs.best_score_:.3f} > Baseline {f1_score(y_val, y_val_pred):.3f}")
final_model = best_rf

y_test_pred = final_model.predict(X_test)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC (jika ada predict_proba)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    plt.tight_layout(); plt.savefig("roc_test.png", dpi=120)



## Langkah 6 (Opsional) — Simpan Model

import joblib
joblib.dump(final_model, "model.pkl")
print("Model tersimpan ke model.pkl")


## Langkah 7 (Opsional) — Endpoint Inference (Flask)

from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
MODEL = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)  # dict fitur
    X = pd.DataFrame([data])
    yhat = MODEL.predict(X)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X)[:,1][0])
    return jsonify({"prediction": int(yhat), "proba": proba})

if __name__ == "__main__":
    # app.run(port=5000)
    pass