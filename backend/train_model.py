"""Train and serialize the HeartAI model bundle."""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [feature for feature in FEATURES if feature not in NUMERIC]
COLUMNS = FEATURES + ["target"]
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/cherngs/heart-disease-cleveland-uci"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 100:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "heart-disease.zip"
            urlretrieve(KAGGLE_URL, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                source_name = next(name for name in archive.namelist() if name.endswith("heart_cleveland_upload.csv"))
                path.write_bytes(archive.read(source_name))
    frame = pd.read_csv(path)
    if "condition" in frame.columns:
        frame = frame.rename(columns={"condition": "target"})
    if "target" not in frame.columns:
        frame = pd.read_csv(path, names=COLUMNS, header=None)
    frame = frame[[*FEATURES, "target"]]
    frame = frame.replace("?", np.nan)
    for column in FEATURES + ["target"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["target"])
    frame["target"] = (frame["target"] > 0).astype(int)
    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def train(data_path: Path = ROOT.parent / "dataset" / "heart_disease.csv") -> dict:
    data = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data["target"], test_size=0.2, random_state=42, stratify=data["target"])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    metrics = {}
    bundles = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        }
        bundles[name] = pipeline
    selected = max(metrics, key=lambda name: (metrics[name]["f1"], metrics[name]["roc_auc"]))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": bundles[selected], "model_name": selected, "features": FEATURES}, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"selected_model": selected, "metrics": metrics, "rows": len(data)}, indent=2), encoding="utf-8")
    data.to_json(MODEL_DIR / "analytics.json", orient="records")
    return {"selected_model": selected, "metrics": metrics, "rows": len(data)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parent / "dataset" / "heart_disease.csv")
    args = parser.parse_args()
    print(json.dumps(train(args.data), indent=2))
