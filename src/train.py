import os
import json
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

params = yaml.safe_load(open("params.yaml"))["train"]

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_tr, y_tr = train.drop(columns="target"), train["target"]
X_te, y_te = test.drop(columns="target"), test["target"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cancer-pipeline")

with mlflow.start_run():
    model = RandomForestClassifier(**params).fit(X_tr, y_tr)
    preds = model.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "f1": f1_score(y_te, preds),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

# Outputs that DVC tracks
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)