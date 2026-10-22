import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# 1. Where MLflow stores run records (a local SQLite file)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Group these runs under one experiment name
mlflow.set_experiment("cancer-classifier")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

for n in [10, 50, 200]:
    for depth in [3, 8, None]:
        # 3. Each 'with' block = one run
        with mlflow.start_run():
            params = {"n_estimators": n, "max_depth": depth}
            model = RandomForestClassifier(**params, random_state=42).fit(X_tr, y_tr)
            preds = model.predict(X_te)

            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds)

            # 4. Record inputs, outputs, and the model itself
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"n={n:>3}, depth={str(depth):>4} -> acc={acc:.4f}, f1={f1:.4f}")