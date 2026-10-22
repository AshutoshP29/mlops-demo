import mlflow
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# No run ID needed: just name + alias
model = mlflow.sklearn.load_model("models:/cancer-classifier@champion")

X, y = load_breast_cancer(return_X_y=True)
print("Predicted:", model.predict(X[:10]))
print("Actual:   ", y[:10])
print("Model:", model)