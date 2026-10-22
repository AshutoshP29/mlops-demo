import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Find the run we want: n_estimators=50, max_depth=3
exp = client.get_experiment_by_name("cancer-classifier")
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=(
        "params.n_estimators = '50' "
        "and params.max_depth = '3' "
        "and attributes.status = 'FINISHED'"
    ),
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
run = runs[0]
print("Best run ID:", run.info.run_id)
print("Accuracy:  ", run.data.metrics["accuracy"])

# 2. Register that run's model under a name
mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "cancer-classifier")
print("Registered version:", mv.version)

# 3. Point the 'champion' alias at this version
client.set_registered_model_alias("cancer-classifier", "champion", mv.version)
print("Alias 'champion' -> version", mv.version)