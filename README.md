# mlops-demo

A hands-on learning project for the two core MLOps tools: **MLflow** (experiment tracking and model registry) and **DVC** (data versioning and reproducible pipelines).

The ML task is intentionally simple: a Random Forest classifier on scikit-learn's Breast Cancer Wisconsin dataset. The focus is on the tooling around the model, not the model itself.

---

## What this project covers

| Concept | Tool | Where |
|---|---|---|
| Logging params, metrics and models for every run | MLflow Tracking | `mlflow_basics.py`, `src/train.py` |
| Comparing runs visually | MLflow UI | `mlflow ui` |
| Naming models and promoting them with aliases | MLflow Model Registry | `register_model.py`, `load_model.py` |
| Versioning a dataset outside Git | DVC | `data/raw/data.csv.dvc` |
| Remote storage for data and models | DVC remote | `.dvc/config` |
| Multi-stage pipeline that reruns only what changed | DVC Pipelines | `dvc.yaml`, `dvc.lock` |
| Centralised hyperparameters | DVC params | `params.yaml` |
| Comparing params/metrics across commits | DVC | `dvc params diff`, `dvc metrics diff` |
| Trying hyperparameters without commits | DVC Experiments | `dvc exp run`, `dvc exp show` |

### How the tools divide the work

- **Git** versions the code, `params.yaml`, `dvc.yaml`, `dvc.lock` and the small `.dvc` pointer files.
- **DVC** versions the data and the model file, and decides *what needs to rerun*.
- **MLflow** records *what happened in every run* and serves models by name.

Any Git commit therefore pins down the exact code, parameters and data, so the result can be reproduced.

---

## Project structure

```
mlops-demo/
├── data/
│   └── raw/
│       └── data.csv.dvc      # DVC pointer to the dataset (the CSV itself is not in Git)
├── src/
│   ├── prepare.py            # stage 1: stratified train/test split
│   └── train.py              # stage 2: train, log to MLflow, write model + metrics
├── dvc.yaml                  # pipeline definition (stages, deps, outs, params)
├── dvc.lock                  # exact hashes of every input/output from the last run
├── params.yaml               # all hyperparameters
├── metrics.json              # latest metrics (tracked by Git for diffing)
├── mlflow_basics.py          # standalone MLflow demo: 9-run hyperparameter sweep
├── register_model.py         # finds the best run and registers it with alias 'champion'
├── load_model.py             # loads models:/cancer-classifier@champion and predicts
├── pyproject.toml / uv.lock  # dependencies (managed with uv)
└── README.md
```

## Pipeline

```
data/raw/data.csv ──► [prepare] ──► data/processed/{train,test}.csv ──► [train] ──► models/model.pkl
                          ▲                                                ▲          metrics.json
                   prepare.test_size                               train.n_estimators   + MLflow run
                   prepare.seed                                    train.max_depth
                                                                   train.random_state
```

`dvc repro` runs only the stages whose code, data or params changed. Changing `train.n_estimators` reruns `train` and skips `prepare`.

---

## Setup (Windows / PowerShell)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/AshutoshP29/mlops-demo.git
cd mlops-demo
uv sync
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> **Note on data:** the DVC remote in this repo points to a local folder on the author's machine, so `dvc pull` will not work after cloning. To run the pipeline, regenerate the dataset:
>
> ```powershell
> mkdir data\raw
> python -c "from sklearn.datasets import load_breast_cancer; load_breast_cancer(as_frame=True).frame.to_csv('data/raw/data.csv', index=False)"
> ```
>
> Because DVC identifies files by content hash, this produces the exact same file the pipeline was built on.

---

## Usage

All commands are run from the project root.

### MLflow

```powershell
python mlflow_basics.py                                  # 9 runs: n_estimators × max_depth grid
mlflow ui --backend-store-uri sqlite:///mlflow.db        # open http://127.0.0.1:5000
python register_model.py                                 # register best run as 'cancer-classifier', alias 'champion'
python load_model.py                                     # load models:/cancer-classifier@champion
```

### DVC pipeline

```powershell
dvc dag                  # show the stage graph
dvc repro                # run the pipeline (skips unchanged stages)
dvc params diff          # params: workspace vs last commit
dvc metrics diff         # metrics: workspace vs last commit
```

### DVC experiments

```powershell
dvc exp run --set-param train.max_depth=5
dvc exp show --only-changed
dvc exp apply <experiment-name>     # bring an experiment into the workspace, then commit it
```

### DVC data versioning

```powershell
dvc add data/raw/data.csv                         # track a new version of the data
git checkout <commit> -- data/raw/data.csv.dvc    # point to an older version...
dvc checkout                                      # ...and restore that data
```

---

## Results

**Hyperparameter sweep (`mlflow_basics.py`, unstratified 80/20 split)**

| n_estimators | max_depth 3 | max_depth 8 | max_depth None |
|---|---|---|---|
| 10 | 0.9561 | 0.9561 | 0.9561 |
| 50 | 0.9649 | 0.9649 | 0.9649 |
| 200 | 0.9649 | 0.9649 | 0.9649 |

**Pipeline (`dvc repro`, stratified 80/20 split, 114 test samples)**

| n_estimators | max_depth | Accuracy | F1 |
|---|---|---|---|
| 50 | 3 | 0.9474 | 0.9583 |
| 200 | 3 | 0.9561 | 0.9655 |

DVC experiments with `max_depth` 5 and 8 and `n_estimators` 100 gave identical metrics.

### Takeaways

- `max_depth` has no effect on this dataset; trees separate the classes easily even when shallow.
- The 50 → 200 tree gain is exactly 1/114, which is **one** extra test sample. On a test set this small that can be noise, and it would need cross-validation to confirm.
- Metrics are only comparable on an identical split. The sweep and the pipeline use different splits, which is why their numbers differ for the same hyperparameters. The DVC pipeline pins the split through `params.yaml` and the data hash.

---

## Lessons learned / gotchas

- **MLflow + skops:** recent MLflow versions save sklearn models with skops, which rejects types it does not trust. Tree-based models need `skops_trusted_types=["sklearn.tree._tree.Tree"]` in `log_model`. Only trust types for models you created yourself.
- **MLflow params are strings:** filters in `search_runs` must compare against `'50'`, not `50`.
- **`dvc exp run` leaves the last experiment in the workspace.** Afterwards, either `dvc exp apply` the one you want or restore with `git checkout -- params.yaml dvc.lock metrics.json` followed by `dvc checkout`, so that the next commit doesn't quietly include it.
- **Keep hyperparameters in `params.yaml`,** never hardcoded in scripts, or DVC cannot detect changes to them.
- **Commit `dvc.lock`.** It is what makes a run reproducible.
- **Always `dvc push` after committing data changes**, or others pulling the repo get pointers with no data behind them.
- **Windows:** PowerShell 5.1 has no `&&` (use separate lines or `;`), and newly installed CLIs (like `gh`) are not visible in a VS Code terminal until VS Code is fully restarted.

---

## Next steps

This repo was for learning the tools. The follow-up project applies the same workflow to a neural network, with DagsHub as a hosted DVC remote and MLflow server, GitHub Actions for CI, and Docker for serving.