import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/raw/data.csv")
train, test = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=df["target"],
)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print(f"train: {len(train)} rows, test: {len(test)} rows")