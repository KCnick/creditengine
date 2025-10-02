import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import mlflow
from features import build_features
from model_utils import build_pipeline, evaluate

def main(data_path: str, model_out: str = "model.joblib"):
    mlflow.set_experiment("credit-scoring")
    df = pd.read_parquet(data_path)

    df = build_features(df)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    print("Metrics:", metrics)
    mlflow.log_metrics(metrics)

    joblib.dump(pipeline, model_out)
    mlflow.log_artifact(model_out, artifact_path="model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-out", default="model.joblib")
    args = parser.parse_args()
    main(args.data_path, args.model_out)
