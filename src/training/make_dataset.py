import pandas as pd
import numpy as np

def make_dataset(n_samples=5000, out_path="data/credit_train.parquet"):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(21, 65, size=n_samples),
        "income": np.random.normal(50000, 15000, size=n_samples).clip(20000, 150000),
        "credit_lines": np.random.randint(1, 10, size=n_samples),
        "delinquencies": np.random.poisson(0.5, size=n_samples),
        "utilization": np.random.uniform(0, 1, size=n_samples),
        "months_active": np.random.randint(6, 240, size=n_samples),
        "region": np.random.choice(["Nairobi", "Mombasa", "Kisumu"], size=n_samples),
    })
    # target: simple heuristic -> more delinquencies & high utilization = higher default risk
    df["default"] = (
        (df["delinquencies"] > 1).astype(int) |
        ((df["utilization"] > 0.7) & (df["income"] < 40000)).astype(int)
    ).astype(int)

    df.to_parquet(out_path, index=False)
    print(f"✅ Dataset saved to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    make_dataset()
