import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def main():
    data = load_breast_cancer()
    feature_names = [name.replace(" ", "_") for name in data.feature_names]
    
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target
    
    # Requirement: check no null values and feature count is 30
    assert len(feature_names) == 30, f"Expected 30 features, got {len(feature_names)}"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/breast_cancer.csv"
    df.to_csv(out_path, index=False)
    print(f"Exported dataset successfully to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
