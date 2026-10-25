import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Train Breast Cancer Triage Model")
    parser.add_argument("--data-path", type=str, default="data/raw/breast_cancer.csv", help="Path to input CSV")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to save model artifacts")
    parser.add_argument("--model-version", type=str, default="1.0.0", help="Model semantic version")
    return parser.parse_args()


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_cv(pipeline, X_train, y_train, cv):
    recalls = []
    accuracies = []
    roc_aucs = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        # In breast cancer dataset: 0 is malignant, 1 is benign.
        # Malignant is positive class (pos_label=0)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 0]  # probability of malignant (class 0)

        # Compute malignant recall (pos_label=0)
        rec = recall_score(y_val, y_pred, pos_label=0)
        acc = accuracy_score(y_val, y_pred)
        # For ROC AUC with class 0 as positive, target 0 -> 1, target 1 -> 0
        roc = roc_auc_score((y_val == 0).astype(int), y_proba)

        recalls.append(rec)
        accuracies.append(acc)
        roc_aucs.append(roc)

    return {
        "cv_malignant_recall_mean": float(np.mean(recalls)),
        "cv_malignant_recall_std": float(np.std(recalls)),
        "cv_accuracy_mean": float(np.mean(accuracies)),
        "cv_accuracy_std": float(np.std(accuracies)),
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std": float(np.std(roc_aucs)),
    }


def main():
    args = parse_args()
    params = load_params()

    seed = params.get("seed", 42)
    test_size = params.get("test_size", 0.2)
    n_splits = params.get("n_splits", 5)

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lr_params = params.get("models", {}).get("logistic_regression", {})
    rf_params = params.get("models", {}).get("random_forest", {})

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=lr_params.get("C", 1.0),
                max_iter=lr_params.get("max_iter", 1000),
                solver=lr_params.get("solver", "lbfgs"),
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", 5),
                random_state=seed,
            )),
        ]),
    }

    results = {}
    best_family = None
    best_cv_recall = -1.0

    print("Evaluating candidate model families with 5-fold Stratified CV...")
    for name, pipe in candidates.items():
        metrics = evaluate_cv(pipe, X_train, y_train, cv)
        # Train on full train split
        pipe.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_test_pred = pipe.predict(X_test)
        y_test_proba = pipe.predict_proba(X_test)[:, 0]  # class 0 (malignant)

        test_recall = float(recall_score(y_test, y_test_pred, pos_label=0))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_roc = float(roc_auc_score((y_test == 0).astype(int), y_test_proba))

        metrics["test_malignant_recall"] = test_recall
        metrics["test_accuracy"] = test_acc
        metrics["test_roc_auc"] = test_roc

        results[name] = metrics
        print(f"[{name}] CV Recall (Malignant): {metrics['cv_malignant_recall_mean']:.4f} +/- {metrics['cv_malignant_recall_std']:.4f} | Test Recall: {test_recall:.4f}")

        if metrics["cv_malignant_recall_mean"] > best_cv_recall:
            best_cv_recall = metrics["cv_malignant_recall_mean"]
            best_family = name

    print(f"\nSelected Model Family: {best_family} with CV Malignant Recall: {best_cv_recall:.4f}")
    selected_pipeline = candidates[best_family]
    selected_metrics = results[best_family]

    os.makedirs(args.output_dir, exist_ok=True)

    # Check test-set malignant recall threshold >= 0.95
    test_recall = selected_metrics["test_malignant_recall"]
    if test_recall < 0.95:
        print(f"ERROR: Held-out test set malignant recall ({test_recall:.4f}) is below 0.95!", file=sys.stderr)
        sys.exit(1)

    # Save artifacts
    model_file = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(selected_pipeline, model_file)

    model_card = {
        "model_version": args.model_version,
        "model_family": best_family,
        "scikit_learn_version": sklearn.__version__,
        "seed": seed,
        "feature_count": len(feature_names),
        "ordered_feature_names": feature_names,
        "test_metrics": {
            "malignant_recall": selected_metrics["test_malignant_recall"],
            "accuracy": selected_metrics["test_accuracy"],
            "roc_auc": selected_metrics["test_roc_auc"],
        },
        "cv_metrics": {
            "malignant_recall_mean": selected_metrics["cv_malignant_recall_mean"],
            "accuracy_mean": selected_metrics["cv_accuracy_mean"],
            "roc_auc_mean": selected_metrics["cv_roc_auc_mean"],
        },
        "all_candidate_metrics": results,
    }

    card_file = os.path.join(args.output_dir, "model_card.json")
    with open(card_file, "w") as f:
        json.dump(model_card, f, indent=2)

    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(selected_metrics, f, indent=2)

    print(f"Model saved to {model_file}")
    print(f"Model card saved to {card_file}")
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
