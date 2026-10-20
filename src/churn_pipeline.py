"""Reusable training and evaluation pipeline for IBM Telco churn data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMN = "customerID"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and clean the public IBM Telco Customer Churn CSV."""
    frame = pd.read_csv(path)
    required = {ID_COLUMN, TARGET, "TotalCharges"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = frame[TARGET].map({"Yes": 1, "No": 0})
    if frame[TARGET].isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No'.")
    return frame


def split_features(frame: pd.DataFrame):
    customer_ids = frame[ID_COLUMN].copy()
    features = frame.drop(columns=[ID_COLUMN, TARGET])
    target = frame[TARGET].astype(int)
    return features, target, customer_ids


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate(frame: pd.DataFrame, output_dir: str | Path = "artifacts"):
    """Select by train-only cross-validation, evaluate once on a held-out test set."""
    output_dir = Path(output_dir)
    image_dir = output_dir.parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    X, y, customer_ids = split_features(frame)
    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    pipelines = {}
    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            [("preprocess", build_preprocessor(X_train)), ("model", estimator)]
        )
        scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        comparison.append(
            {"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()}
        )
        pipelines[name] = pipeline

    comparison_frame = pd.DataFrame(comparison).sort_values(
        "cv_roc_auc_mean", ascending=False
    )
    best_name = comparison_frame.iloc[0]["model"]
    best_pipeline = pipelines[best_name].fit(X_train, y_train)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "selected_model": best_name,
        "selection_method": "Highest mean ROC-AUC in stratified 5-fold CV on training data",
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions)),
        "test_recall": float(recall_score(y_test, predictions)),
        "random_state": RANDOM_STATE,
    }

    comparison_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    joblib.dump(best_pipeline, output_dir / "churn_model.joblib")

    scored = ids_test.to_frame().reset_index(drop=True)
    scored["actual_churn"] = y_test.reset_index(drop=True)
    scored["churn_probability"] = probabilities
    scored["predicted_churn"] = predictions
    scored["risk_band"] = pd.cut(
        scored["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )
    scored.sort_values("churn_probability", ascending=False).to_csv(
        output_dir / "customer_risk_scores.csv", index=False
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Held-out Test ROC Curve")
    fig.tight_layout()
    fig.savefig(image_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix at 0.50 Threshold")
    fig.tight_layout()
    fig.savefig(image_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    report = classification_report(y_test, predictions)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    return best_pipeline, comparison_frame, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the IBM Telco Customer Churn CSV",
    )
    parser.add_argument("--output", default="artifacts", help="Artifact output directory")
    args = parser.parse_args()
    _, comparison, metrics = train_and_evaluate(load_data(args.data), args.output)
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

