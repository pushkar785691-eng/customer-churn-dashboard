"""Generate the self-contained Kaggle/Jupyter portfolio notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Customer Churn Prediction and Retention Analysis

        **Objective:** build a reproducible churn-risk pipeline on the public IBM Telco
        Customer Churn sample, compare models using train-only cross-validation, evaluate
        the selected model once on a held-out test set, and translate results into retention
        actions.

        The notebook is designed to run both on **Kaggle** and from this GitHub repository.
        It uses a fixed random seed and keeps preprocessing inside each model pipeline to
        prevent train/test leakage.
        """
    ),
    md(
        """
        ## 1. Imports and configuration

        No internet access is needed after the dataset has been attached to the Kaggle
        notebook or placed in `data/raw/` locally.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import warnings

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
            ConfusionMatrixDisplay, accuracy_score, classification_report,
            precision_score, recall_score, roc_auc_score, RocCurveDisplay
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", palette="deep")
        RANDOM_STATE = 42
        pd.set_option("display.max_columns", 30)
        """
    ),
    md("## 2. Locate and load the dataset"),
    code(
        """
        LOCAL_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        candidates = [LOCAL_PATH]
        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            candidates.extend(kaggle_root.rglob("*.csv"))

        data_path = next(
            (path for path in candidates if path.exists() and "churn" in path.name.lower()),
            None,
        )
        if data_path is None:
            raise FileNotFoundError(
                "Attach the Kaggle 'Telco Customer Churn' dataset or place the CSV at "
                f"{LOCAL_PATH}."
            )

        df = pd.read_csv(data_path)
        print(f"Loaded: {data_path}")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        df.head()
        """
    ),
    md("## 3. Data audit and cleaning"),
    code(
        """
        audit = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing_before_cleaning": df.isna().sum(),
            "unique_values": df.nunique(),
        })
        display(audit)
        print("Duplicate rows:", df.duplicated().sum())
        print("Blank TotalCharges values:", df["TotalCharges"].astype(str).str.strip().eq("").sum())
        """
    ),
    code(
        """
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        print("Rows:", len(df))
        print("Missing TotalCharges after conversion:", df["TotalCharges"].isna().sum())
        print("Overall churn rate: {:.1%}".format(df["Churn"].mean()))
        """
    ),
    md(
        """
        The 11 blank `TotalCharges` values correspond to customers with zero tenure.
        They are not filled globally here: median imputation is learned inside each
        cross-validation training fold, which avoids leakage.
        """
    ),
    md("## 4. Exploratory data analysis"),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        churn_counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[0])
        axes[0].set(title="Class distribution", ylabel="Customers", xlabel="")

        sns.histplot(data=df, x="tenure", hue="Churn", bins=24, multiple="stack", ax=axes[1])
        axes[1].set(title="Churn concentration by tenure", xlabel="Tenure (months)")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        drivers = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for column, ax in zip(drivers, axes.ravel()):
            rates = (
                df.groupby(column, observed=True)["Churn"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=False)
                .reset_index()
            )
            sns.barplot(data=rates, y=column, x="mean", ax=ax)
            ax.set(title=f"Churn rate by {column}", xlabel="Churn rate", ylabel="")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        segment_summary = []
        for column in drivers:
            summary = (
                df.groupby(column, observed=True)["Churn"]
                .agg(customers="size", churn_rate="mean")
                .reset_index()
                .rename(columns={column: "segment"})
            )
            summary.insert(0, "feature", column)
            segment_summary.append(summary)
        segment_summary = pd.concat(segment_summary, ignore_index=True)
        top_segments = segment_summary.sort_values("churn_rate", ascending=False).head(10).copy()
        top_segments["churn_rate"] = top_segments["churn_rate"].map(lambda value: f"{value:.1%}")
        display(top_segments)
        """
    ),
    md("## 5. Leakage-safe preprocessing and train/test split"),
    code(
        """
        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].astype(int)
        customer_ids = df["customerID"]

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ])

        print(f"Training rows: {len(X_train):,}")
        print(f"Held-out test rows: {len(X_test):,}")
        """
    ),
    md(
        """
        ## 6. Model comparison

        Model selection uses only the training partition with stratified 5-fold
        cross-validation and ROC-AUC. The held-out test set remains untouched until a
        model has been selected.
        """
    ),
    code(
        """
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="liblinear",
                random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=400, min_samples_leaf=4, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE
            ),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        results, pipelines = [], {}
        for name, estimator in models.items():
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            results.append({
                "model": name,
                "cv_roc_auc_mean": scores.mean(),
                "cv_roc_auc_std": scores.std(),
            })
            pipelines[name] = pipeline

        comparison = pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False)
        display(comparison.round(3))
        """
    ),
    md("## 7. Final held-out evaluation"),
    code(
        """
        best_name = comparison.iloc[0]["model"]
        best_model = pipelines[best_name].fit(X_train, y_train)
        test_probability = best_model.predict_proba(X_test)[:, 1]
        test_prediction = (test_probability >= 0.50).astype(int)

        metrics = {
            "dataset_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate": y.mean(),
            "selected_model": best_name,
            "selection_method": "Highest mean ROC-AUC in stratified 5-fold CV on training data",
            "test_roc_auc": roc_auc_score(y_test, test_probability),
            "test_accuracy": accuracy_score(y_test, test_prediction),
            "test_precision": precision_score(y_test, test_prediction),
            "test_recall": recall_score(y_test, test_prediction),
            "random_state": RANDOM_STATE,
        }
        pd.Series(metrics).to_frame("held_out_test_result")
        """
    ),
    code(
        """
        print(classification_report(y_test, test_prediction, target_names=["Stayed", "Churned"]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        RocCurveDisplay.from_predictions(
            y_test, test_probability, name=best_name, ax=axes[0]
        )
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set_title("Held-out ROC curve")
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_prediction, display_labels=["Stayed", "Churned"],
            cmap="Blues", ax=axes[1]
        )
        axes[1].set_title("Confusion matrix (threshold = 0.50)")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 8. Explain the strongest model signals"),
    code(
        """
        transformed_names = best_model.named_steps["preprocess"].get_feature_names_out()
        estimator = best_model.named_steps["model"]

        if hasattr(estimator, "coef_"):
            importance = pd.Series(estimator.coef_[0], index=transformed_names)
            signal_table = pd.concat([
                importance.nlargest(10).rename("increases_churn_risk"),
                importance.nsmallest(10).sort_values().rename("reduces_churn_risk")
            ], axis=1)
            display(signal_table)
        else:
            importance = pd.Series(estimator.feature_importances_, index=transformed_names)
            display(importance.nlargest(15).to_frame("feature_importance"))
        """
    ),
    md("## 9. Create a customer risk list for a retention team"),
    code(
        """
        risk_scores = pd.DataFrame({
            "customerID": ids_test.reset_index(drop=True),
            "actual_churn": y_test.reset_index(drop=True),
            "churn_probability": test_probability,
            "predicted_churn": test_prediction,
        })
        risk_scores["risk_band"] = pd.cut(
            risk_scores["churn_probability"],
            bins=[-0.01, 0.30, 0.60, 1.00],
            labels=["Low", "Medium", "High"],
        )
        risk_scores = risk_scores.sort_values("churn_probability", ascending=False)
        display(risk_scores.head(15))
        """
    ),
    md(
        """
        ## 10. Business recommendations

        - Prioritize high-risk month-to-month customers for proactive outreach.
        - Test annual-contract incentives instead of offering blanket discounts.
        - Investigate the high churn observed in fiber-optic and electronic-check segments.
        - Bundle technical support and online security trials for at-risk accounts.
        - Measure campaign lift with a randomized holdout group; model score alone does not
          prove that an offer will prevent churn.

        **Limitations:** this is a small, fictional sample with no timestamps, usage history,
        acquisition cost, retention-offer outcome, or causal treatment label. Results are a
        portfolio demonstration and should not be represented as production deployment.
        """
    ),
    md("## 11. Save reproducible outputs"),
    code(
        """
        output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, output_dir / "churn_model.joblib")
        risk_scores.to_csv(output_dir / "customer_risk_scores.csv", index=False)
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        with (output_dir / "metrics.json").open("w") as handle:
            json.dump({key: (float(value) if isinstance(value, (float, np.floating)) else value)
                       for key, value in metrics.items()}, handle, indent=2)
        print(f"Saved model and reports to: {output_dir.resolve()}")
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)

output = Path("notebooks/customer_churn_analysis.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, output)
print(f"Wrote {output}")

