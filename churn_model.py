import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "Tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
    "PaymentMethod", "TechSupport", "SeniorCitizen", "Partner", "PaperlessBilling",
]
NUMERIC_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]


def make_pipeline(classifier):
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def calculate_metrics(name, y_true, predictions, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def load_or_train_model(data):
    X = data[FEATURE_COLUMNS]
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=42),
    }
    metrics, trained = [], {}
    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= .5).astype(int)
        metrics.append(calculate_metrics(name, y_test, predictions, probabilities))
        trained[name] = pipeline

    best_name = max(metrics, key=lambda row: row["ROC-AUC"])["Model"]
    best_model = trained[best_name]
    classifier = best_model.named_steps["classifier"]
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importance_values = abs(classifier.coef_[0]) if hasattr(classifier, "coef_") else classifier.feature_importances_
    importance = sorted(zip(feature_names, importance_values), key=lambda pair: pair[1], reverse=True)
    clean_importance = [(name.replace("numeric__", "").replace("categorical__", ""), value) for name, value in importance]
    return {"model": best_model, "selected_model": best_name, "metrics": metrics, "feature_importance": clean_importance}


def predict_customer(model, customer):
    customer_frame = pd.DataFrame([customer])[FEATURE_COLUMNS]
    return float(model.predict_proba(customer_frame)[0, 1])
