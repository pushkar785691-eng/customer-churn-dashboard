import numpy as np
import pandas as pd


def generate_customer_data(n_customers=1500, random_state=42):
    """Generate a reproducible telecom-style customer churn dataset."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(0, 73, n_customers)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_customers, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_customers, p=[.35, .50, .15])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_customers, p=[.34, .18, .24, .24])
    senior = rng.binomial(1, .16, n_customers)
    partner = rng.choice(["Yes", "No"], n_customers, p=[.48, .52])
    paperless = rng.choice(["Yes", "No"], n_customers, p=[.60, .40])

    tech_support = []
    for service in internet:
        tech_support.append("No internet service" if service == "No" else rng.choice(["Yes", "No"], p=[.34, .66]))

    base_charge = np.where(internet == "Fiber optic", 82, np.where(internet == "DSL", 55, 24))
    monthly = np.clip(base_charge + rng.normal(0, 11, n_customers), 18, 120).round(2)
    total = np.maximum(monthly * tenure + rng.normal(0, 120, n_customers), 0).round(2)

    logit = (
        -1.35 + 1.15 * (contract == "Month-to-month") + .55 * (internet == "Fiber optic")
        + .45 * (payment == "Electronic check") + .45 * (np.array(tech_support) == "No")
        + .35 * senior + .25 * (paperless == "Yes") - .035 * tenure
        + .012 * (monthly - 65) - .25 * (partner == "Yes")
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame({
        "CustomerID": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
        "Tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
        "TechSupport": tech_support, "SeniorCitizen": senior, "Partner": partner,
        "PaperlessBilling": paperless, "Churn": churn,
    })


if __name__ == "__main__":
    data = generate_customer_data()
    data.to_csv("customer_churn.csv", index=False)
    print(f"Created customer_churn.csv with {len(data)} rows")
