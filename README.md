# ChurnLens — Customer Churn Prediction Dashboard

A CV-ready data science project that analyzes telecom customer behavior and predicts which customers are likely to leave. The project combines exploratory data analysis, machine learning, explainability, and an interactive Streamlit dashboard.

## Features

- Generates a reproducible synthetic dataset of 1,500 telecom customers
- Analyzes churn by contract, internet service, payment method, support, tenure, and charges
- Trains and compares Logistic Regression and Random Forest classifiers
- Evaluates accuracy, precision, recall, F1 score, and ROC-AUC
- Displays the most important churn drivers
- Predicts an individual customer's churn probability
- Provides a business-friendly retention recommendation

## Tech stack

Python, Pandas, NumPy, scikit-learn, Plotly, and Streamlit.

## Run locally

```bash
git clone https://github.com/pushkar785691-eng/customer-churn-dashboard.git
cd customer-churn-dashboard
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
customer-churn-dashboard/
├── app.py              # Interactive dashboard
├── churn_model.py      # Preprocessing, training, evaluation, prediction
├── data_generator.py   # Reproducible synthetic telecom data
├── requirements.txt    # Python dependencies
└── README.md
```

## Machine-learning workflow

1. Generate and validate customer data.
2. Split features and churn target using a stratified train/test split.
3. Scale numeric features and one-hot encode categorical features.
4. Train Logistic Regression and Random Forest models.
5. Select the model with the highest test ROC-AUC.
6. Explain results through feature importance and business recommendations.

## CV description

Built an interactive customer churn prediction dashboard using Python, scikit-learn, and Streamlit. Analyzed churn drivers, compared classification models using ROC-AUC and F1 score, and delivered individual risk predictions with retention recommendations.

## Disclaimer

The dataset is synthetic and designed for learning and portfolio demonstration. Model results should not be used for real customer decisions without validation on real organizational data.

## License

MIT
