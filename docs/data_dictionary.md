# Data dictionary

Source: IBM Telco Customer Churn sample data. Each row represents one fictional
telecom customer. `Churn` indicates whether the customer left within the last month.

| Field | Meaning |
|---|---|
| customerID | Unique customer identifier; excluded from model features |
| gender | Customer gender in the source sample |
| SeniorCitizen | Whether the customer is a senior citizen (1/0) |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Months with the company |
| PhoneService | Whether phone service is active |
| MultipleLines | Multiple-line service status |
| InternetService | DSL, fiber optic, or no internet service |
| OnlineSecurity | Online security service status |
| OnlineBackup | Online backup service status |
| DeviceProtection | Device protection service status |
| TechSupport | Technical support service status |
| StreamingTV | Streaming TV service status |
| StreamingMovies | Streaming movie service status |
| Contract | Month-to-month, one-year, or two-year contract |
| PaperlessBilling | Whether paperless billing is active |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Current monthly charge |
| TotalCharges | Cumulative charges; 11 blank source values are median-imputed in the pipeline |
| Churn | Prediction target: Yes or No |

