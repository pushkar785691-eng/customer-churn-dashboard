import streamlit as st
import pandas as pd
import plotly.express as px

from churn_model import FEATURE_COLUMNS, load_or_train_model, predict_customer
from data_generator import generate_customer_data


st.set_page_config(page_title="ChurnLens", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; }
    .block-container { max-width: 1200px; padding-top: 2rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e5e7ef; padding: 18px; border-radius: 14px; }
    .hero { padding: 26px 30px; border-radius: 18px; color: white; background: linear-gradient(120deg,#172554,#4338ca); margin-bottom: 22px; }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: 8px 0 0; color: #dbeafe; }
    </style>
    <div class="hero">
      <h1>ChurnLens</h1>
      <p>Customer churn prediction and business analysis dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return generate_customer_data(1500, random_state=42)


@st.cache_resource
def get_model():
    return load_or_train_model(get_data())


df = get_data()
model_bundle = get_model()
model = model_bundle["model"]

page = st.segmented_control(
    "Dashboard navigation",
    ["Overview", "Churn Analysis", "Predict Customer", "Model Performance"],
    default="Overview",
    label_visibility="collapsed",
    key="dashboard_page",
)

if page == "Overview":
    churn_rate = df["Churn"].mean()
    monthly_revenue = df["MonthlyCharges"].sum()
    revenue_at_risk = df.loc[df["Churn"] == 1, "MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate:.1%}")
    c3.metric("Monthly revenue", f"${monthly_revenue:,.0f}")
    c4.metric("Revenue at risk", f"${revenue_at_risk:,.0f}")

    left, right = st.columns(2)
    with left:
        counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().reset_index()
        counts.columns = ["Status", "Customers"]
        st.plotly_chart(
            px.pie(counts, values="Customers", names="Status", hole=.58,
                   color="Status", color_discrete_map={"Stayed": "#4338ca", "Churned": "#f97316"},
                   title="Customer retention"),
            width="stretch",
        )
    with right:
        contract = df.assign(Status=df["Churn"].map({0: "Stayed", 1: "Churned"}))
        st.plotly_chart(
            px.histogram(contract, x="Contract", color="Status", barmode="group",
                         color_discrete_map={"Stayed": "#4338ca", "Churned": "#f97316"},
                         title="Churn by contract type"),
            width="stretch",
        )

    st.subheader("Customer data sample")
    st.dataframe(df.head(12), width="stretch", hide_index=True)

elif page == "Churn Analysis":
    st.subheader("Explore the drivers of customer churn")
    filter_col, chart_col = st.columns([1, 3])
    with filter_col:
        contract_filter = st.multiselect("Contract", sorted(df["Contract"].unique()), default=list(df["Contract"].unique()))
        internet_filter = st.multiselect("Internet service", sorted(df["InternetService"].unique()), default=list(df["InternetService"].unique()))
    filtered = df[df["Contract"].isin(contract_filter) & df["InternetService"].isin(internet_filter)]
    with chart_col:
        dimension = st.selectbox("Analyze churn by", ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling"])
        rates = filtered.groupby(dimension, as_index=False)["Churn"].mean()
        rates["ChurnRate"] = rates["Churn"] * 100
        fig = px.bar(rates, x=dimension, y="ChurnRate", text_auto=".1f",
                     title=f"Churn rate by {dimension}", color="ChurnRate",
                     color_continuous_scale=["#c7d2fe", "#4338ca"])
        fig.update_layout(yaxis_title="Churn rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.box(filtered, x="Churn", y="MonthlyCharges", color="Churn",
                               labels={"Churn": "Churned"}, title="Monthly charges vs churn",
                               color_discrete_sequence=["#4338ca", "#f97316"]), width="stretch")
    with b:
        st.plotly_chart(px.histogram(filtered, x="Tenure", color="Churn", nbins=20, barmode="overlay",
                                     title="Tenure distribution", opacity=.7,
                                     color_discrete_sequence=["#4338ca", "#f97316"]), width="stretch")

elif page == "Predict Customer":
    st.subheader("Predict an individual customer's churn risk")
    st.caption("Adjust the customer profile and click Predict. The model returns a probability and suggested action.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        with c3:
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        submitted = st.form_submit_button("Predict churn risk", width="stretch")

    if submitted:
        customer = {
            "Tenure": tenure, "MonthlyCharges": monthly,
            "TotalCharges": monthly * max(tenure, 1), "Contract": contract,
            "InternetService": internet, "PaymentMethod": payment,
            "TechSupport": support, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "PaperlessBilling": paperless,
        }
        probability = predict_customer(model, customer)
        st.metric("Predicted churn probability", f"{probability:.1%}")
        if probability >= .65:
            st.error("High risk — prioritize a retention call, plan review, or loyalty offer.")
        elif probability >= .35:
            st.warning("Medium risk — monitor engagement and consider a targeted offer.")
        else:
            st.success("Low risk — continue regular customer engagement.")

elif page == "Model Performance":
    st.subheader("Model evaluation")
    metrics = pd.DataFrame(model_bundle["metrics"])
    st.dataframe(metrics.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}),
                 width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                               x="Model", y="Score", color="Metric", barmode="group",
                               title="Model comparison", range_y=[0, 1]), width="stretch")
    with right:
        importance = pd.DataFrame(model_bundle["feature_importance"], columns=["Feature", "Importance"]).head(12)
        st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature",
                               orientation="h", title="Top churn drivers",
                               color="Importance", color_continuous_scale="Purples"), width="stretch")

    st.info(f"Selected model: **{model_bundle['selected_model']}**. Data is synthetic and intended for portfolio demonstration.")
