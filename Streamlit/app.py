import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Load Model & Pipeline
# ----------------------------------------------------

pipeline = joblib.load("preprocess_pipeline.pkl")
model = joblib.load("revenue_model.pkl")

# ----------------------------------------------------
# Required Columns
# ----------------------------------------------------

required_columns = [
    "customer_id",
    "invoice_id",
    "transaction_date",
    "product_id",
    "quantity",
    "unit_price",
    "discount_rate",
    "refund_flag",
    "payment_method",
    "country"
]

# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------

def create_features(df):

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    df["revenue"] = df["quantity"] * df["unit_price"]

    today = df["transaction_date"].max()

    group = df.groupby("customer_id")

    recency = (today - group["transaction_date"].max()).dt.days
    frequency = group["invoice_id"].nunique()
    monetary = group["revenue"].sum()

    aov = monetary / frequency.replace(0,1)

    tenure = (
        group["transaction_date"].max() -
        group["transaction_date"].min()
    ).dt.days

    last30 = df[df["transaction_date"] >= today - pd.Timedelta(days=30)]
    last90 = df[df["transaction_date"] >= today - pd.Timedelta(days=90)]

    revenue_last_30 = last30.groupby("customer_id")["revenue"].sum()
    revenue_last_90 = last90.groupby("customer_id")["revenue"].sum()

    orders_last_30 = last30.groupby("customer_id")["invoice_id"].nunique()
    orders_last_90 = last90.groupby("customer_id")["invoice_id"].nunique()

    purchase_velocity = frequency / tenure.replace(0,1)

    purchase_trend = revenue_last_30 / revenue_last_90.replace(0,1)

    mean_discount = group["discount_rate"].mean()

    customer_df = pd.DataFrame({
        "recency":recency,
        "frequency":frequency,
        "monetary":monetary,
        "aov":aov,
        "tenure":tenure,
        "revenue_last_30_days":revenue_last_30,
        "revenue_last_90_days":revenue_last_90,
        "orders_last_30_days":orders_last_30,
        "orders_last_90_days":orders_last_90,
        "purchase_velocity":purchase_velocity,
        "purchase_trend":purchase_trend,
        "mean_discount":mean_discount
    })

    customer_df.fillna(0,inplace=True)

    customer_df.reset_index(inplace=True)

    return customer_df


# ----------------------------------------------------
# Segmentation
# ----------------------------------------------------

def segment_customers(df):

    q1 = df["predicted_revenue"].quantile(0.33)
    q2 = df["predicted_revenue"].quantile(0.66)

    conditions = [
        df["predicted_revenue"] <= q1,
        (df["predicted_revenue"] > q1) & (df["predicted_revenue"] <= q2),
        df["predicted_revenue"] > q2
    ]

    choices = ["Low Value","Medium Value","High Value"]

    df["segment"] = np.select(conditions,choices,default="Medium Value")

    return df


# ----------------------------------------------------
# Offer Strategy
# ----------------------------------------------------

def assign_offer_strategy(df):

    conditions = [
        df["segment"]=="High Value",
        df["segment"]=="Medium Value",
        df["segment"]=="Low Value"
    ]

    choices = [
        "VIP Loyalty Program",
        "Cross Sell Recommendations",
        "Discount Campaign"
    ]

    df["offer_strategy"] = np.select(
        conditions,
        choices,
        default="Standard Marketing"
    )

    return df


# ----------------------------------------------------
# UI
# ----------------------------------------------------

st.title("Customer Value Forecasting & Offer Optimization")

st.sidebar.header("Instructions")

st.sidebar.write("Upload transaction dataset to predict customer revenue and generate marketing strategies.")

st.sidebar.subheader("Required Columns")

for col in required_columns:
    st.sidebar.write(col)

st.divider()

# ----------------------------------------------------
# Excel → CSV
# ----------------------------------------------------

st.header("Excel to CSV Converter")

excel_file = st.file_uploader("Upload Excel",type=["xlsx"])

if excel_file:

    df_excel = pd.read_excel(excel_file)

    st.dataframe(df_excel.head())

    csv = df_excel.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "converted_file.csv",
        "text/csv"
    )

st.divider()

# ----------------------------------------------------
# Prediction Section
# ----------------------------------------------------

st.header("Customer Revenue Prediction")

uploaded_file = st.file_uploader("Upload CSV Dataset",type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # Dataset Summary

    col1,col2,col3 = st.columns(3)

    col1.metric("Transactions",len(df))

    col2.metric("Customers",df["customer_id"].nunique())

    col3.metric("Products",df["product_id"].nunique())

    missing = [c for c in required_columns if c not in df.columns]

    if missing:

        st.error("Missing Columns")

        st.write(missing)

    else:

        st.success("All required columns present")

        if st.button("Run Prediction"):

            # Feature Engineering
            customer_features = create_features(df)

            X = customer_features.drop("customer_id",axis=1)

            # Pipeline
            X_processed = pipeline.transform(X)

            # Prediction
            preds = model.predict(X_processed)

            customer_features["predicted_revenue"] = preds

            # Segmentation
            customer_features = segment_customers(customer_features)

            # Offer Strategy
            customer_features = assign_offer_strategy(customer_features)

            st.success("Prediction Completed")

            # ------------------------------------------------
            # KPI Metrics
            # ------------------------------------------------

            total_customers = customer_features["customer_id"].nunique()

            avg_revenue = customer_features["predicted_revenue"].mean()

            high_value = (customer_features["segment"]=="High Value").sum()

            col1,col2,col3 = st.columns(3)

            col1.metric("Total Customers",total_customers)

            col2.metric("Avg Predicted Revenue",round(avg_revenue,2))

            col3.metric("High Value Customers",high_value)

            # ------------------------------------------------
            # Charts
            # ------------------------------------------------

            st.subheader("Revenue Distribution")

            fig,ax = plt.subplots()

            ax.hist(customer_features["predicted_revenue"],bins=20)

            ax.set_xlabel("Predicted Revenue")

            st.pyplot(fig)

            st.subheader("Customer Segment Distribution")

            st.bar_chart(customer_features["segment"].value_counts())

            st.subheader("Discount vs Predicted Revenue")

            fig2,ax2 = plt.subplots()

            ax2.scatter(
                customer_features["mean_discount"],
                customer_features["predicted_revenue"]
            )

            ax2.set_xlabel("Average Discount")

            ax2.set_ylabel("Predicted Revenue")

            st.pyplot(fig2)

            # ------------------------------------------------
            # Segment Filter
            # ------------------------------------------------

            st.subheader("Filter Customers")

            seg = st.selectbox(
                "Select Segment",
                ["All","High Value","Medium Value","Low Value"]
            )

            if seg != "All":

                filtered = customer_features[
                    customer_features["segment"]==seg
                ]

            else:

                filtered = customer_features

            st.dataframe(filtered.head(20))

            # ------------------------------------------------
            # Customer Lookup
            # ------------------------------------------------

            st.subheader("Customer Lookup")

            cust_id = st.selectbox(
                "Select Customer",
                customer_features["customer_id"]
            )

            cust = customer_features[
                customer_features["customer_id"]==cust_id
            ].iloc[0]

            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Predicted Revenue",
                round(cust["predicted_revenue"],2)
            )

            col2.metric("Segment",cust["segment"])

            col3.metric("Offer Strategy",cust["offer_strategy"])

            # ------------------------------------------------
            # Top 10 Customers
            # ------------------------------------------------

            st.subheader("Top 10 High Value Customers")

            top_customers = customer_features.sort_values(
                "predicted_revenue",
                ascending=False
            ).head(10)

            st.dataframe(
                top_customers[
                    ["customer_id","predicted_revenue","offer_strategy"]
                ]
            )

            # ------------------------------------------------
            # Download Results
            # ------------------------------------------------

            csv = customer_features.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "customer_predictions.csv",
                "text/csv"
            )