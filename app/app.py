import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

# Load model
model = joblib.load("C:/Users/sumit_zhvqwxt/OneDrive/Desktop/df_project/models/xgboost_model.pkl")

# Title
st.title("📊 Demand Forecasting Dashboard")
st.markdown("### Predict and analyze retail sales using Machine Learning")

st.divider()

# Layout
col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    st.subheader("📦 Store & Economic Data")

    store = st.selectbox("Store ID", list(range(1, 46)))
    holiday = st.selectbox("Holiday", ["No", "Yes"])
    holiday = 1 if holiday == "Yes" else 0

    temperature = st.slider("Temperature", -10.0, 50.0, 20.0)
    fuel_price = st.number_input("Fuel Price", value=3.0)
    cpi = st.number_input("CPI", value=200.0)
    unemployment = st.number_input("Unemployment", value=8.0)

# RIGHT COLUMN
with col2:
    st.subheader("📅 Time & Historical Data")

    year = st.selectbox("Year", [2010, 2011, 2012])
    month = st.selectbox("Month", list(range(1, 13)))
    week = st.slider("Week", 1, 52, 10)
    day = st.slider("Day", 1, 31, 15)

    lag_1 = st.number_input("Last Week Sales", value=1000000)
    lag_2 = st.number_input("2 Weeks Ago Sales", value=1000000)
    lag_4 = st.number_input("4 Weeks Ago Sales", value=1000000)

# Calculate rolling mean
rolling_mean_4 = (lag_1 + lag_2 + lag_4) / 3

st.write(f"📊 Rolling Mean (Auto): {rolling_mean_4:,.2f}")

st.divider()

# Prediction
if st.button("🚀 Predict Demand", use_container_width=True):

    input_data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Year": [year],
        "Month": [month],
        "Week": [week],
        "Day": [day],
        "lag_1": [lag_1],
        "lag_2": [lag_2],
        "lag_4": [lag_4],
        "rolling_mean_4": [rolling_mean_4]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Sales: ₹ {prediction:,.2f}")

    st.divider()

    # 📊 1. BAR GRAPH (Comparison)
    st.subheader("📊 Sales Comparison")

    labels = ["Last Week", "Predicted"]
    values = [lag_1, prediction]

    fig1 = plt.figure()
    plt.bar(labels, values)
    plt.title("Sales Comparison")
    st.pyplot(fig1)

    # 📈 2. LINE CHART (Trend)
    st.subheader("📈 Sales Trend")

    trend_data = [lag_4, lag_2, lag_1, prediction]

    fig2 = plt.figure()
    plt.plot(trend_data, marker='o')
    plt.xticks([0,1,2,3], ["4 Weeks Ago", "2 Weeks Ago", "Last Week", "Predicted"])
    plt.title("Sales Trend Over Time")
    st.pyplot(fig2)

    # 🔁 3. MULTI-WEEK FORECAST
    st.subheader("🔁 Multi-Week Forecast (Next 4 Weeks)")

    future_preds = []
    temp_lag1 = lag_1
    temp_lag2 = lag_2
    temp_lag4 = lag_4

    for i in range(4):
        rolling = (temp_lag1 + temp_lag2 + temp_lag4) / 3

        temp_input = pd.DataFrame({
            "Store": [store],
            "Holiday_Flag": [holiday],
            "Temperature": [temperature],
            "Fuel_Price": [fuel_price],
            "CPI": [cpi],
            "Unemployment": [unemployment],
            "Year": [year],
            "Month": [month],
            "Week": [week+i],
            "Day": [day],
            "lag_1": [temp_lag1],
            "lag_2": [temp_lag2],
            "lag_4": [temp_lag4],
            "rolling_mean_4": [rolling]
        })

        pred = model.predict(temp_input)[0]
        future_preds.append(pred)

        # Update lags
        temp_lag4 = temp_lag2
        temp_lag2 = temp_lag1
        temp_lag1 = pred

    fig3 = plt.figure()
    plt.plot(future_preds, marker='o')
    plt.title("Next 4 Weeks Forecast")
    st.pyplot(fig3)

    # 📉 4. ERROR ESTIMATION (Simple)
    st.subheader("📉 Approx Error Insight")

    error_estimate = abs(prediction - lag_1)
    st.info(f"Estimated variation from last week: ₹ {error_estimate:,.2f}")