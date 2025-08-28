import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load trained model
with open("walmart_sales_forecast_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset for plotting and structure
df = pd.read_csv("walmart_cleaned_no_index.csv")
# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Drop unnecessary columns
df.drop(columns=["Date", "IsHoliday", "Dept", "Type", "MarkDown1", "MarkDown2", 
                 "MarkDown3", "MarkDown4", "MarkDown5"], inplace=True, errors="ignore")

# Split into features and target
X = df.drop(columns=["Weekly_Sales"])
y = df["Weekly_Sales"]

# Split for plotting
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Predict
y_pred = model.predict(X_test)

# Streamlit UI
st.title("🛒 Walmart Weekly Sales Forecasting")

st.markdown("### 📈 Actual vs. Predicted Sales")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.arange(len(y_test)), y_test.values, label="Actual", linewidth=2)
ax.plot(np.arange(len(y_test)), y_pred, label="Predicted", linewidth=2)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Weekly Sales")
ax.set_title("Actual vs Predicted Weekly Sales")
ax.legend()
st.pyplot(fig)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.markdown("### 📊 Model Performance Metrics")
st.write(f"**MAE (Mean Absolute Error):** ${mae:,.2f}")
st.write(f"**RMSE (Root Mean Squared Error):** ${rmse:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Prediction form
st.markdown("### 🔮 Forecast New Weekly Sales")

# User Inputs
store = st.number_input("Store Number", min_value=1, max_value=50, value=1)
temperature = st.number_input("Temperature (°F)", value=70.0)
fuel_price = st.number_input("Fuel Price", value=3.0)
cpi = st.number_input("CPI (Consumer Price Index)", value=220.0)
unemployment = st.number_input("Unemployment Rate", value=7.0)
size = st.number_input("Store Size (in sq ft)", value=150000)  # ✅ Added this
year = st.selectbox("Year", [2010, 2011, 2012])
month = st.selectbox("Month", list(range(1, 13)))
week = st.selectbox("Week", list(range(1, 54)))
day_of_week = st.selectbox("Day of Week (0 = Monday)", list(range(0, 7)))

# Predict on user input
if st.button("📌 Predict Weekly Sales"):
    input_data = pd.DataFrame({
        "Store": [store],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Size": [size],  # ✅ Must be included!
        "Year": [year],
        "Month": [month],
        "Week": [week],
        "DayOfWeek": [day_of_week]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"📊 Forecasted Weekly Sales: **${prediction:,.2f}**")
