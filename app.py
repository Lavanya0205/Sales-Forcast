import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load trained model
# ===============================
with open("walmart_sales_forecast_model.pkl", "rb") as file:
    model = pickle.load(file)

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide", page_icon="🛒")

# ===============================
# Page Title & Intro
# ===============================
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🛒 Walmart Weekly Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; font-size:18px'>
    Enter store & economic details to forecast <b>weekly sales</b> 📊
    </p>
    """, unsafe_allow_html=True
)
st.divider()

# ===============================
# Input Parameters with Sidebar
# ===============================
st.sidebar.header("🔧 Input Parameters")

store = st.sidebar.slider("Store Number", 1, 50, 1)
temperature = st.sidebar.slider("🌡️ Temperature (°F)", 50.0, 100.0, 70.0)
fuel_price = st.sidebar.slider("⛽ Fuel Price ($)", 2.0, 5.0, 3.0, 0.01)
cpi = st.sidebar.slider("📈 CPI (Consumer Price Index)", 200.0, 250.0, 220.0, 0.1)
unemployment = st.sidebar.slider("💼 Unemployment Rate (%)", 3.0, 15.0, 7.0, 0.1)
size = st.sidebar.slider("🏬 Store Size (sq ft)", 50000, 250000, 150000, 500)
year = st.sidebar.selectbox("📅 Year", [2010, 2011, 2012])
month = st.sidebar.selectbox("📆 Month", list(range(1, 13)))
week = st.sidebar.selectbox("🗓️ Week", list(range(1, 54)))
day_of_week = st.sidebar.selectbox("📍 Day of Week (0 = Monday)", list(range(0, 7)))

input_data = pd.DataFrame({
    "Store": [store],
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "Size": [size],
    "Year": [year],
    "Month": [month],
    "Week": [week],
    "DayOfWeek": [day_of_week]
})

# ===============================
# Tabs for Prediction & Info
# ===============================
tab1, tab2 = st.tabs(["📌 Predict Sales", "ℹ️ About"])

with tab1:
    st.markdown("### Enter the store and economic details and click predict")
    
    if st.button("Predict Weekly Sales"):
        with st.spinner("⏳ Predicting..."):
            prediction = model.predict(input_data)[0]
        
        st.success(f"### ✅ Forecasted Weekly Sales: **${prediction:,.2f}**")

        # Advanced Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=["Predicted Sales"], y=[prediction], palette="Blues_d", ax=ax)
        ax.set_ylabel("Weekly Sales ($)")
        ax.set_title("Predicted Weekly Sales")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Fill the inputs on the sidebar and click **Predict Weekly Sales**.")

with tab2:
    st.markdown("""
    ### About This App
    - Built with **Streamlit**, **Pandas**, **NumPy**, and **Matplotlib**.
    - Predicts weekly sales of Walmart stores based on economic indicators.
    - 🎯 Ideal for store managers, analysts, and enthusiasts!
    """)

# ===============================
# Footer
# ===============================
st.divider()


