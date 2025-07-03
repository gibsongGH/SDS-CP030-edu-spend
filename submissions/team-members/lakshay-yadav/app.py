import streamlit as st
import pandas as pd
import joblib

# Load saved models
regressor = joblib.load("deployment_models/best_regressor.pkl")
classifier = joblib.load("deployment_models/best_classifier.pkl")

# Load dataset used for dropdowns
df = pd.read_csv("data/International_Education_Costs.csv")  # ✅ Use forward slashes or raw string

st.set_page_config(page_title="EduSpend Budget Planner", layout="centered")
st.title("🎓 EduSpend — Budget Planner")
st.markdown("Predict your **Total Cost of Attendance (TCA)** and **Affordability Tier** based on your study profile.")

# Input form
with st.form("input_form"):
    st.subheader("📋 Enter Program Details")

    # Dynamically populate countries
    unique_countries = sorted(df['Country'].dropna().unique().tolist())
    country = st.selectbox("Country", unique_countries)

    # Dynamically populate cities based on selected country
    filtered_df = df[df['Country'] == country]
    unique_cities = sorted(filtered_df['City'].dropna().unique().tolist())
    city = st.selectbox("City", unique_cities)

    # Populate levels based on filtered data
    unique_levels = sorted(filtered_df['Level'].dropna().unique().tolist())
    level = st.selectbox("Degree Level", unique_levels)

    tuition = st.number_input("Tuition (USD)", min_value=0, value=15000, step=500)
    rent = st.number_input("Monthly Rent (USD)", min_value=0, value=800, step=50)
    insurance = st.number_input("Annual Insurance (USD)", min_value=0, value=1000, step=50)

    submitted = st.form_submit_button("🔍 Predict")

# Predict and display results
if submitted:
    input_df = pd.DataFrame([{
        'Country': country,
        'City': city,
        'Level': level,
        'Tuition_USD': tuition,
        'Rent_USD': rent,
        'Insurance_USD': insurance
    }])

    # Predict TCA
    tca_pred = regressor.predict(input_df)[0]
    tier_pred = classifier.predict(input_df)[0]

    # Decode label
    tier_label = {0: "Low", 1: "Medium", 2: "High"}.get(tier_pred, "Unknown")

    st.success(f"📊 **Estimated Total Cost of Attendance:** ${tca_pred:,.2f}")
    st.info(f"🏷️ **Affordability Tier:** {tier_label}")