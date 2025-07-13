import streamlit as st
import pandas as pd
import joblib

# Load saved models
regressor = joblib.load("deployment_models/best_regressor.pkl")
classifier = joblib.load("deployment_models/best_classifier.pkl")

# Loading dataset used for dropdowns and cache it
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/yadavLakshay/CP30_data/main/International_Education_Costs.csv"
    return pd.read_csv(url)

df = load_data()

# Set page config and title
st.set_page_config(page_title="EduSpend Budget Planner", layout="centered")
st.title("🎓 EduSpend — Budget Planner")
st.markdown("Predict your **Total Cost of Attendance (TCA)** and **Affordability Tier** based on your study profile.")

# 📋 Input section (unified layout)
with st.container():
    st.subheader("📋 Enter Program Details")

    # Dropdowns
    unique_countries = sorted(df['Country'].dropna().unique())
    country = st.selectbox("Country", unique_countries)

    filtered_df = df[df['Country'] == country]
    unique_cities = sorted(filtered_df['City'].dropna().unique())
    city = st.selectbox("City", unique_cities)

    unique_levels = sorted(filtered_df[filtered_df['City'] == city]['Level'].dropna().unique())
    level = st.selectbox("Degree Level", unique_levels)

    # Cost inputs
    tuition = st.number_input("Tuition (USD)", min_value=0, value=15000, step=500)
    rent = st.number_input("Monthly Rent (USD)", min_value=0, value=800, step=50)
    insurance = st.number_input("Annual Insurance (USD)", min_value=0, value=1000, step=50)

    # Prediction button
    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([{
            'Country': country,
            'City': city,
            'Level': level,
            'Tuition_USD': tuition,
            'Rent_USD': rent,
            'Insurance_USD': insurance
        }])

        # Predictions
        tca_pred = regressor.predict(input_df)[0]
        tier_pred = classifier.predict(input_df)[0]
        tier_label = {0: "Low", 1: "Medium", 2: "High"}.get(tier_pred, "Unknown")

        st.success(f"📊 **Estimated Total Cost of Attendance:** ${tca_pred:,.2f}")
        st.info(f"🏷️ **Affordability Tier:** {tier_label}")