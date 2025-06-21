import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_tca_model.joblib')

st.title("EduSpend: TCA Predictor")

# Example options (replace with your actual data if available)
country_options = ["USA", "UK", "Canada", "Australia"]
city_options = {
    "USA": ["New York", "Los Angeles", "Chicago"],
    "UK": ["London", "Manchester", "Edinburgh"],
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "Australia": ["Sydney", "Melbourne", "Brisbane"]
}
level_options = ["Bachelor", "Master", "PhD"]
program_options = ["Engineering", "Business", "Arts", "Science"]  # Example

country = st.selectbox("Country", country_options, help="Select the country of study")
city = st.selectbox("City", city_options[country], help="Select the city of study")
level = st.selectbox("Level", level_options, help="Select the level of study")
program = st.selectbox("Program", program_options, help="Select your program")  # Use text_input if options unknown

living_cost_index = st.number_input("Living Cost Index", min_value=0.0, help="Relative cost of living in the city")
rent_usd = st.number_input("Monthly Rent (USD)", min_value=0.0, help="Average monthly rent in USD")
visa_usd = st.number_input("Visa Fee Cost (USD)", min_value=0.0, help="Visa application fee in USD")
insurance_usd = st.number_input("Insurance Cost (USD)", min_value=0.0, help="Annual insurance cost in USD")
exchange_rate = st.number_input("Exchange Rate", min_value=0.0, help="Exchange rate to USD")
tuition_usd = st.number_input("Tuition (USD)", min_value=0.0, help="Annual tuition fee in USD")

# Collect input into a DataFrame
input_df = pd.DataFrame([{
    "Country": country,
    "City": city,
    "Level": level,
    "Program": program,
    "Living_Cost_Index": living_cost_index,
    "Rent_USD": rent_usd,
    "Visa_Fee_USD": visa_usd,
    "Insurance_USD": insurance_usd,
    "Exchange_Rate": exchange_rate,
    "Tuition_USD": tuition_usd
}])

if st.button("Predict TCA"):
    st.write("Input DataFrame:", input_df)  # For debugging, can remove later
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Total Cost of Attendance (TCA): ${prediction:,.2f}")