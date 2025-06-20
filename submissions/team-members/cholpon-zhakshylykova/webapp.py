import streamlit as st
st.set_page_config(page_title='International Education Budget Planner', layout='centered')

import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --- Page Config ---
st.title("🎓 International Education Budget Planner")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data_full.csv")

data = load_data()

# --- Load Model and Preprocessors from local file ---
@st.cache_resource
def load_model_components():
    path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    return joblib.load(path)

components = load_model_components()
regressor = components["regressor"]
encoder = components["encoder"]
scaler = components["scaler"]

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
target_country = st.sidebar.selectbox("Select Country", sorted(data["Country"].unique()))
level = st.sidebar.selectbox("Select Level", sorted(data["Level"].unique()))
#duration = st.sidebar.slider("Duration (Years)", min_value=1, max_value=6, value=4)

# Placeholders based on target country
filtered = data[data["Country"] == target_country]
most_common_city = filtered["City"].mode()[0] if not filtered.empty else "Unknown"
most_common_university = filtered["University"].mode()[0] if not filtered.empty else "Unknown"
most_common_program = data["Program"].mode()[0]

# --- Define feature categories ---
categorical_features = ["Country", "City", "University", "Program", "Level"]
numeric_features = ["Tuition_USD", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD"]

# --- User Input ---
user_input = pd.DataFrame({
    "Country": [target_country],
    "City": [most_common_city],
    "University": [most_common_university],
    "Program": [most_common_program],
    "Level": [level],
    #"Duration_Years": [duration],
    "Tuition_USD": [0],
    "Living_Cost_Index": [0],
    "Rent_USD": [0],
    "Visa_Fee_USD": [0],
    "Insurance_USD": [0]
})

# --- Encode & Scale ---
encoded_input = encoder.transform(user_input[categorical_features])
scaled_input = scaler.transform(user_input[numeric_features])
user_features = np.hstack([encoded_input, scaled_input])

# --- Predict ---
predicted_tca = regressor.predict(user_features)[0]
st.sidebar.markdown(f"### 💰 Predicted TCA:\n**${predicted_tca:,.2f} USD**")

# --- Affordability Map ---
st.header("🌍 Affordability Map")
affordability_map = px.choropleth(
    data,
    locations="Country",
    locationmode="country names",
    color="Total_cost",
    title="Affordability by Country",
    color_continuous_scale="Viridis"
)
st.plotly_chart(affordability_map)

# --- Cluster Explorer ---
st.header("📊 Cluster Explorer")
cluster_cols = [col for col in data.columns if col.endswith("_Cluster")]
if cluster_cols:
    cluster_option = st.selectbox("Select Cluster Type", cluster_cols)
    cluster_summary = data.groupby(cluster_option).mean(numeric_only=True).reset_index()
    st.dataframe(cluster_summary)
else:
    st.warning("No cluster columns found in dataset.")

# --- Footer Instructions ---
st.markdown("""
---
### ℹ️ How to Use
1. Use the sidebar to input your target country, level, and study duration.
2. View the predicted Total Cost of Attendance (TCA).
3. Explore cost maps and clustering by region or institution.
""")
