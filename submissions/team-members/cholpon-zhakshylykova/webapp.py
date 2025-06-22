import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title='International Education Budget Planner', layout='wide')
st.title("🎓 International Education Budget Planner")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data_full.csv")

data = load_data()

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_model_components():
    path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    return joblib.load(path)

components = load_model_components()
regressor = components["regressor"]
encoder = components["encoder"]
scaler = components["scaler"]

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_model_components():
    path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    st.write("🔍 Loading model from:", path)  # Debug: Show full path
    return joblib.load(path)

components = load_model_components()
regressor = components["regressor"]
encoder = components["encoder"]
scaler = components["scaler"]

# --- Debug: Print expected feature names ---
st.write("✅ Regressor expects features:", getattr(regressor, 'feature_names_in_', 'Not available'))
st.write("✅ Scaler expects numeric features:", getattr(scaler, 'feature_names_in_', 'Not available'))


# --- Sidebar Inputs ---
st.sidebar.header("📥 Input Parameters")
target_country = st.sidebar.selectbox("Select Country", sorted(data["Country"].dropna().unique()))
level = st.sidebar.selectbox("Select Level", sorted(data["Level"].dropna().unique()))
duration = st.sidebar.slider("Select Duration (Years)", min_value=1, max_value=6, value=4)

# Suggest common options or allow further UI expansion later
filtered = data[data["Country"] == target_country]
most_common_city = filtered["City"].mode()[0] if not filtered.empty else "Unknown"
most_common_university = filtered["University"].mode()[0] if not filtered.empty else "Unknown"
most_common_program = data["Program"].mode()[0] if not data.empty else "Unknown"

# --- User Numeric Inputs ---
tuition = st.sidebar.number_input("Tuition per Year (USD)", min_value=0, value=10000)
living_index = st.sidebar.number_input("Living Cost Index", min_value=0.0, value=65.0)
rent = st.sidebar.number_input("Monthly Rent (USD)", min_value=0, value=500)
visa_fee = st.sidebar.number_input("Visa Fee (USD)", min_value=0, value=200)
insurance = st.sidebar.number_input("Insurance per Year (USD)", min_value=0, value=600)

# --- User Input Frame ---
user_input = pd.DataFrame({
    "Country": [target_country],
    "City": [most_common_city],
    "University": [most_common_university],
    "Program": [most_common_program],
    "Level": [level],
    "Tuition_USD": [tuition],
    "Living_Cost_Index": [living_index],
    "Rent_USD": [rent * 12],  # annualized
    "Visa_Fee_USD": [visa_fee],
    "Insurance_USD": [insurance],
    "Duration_Years": [duration]
})

categorical_features = ["Country", "City", "University", "Program", "Level"]
numeric_features = ["Tuition_USD", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Duration_Years"]

# --- Preprocessing ---
try:
    encoded_input = encoder.transform(user_input[categorical_features])
    scaled_input = scaler.transform(user_input[numeric_features])
    user_features = np.hstack([encoded_input, scaled_input])
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# --- Input Validation ---
if user_features.shape[1] != regressor.n_features_in_:
    st.error(f"Feature mismatch: model expects {regressor.n_features_in_} features but got {user_features.shape[1]}.")
    st.write("A category may be unknown. Try a different combination.")
    st.stop()

# --- Prediction ---
predicted_tca = regressor.predict(user_features)[0]
st.sidebar.markdown("### 💰 Predicted TCA")
st.sidebar.metric(label="Estimated Total Cost (USD)", value=f"${predicted_tca:,.2f}")

# --- Affordability Map ---
st.subheader("🌍 Global Affordability Map")
affordability_map = px.choropleth(
    data,
    locations="Country",
    locationmode="country names",
    color="Total_cost",
    color_continuous_scale="Viridis",
    title="Average Total Cost of Attendance by Country"
)
st.plotly_chart(affordability_map, use_container_width=True)

# --- Cluster Explorer ---
st.subheader("📊 Cost Cluster Explorer")
cluster_cols = [col for col in data.columns if col.endswith("_Cluster")]
if cluster_cols:
    cluster_option = st.selectbox("Select Cluster Type", cluster_cols)
    cluster_summary = data.groupby(cluster_option).mean(numeric_only=True).reset_index()
    st.dataframe(cluster_summary)
    
    with st.expander("ℹ️ Cluster Segment Descriptions"):
        st.markdown("""
        - **Low Cost Cluster**: Countries or institutions with total annual costs typically below $20,000.
        - **Medium Cost Cluster**: Costs between $20,000 and $40,000.
        - **High Cost Cluster**: Prestigious or expensive locations typically exceeding $40,000 annually.
        """)
else:
    st.warning("No cluster columns found in the dataset.")

# --- Instructions ---
st.markdown("""
---
### 📘 How to Use This App

1. Select your **target country**, **education level**, and **study duration**.
2. Enter estimated costs like **tuition**, **rent**, **insurance**, etc.
3. View your **Total Cost of Attendance** prediction.
4. Explore cost trends with:
   - 🌍 Affordability Map
   - 📊 Cost Cluster Explorer
5. Use the cluster segment guide to compare between countries or schools.
""")
