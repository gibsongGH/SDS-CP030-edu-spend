import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title='International Education Budget Planner', layout='wide')
st.title("International Education Budget Planner")

# --- Load Data ---
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data_tca_clusters_raw.csv")
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path)

data = load_data()

# --- Load Pipeline ---
@st.cache_resource
def load_model_pipeline():
    path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    return joblib.load(path)

pipeline = load_model_pipeline()

# --- Sidebar Inputs ---
st.sidebar.header("📥 Input Parameters")
target_country = st.sidebar.selectbox("Select Country", sorted(data["Country"].dropna().unique()))
filtered = data[data["Country"] == target_country]

city = st.sidebar.selectbox("Select City", sorted(filtered["City"].dropna().unique()))
level = st.sidebar.selectbox("Select Level", sorted(data["Level"].dropna().unique()))
university = st.sidebar.selectbox("Select University", sorted(filtered["University"].dropna().unique()))
program = st.sidebar.selectbox("Select Program", sorted(filtered["Program"].dropna().unique()))

# --- Construct Input Frame (categorical only) ---
user_input = pd.DataFrame({
    "Country": [target_country],
    "City": [city],
    "University": [university],
    "Program": [program],
    "Level": [level]
})

# --- Prediction ---
try:
    predicted_tca = pipeline.predict(user_input)[0]
    st.sidebar.markdown("### 💰 Predicted TCA")
    st.sidebar.metric(label="Estimated Total Cost (USD)", value=f"${predicted_tca:,.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- Affordability Map ---
st.subheader("🌍 Global Affordability Map")
fig = px.choropleth(
    data_frame=data,
    locations="Country",
    locationmode="country names",
    color="Total_cost",
    hover_name="Country",
    color_continuous_scale="Turbo",
    range_color=(data["Total_cost"].min(), data["Total_cost"].max()),
    title="Average Total Cost of Attendance by Country",
    labels={"Total_cost": "Total Cost (USD)"}
)
fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- Cluster Explorer ---
st.subheader("📊 Cost Cluster Explorer")
cluster_cols = [col for col in data.columns if col.endswith("_Cluster")]
if cluster_cols:
    cluster_option = st.selectbox("Select Cluster Type", cluster_cols)
    cluster_summary = data.groupby(cluster_option).mean(numeric_only=True).reset_index()
    st.dataframe(cluster_summary)

    with st.expander("ℹ️ Cluster Segment Descriptions"):
        st.markdown("""
        <ul>
        <li><b>Low Cost Cluster</b>: Typically below <b>$20,000</b> annually.</li>
        <li><b>Medium Cost Cluster</b>: Between <b>$20,000</b> and <b>$40,000</b>.</li>
        <li><b>High Cost Cluster</b>: Above <b>$40,000</b>.</li>
        </ul>
        """, unsafe_allow_html=True)
else:
    st.warning("No cluster columns found in the dataset.")

# --- Instructions ---
st.markdown("""
---
### 📘 How to Use This App

1. Select your **country**, **city**, **university**, **program**, and **level** of study.
2. View your **Total Cost of Attendance** prediction.
3. Explore cost trends using:
   - 🌍 Global Affordability Map
   - 📊 Cost Cluster Explorer
""")
