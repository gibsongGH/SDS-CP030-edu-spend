import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.decomposition import PCA
import io

# Load model and scaler
model = joblib.load("best_classifier_xgb.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("processed_data.csv")

# Title & Description
st.title("🎓 EduSpend Affordability Classifier")
st.markdown("""
Welcome to the **EduSpend App**! 🚀

This interactive app helps you estimate the **affordability tier** of a study program based on various cost factors 💸💰🏦.
""")

# Sidebar for user input
st.sidebar.header("📋 Enter Cost Details")
tuition = st.sidebar.slider("Tuition Fee (USD)", 1000, 60000, 15000, step=500)
living_index = st.sidebar.slider("Living Cost Index", 30, 120, 65, step=1)
rent = st.sidebar.slider("Monthly Rent (USD)", 100, 3000, 900, step=50)
visa = st.sidebar.slider("Visa Fee (USD)", 0, 500, 150, step=10)
insurance = st.sidebar.slider("Insurance Fee (USD)", 100, 2000, 600, step=50)
duration = st.sidebar.selectbox("Program Duration (Years)", [1, 2, 3, 4, 5])

# Prepare and scale input
data_input = pd.DataFrame({
    "Tuition_USD": [tuition],
    "Living_Cost_Index": [living_index],
    "Rent_USD": [rent],
    "Visa_Fee_USD": [visa],
    "Insurance_USD": [insurance],
    "Duration_Years": [duration],
    "Total_Cost_USD": [tuition + (rent + insurance + visa) * duration]  # Synthetic total cost
})

scaled_input = scaler.transform(data_input)

# Make prediction
prediction = model.predict(scaled_input)[0]
tier_map = {0: "Low 💸", 1: "Medium 💰", 2: "High 🏦"}
st.success(f"### 🎯 Predicted Affordability Tier: {tier_map[prediction]}")

# Visualizations
st.markdown("---")
st.subheader("📊 Affordability Tier Distribution")
tier_counts = df["Affordability_Tier"].value_counts().sort_index()
st.bar_chart(tier_counts.rename(index={0: "Low", 1: "Medium", 2: "High"}))

st.markdown("---")
st.subheader("📈 Compare Your Input With Dataset")
col1, col2 = st.columns(2)

with col1:
    st.write("**Your Input vs. Tuition Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["Tuition_USD"], bins=30, kde=True, ax=ax)
    ax.axvline(tuition, color='red', linestyle='--', label='Your Input')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.write("**Your Input vs. Rent Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["Rent_USD"], bins=30, kde=True, ax=ax)
    ax.axvline(rent, color='red', linestyle='--', label='Your Input')
    ax.legend()
    st.pyplot(fig)

# File upload for batch predictions
st.markdown("---")
st.subheader("📁 Upload File for Batch Predictions")
uploaded_file = st.file_uploader("Upload a CSV file with the same structure as the input fields", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_data["Total_Cost_USD"] = batch_data["Tuition_USD"] + (
        batch_data["Rent_USD"] + batch_data["Insurance_USD"] + batch_data["Visa_Fee_USD"]
    ) * batch_data["Duration_Years"]
    scaled_batch = scaler.transform(batch_data)
    batch_preds = model.predict(scaled_batch)
    batch_data["Predicted_Tier"] = [tier_map[p] for p in batch_preds]
    st.write("### 📄 Results:")
    st.dataframe(batch_data)
    csv = batch_data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# SHAP Explanation
st.markdown("---")
st.subheader("🧠 Model Explainability (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(scaled_input)

# Safely display SHAP explanation for a single instance
shap_value = shap_values[0]

st.write("**Feature Impact for Your Input:**")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_value, max_display=7, show=False)
plt.tight_layout()
st.pyplot(fig)

# Export SHAP plot
buf = io.BytesIO()
fig.savefig(buf, format="png")
st.download_button(
    label="🖼️ Download SHAP Plot",
    data=buf.getvalue(),
    file_name="shap_explanation.png",
    mime="image/png"
)

# Clustering comparison using PCA
st.markdown("---")
st.subheader("✨ Clustering Comparison (PCA Projection)")
if "Cost_Archetype" in df.columns:
    X_cluster = df[[
        "Tuition_USD", "Living_Cost_Index", "Rent_USD",
        "Visa_Fee_USD", "Insurance_USD", "Duration_Years", "Total_Cost_USD"
    ]]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaler.transform(X_cluster))

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cost_Archetype"], cmap="Set1", alpha=0.6)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Cost Archetypes Clustering (PCA Projection)")
    legend = ax.legend(*scatter.legend_elements(), title="Archetype")
    ax.add_artist(legend)
    st.pyplot(fig)

st.markdown("---")
st.caption("Made with 💖 using Streamlit, SHAP & XGBoost")






