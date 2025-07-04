import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# --- Page Config ---
st.set_page_config(
   page_title='🎓 International Education Budget Planner',
   page_icon='🎓',
   layout='wide',
   initial_sidebar_state='expanded'
)


# --- Custom CSS ---
st.markdown("""
<style>
   .main-header {
       font-size: 3rem;
       font-weight: bold;
       text-align: center;
       background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
       -webkit-background-clip: text;
       -webkit-text-fill-color: transparent;
       margin-bottom: 2rem;
   }
  
   .metric-card {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       padding: 1.5rem;
       border-radius: 15px;
       text-align: center;
       color: white;
       margin: 1rem 0;
       box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
   }
  
   .cost-tier {
       padding: 0.5rem 1rem;
       border-radius: 25px;
       font-weight: bold;
       text-align: center;
       margin: 0.5rem;
       display: inline-block;
   }
  
   .cost-low {
       background: linear-gradient(135deg, #4CAF50, #45a049);
       color: white;
   }
  
   .cost-medium {
       background: linear-gradient(135deg, #FF9800, #f57c00);
       color: white;
   }
  
   .cost-high {
       background: linear-gradient(135deg, #f44336, #d32f2f);
       color: white;
   }
  
   .info-box {
       background: rgba(255, 255, 255, 0.1);
       padding: 1rem;
       border-radius: 10px;
       border-left: 4px solid #667eea;
       margin: 1rem 0;
   }
  
   .feature-card {
       background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
       padding: 1.5rem;
       border-radius: 15px;
       margin: 1rem 0;
       border: 1px solid rgba(255, 255, 255, 0.1);
       backdrop-filter: blur(10px);
   }
  
   .sidebar .sidebar-content {
       background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
   }
  
   .stSelectbox > div > div > select {
       background: rgba(255, 255, 255, 0.1);
       border-radius: 10px;
   }
</style>
""", unsafe_allow_html=True)


# --- Main Header ---
st.markdown('<h1 class="main-header">🎓 International Education Budget Planner</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">ML-Powered Cost Prediction</p>', unsafe_allow_html=True)


# --- Load Data ---
@st.cache_data
def load_data():
   script_dir = os.path.dirname(os.path.abspath(__file__))
   csv_path = os.path.join(script_dir, "data_tca_clusters_raw.csv")
  
   if not os.path.exists(csv_path):
       st.error(f"📁 CSV file not found at: {csv_path}")
       st.info(f"📂 Current working directory: {os.getcwd()}")
       st.info(f"📂 Script directory: {script_dir}")
       if os.path.exists(script_dir):
           st.info(f"📄 Files in script directory: {os.listdir(script_dir)}")
       st.stop()
  
   return pd.read_csv(csv_path)


# --- Load Pipeline ---
@st.cache_resource
def load_model_pipeline():
   path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
   if not os.path.exists(path):
       st.error("🤖 Model pipeline not found! Please run the training script first.")
       st.stop()
   return joblib.load(path)


# --- Load data and model ---
try:
   data = load_data()
   pipeline = load_model_pipeline()
  
   # Define categorical features (same as in training script)
   categorical_features = ["Country", "City", "University", "Program", "Level"]
  


  
except Exception as e:
   st.error(f"❌ Error loading data or model: {str(e)}")
   st.stop()




# --- Sidebar Inputs ---
st.sidebar.markdown("## 📥 Select Your Preferences")
st.sidebar.markdown("---")


# Country selection with search
st.sidebar.markdown("### 🌍 Destination")
countries = sorted(data["Country"].dropna().unique())
target_country = st.sidebar.selectbox(
   "Select Country",
   countries,
   help="Choose your destination country"
)


# Filter data based on country
filtered_data = data[data["Country"] == target_country]


# City selection
cities = sorted(filtered_data["City"].dropna().unique()) if not filtered_data.empty else ["No cities available"]
target_city = st.sidebar.selectbox(
   "Select City",
   cities,
   help="Choose your destination city"
)


# University selection
if target_city != "No cities available":
   city_filtered = filtered_data[filtered_data["City"] == target_city]
   universities = sorted(city_filtered["University"].dropna().unique()) if not city_filtered.empty else ["No universities available"]
else:
   universities = ["No universities available"]


target_university = st.sidebar.selectbox(
   "Select University",
   universities,
   help="Choose your target university"
)


# Program and Level
st.sidebar.markdown("### 📚 Academic Details")
programs = sorted(data["Program"].dropna().unique())
target_program = st.sidebar.selectbox(
   "Select Program",
   programs,
   help="Choose your field of study"
)


levels = sorted(data["Level"].dropna().unique())
target_level = st.sidebar.selectbox(
   "Select Level",
   levels,
   help="Choose your education level"
)


st.sidebar.markdown("---")


# --- Create prediction input ---
if (target_country and target_city != "No cities available" and
   target_university != "No universities available" and target_program and target_level):
  
   user_input = pd.DataFrame({
       "Country": [target_country],
       "City": [target_city],
       "University": [target_university],
       "Program": [target_program],
       "Level": [target_level]
   })
  
   # --- Make Prediction ---
   try:
       predicted_cost = pipeline.predict(user_input)[0]
      
       # Determine cost tier
       cost_percentiles = data['Total_cost'].quantile([0.33, 0.67])
       if predicted_cost <= cost_percentiles[0.33]:
           cost_tier = "Low"
           tier_color = "cost-low"
           tier_emoji = "💚"
       elif predicted_cost <= cost_percentiles[0.67]:
           cost_tier = "Medium"
           tier_color = "cost-medium"
           tier_emoji = "🟡"
       else:
           cost_tier = "High"
           tier_color = "cost-high"
           tier_emoji = "🔴"
      
       # Display prediction
       st.sidebar.markdown("### 🎯 Prediction Results")
       st.sidebar.markdown(f"""
       <div class="metric-card">
           <h3>💰 Predicted Total Cost</h3>
           <h2>${predicted_cost:,.0f}</h2>
           <div class="cost-tier {tier_color}">
               {tier_emoji} {cost_tier} Cost Tier
           </div>
       </div>
       """, unsafe_allow_html=True)
      
       # Show prediction confidence
       similar_combinations = data[
           (data["Country"] == target_country) &
           (data["Program"] == target_program) &
           (data["Level"] == target_level)
       ]
      
       if not similar_combinations.empty:
           actual_range = f"${similar_combinations['Total_cost'].min():,.0f} - ${similar_combinations['Total_cost'].max():,.0f}"
           st.sidebar.info(f"📊 Similar combinations in dataset: {len(similar_combinations)}\n\n💡 Actual cost range: {actual_range}")
      
   except Exception as e:
       st.sidebar.error(f"❌ Prediction failed: {str(e)}")
       predicted_cost = None
       cost_tier = None
      
else:
   st.sidebar.warning("⚠️ Please select all options to get a prediction")
   predicted_cost = None
   cost_tier = None


# --- Main Content ---
# Create tabs for different views
tab1, tab2 = st.tabs(["🌍 Global Overview", "📈 Program and Level Costs Insights"])


with tab1:
   st.markdown("### 🌍 Global Education Cost Map")
  
   # Create choropleth map
   country_avg = data.groupby('Country')['Total_cost'].mean().reset_index()
  
   fig_map = px.choropleth(
       country_avg,
       locations="Country",
       locationmode="country names",
       color="Total_cost",
       hover_name="Country",
       color_continuous_scale="Viridis",
       title="Average Total Cost of Education by Country",
       labels={"Total_cost": "Total Cost (USD)"},
   )
  
   fig_map.update_geos(
       showframe=False,
       showcoastlines=True,
       projection_type="natural earth",
   )
  
   fig_map.update_layout(
       height=500,
       margin=dict(l=0, r=0, t=40, b=0),
       title_font=dict(size=16),
       coloraxis_colorbar=dict(
           title="Cost (USD)",
           tickprefix="$",
           thickness=15,
           len=0.7,
       )
   )
  
   st.plotly_chart(fig_map, use_container_width=True)
  




with tab2:
   st.markdown("### 📈 Program and Level Costs Insights")
  
   # Generate insights
   col1, col2 = st.columns(2)
  
   with col1:
       st.markdown("#### 🎓 Education Level Insights")
       level_stats = data.groupby('Level')['Tuition_USD'].agg(['mean', 'count']).sort_values('mean', ascending=False)
      
       fig_level_bar = px.bar(
           x=level_stats.index,
           y=level_stats['mean'],
           title="Average Tuition Costs by Education Level",
           labels={'x': 'Education Level', 'y': 'Average Cost (USD)'},
           color=level_stats['mean'],
           color_continuous_scale="Viridis"
       )
       fig_level_bar.update_layout(height=400)
       st.plotly_chart(fig_level_bar, use_container_width=True)
  
   with col2:
       st.markdown("#### 📚 Program Insights")
       program_stats = data.groupby('Program')['Tuition_USD'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
      
       fig_program_bar = px.bar(
           x=program_stats['mean'],
           y=program_stats.index,
           orientation='h',
           title="Top 10 Program with Highest Average Tuition Costs",
           labels={'x': 'Average Cost (USD)', 'y': 'Program'},
           color=program_stats['mean'],
           color_continuous_scale="Plasma"
       )
       fig_program_bar.update_layout(height=400)
       st.plotly_chart(fig_program_bar, use_container_width=True)
  
   # Key statistics


  
   insights_col1, insights_col2, insights_col3, insights_col4 = st.columns(4)
  
   with insights_col1:
       most_expensive_country = data.groupby('Country')['Total_cost'].mean().idxmax()
       st.metric(
           "🌍 Most Expensive Country",
           most_expensive_country,
           f"${data.groupby('Country')['Total_cost'].mean().max():,.0f}"
       )
  
   with insights_col2:
       most_affordable_country = data.groupby('Country')['Total_cost'].mean().idxmin()
       st.metric(
           "💚 Most Affordable Country",
           most_affordable_country,
           f"${data.groupby('Country')['Total_cost'].mean().min():,.0f}"
       )
  
   with insights_col3:
       most_expensive_program = data.groupby('Program')['Total_cost'].mean().idxmax()
       st.metric(
           "📚 Most Expensive Program",
           most_expensive_program,
           f"${data.groupby('Program')['Total_cost'].mean().max():,.0f}"
       )
  

