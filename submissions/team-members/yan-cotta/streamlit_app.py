import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EduSpend: Education Cost Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing components."""
    try:
        # Load the dataset to get categorical values
        df = pd.read_csv('data/International_Education_Costs.csv')
        
        # Get unique values for categorical features
        countries = sorted(df['Country'].unique())
        programs = sorted(df['Program'].unique())
        levels = sorted(df['Level'].unique())
        cities = sorted(df['City'].unique())
        
        # Load top cities if available
        try:
            with open('top_cities_list.pkl', 'rb') as f:
                top_cities = pickle.load(f)
        except:
            top_cities = cities[:20]  # Fallback to first 20 cities
        
        return {
            'countries': countries,
            'programs': programs,
            'levels': levels,
            'cities': cities,
            'top_cities': top_cities,
            'dataset': df
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None

# TCA Prediction Function (Simplified for Streamlit)
def predict_tca(country, program, level, city, living_cost_index, rent_usd, insurance_usd, visa_fee_usd, duration_years):
    """
    Simplified TCA prediction based on the main cost components.
    This is a simplified version for demonstration. In production, you would load the actual trained model.
    """
    # Base calculation similar to the actual TCA formula
    base_tca = (rent_usd * 12 * duration_years) + insurance_usd + visa_fee_usd
    
    # Country-based multipliers (based on our model findings)
    country_multipliers = {
        'USA': 1.3, 'UK': 1.25, 'Canada': 1.2, 'Australia': 1.15,
        'Germany': 0.8, 'France': 0.85, 'Netherlands': 0.9, 'Sweden': 0.95,
        'India': 0.3, 'China': 0.4, 'Japan': 1.0, 'South Korea': 0.7
    }
    
    # Program-based multipliers
    program_multipliers = {
        'Medicine': 1.4, 'Business': 1.2, 'Engineering': 1.1, 'Law': 1.15,
        'Computer Science': 1.05, 'Arts': 0.9, 'Science': 0.95, 'Social Sciences': 0.85
    }
    
    # Level-based multipliers
    level_multipliers = {
        'PhD': 1.2, 'Masters': 1.0, 'Bachelors': 0.85, 'Diploma': 0.7
    }
    
    # Apply multipliers
    country_mult = country_multipliers.get(country, 1.0)
    program_mult = program_multipliers.get(program, 1.0)
    level_mult = level_multipliers.get(level, 1.0)
    
    # Living cost index adjustment
    living_cost_adj = living_cost_index / 100
    
    # Calculate estimated tuition based on patterns
    estimated_tuition = 25000 * country_mult * program_mult * level_mult * living_cost_adj
    
    # Final TCA calculation
    predicted_tca = base_tca + estimated_tuition
    
    return max(predicted_tca, 5000)  # Minimum reasonable TCA

# Main Streamlit App
def main():
    # Load model artifacts
    artifacts = load_model_artifacts()
    if not artifacts:
        st.error("Failed to load model artifacts. Please ensure the data files are available.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🎓 EduSpend: Global Education Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict your Total Cost of Attendance (TCA) for international education")
    
    # Sidebar for input parameters
    st.sidebar.markdown('<div class="sidebar-header">📊 Input Parameters</div>', unsafe_allow_html=True)
    
    # Input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        country = st.selectbox("🌍 Country", artifacts['countries'], index=artifacts['countries'].index('USA') if 'USA' in artifacts['countries'] else 0)
        program = st.selectbox("📚 Program", artifacts['programs'], index=artifacts['programs'].index('Computer Science') if 'Computer Science' in artifacts['programs'] else 0)
    
    with col2:
        level = st.selectbox("🎓 Level", artifacts['levels'], index=artifacts['levels'].index('Masters') if 'Masters' in artifacts['levels'] else 0)
        duration_years = st.number_input("⏱️ Duration (Years)", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
    
    # City selection
    city = st.sidebar.selectbox("🏙️ City", artifacts['top_cities'])
    
    # Cost parameters
    st.sidebar.markdown("### 💰 Cost Parameters")
    living_cost_index = st.sidebar.slider("📈 Living Cost Index", min_value=30, max_value=200, value=100)
    rent_usd = st.sidebar.number_input("🏠 Monthly Rent (USD)", min_value=200, max_value=5000, value=1200, step=50)
    insurance_usd = st.sidebar.number_input("🏥 Health Insurance (USD/year)", min_value=0, max_value=5000, value=1500, step=100)
    visa_fee_usd = st.sidebar.number_input("🛂 Visa Fee (USD)", min_value=0, max_value=2000, value=500, step=50)
    
    # Prediction button
    if st.sidebar.button("🔮 Predict TCA", type="primary"):
        # Make prediction
        predicted_tca = predict_tca(
            country, program, level, city, living_cost_index, 
            rent_usd, insurance_usd, visa_fee_usd, duration_years
        )
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Total Cost of Attendance</h2>
            <h1>${predicted_tca:,.0f}</h1>
            <p>For {level} in {program} at {city}, {country}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cost breakdown
        st.markdown("### 📊 Cost Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        
        annual_rent = rent_usd * 12
        total_rent = annual_rent * duration_years
        estimated_tuition = predicted_tca - total_rent - insurance_usd - visa_fee_usd
        
        with col1:
            st.metric("🏠 Total Housing", f"${total_rent:,.0f}", f"{total_rent/predicted_tca*100:.1f}%")
        with col2:
            st.metric("📚 Estimated Tuition", f"${estimated_tuition:,.0f}", f"{estimated_tuition/predicted_tca*100:.1f}%")
        with col3:
            st.metric("🏥 Health Insurance", f"${insurance_usd:,.0f}", f"{insurance_usd/predicted_tca*100:.1f}%")
        with col4:
            st.metric("🛂 Visa Fee", f"${visa_fee_usd:,.0f}", f"{visa_fee_usd/predicted_tca*100:.1f}%")
        
        # Visualization
        st.markdown("### 📈 Cost Visualization")
        
        # Pie chart for cost breakdown
        labels = ['Housing', 'Tuition', 'Insurance', 'Visa Fee']
        values = [total_rent, estimated_tuition, insurance_usd, visa_fee_usd]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Cost Distribution",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset insights
    st.markdown("### 📊 Dataset Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Countries Available")
        country_counts = artifacts['dataset']['Country'].value_counts().head(10)
        fig_countries = px.bar(
            x=country_counts.values, 
            y=country_counts.index, 
            orientation='h',
            title="Top 10 Countries by Number of Programs"
        )
        fig_countries.update_layout(height=400)
        st.plotly_chart(fig_countries, use_container_width=True)
    
    with col2:
        st.markdown("#### 📚 Programs Available")
        program_counts = artifacts['dataset']['Program'].value_counts().head(10)
        fig_programs = px.bar(
            x=program_counts.values, 
            y=program_counts.index, 
            orientation='h',
            title="Top 10 Programs by Availability",
            color=program_counts.values,
            color_continuous_scale='viridis'
        )
        fig_programs.update_layout(height=400)
        st.plotly_chart(fig_programs, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>🎓 EduSpend Education Cost Predictor | Built with Streamlit & Machine Learning</p>
        <p>Developed by yan-cotta | Data-driven insights for international education planning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
