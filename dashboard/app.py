import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import joblib

# ==========================================
# 1. CONFIG & LOAD MODEL
# ==========================================
st.set_page_config(page_title="MMM Optimizer Pro", layout="wide")

# Helper function to load the model safely
@st.cache_resource
def load_model():
    try:
        return joblib.load('src/mmm_model.pkl') # If run from root
    except:
        return joblib.load('../src/mmm_model.pkl') # If run from dashboard folder

try:
    model_artifacts = load_model()
    model = model_artifacts['model']
    features = model_artifacts['features']
    sat_params = model_artifacts['saturation_params']
    
    # Extract coefficients
    coefs = dict(zip(features, model.coef_))
    
    # Metrics
    r2_score = model_artifacts['performance']['R2']
    mape_score = model_artifacts['performance']['MAPE']

except FileNotFoundError:
    st.error("Model file not found! Please run '03_modeling_pro.ipynb' to generate 'src/mmm_model.pkl'.")
    st.stop()

# ==========================================
# 2. UTILITY FUNCTIONS (The Math)
# ==========================================
def hill_saturation(spend, alpha, beta):
    # FIX: Avoid division by zero error when spend is 0
    # We replace 0 with a tiny number (1e-9) so the math doesn't break
    safe_spend = np.maximum(spend, 1e-9) 
    return 1 / (1 + (safe_spend / alpha)**(-beta))

def calculate_prediction(budget_allocation):
    tv, social, radio = budget_allocation
    
    # 1. Apply Saturation (Diminishing Returns)
    tv_eff = hill_saturation(tv, sat_params['TV']['alpha'], sat_params['TV']['beta'])
    social_eff = hill_saturation(social, sat_params['Social']['alpha'], sat_params['Social']['beta'])
    radio_eff = hill_saturation(radio, sat_params['Radio']['alpha'], sat_params['Radio']['beta'])
    
    # 2. Apply ROI Coefficients
    pred_sales = (tv_eff * coefs['TV_Adstock'] * sat_params['TV']['alpha']) + \
                 (social_eff * coefs['Social_Adstock'] * sat_params['Social']['alpha']) + \
                 (radio_eff * coefs['Radio_Adstock'] * sat_params['Radio']['alpha'])
                 
    return pred_sales

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.title("Enterprise Marketing Mix Modeler")
st.markdown(f"""
**Model Status:** Trained on 156 weeks of data.  
**Precision:** MAPE: `{mape_score:.2%}` | R²: `{r2_score:.4f}`
""")

st.markdown("---")

# SIDEBAR
st.sidebar.header("Scenario Planner")
total_budget = st.sidebar.number_input("Total Budget ($)", min_value=1000, max_value=200000, value=50000, step=1000)

st.sidebar.subheader("Manual Allocation")
tv_alloc = st.sidebar.slider("TV Spend", 0, total_budget, int(total_budget*0.33))
remaining = total_budget - tv_alloc
social_alloc = st.sidebar.slider("Social Spend", 0, remaining, int(remaining*0.5))
radio_alloc = total_budget - tv_alloc - social_alloc
st.sidebar.write(f"**Radio Spend (Auto-calc):** ${radio_alloc:,.2f}")

# MAIN PANEL
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Response Curves (The 'Why')")
    st.write("See how each channel saturates. Notice where the curve flattens!")
    
    # Plotting the Hill Curves dynamically
    x_values = np.linspace(0, 100000, 100)
    
    fig_curves = go.Figure()
    for channel, color in zip(['TV', 'Social', 'Radio'], ['blue', 'orange', 'green']):
        # Calculate theoretical return for plotting
        y_values = hill_saturation(x_values, sat_params[channel]['alpha'], sat_params[channel]['beta']) 
        y_values = y_values * coefs[f'{channel}_Adstock']
        
        fig_curves.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=channel, line=dict(color=color)))
    
    fig_curves.update_layout(title="Diminishing Returns per Channel", xaxis_title="Spend ($)", yaxis_title="Impact Index")
    st.plotly_chart(fig_curves, use_container_width=True)

with col2:
    st.subheader("Prediction")
    
    # Manual Prediction
    manual_pred = calculate_prediction([tv_alloc, social_alloc, radio_alloc])
    
    st.metric("Predicted Sales (Manual)", f"${manual_pred:,.0f}")
    
    if st.button("Run AI Optimization"):
        with st.spinner("Solving non-linear optimization problem..."):
            # Run Optimization
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
            bounds = ((0, total_budget), (0, total_budget), (0, total_budget))
            
            res = minimize(
                lambda x: -calculate_prediction(x), 
                [total_budget/3]*3, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            opt_sales = -res.fun
            opt_tv, opt_social, opt_radio = res.x
            
            st.success(f"Found Optimal Split! Potential Lift: +${opt_sales - manual_pred:,.0f}")
            
            # Show Optimal Split
            df_opt = pd.DataFrame({
                'Channel': ['TV', 'Social', 'Radio'],
                'Recommended Spend': [opt_tv, opt_social, opt_radio]
            })
            
            # FIX: Use Streamlit's native formatting to prevent crashing
            st.dataframe(
                df_opt,
                column_config={
                    "Recommended Spend": st.column_config.NumberColumn(
                        "Recommended Spend",
                        format="$%.2f"
                    )
                },
                hide_index=True
            )
            
            # Visualization of Split
            fig_pie = px.pie(df_opt, values='Recommended Spend', names='Channel', title="AI Budget Allocation")
            st.plotly_chart(fig_pie, use_container_width=True)