import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Media Mix Optimizer", layout="wide")
st.title("📊 Algorithmic Media Mix Budget Optimizer")
st.markdown("""
*Built for Growth Marketers & Strategists to solve the problem of diminishing returns at scale.*
""")

# Industry Presets
PRESETS = {
    "B2B SaaS": {"Paid Search": 150, "Paid Social": 100, "Cold Outbound": 80},
    "E-commerce": {"Paid Search": 200, "Paid Social": 250, "Influencer Mktg": 180},
    "Custom": {"Paid Search": 100, "Paid Social": 100, "Other Channel": 100}
}

# Sidebar Inputs
st.sidebar.header("🎯 Campaign Parameters")
industry = st.sidebar.selectbox("Select Industry Preset", list(PRESETS.keys()))
target_leads = st.sidebar.number_input("Target Total Leads", min_value=10, max_value=5000, value=500, step=50)

st.sidebar.subheader("Adjust Channel Weights (Alpha)")
channels = {}
for channel, default_alpha in PRESETS[industry].items():
    channels[channel] = st.sidebar.slider(f"{channel} Efficiency", 10, 500, default_alpha)

# Mathematical Backend
def lead_function(budget, alpha):
    return alpha * np.log(budget + 1)

def objective_function(budgets):
    return np.sum(budgets)

def constraint_target_leads(budgets):
    total_leads = sum(lead_function(b, alpha) for b, alpha in zip(budgets, channels.values()))
    return total_leads - target_leads

init_guess = [1000] * len(channels)
bounds = [(0, 1000000)] * len(channels)
constraints = {'type': 'ineq', 'fun': constraint_target_leads}

result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Dashboard UI Layout
if result.success:
    optimized_budgets = result.x
    channel_leads = [lead_function(b, a) for b, a in zip(optimized_budgets, channels.values())]
    total_spend = np.sum(optimized_budgets)
    blended_cac = total_spend / target_leads
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimized Budget Needed", f"${total_spend:,.2f}")
    col2.metric("Target Leads Met", f"{int(sum(channel_leads))}")
    col3.metric("Blended CAC", f"${blended_cac:,.2f}")
    
    st.markdown("---")
    
    df_results = pd.DataFrame({
        "Channel": list(channels.keys()),
        "Recommended Budget": optimized_budgets,
        "Expected Leads": channel_leads,
        "Effective CAC": [b/l if l > 0 else 0 for b, l in zip(optimized_budgets, channel_leads)]
    })
    
    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.subheader("Budget Allocation Breakdown")
        st.dataframe(df_results.style.format({
            "Recommended Budget": "${:,.2f}",
            "Expected Leads": "{:.1f}",
            "Effective CAC": "${:,.2f}"
        }))
        
    with col_chart:
        st.subheader("Budget Distribution")
        fig = px.pie(df_results, values="Recommended Budget", names="Channel", hole=0.4)
        st.plotly_chart(fig, theme="streamlit")

else:
    st.error("The optimizer could not find a feasible solution. Try reducing the target lead count or increasing channel efficiencies.")

# Recruiter Narrative Block
st.markdown("---")
with st.expander("🔬 Methodology & Data Science Breakdown (For Recruiters)"):
    st.markdown("""
    ### Why this beats a standard linear calculator:
    Standard budget tools split ratios linearly. In reality, ad networks saturate. 
    This tool utilizes a non-linear **Logarithmic Saturation Curve** to model diminishing returns. 
    
    **Technical Stack:**
    * **Optimization Engine:** Sequential Least Squares Programming (`SLSQP`) via `scipy.optimize`.
    * **Frontend:** Streamlit UI mapped to live interactive Plotly visual vectors.
    """)
