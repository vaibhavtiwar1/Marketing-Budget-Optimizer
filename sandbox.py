import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Media Mix Optimizer", layout="wide")
st.title("📊 Algorithmic Media Mix Budget Optimizer")
st.markdown("""
*Built for Growth Marketers & Strategists to solve the problem of diminishing returns at scale.*
""")

# 2. Sidebar - Historical Data Input (Option 1)
st.sidebar.header("🎯 Campaign Parameters")
target_leads = st.sidebar.number_input("Target Total Leads", min_value=10, max_value=10000, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Historical Channel Data")
st.sidebar.caption("Input your past 30-day performance. The algorithm will calculate the true efficiency of each channel.")

# Default data to make the portfolio look good instantly
default_data = pd.DataFrame({
    "Channel": ["Paid Search", "Paid Social", "Cold Outbound", "Content SEO"],
    "Past Spend (K)": [5.0, 3.0, 2.0, 1.5],
    "Past Leads": [268, 138, 92, 55]
})

# Streamlit Data Editor allows users to add/remove rows directly!
edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", hide_index=True)

# 3. Mathematical Backend (Calculating Alpha dynamically)
channels = {}
for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    spend = float(row["Past Spend (K)"])
    leads = float(row["Past Leads"])
    
    # Safeguard against division by zero or negative inputs
    if spend > 0 and leads > 0 and pd.notna(channel_name):
        # Reverse engineer Alpha: Alpha = Leads / ln(Spend + 1)
        alpha = leads / np.log(spend + 1)
        channels[channel_name] = alpha

if len(channels) == 0:
    st.warning("Please add at least one valid channel with Spend and Leads greater than 0.")
    st.stop()

def lead_function(budget, alpha):
    return alpha * np.log(budget + 1)

def objective_function(budgets):
    return np.sum(budgets)

def constraint_target_leads(budgets):
    total_leads = sum(lead_function(b, alpha) for b, alpha in zip(budgets, channels.values()))
    return total_leads - target_leads

init_guess = [1.0] * len(channels)
bounds = [(0, 1000)] * len(channels) # Max 1 million (1000K) per channel
constraints = {'type': 'ineq', 'fun': constraint_target_leads}

# Calculate the theoretical ceiling before optimizing
theoretical_max_leads = sum(lead_function(1000, alpha) for alpha in channels.values())

if target_leads > theoretical_max_leads:
    st.error("🛑 **Mathematical Limit Reached!**")
    st.write(f"Based on your historical efficiency, the absolute maximum number of leads this mix can generate before hitting the budget ceiling ($1,000K per channel) is **{int(theoretical_max_leads)}**.")
    st.info("💡 **How to fix:** Lower your 'Target Total Leads' below this number, or input more efficient historical channel data.")
    st.stop() # This halts the app gracefully so the optimizer doesn't crash!

# Run Optimizer
result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# 4. Dashboard UI Layout
if result.success:
    optimized_budgets = result.x
    channel_leads = [lead_function(b, a) for b, a in zip(optimized_budgets, channels.values())]
    total_spend = np.sum(optimized_budgets)
    blended_cac = total_spend / target_leads
    
    # Display KPI Metrics with the (K) explicitly added
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimized Budget Needed (K)", f"${total_spend:,.2f}")
    col2.metric("Target Leads Met", f"{int(sum(channel_leads))}")
    col3.metric("Blended CAC (K)", f"${blended_cac:,.2f}")
    
    st.markdown("---")
    
    # Results Dataframe
    df_results = pd.DataFrame({
        "Channel": list(channels.keys()),
        "Recommended Budget (K)": optimized_budgets,
        "Expected Leads": channel_leads,
        "Effective CAC (K)": [b/l if l > 0 else 0 for b, l in zip(optimized_budgets, channel_leads)]
    })
    
    col_table, col_chart = st.columns([1, 1.2])
    with col_table:
        st.subheader("Budget Allocation Breakdown")
        st.dataframe(df_results.style.format({
            "Recommended Budget (K)": "${:,.2f}",
            "Expected Leads": "{:.1f}",
            "Effective CAC (K)": "${:,.2f}"
        }))
        
        st.subheader("Budget Distribution")
        fig_pie = px.pie(df_results, values="Recommended Budget (K)", names="Channel", hole=0.4)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 5. Visual Overhaul: The Saturation Heatmap (Option 2)
    with col_chart:
        st.subheader("Diminishing Returns Saturation Point")
        st.caption("The dots show exactly where the algorithm stopped spending on each channel before it became inefficient.")
        
        fig_curve = go.Figure()
        max_b = max(optimized_budgets) * 1.5 if max(optimized_budgets) > 0 else 10
        budget_range = np.linspace(0, max_b, 100)
        
        # Plot curves and optimal points for each channel
        for i, (channel, alpha) in enumerate(channels.items()):
            opt_b = optimized_budgets[i]
            opt_l = channel_leads[i]
            
            # The continuous curve
            fig_curve.add_trace(go.Scatter(
                x=budget_range, 
                y=[lead_function(b, alpha) for b in budget_range],
                mode='lines',
                name=channel,
                line=dict(width=2),
                opacity=0.6
            ))
            
            # The 'optimized stop point' dot
            fig_curve.add_trace(go.Scatter(
                x=[opt_b], 
                y=[opt_l],
                mode='markers+text',
                marker=dict(size=12, line=dict(width=2, color='white')),
                showlegend=False,
                hoverinfo="text",
                textposition="top center",
                hovertext=f"{channel}<br>Stop Point: ${opt_b:.2f}K"
            ))

        fig_curve.update_layout(
            xaxis_title="Budget Spend (K)",
            yaxis_title="Leads Generated",
            hovermode="x unified",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

else:
    st.error("The optimizer could not find a feasible solution. Check your historical data inputs.")

# 6. Recruiter Narrative Block
st.markdown("---")
with st.expander("🔬 Methodology & Data Science Breakdown (For Recruiters)"):
    st.markdown("""
    ### Turning Historical Data into Predictive Models
    Standard budget tools split ratios linearly based on flat CACs. In reality, ad networks saturate. 
    This tool reverses engineers historical campaign data to find a channel's intrinsic efficiency (Alpha), and then applies a **Logarithmic Saturation Curve** to model diminishing returns. 
    
    * **Math Formula used:** $\\text{Leads} = \\alpha \\cdot \\ln(\\text{Budget} + 1)$
    * **Optimization Engine:** Sequential Least Squares Programming (`SLSQP`) via `scipy.optimize` minimizes total budget while ensuring lead targets are met.
    * **Note on Scaling:** To keep the math highly responsive, inputs and outputs are scaled in **thousands (K)**.
    """)
