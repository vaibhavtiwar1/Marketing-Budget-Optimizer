import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Media Mix Optimizer", layout="wide")
st.title("📊 Algorithmic Media Mix Budget Optimizer")
st.markdown("*Built for Growth Marketers & Strategists to solve the problem of diminishing returns at scale.*")

# 2. Sidebar - Basic Inputs
st.sidebar.header("🎯 Campaign Parameters")
target_leads = st.sidebar.number_input("Target Total Leads", min_value=10, max_value=10000, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Historical Channel Data")
default_data = pd.DataFrame({
    "Channel": ["Paid Search", "Paid Social", "Cold Email"],
    "Past Spend (K)": [5.0, 4.0, 1.5],
    "Past Leads": [268, 110, 80]
})
edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", hide_index=True)

# 3. The "Advanced Mode" Toggle (UI Magic)
st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("⚙️ Enable Advanced Mode")

# Dictionary to hold dynamic channel configurations
adv_settings = {}

if advanced_mode:
    st.sidebar.caption("Configure individual channel economics.")
    for index, row in edited_df.iterrows():
        channel_name = row["Channel"]
        if pd.notna(channel_name) and str(channel_name).strip() != "":
            with st.sidebar.expander(f"{channel_name} Settings"):
                model = st.selectbox("Growth Model", ["Logarithmic (Search)", "S-Curve (Social)", "Linear (Email)"], key=f"mod_{channel_name}")
                fixed_cost = st.number_input("Fixed Cost / Retainer (K)", min_value=0.0, value=0.0, step=0.5, key=f"fix_{channel_name}")
                learning = st.checkbox("In Learning Phase?", key=f"lrn_{channel_name}")
                adv_settings[channel_name] = {"model": model, "fixed_cost": fixed_cost, "learning": learning}
else:
    # Default settings if advanced mode is off
    for index, row in edited_df.iterrows():
        channel_name = row["Channel"]
        if pd.notna(channel_name):
            # Auto-guess model based on name for a better default experience
            default_model = "Linear (Email)" if "email" in str(channel_name).lower() else "S-Curve (Social)" if "social" in str(channel_name).lower() else "Logarithmic (Search)"
            adv_settings[channel_name] = {"model": default_model, "fixed_cost": 0.0, "learning": False}

# 4. Dynamic Math Engine (Calculating Alpha per Model)
channels = {}
for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    spend = float(row["Past Spend (K)"])
    leads = float(row["Past Leads"])
    
    if spend > 0 and leads > 0 and pd.notna(channel_name):
        settings = adv_settings[channel_name]
        var_spend = max(0.01, spend - settings["fixed_cost"]) # Isolate variable spend
        
        # Reverse engineer Alpha based on the specific math model
        if "Logarithmic" in settings["model"]:
            alpha = leads / np.log(var_spend + 1)
        elif "Linear" in settings["model"]:
            alpha = leads / var_spend
        elif "S-Curve" in settings["model"]:
            alpha = leads / ((var_spend**1.5) / (var_spend + 5))
            
        channels[channel_name] = {"alpha": alpha, "settings": settings}

if len(channels) == 0:
    st.warning("Please add at least one valid channel with Spend and Leads.")
    st.stop()

# Dynamic lead calculation function
def lead_function(budget, channel_name):
    ch_data = channels[channel_name]
    alpha = ch_data["alpha"]
    settings = ch_data["settings"]
    
    eff_budget = max(0.0, budget - settings["fixed_cost"])
    if eff_budget == 0:
        return 0.0
        
    if "Logarithmic" in settings["model"]:
        return alpha * np.log(eff_budget + 1)
    elif "Linear" in settings["model"]:
        return alpha * eff_budget
    elif "S-Curve" in settings["model"]:
        return alpha * ((eff_budget**1.5) / (eff_budget + 5))

# Optimizer Setup
def objective_function(budgets):
    return np.sum(budgets)

def constraint_target_leads(budgets):
    total_leads = sum(lead_function(b, ch) for b, ch in zip(budgets, channels.keys()))
    return total_leads - target_leads

init_guess = []
bounds = []
for ch_name, ch_data in channels.items():
    settings = ch_data["settings"]
    min_bound = settings["fixed_cost"]
    if settings["learning"]:
        min_bound += 2.0 # Force a minimum 2K exploration budget
    init_guess.append(max(1.0, min_bound))
    bounds.append((min_bound, 1000))

constraints = {'type': 'ineq', 'fun': constraint_target_leads}

# 5. Smart Error Handling (The Ceiling Check)
theoretical_max_leads = sum(lead_function(1000, ch) for ch in channels.keys())
if target_leads > theoretical_max_leads:
    st.error("🛑 **Mathematical Limit Reached!**")
    st.write(f"Based on your historical efficiency, the absolute maximum number of leads this mix can generate before hitting the budget ceiling ($1,000K per channel) is **{int(theoretical_max_leads)}**.")
    st.stop()

# Run Optimizer
result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# 6. Split-Screen Layout (Dashboard + AI Advisor)
if result.success:
    optimized_budgets = result.x
    channel_leads = [lead_function(b, ch) for b, ch in zip(optimized_budgets, channels.keys())]
    total_spend = np.sum(optimized_budgets)
    blended_cac = total_spend / target_leads
    
    # --- UI SPLIT: 70% Dashboard / 30% AI Chat ---
    dash_col, chat_col = st.columns([7, 3], gap="large")
    
    # ==========================================
    # LEFT COLUMN: The Math & Charts Dashboard
    # ==========================================
    with dash_col:
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimized Budget Needed (K)", f"${total_spend:,.2f}")
        col2.metric("Target Leads Met", f"{int(sum(channel_leads))}")
        col3.metric("Blended CAC (K)", f"${blended_cac:,.2f}")
        
        st.markdown("---")
        
        df_results = pd.DataFrame({
            "Channel": list(channels.keys()),
            "Model Type": [channels[ch]["settings"]["model"].split(" ")[0] for ch in channels.keys()],
            "Recommended Budget (K)": optimized_budgets,
            "Expected Leads": channel_leads,
            "Effective CAC (K)": [b/l if l > 0 else 0 for b, l in zip(optimized_budgets, channel_leads)]
        })
        
        col_table, col_chart = st.columns([1.2, 1])
        with col_table:
            st.subheader("Budget Allocation Breakdown")
            st.dataframe(df_results.style.format({
                "Recommended Budget (K)": "${:,.2f}",
                "Expected Leads": "{:.1f}",
                "Effective CAC (K)": "${:,.2f}"
            }))
            
        with col_chart:
            st.subheader("Budget Distribution")
            fig_pie = px.pie(df_results, values="Recommended Budget (K)", names="Channel", hole=0.4)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("Multi-Model Saturation Curves")
        
        fig_curve = go.Figure()
        max_b = max(optimized_budgets) * 1.5 if max(optimized_budgets) > 0 else 10
        budget_range = np.linspace(0, max_b, 100)
        
        for i, ch_name in enumerate(channels.keys()):
            opt_b = optimized_budgets[i]
            opt_l = channel_leads[i]
            fig_curve.add_trace(go.Scatter(x=budget_range, y=[lead_function(b, ch_name) for b in budget_range], mode='lines', name=ch_name, line=dict(width=2), opacity=0.6))
            fig_curve.add_trace(go.Scatter(x=[opt_b], y=[opt_l], mode='markers+text', marker=dict(size=12, line=dict(width=2, color='white')), showlegend=False, hoverinfo="text", textposition="top left", hovertext=f"{ch_name}<br>${opt_b:.2f}K"))

        fig_curve.update_layout(xaxis_title="Budget Spend (K)", yaxis_title="Leads Generated", hovermode="x unified", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_curve, use_container_width=True)

    # ==========================================
    # RIGHT COLUMN: AI Marketing Advisor Chat
    # ==========================================
    with chat_col:
        st.subheader("🧠 AI Marketing Advisor")
        st.caption("Ask for strategic advice based on your current budget mix.")
        
        # 1. Initialize chat memory
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your AI Marketing Advisor. Looking at your current saturation curves, how can I help you optimize this budget?"}]

        # 2. Display previous chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Chat Input Trigger
        if prompt := st.chat_input("E.g., Which channel should I cut first?"):
            # Add user message to UI
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 4. Generate AI Response (Mocked for now until we add the API!)
            # In the next step, we will replace this mock text with a live LLM API call
            mock_response = f"*(This is a placeholder for the AI.)* To answer your question about '{prompt}', I would look at the Effective CAC column. Since {list(channels.keys())[0]} has the current highest efficiency, we should protect its budget first."
            
            # Add AI message to UI
            with st.chat_message("assistant"):
                st.markdown(mock_response)
            st.session_state.messages.append({"role": "assistant", "content": mock_response})

else:
    st.error("The optimizer could not find a feasible solution. Check your historical data inputs.")
