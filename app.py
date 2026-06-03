import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types

# ==========================================
# 1. Page Configuration & Theme
# ==========================================
st.set_page_config(page_title="Enterprise Marketing Optimizer", layout="wide")

# --- Define the Pop-Up Modal ---
@st.dialog("📖 How to Leverage this Optimizer")
def guide_modal():
    st.markdown("""
    **Welcome to the Enterprise Marketing Optimizer!** Standard marketing calculators use flawed linear math (assuming $1,000 always gets 10x more leads than $100). This engine uses non-linear calculus to find the exact point of diminishing returns for every channel.
    
    ### 🚀 Step-by-Step Guide
    1. **Input Your Baseline:** Go to the left sidebar and enter your historical spend and total leads for each channel.
    2. **Set Your Target:** Adjust the *Target Total Leads* slider to define your pipeline goal.
    3. **Refine the Math (Advanced):** Toggle *Advanced Controls* to manually assign different mathematical curves to specific channels or set fixed infrastructure costs (like software subscriptions).
    4. **Analyze the Matrix:** Review the *Optimal Budget Allocation* table. The algorithm actively shifts money away from saturated channels and redirects it into high-efficiency, scaling channels to minimize your Blended CAC.
    5. **Consult the AI:** Switch to the *AI Strategic Advisor* tab. The agent reads your live matrix and provides Fortune-500 level strategic advice on your next move.

    ### 🧮 The 5 Mathematical Archetypes
    * **Logarithmic (Search):** High initial efficiency, but hits a hard saturation ceiling rapidly.
    * **Hill Function (Paid Social):** Starts slow (platform learning phase), hits a hyper-efficient scaling period, then flattens out.
    * **Negative Exponential (Email):** Scales linearly until it hits the absolute limit of your audience/list size.
    * **Power Law (SEO):** Diminishes slowly and compounds over time without a rigid ceiling.
    * **Shifted Sigmoid (Events):** Yields zero results until a massive minimum entry cost is crossed.
    """)

# --- Header with Top-Right Button ---
col_title, col_btn = st.columns([8.5, 1.5])
with col_title:
    st.title("📊 Enterprise Algorithmic Marketing Budget Optimizer")
    st.markdown("*An advanced, multi-model budget allocation simulator powered by continuous curves and live AI analysis.*")

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True) # Adds padding to align button with title
    if st.button("📖 How to Use Tool", use_container_width=True):
        guide_modal()
# ==========================================
# 2. Sidebar - Basic Parameters & Data Input
# ==========================================
st.sidebar.header("🎯 Campaign Parameters")
target_leads = st.sidebar.number_input("Target Total Leads", min_value=10, max_value=10000, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Historical Channel Data")
st.sidebar.caption("Enter historical monthly spend and total conversions achieved.")

default_data = pd.DataFrame({
    "Channel": ["Google Paid Search", "Meta Paid Social", "HubSpot Email Blast", "SEO Blog Content"],
    "Past Spend (K)": [5.0, 4.0, 1.0, 3.0],
    "Past Leads": [250, 180, 95, 120]
})
edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", hide_index=True)

# ==========================================
# 3. Intelligent Default Keyword Matching
# ==========================================
def guess_default_model(channel_name):
    name = str(channel_name).lower()
    if any(k in name for k in ["search", "google", "bing"]):
        return "Logarithmic (Search)"
    elif any(k in name for k in ["social", "meta", "tiktok", "linkedin", "instagram", "facebook"]):
        return "Hill Function (S-Curve)"
    elif any(k in name for k in ["email", "newsletter", "sms", "hubspot", "mailchimp"]):
        return "Negative Exponential (Email Ceiling)"
    elif any(k in name for k in ["seo", "content", "organic", "blog"]):
        return "Power Law (Compounding)"
    elif any(k in name for k in ["show", "event", "billboard", "pr", "sponsorship"]):
        return "Shifted Sigmoid (Threshold)"
    else:
        return "Logarithmic (Search)"

# ==========================================
# 4. Advanced Configuration Panel
# ==========================================
st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("⚙️ Enable Advanced Controls")

adv_settings = {}
model_options = [
    "Logarithmic (Search)", 
    "Hill Function (S-Curve)", 
    "Negative Exponential (Email Ceiling)", 
    "Power Law (Compounding)", 
    "Shifted Sigmoid (Threshold)"
]

for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    if pd.notna(channel_name) and str(channel_name).strip() != "":
        default_guessed = guess_default_model(channel_name)
        
        if advanced_mode:
            with st.sidebar.expander(f"🛠️ {channel_name} Options"):
                idx = model_options.index(default_guessed)
                model = st.selectbox("Algorithmic Fit", model_options, index=idx, key=f"mod_{channel_name}")
                fixed_cost = st.number_input("Fixed Infrastructure Cost (K)", min_value=0.0, value=0.0, step=0.5, key=f"fix_{channel_name}")
                learning = st.checkbox("Force Platform Learning Budget?", key=f"lrn_{channel_name}")
                adv_settings[channel_name] = {"model": model, "fixed_cost": fixed_cost, "learning": learning}
        else:
            adv_settings[channel_name] = {"model": default_guessed, "fixed_cost": 0.0, "learning": False}

# ==========================================
# 5. Core Algorithmic Math Engine
# ==========================================
channels = {}
for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    spend = float(row["Past Spend (K)"])
    leads = float(row["Past Leads"])
    
    if spend > 0 and leads > 0 and pd.notna(channel_name):
        settings = adv_settings[channel_name]
        var_spend = max(0.01, spend - settings["fixed_cost"])
        model_type = settings["model"]
        
        # Reverse-engineer scaling factor (Alpha) safely per model type
        if "Logarithmic" in model_type:
            alpha = leads / np.log(var_spend + 1)
        elif "Hill Function" in model_type:
            alpha = leads / (var_spend**2 / (var_spend**2 + 16))
        elif "Negative Exponential" in model_type:
            alpha = leads / (1 - np.exp(-0.2 * var_spend))
        elif "Power Law" in model_type:
            alpha = leads / (var_spend**0.5)
        elif "Shifted Sigmoid" in model_type:
            alpha = leads / (1 / (1 + np.exp(-2 * (var_spend - 4))))
            
        channels[channel_name] = {"alpha": alpha, "settings": settings}

if len(channels) == 0:
    st.warning("Please enter valid historical data metrics in the sidebar table to start optimization.")
    st.stop()

# Continuous optimization execution function
def lead_function(budget, channel_name):
    ch_data = channels[channel_name]
    alpha = ch_data["alpha"]
    model_type = ch_data["settings"]["model"]
    fixed_cost = ch_data["settings"]["fixed_cost"]
    
    eff_budget = max(0.0, budget - fixed_cost)
    if eff_budget == 0:
        return 0.0
        
    if "Logarithmic" in model_type:
        return alpha * np.log(eff_budget + 1)
    elif "Hill Function" in model_type:
        return alpha * (eff_budget**2 / (eff_budget**2 + 16))
    elif "Negative Exponential" in model_type:
        return alpha * (1 - np.exp(-0.2 * eff_budget))
    elif "Power Law" in model_type:
        return alpha * (eff_budget**0.5)
    elif "Shifted Sigmoid" in model_type:
        return alpha * (1 / (1 + np.exp(-2 * (eff_budget - 4))))

# SLSQP Mathematical Constraints Setup
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
        min_bound += 2.0  # Safe boundary limit for platform algorithm stability
    init_guess.append(max(1.0, min_bound))
    bounds.append((min_bound, 1000.0))

constraints = {'type': 'ineq', 'fun': constraint_target_leads}

# Limit ceiling validation check
theoretical_max_leads = sum(lead_function(1000.0, ch) for ch in channels.keys())
if target_leads > theoretical_max_leads:
    st.error("🛑 **Mathematical Boundary Alert**")
    st.write(f"The entered performance ceiling can generate an absolute maximum of **{int(theoretical_max_leads)}** leads under full system saturation. Reduce target values or improve core channel metrics.")
    st.stop()

result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# ==========================================
# 6. Premium UI Layout (Cyberpunk Theme)
# ==========================================
if result.success:
    optimized_budgets = result.x
    channel_leads = [lead_function(b, ch) for b, ch in zip(optimized_budgets, channels.keys())]
    total_spend = np.sum(optimized_budgets)
    blended_cac = total_spend / target_leads

# --- CUSTOM TECHNO CSS ---
    st.markdown("""
    <style>
    /* Lighter Synthwave Radial Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #431c7a 0%, #130f26 50%, #0b0817 100%);
    }
    
    /* Layered Neon Text Glow for Metrics */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* Keeps the numbers crisp and readable */
        text-shadow: 0 0 10px rgba(0, 255, 170, 0.8), 0 0 20px rgba(0, 255, 170, 0.3);
    }
    
    /* Soften Metric Labels for better visual hierarchy */
    [data-testid="stMetricLabel"] {
        color: #b8b1ce !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Create the Tabbed Workspace ---
    tab1, tab2 = st.tabs(["📊 Optimization Matrix", "✨ AI Strategic Advisor"])
    
    # ------------------------------------------
    # TAB 1: The Math & Data Dashboard
    # ------------------------------------------
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        # Clean metrics, no messy deltas!
        col1.metric("Optimized Budget (K)", f"${total_spend:,.2f}")
        col2.metric("Target Leads Met", f"{int(round(sum(channel_leads)))}")
        col3.metric("Blended CAC (K)", f"${blended_cac:,.2f}")
        
        st.markdown("---")
        
        df_results = pd.DataFrame({
            "Channel": list(channels.keys()),
            "Assigned Model": [channels[ch]["settings"]["model"].split(" ")[0] for ch in channels.keys()],
            "Recommended Budget (K)": optimized_budgets,
            "Expected Leads": channel_leads,
            "Effective CAC (K)": [b/l if l > 0 else 0 for b, l in zip(optimized_budgets, channel_leads)]
        })
        
        col_table, col_chart = st.columns([1.5, 1], gap="large")
        
        with col_table:
            st.subheader("Optimal Budget Allocation")
            
            # Clean, fast formatting without requiring external color libraries
            styled_df = df_results.style.format({
                "Recommended Budget (K)": "${:,.2f}",
                "Expected Leads": "{:.1f}",
                "Effective CAC (K)": "${:,.2f}"
            })
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
        with col_chart:
            st.subheader("Budget Share")
            # Sleek Donut Chart
            fig_donut = px.pie(df_results, values="Recommended Budget (K)", names="Channel", hole=0.65)
            fig_donut.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
            fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        st.subheader("Comparative Curve Saturation Models")
        st.caption("Inspect performance trajectories, inflection vectors, and diminishing return plateaus across distinct algorithmic types.")
        
        fig_curve = go.Figure()
        max_b = max(optimized_budgets) * 1.6 if max(optimized_budgets) > 0 else 15
        budget_range = np.linspace(0, max_b, 150)
        
        for i, ch_name in enumerate(channels.keys()):
            opt_b = optimized_budgets[i]
            opt_l = channel_leads[i]
            
            fig_curve.add_trace(go.Scatter(x=budget_range, y=[lead_function(b, ch_name) for b in budget_range], mode='lines', name=ch_name, line=dict(width=2.5)))
            fig_curve.add_trace(go.Scatter(x=[opt_b], y=[opt_l], mode='markers', marker=dict(size=11, line=dict(width=1.5, color='white')), showlegend=False, hoverinfo="skip"))

        fig_curve.update_layout(
            xaxis_title="Budget Allocated (K)", 
            yaxis_title="Leads Generated", 
            hovermode="x unified", 
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # ------------------------------------------
    # TAB 2: AI Advisor Agent (Full Width)
    # ------------------------------------------
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("✨ AI Strategic Advisor")
        st.caption("Consult your dedicated partner agent regarding cross-channel scaling recommendations. The agent has full visibility into your active matrix.")
        
        # Pull key cleanly from native vault
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Greetings. I've parsed your dashboard's structural mathematical models and channel efficiencies. Ask me any cross-functional scaling or optimization question."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about budget reallocations or performance plateaus..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Removed deltas from the AI context as well so the chat doesn't crash!
            dashboard_context = f"""
            You are an elite, highly-paid Fractional CMO and Enterprise Growth Advisor. 
            The user is viewing their Media Mix Optimization dashboard with you.
            
            Current Dashboard State:
            Total Recommended Spend: ${total_spend:,.2f}K
            Required Scale Target: {int(round(sum(channel_leads)))} Leads
            Calculated Blended System CAC: ${blended_cac:,.2f}K
            
            Granular Data Points Matrix:
            {df_results.to_string()}
            
            Your Directive:
            1. DO NOT just recite the data back to the user. They can already see the numbers.
            2. Act as a strategic partner. Use the data matrix as a starting point to diagnose broader business bottlenecks (e.g., if Paid Search is saturated, discuss transitioning from 'demand capture' to 'demand generation').
            3. Push back on the user if their goals seem misaligned with their budget constraints.
            4. End every response by asking ONE sharp, high-leverage question to uncover their business context (e.g., "What is the LTV to CAC ratio here?", "Who is our target demographic?", or "Do we have the creative bandwidth to scale the S-Curve on Social?").
            5. Be conversational, authoritative, and concise (max 3 short paragraphs).
            """
            
            api_messages = []
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "model"
                api_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
            config = types.GenerateContentConfig(system_instruction=dashboard_context, temperature=0.6)
            
            with st.spinner("Processing scenario parameters..."):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=api_messages,
                        config=config
                    )
                    ai_reply = response.text
                except Exception as e:
                    ai_reply = f"🚨 Secure Agent Bridge Error: {e}"
            
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

else:
    st.error("The optimization matrix could not define a realistic convergent solution. Check boundary parameters or historical CAC entries.")
